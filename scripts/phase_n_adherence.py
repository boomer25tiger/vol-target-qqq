#!/usr/bin/env python3
"""
N0 reconciliation + N1 bootstrap of the target-adherence statistic.

Adherence = mean absolute deviation of the annualized realized volatility over
NON-OVERLAPPING 21-day blocks from the 20% target (SPEC 10.2). Point estimates are
recomputed for all 18 strategies through one code path; the statistic is then
stationary-bootstrapped on the daily portfolio return series per strategy.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
from volteq.eval.layer1 import stationary_bootstrap_idx

cfg = load_config()
TGT = cfg["frozen"]["target_vol"]
W = 21
TAB = os.path.join(repo_root(), "outputs", "tables")
B = 2000
SEED1, SEED2 = 20260806, 19990310
BLOCKS_DAYS = [42, 105, 210]        # = MCS grid {2,5,10} months in trading days; all > 21


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def adherence_vec(Rmat):
    """MAD-from-target of non-overlapping 21-day annualized vol; Rmat (T, k) -> (k,)."""
    T = Rmat.shape[0]; nb = T // W
    Rb = Rmat[:nb * W].reshape(nb, W, -1)
    bvol = Rb.std(axis=1, ddof=1) * np.sqrt(ANNUAL)
    return np.abs(bvol - TGT).mean(axis=0)


def main():
    be = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "backtest_equity.parquet"))
    rets = be.pct_change().dropna()                    # daily returns == bt['ret'] (E2 fix)
    # trailing_rv21 is not in backtest_equity; recompute its daily returns via the same engine
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    qqq_ret = _raw("qqq_daily", "close").pct_change().dropna()
    dff = _raw("dff", "dff")
    ws_tr = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    bt_tr = run_backtest(ws_tr, qqq_ret, dff, cfg)
    rets = rets.join(bt_tr["ret"].rename("trailing_rv21"), how="inner")

    STRAT = list(be.columns) + ["trailing_rv21"]
    R = rets[STRAT].to_numpy()
    T = R.shape[0]
    point = dict(zip(STRAT, adherence_vec(R)))

    # ---- N0 reconciliation: confirm point estimates match layer2_adherence.csv ----
    la = pd.read_csv(os.path.join(TAB, "layer2_adherence.csv")).rename(
        columns={"Unnamed: 0": "id"}).set_index("id")
    print("=" * 78); print("N0 RECONCILIATION (point adherence, one code path)"); print("=" * 78)
    print(f"{'strategy':16s} {'this':>8} {'layer2_adh':>11} {'match':>6}")
    for s in STRAT:
        prior = la.loc[s, "mad_nonoverlap"] if s in la.index else np.nan
        ok = "" if np.isnan(prior) else ("OK" if abs(prior - point[s]) < 5e-4 else "DIFF")
        print(f"{s:16s} {point[s]:8.4f} {prior:11.4f} {ok:>6}")

    # ---- N1 bootstrap ----
    print("\n" + "=" * 78); print(f"N1 BOOTSTRAP  B={B}  seed={SEED1}  blocks(days)={BLOCKS_DAYS}"); print("=" * 78)
    print(f"  21-day adherence window; every block length exceeds it: "
          f"{[b for b in BLOCKS_DAYS]} all > {W} -> none dropped")

    def run_boot(block, seed):
        rng = np.random.RandomState(seed)
        boot = stationary_bootstrap_idx(T, B, block, rng)          # (B, T)
        reps = np.empty((B, len(STRAT)))
        for b in range(B):
            reps[b] = adherence_vec(R[boot[b]])
        return reps

    # CI per block (percentile, 90% = MCS level)
    rows = []
    reps_by_block = {}
    for block in BLOCKS_DAYS:
        reps = run_boot(block, SEED1)
        reps_by_block[block] = reps
        lo = np.percentile(reps, 5, axis=0); hi = np.percentile(reps, 95, axis=0)
        for k, s in enumerate(STRAT):
            rows.append({"strategy": s, "block_days": block, "point": point[s],
                         "ci90_lo": lo[k], "ci90_hi": hi[k]})
    inf = pd.DataFrame(rows)
    inf.to_csv(os.path.join(TAB, "adherence_inference.csv"), index=False)

    # second-seed stability at headline block 105
    reps_s2 = run_boot(105, SEED2)
    lo1 = np.percentile(reps_by_block[105], 5, axis=0); hi1 = np.percentile(reps_by_block[105], 95, axis=0)
    lo2 = np.percentile(reps_s2, 5, axis=0); hi2 = np.percentile(reps_s2, 95, axis=0)
    max_shift = float(np.max(np.abs(np.r_[lo1 - lo2, hi1 - hi2])))
    print(f"\n  second-seed ({SEED2}) max CI-bound shift at block 105: {max_shift:.5f}")

    print(f"\n  Adherence point + 90% CI (block 105 = 5 months), sorted:")
    order = sorted(STRAT, key=lambda s: point[s])
    for s in order:
        k = STRAT.index(s)
        print(f"    {s:16s} {point[s]:.4f}  [{lo1[k]:.4f}, {hi1[k]:.4f}]")

    # ---- pairwise fraction matrix (headline block 105): P(adh_i < adh_j) ----
    reps = reps_by_block[105]
    P = np.zeros((len(STRAT), len(STRAT)))
    for i in range(len(STRAT)):
        for j in range(len(STRAT)):
            P[i, j] = np.mean(reps[:, i] < reps[:, j])
    pw = pd.DataFrame(P, index=STRAT, columns=STRAT)
    pw.to_csv(os.path.join(TAB, "adherence_pairwise.csv"))

    # headline reads: resizers vs static; resizers among themselves
    resizers = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv", "trailing_rv21"]
    statics = ["bench_buy_hold", "bench_const_lev", "bench_uncond_vol"]
    print("\n  P(resizer adheres better than static) - every resizer vs every static:")
    frac = [P[STRAT.index(r), STRAT.index(s)] for r in resizers for s in statics]
    print(f"    min={min(frac):.3f} max={max(frac):.3f} (1.0 = resizer better in all reps)")
    # gjr (best point) vs the naive resizers
    print("\n  gjr_skewt vs naive resizers, P(gjr better):")
    for s in ["ewma", "rv", "rfsv", "trailing_rv21"]:
        print(f"    gjr vs {s:14s}: {P[STRAT.index('gjr_skewt'), STRAT.index(s)]:.3f}")
    print(f"\ntables -> adherence_inference.csv, adherence_pairwise.csv")


if __name__ == "__main__":
    main()

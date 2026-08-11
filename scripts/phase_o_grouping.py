#!/usr/bin/env python3
"""
O0 group-membership audit + O1 multiplicity correction.

Rebuilds the dynamic/naive partition on MODEL CLASS (declared independently of the
adherence values), reruns the paired comparison per pair, and runs Hansen's MCS on the
adherence loss directly for a multiplicity-aware retained set.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
from volteq.eval.layer1 import stationary_bootstrap_idx, model_confidence_set

cfg = load_config(); TGT = cfg["frozen"]["target_vol"]; W = 21
TAB = os.path.join(repo_root(), "outputs", "tables")
B = 2000; SEED = 20260806

# ---- model-class partition (independent of adherence values) ----
# criterion: does the model carry conditional-variance dynamics (mean-reverting GARCH-type
# or fractional) and sit in the Layer-1 QLIKE MCS? garch/egarch/gjr (GARCH family), har
# (HAR-RV), rfsv (rough/fractional) -> dynamic. ewma (IGARCH, flat), rv & trailing_rv21
# (no model) -> naive. This is exactly the QLIKE MCS retained/excluded split.
DYNAMIC = ["garch_skewt", "egarch_skewt", "gjr_skewt", "har", "rfsv"]
NAIVE = ["ewma", "rv", "trailing_rv21"]
PRIMARY8 = DYNAMIC + NAIVE


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def block_vols(Rmat):
    T = Rmat.shape[0]; nb = T // W
    Rb = Rmat[:nb * W].reshape(nb, W, -1)
    return Rb.std(axis=1, ddof=1) * np.sqrt(ANNUAL)     # (nb, M)


def main():
    be = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "backtest_equity.parquet"))
    rets = be.pct_change().dropna()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    qqq = _raw("qqq_daily", "close").pct_change().dropna(); dff = _raw("dff", "dff")
    ws = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    rets = rets.join(run_backtest(ws, qqq, dff, cfg)["ret"].rename("trailing_rv21"), how="inner")
    R = rets[PRIMARY8].to_numpy(); T = R.shape[0]

    print("=" * 82); print("O0 GROUP-MEMBERSHIP AUDIT (partition on model class, not values)"); print("=" * 82)
    bv = block_vols(R); adh = np.abs(bv - TGT).mean(axis=0)
    crit = {"garch_skewt": "GARCH(1,1) conditional dynamics; in QLIKE MCS",
            "egarch_skewt": "EGARCH conditional dynamics; in QLIKE MCS",
            "gjr_skewt": "GJR conditional dynamics; in QLIKE MCS",
            "har": "HAR-RV lag dynamics; in QLIKE MCS",
            "rfsv": "rough/fractional dynamics; in QLIKE MCS",
            "ewma": "IGARCH flat term structure (no mean reversion); excluded from QLIKE MCS",
            "rv": "trailing RV, no model; excluded from QLIKE MCS",
            "trailing_rv21": "trailing YZ_21 rung, no model; excluded from QLIKE MCS"}
    for s in PRIMARY8:
        grp = "DYNAMIC" if s in DYNAMIC else "naive"
        print(f"  {s:15s} adh={adh[PRIMARY8.index(s)]:.4f}  [{grp:7s}]  {crit[s]}")
    print("\n  N1 had placed rfsv in the NAIVE group (adh 0.0539 sits in the naive band) -")
    print("  that read membership off the values and was circular. Corrected: rfsv is DYNAMIC.")

    # ---- O1 req 1: every dynamic-vs-naive pair, individually (rfsv on dynamic side) ----
    print("\n" + "=" * 82); print("O1 PAIRED FRACTIONS, per pair (model-class partition, B=%d)" % B); print("=" * 82)
    rng = np.random.RandomState(SEED)
    boot = stationary_bootstrap_idx(T, B, 105, rng)     # block 105 d = N1 headline
    reps = np.empty((B, len(PRIMARY8)))
    for b in range(B):
        reps[b] = np.abs(block_vols(R[boot[b]]) - TGT).mean(axis=0)
    idx = {s: i for i, s in enumerate(PRIMARY8)}
    print(f"  {'pair':30s} P(dynamic adheres better)")
    survive = 0; total = 0
    for d in DYNAMIC:
        for n in NAIVE:
            p = np.mean(reps[:, idx[d]] < reps[:, idx[n]])
            total += 1; survive += int(p >= 0.95)
            flag = "" if p >= 0.95 else "  <-- below 0.95"
            print(f"  {d+' vs '+n:30s} {p:.3f}{flag}")
    print(f"\n  pairs clearing 0.95 (per-pair, uncorrected): {survive}/{total}")

    # ---- O1 req 2: multiplicity-aware MCS on the adherence loss ----
    print("\n" + "=" * 82); print("O1 ADHERENCE MODEL CONFIDENCE SET (Hansen, block-loss |vol-0.20|)"); print("=" * 82)
    L = np.abs(block_vols(R) - TGT)                     # (nb, 8) per-block adherence loss
    print(f"  loss = |block_vol - 0.20| over {L.shape[0]} non-overlapping 21-day blocks; "
          f"MCS alpha=0.10, B=10000")
    print("  block grid in BLOCK units {2,5,10} = N1's {42,105,210} days / 21:")
    mcs_rows = []
    for blk in (2, 5, 10):
        res = model_confidence_set(L, PRIMARY8, alpha=0.10, B=10000, block=blk, seed=SEED)
        ret = [m for m in PRIMARY8 if m in res["retained"]]          # canonical order
        print(f"    block={blk:2d}: retained ({len(ret)}/8) {ret}")
        mcs_rows.append({"block_units": blk, "block_days": blk * 21,
                         "retained": ",".join(ret), "n_retained": len(ret)})
        if blk == 5:
            print(f"            elimination order: {res['elim_order']}")
    pd.DataFrame(mcs_rows).to_csv(os.path.join(TAB, "adherence_mcs.csv"), index=False)
    print(f"  wrote {os.path.join(TAB, 'adherence_mcs.csv')}")
    print("\n  compare: Layer-1 QLIKE MCS retains {garch,egarch,gjr,har,rfsv}, excludes {ewma,rv,trailing}")


if __name__ == "__main__":
    main()

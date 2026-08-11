#!/usr/bin/env python3
"""
P0 block-length power check for the adherence MCS.

Diagnostic: at the 210-day block, does the MCS still EXCLUDE the static benchmarks
(then the finer dynamics-vs-naive effect is genuinely small) or retain them too
(then the set is uninformative and the result is about power)?
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
from volteq.eval.layer1 import model_confidence_set

cfg = load_config(); TGT = cfg["frozen"]["target_vol"]; W = 21; SEED = 20260806
RESIZERS = ["garch_skewt", "egarch_skewt", "gjr_skewt", "har", "rfsv", "ewma", "rv", "trailing_rv21"]
STATIC = ["bench_buy_hold", "bench_const_lev", "bench_uncond_vol"]
ALL = RESIZERS + STATIC


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def block_vols(Rmat):
    T = Rmat.shape[0]; nb = T // W
    return (Rmat[:nb * W].reshape(nb, W, -1)).std(axis=1, ddof=1) * np.sqrt(ANNUAL)


def main():
    be = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "backtest_equity.parquet"))
    rets = be.pct_change().dropna()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    qqq = _raw("qqq_daily", "close").pct_change().dropna(); dff = _raw("dff", "dff")
    ws = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    rets = rets.join(run_backtest(ws, qqq, dff, cfg)["ret"].rename("trailing_rv21"), how="inner")
    R = rets[ALL].to_numpy(); T = R.shape[0]
    nb = T // W

    print("=" * 84); print("P0 BLOCK-LENGTH POWER CHECK"); print("=" * 84)
    print(f"  daily sample T={T}; non-overlapping 21-day blocks nb={nb}")
    print(f"  {'block(days)':>11} {'block(units)':>12} {'eff. resampled units ~ nb/units':>32}")
    for bd, bu in [(42, 2), (105, 5), (210, 10)]:
        print(f"  {bd:>11} {bu:>12} {f'{nb}/{bu} = {nb//bu}':>32}")

    # CI widths per strategy at each block (from adherence_inference.csv)
    inf = pd.read_csv(os.path.join(repo_root(), "outputs", "tables", "adherence_inference.csv"))
    print("\n  adherence 90% CI width per strategy (loss of precision as a number):")
    show = ["gjr_skewt", "har", "ewma", "rfsv", "bench_buy_hold"]
    print(f"    {'strategy':16s} {'42d':>8} {'105d':>8} {'210d':>8}")
    for s in show:
        w = {}
        for bd in (42, 105, 210):
            r = inf[(inf.strategy == s) & (inf.block_days == bd)]
            w[bd] = float(r.ci90_hi.iloc[0] - r.ci90_lo.iloc[0]) if len(r) else np.nan
        print(f"    {s:16s} {w[42]:8.4f} {w[105]:8.4f} {w[210]:8.4f}")

    # MCS on the adherence loss with static INCLUDED, per block
    L = np.abs(block_vols(R) - TGT)     # (nb, 11)
    print("\n  Adherence MCS (alpha=0.10, B=10000) on ALL 11 (resizers + static):")
    for bu in (2, 5, 10):
        res = model_confidence_set(L, ALL, alpha=0.10, B=10000, block=bu, seed=SEED)
        ret = res["retained"]
        static_in = [s for s in STATIC if s in ret]
        naive_in = [s for s in ["ewma", "rv", "trailing_rv21"] if s in ret]
        print(f"    block={bu:2d} ({bu*21}d): retained {len(ret)}/11 -> "
              f"static retained: {static_in if static_in else 'NONE (excluded)'}; "
              f"naive resizers retained: {naive_in}")

    print("\n  DIAGNOSTIC: static excluded at 210d + naive resizers back -> finer effect is small,")
    print("  not a power vacuum. static retained at 210d -> uninformative (a statement about power).")


if __name__ == "__main__":
    main()

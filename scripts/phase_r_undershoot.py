#!/usr/bin/env python3
"""
R1 signed vs absolute adherence deviation (all 18 strategies) and R2 undershoot diagnostic.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL

cfg = load_config(); TGT = cfg["frozen"]["target_vol"]; W = 21


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def blockstats(r):
    nb = len(r) // W
    bv = (r[:nb * W].reshape(nb, W)).std(axis=1, ddof=1) * np.sqrt(ANNUAL)
    return bv.mean(), (bv - TGT).mean(), np.abs(bv - TGT).mean(), bv.std(ddof=1)


def main():
    be = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "backtest_equity.parquet"))
    rets = be.pct_change().dropna()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel(); qqq = _raw("qqq_daily", "close").pct_change().dropna(); dff = _raw("dff", "dff")
    ws = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    rets = rets.join(run_backtest(ws, qqq, dff, cfg)["ret"].rename("trailing_rv21"), how="inner")
    ALL = list(be.columns) + ["trailing_rv21"]
    bm = pd.read_csv(os.path.join(repo_root(), "outputs", "tables", "backtest_metrics.csv"), index_col=0)

    print("=" * 96)
    print("R1/R2  signed vs absolute adherence deviation (block-vol), all strategies")
    print("=" * 96)
    print(f"{'strategy':16s} {'mean vol':>8} {'signed':>8} {'MAD':>7} {'block-vol sd':>12} "
          f"{'full-samp vol':>13} {'cap%':>6}")
    rows = []
    for s in ALL:
        mv, sd, mad, bsd = blockstats(rets[s].to_numpy())
        fs = bm.loc[s, "realized_vol"] if s in bm.index else np.nan
        cap = bm.loc[s, "pct_months_at_cap"] if s in bm.index else np.nan
        rows.append((s, mv, sd, mad, bsd, fs, cap))
        print(f"{s:16s} {mv:8.4f} {sd:+8.4f} {mad:7.4f} {bsd:12.4f} {fs:13.4f} {cap*100 if cap==cap else float('nan'):6.1f}")
    df = pd.DataFrame(rows, columns=["s", "meanvol", "signed", "mad", "bsd", "fullvol", "cap"])
    print(f"\n  ALL {len(ALL)} strategies undershoot (signed<0): {(df.signed<0).all()}; "
          f"mean signed across set = {df.signed.mean():+.4f}")
    print(f"  corr(signed, block-vol sd) = {np.corrcoef(df.signed, df.bsd)[0,1]:+.3f} "
          f"(more dispersed -> more undershoot?)")
    print(f"  corr(signed, cap%) = {np.corrcoef(df.signed, df.cap.fillna(0))[0,1]:+.3f}")

    # R2 candidate (c): forecast vol-level bias mean(√V̂)/mean(√RV) for the 8 primary
    print("\n  R2(c) forecast vol-level bias mean(√V̂)/mean(√RV) (>1 = forecast vol high -> weights low):")
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], W)
    prim = {"garch_skewt": "garch_skewt", "egarch_skewt": "egarch_skewt", "gjr_skewt": "gjr_skewt",
            "ewma": "ewma", "rv": "rv", "har": "har", "rfsv": "rfsv"}
    for s, col in prim.items():
        d = pd.concat([fwd.reindex(v.index).rename("RV"), v[col].rename("Vh")], axis=1)
        d = d.loc[d.index >= pd.Timestamp(cfg["frozen"]["eval_start"])].dropna()
        ratio = np.sqrt(d["Vh"]).mean() / np.sqrt(d["RV"]).mean()
        print(f"    {s:14s} mean√V̂/mean√RV = {ratio:.3f}")

    # R2 baseline: const_lev has fixed leverage, no forecast, ~no cap -> isolates measurement/drift
    print("\n  R2 baseline (const_lev: fixed leverage, no forecast, no cap):")
    s = "bench_const_lev"
    mv, sd, mad, bsd = blockstats(rets[s].to_numpy())
    print(f"    const_lev signed dev {sd:+.4f}, full-sample vol {bm.loc[s,'realized_vol']:.4f} "
          f"-> block-vol mean sits {sd:+.4f} below target though full-sample vol is ~0.20")
    # Jensen estimator bias: sqrt of sample variance underestimates sigma by ~(1 - 1/(4(W-1)))
    jb = 1 - 1/(4*(W-1))
    print(f"    √-estimator downward bias for W={W} blocks ≈ {(jb-1)*TGT:+.4f} on a 0.20 target")


if __name__ == "__main__":
    main()

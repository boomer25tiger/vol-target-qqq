#!/usr/bin/env python3
"""
R1/R2 canonical-aligned signed vs absolute adherence deviation for all 18 strategies.

Reuses layer2_eval.schedules / blocks / run_backtest so the non-overlapping-block MAD
reproduces outputs/tables/layer2_adherence.csv exactly (same series, same block
alignment), and reports the signed deviation (mean block vol - 0.20) and the mean
block vol on the identical construction. The dispersion number is the published
vol-of-vol column (bvol.std(ddof=1)); it is re-derived here as a check.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
import layer2_eval as L2  # reuse schedules(), blocks(), _raw(), TARGET

TGT = L2.TARGET
W = 21


def blockstats(ret: pd.Series):
    """Signed dev, MAD, mean block vol, block-vol sd on layer2_eval's exact blocks."""
    bvol = np.array([b.std(ddof=1) * np.sqrt(ANNUAL) for b in L2.blocks(ret.values)])
    return dict(mean_bvol=bvol.mean(), signed=(bvol - TGT).mean(),
                mad=np.abs(bvol - TGT).mean(), vov=bvol.std(ddof=1), n=len(bvol))


def main():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = L2._raw("qqq_daily", "close").pct_change().dropna()
    dff = L2._raw("dff", "dff")
    sc = L2.schedules(cfg, v, qqq_ret)                       # 17 schedules (14 fcst + 3 bench)
    # add the trailing-RV sizing rung (yz_21), the 18th strategy
    panel = load_panel()
    sc["trailing_rv21"] = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg),
                                    index=v.index)
    bts = {n: run_backtest(ws, qqq_ret, dff, cfg) for n, ws in sc.items()}

    ORDER = ["garch_skewt", "garch_normal", "gjr_skewt", "gjr_normal", "egarch_skewt",
             "egarch_normal", "har", "rfsv", "ewma", "rv", "trailing_rv21",
             "rfsv_h002", "rfsv_h005", "rfsv_h010", "rfsv_h015",
             "bench_const_lev", "bench_uncond_vol", "bench_buy_hold"]

    print("=" * 92)
    print("R1/R2 canonical-aligned (layer2_eval blocks): signed vs absolute block-vol deviation")
    print("=" * 92)
    print(f"{'strategy':17s} {'mean bvol':>9} {'signed':>8} {'MAD':>7} {'vov(disp)':>9} {'n':>4}")
    rows = {}
    for n in ORDER:
        s = blockstats(bts[n]["ret"]); rows[n] = s
        print(f"{n:17s} {s['mean_bvol']:9.4f} {s['signed']:+8.4f} {s['mad']:7.4f} "
              f"{s['vov']:9.4f} {s['n']:4d}")

    df = pd.DataFrame(rows).T
    resizers = [n for n in ORDER if not n.startswith("bench") and n != "trailing_rv21"] + ["trailing_rv21"]
    print("\n  --- universality ---")
    print(f"  signed<0 (undershoot) for ALL 18: {(df['signed'] < 0).all()}")
    over = df.index[df['signed'] > 0].tolist()
    print(f"  OVER-shoot (signed>0): {over}")
    print(f"  mean signed, all 18            = {df['signed'].mean():+.4f}")
    print(f"  mean signed, 15 forecast+trail = {df.loc[resizers,'signed'].mean():+.4f}")

    # canonical cross-check vs layer2_adherence.csv
    adh = pd.read_csv(os.path.join(repo_root(), "outputs", "tables", "layer2_adherence.csv"), index_col=0)

    # full-sample vol per strategy (block-mean sits BELOW this by the Jensen/dispersion gap)
    fullvol = {}
    for n in ORDER:
        fullvol[n] = float(adh.loc[n, "realized_vol"]) if n in adh.index \
            else float(bts[n]["ret"].std() * np.sqrt(ANNUAL))
    df["fullvol"] = pd.Series(fullvol)
    # R2(a) Jensen: predicted (fullvol - block-mean) gap = vov^2 / (2*mean_bvol) [power-mean]
    df["jensen_pred"] = df["vov"] ** 2 / (2 * df["mean_bvol"])
    df["jensen_act"] = df["fullvol"] - df["mean_bvol"]
    print("\n  R2(a) Jensen/dispersion: full-sample vol - mean block vol, predicted vs actual")
    print(f"  {'strategy':17s} {'fullvol':>8} {'blkmean':>8} {'gap_act':>8} {'gap_pred':>9} {'vov':>7}")
    for n in ORDER:
        r = df.loc[n]
        print(f"  {n:17s} {r['fullvol']:8.4f} {r['mean_bvol']:8.4f} {r['jensen_act']:8.4f} "
              f"{r['jensen_pred']:9.4f} {r['vov']:7.4f}")
    print(f"  corr(gap_act, vov) = {np.corrcoef(df['jensen_act'], df['vov'])[0,1]:+.3f}; "
          f"max|pred-act| = {(df['jensen_pred']-df['jensen_act']).abs().max():.4f}")

    # export master signed-deviation table (prose in paper cites this, like adherence_inference.csv)
    out = df[["fullvol", "mean_bvol", "signed", "mad", "vov"]].copy()
    out.columns = ["realized_vol", "mean_block_vol", "signed_dev", "mad_nonoverlap", "dispersion_vov"]
    out.index.name = "strategy"
    outp = os.path.join(repo_root(), "outputs", "tables", "adherence_signed.csv")
    out.round(6).to_csv(outp)
    print(f"\n  wrote {outp}")
    print("\n  --- MAD cross-check vs layer2_adherence.csv (must match) ---")
    for n in ["rfsv", "har", "gjr_skewt", "ewma"]:
        pub = adh.loc[n, "mad_nonoverlap"]; got = rows[n]["mad"]
        print(f"    {n:14s} published {pub:.4f}  recomputed {got:.4f}  Δ={got-pub:+.5f}")
    print("  --- vov cross-check ---")
    for n in ["rfsv", "har"]:
        pub = adh.loc[n, "vol_of_vol"]; got = rows[n]["vov"]
        print(f"    {n:14s} published {pub:.4f}  recomputed {got:.4f}  Δ={got-pub:+.5f}")

    # R2 source (b): cap truncation - signed dev vs pct at cap
    bm = pd.read_csv(os.path.join(repo_root(), "outputs", "tables", "backtest_metrics.csv"), index_col=0)
    cap = bm["pct_months_at_cap"].reindex(df.index)
    print(f"\n  R2(b) corr(signed, cap%) = {np.corrcoef(df['signed'], cap.fillna(0))[0,1]:+.3f} "
          f"(cap binds only ewma {cap.get('ewma',np.nan)*100:.1f}%, rv {cap.get('rv',np.nan)*100:.1f}%, "
          f"egarch_skewt {cap.get('egarch_skewt',np.nan)*100:.1f}%)")

    # R2 source (a) isolation: const_lev has fixed leverage, no forecast, cap never binds
    cl = rows["bench_const_lev"]
    print(f"\n  R2(a) const_lev: mean bvol {cl['mean_bvol']:.4f} signed {cl['signed']:+.4f}, "
          f"full-sample vol {adh.loc['bench_const_lev','realized_vol']:.4f}")
    print(f"        Jensen gap: full-sample vol ~0.198 but MEAN of block vols is {cl['mean_bvol']:.4f} "
          f"({cl['signed']:+.4f}); concavity of sqrt on dispersed block variance.")
    jb = 1 - 1 / (4 * (W - 1))
    print(f"        small-sample sqrt-estimator bias for W={W}: ~{(jb-1)*TGT:+.4f} of the gap (minor).")


if __name__ == "__main__":
    main()

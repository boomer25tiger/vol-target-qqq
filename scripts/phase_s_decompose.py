#!/usr/bin/env python3
"""
S0 - decompose the signed target deviation into a level term and a concavity term.

Identity (exact):  signed_dev = (block_mean - 0.20)
                             = (fullvol - 0.20) - (fullvol - block_mean)
                             = level_term      -  concavity_gap(measured)

The R2 passage attributed the undershoot to concavity (concavity_pred = vov^2/(2*mean)).
This script tests whether that single term predicts the signed deviation, and if not,
whether the residual is the level term, tested against each strategy's volatility-level
forecast ratio mean(sqrt(V_hat))/mean(sqrt(RV)).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
import layer2_eval as L2

TGT = L2.TARGET
W = 21


def blockstats(ret: pd.Series):
    bvol = np.array([b.std(ddof=1) * np.sqrt(ANNUAL) for b in L2.blocks(ret.values)])
    return dict(mean_bvol=float(bvol.mean()), vov=float(bvol.std(ddof=1)))


def main():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = L2._raw("qqq_daily", "close").pct_change().dropna()
    dff = L2._raw("dff", "dff")
    sc = L2.schedules(cfg, v, qqq_ret)
    panel = load_panel()
    sc["trailing_rv21"] = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    bts = {n: run_backtest(ws, qqq_ret, dff, cfg) for n, ws in sc.items()}
    adh = pd.read_csv(os.path.join(repo_root(), "outputs", "tables", "layer2_adherence.csv"), index_col=0)

    # forecast volatility-level ratio mean(sqrt(V_hat))/mean(sqrt(RV)), eval window
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], W)
    def fratio(col):
        if col not in v.columns:
            return np.nan
        d = pd.concat([fwd.reindex(v.index).rename("RV"), v[col].rename("Vh")], axis=1)
        d = d.loc[d.index >= pd.Timestamp(cfg["frozen"]["eval_start"])].dropna()
        return float(np.sqrt(d["Vh"]).mean() / np.sqrt(d["RV"]).mean())

    ORDER = ["garch_skewt", "garch_normal", "gjr_skewt", "gjr_normal", "egarch_skewt",
             "egarch_normal", "har", "rfsv", "ewma", "rv", "trailing_rv21",
             "rfsv_h002", "rfsv_h005", "rfsv_h010", "rfsv_h015",
             "bench_const_lev", "bench_uncond_vol", "bench_buy_hold"]
    fcol = {"trailing_rv21": None, "bench_const_lev": None, "bench_uncond_vol": None,
            "bench_buy_hold": None}

    rows = []
    for n in ORDER:
        s = blockstats(bts[n]["ret"])
        fullvol = float(adh.loc[n, "realized_vol"]) if n in adh.index \
            else float(bts[n]["ret"].std() * np.sqrt(ANNUAL))
        mean_bvol = s["mean_bvol"]; vov = s["vov"]
        level = fullvol - TGT
        conc_meas = fullvol - mean_bvol
        conc_pred = vov ** 2 / (2 * mean_bvol)
        signed = mean_bvol - TGT
        residual = signed + conc_pred          # deviation from a pure-concavity account
        fr = fratio(fcol.get(n, n))
        rows.append(dict(strategy=n, fullvol=fullvol, mean_bvol=mean_bvol, vov=vov,
                         level=level, conc_meas=conc_meas, conc_pred=conc_pred,
                         signed=signed, residual=residual, fratio=fr))
    df = pd.DataFrame(rows).set_index("strategy")

    print("=" * 108)
    print("S0  signed_dev = level_term - concavity_term   (level = fullvol-0.20; conc = fullvol-blockmean)")
    print("=" * 108)
    print(f"{'strategy':16s} {'fullvol':>8} {'blkmean':>8} {'vov':>7} {'level':>8} "
          f"{'conc_meas':>9} {'conc_pred':>9} {'signed':>8} {'resid':>8} {'fratio':>7}")
    for n in ORDER:
        r = df.loc[n]
        print(f"{n:16s} {r.fullvol:8.4f} {r.mean_bvol:8.4f} {r.vov:7.4f} {r.level:+8.4f} "
              f"{r.conc_meas:9.4f} {r.conc_pred:9.4f} {r.signed:+8.4f} {r.residual:+8.4f} "
              f"{r.fratio:7.3f}" if r.fratio == r.fratio else
              f"{n:16s} {r.fullvol:8.4f} {r.mean_bvol:8.4f} {r.vov:7.4f} {r.level:+8.4f} "
              f"{r.conc_meas:9.4f} {r.conc_pred:9.4f} {r.signed:+8.4f} {r.residual:+8.4f} {'--':>7}")

    # the rfsv vs har question, own means
    print("\n--- rfsv vs har with each strategy's OWN mean (user assumed 0.19 for both) ---")
    for n in ["har", "rfsv"]:
        r = df.loc[n]
        print(f"  {n:6s} vov {r.vov:.4f} / own blkmean {r.mean_bvol:.4f} -> conc_pred {r.conc_pred:.4f}; "
              f"level {r.level:+.4f}; signed = level - conc = {r.level:+.4f} - {r.conc_meas:.4f} = {r.signed:+.4f}")
    print(f"  concavity gap alone orders rfsv ({df.loc['rfsv','conc_pred']:.4f}) > har "
          f"({df.loc['har','conc_pred']:.4f}); signed orders rfsv ({df.loc['rfsv','signed']:+.4f}) "
          f"vs har ({df.loc['har','signed']:+.4f}) -> level term flips it.")

    # is the residual systematic? test vs level and vs forecast ratio
    print("\n--- residual (signed + conc_pred) vs level term and forecast ratio ---")
    print(f"  corr(residual, level_term)     = {np.corrcoef(df.residual, df.level)[0,1]:+.3f}")
    fr = df.dropna(subset=["fratio"])
    print(f"  corr(residual, forecast ratio) = {np.corrcoef(fr.residual, fr.fratio)[0,1]:+.3f}  "
          f"(n={len(fr)}, resizers with a forecast)")
    print(f"  corr(level_term, forecast ratio)= {np.corrcoef(fr.level, fr.fratio)[0,1]:+.3f}")
    print(f"  mean |residual| = {df.residual.abs().mean():.4f}; mean |approx err conc_pred-conc_meas| "
          f"= {(df.conc_pred-df.conc_meas).abs().mean():.4f}")

    # overshooters
    over = df.index[df.signed > 0].tolist()
    print(f"\n--- overshooters (signed>0): {over} ---")
    for n in over:
        r = df.loc[n]
        print(f"  {n:16s} level {r.level:+.4f}  conc {r.conc_meas:.4f}  signed {r.signed:+.4f}  "
              f"(fullvol {r.fullvol:.4f})")
    print("  const_lev/uncond_vol (scaled to ~target fullvol) UNDER-shoot; buy_hold (fixed exposure, "
          "fullvol 0.268 far above target) OVER-shoots: its huge +level dominates the concavity gap.")

    df.round(5).to_csv(os.path.join(repo_root(), "outputs", "tables", "adherence_decomposition.csv"))
    print(f"\nwrote outputs/tables/adherence_decomposition.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the volatility-targeting backtest (SPEC.md Sections 8-9) for every model in
the V_t(21) panel plus the four benchmark rungs, and report Layer-2 metrics
(target adherence first, then risk/return). Reads data/processed + data/raw. No
network. Writes outputs/tables/backtest_metrics.csv, data/processed/backtest_equity.parquet,
and figures under figures/.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.backtest.engine import run_backtest, metrics, sizing_weight   # noqa: E402

PROC = os.path.join(repo_root(), "data", "processed")
FIG = os.path.join(repo_root(), "figures")
TAB = os.path.join(repo_root(), "outputs", "tables")


def _load_raw(name, col):
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[col]


def main():
    cfg = load_config()
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])

    v = pd.read_parquet(os.path.join(PROC, "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_close = _load_raw("qqq_daily", "close")
    qqq_ret = qqq_close.pct_change().dropna()
    dff = _load_raw("dff", "dff")
    panel = load_panel()

    model_cols = list(v.columns)
    # weight schedules
    scheds = {c: pd.Series(sizing_weight(v[c].to_numpy(), cfg), index=v.index) for c in model_cols}

    # benchmarks
    scheds["bench_buy_hold"] = pd.Series(1.0, index=v.index)
    k = cfg["frozen"]["target_vol"] / (qqq_ret.loc[eval_start:].std() * np.sqrt(252))
    scheds["bench_const_lev"] = pd.Series(float(k), index=v.index)
    rv_exp = panel["rv_daily"].expanding().mean()          # unconditional (expanding-mean) variance
    v_uncond = rv_exp.reindex(v.index, method="ffill")
    scheds["bench_uncond_vol"] = pd.Series(sizing_weight(v_uncond.to_numpy(), cfg), index=v.index)

    rows, eqs = {}, {}
    for name, ws in scheds.items():
        bt = run_backtest(ws, qqq_ret, dff, cfg)
        rows[name] = metrics(bt, cfg)
        eqs[name] = bt["equity"]

    M = pd.DataFrame(rows).T
    M.index.name = "strategy"
    os.makedirs(TAB, exist_ok=True)
    M.to_csv(os.path.join(TAB, "backtest_metrics.csv"))
    pd.DataFrame(eqs).to_parquet(os.path.join(PROC, "backtest_equity.parquet"))

    key = ["bench_buy_hold", "bench_const_lev", "gjr_skewt", "garch_skewt",
           "egarch_skewt", "ewma", "har", "rv", "rfsv", "bench_uncond_vol"]
    show = ["realized_vol", "vol_of_vol", "mad_roll_vol_from_target", "sharpe_excess",
            "max_drawdown", "cagr", "avg_annual_turnover", "pct_months_at_cap",
            "regt_breach_months"]
    print("=" * 108)
    print("LAYER-2 METRICS  (target vol 0.20; buy&hold + constant-lev are the bounds)")
    print("=" * 108)
    hdr = f"  {'strategy':17s}" + "".join(f"{s[:9]:>11s}" for s in show)
    print(hdr)
    for name in key + [c for c in M.index if c not in key]:
        m = M.loc[name]
        print(f"  {name:17s}" + "".join(
            (f"{m[s]:11.3f}" if abs(m[s]) < 100 else f"{m[s]:11.0f}") for s in show))

    print("\n" + "=" * 70)
    print("RFSV H-GRID: is the strategy outcome sensitive to H?")
    print("=" * 70)
    print(f"  {'variant':11s} {'real_vol':>9s} {'vov':>7s} {'MAD':>7s} {'sharpe':>7s} "
          f"{'maxDD':>7s} {'cagr':>7s} {'%cap':>6s}")
    for name in ["rfsv", "rfsv_h002", "rfsv_h005", "rfsv_h010", "rfsv_h015"]:
        m = M.loc[name]
        print(f"  {name:11s} {m['realized_vol']:9.3f} {m['vol_of_vol']:7.3f} "
              f"{m['mad_roll_vol_from_target']:7.3f} {m['sharpe_excess']:7.3f} "
              f"{m['max_drawdown']:7.3f} {m['cagr']:7.3f} {m['pct_months_at_cap']:6.1%}")
    rr = M.loc[["rfsv", "rfsv_h002", "rfsv_h005", "rfsv_h010", "rfsv_h015"]]
    print(f"\n  spread across the H grid: realized_vol {rr['realized_vol'].max()-rr['realized_vol'].min():.4f}, "
          f"sharpe {rr['sharpe_excess'].max()-rr['sharpe_excess'].min():.4f}, "
          f"cagr {rr['cagr'].max()-rr['cagr'].min():.4f}")

    _plots(eqs, M, cfg, key)


def _plots(eqs, M, cfg, key):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in key:
        ax.plot(eqs[name].index, eqs[name].values, lw=1.1, label=name)
    ax.set_yscale("log"); ax.set_title("Cumulative growth of $100,000 (log scale)")
    ax.set_ylabel("equity ($)"); ax.legend(ncol=3, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cumulative_growth.png"), dpi=130)
    plt.close(fig)

    # rolling 21d realized vol vs target
    eqdf = pd.DataFrame(eqs)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name in ["bench_buy_hold", "gjr_skewt", "ewma"]:
        rp = eqdf[name].pct_change()
        ax.plot(rp.index, rp.rolling(21).std() * np.sqrt(252), lw=0.8, label=name)
    ax.axhline(cfg["frozen"]["target_vol"], color="k", ls="--", lw=1, label="20% target")
    ax.set_title("Rolling 21-day realized portfolio volatility vs target")
    ax.set_ylabel("annualized vol"); ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rolling_vol.png"), dpi=130)
    plt.close(fig)
    print(f"\nfigures -> {FIG}/cumulative_growth.png, rolling_vol.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Subsample robustness: recompute the portfolio statistics over shorter histories that
drop the early crises. The full sample (2000) contains both the dot-com collapse and
2008; a 2003 start drops dot-com but keeps 2008; a 2010 start drops both. If the
volatility-targeting advantage over buy-and-hold survives in the later windows, it is
not an artifact of a single crash.

Reuses the exact backtest engine and metric formulas from run_backtest.py, so the full
window reproduces outputs/tables/backtest_metrics.csv (asserted below).

Writes outputs/tables/portfolio_summary_from2003.csv, _from2010.csv, and the matching
markdown tables under assets/ for the README.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.backtest.engine import run_backtest, metrics, sizing_weight   # noqa: E402

ROOT = repo_root()
PROC = os.path.join(ROOT, "data", "processed")
TAB = os.path.join(ROOT, "outputs", "tables")
ASSETS = os.path.join(ROOT, "assets")
ANNUAL = 252
START_CAPITAL = 100_000.0

NAME = {
    "garch_skewt": "GARCH(1,1)", "egarch_skewt": "EGARCH", "gjr_skewt": "GJR-GARCH",
    "ewma": "EWMA", "rv": "Realized variance", "har": "HAR-RV", "rfsv": "Rough volatility",
    "bench_buy_hold": "Buy & hold", "bench_const_lev": "Constant leverage",
    "bench_uncond_vol": "Unconditional vol",
}
ROWS = list(NAME)
WINDOWS = {"2003": "2003-01-01", "2010": "2010-01-01"}


def _load_raw(name, col):
    df = pd.read_parquet(os.path.join(ROOT, "data", "raw", f"{name}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[col]


def build_backtests():
    cfg = load_config()
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    v = pd.read_parquet(os.path.join(PROC, "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = _load_raw("qqq_daily", "close").pct_change().dropna()
    dff = _load_raw("dff", "dff")
    panel = load_panel()

    scheds = {c: pd.Series(sizing_weight(v[c].to_numpy(), cfg), index=v.index) for c in v.columns}
    scheds["bench_buy_hold"] = pd.Series(1.0, index=v.index)
    k = cfg["frozen"]["target_vol"] / (qqq_ret.loc[eval_start:].std() * np.sqrt(ANNUAL))
    scheds["bench_const_lev"] = pd.Series(float(k), index=v.index)
    v_uncond = panel["rv_daily"].expanding().mean().reindex(v.index, method="ffill")
    scheds["bench_uncond_vol"] = pd.Series(sizing_weight(v_uncond.to_numpy(), cfg), index=v.index)

    bts = {name: run_backtest(ws, qqq_ret, dff, cfg) for name, ws in scheds.items()}
    return bts, cfg


def wmetrics(bt: dict, start) -> dict:
    eq = bt["equity"].loc[start:]
    ret = bt["ret"].loc[start:]
    ex = ret - bt["cash_ret"].loc[start:]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    growth = eq.iloc[-1] / eq.iloc[0]
    maxdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float(growth ** (1 / yrs) - 1)
    return {
        "cagr": cagr,
        "ann_vol": float(ret.std() * np.sqrt(ANNUAL)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(ANNUAL)),
        "max_dd": maxdd,
        "calmar": cagr / abs(maxdd),
        "final_100k": START_CAPITAL * growth,
    }


def _money(x: float) -> str:
    return f"${x/1e6:.2f}M" if x >= 1e6 else (f"${x/1e3:.0f}K" if x >= 1e3 else f"${x:,.0f}")


def table(bts: dict, start) -> pd.DataFrame:
    rec = []
    for s in ROWS:
        m = wmetrics(bts[s], start)
        rec.append({
            "Strategy": NAME[s],
            "CAGR": f"{m['cagr']:.1%}",
            "Ann. volatility": f"{m['ann_vol']:.1%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Max drawdown": f"{m['max_dd']:.0%}",
            "Calmar": f"{m['calmar']:.2f}",
            "Final value of $100k": _money(m["final_100k"]),
        })
    return pd.DataFrame(rec)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join([":--"] + [":-:"] * (len(cols) - 1)) + "|"]
    for i, r in df.iterrows():
        if i == 7:  # blank row separating the seven models from the three benchmarks
            lines.append("|" + " | ".join([""] * len(cols)) + "|")
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    bts, cfg = build_backtests()
    eval_start = cfg["frozen"]["eval_start"]

    # validate: full-window recompute must reproduce the committed metrics
    bm = pd.read_csv(os.path.join(TAB, "backtest_metrics.csv")).set_index("strategy")
    worst = 0.0
    for s in ROWS:
        m = wmetrics(bts[s], eval_start)
        for k, col in [("sharpe", "sharpe_excess"), ("cagr", "cagr"), ("max_dd", "max_drawdown")]:
            worst = max(worst, abs(m[k] - float(bm.loc[s, col])))
    assert worst < 1e-9, f"full-window recompute drifts from backtest_metrics by {worst:.2e}"
    print(f"full-window validation OK (max abs diff vs backtest_metrics = {worst:.2e})")

    for tag, start in WINDOWS.items():
        df = table(bts, start)
        df.to_csv(os.path.join(TAB, f"portfolio_summary_from{tag}.csv"), index=False)
        md = to_markdown(df)
        with open(os.path.join(ASSETS, f"portfolio_summary_from{tag}.md"), "w") as f:
            f.write(md)
        span = bts["rv"]["equity"].loc[start:]
        print(f"\n=== From {tag} ({span.index[0].date()} to {span.index[-1].date()}) ===\n{md}")


if __name__ == "__main__":
    main()

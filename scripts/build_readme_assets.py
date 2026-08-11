"""
Reader-facing performance assets: a static growth chart and a clean portfolio
summary table, both regenerated from cached backtest output (no recomputation).

Outputs
  assets/performance.png              growth of $100,000, seven models vs buy & hold
  outputs/tables/portfolio_summary.csv  headline portfolio statistics, all strategies
  assets/portfolio_summary.md         the same table as GitHub-flavored markdown

Run:  .venv/bin/python scripts/build_readme_assets.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ = os.path.join(ROOT, "data", "processed", "backtest_equity.parquet")
BM = os.path.join(ROOT, "outputs", "tables", "backtest_metrics.csv")
RR = os.path.join(ROOT, "outputs", "tables", "layer2_riskreturn.csv")
AD = os.path.join(ROOT, "outputs", "tables", "layer2_adherence.csv")
ASSETS = os.path.join(ROOT, "assets")
TABLES = os.path.join(ROOT, "outputs", "tables")
os.makedirs(ASSETS, exist_ok=True)

START_CAPITAL = 100_000.0

# clean, reader-facing labels (no internal jargon) and the Okabe-Ito colorblind palette
NAME = {
    "garch_skewt": "GARCH(1,1)", "egarch_skewt": "EGARCH", "gjr_skewt": "GJR-GARCH",
    "ewma": "EWMA", "rv": "Realized variance", "har": "HAR-RV", "rfsv": "Rough volatility",
    "bench_buy_hold": "Buy & hold", "bench_const_lev": "Constant leverage",
    "bench_uncond_vol": "Unconditional vol",
}
COLOR = {
    "garch_skewt": "#0072B2", "egarch_skewt": "#56B4E9", "gjr_skewt": "#009E73",
    "ewma": "#E69F00", "rv": "#CC79A7", "har": "#000000", "rfsv": "#D55E00",
    "bench_buy_hold": "#B0B0B0", "bench_const_lev": "#8C8C8C", "bench_uncond_vol": "#C7C7C7",
}
# solid for conditional-variance dynamics, dashed/dotted for the naive resizers
LS = {"ewma": "--", "rv": "--", "rfsv": ":"}

MODELS = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv"]
BENCH = ["bench_buy_hold", "bench_const_lev", "bench_uncond_vol"]
ROWS = MODELS + BENCH


def _money(x: float) -> str:
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:,.0f}"


def build_table() -> pd.DataFrame:
    bm = pd.read_csv(BM).set_index("strategy")
    rr = pd.read_csv(RR).rename(columns={"Unnamed: 0": "strategy"}).set_index("strategy")
    ad = pd.read_csv(AD).rename(columns={"Unnamed: 0": "strategy"}).set_index("strategy")
    rec = []
    for s in ROWS:
        fin = float(bm.loc[s, "final_equity"])
        rec.append({
            "Strategy": NAME[s],
            "Final value of $100k": _money(fin),
            "Total return": f"{fin/START_CAPITAL - 1:.0%}",
            "CAGR": f"{bm.loc[s, 'cagr']:.1%}",
            "Ann. volatility": f"{bm.loc[s, 'realized_vol']:.1%}",
            "Sharpe": f"{bm.loc[s, 'sharpe_excess']:.2f}",
            "Max drawdown": f"{bm.loc[s, 'max_drawdown']:.0%}",
            "Calmar": f"{rr.loc[s, 'calmar']:.2f}",
            "Longest drawdown (yrs)": f"{rr.loc[s, 'days_underwater_longest']/252:.1f}",
            "Vol-target error": f"{ad.loc[s, 'mad_nonoverlap']:.3f}",
        })
    return pd.DataFrame(rec)


def write_markdown(df: pd.DataFrame) -> str:
    cols = ["Strategy", "CAGR", "Ann. volatility", "Sharpe", "Max drawdown",
            "Calmar", "Final value of $100k"]
    d = df[cols]
    head = "| " + " | ".join(cols) + " |"
    align = "|" + "|".join([":--"] + [":-:"] * (len(cols) - 1)) + "|"
    lines = [head, align]
    for i, r in d.iterrows():
        cells = [str(r[c]) for c in cols]
        # visually separate the three benchmark rows from the seven models
        if i == len(MODELS):
            lines.append("|" + " | ".join([""] * len(cols)) + "|")
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    with open(os.path.join(ASSETS, "portfolio_summary.md"), "w") as f:
        f.write(md + "\n")
    return md


def build_figure() -> None:
    eq = pd.read_parquet(EQ)
    eq = eq / eq.iloc[0] * START_CAPITAL          # every strategy starts at $100,000
    shown = MODELS + ["bench_buy_hold"]
    order = sorted(shown, key=lambda s: -eq[s].iloc[-1])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": "#444444", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(10.5, 5.9), dpi=170)
    fig.subplots_adjust(top=0.85, left=0.085, right=0.975, bottom=0.095)

    for s in shown:
        is_bench = s == "bench_buy_hold"
        ax.plot(eq.index, eq[s], color=COLOR[s],
                lw=1.5 if is_bench else 1.0,
                ls="-" if is_bench else LS.get(s, "-"),
                zorder=3 if is_bench else 2, label=NAME[s])

    ax.set_yscale("log")
    ax.set_yticks([1e4, 1e5, 1e6])
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(1e4, 4e6)                         # log floor at 10^4; start is 10^5
    ax.grid(True, which="major", axis="both", color="#E6E6E6", lw=0.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#F5F5F5", lw=0.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("Portfolio value (log scale, $100,000 start)")

    fig.text(0.085, 0.965, "Growth of $100,000 in a volatility-targeted QQQ, 2000-2026",
             ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.085, 0.900, "Seven volatility models sizing a 20% target, against buy-and-hold",
             ha="left", va="top", fontsize=10.5, color="#555555")

    handles, labels = ax.get_legend_handles_labels()
    idx = [labels.index(NAME[s]) for s in order]
    ax.legend([handles[i] for i in idx], [labels[i] for i in idx],
              loc="upper left", frameon=False, fontsize=9.5, labelspacing=0.5,
              handlelength=2.4, borderaxespad=0.8)
    out = os.path.join(ASSETS, "performance.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    df = build_table()
    df.to_csv(os.path.join(TABLES, "portfolio_summary.csv"), index=False)
    print("wrote", os.path.join(TABLES, "portfolio_summary.csv"))
    md = write_markdown(df)
    build_figure()
    print("\n--- markdown table ---\n" + md)

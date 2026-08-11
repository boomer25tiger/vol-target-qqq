"""
Regenerate every write-up figure from cached data using the single shared visual
standard in src/volteq/viz/style.py.

MECHANICAL BUILD ONLY. This script computes no model results and changes no cached
number. It reads tables/parquets on disk, applies the style module, and writes PNGs
to figures/. Titles on the canvas are short axis-level quantity names; the
descriptive claim lives in the markdown caption added later (figure_style.md).

Every series is coloured through style.style_for / style.COLOR, ordered with
style.canonical_sort, and labelled with the reader-facing style.DISPLAY name. Three
documented exceptions to the model palette (all still colourblind-safe by lightness +
a second channel):
  - har_coef_path: three WITHIN-model HAR coefficients (not the 11 canonical series)
    -> a small distinct 3-colour set, each with its own line style.
  - return_gap_decomposition: five gap COMPONENTS (not model series) -> a distinct
    categorical set, each with its own hatch as a second channel.
  - sharpe_lw_pmatrix: a p-value matrix -> a sequential (cividis) colormap, which is
    itself perceptually uniform / CB-safe.
Run:  OMP_NUM_THREADS=1 .venv/bin/python scripts/build_figures.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# --- paths (robust to cwd; agent shells reset between calls) -----------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from volteq.viz import style                       # noqa: E402
from volteq.config import load_config              # noqa: E402
from volteq.backtest.engine import sizing_weight   # noqa: E402

DATA = os.path.join(REPO, "data", "processed")
TABLES = os.path.join(REPO, "outputs", "tables")
FIGDIR = os.path.join(REPO, "figures")
os.makedirs(FIGDIR, exist_ok=True)

style.apply(matplotlib)          # set rcParams once (dpi 200 savefig, DejaVu Sans, ...)
ANNUAL = 252
_WRITTEN: list[str] = []


# --- id / column helpers -----------------------------------------------------
def to_id(col: str) -> str:
    """Cached tables prefix benchmarks with 'bench_'; the style module keys on the
    canonical id (buy_hold, const_lev, uncond_vol). Strip the prefix."""
    return col[len("bench_"):] if col.startswith("bench_") else col


def order_cols(cols):
    """[(canonical_id, raw_column)] in canonical order for arbitrary table columns."""
    id2col = {}
    for c in cols:
        id2col[to_id(c)] = c
    return [(i, id2col[i]) for i in style.canonical_sort(list(id2col.keys()))]


def raw_for(i: str, available: set) -> str | None:
    """Resolve canonical id -> the raw label present in a table (id or 'bench_'+id)."""
    if i in available:
        return i
    if "bench_" + i in available:
        return "bench_" + i
    return None


def line_kw(i: str, markevery=None) -> dict:
    """style_for kwargs for a line. Drop the per-point marker on dense series unless a
    markevery spacing is supplied (a sparse redundant channel that aids CB separation)."""
    kw = style.style_for(i)
    if markevery is None:
        kw.pop("marker", None)
        kw.pop("markeredgecolor", None)
    else:
        kw["markevery"] = markevery
    return kw


def scatter_kw(i: str) -> dict:
    """color/marker/label for a scatter point (the marker is the redundant CB channel)."""
    s = style.style_for(i)
    kw = {"color": s["color"], "marker": s["marker"] or "o", "label": s["label"]}
    if "markeredgecolor" in s:          # weak yellow rv -> dark edge
        kw["edgecolors"] = s["markeredgecolor"]
    else:
        kw["edgecolors"] = "#333333"
    return kw


def save(fig, name: str):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)               # style sets dpi=200, bbox='tight', white bg
    plt.close(fig)
    _WRITTEN.append(name)
    print(f"  wrote {name}")


# ============================================================ 1. REBUILDS =====
def fig_cumulative_growth():
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    for i, col in order_cols(eq.columns):        # all 14 model paths + 3 benchmarks
        ax.plot(eq.index, eq[col], **line_kw(i))
    ax.set_yscale("log")
    ax.set_ylabel("$ (log scale)")
    ax.set_xlabel("")
    ax.set_title("Cumulative growth of $100,000")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5,
              fontsize=7.2, handlelength=2.2, columnspacing=1.1)
    save(fig, "cumulative_growth.png")


def _longest_spell(e):
    """(start_date, end_date, n_days) of the longest consecutive spell below a prior peak."""
    below = (e < e.cummax() - 1e-9).to_numpy()
    best = (0, -1); cur = None
    for k, x in enumerate(below):
        if x and cur is None:
            cur = k
        if (not x) and cur is not None:
            if (k - 1 - cur) > (best[1] - best[0]):
                best = (cur, k - 1)
            cur = None
    if cur is not None and (len(below) - 1 - cur) > (best[1] - best[0]):
        best = (cur, len(below) - 1)
    return e.index[best[0]], e.index[best[1]], best[1] - best[0] + 1


def fig_drawdown():
    """Underwater curves per strategy (top) with each strategy's longest spell below a prior
    peak marked as a bar (bottom). Serves drawdown depth (compression) AND duration/recovery:
    buy-hold 14.9y underwater vs targeting 6.8-10.0y (N6)."""
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    subset = ["bench_buy_hold", "gjr_skewt", "har", "ewma", "rv"]
    ordered = order_cols(subset)
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(8.5, 4.8), height_ratios=[3, 1.15],
                                  sharex=True)
    for i, col in ordered:
        e = eq[col]
        ax.plot(eq.index, e / e.cummax() - 1.0, **line_kw(i))
    ax.axhline(0.0, color="#888888", linewidth=0.6)
    ax.set_ylabel("drawdown")
    ax.set_title("Drawdown (underwater curve) with longest spell below a prior peak")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8,
              columnspacing=1.3)
    # bottom: longest-spell bars (a small gantt), one row per strategy, top-to-bottom canonical
    for row, (i, col) in enumerate(ordered):
        s, en, n = _longest_spell(eq[col])
        y = len(ordered) - 1 - row
        c = style.style_for(i)["color"]
        axg.hlines(y, s, en, color=c, linewidth=7,
                   path_effects=[] , alpha=0.9)
        axg.text(en, y, f"  {n/252:.1f}y", va="center", ha="left", fontsize=7.5, color="#333333")
    axg.set_yticks(range(len(ordered)))
    axg.set_yticklabels([style.style_for(i)["label"] for i, _ in ordered][::-1], fontsize=7.5)
    axg.set_ylim(-0.6, len(ordered) - 0.4)
    axg.set_xlabel("")
    axg.set_ylabel("longest spell", fontsize=8)
    axg.grid(axis="y", visible=False)
    save(fig, "drawdown.png")


def fig_rolling_vol():
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    # Representative subset (stated per the request): buy_hold + the 4 dynamics models
    # (gjr/garch/egarch skew-t, har) + ewma. Full canonical set is too dense to read.
    subset = ["bench_buy_hold", "gjr_skewt", "garch_skewt", "egarch_skewt", "har", "ewma"]
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    for i, col in order_cols(subset):
        r = eq[col].pct_change()
        roll = r.rolling(21).std() * np.sqrt(ANNUAL)
        ax.plot(eq.index, roll, **line_kw(i, markevery=650))
    ax.axhline(style.TARGET_VOL, color="black", linestyle="--", linewidth=1.1,
               label=f"Target {style.TARGET_VOL:.2f}")
    ax.set_ylim(0.0, 0.9)
    ax.set_ylabel("Realized volatility (ann.)")
    ax.set_xlabel("")
    ax.set_title("Rolling 21-day realized volatility")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=8, columnspacing=1.3)
    save(fig, "rolling_vol.png")


def fig_persistence_path():
    pp = pd.read_parquet(os.path.join(DATA, "param_paths.parquet"))
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    xmin = xmax = None
    for m in ["garch", "egarch", "gjr"]:
        sub = pp[(pp.model == m) & (pp.dist == "skewt")].sort_values("date")
        ax.plot(sub.date, sub.persistence, **line_kw(m, markevery=26))
        xmin = sub.date.min() if xmin is None else min(xmin, sub.date.min())
        xmax = sub.date.max() if xmax is None else max(xmax, sub.date.max())
    ek = line_kw("ewma")                                   # ewma reference: phi = 1 (IGARCH)
    ax.plot([xmin, xmax], [1.0, 1.0], **ek)
    ax.set_ylim(0.90, 1.02)
    ax.set_ylabel(r"persistence $\varphi$")
    ax.set_title("Fitted persistence over refits")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    save(fig, "persistence_path.png")


def fig_lambda21_path():
    pp = pd.read_parquet(os.path.join(DATA, "param_paths.parquet"))
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    xmin = xmax = None
    for m in ["garch", "gjr"]:            # closed-form lambda_h only (egarch aggregates by MC)
        sub = pp[(pp.model == m) & (pp.dist == "skewt")].sort_values("date")
        phi = sub.persistence.to_numpy()
        lam = (1.0 - phi ** 21) / (21.0 * (1.0 - phi))
        ax.plot(sub.date, lam, **line_kw(m, markevery=26))
        xmin = sub.date.min() if xmin is None else min(xmin, sub.date.min())
        xmax = sub.date.max() if xmax is None else max(xmax, sub.date.max())
    ek = line_kw("ewma")                                   # ewma flat at 1.0 (lambda_h == 1)
    ax.plot([xmin, xmax], [1.0, 1.0], **ek)
    ax.set_ylabel(r"$\lambda_{h=21}$")
    ax.set_title("h=21 aggregation weight")
    ax.legend(ncol=3, fontsize=8, loc="best")
    save(fig, "lambda21_path.png")


def fig_har_coef_path():
    hc = pd.read_parquet(os.path.join(DATA, "har_coef_path.parquet"))
    # DOCUMENTED EXCEPTION: the three lines are within-model HAR daily/weekly/monthly
    # coefficients, not the 11 canonical series -> a small distinct 3-colour set
    # (ColorBrewer Dark2, CB-safe) with a distinct line style each as a second channel.
    trio = [("beta_d", "Daily", "#1B9E77", "-"),
            ("beta_w", "Weekly", "#7570B3", "--"),
            ("beta_m", "Monthly", "#D95F02", "-.")]
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    for col, label, color, ls in trio:
        ax.plot(hc.date, hc[col], color=color, linestyle=ls, linewidth=1.5, label=label)
    ax.set_ylabel("HAR coefficient")
    ax.set_title("HAR component coefficients over refits")
    ax.legend(ncol=3, fontsize=9, loc="best")
    save(fig, "har_coef_path.png")


def fig_hurst_path():
    hp = pd.read_parquet(os.path.join(DATA, "rfsv_hurst_path.parquet"))
    c = style.COLOR["rfsv"]                                 # recolour single line to rfsv hue
    fig, ax = plt.subplots(figsize=style.FIG_WIDE)
    ax.fill_between(hp.date, hp.H - 2 * hp.H_se, hp.H + 2 * hp.H_se,
                    color=c, alpha=0.20, linewidth=0, label=r"$\pm 2$ SE")
    ax.plot(hp.date, hp.H, color=c, linewidth=1.6, label=style.DISPLAY["rfsv"] + r" $H$")
    ax.axhline(0.5, color="#444444", linestyle=":", linewidth=1.2, label="H = 0.5")
    ax.set_ylim(0.0, 0.55)
    ax.set_ylabel(r"Hurst $H$")
    ax.set_title("Estimated Hurst exponent over refits")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    save(fig, "hurst_path.png")


def fig_h_stage_spread():
    hs = pd.read_csv(os.path.join(TABLES, "h_stage_spread.csv"))
    # NOTE: the on-disk CSV holds five PIPELINE STAGES (V -> sigma -> w_unclipped ->
    # w_clipped -> realized_vol) with mean/p95 cross-model spread; it has no rfsv-grid
    # variant columns, so the "grid style" note is inapplicable here. Stages are not the
    # 11 canonical series, so this is a within-figure diagnostic: two neutral fills.
    stage_lbl = {"1_V": "V(21)", "2_sigma": r"$\sigma$", "3_w_unclipped": "w (unclipped)",
                 "4_w_clipped": "w (clipped)", "5_realized_vol": "realized vol"}
    labels = [stage_lbl.get(s, s) for s in hs["Unnamed: 0"]]
    x = np.arange(len(hs))
    w = 0.38
    fig, ax = plt.subplots(figsize=style.FIG_BAR)
    ax.bar(x - w / 2, hs["mean_spread"], w, color="#4E79A7", label="mean spread")
    ax.bar(x + w / 2, hs["p95_spread"], w, color="#9C9C9C", label="p95 spread")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cross-model spread (log)")
    ax.set_title("Cross-model spread by pipeline stage")
    ax.legend(fontsize=9, loc="best")
    save(fig, "h_stage_spread.png")


# ====================================================== 2. DRAFT STATICS ======
PRIMARY8 = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har",
            "rfsv", "trailing_rv21"]


def _mcs_retained_primary_qlike():
    mcs = pd.read_csv(os.path.join(TABLES, "mcs_results.csv"))
    pq = mcs[(mcs["set"] == "primary") & (mcs["loss"] == "qlike")]
    return pq.groupby("model")["retained"].all()           # stable across blocks {2,5,10}


def fig_layer1_qlike_mcs():
    l1 = pd.read_csv(os.path.join(TABLES, "layer1_losses.csv")).set_index("model")
    retained = _mcs_retained_primary_qlike()
    order = style.canonical_sort(PRIMARY8)
    fig, ax = plt.subplots(figsize=style.FIG_BAR)
    for j, i in enumerate(order):
        s = style.style_for(i)
        keep = bool(retained.get(i, False))
        if keep:
            ax.bar(j, l1.loc[i, "qlike_mean"], color=s["color"], edgecolor="#333333",
                   linewidth=0.6)
        else:                                              # excluded -> hollow + hatch
            # weak yellow (rv) uses the module's designated dark edge so it stays visible
            ax.bar(j, l1.loc[i, "qlike_mean"], facecolor="white",
                   edgecolor=s.get("markeredgecolor", s["color"]), linewidth=1.6, hatch="////")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([style.style_for(i)["label"] for i in order], rotation=35,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("QLIKE (mean)")
    ax.set_title("Mean QLIKE by model")
    handles = [Patch(facecolor="#777777", edgecolor="#333333", label="MCS-retained"),
               Patch(facecolor="white", edgecolor="#777777", hatch="////",
                     label="MCS-excluded")]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    save(fig, "layer1_qlike_mcs.png")


def _adherence_mcs_retained():
    """{block_days: set(retained ids)} from the adherence MCS (Layer-2 analog of the
    QLIKE MCS)."""
    m = pd.read_csv(os.path.join(TABLES, "adherence_mcs.csv"))
    return {int(r["block_days"]): set(str(r["retained"]).split(",")) for _, r in m.iterrows()}


def fig_adherence_intervals():
    """Per-strategy adherence MAD with 90% bootstrap CI and adherence-MCS membership,
    the Layer-2 counterpart of fig_layer1_qlike_mcs: solid = retained, hollow+hatch =
    excluded. Two panels, the 42- and 105-day blocks that resolve the finer separation;
    the 210-day block (all eight retained) is stated in the caption, not drawn."""
    inf = pd.read_csv(os.path.join(TABLES, "adherence_inference.csv"))
    retained = _adherence_mcs_retained()
    order = style.canonical_sort(PRIMARY8)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4), sharey=True)
    for ax, bd in zip(axes, (42, 105)):
        sub = inf[inf["block_days"] == bd].set_index("strategy")
        keep = retained.get(bd, set())
        for j, i in enumerate(order):
            s = style.style_for(i)
            pt, lo, hi = (float(sub.loc[i, c]) for c in ("point", "ci90_lo", "ci90_hi"))
            if i in keep:
                ax.bar(j, pt, color=s["color"], edgecolor="#333333", linewidth=0.6)
            else:                                          # excluded -> hollow + hatch
                ax.bar(j, pt, facecolor="white", edgecolor=s.get("markeredgecolor", s["color"]),
                       linewidth=1.6, hatch="////")
            ax.errorbar(j, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor="#333333",
                        capsize=3, linewidth=1.1, zorder=5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([style.style_for(i)["label"] for i in order], rotation=35,
                           ha="right", fontsize=8)
        ax.set_title(f"{bd}-day blocks", fontsize=10)
    axes[0].set_ylabel("adherence MAD (non-overlap)")
    axes[0].set_ylim(0, 0.062)
    handles = [Patch(facecolor="#777777", edgecolor="#333333", label="MCS-retained"),
               Patch(facecolor="white", edgecolor="#777777", hatch="////", label="MCS-excluded"),
               Line2D([0], [0], color="#333333", lw=1.1, label="90% bootstrap CI")]
    fig.suptitle("Adherence MAD by model", fontsize=10.5, y=0.99)
    fig.legend(handles=handles, fontsize=8, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.945), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "adherence_intervals.png")


def fig_loss_concentration():
    lc = pd.read_csv(os.path.join(TABLES, "loss_concentration.csv"))
    lc = lc[lc["sample"] == "primary"].set_index("model")
    order = style.canonical_sort(list(lc.index))
    x = np.arange(len(order))
    w = 0.40
    fig, ax = plt.subplots(figsize=style.FIG_BAR)
    for j, i in enumerate(order):
        c = style.style_for(i)["color"]
        ax.bar(j - w / 2, lc.loc[i, "mse_top1pct_share"], w, color=c, edgecolor="#333333",
               linewidth=0.5)
        ax.bar(j + w / 2, lc.loc[i, "qlike_top1pct_share"], w, color=c, edgecolor="#333333",
               linewidth=0.5, hatch="////", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([style.style_for(i)["label"] for i in order], rotation=35,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("top-1% share of total loss")
    ax.set_title("Loss concentration in the top 1%")
    handles = [Patch(facecolor="#777777", edgecolor="#333333", label="MSE"),
               Patch(facecolor="#777777", edgecolor="#333333", hatch="////", label="QLIKE")]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    save(fig, "loss_concentration.png")


def fig_qlike_mse_rank():
    # Re-rank WITHIN the 8 primary per sample (the on-disk qlike_rank/mse_rank rank over
    # the full 15-row set). loss_concentration.csv carries mean QLIKE/MSE for both samples,
    # so both panels reproduce the H1 within-8 Spearman coefficients (0.881 / 0.619).
    lc = pd.read_csv(os.path.join(TABLES, "loss_concentration.csv"))
    order = style.canonical_sort(PRIMARY8)
    n = 8
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    for ax, samp, title in [(axA, "primary", "Primary (monthly)"),
                            (axB, "secondary", "Secondary (daily)")]:
        s = lc[lc["sample"] == samp].set_index("model").loc[list(order)]
        qr = s["qlike_full"].rank()
        mr = s["mse_full"].rank()
        d = (qr - mr).to_numpy()
        rho = 1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1))       # Spearman, n=8
        ax.plot([1, n], [1, n], color="#BBBBBB", linestyle="--", linewidth=1.0, zorder=1)
        for i in order:
            ax.scatter(qr[i], mr[i], s=70, zorder=3, linewidths=0.8, **scatter_kw(i))
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.set_xlabel("QLIKE rank (1 = best, of 8)")
        ax.set_title(title)
        ax.annotate(f"Spearman +{rho:.3f}", xy=(0.04, 0.96), xycoords="axes fraction",
                    va="top", ha="left", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    axA.set_ylabel("MSE rank (1 = best, of 8)")
    handles = [Line2D([0], [0], marker=style.style_for(i)["marker"] or "o", linestyle="none",
                      color=style.style_for(i)["color"], markeredgecolor="#333333",
                      label=style.style_for(i)["label"]) for i in order]
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    save(fig, "qlike_mse_rank.png")


def fig_cross_layer_scatter():
    cl = pd.read_csv(os.path.join(TABLES, "cross_layer_ranks.csv")).rename(
        columns={"Unnamed: 0": "model"}).set_index("model")
    order = style.canonical_sort(list(cl.index))
    n = len(order)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 4.4))
    # note goes in each panel's data-free corner: +corr fills bottom-left->top-right,
    # so its empty corner is bottom-right; -corr fills top-left->bottom-right, empty
    # corner bottom-left.
    for ax, ycol, note, nxy, nha in [
        (axA, "adherence_rank", "Spearman +0.833\nn=8, descriptive", (0.96, 0.04), "right"),
        (axB, "sharpe_rank", "Spearman -0.690\nn=8, descriptive", (0.04, 0.04), "left"),
    ]:
        ax.plot([1, n], [1, n], color="#DDDDDD", linestyle="--", linewidth=1.0, zorder=1)
        for i in order:
            ax.scatter(cl.loc[i, "qlike_rank"], cl.loc[i, ycol], s=60, zorder=3,
                       linewidths=0.7, **scatter_kw(i))
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(0.5, n + 0.5)
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.set_xlabel("QLIKE rank (1 = best)")
        ax.annotate(note, xy=nxy, xycoords="axes fraction", va="bottom", ha=nha,
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    axA.set_ylabel("adherence-MAD rank (1 = best)")
    axB.set_ylabel("Sharpe rank (1 = best)")
    axA.set_title("Accuracy vs adherence")
    axB.set_title("Accuracy vs Sharpe")
    handles = [Line2D([0], [0], marker=style.style_for(i)["marker"] or "o", linestyle="none",
                      color=style.style_for(i)["color"], markeredgecolor="#333333",
                      label=style.style_for(i)["label"]) for i in order]
    fig.tight_layout(rect=(0, 0.14, 1, 1))          # reserve a band so the legend clears x-labels
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    save(fig, "cross_layer_scatter.png")


def fig_return_gap_decomposition():
    rg = pd.read_csv(os.path.join(TABLES, "return_gap_decomposition.csv")).set_index("strategy")
    order = order_cols(list(rg.index))          # 14 models + const_lev + uncond_vol (no buy_hold)
    # DOCUMENTED EXCEPTION: segments are the five gap COMPONENTS, not model series ->
    # a distinct categorical set (CB-safe by lightness) each with its own hatch.
    comps = [("market_exposure", "market exposure", "#4E79A7", ""),
             ("vol_timing", "vol timing", "#59A14F", ""),
             ("financing", "financing", "#E15759", "///"),
             ("transaction_cost", "transaction cost", "#B07AA1", "xxx"),
             ("residual", "residual", "#BAB0AC", "...")]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for j, (i, col) in enumerate(order):
        pos = neg = 0.0
        for key, _lbl, color, hatch in comps:
            v = float(rg.loc[col, key])
            bottom = pos if v >= 0 else neg
            ax.bar(j, v, bottom=bottom, color=color, edgecolor="#444444", linewidth=0.4,
                   hatch=hatch, width=0.72)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.plot(j, float(rg.loc[col, "gap_ann"]), marker="D", color="black", markersize=4,
                zorder=5)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([style.style_for(i)["label"] for i, _ in order], rotation=40,
                       ha="right", fontsize=7.8)
    ax.set_ylabel("annualized return gap")
    ax.set_title("Return-gap decomposition")
    handles = [Patch(facecolor=c, edgecolor="#444444", hatch=h, label=lbl)
               for _k, lbl, c, h in comps]
    handles.append(Line2D([0], [0], marker="D", linestyle="none", color="black",
                          label="total gap"))
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper right")
    save(fig, "return_gap_decomposition.png")


def fig_sharpe_lw_pmatrix():
    pm = pd.read_csv(os.path.join(TABLES, "sharpe_inference_lw_pmatrix.csv")).set_index(
        "Unnamed: 0")
    pairs = order_cols(list(pm.columns))                   # canonical order, bench_ stripped
    ids = [i for i, _ in pairs]
    raw = [c for _, c in pairs]
    labels = [style.style_for(i)["label"] for i in ids]
    M = pm.loc[raw, raw].to_numpy(dtype=float)
    n = len(raw)
    fig, ax = plt.subplots(figsize=style.FIG_SQUARE)
    # DOCUMENTED EXCEPTION: a p-value matrix legitimately uses a sequential colormap.
    # cividis is perceptually uniform and colourblind-safe.
    im = ax.imshow(M, cmap="cividis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.contour(M, levels=[0.05], colors="#D55E00", linewidths=1.1)   # 0.05 significance edge
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.6)
    ax.set_yticklabels(labels, fontsize=5.6)
    ax.set_title("Ledoit-Wolf Sharpe p-values")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("p-value", fontsize=9)
    cbar.ax.axhline(0.05, color="#D55E00", linewidth=1.4)
    cbar.ax.annotate("0.05", xy=(1.0, 0.05), xytext=(2.6, 0.05), xycoords=("axes fraction",
                     "data"), va="center", fontsize=7, color="#D55E00")
    save(fig, "sharpe_lw_pmatrix.png")


def fig_benchmark_comparison():
    la = pd.read_csv(os.path.join(TABLES, "layer2_adherence.csv")).rename(
        columns={"Unnamed: 0": "id"}).set_index("id")
    si = pd.read_csv(os.path.join(TABLES, "sharpe_inference.csv")).set_index("strategy")
    # trailing_rv21 postdates layer2_adherence.csv; its adherence MAD lives in cross_layer_ranks.
    cl = pd.read_csv(os.path.join(TABLES, "cross_layer_ranks.csv")).rename(
        columns={"Unnamed: 0": "model"}).set_index("model")
    models = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv"]
    benches = ["buy_hold", "const_lev", "trailing_rv21", "uncond_vol"]
    order = style.canonical_sort(models + benches)      # 11 ids, shared by both panels

    def val_adh(i):
        raw = raw_for(i, set(la.index))
        if raw is not None and pd.notna(la.loc[raw, "mad_nonoverlap"]):
            return float(la.loc[raw, "mad_nonoverlap"])
        if i in cl.index and pd.notna(cl.loc[i, "adherence_mad"]):   # trailing_rv21 fallback
            return float(cl.loc[i, "adherence_mad"])
        return None

    def val_shp(i):
        raw = raw_for(i, set(si.index))
        return float(si.loc[raw, "sharpe_excess_ann"]) if raw is not None else None

    def panel(ax, valfn, xlabel):
        ids = [i for i in order if valfn(i) is not None]
        ys = np.arange(len(ids))[::-1]                     # canonical top-to-bottom
        for y, i in zip(ys, ids):
            val = valfn(i)
            s = style.style_for(i)
            ax.hlines(y, 0, val, color=s["color"], linewidth=1.6, alpha=0.8)
            ax.scatter(val, y, s=60, color=s["color"], marker=s["marker"] or "o",
                       edgecolors=s.get("markeredgecolor", "#333333"), zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([style.style_for(i)["label"] for i in ids], fontsize=8)
        ax.set_xlabel(xlabel)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.4))
    panel(axA, val_adh, "adherence MAD (non-overlap)")
    panel(axB, val_shp, "Sharpe (excess, ann.)")
    axA.set_title("Target adherence")
    axB.set_title("Sharpe ratio")
    fig.tight_layout(w_pad=3.0)
    save(fig, "benchmark_comparison.png")


def _nonoverlap_block_vol(equity: pd.Series, block: int = 21) -> np.ndarray:
    r = equity.pct_change().dropna().to_numpy()
    n = (len(r) // block) * block
    blocks = r[:n].reshape(-1, block)
    return blocks.std(axis=1, ddof=1) * np.sqrt(ANNUAL)


def fig_adherence_distribution():
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    # Representative strategies (trailing_rv21 not in this panel): three models spanning
    # dynamics/naive/rough + the three benchmarks present in the equity panel.
    reps = ["gjr_skewt", "ewma", "rfsv", "bench_buy_hold", "bench_const_lev",
            "bench_uncond_vol"]
    pairs = order_cols(reps)
    data = [_nonoverlap_block_vol(eq[col]) for _, col in pairs]
    ids = [i for i, _ in pairs]
    ypos = np.arange(len(ids))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    parts = ax.violinplot(data, positions=ypos, orientation="horizontal",
                          showmeans=False, showextrema=False, widths=0.85)
    for body, i in zip(parts["bodies"], ids):
        body.set_facecolor(style.style_for(i)["color"])
        body.set_edgecolor("#333333")
        body.set_alpha(0.75)
    ax.axvline(style.TARGET_VOL, color="black", linestyle="--", linewidth=1.2,
               label=f"target {style.TARGET_VOL:.2f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([style.style_for(i)["label"] for i in ids], fontsize=8.5)
    ax.set_xlabel("21-day realized vol (ann., non-overlapping)")
    ax.set_title("Realized-volatility distribution vs target")
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "adherence_distribution.png")


def fig_leverage_distribution():
    cfg = load_config()
    fv = pd.read_parquet(os.path.join(DATA, "forecast_v21.parquet"))
    models = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv"]
    order = style.canonical_sort(models)
    data = [sizing_weight(fv[i].to_numpy(), cfg) for i in order]
    ypos = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    parts = ax.violinplot(data, positions=ypos, orientation="horizontal",
                          showmeans=False, showextrema=False, widths=0.85)
    for body, i in zip(parts["bodies"], order):
        body.set_facecolor(style.style_for(i)["color"])
        body.set_edgecolor("#333333")
        body.set_alpha(0.75)
    ax.axvline(style.LEVERAGE_CAP, color="black", linestyle="--", linewidth=1.2,
               label=f"cap {style.LEVERAGE_CAP:.1f}x")
    ax.axvline(1.0, color="#999999", linestyle="-", linewidth=0.9, label="fully invested")
    ax.axvline(0.0, color="#444444", linestyle=":", linewidth=1.0, label="floor 0")
    ax.set_xlim(-0.05, 2.15)
    ax.set_yticks(ypos)
    ax.set_yticklabels([style.style_for(i)["label"] for i in order], fontsize=8.5)
    ax.set_xlabel("leverage w")
    ax.set_title("Leverage distribution by strategy")
    ax.legend(fontsize=8, loc="lower right", ncol=1)
    save(fig, "leverage_distribution.png")


def fig_subperiod_panel():
    sp = pd.read_csv(os.path.join(TABLES, "layer2_subperiods.csv"))
    subs = ["2000-2002", "2008-2009", "2010-2019", "2020", "2022"]
    wanted = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv",
              "buy_hold", "const_lev", "uncond_vol"]
    avail = set(sp["strategy"])
    ids = [i for i in style.canonical_sort(wanted) if raw_for(i, avail)]
    ypos = {i: y for i, y in zip(ids, np.arange(len(ids))[::-1])}
    rows = [("realized_vol", "realized vol", True),
            ("mad_nonoverlap", "adherence MAD", False)]
    fig, axes = plt.subplots(len(rows), len(subs), figsize=(12.5, 5.4), sharey=True)
    for ri, (metric, rlabel, show_target) in enumerate(rows):
        for ci, s in enumerate(subs):
            ax = axes[ri, ci]
            d = sp[sp["subperiod"] == s].set_index("strategy")
            for i in ids:
                raw = raw_for(i, avail)
                val = float(d.loc[raw, metric])
                sc = style.style_for(i)
                y = ypos[i]
                ax.hlines(y, 0, val, color=sc["color"], linewidth=1.2, alpha=0.7)
                ax.scatter(val, y, s=32, color=sc["color"], marker=sc["marker"] or "o",
                           edgecolors=sc.get("markeredgecolor", "#333333"), linewidths=0.5,
                           zorder=4)
            if show_target:
                ax.axvline(style.TARGET_VOL, color="black", linestyle="--", linewidth=1.0)
            if ri == 0:
                ax.set_title(s, fontsize=10)
            if ci == 0:
                ax.set_yticks(list(ypos.values()))
                ax.set_yticklabels([style.style_for(i)["label"] for i in ids], fontsize=7.5)
                ax.set_ylabel(rlabel, fontsize=10)
            ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Adherence and realized volatility by subperiod", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "subperiod_panel.png")


# ================================================================= main =======
def main():
    builders = [
        fig_cumulative_growth, fig_drawdown, fig_rolling_vol, fig_persistence_path, fig_lambda21_path,
        fig_har_coef_path, fig_hurst_path, fig_h_stage_spread,
        fig_layer1_qlike_mcs, fig_adherence_intervals, fig_loss_concentration, fig_qlike_mse_rank,
        fig_cross_layer_scatter, fig_return_gap_decomposition, fig_sharpe_lw_pmatrix,
        fig_benchmark_comparison, fig_adherence_distribution, fig_leverage_distribution,
        fig_subperiod_panel,
    ]
    print(f"Building {len(builders)} figures into {FIGDIR}")
    for b in builders:
        b()
    print(f"\nDone. {len(_WRITTEN)} PNGs written:")
    for n in _WRITTEN:
        print("  -", n)


if __name__ == "__main__":
    main()

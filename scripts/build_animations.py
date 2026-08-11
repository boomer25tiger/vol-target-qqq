"""
S4 - five gallery animations, all routed through the shared style module so the gallery
and the paper read as one project. GIF (pillow) + poster PNG each; MP4 is emitted too if
an ffmpeg writer is available, and skipped with a notice otherwise (this environment has
no ffmpeg). Every frame reads cached data; no number is recomputed here.

Colorblind under motion: more than three series appear in four of the five, so strategy
identity is carried by the Okabe-Ito hue AND a per-strategy marker (o s ^ D v P X) drawn
at each line's leading edge, plus the ewma/rv dashed linestyle - never hue alone.

Frame cadence: 12 fps. Frames step semiannually (Jun/Dec) through calm years and monthly
through 2007-2009, so the 2008 window unfolds one frame per month where every animation
carries the most information; a one-second hold closes each. ~80 frames, ~7.5 s.

Run:  OMP_NUM_THREADS=1 .venv/bin/python scripts/build_animations.py
"""
from __future__ import annotations
import os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from volteq.viz import style                        # noqa: E402
from volteq.config import load_config               # noqa: E402
from volteq.backtest.engine import sizing_weight     # noqa: E402

style.apply(matplotlib)
DATA = os.path.join(REPO, "data", "processed")
TAB = os.path.join(REPO, "outputs", "tables")
ANIM = os.path.join(REPO, "gallery")
os.makedirs(ANIM, exist_ok=True)
ANNUAL = 252
FPS = 12
HOLD = 12                       # ~1 s hold on the final frame
MODELS7 = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv"]
_written = []


def sid(col):
    return col[len("bench_"):] if col.startswith("bench_") else col


def frame_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Month-end trading days: monthly through 2007-2009, else Jun/Dec."""
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).last().to_numpy()
    out = [pd.Timestamp(d) for d in me
           if (2007 <= pd.Timestamp(d).year <= 2009) or pd.Timestamp(d).month in (6, 12)]
    return pd.DatetimeIndex(sorted(out))


def _line_style(col):
    s = style.style_for(sid(col))
    return dict(color=s["color"], linestyle=s["linestyle"], linewidth=s.get("linewidth", 1.5),
                label=s["label"]), s


def _lead_marker(ax, x, y, col):
    s = style.style_for(sid(col))
    mk = s.get("marker") or "o"
    ax.plot([x], [y], marker=mk, color=s["color"], markersize=6,
            markeredgecolor=s.get("markeredgecolor", "#222222"), markeredgewidth=0.6, zorder=6)


def _save(fig, update, n_frames, name, poster_i, note=""):
    frames = list(range(n_frames)) + [n_frames - 1] * HOLD
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    gif = os.path.join(ANIM, f"{name}.gif")
    anim.save(gif, writer=PillowWriter(fps=FPS))
    update(poster_i)
    poster = os.path.join(ANIM, f"{name}_poster.png")
    fig.savefig(poster, dpi=110)
    # MP4 if a writer exists (none in this env); report either way
    mp4 = os.path.join(ANIM, f"{name}.mp4")
    mp4_done = False
    if matplotlib.animation.writers.is_available("ffmpeg"):
        try:
            anim.save(mp4, writer=matplotlib.animation.FFMpegWriter(fps=FPS, bitrate=2400))
            mp4_done = True
        except Exception:
            mp4_done = False
    plt.close(fig)
    gz = os.path.getsize(gif) / 1e6
    pz = os.path.getsize(poster) / 1e6
    _written.append((name, gz, pz, mp4_done, note))
    print(f"  {name:26s} gif {gz:5.2f} MB  poster {pz:4.2f} MB  mp4 {'yes' if mp4_done else 'SKIP (no ffmpeg)'}")


# ---------------------------------------------------------------- 1. surface
def anim_term_structure():
    vh = pd.read_parquet(os.path.join(DATA, "forecast_vh.parquet"))
    vh["date"] = pd.to_datetime(vh["date"])
    dates = frame_dates(pd.DatetimeIndex(sorted(vh["date"].unique())))
    hs = np.arange(1, 64)
    # per-date, per-model annualized vol curve
    piv = {c: vh[vh["col"] == c].pivot(index="date", columns="h", values="V") for c in MODELS7}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    def update(k):
        ax.clear()
        d = dates[k]
        for c in style.canonical_sort([sid(m) for m in MODELS7]):
            col = next(m for m in MODELS7 if sid(m) == c)
            row = piv[col].loc[piv[col].index.asof(d)]
            vol = np.sqrt(ANNUAL * row.reindex(hs).to_numpy())
            kw, _ = _line_style(col)
            ax.plot(hs, vol, **kw)
            _lead_marker(ax, hs[-1], vol[-1], col)
        ax.axhline(style.TARGET_VOL, color="#555555", linestyle=":", linewidth=1.0)
        ax.set_xlim(1, 63); ax.set_ylim(0.05, 0.85)   # fixed across regimes: calm ~0.13, 2008 ~0.62
        ax.set_xlabel("forecast horizon h (trading days)")
        ax.set_ylabel("forecast volatility (annualized)")
        ax.set_title(f"Model forecast term structure  V̂ₜ(h)   -   {pd.Timestamp(d):%Y-%m}")
        ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)
        ax.text(0.015, 0.05, "dotted = 20% target; rv near-term can run above axis in crises",
                transform=ax.transAxes, fontsize=7.0, color="#555555")

    poster = int(np.argmin(np.abs(dates - pd.Timestamp("2008-11-30"))))
    _save(fig, update, len(dates), "term_structure_surface", poster,
          "forward V_t(h) h=1..63 per model, evolving over rebalance dates (from forecast_vh)")


# --------------------------------------------------- growing-line scaffolding
def _growing_line(name, series_cols, ytransform, ylabel, title, logy, poster_date,
                  note, shade=None, value_labels=None, hline=None, ylim=None):
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    y = ytransform(eq)
    dates = frame_dates(eq.index)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ordered = style.canonical_sort([sid(c) for c in series_cols])
    col_of = {sid(c): c for c in series_cols}

    def update(k):
        ax.clear()
        d = dates[k]
        sub = y.loc[:d]
        if shade is not None:
            a, b = shade
            if sub.index[-1] >= pd.Timestamp(a):
                ax.axvspan(pd.Timestamp(a), min(pd.Timestamp(b), sub.index[-1]),
                           color="#CCCCCC", alpha=0.35, zorder=0)
        for cid in ordered:
            c = col_of[cid]
            kw, _ = _line_style(c)
            ax.plot(sub.index, sub[c].to_numpy(), **kw)
            _lead_marker(ax, sub.index[-1], sub[c].iloc[-1], c)
            if value_labels and cid in value_labels:
                ax.annotate(f"${sub[c].iloc[-1]/1e3:,.0f}k", (sub.index[-1], sub[c].iloc[-1]),
                            fontsize=7, color=style.style_for(cid)["color"],
                            xytext=(4, 0), textcoords="offset points", va="center")
        if hline is not None:
            ax.axhline(hline, color="#555555", linestyle=":", linewidth=1.0)
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(y.index[0], y.index[-1])
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}   -   {pd.Timestamp(d):%Y-%m}")
        ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)

    poster = int(np.argmin(np.abs(dates - pd.Timestamp(poster_date))))
    _save(fig, update, len(dates), name, poster, note)


# ------------------------------------------------------------ 2. performance
def anim_performance():
    cols = MODELS7 + ["bench_buy_hold"]
    _growing_line(
        "strategy_performance", cols,
        ytransform=lambda eq: eq[cols] / eq[cols].iloc[0],   # growth of $1
        ylabel="growth of $1 (log)", title="Strategy performance", logy=True,
        poster_date="2026-07-31",
        note="growth of $1, 7 models + buy-and-hold, log scale, lines extend left to right")


# --------------------------------------------------------- 3. account balance
def anim_balance():
    cols = ["gjr_skewt", "ewma", "rfsv", "bench_buy_hold"]
    _growing_line(
        "account_balance", cols,
        ytransform=lambda eq: eq[cols], ylabel="account balance ($, log)",
        title="Running account balance", logy=True, poster_date="2009-03-31",
        note="$ balance of four representative strategies, 2008-2009 shaded, live $ labels",
        shade=("2007-10-01", "2009-06-30"), value_labels={"gjr_skewt", "ewma", "rfsv", "buy_hold"})


# ------------------------------------------------------------ 4. rolling Sharpe
def anim_rolling_sharpe():
    cols = MODELS7 + ["bench_buy_hold"]
    eq = pd.read_parquet(os.path.join(DATA, "backtest_equity.parquet"))
    ret = eq[cols].pct_change()
    W = 504                                                    # 2-year rolling window
    rs = (ret.rolling(W).mean() / ret.rolling(W).std()) * np.sqrt(ANNUAL)
    dates = frame_dates(eq.index)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ordered = style.canonical_sort([sid(c) for c in cols])
    col_of = {sid(c): c for c in cols}

    def update(k):
        ax.clear()
        d = dates[k]
        sub = rs.loc[:d]
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        for cid in ordered:
            c = col_of[cid]
            kw, _ = _line_style(c)
            ax.plot(sub.index, sub[c].to_numpy(), **kw)
            if np.isfinite(sub[c].iloc[-1]):
                _lead_marker(ax, sub.index[-1], sub[c].iloc[-1], c)
        ax.set_xlim(rs.first_valid_index(), rs.index[-1]); ax.set_ylim(-1.6, 2.6)
        ax.set_ylabel("rolling 2-year Sharpe (annualized)")
        ax.set_title(f"Rolling Sharpe by strategy   -   {pd.Timestamp(d):%Y-%m}")
        ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.9)
        ax.text(0.015, 0.96, "ranking is noisy; no significance claimed", transform=ax.transAxes,
                fontsize=7.5, color="#555555", va="top")

    poster = int(np.argmin(np.abs(dates - pd.Timestamp("2009-06-30"))))
    _save(fig, update, len(dates), "rolling_sharpe", poster,
          "rolling 2-year Sharpe, 7 models + buy-and-hold; ranking re-orders across regimes")


# ------------------------------------------------------------ 5. leverage path
def anim_leverage():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(DATA, "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    v = v.loc[v.index >= pd.Timestamp(cfg["frozen"]["eval_start"])]
    W = {m: pd.Series(sizing_weight(v[m].to_numpy(), cfg), index=v.index) for m in MODELS7}
    W = pd.DataFrame(W)
    dates = frame_dates(v.index)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ordered = style.canonical_sort([sid(c) for c in MODELS7])

    def update(k):
        ax.clear()
        d = dates[k]
        sub = W.loc[:d]
        ax.axhspan(2007.0, 2007.0, color="none")   # noop keeps axhline layering simple
        ax.axhline(style.LEVERAGE_CAP, color="#B00000", linestyle="--", linewidth=1.1)
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        for cid in ordered:
            c = next(m for m in MODELS7 if sid(m) == cid)
            kw, _ = _line_style(c)
            ax.plot(sub.index, sub[c].to_numpy(), drawstyle="steps-post", **kw)
            _lead_marker(ax, sub.index[-1], sub[c].iloc[-1], c)
        ax.set_xlim(W.index[0], W.index[-1]); ax.set_ylim(-0.05, 2.15)
        ax.set_ylabel("portfolio leverage w")
        ax.set_title(f"Leverage path (monthly hold)   -   {pd.Timestamp(d):%Y-%m}")
        ax.text(W.index[3], style.LEVERAGE_CAP - 0.10, "2× cap", color="#B00000", fontsize=8)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)

    poster = int(np.argmin(np.abs(dates - pd.Timestamp("2008-12-31"))))
    _save(fig, update, len(dates), "leverage_path", poster,
          "monthly weight per model, 2x cap and 0 floor drawn; exposure collapses in 2008, ewma/rv pin the cap")


def main():
    # anim_rolling_sharpe() is defined above but NOT built: K3 and S5 both found its motion adds
    # nothing over a static rolling-Sharpe line, and the paper cites no rolling-Sharpe figure (T4).
    print(f"Building 4 animations into {ANIM}  (12 fps, crisis-dwelling frame schedule)")
    anim_term_structure()
    anim_performance()
    anim_balance()
    anim_leverage()
    print("\nDone. Files:")
    for name, gz, pz, mp4, note in _written:
        print(f"  {name}.gif ({gz:.2f} MB) + {name}_poster.png ({pz:.2f} MB)"
              f"{' + .mp4' if mp4 else '  [mp4 skipped: no ffmpeg]'}")
    total = sum(gz + pz for _, gz, pz, _, _ in _written)
    print(f"\ntotal GIF+poster size: {total:.2f} MB")


if __name__ == "__main__":
    main()

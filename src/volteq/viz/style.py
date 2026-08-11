"""
Shared figure style for the write-up figures. This module is the single source of truth the figure
scripts import. It generates no figures.

Design summary:
- Seven models carry Okabe-Ito hues (constructed for deuteranopia/protanopia);
  four benchmarks are achromatic (black + three grays) so they never confuse with
  a model under any color vision. Every series also carries a non-hue channel
  (line style for lines, marker for scatter), so no distinction rests on hue alone.
- Canonical order follows SPEC Section 5 (models) then Section 9 (benchmarks); it is
  not sorted by any result.
- Display names are reader-facing; internal ids (garch_skewt, ...) never appear on
  a figure.
"""
from __future__ import annotations

# ------------------------------------------------------------------ identity
MODELS = ["garch", "egarch", "gjr", "ewma", "rv", "har", "rfsv"]      # SPEC Section 5 order
BENCHMARKS = ["buy_hold", "const_lev", "trailing_rv21", "uncond_vol"]  # SPEC Section 9 order
ORDER = MODELS + BENCHMARKS                                            # canonical, every figure/table

DISPLAY = {
    "garch": "GARCH(1,1)", "egarch": "EGARCH(1,1)", "gjr": "GJR-GARCH",
    "ewma": "EWMA (λ=0.94)", "rv": "RV (direct-h)", "har": "HAR-RV", "rfsv": "RFSV",
    "buy_hold": "Buy & hold", "const_lev": "Constant leverage",
    "trailing_rv21": "Trailing-RV rung", "uncond_vol": "Unconditional-vol",
}

# ------------------------------------------------------------------ palette
# Okabe & Ito (2008) colorblind-safe qualitative palette for the seven models.
COLOR = {
    "garch":  "#0072B2",   # blue
    "egarch": "#56B4E9",   # sky blue
    "gjr":    "#009E73",   # bluish green
    "ewma":   "#E69F00",   # orange
    "rv":     "#F0E442",   # yellow  (weak on white -> dark marker edge, thicker line)
    "har":    "#CC79A7",   # reddish purple
    "rfsv":   "#D55E00",   # vermillion
    # benchmarks: achromatic references
    "buy_hold":      "#000000",
    "const_lev":     "#777777",
    "trailing_rv21": "#333333",
    "uncond_vol":    "#AAAAAA",
}

# Redundant non-hue channel. Dynamics models solid, naive models dashed; each
# benchmark a distinct gray dash. Markers used when a plot is scatter/bar or when a
# line needs grayscale separation (apply via markevery).
LINESTYLE = {
    "garch": "-", "egarch": "-", "gjr": "-", "har": "-", "rfsv": "-",
    "ewma": "--", "rv": "--",
    "buy_hold": "-", "const_lev": "-.", "trailing_rv21": "--", "uncond_vol": ":",
}
MARKER = {
    "garch": "o", "egarch": "s", "gjr": "^", "ewma": "D", "rv": "v",
    "har": "P", "rfsv": "X",
    "buy_hold": None, "const_lev": None, "trailing_rv21": None, "uncond_vol": None,
}

# recurring reference lines / limits (frozen values; do not hardcode elsewhere)
TARGET_VOL = 0.20
LEVERAGE_CAP = 2.0

# figure sizes in inches at 200 dpi -> ~1700 px wide, displayed at ~850 css px (2x retina)
FIG_WIDE = (8.5, 3.6)     # time series
FIG_BAR = (8.5, 4.0)      # ranked bars
FIG_SQUARE = (5.2, 5.0)   # scatter / heatmap


# ------------------------------------------------------------------ resolver
def _split(name: str):
    """Map any internal id (garch_skewt, garch_normal, rfsv_h010) to
    (base, variant) where base is one of ORDER and variant in
    {'skewt','normal','grid',None}."""
    if name in ORDER:
        return name, None
    if name.endswith("_skewt"):
        return name[:-6], "skewt"
    if name.endswith("_normal"):
        return name[:-7], "normal"
    if name.startswith("rfsv_h"):
        return "rfsv", "grid"
    return name, None


def style_for(name: str) -> dict:
    """matplotlib kwargs for one series, keyed by internal id. Skew-t is the primary
    (solid/base); Gaussian and rfsv-grid variants are sensitivity (same hue, dotted,
    thinner, de-emphasized label)."""
    base, variant = _split(name)
    color = COLOR.get(base, "#444444")
    kw = {
        "color": color,
        "linestyle": LINESTYLE.get(base, "-"),
        "marker": MARKER.get(base, None),
        "linewidth": 1.4,
        "label": DISPLAY.get(base, base),
    }
    if base == "rv":                       # weak yellow: dark edge + heavier line
        kw["linewidth"] = 1.7
        kw["markeredgecolor"] = "#333333"
    if base == "buy_hold":
        kw["linewidth"] = 2.0
    if variant == "normal":
        kw["linestyle"] = ":"; kw["linewidth"] = 1.1
        kw["label"] = f"{DISPLAY.get(base, base)} (Gaussian)"
    elif variant == "grid":
        kw["linestyle"] = ":"; kw["linewidth"] = 1.0; kw["alpha"] = 0.7
        h = name.split("_h")[-1]              # "002","005","010","015" -> 0.02..0.15
        kw["label"] = f"RFSV (H={int(h) / 100:.2f})"
    return kw


def canonical_sort(names):
    """Return `names` in canonical ORDER; unknown ids go last, stable."""
    rank = {n: i for i, n in enumerate(ORDER)}
    return sorted(names, key=lambda n: rank.get(_split(n)[0], len(ORDER)))


# ------------------------------------------------------------------ rcParams
def apply(mpl=None):
    """Set global rcParams for the write-up look. Call once per figure script.
    Pass the matplotlib module to avoid importing it at module import time."""
    if mpl is None:
        import matplotlib as mpl  # noqa: PLC0415
    mpl.rcParams.update({
        "figure.figsize": FIG_WIDE,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],   # bundled with matplotlib; no install needed
        "font.size": 10,
        "axes.titlesize": 12,                 # short axis-level title only; claim goes in the caption
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
    })

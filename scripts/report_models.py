#!/usr/bin/env python3
"""
GARCH-family diagnostics report (Phase C).

Figures (figures/): fitted-persistence path and lambda_21 path per model.
Prints: convergence failures, closed-form-vs-simulation discrepancy,
Mincer-Zarnowitz (a, b, HAC SE, R^2) per model, and skewed-t vs Gaussian
differences in persistence and V_t(21). Reads data/processed/. No network.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                            # noqa: E402
from volteq.models.returns import build_garch_returns                      # noqa: E402
from volteq.models.garch_family import fit_vt_garch                        # noqa: E402
from volteq.forecast.aggregate import (                                     # noqa: E402
    v_closed_form, simulate_garch_avg_var, lambda_h)
from volteq.rv.panel import load_panel                                      # noqa: E402
from volteq.rv.forward_target import forward_realized_variance_avg          # noqa: E402

PROC = os.path.join(repo_root(), "data", "processed")
FIG = os.path.join(repo_root(), "figures")
MODELS = ["garch", "gjr", "egarch", "ewma"]
COLORS = {"garch": "#1f77b4", "gjr": "#ff7f0e", "egarch": "#2ca02c", "ewma": "#d62728"}


def _load():
    pp = pd.read_parquet(os.path.join(PROC, "param_paths.parquet"))
    pp["date"] = pd.to_datetime(pp["date"])
    v = pd.read_parquet(os.path.join(PROC, "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date")
    with open(os.path.join(PROC, "convergence_summary.json")) as fh:
        summ = json.load(fh)
    return pp, v, summ


def fig_persistence(pp):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for m in ["garch", "gjr", "egarch"]:
        d = pp[(pp.model == m) & (pp.dist == "skewt")].sort_values("date")
        ax.plot(d["date"], d["persistence"], label=f"{m} (skew-t)", color=COLORS[m], lw=1.2)
    ax.axhline(1.0, color=COLORS["ewma"], ls="--", lw=1.2, label="ewma (phi=1, IGARCH)")
    ax.set_title("Fitted persistence phi across the sample (skew-t)")
    ax.set_ylabel("phi"); ax.legend(ncol=4, fontsize=8); ax.set_ylim(0.9, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "persistence_path.png"), dpi=130)
    plt.close(fig)


def fig_lambda21(pp, h):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for m in ["garch", "gjr", "egarch"]:
        d = pp[(pp.model == m) & (pp.dist == "skewt")].sort_values("date")
        lam = d["persistence"].apply(lambda p: lambda_h(p, h))
        ax.plot(d["date"], lam, label=f"{m} (skew-t)", color=COLORS[m], lw=1.2)
    ax.axhline(1.0, color=COLORS["ewma"], ls="--", lw=1.2, label="ewma (lambda_21=1, flat)")
    ax.set_title(f"lambda_{h}: fraction of the one-day signal surviving to the h={h} sizing decision")
    ax.set_ylabel(f"lambda_{h}"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lambda21_path.png"), dpi=130)
    plt.close(fig)


def mincer_zarnowitz(v, cfg):
    h = int(cfg["frozen"]["forecast_horizon_days"])
    panel = load_panel()
    rv = panel.loc[panel["source"] == "qqq", "rv_daily"]
    fwd = forward_realized_variance_avg(rv, h)          # evaluation-only target
    rows = []
    for col in ["garch_skewt", "gjr_skewt", "egarch_skewt", "ewma"]:
        j = pd.concat([v[col].rename("f"), fwd.reindex(v.index).rename("y")], axis=1).dropna()
        X = sm.add_constant(j["f"].to_numpy())
        res = sm.OLS(j["y"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        a, b = res.params
        sea, seb = res.bse
        rows.append((col, a, sea, b, seb, res.rsquared, len(j)))
    return rows


def closed_form_check(cfg):
    h = int(cfg["frozen"]["forecast_horizon_days"]); seed = int(cfg["meta"]["random_seed"])
    rets = build_garch_returns(cfg)
    out = []
    for tstr in ["2000-01-31", "2008-10-31", "2020-03-31"]:
        r = rets.loc[:tstr, "ret"].to_numpy()
        f = fit_vt_garch(r, "normal")
        cf = v_closed_form(f["sig2bar"], f["sigma2_next"], f["persistence"], h)
        sim = simulate_garch_avg_var(f["sig2bar"], f["sigma2_next"], f["params"]["alpha"],
                                     f["params"]["beta"], h, 300_000, seed)
        out.append((tstr, cf, sim["mc_mean"], sim["mc_se"], (sim["mc_mean"] - cf) / cf))
    return out


def skewt_vs_gaussian(pp, v):
    rows = []
    for m in ["garch", "gjr", "egarch"]:
        ps = pp[(pp.model == m) & (pp.dist == "skewt")].set_index("date")["persistence"]
        pn = pp[(pp.model == m) & (pp.dist == "normal")].set_index("date")["persistence"]
        dpers = (ps - pn).abs().mean()
        vs, vn = v[f"{m}_skewt"], v[f"{m}_normal"]
        dv = ((vs - vn).abs() / vs).mean()
        rows.append((m, dpers, float(ps.mean()), float(pn.mean()), dv))
    return rows


def main():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    os.makedirs(FIG, exist_ok=True)
    pp, v, summ = _load()

    fig_persistence(pp); fig_lambda21(pp, h)
    print(f"figures -> {FIG}/persistence_path.png, lambda21_path.png\n")

    print("=" * 70); print("CONVERGENCE FAILURES"); print("=" * 70)
    print(" ", summ["convergence_failures"] or "none  (all fits converged)")

    print("\n" + "=" * 70); print("CLOSED-FORM vs SIMULATION (garch, V_t(21))"); print("=" * 70)
    for t, cf, mc, se, rd in closed_form_check(cfg):
        print(f"  {t}: closed {cf:.4e}  sim {mc:.4e} +/-{se:.1e}  reldiff {rd:+.3%}")

    print("\n" + "=" * 70); print("MINCER-ZARNOWITZ  RV_fwd = a + b*V_t(21) + e  (HAC lag 3)"); print("=" * 70)
    print(f"  {'model':13s} {'a':>10s} {'SE(a)':>9s} {'b':>8s} {'SE(b)':>8s} {'R2':>7s} {'n':>5s}")
    for col, a, sea, b, seb, r2, n in mincer_zarnowitz(v, cfg):
        print(f"  {col:13s} {a:10.2e} {sea:9.1e} {b:8.3f} {seb:8.3f} {r2:7.3f} {n:5d}")

    print("\n" + "=" * 70); print("SKEW-T vs GAUSSIAN"); print("=" * 70)
    print(f"  {'model':7s} {'mean|dphi|':>11s} {'phi_skewt':>10s} {'phi_norm':>9s} {'mean|dV21|/V':>13s}")
    for m, dp, ps, pn, dv in skewt_vs_gaussian(pp, v):
        print(f"  {m:7s} {dp:11.4f} {ps:10.4f} {pn:9.4f} {dv:13.2%}")

    # annualized-vol summary of V_t(21) per model (skewt)
    print("\n" + "=" * 70); print("V_t(21) ANNUALIZED VOL, mean across dates (skew-t)"); print("=" * 70)
    for col in ["garch_skewt", "gjr_skewt", "egarch_skewt", "ewma"]:
        print(f"  {col:13s} {np.sqrt(v[col] * 252).mean():.3f}")


if __name__ == "__main__":
    main()

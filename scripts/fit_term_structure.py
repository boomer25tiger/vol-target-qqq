#!/usr/bin/env python3
"""
S3 - term structure V_t(h), h = 1..63, at every rebalance date, for every model, each
by its declared aggregation method:

  garch, gjr : closed form from the stored VT-GARCH state (param_paths.parquet), no refit
  ewma       : identity (flat, IGARCH)
  egarch     : Monte Carlo path, one horizon-63 simulation per date (the cost)
  har        : direct-h OLS, one regression per horizon
  rfsv (+grid): RFSV prediction formula, running mean of the per-horizon point forecasts
  rv         : trailing-h realized variance (running mean)

Writes data/processed/forecast_vh.parquet (long: date, col, h, V). Does NOT touch
forecast_v21.parquet. The h=21 slice is reconciled against forecast_v21 and the maximum
absolute difference per column is reported: exact for every deterministic model, MC noise
for egarch.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.models.returns import build_garch_returns
from volteq.models.rebalance import rebalance_dates
from volteq.models.garch_family import fit_egarch, egarch_apply
from volteq.forecast.aggregate import lambda_h, v_egarch_mc_path, v_egarch_mc
from volteq.models.direct_h import (rv_forecast, har_term_structure, rfsv_term_structure)

PROC = os.path.join(repo_root(), "data", "processed")
HMAX = 63
H_GRID = [0.02, 0.05, 0.10, 0.15]
def _gcol(H): return f"rfsv_h{int(round(H*100)):03d}"


def closed_form_ts(sig2bar, sigma2_next, phi, hmax):
    """V_t(h) = sig2bar + lambda_h(phi) (sigma2_next - sig2bar), vectorised over h."""
    hs = np.arange(1, hmax + 1)
    lam = np.array([lambda_h(float(phi), int(h)) for h in hs])
    return sig2bar + lam * (sigma2_next - sig2bar)


def main():
    cfg = load_config()
    seed = int(cfg["meta"]["random_seed"])
    min_obs = int(cfg["estimation"]["min_observations"])
    n_paths = int(next(m for m in cfg["models"] if m["id"] == "egarch")["n_paths"])
    hs = np.arange(1, HMAX + 1)

    v21 = pd.read_parquet(os.path.join(PROC, "forecast_v21.parquet"))
    v21["date"] = pd.to_datetime(v21["date"]); v21 = v21.set_index("date").sort_index()

    rows = []   # (date, col, h, V)

    # ---- 1. closed-form garch/gjr/ewma from param_paths (no refit) ----
    pp = pd.read_parquet(os.path.join(PROC, "param_paths.parquet"))
    pp["date"] = pd.to_datetime(pp["date"])
    for _, p in pp.iterrows():
        mid, dist, t = p["model"], p["dist"], p["date"]
        if mid in ("garch", "gjr"):
            col = f"{mid}_{dist}"
            ts = closed_form_ts(p["sig2bar"], p["sigma2_next"], p["persistence"], HMAX)
        elif mid == "ewma":
            col = "ewma"
            ts = np.full(HMAX, float(p["sigma2_next"]))     # identity, flat
        else:
            continue
        for h, V in zip(hs, ts):
            rows.append((t, col, int(h), float(V)))
    print(f"[ts] closed-form garch/gjr/ewma done ({pp['model'].isin(['garch','gjr','ewma']).sum()} rows)",
          flush=True)

    # ---- 2. egarch Monte Carlo path (refit per date; the cost) ----
    rets = build_garch_returns(cfg)
    dates = rebalance_dates(cfg, data_end=rets.index.max(), include_warmup=True)
    last_good: dict = {}
    egarch_v21_check = {}   # (date,dist) -> horizon-21 MC value, to validate the refit
    for i, t in enumerate(dates):
        r = rets.loc[:t, "ret"].to_numpy()
        if len(r) < min_obs:
            continue
        for dist in ("skewt", "normal"):
            key = ("egarch", dist)
            f = fit_egarch(r, dist)
            if not f["converged"] and key in last_good:
                f = egarch_apply(r, last_good[key]["params"], dist)
            elif f["converged"]:
                last_good[key] = {k: v for k, v in f.items() if k != "_res"}
            path = v_egarch_mc_path(f["_res"], HMAX, n_paths, seed + i)   # E[sigma^2_{t+k}] k=1..63
            ts = np.cumsum(path) / hs                                     # V_t(h)
            col = f"egarch_{dist}"
            for h, V in zip(hs, ts):
                rows.append((t, col, int(h), float(V)))
            # validation: horizon-21 MC with the same seed reproduces forecast_v21 exactly
            egarch_v21_check[(t, dist)] = v_egarch_mc(f["_res"], 21, n_paths, seed + i)
        if (i + 1) % 40 == 0 or i == len(dates) - 1:
            print(f"[ts] egarch {i+1}/{len(dates)} {t.date()}", flush=True)

    # ---- 3. direct-h har / rfsv(+grid) / rv from the RV panel ----
    panel = load_panel(); rv = panel["rv_daily"]
    pdates = rebalance_dates(cfg, data_end=panel.index.max(), include_warmup=True)
    for j, t in enumerate(pdates):
        r = rv.loc[:t].to_numpy()
        if len(r) < min_obs:
            continue
        har_ts = har_term_structure(r, HMAX)
        rfsv_ts = rfsv_term_structure(r, HMAX)
        rv_ts = np.cumsum(r[::-1][:HMAX]) / hs      # trailing-h mean, h=1..63
        cols = {"har": har_ts, "rfsv": rfsv_ts, "rv": rv_ts}
        for H in H_GRID:
            cols[_gcol(H)] = rfsv_term_structure(r, HMAX, H_fixed=H)
        for col, ts in cols.items():
            for h, V in zip(hs, ts):
                rows.append((t, col, int(h), float(V)))
        if (j + 1) % 40 == 0 or j == len(pdates) - 1:
            print(f"[ts] direct-h {j+1}/{len(pdates)} {t.date()}", flush=True)

    vh = pd.DataFrame(rows, columns=["date", "col", "h", "V"])
    outp = os.path.join(PROC, "forecast_vh.parquet")
    vh.to_parquet(outp, index=False)
    print(f"\n[ts] wrote {outp}: {len(vh)} rows, {vh['col'].nunique()} cols, "
          f"{vh['date'].nunique()} dates, h={vh['h'].min()}..{vh['h'].max()}", flush=True)

    # ---- 4. reconcile the h=21 slice against forecast_v21 ----
    h21 = vh[vh["h"] == 21].pivot(index="date", columns="col", values="V").sort_index()
    common = h21.index.intersection(v21.index)
    print("\n=== h=21 reconciliation vs forecast_v21 (max abs diff per column) ===")
    worst = 0.0
    for col in sorted(set(h21.columns) & set(v21.columns)):
        d = (h21.loc[common, col] - v21.loc[common, col]).abs()
        mad = float(d.max()); worst = max(worst, mad if not col.startswith("egarch") else 0.0)
        tag = "  (MC)" if col.startswith("egarch") else ""
        print(f"  {col:16s} max|Δ| = {mad:.3e}{tag}")
    # egarch refit validation: horizon-21 MC vs forecast_v21 (should be ~0)
    print("\n=== egarch refit validation: horizon-21 MC vs forecast_v21 (should be exact) ===")
    for dist in ("skewt", "normal"):
        col = f"egarch_{dist}"
        d = [abs(egarch_v21_check[(t, dist)] - v21.loc[t, col])
             for t in common if (t, dist) in egarch_v21_check]
        print(f"  {col:16s} max|Δ| = {max(d):.3e}  (same-horizon, same-seed refit)")
    print(f"\ndeterministic columns worst max|Δ| = {worst:.3e}")


if __name__ == "__main__":
    main()

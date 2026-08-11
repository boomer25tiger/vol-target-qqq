#!/usr/bin/env python3
"""
Direct-h models (rv, har, rfsv) + the RFSV Hurst grid.

Computes V_t(21) at every rebalance date (incl. the warmup) on data through t:
  rv, har, rfsv (estimated H), and rfsv at fixed H in {0.02, 0.05, 0.10, 0.15}.
Appends all columns to data/processed/forecast_v21.parquet, and persists the HAR
coefficient path, the RFSV Hurst path, and the RFSV-grid diagnostics (MZ inputs,
retransformation size, cap-binding frequency). Reads data/processed + config.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.models.rebalance import rebalance_dates                       # noqa: E402
from volteq.models.direct_h import (                                       # noqa: E402
    rv_forecast, har_fit_forecast, rfsv_fit_forecast)

PROC = os.path.join(repo_root(), "data", "processed")
H_GRID = [0.02, 0.05, 0.10, 0.15]


def _gcol(H):
    return f"rfsv_h{int(round(H*100)):03d}"


def main():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    min_obs = int(cfg["estimation"]["min_observations"])
    panel = load_panel()
    rv = panel["rv_daily"]
    dates = rebalance_dates(cfg, data_end=panel.index.max(), include_warmup=True)
    print(f"[direct-h] {len(rv)} rv obs, {len(dates)} rebalance dates "
          f"{dates[0].date()}..{dates[-1].date()}", flush=True)

    v_rows, har_rows, rfsv_rows = {}, [], []
    grid_diag = {_gcol(H): {"retrans": [], "cap_frac": []} for H in H_GRID}
    grid_diag["rfsv"] = {"retrans": [], "cap_frac": []}

    for i, t in enumerate(dates):
        r = rv.loc[:t].to_numpy()
        if len(r) < min_obs:
            continue
        ha = har_fit_forecast(r, h)
        rf = rfsv_fit_forecast(r, h)                 # estimated H
        row = {"rv": rv_forecast(r, h), "har": ha["V21"], "rfsv": rf["V21"]}
        grid_diag["rfsv"]["retrans"].append(rf["retransform_pct"])
        grid_diag["rfsv"]["cap_frac"].append(rf["cap_frac"])
        for H in H_GRID:
            g = rfsv_fit_forecast(r, h, H_fixed=H)
            row[_gcol(H)] = g["V21"]
            grid_diag[_gcol(H)]["retrans"].append(g["retransform_pct"])
            grid_diag[_gcol(H)]["cap_frac"].append(g["cap_frac"])
        v_rows[t] = row
        har_rows.append({"date": t, **{k: ha[k] for k in
                        ["beta0", "beta_d", "beta_w", "beta_m", "sigma2_resid",
                         "retransform_pct", "n_train"]}})
        rfsv_rows.append({"date": t, **{k: rf[k] for k in
                         ["H", "H_se", "nu2", "H_clipped", "cap_frac",
                          "retransform_pct"]}})
        if (i + 1) % 80 == 0 or i == len(dates) - 1:
            print(f"[direct-h] {i+1}/{len(dates)} {t.date()}", flush=True)

    newv = pd.DataFrame(v_rows).T
    newv.index.name = "date"
    cols = ["rv", "har", "rfsv"] + [_gcol(H) for H in H_GRID]

    vpath = os.path.join(PROC, "forecast_v21.parquet")
    v = pd.read_parquet(vpath); v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date")
    for c in cols:
        v[c] = newv[c].reindex(v.index)
    v.reset_index().to_parquet(vpath, index=False)

    pd.DataFrame(har_rows).to_parquet(os.path.join(PROC, "har_coef_path.parquet"), index=False)
    pd.DataFrame(rfsv_rows).to_parquet(os.path.join(PROC, "rfsv_hurst_path.parquet"), index=False)

    grid_summary = {}
    for k, dd in grid_diag.items():
        grid_summary[k] = {
            "retransform_pct_mean": float(np.mean(dd["retrans"])),
            "retransform_pct_max": float(np.max(dd["retrans"])),
            "cap_frac_mean": float(np.mean(dd["cap_frac"])),
        }
    with open(os.path.join(PROC, "rfsv_grid_summary.json"), "w") as fh:
        json.dump(grid_summary, fh, indent=2)

    print("\n[direct-h] panel columns:", list(v.columns))
    print("[direct-h] RFSV grid (retransform mean / cap-bind mean):")
    for k in ["rfsv"] + [_gcol(H) for H in H_GRID]:
        s = grid_summary[k]
        print(f"    {k:11s} retrans {s['retransform_pct_mean']:6.1%}  "
              f"cap-bind {s['cap_frac_mean']:6.1%}")


if __name__ == "__main__":
    main()

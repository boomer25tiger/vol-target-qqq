#!/usr/bin/env python3
"""
GARCH-family refit loop + h=21 aggregation.

Expanding window, refit at every monthly rebalance date on data through t only.
Models: garch, gjr (variance-targeted closed form), egarch (Monte Carlo), ewma
(identity). Distributions: skewed-t (primary) and Gaussian (robustness) for the
three estimated models. On a convergence failure the last successful parameter
set is carried forward onto the current window (never a default), logged, and
counted.

Writes data/processed/{param_paths.parquet, forecast_v21.parquet,
convergence_summary.json}. Reads data/raw + config. No network.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                              # noqa: E402
from volteq.models.returns import build_garch_returns                        # noqa: E402
from volteq.models.rebalance import rebalance_dates                          # noqa: E402
from volteq.models.garch_family import (                                     # noqa: E402
    fit_vt_garch, fit_vt_gjr, fit_egarch, fit_ewma,
    garch_state, gjr_state, egarch_apply)
from volteq.forecast.aggregate import (                                      # noqa: E402
    v_closed_form, v_identity, v_egarch_mc)

PROC = os.path.join(repo_root(), "data", "processed")


def main():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    seed = int(cfg["meta"]["random_seed"])
    min_obs = int(cfg["estimation"]["min_observations"])
    lam = float(next(m for m in cfg["models"] if m["id"] == "ewma")["lambda"])
    n_paths = int(next(m for m in cfg["models"] if m["id"] == "egarch")["n_paths"])

    rets = build_garch_returns(cfg)
    dates = rebalance_dates(cfg, data_end=rets.index.max(), include_warmup=True)
    print(f"[fit] {len(rets)} returns, {len(dates)} rebalance dates "
          f"{dates[0].date()}..{dates[-1].date()}", flush=True)

    est_models = [("garch", fit_vt_garch, garch_state),
                  ("gjr", fit_vt_gjr, gjr_state)]
    dists = ["skewt", "normal"]

    last_good: dict[tuple, dict] = {}
    failures: dict[str, int] = {}
    fail_log = []
    param_rows = []
    v21 = {}  # date -> {col: V}

    for i, t in enumerate(dates):
        r = rets.loc[:t, "ret"].to_numpy()
        if len(r) < min_obs:
            continue
        row_v = {}

        # --- variance-targeted closed-form models ---
        for mid, fit_fn, state_fn in est_models:
            for dist in dists:
                key = (mid, dist)
                f = fit_fn(r, dist)
                carried = False
                if not f["converged"] and key in last_good:
                    lg = last_good[key]
                    sig2bar, s2_next = state_fn(r, *_dyn(lg["params"], mid))
                    f = {**lg, "sig2bar": sig2bar, "sigma2_next": s2_next,
                         "converged": False}
                    carried = True
                    _bump(failures, fail_log, mid, dist, t, "carried_forward")
                elif not f["converged"]:
                    _bump(failures, fail_log, mid, dist, t, "failed_no_prior")
                else:
                    last_good[key] = f
                V = v_closed_form(f["sig2bar"], f["sigma2_next"], f["persistence"], h)
                row_v[f"{mid}_{dist}"] = V
                param_rows.append(_prow(t, mid, dist, f, V, carried))

        # --- egarch (Monte Carlo) ---
        for dist in dists:
            key = ("egarch", dist)
            f = fit_egarch(r, dist)
            carried = False
            if not f["converged"] and key in last_good:
                f = egarch_apply(r, last_good[key]["params"], dist)
                carried = True
                _bump(failures, fail_log, "egarch", dist, t, "carried_forward")
            elif not f["converged"]:
                _bump(failures, fail_log, "egarch", dist, t, "failed_no_prior")
            else:
                last_good[key] = {k: v for k, v in f.items() if k != "_res"}
            V = v_egarch_mc(f["_res"], h, n_paths, seed + i)
            row_v[f"egarch_{dist}"] = V
            param_rows.append(_prow(t, "egarch", dist, f, V, carried))

        # --- ewma (identity, no estimation) ---
        f = fit_ewma(r, lam)
        V = v_identity(f["sigma2_next"])
        row_v["ewma"] = V
        param_rows.append(_prow(t, "ewma", "none", f, V, False))

        v21[t] = row_v
        if (i + 1) % 40 == 0 or i == len(dates) - 1:
            print(f"[fit] {i+1}/{len(dates)} {t.date()}", flush=True)

    # --- persist ---
    os.makedirs(PROC, exist_ok=True)
    pp = pd.DataFrame(param_rows)
    pp.to_parquet(os.path.join(PROC, "param_paths.parquet"), index=False)
    v = pd.DataFrame(v21).T.sort_index()
    v.index.name = "date"
    v.reset_index().to_parquet(os.path.join(PROC, "forecast_v21.parquet"), index=False)

    summary = {
        "n_dates": len(dates), "h": h, "n_paths_egarch": n_paths, "seed": seed,
        "convergence_failures": failures,
        "failure_log": fail_log[:200],
        "date_min": str(dates[0].date()), "date_max": str(dates[-1].date()),
    }
    with open(os.path.join(PROC, "convergence_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[fit] convergence failures:", failures or "none")
    print(f"[fit] wrote param_paths ({len(pp)} rows), forecast_v21 ({len(v)} dates)")


def _dyn(params, mid):
    if mid == "garch":
        return (params["alpha"], params["beta"])
    return (params["alpha"], params["gamma"], params["beta"])


def _bump(failures, log, mid, dist, t, kind):
    k = f"{mid}_{dist}"
    failures[k] = failures.get(k, 0) + 1
    log.append({"model": mid, "dist": dist, "date": str(t.date()), "kind": kind})


def _prow(t, mid, dist, f, V, carried):
    p = f["params"]
    return {
        "date": t, "model": mid, "dist": dist,
        "converged": bool(f["converged"]), "carried": bool(carried),
        "persistence": float(f["persistence"]),
        "alpha": _g(p, "alpha"), "beta": _g(p, "beta", "beta[1]"),
        "gamma": _g(p, "gamma", "gamma[1]"),
        "eta": _g(p, "eta"), "lambda": _g(p, "lambda"),
        "sig2bar": f.get("sig2bar", np.nan), "sigma2_next": f["sigma2_next"],
        "V21": float(V), "n": int(f["n"]),
        "params_json": json.dumps({k: _round(v) for k, v in p.items()}),
    }


def _g(p, *keys):
    for k in keys:
        if k in p:
            return float(p[k])
    return np.nan


def _round(v):
    try:
        return round(float(v), 8)
    except Exception:
        return v


if __name__ == "__main__":
    main()

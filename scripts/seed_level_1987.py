#!/usr/bin/env python3
"""
Seed level-matching scalar c, and the October 1987 sensitivity gate.

c = var(NDX close-to-close returns) / mean(IXIC daily RV), over the seed window
[range_seed_source.start, seed_source.end]. Rescaling the IXIC realized-variance
seed by c gives the RV family the same unconditional volatility level that
variance targeting pins for the GARCH family (var of NDX cc returns).

Also quantifies the October 1987 effect: seed-window annualized vol for NDX cc
and IXIC RV, and the GARCH(1,1) implied sigma-bar-squared, computed WITH and
WITHOUT October 1987. Flags any sigma-bar move > 10%. Excludes nothing from the
build; the default seed includes October 1987.

Reads data/raw/. Writes data/processed/seed_level.json. No network.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                         # noqa: E402
from volteq.rv.estimators import daily_proxy_rv, annualized_vol_from_daily_var  # noqa: E402

ANNUAL = 252
FLAG_PCT = 0.10


def _load_ohlc(name):
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _oct1987_mask(idx):
    return (idx.year == 1987) & (idx.month == 10)


def _garch_sigma_bar(returns: pd.Series) -> dict:
    """Fit GARCH(1,1) Normal-QMLE; return implied unconditional variance/vol."""
    from arch import arch_model
    r = returns.dropna() * 100.0  # percent scale for numerical stability
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    p = res.params
    omega = float(p["omega"])
    alpha = float(p.get("alpha[1]", 0.0))
    beta = float(p.get("beta[1]", 0.0))
    persist = alpha + beta
    uvar_pct = omega / (1.0 - persist)          # unconditional variance, percent^2
    sigma2_bar_daily = uvar_pct / (100.0 ** 2)  # back to raw return variance
    return {
        "alpha": alpha, "beta": beta, "persistence": persist,
        "sigma2_bar_daily": sigma2_bar_daily,
        "sigma_bar_annual": float(np.sqrt(sigma2_bar_daily * ANNUAL)),
        "n": int(len(r)),
    }


def main():
    cfg = load_config()
    seed_start = cfg["realized_measures"]["seed"]["start"]
    seed_end = cfg["realized_measures"]["seed"]["end"]     # 1999-03-09
    print(f"Seed window: {seed_start} .. {seed_end}\n")

    # --- series over the common seed window ---
    ndx = _load_ohlc("ndx_daily").loc[seed_start:seed_end]
    ixic = _load_ohlc("ixic_daily").loc[seed_start:seed_end]

    ndx_cc = np.log(ndx["close"]).diff().dropna()
    ixic_rv = daily_proxy_rv(ixic)["rv_daily"].dropna()

    # align to common dates (both indices trade the same NYSE calendar)
    common = ndx_cc.index.intersection(ixic_rv.index)
    ndx_cc = ndx_cc.loc[common]
    ixic_rv = ixic_rv.loc[common]
    oct87 = _oct1987_mask(common)
    n_oct = int(oct87.sum())
    print(f"seed sessions (common): {len(common)}   October 1987 sessions: {n_oct}\n")

    def _report(mask_excl):
        cc = ndx_cc[~mask_excl] if mask_excl is not None else ndx_cc
        rv = ixic_rv[~mask_excl] if mask_excl is not None else ixic_rv
        return {
            "ndx_cc_ann_vol": float(cc.std(ddof=1) * np.sqrt(ANNUAL)),
            "ndx_cc_var_daily": float(cc.var(ddof=1)),
            "ixic_rv_ann_vol": annualized_vol_from_daily_var(rv, ANNUAL),
            "ixic_rv_mean_daily": float(rv.mean()),
            "c": float(cc.var(ddof=1) / rv.mean()),
            "n": int(len(cc)),
        }

    incl = _report(None)
    excl = _report(oct87)

    # c used by the build = include-everything (default)
    c = incl["c"]

    print("=" * 72)
    print("LEVEL MATCH  (seed window, common dates)")
    print("=" * 72)
    print(f"  {'':22s} {'incl Oct-87':>14s} {'excl Oct-87':>14s} {'delta':>9s}")
    for key, lab in [("ndx_cc_ann_vol", "NDX cc ann vol"),
                     ("ixic_rv_ann_vol", "IXIC RV ann vol"),
                     ("c", "scalar c")]:
        a, b = incl[key], excl[key]
        d = (b - a) / a
        print(f"  {lab:22s} {a:14.5f} {b:14.5f} {d:+8.1%}")
    # rescaled IXIC RV annualized (sanity: should equal NDX cc ann vol, incl)
    resc_ann = annualized_vol_from_daily_var(ixic_rv * c, ANNUAL)
    print(f"\n  IXIC RV x c annualized (incl): {resc_ann:.5f}  "
          f"(target NDX cc: {incl['ndx_cc_ann_vol']:.5f})")
    print(f"  --> c (build value, includes Oct-87) = {c:.6f}")

    print("\n" + "=" * 72)
    print("GARCH(1,1) sigma-bar on NDX cc returns")
    print("=" * 72)
    g_incl = _garch_sigma_bar(ndx_cc)
    g_excl = _garch_sigma_bar(ndx_cc[~oct87])
    d_sig = (g_excl["sigma_bar_annual"] - g_incl["sigma_bar_annual"]) / g_incl["sigma_bar_annual"]
    print(f"  incl Oct-87:  a={g_incl['alpha']:.3f} b={g_incl['beta']:.3f} "
          f"persist={g_incl['persistence']:.3f}  sigma_bar(ann)={g_incl['sigma_bar_annual']:.5f}")
    print(f"  excl Oct-87:  a={g_excl['alpha']:.3f} b={g_excl['beta']:.3f} "
          f"persist={g_excl['persistence']:.3f}  sigma_bar(ann)={g_excl['sigma_bar_annual']:.5f}")
    print(f"  sigma_bar move from excluding Oct-87: {d_sig:+.1%}")

    # --- flags ---
    print("\n" + "=" * 72)
    print("FLAGS (threshold: |move| > 10% from excluding October 1987)")
    print("=" * 72)
    flags = {
        "NDX cc annualized vol": (excl["ndx_cc_ann_vol"] - incl["ndx_cc_ann_vol"]) / incl["ndx_cc_ann_vol"],
        "IXIC RV annualized vol": (excl["ixic_rv_ann_vol"] - incl["ixic_rv_ann_vol"]) / incl["ixic_rv_ann_vol"],
        "scalar c": (excl["c"] - incl["c"]) / incl["c"],
        "GARCH sigma_bar (NDX cc)": d_sig,
    }
    any_flag = False
    for k, v in flags.items():
        hit = abs(v) > FLAG_PCT
        any_flag = any_flag or hit
        print(f"  {k:28s} {v:+8.1%}   {'*** FLAG' if hit else 'ok'}")
    print(f"\n  >>> {'AT LEAST ONE FLAG > 10% - user decides on an exclude-Oct-1987 robustness row.' if any_flag else 'No flags.'}")
    print("      Main build INCLUDES October 1987 (no self-initiated exclusion).")

    # --- persist c and the sensitivity for the panel build & the memo ---
    out = {
        "seed_window": [seed_start, seed_end],
        "seed_sessions_common": int(len(common)),
        "october_1987_sessions": n_oct,
        "c_build": c,
        "include_october_1987": True,
        "level_match": {"incl_oct87": incl, "excl_oct87": excl,
                        "ixic_rv_x_c_ann_incl": resc_ann},
        "garch_sigma_bar": {"incl_oct87": g_incl, "excl_oct87": g_excl,
                            "sigma_bar_move_excl": d_sig},
        "flags_pct": flags,
        "any_flag_gt_10pct": any_flag,
    }
    os.makedirs(os.path.join(repo_root(), "data", "processed"), exist_ok=True)
    p = os.path.join(repo_root(), "data", "processed", "seed_level.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete Mincer-Zarnowitz reporting for every forecast in the V_t(21) panel.

RV_{t+1:t+21} = a + b * V_t(21) + e, realized target = forward 21-day average of
the QQQ daily proxy (evaluation-only). For each model:
  - a, b with HAC standard errors,
  - joint Wald test of H0: (a, b) = (0, 1) with a p-value (slope alone is not
    calibration),
  - HAC lags 3, 5, 8 to show slope sensitivity.
Covers skew-t and Gaussian fits (and any direct-h columns present). No network.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.rv.forward_target import forward_realized_variance_avg        # noqa: E402

LAGS = [3, 5, 8]


def _mz(y, f, lag):
    X = sm.add_constant(f)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    a, b = res.params
    sea, seb = res.bse
    wald = res.wald_test((np.eye(2), np.array([0.0, 1.0])), use_f=False, scalar=True)
    return {"a": a, "sea": sea, "b": b, "seb": seb, "r2": res.rsquared,
            "wald_chi2": float(wald.statistic), "wald_p": float(wald.pvalue),
            "n": int(len(y))}


def main():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date")
    panel = load_panel()
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], h)

    cols = [c for c in v.columns]
    print(f"Mincer-Zarnowitz | realized = forward {h}-day avg rv_daily | models: {len(cols)}\n")

    # primary table at HAC lag 5
    print("=" * 92)
    print(f"PRIMARY  (HAC lag 5)   H0:(a,b)=(0,1)")
    print("=" * 92)
    print(f"  {'model':14s} {'a':>10s} {'SE(a)':>9s} {'b':>7s} {'SE(b)':>7s} "
          f"{'R2':>6s} {'Wald_chi2':>10s} {'Wald_p':>8s} {'n':>5s}")
    res5 = {}
    for c in cols:
        j = pd.concat([fwd.reindex(v.index).rename("y"), v[c].rename("f")], axis=1).dropna()
        m = _mz(j["y"].to_numpy(), j["f"].to_numpy(), 5)
        res5[c] = m
        star = "" if m["wald_p"] >= 0.05 else "  * reject (0,1)"
        print(f"  {c:14s} {m['a']:10.2e} {m['sea']:9.1e} {m['b']:7.3f} {m['seb']:7.3f} "
              f"{m['r2']:6.3f} {m['wald_chi2']:10.2f} {m['wald_p']:8.3f} {m['n']:5d}{star}")

    # slope sensitivity across HAC lags
    print("\n" + "=" * 72)
    print("SLOPE b AND Wald p ACROSS HAC LAGS 3 / 5 / 8")
    print("=" * 72)
    print(f"  {'model':14s} {'b@3':>7s} {'b@5':>7s} {'b@8':>7s}  | "
          f"{'p@3':>6s} {'p@5':>6s} {'p@8':>6s}")
    for c in cols:
        j = pd.concat([fwd.reindex(v.index).rename("y"), v[c].rename("f")], axis=1).dropna()
        bl, pl = {}, {}
        for L in LAGS:
            m = _mz(j["y"].to_numpy(), j["f"].to_numpy(), L)
            bl[L], pl[L] = m["b"], m["wald_p"]
        # SE(b) is what moves with lag; b itself is lag-invariant (same OLS point est)
        print(f"  {c:14s} {bl[3]:7.3f} {bl[5]:7.3f} {bl[8]:7.3f}  | "
              f"{pl[3]:6.3f} {pl[5]:6.3f} {pl[8]:6.3f}")
    print("\n  (b is the OLS point estimate, invariant to the HAC lag; only the SEs, and")
    print("   hence the Wald p-values, move. Sensitivity = whether inference flips.)")


if __name__ == "__main__":
    main()

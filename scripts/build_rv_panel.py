#!/usr/bin/env python3
"""
Build and cache the realized-measure panel, then report Phase C diagnostics.

Writes data/processed/rv_panel.parquet (+ .meta.json). Reports the overnight
variance share, full-sample annualized vol per measure, the cross-measure
correlation matrix, and the per-model estimation-history length as of 2000-01-03.
Reads data/raw/ and data/processed/. No network.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                       # noqa: E402
from volteq.rv.panel import build_panel, write_panel, VAR_COLS         # noqa: E402
from volteq.rv.forward_target import (                                 # noqa: E402
    forward_realized_variance_avg, forward_realized_variance_yz)

ANNUAL = 252


def _load_raw(name):
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def _ann(daily_var):
    return float(np.sqrt(np.nanmean(np.asarray(daily_var, float)) * ANNUAL))


def main():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    eval_start = cfg["frozen"]["eval_start"]

    panel, meta = build_panel(cfg)
    p = write_panel(panel, meta)
    print(f"cached {p}")
    print(f"c = {meta['c_seed_scalar']:.6f}  | rows {meta['rows']} "
          f"(seed {meta['rows_seed']}, qqq {meta['rows_qqq']})  "
          f"{meta['date_min']}..{meta['date_max']}\n")

    # forward (evaluation-only) measures for the diagnostics table
    fwd_avg = forward_realized_variance_avg(panel["rv_daily"], h)
    fwd_yz = forward_realized_variance_yz(panel["yz_21"], h)

    qqq_mask = panel["source"] == "qqq"

    print("=" * 72)
    print("OVERNIGHT VARIANCE SHARE  = mean(overnight_var) / mean(rv_daily)")
    print("=" * 72)
    for lab, m in [("full sample", slice(None)), ("QQQ era only", qqq_mask)]:
        sub = panel[m] if not isinstance(m, slice) else panel
        share = sub["overnight_var"].mean() / sub["rv_daily"].mean()
        print(f"  {lab:14s}: {share:.1%}")

    print("\n" + "=" * 72)
    print("FULL-SAMPLE ANNUALIZED VOL PER MEASURE")
    print("=" * 72)
    measures = {
        "rv_daily (overnight^2+RS)": panel["rv_daily"],
        "rs (Rogers-Satchell)": panel["rs"],
        "overnight_var": panel["overnight_var"],
        "yz_21 (trailing)": panel["yz_21"],
        f"fwd_rv_avg_{h} (eval)": fwd_avg,
        f"fwd_yz_{h} (eval)": fwd_yz,
    }
    for lab, s in measures.items():
        print(f"  {lab:28s} full {_ann(s):.4f}   QQQ {_ann(s[qqq_mask]):.4f}")

    print("\n" + "=" * 72)
    print("CROSS-MEASURE CORRELATION  (Spearman rank; robust to RV spikes)")
    print("=" * 72)
    corr_df = pd.DataFrame({
        "rv_daily": panel["rv_daily"],
        "rs": panel["rs"],
        "overnight": panel["overnight_var"],
        "yz_21": panel["yz_21"],
        f"fwd_rv_{h}": fwd_avg,
        f"fwd_yz_{h}": fwd_yz,
    })
    print("\n  [full sample]")
    print(corr_df.corr(method="spearman").round(3).to_string())
    print("\n  [QQQ era only]")
    print(corr_df[qqq_mask].corr(method="spearman").round(3).to_string())

    # ---- per-model estimation-history length as of eval_start ----
    print("\n" + "=" * 72)
    print(f"ESTIMATION-HISTORY LENGTH AS OF {eval_start}")
    print("=" * 72)
    cut = pd.Timestamp(eval_start)

    # GARCH family: close-to-close returns, NDX seed + QQQ traded
    ndx = _load_raw("ndx_daily")
    qqq = _load_raw("qqq_daily")
    ndx_cc = np.log(ndx["close"]).diff().dropna().loc[:"1999-03-09"]
    qqq_cc = np.log(qqq["close"]).diff().dropna().loc["1999-03-10":cut]
    garch_obs = len(ndx_cc) + len(qqq_cc)

    # RV family: daily RV proxy, IXIC seed (rescaled) + QQQ traded
    rv_hist = panel.loc[:cut, "rv_daily"].dropna()
    rv_seed = panel.loc[(panel.index <= cut) & (panel["source"] == "ixic_seed"), "rv_daily"].dropna()
    rv_qqq = panel.loc[(panel.index <= cut) & (panel["source"] == "qqq"), "rv_daily"].dropna()
    rv_obs = len(rv_hist)

    rows = [
        ("garch",  "GARCH family", "NDX cc + QQQ cc", garch_obs),
        ("egarch", "GARCH family", "NDX cc + QQQ cc", garch_obs),
        ("gjr",    "GARCH family", "NDX cc + QQQ cc", garch_obs),
        ("ewma",   "GARCH family", "NDX cc + QQQ cc", garch_obs),
        ("rv",     "RV family",    "c*IXIC RV + QQQ RV", rv_obs),
        ("har",    "RV family",    "c*IXIC RV + QQQ RV", rv_obs),
        ("rfsv",   "RV family",    "c*IXIC RV + QQQ RV", rv_obs),
    ]
    print(f"  {'model':7s} {'family':13s} {'series':22s} {'obs':>6s}  vs 1000-min")
    for mid, fam, series, obs in rows:
        print(f"  {mid:7s} {fam:13s} {series:22s} {obs:6d}  {'OK' if obs>=1000 else 'UNDER'}")
    print(f"\n  GARCH family: {garch_obs}  (NDX cc {len(ndx_cc)} + QQQ cc {len(qqq_cc)})")
    print(f"  RV family:    {rv_obs}  (IXIC seed {len(rv_seed)} + QQQ {len(rv_qqq)})")
    print(f"  RV family WITHOUT the IXIC seed would be {len(rv_qqq)} obs "
          f"(< 1000): the salvage adds {len(rv_seed)}.")


if __name__ == "__main__":
    main()

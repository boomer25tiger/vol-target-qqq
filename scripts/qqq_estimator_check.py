#!/usr/bin/env python3
"""
Phase A: does the range-proxy estimator bias persist in the QQQ era?

Ratio d = mean(rv_daily) / var(close-to-close returns). If the daily proxy
(overnight^2 + Rogers-Satchell) is an unbiased estimator of daily close-to-close
variance, d ~ 1.0. On the IXIC seed d ~ 0.70 (the 1.43x estimator effect). The
question is whether QQQ trade-print OHLC gives d near 1.0.

Reads data/raw/. No network.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                        # noqa: E402
from volteq.rv.estimators import daily_proxy_rv, yang_zhang             # noqa: E402

ANNUAL = 252


def _load(name):
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["open", "high", "low", "close"]]


def _measures(df, n):
    cc = np.log(df["close"]).diff()
    rv = daily_proxy_rv(df)["rv_daily"]
    yz = yang_zhang(df, n)
    return pd.DataFrame({"cc": cc, "rv_daily": rv, "yz_21": yz})


def _ratio(sub):
    m = sub.dropna(subset=["cc", "rv_daily"])
    return float(m["rv_daily"].mean() / m["cc"].var(ddof=1))


def main():
    cfg = load_config()
    n = int(cfg["realized_measures"]["seed"]["yz_window_days"])
    eval_start = cfg["frozen"]["eval_start"]

    qqq = _measures(_load("qqq_daily"), n)
    q = qqq.loc[eval_start:]

    cc = q["cc"].dropna()
    ann = lambda s: float(np.sqrt(np.nanmean(s) * ANNUAL))
    print("=" * 70)
    print(f"QQQ, {eval_start} .. present   (n={len(cc)} sessions)")
    print("=" * 70)
    print(f"  a. close-to-close annualized vol : {cc.std(ddof=1)*np.sqrt(ANNUAL):.4f}")
    print(f"  b. rv_daily annualized vol       : {ann(q['rv_daily']):.4f}")
    print(f"  c. yz_21 annualized vol          : {ann(q['yz_21']):.4f}")
    d = _ratio(q)
    # uncentered cross-check denominator E[cc^2]
    m = q.dropna(subset=["cc", "rv_daily"])
    d_unc = float(m["rv_daily"].mean() / (m["cc"] ** 2).mean())
    print(f"  d. mean(rv_daily) / var(cc)      : {d:.4f}   (uncentered E[cc^2]: {d_unc:.4f})")
    print(f"     sqrt(1/d) weight inflation     : {np.sqrt(1.0/d):.4f}")

    print("\n  by-year ratio d = mean(rv_daily)/var(cc):")
    by = []
    for y, g in q.groupby(q.index.year):
        by.append((y, len(g.dropna(subset=['cc','rv_daily'])), _ratio(g)))
    for y, ny, r in by:
        bar = "#" * int(round(r * 30))
        print(f"    {y}  n={ny:3d}  d={r:5.3f}  {bar}")
    yrs = np.array([r for _, _, r in by])
    print(f"    -> by-year mean {yrs.mean():.3f}, min {yrs.min():.3f}, max {yrs.max():.3f}")

    # IXIC seed window comparison (raw, no c-rescale)
    s = cfg["realized_measures"]["seed"]
    ixic = _measures(_load("ixic_daily").loc[s["start"]:s["end"]], n)
    d_ixic = _ratio(ixic)
    print("\n" + "=" * 70)
    print(f"IXIC seed window {s['start']}..{s['end']} (raw, no c):  d = {d_ixic:.4f}")
    print(f"   1/d = {1.0/d_ixic:.3f}  (the estimator effect ~1.43x variance)")
    print("=" * 70)

    print("\nVERDICT:")
    if 0.95 <= d <= 1.05:
        print(f"  QQQ ratio d={d:.3f} is within [0.95, 1.05]. The estimator bias does NOT")
        print("  persist in QQQ trade-print OHLC. No correction needed; flag withdrawn.")
    else:
        print(f"  QQQ ratio d={d:.3f} is OUTSIDE [0.95, 1.05]. Proxy biased; correction needed.")


if __name__ == "__main__":
    main()

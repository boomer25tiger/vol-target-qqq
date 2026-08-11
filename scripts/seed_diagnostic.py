#!/usr/bin/env python3
"""
Seed-candidate diagnostic (Phase A): can the RV-family estimation seed be salvaged?

The NDX close-to-close decision leaves RV / HAR-RV / rough-vol with too little
realized-variance history by 2000-01-03 (range estimators need trustworthy OHLC,
and NDX opens are mechanically equal to the prior close pre-2000). This script
tests ^NDX, ^IXIC, and ^GSPC for the mechanical-open pathology, BY YEAR, over
1985-1999, and reports whether any is a clean range-estimator seed.

Reads data/raw/{ndx,ixic,gspc}_daily.parquet. No network.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RAW = "data/raw"
CANDIDATES = ["ndx", "ixic", "gspc"]
LABEL = {"ndx": "^NDX (Nasdaq-100)", "ixic": "^IXIC (Nasdaq Composite)",
         "gspc": "^GSPC (S&P 500)"}
SEED_END = "2000-01-03"       # eval opens here; seed must precede it
CLEAN_THRESHOLD = 0.05        # <5% mechanical opens == clean


def _load(name):
    p = os.path.join(RAW, f"{name}_daily.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def by_year(df, lo="1985-01-01", hi="1999-12-31"):
    d = df.loc[lo:hi]
    prior = d["close"].shift(1)
    exact = (d["open"] == prior)
    near = ((d["open"] - prior).abs() / prior < 1e-6)
    t = pd.DataFrame({"exact": exact, "near": near})
    t["year"] = d.index.year
    g = t.groupby("year").agg(n=("exact", "size"),
                              exact=("exact", "sum"),
                              near=("near", "sum"))
    g["exact_pct"] = (g["exact"] / g["n"] * 100).round(1)
    return g


def first_clean_year(g):
    """First year whose rate is < threshold AND stays < threshold to the end."""
    years = list(g.index)
    for i, y in enumerate(years):
        if all(g.loc[yy, "exact_pct"] < CLEAN_THRESHOLD * 100 for yy in years[i:]):
            return y
    return None


def main():
    print("Mechanical-open diagnostic: fraction of sessions with open == prior close")
    print(f"Clean threshold: < {CLEAN_THRESHOLD:.0%}.  Seed must precede {SEED_END}.\n")

    ndx = _load("ndx")
    ndx_ret = np.log(ndx["close"]).diff().loc["1985-01-01":"1999-12-31"] if ndx is not None else None

    summary = {}
    for name in CANDIDATES:
        df = _load(name)
        print("=" * 68)
        print(f"{LABEL[name]}   [{name}_daily]")
        print("=" * 68)
        if df is None:
            print("  MISSING\n")
            continue
        print(f"  available from {df.index.min().date()}")
        g = by_year(df)
        print(g[["n", "exact", "exact_pct"]].to_string())
        fc = first_clean_year(g)
        summary[name] = {"first_clean_year": fc}
        if fc is not None:
            # obs from first clean year's Jan 1 to SEED_END
            span = df.loc[f"{fc}-01-01":SEED_END]
            n_obs = len(span) - 1   # one lost to the diff for range/return
            summary[name]["clean_span"] = (f"{fc}-01-01", SEED_END)
            summary[name]["n_obs_by_2000"] = n_obs
            print(f"\n  -> clean (<5%) from {fc} onward; "
                  f"{n_obs} sessions {fc}-01-01 .. {SEED_END}")
        else:
            print("\n  -> never clean over 1985-1999 (mechanical-open pathology present)")
        # proxy quality vs NDX returns over the overlap
        if ndx_ret is not None and name != "ndx":
            r = np.log(df["close"]).diff().loc["1985-01-01":"1999-12-31"]
            j = pd.concat([r.rename("cand"), ndx_ret.rename("ndx")], axis=1).dropna()
            if len(j) > 100:
                corr = j["cand"].corr(j["ndx"])
                summary[name]["ret_corr_vs_ndx"] = round(float(corr), 3)
                print(f"  -> close-to-close return corr with ^NDX (1985-1999, "
                      f"n={len(j)}): {corr:.3f}")
        print()

    print("=" * 68)
    print("RECOMMENDATION")
    print("=" * 68)
    # prefer a clean series that best proxies the Nasdaq-100
    order = ["ndx", "ixic", "gspc"]   # preference: exact underlying, then same market
    pick = None
    for name in order:
        s = summary.get(name, {})
        if s.get("first_clean_year") is not None and s.get("n_obs_by_2000", 0) >= 1000:
            pick = name
            break
    if pick:
        s = summary[pick]
        print(f"  Use {LABEL[pick]} as the range-estimator seed.")
        print(f"    clean span: {s['clean_span'][0]} .. {s['clean_span'][1]} "
              f"({s['n_obs_by_2000']} sessions, >= 1000 minimum)")
        if "ret_corr_vs_ndx" in s:
            print(f"    return corr with ^NDX: {s['ret_corr_vs_ndx']}")
    else:
        print("  No candidate is clean over a span long enough to reach 1000")
        print("  observations by 2000-01-03. Fall back to squared close-to-close")
        print("  NDX returns as the daily variance proxy for the RV-family seed")
        print("  (Phase B path 2).")
    print("\n  per-candidate summary:")
    for name in CANDIDATES:
        print(f"    {name}: {summary.get(name, {})}")


if __name__ == "__main__":
    main()

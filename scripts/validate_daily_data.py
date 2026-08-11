#!/usr/bin/env python3
"""
Phase C validation of the cached daily data (reads data/raw/, no network).

Emits numbers for each acquisition check.

Checks
------
  1. Row counts and date ranges per series; does QQQ start 1999-03-10.
  2. Session count per year vs the NYSE calendar; years missing > 2 sessions.
  3. Yahoo vs Stooq daily closes (if a Stooq file is present).
  4. Split events from yfinance actions; 2000-03-20 2-for-1; no adjusted jump.
  5. NDX open == prior close (exact) by decade; range-estimator viability.
  6. DFF coverage 2000-present; weekends/holidays filled or absent.
  7. Zero / negative / null OHLC in either equity series.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RAW = "data/raw"


def _load(name):
    p = os.path.join(RAW, f"{name}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def hr(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76, flush=True)


def check1(qqq, ndx, dff):
    hr("1. ROW COUNTS AND DATE RANGES")
    for name, df in [("qqq_daily", qqq), ("ndx_daily", ndx), ("dff", dff)]:
        if df is None:
            print(f"  {name:<12} MISSING")
            continue
        print(f"  {name:<12} rows={len(df):>7,}  {df.index.min().date()} .. {df.index.max().date()}")
    q0 = qqq.index.min().date()
    print(f"\n  QQQ starts {q0} -> expected 1999-03-10 : "
          f"{'YES' if str(q0)=='1999-03-10' else 'NO'}")


def check2(qqq, ndx):
    hr("2. SESSION COUNT PER YEAR vs NYSE CALENDAR")
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    for label, df in [("QQQ", qqq), ("NDX", ndx)]:
        print(f"\n  [{label}]")
        yrs = range(df.index.min().year, df.index.max().year + 1)
        flagged = []
        rows = []
        for y in yrs:
            sched = nyse.schedule(start_date=f"{y}-01-01", end_date=f"{y}-12-31")
            n_cal = len(sched)
            have = df.loc[f"{y}-01-01":f"{y}-12-31"]
            # clip calendar to the data's actual span in the first/last year
            lo = max(pd.Timestamp(f"{y}-01-01"), df.index.min())
            hi = min(pd.Timestamp(f"{y}-12-31"), df.index.max())
            cal_days = nyse.valid_days(start_date=lo, end_date=hi)
            n_cal_clip = len(cal_days)
            n_have = len(have)
            miss = n_cal_clip - n_have
            rows.append((y, n_cal_clip, n_have, miss))
            if abs(miss) > 2:
                flagged.append((y, n_cal_clip, n_have, miss))
        # compact print: only show flagged + head/tail
        print(f"    years covered: {rows[0][0]}..{rows[-1][0]}")
        print(f"    years with |calendar - data| > 2 sessions: {len(flagged)}")
        for y, c, h, m in flagged:
            print(f"      {y}: calendar {c}, data {h}, diff {m:+d}")
        if not flagged:
            print("      (none)")


def check3(qqq):
    hr("3. YAHOO vs STOOQ DAILY CLOSES")
    st = _load("qqq_stooq")
    if st is None and os.path.exists(os.path.join(RAW, "qqq_stooq_manual.csv")):
        st = pd.read_csv(os.path.join(RAW, "qqq_stooq_manual.csv"))
        st.columns = [c.strip().lower() for c in st.columns]
        st["date"] = pd.to_datetime(st["date"])
        st = st.set_index("date").sort_index()
    if st is None:
        print("  Stooq series UNAVAILABLE this session.")
        print("  Reason: pandas_datareader 0.11.1 has no stooq route/module, and the")
        print("  stooq.com CSV endpoint is behind a JavaScript proof-of-work anti-bot")
        print("  challenge that was NOT bypassed. Independent cross-check deferred.")
        print("  To supply it: drop a manual export at data/raw/qqq_stooq_manual.csv")
        print("  (columns: Date,Open,High,Low,Close,Volume) and re-run.")
        return
    y = qqq["close"].rename("yahoo")
    s = st["close"].rename("stooq")
    j = pd.concat([y, s], axis=1)
    only_y = j["yahoo"].notna() & j["stooq"].isna()
    only_s = j["stooq"].notna() & j["yahoo"].isna()
    both = j.dropna()
    reldiff = (both["yahoo"] - both["stooq"]).abs() / both["stooq"]
    print(f"  dates only in Yahoo: {int(only_y.sum())}")
    print(f"  dates only in Stooq: {int(only_s.sum())}")
    print(f"  shared dates: {len(both)}")
    print(f"  abs rel diff: mean {reldiff.mean():.4%}  median {reldiff.median():.4%}  "
          f"p95 {reldiff.quantile(.95):.4%}  max {reldiff.max():.4%}")
    flagged = reldiff[reldiff > 0.005]
    print(f"  dates differing > 0.5%: {len(flagged)}")
    if len(flagged):
        print("   ", ", ".join(str(d.date()) for d in flagged.index[:12]),
              "..." if len(flagged) > 12 else "")


def check4(qqq):
    hr("4. SPLIT EVENTS (yfinance actions)")
    sp = qqq[qqq["stock_splits"] != 0][["stock_splits"]]
    print(f"  split events: {len(sp)}")
    for d, row in sp.iterrows():
        print(f"    {d.date()}  ratio {row['stock_splits']}")
    has = any(str(d.date()) == "2000-03-20" for d in sp.index)
    print(f"\n  2000-03-20 2-for-1 present: {'YES' if has else 'NO'}")
    r = np.log(qqq["close"]).diff()
    if pd.Timestamp("2000-03-20") in r.index:
        v = float(r.loc["2000-03-20"])
        print(f"  adjusted close-to-close log return on 2000-03-20: {v:+.4f} "
              f"({np.expm1(v)*100:+.2f}%)  -> {'OK (<5%)' if abs(v)<0.05 else 'ANOMALY'}")


def check5(ndx):
    hr("5. NDX OPEN == PRIOR CLOSE (range-estimator viability)")
    prior_close = ndx["close"].shift(1)
    exact = (ndx["open"] == prior_close)
    # near-equal within 1e-6 relative, to catch rounding
    near = ((ndx["open"] - prior_close).abs() / prior_close < 1e-6)
    df = pd.DataFrame({"exact": exact, "near": near})
    df["decade"] = (ndx.index.year // 10 * 10).astype(int)
    g = df.groupby("decade").agg(n=("exact", "size"),
                                 exact=("exact", "sum"),
                                 near=("near", "sum"))
    g["exact_pct"] = (g["exact"] / g["n"] * 100).round(1)
    g["near_pct"] = (g["near"] / g["n"] * 100).round(1)
    print(g.to_string())
    pre95 = df[ndx.index < "1995-01-01"]
    frac = pre95["exact"].mean() if len(pre95) else float("nan")
    print(f"\n  pre-1995 rows: {len(pre95)}, open==prior_close exact: {frac:.1%}")
    print(f"  VERDICT: {'range estimators UNSAFE pre-1995 -> seed uses close-to-close' if frac>0.20 else 'range estimators usable across the seed window'}")
    return frac


def check6(dff):
    hr("6. DFF COVERAGE (2000-present)")
    d = dff.loc["2000-01-01":]
    cal_days = pd.date_range(d.index.min(), d.index.max(), freq="D")
    present = d.index.normalize().unique()
    missing = cal_days.difference(present)
    is_weekend = pd.Series(present).dt.dayofweek.isin([5, 6])
    nan_rows = int(d["dff"].isna().sum())
    print(f"  span: {d.index.min().date()} .. {d.index.max().date()}")
    print(f"  rows: {len(d):,}  distinct calendar days present: {len(present):,}")
    print(f"  calendar days in span: {len(cal_days):,}  missing calendar days: {len(missing)}")
    print(f"  weekend rows present: {int(is_weekend.sum()):,}  -> weekends are "
          f"{'FILLED' if is_weekend.sum() > 100 else 'ABSENT'}")
    print(f"  NaN dff values: {nan_rows}")
    if len(missing):
        print("  first few missing:", ", ".join(str(x.date()) for x in missing[:8]))


def check7(qqq, ndx):
    hr("7. OHLC INTEGRITY (zero / negative / null)")
    for name, df in [("qqq_daily", qqq), ("ndx_daily", ndx)]:
        cols = ["open", "high", "low", "close"]
        nulls = int(df[cols].isna().any(axis=1).sum())
        nonpos = int((df[cols] <= 0).any(axis=1).sum())
        bad_hl = int((df["high"] < df["low"]).sum())
        bad_hi = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
        bad_lo = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
        zero_vol = int((df["volume"] == 0).sum()) if "volume" in df else -1
        print(f"  [{name}] null OHLC rows: {nulls}  nonpositive: {nonpos}  "
              f"high<low: {bad_hl}  high<max(o,c): {bad_hi}  low>min(o,c): {bad_lo}  "
              f"zero-volume: {zero_vol}")


def main():
    qqq = _load("qqq_daily")
    ndx = _load("ndx_daily")
    dff = _load("dff")
    check1(qqq, ndx, dff)
    check2(qqq, ndx)
    check3(qqq)
    check4(qqq)
    check5(ndx)
    check6(dff)
    check7(qqq, ndx)
    print("\nDONE.")


if __name__ == "__main__":
    main()

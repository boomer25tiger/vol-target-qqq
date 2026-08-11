#!/usr/bin/env python3
"""
QQQ intraday data validation for the vol-targeting project.

Evaluates whether the HuggingFace dataset `mito0o852/OHLCV-1m` is a viable
source of 1-minute bars for QQQ from 1999 onward.

Stages
------
  selftest   Run every diagnostic against synthetic data. No network. Run first.
  inventory  List the repo file tree. Checks for per-ticker files that would
             make the full 87.7 GB scan unnecessary.
  probe      Inspect one monthly Parquet file's internal layout. Determines
             whether row-group pruning on `ticker` is possible.
  extract    Pull QQQ / QQQQ rows from every monthly file into one local
             Parquet. Resumable: re-running skips months already cached.
  checks     Run all data-quality diagnostics and write plots + CSVs.
  all        inventory -> probe -> extract -> checks

Usage
-----
  pip install pandas pyarrow numpy matplotlib huggingface_hub
  python qqq_data_validation.py selftest
  python qqq_data_validation.py inventory
  python qqq_data_validation.py probe --month 2010-06
  python qqq_data_validation.py extract --start 1999-01 --end 2026-03
  python qqq_data_validation.py extract --mode stream          # datasets fallback
  python qqq_data_validation.py checks

Do NOT use the loader shown on the dataset card. `load_dataset(repo, split="train")`
resolves every ticker across every month (87.7 GB), offers no ticker filter, and
goes through a split configuration that the repo's own dataset viewer fails on.

Outputs land in ./validation_output/ (CSVs + PNGs) and ./cache/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

REPO_ID = "mito0o852/OHLCV-1m"
REPO_TYPE = "dataset"

# QQQ traded under QQQQ from 2004-12-01 to 2011-03-23. Pulling only "QQQ"
# silently drops the financial crisis.
TICKERS = ["QQQ", "QQQQ"]

# Known corporate actions. VERIFY against an independent split calendar before
# relying on this; the dataset is NOT split-adjusted (confirmed by the repo
# owner in community discussion #4).
KNOWN_SPLITS = {
    "2000-03-20": 2.0,  # QQQ 2-for-1
}

EXCHANGE_TZ = "America/New_York"
RTH_OPEN = (9, 30)
RTH_CLOSE = (16, 0)          # exclusive; last regular bar starts 15:59
HALF_DAY_CLOSE = (13, 0)     # exclusive; last half-day bar starts 12:59
FULL_SESSION_BARS = 390
HALF_SESSION_BARS = 210

SIGNATURE_FREQS = [1, 2, 3, 5, 10, 15, 20, 30]
SPLIT_FLAG_THRESHOLD = 0.20  # |log return| overnight

CACHE_DIR = "cache"
MONTH_CACHE_DIR = os.path.join(CACHE_DIR, "months")
COMBINED_PATH = os.path.join(CACHE_DIR, "qqq_1m.parquet")
OUT_DIR = "validation_output"

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]


def _ensure_dirs():
    for d in (CACHE_DIR, MONTH_CACHE_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


def _log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------
# Stage 1: inventory
# --------------------------------------------------------------------------

def stage_inventory(args):
    from huggingface_hub import HfApi

    _hr("STAGE: inventory")
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    _log(f"{len(files)} files in repo")

    prefixes = {}
    for f in files:
        head = f.split("/")[0] if "/" in f else "(root)"
        prefixes[head] = prefixes.get(head, 0) + 1
    print("\nTop-level structure")
    for k, v in sorted(prefixes.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v} files")

    monthly = sorted(f for f in files if f.startswith("data/ohlcv_") and f.endswith(".parquet"))
    print(f"\nMonthly files: {len(monthly)}")
    if monthly:
        print(f"  first: {monthly[0]}")
        print(f"  last:  {monthly[-1]}")

    # If per-ticker files exist, the 87.7 GB scan is avoidable entirely.
    ticker_hits = [f for f in files if "QQQ" in f.upper()]
    print(f"\nFiles with 'QQQ' in the path: {len(ticker_hits)}")
    for f in ticker_hits[:20]:
        print(f"  {f}")
    if not ticker_hits:
        print("  none -- extraction must scan the monthly files")

    # Sizes, if the API exposes them.
    try:
        info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
        sizes = [(s.rfilename, s.size) for s in info.siblings if s.size]
        if sizes:
            total = sum(s for _, s in sizes)
            print(f"\nTotal size across {len(sizes)} sized files: {total / 1e9:.1f} GB")
            recent = sorted(sizes)[-5:]
            print("Sample file sizes")
            for name, size in recent:
                print(f"  {name:<40} {size / 1e6:8.1f} MB")
    except Exception as exc:
        print(f"\n(size metadata unavailable: {exc})")

    pd.DataFrame({"file": files}).to_csv(os.path.join(OUT_DIR, "repo_files.csv"), index=False)
    _log(f"wrote {OUT_DIR}/repo_files.csv")


# --------------------------------------------------------------------------
# Stage 2: probe layout
# --------------------------------------------------------------------------

def _month_path(month: str) -> str:
    return f"data/ohlcv_{month}.parquet"


def stage_probe(args):
    from huggingface_hub import HfFileSystem

    _hr(f"STAGE: probe ({args.month})")
    fs = HfFileSystem()
    remote = f"datasets/{REPO_ID}/{_month_path(args.month)}"

    try:
        size = fs.info(remote)["size"]
        _log(f"file size: {size / 1e6:.1f} MB")
    except Exception as exc:
        print(f"could not stat {remote}: {exc}")
        return

    with fs.open(remote, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        print(f"\nrows:        {md.num_rows:,}")
        print(f"row groups:  {md.num_row_groups}")
        print(f"columns:     {[md.schema.column(i).name for i in range(md.num_columns)]}")
        print(f"compression: {md.row_group(0).column(0).compression}")

        # Locate the ticker column and read its per-row-group statistics.
        names = [md.schema.column(i).name for i in range(md.num_columns)]
        if "ticker" not in names:
            print("\nno 'ticker' column -- schema differs from the dataset card")
            return
        tcol = names.index("ticker")

        stats = []
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(tcol).statistics
            if st is None:
                stats.append((rg, None, None, md.row_group(rg).num_rows))
            else:
                stats.append((rg, st.min, st.max, md.row_group(rg).num_rows))

        have_stats = sum(1 for _, mn, _, _ in stats if mn is not None)
        print(f"\nrow groups with ticker statistics: {have_stats}/{len(stats)}")

        if have_stats == 0:
            print("VERDICT: no statistics -- every row group must be read. Full scan.")
            return

        mins = [mn for _, mn, _, _ in stats if mn is not None]
        sorted_by_ticker = all(mins[i] <= mins[i + 1] for i in range(len(mins) - 1))
        matching = [rg for rg, mn, mx, _ in stats
                    if mn is not None and any(mn <= t <= mx for t in TICKERS)]
        frac = len(matching) / len(stats)

        print(f"row groups sorted by ticker: {sorted_by_ticker}")
        print(f"row groups that could contain {TICKERS}: {len(matching)}/{len(stats)} ({frac:.1%})")
        print("\nfirst 5 row groups (rg, ticker_min, ticker_max, rows)")
        for row in stats[:5]:
            print(f"  {row}")

        if frac < 0.25:
            print(f"\nVERDICT: pruning works. Expect roughly {frac:.0%} of bytes transferred.")
            print("         Use --mode remote in the extract stage.")
        else:
            print("\nVERDICT: pruning saves little. Every row group holds all tickers.")
            print("         Use --mode download in the extract stage.")


# --------------------------------------------------------------------------
# Stage 3: extract
# --------------------------------------------------------------------------

def _month_range(start: str, end: str):
    s = pd.Period(start, freq="M")
    e = pd.Period(end, freq="M")
    out = []
    while s <= e:
        out.append(str(s))
        s += 1
    return out


def _read_month_remote(month: str) -> pd.DataFrame | None:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    remote = f"datasets/{REPO_ID}/{_month_path(month)}"
    with fs.open(remote, "rb") as fh:
        table = pq.read_table(
            fh,
            columns=COLUMNS,
            filters=[("ticker", "in", TICKERS)],
        )
    return table.to_pandas()


def _read_month_download(month: str, keep: bool) -> pd.DataFrame | None:
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=_month_path(month),
        local_dir=os.path.join(CACHE_DIR, "raw"),
    )
    try:
        table = pq.read_table(local, columns=COLUMNS, filters=[("ticker", "in", TICKERS)])
        return table.to_pandas()
    finally:
        if not keep:
            try:
                os.remove(local)
            except OSError:
                pass


def _read_month_stream(month: str) -> pd.DataFrame | None:
    """Fallback reader using the `datasets` library.

    Points at the Parquet file directly rather than going through the repo's
    split configuration, which is broken (the dataset viewer fails with
    FileFormatMismatchBetweenSplitsError). Slower than the pyarrow paths since
    it decodes every row before filtering, but it needs no Parquet statistics.
    """
    from datasets import load_dataset

    url = f"hf://datasets/{REPO_ID}/{_month_path(month)}"
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    keep = []
    wanted = set(TICKERS)
    for row in ds:
        if row.get("ticker") in wanted:
            keep.append({c: row.get(c) for c in COLUMNS})
    return pd.DataFrame(keep, columns=COLUMNS) if keep else None


def stage_extract(args):
    _hr(f"STAGE: extract ({args.start} to {args.end}, mode={args.mode})")
    months = _month_range(args.start, args.end)
    _log(f"{len(months)} months to process")

    rows_by_month = {}
    failures = []

    for i, month in enumerate(months, 1):
        cached = os.path.join(MONTH_CACHE_DIR, f"{month}.parquet")
        if os.path.exists(cached):
            try:
                n = pq.ParquetFile(cached).metadata.num_rows
                rows_by_month[month] = n
                continue
            except Exception:
                os.remove(cached)

        try:
            if args.mode == "remote":
                df = _read_month_remote(month)
            elif args.mode == "stream":
                df = _read_month_stream(month)
            else:
                df = _read_month_download(month, keep=args.keep_raw)
        except Exception as exc:
            msg = str(exc).split("\n")[0][:120]
            failures.append((month, msg))
            _log(f"[{i}/{len(months)}] {month}: FAILED ({msg})")
            continue

        n = 0 if df is None else len(df)
        rows_by_month[month] = n
        if df is not None and n:
            df.to_parquet(cached, index=False)
        else:
            # Write an empty marker so the month is not retried on resume.
            pd.DataFrame(columns=COLUMNS).to_parquet(cached, index=False)
        _log(f"[{i}/{len(months)}] {month}: {n:,} rows")

    _hr("extract summary")
    cov = pd.Series(rows_by_month).sort_index()
    cov.to_csv(os.path.join(OUT_DIR, "rows_per_month.csv"), header=["rows"])
    nonzero = cov[cov > 0]
    print(f"months processed:     {len(cov)}")
    print(f"months with QQQ rows: {len(nonzero)}")
    if len(nonzero):
        print(f"first month with data: {nonzero.index[0]}")
        print(f"last month with data:  {nonzero.index[-1]}")
    if failures:
        print(f"\nfailures: {len(failures)}")
        for m, e in failures[:10]:
            print(f"  {m}: {e}")

    # Combine.
    frames = []
    for month in months:
        p = os.path.join(MONTH_CACHE_DIR, f"{month}.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            if len(d):
                frames.append(d)
    if not frames:
        print("\nno rows extracted -- nothing to combine")
        return
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)
    full.to_parquet(COMBINED_PATH, index=False)
    _log(f"wrote {COMBINED_PATH}: {len(full):,} rows, {os.path.getsize(COMBINED_PATH)/1e6:.1f} MB")


# --------------------------------------------------------------------------
# Stage 4: quality checks
# --------------------------------------------------------------------------

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"], utc=True)
    d["ts_et"] = ts.dt.tz_convert(EXCHANGE_TZ)
    d["date"] = d["ts_et"].dt.date
    d["tod"] = d["ts_et"].dt.hour * 60 + d["ts_et"].dt.minute
    d = d.sort_values("ts_et").reset_index(drop=True)
    return d


def _rth_mask(d: pd.DataFrame) -> pd.Series:
    lo = RTH_OPEN[0] * 60 + RTH_OPEN[1]
    hi = RTH_CLOSE[0] * 60 + RTH_CLOSE[1]
    return (d["tod"] >= lo) & (d["tod"] < hi)


def check_coverage(d, report):
    _hr("1. COVERAGE")
    print(f"rows:  {len(d):,}")
    print(f"first: {d['ts_et'].iloc[0]}")
    print(f"last:  {d['ts_et'].iloc[-1]}")

    per_month = d.groupby(d["ts_et"].dt.to_period("M")).size()
    full_idx = pd.period_range(per_month.index[0], per_month.index[-1], freq="M")
    per_month = per_month.reindex(full_idx, fill_value=0)
    gaps = per_month[per_month == 0]
    print(f"\nmonths in span:    {len(per_month)}")
    print(f"months with 0 bars: {len(gaps)}")
    if len(gaps):
        print("  " + ", ".join(str(g) for g in gaps.index[:24]))

    thin = per_month[(per_month > 0) & (per_month < 8000)]
    print(f"months under 8,000 bars (incomplete): {len(thin)}")
    if len(thin):
        print("  " + ", ".join(f"{i}({v})" for i, v in list(thin.items())[:24]))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(per_month.index.to_timestamp(), per_month.values, width=25)
    ax.axhline(FULL_SESSION_BARS * 19, ls="--", lw=1, color="k", label="~19 full sessions")
    ax.set_title("QQQ 1-minute bars per month")
    ax.set_ylabel("bars")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_bars_per_month.png"), dpi=130)
    plt.close(fig)

    per_month.to_csv(os.path.join(OUT_DIR, "bars_per_month.csv"), header=["bars"])
    report["months_empty"] = int(len(gaps))
    report["months_thin"] = int(len(thin))
    report["first_bar"] = str(d["ts_et"].iloc[0])


def check_tickers(d, report):
    _hr("2. TICKER HISTORY (QQQ vs QQQQ)")
    tab = d.groupby([d["ts_et"].dt.year, "ticker"]).size().unstack(fill_value=0)
    print(tab.to_string())
    for tk in TICKERS:
        if tk in d["ticker"].unique():
            sub = d[d["ticker"] == tk]
            print(f"\n{tk}: {sub['ts_et'].iloc[0].date()} to {sub['ts_et'].iloc[-1].date()}")
    print("\nExpected: QQQ through 2004-11, QQQQ 2004-12 to 2011-03, QQQ after.")
    overlap = d.groupby("ts_et")["ticker"].nunique()
    n_ov = int((overlap > 1).sum())
    print(f"timestamps carrying both tickers: {n_ov:,}")
    tab.to_csv(os.path.join(OUT_DIR, "ticker_by_year.csv"))
    report["ticker_overlap_bars"] = n_ov


def check_sessions(d, report):
    _hr("3. SESSION COMPOSITION")
    rth = _rth_mask(d)
    pre = d["tod"] < RTH_OPEN[0] * 60 + RTH_OPEN[1]
    post = d["tod"] >= RTH_CLOSE[0] * 60 + RTH_CLOSE[1]
    print(f"regular hours (09:30-15:59): {rth.sum():,} ({rth.mean():.1%})")
    print(f"pre-market:                  {pre.sum():,} ({pre.mean():.1%})")
    print(f"post-market:                 {post.sum():,} ({post.mean():.1%})")
    if pre.sum() + post.sum() > 0.02 * len(d):
        print("\nExtended-hours bars present. Exclude them from RV.")
    else:
        print("\nRegular hours only.")

    per_day = d[rth].groupby("date").size()
    print(f"\nsessions: {len(per_day):,}")
    print(f"bars/session  mean {per_day.mean():.1f}  median {per_day.median():.0f}  "
          f"min {per_day.min()}  max {per_day.max()}")

    full = int((per_day == FULL_SESSION_BARS).sum())
    over = int((per_day > FULL_SESSION_BARS).sum())
    print(f"sessions with exactly 390 bars: {full:,} ({full/len(per_day):.1%})")
    print(f"sessions with more than 390:    {over:,}")

    # Half-day detection: last regular bar before 13:00 ET.
    last_bar = d[rth].groupby("date")["tod"].max()
    half = last_bar[last_bar < HALF_DAY_CLOSE[0] * 60 + HALF_DAY_CLOSE[1]]
    print(f"\nlikely half-days (last bar before 13:00): {len(half):,}")
    if len(half):
        print("  " + ", ".join(str(x) for x in list(half.index)[:10]))

    short = per_day[(per_day < FULL_SESSION_BARS) & (~per_day.index.isin(half.index))]
    print(f"short sessions not explained by half-days: {len(short):,}")
    if len(short):
        print(f"  worst 10: {short.nsmallest(10).to_dict()}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(per_day.values, bins=80)
    ax.axvline(FULL_SESSION_BARS, color="k", ls="--", label="390")
    ax.axvline(HALF_SESSION_BARS, color="r", ls="--", label="210 (half-day)")
    ax.set_title("Regular-hours bars per session")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_bars_per_session.png"), dpi=130)
    plt.close(fig)

    per_day.to_csv(os.path.join(OUT_DIR, "bars_per_session.csv"), header=["bars"])
    report["pct_full_sessions"] = round(full / len(per_day), 4)
    report["n_half_days"] = int(len(half))
    report["n_short_sessions"] = int(len(short))
    return per_day


def check_integrity(d, report):
    _hr("4. BAR INTEGRITY")
    dup = int(d.duplicated(subset=["ts_et", "ticker"]).sum())
    print(f"duplicate (timestamp, ticker) rows: {dup:,}")

    nn = d[["open", "high", "low", "close"]].isna().sum()
    print(f"\nnull OHLC:\n{nn.to_string()}")

    bad_hl = int((d["high"] < d["low"]).sum())
    bad_hi = int((d["high"] < d[["open", "close"]].max(axis=1)).sum())
    bad_lo = int((d["low"] > d[["open", "close"]].min(axis=1)).sum())
    nonpos = int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    print(f"\nhigh < low:                {bad_hl:,}")
    print(f"high < max(open, close):   {bad_hi:,}")
    print(f"low  > min(open, close):   {bad_lo:,}")
    print(f"non-positive prices:       {nonpos:,}")

    zero_vol = int((d["volume"] == 0).sum())
    print(f"\nzero-volume bars: {zero_vol:,} ({zero_vol/len(d):.2%})")
    flat = int((d["high"] == d["low"]).sum())
    print(f"flat bars (high == low): {flat:,} ({flat/len(d):.2%})")

    report.update(dict(duplicates=dup, bad_high_low=bad_hl + bad_hi + bad_lo,
                       nonpositive=nonpos, zero_volume_bars=zero_vol))


def build_daily(d):
    """Daily OHLC from regular-hours minute bars, plus overnight log return."""
    rth = d[_rth_mask(d)]
    g = rth.groupby("date")
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "bars": g.size(),
    })
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    daily["ret_cc"] = np.log(daily["close"]).diff()
    daily["ret_on"] = np.log(daily["open"] / daily["close"].shift(1))
    daily["ret_oc"] = np.log(daily["close"] / daily["open"])
    return daily


def check_splits(daily, report):
    _hr("5. SPLIT / CORPORATE ACTION DETECTION")
    print("The source is NOT split-adjusted (repo owner, community discussion #4).")
    flags = daily[daily["ret_cc"].abs() > SPLIT_FLAG_THRESHOLD].copy()
    flags["implied_ratio"] = np.exp(-flags["ret_cc"])
    print(f"\ndays with |close-to-close log return| > {SPLIT_FLAG_THRESHOLD}: {len(flags)}")
    if len(flags):
        show = flags[["close", "ret_cc", "implied_ratio"]].copy()
        show.index = show.index.date
        print(show.to_string())

    print("\nKnown splits to verify:")
    for date_str, ratio in KNOWN_SPLITS.items():
        dt = pd.Timestamp(date_str)
        if dt in daily.index:
            i = daily.index.get_loc(dt)
            if i > 0:
                prev_c = daily["close"].iloc[i - 1]
                this_o = daily["open"].iloc[i]
                obs = prev_c / this_o
                print(f"  {date_str} expected {ratio}:1 | prev close {prev_c:.2f} "
                      f"-> open {this_o:.2f} | observed ratio {obs:.3f}")
                if abs(obs - ratio) < 0.15:
                    print("    UNADJUSTED. Build the adjustment factor yourself.")
                elif abs(obs - 1.0) < 0.15:
                    print("    already adjusted.")
                else:
                    print("    inconclusive.")
        else:
            print(f"  {date_str}: not in sample")

    flags.to_csv(os.path.join(OUT_DIR, "split_candidates.csv"))
    report["split_flag_days"] = int(len(flags))


def check_missing_minutes(d, report):
    _hr("6. MISSING MINUTES WITHIN REGULAR HOURS")
    rth = d[_rth_mask(d)]
    per_day = rth.groupby("date").size()
    last_bar = rth.groupby("date")["tod"].max()
    half = set(last_bar[last_bar < HALF_DAY_CLOSE[0] * 60 + HALF_DAY_CLOSE[1]].index)
    expected = per_day.index.map(lambda x: HALF_SESSION_BARS if x in half else FULL_SESSION_BARS)
    missing = pd.Series(expected, index=per_day.index) - per_day
    missing = missing.clip(lower=0)
    print(f"total missing minutes: {int(missing.sum()):,}")
    print(f"sessions with any gap: {int((missing > 0).sum()):,} of {len(missing):,}")
    by_year = missing.groupby(pd.to_datetime(missing.index).year).mean()
    print("\nmean missing minutes per session, by year")
    print(by_year.round(1).to_string())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(missing.index), missing.values, lw=0.5)
    ax.set_title("Missing regular-hours minutes per session")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_missing_minutes.png"), dpi=130)
    plt.close(fig)
    report["total_missing_minutes"] = int(missing.sum())


def volatility_signature(d, report, subperiods=True):
    _hr("7. VOLATILITY SIGNATURE PLOT")
    print("Average annualized RV against sampling frequency. A feed with usable")
    print("microstructure properties flattens out somewhere near 5 minutes.")
    rth = d[_rth_mask(d)].set_index("ts_et")

    def rv_by_freq(frame):
        out = {}
        for k in SIGNATURE_FREQS:
            rvs = []
            for _, day in frame.groupby(frame.index.date):
                px = day["close"].resample(f"{k}min").last().dropna()
                if len(px) < 3:
                    continue
                r = np.diff(np.log(px.values))
                rvs.append(np.sum(r ** 2))
            out[k] = np.sqrt(np.mean(rvs) * 252) if rvs else np.nan
        return out

    full = rv_by_freq(rth)
    print("\nfull sample (annualized vol, open-to-close)")
    for k, v in full.items():
        print(f"  {k:>2} min: {v:.4f}")

    curves = {"full sample": full}
    if subperiods:
        bounds = [("2000-2004", "2000", "2004"), ("2005-2009", "2005", "2009"),
                  ("2010-2019", "2010", "2019"), ("2020-2026", "2020", "2026")]
        for label, a, b in bounds:
            sub = rth[(rth.index.year >= int(a)) & (rth.index.year <= int(b))]
            if len(sub) > 5000:
                curves[label] = rv_by_freq(sub)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in curves.items():
        ax.plot(list(curve.keys()), list(curve.values()), marker="o", label=label)
    ax.axvline(5, color="k", ls="--", lw=1, label="5 min")
    ax.set_xlabel("sampling frequency (minutes)")
    ax.set_ylabel("average annualized RV")
    ax.set_title("Volatility signature plot, QQQ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_signature_plot.png"), dpi=130)
    plt.close(fig)

    pd.DataFrame(curves).to_csv(os.path.join(OUT_DIR, "signature_plot.csv"))
    if not np.isnan(full[1]) and not np.isnan(full[5]) and full[5] > 0:
        ratio = full[1] / full[5]
        print(f"\nRV(1min) / RV(5min) = {ratio:.3f}")
        if ratio > 1.4:
            print("  Heavy noise at 1 minute, as expected. 5-minute sampling justified.")
        elif ratio < 1.02:
            print("  Nearly flat, which suggests stale or interpolated prices. Investigate.")
        report["signature_ratio_1_over_5"] = round(float(ratio), 3)


def check_rv_vs_daily(d, daily, report):
    _hr("8. INTRADAY RV vs DAILY CLOSE-TO-CLOSE VOL")
    rth = d[_rth_mask(d)].set_index("ts_et")
    rows = []
    for day, frame in rth.groupby(rth.index.date):
        px = frame["close"].resample("5min").last().dropna()
        if len(px) < 20:
            continue
        r = np.diff(np.log(px.values))
        rows.append((day, np.sum(r ** 2)))
    rv = pd.Series(dict(rows))
    rv.index = pd.to_datetime(rv.index)
    joined = pd.DataFrame({"rv5": rv}).join(daily[["ret_cc", "ret_on"]], how="inner").dropna()
    joined["rv_total"] = joined["rv5"] + joined["ret_on"] ** 2

    ann_rv = np.sqrt(joined["rv_total"].mean() * 252)
    ann_cc = joined["ret_cc"].std() * np.sqrt(252)
    print(f"annualized vol from 5-min RV + overnight: {ann_rv:.4f}")
    print(f"annualized vol from daily close-to-close: {ann_cc:.4f}")
    print(f"ratio: {ann_rv / ann_cc:.3f}   (expect roughly 0.9 to 1.1)")
    print(f"\novernight share of total variance: "
          f"{(joined['ret_on'] ** 2).mean() / joined['rv_total'].mean():.1%}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(joined.index, np.sqrt(joined["rv_total"] * 252), lw=0.6, label="RV (5min + overnight)")
    ax.plot(joined.index, joined["ret_cc"].rolling(21).std() * np.sqrt(252),
            lw=1.0, label="21d close-to-close")
    ax.set_title("Annualized volatility, QQQ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_rv_vs_daily.png"), dpi=130)
    plt.close(fig)

    joined.to_csv(os.path.join(OUT_DIR, "daily_rv.csv"))
    report["ann_vol_rv"] = round(float(ann_rv), 4)
    report["ann_vol_cc"] = round(float(ann_cc), 4)


def stage_checks(args):
    path = args.input or COMBINED_PATH
    if not os.path.exists(path):
        print(f"missing {path}. Run the extract stage first.")
        return
    _log(f"loading {path}")
    df = pd.read_parquet(path)
    _log(f"{len(df):,} rows")
    d = _prepare(df)

    report = {}
    check_coverage(d, report)
    check_tickers(d, report)
    check_sessions(d, report)
    check_integrity(d, report)
    daily = build_daily(d)
    check_splits(daily, report)
    check_missing_minutes(d, report)
    volatility_signature(d, report, subperiods=not args.fast)
    check_rv_vs_daily(d, daily, report)

    _hr("SUMMARY")
    for k, v in report.items():
        print(f"  {k:<28} {v}")
    pd.Series(report).to_csv(os.path.join(OUT_DIR, "summary.csv"), header=["value"])
    daily.to_csv(os.path.join(OUT_DIR, "daily_ohlc.csv"))
    _log(f"outputs in {OUT_DIR}/")


# --------------------------------------------------------------------------
# Stage 0: selftest
# --------------------------------------------------------------------------

def _synthetic(n_days=520, seed=7):
    """Synthetic minute bars with a planted split, gaps, and half-days."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2000-01-03", periods=n_days, tz=EXCHANGE_TZ)
    rows = []
    price = 100.0
    split_day = days[54]
    for day in days:
        n_bars = HALF_SESSION_BARS if day.dayofyear % 97 == 0 else FULL_SESSION_BARS
        idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                            periods=n_bars, freq="1min")
        if day == split_day:
            price /= 2.0
        price *= np.exp(rng.normal(0, 0.004))  # overnight
        sig = 0.0006 * (1 + 0.5 * rng.random())
        steps = rng.normal(0, sig, n_bars)
        closes = price * np.exp(np.cumsum(steps))
        noise = rng.normal(0, 0.0004, n_bars)  # microstructure noise
        obs = closes * np.exp(noise)
        opens = np.concatenate([[price], obs[:-1]])
        highs = np.maximum(opens, obs) * (1 + rng.random(n_bars) * 0.0003)
        lows = np.minimum(opens, obs) * (1 - rng.random(n_bars) * 0.0003)
        vols = rng.integers(100, 90000, n_bars).astype(float)
        keep = rng.random(n_bars) > 0.004  # planted gaps
        tk = "QQQ" if day < pd.Timestamp("2000-06-01", tz=EXCHANGE_TZ) else "QQQQ"
        rows.append(pd.DataFrame({
            "timestamp": idx.tz_convert("UTC")[keep],
            "open": opens[keep], "high": highs[keep], "low": lows[keep],
            "close": obs[keep], "volume": vols[keep], "ticker": tk,
        }))
        price = closes[-1]
    return pd.concat(rows, ignore_index=True)


def stage_selftest(args):
    _hr("STAGE: selftest (synthetic data, no network)")
    df = _synthetic()
    _log(f"synthetic rows: {len(df):,}")
    d = _prepare(df)
    report = {}
    ok = True
    for name, fn in [
        ("coverage", lambda: check_coverage(d, report)),
        ("tickers", lambda: check_tickers(d, report)),
        ("sessions", lambda: check_sessions(d, report)),
        ("integrity", lambda: check_integrity(d, report)),
    ]:
        try:
            fn()
        except Exception:
            ok = False
            print(f"\n!! {name} FAILED")
            traceback.print_exc()

    try:
        daily = build_daily(d)
        check_splits(daily, report)
        check_missing_minutes(d, report)
        volatility_signature(d, report, subperiods=False)
        check_rv_vs_daily(d, daily, report)
    except Exception:
        ok = False
        print("\n!! downstream check FAILED")
        traceback.print_exc()

    _hr("SELFTEST RESULT")
    print("PASS" if ok else "FAIL")
    print("\nThe planted 2-for-1 split should appear in section 5, the planted")
    print("half-days in section 3, and the signature plot should slope down from")
    print("1 minute given the planted microstructure noise.")
    for k, v in report.items():
        print(f"  {k:<28} {v}")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    sub.add_parser("selftest")
    sub.add_parser("inventory")

    pp = sub.add_parser("probe")
    pp.add_argument("--month", default="2010-06", help="YYYY-MM")

    pe = sub.add_parser("extract")
    pe.add_argument("--start", default="1999-01")
    pe.add_argument("--end", default="2026-03")
    pe.add_argument("--mode", choices=["remote", "download", "stream"], default="remote",
                    help="remote = HTTP range reads with row-group pruning; "
                         "download = fetch and discard each monthly file; "
                         "stream = `datasets` library fallback")
    pe.add_argument("--keep-raw", action="store_true",
                    help="keep the downloaded monthly files (needs ~88 GB)")

    pc = sub.add_parser("checks")
    pc.add_argument("--input", default=None)
    pc.add_argument("--fast", action="store_true", help="skip signature subperiods")

    pa_ = sub.add_parser("all")
    pa_.add_argument("--start", default="1999-01")
    pa_.add_argument("--end", default="2026-03")
    pa_.add_argument("--mode", choices=["remote", "download", "stream"], default="remote")
    pa_.add_argument("--keep-raw", action="store_true")
    pa_.add_argument("--month", default="2010-06")
    pa_.add_argument("--input", default=None)
    pa_.add_argument("--fast", action="store_true")

    args = p.parse_args()
    _ensure_dirs()

    if args.stage == "selftest":
        return stage_selftest(args)
    if args.stage == "inventory":
        return stage_inventory(args)
    if args.stage == "probe":
        return stage_probe(args)
    if args.stage == "extract":
        return stage_extract(args)
    if args.stage == "checks":
        return stage_checks(args)
    if args.stage == "all":
        stage_inventory(args)
        stage_probe(args)
        stage_extract(args)
        stage_checks(args)
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

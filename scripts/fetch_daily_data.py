#!/usr/bin/env python3
"""
One-shot daily-data acquisition for the vol-targeted QQQ study (fallback path).

Pulls each series EXACTLY ONCE and caches it to data/raw/ as Parquet plus a JSON
sidecar (source, call, retrieval timestamp, row count, date range). Re-running is
idempotent: any series whose Parquet already exists is skipped, so development
never re-hits a rate-limited API. To force a refetch, delete the cached file.

Series
------
  qqq_daily   yfinance  QQQ  from 1999-03-10, auto_adjust=True, actions=True
  ndx_daily   yfinance  ^NDX from 1985-01-01, auto_adjust=True
  qqq_stooq   stooq     QQQ.US daily (independent close cross-check)
  dff         FRED      DFF (effective fed funds, daily), full history

Usage:  python scripts/fetch_daily_data.py [--force name[,name...]]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

RAW_DIR = os.path.join("data", "raw")

QQQ_START = "1999-03-10"
NDX_START = "1985-01-01"
DFF_START = "1954-07-01"   # DFF begins 1954-07-01 on FRED


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _write(name: str, df: pd.DataFrame, meta: dict):
    os.makedirs(RAW_DIR, exist_ok=True)
    pq_path = os.path.join(RAW_DIR, f"{name}.parquet")
    df.to_parquet(pq_path, index=False)
    meta = {
        **meta,
        "name": name,
        "retrieved_at": _now_iso(),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "parquet": pq_path,
        "parquet_bytes": os.path.getsize(pq_path),
    }
    # date range for the record
    datecol = "date" if "date" in df.columns else df.columns[0]
    try:
        d = pd.to_datetime(df[datecol])
        meta["date_min"] = str(d.min().date())
        meta["date_max"] = str(d.max().date())
    except Exception:
        pass
    with open(os.path.join(RAW_DIR, f"{name}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    _log(f"wrote {pq_path}: {len(df):,} rows ({meta.get('date_min','?')} .. "
         f"{meta.get('date_max','?')})")


def _cached(name: str) -> bool:
    return os.path.exists(os.path.join(RAW_DIR, f"{name}.parquet"))


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    """yf.download can return a (field, ticker) MultiIndex; drop the ticker level."""
    if isinstance(df.columns, pd.MultiIndex):
        # keep the price-field level; the other level is the single ticker
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        # the level with >1 unique value is the field level
        if df.columns.get_level_values(0).nunique() >= df.columns.get_level_values(1).nunique():
            df.columns = lvl0
        else:
            df.columns = lvl1
    return df


def _yf_download(ticker: str, start: str, actions: bool) -> pd.DataFrame:
    import yfinance as yf
    last_exc = None
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, actions=actions,
                             progress=False, threads=False)
            if df is not None and len(df):
                df = _flatten_yf(df.copy())
                df = df.reset_index()
                df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
                if "date" not in df.columns and "index" in df.columns:
                    df = df.rename(columns={"index": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                return df
            _log(f"  {ticker}: empty result (attempt {attempt})")
        except Exception as e:
            last_exc = e
            _log(f"  {ticker}: attempt {attempt} error: {str(e)[:100]}")
        time.sleep(5 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{ticker}: empty after retries")


def fetch_qqq_daily():
    df = _yf_download("QQQ", QQQ_START, actions=True)
    _write("qqq_daily", df, {
        "source": "yahoo (yfinance)",
        "call": f'yf.download("QQQ", start="{QQQ_START}", auto_adjust=True, actions=True)',
    })


def fetch_ndx_daily():
    df = _yf_download("^NDX", NDX_START, actions=True)
    _write("ndx_daily", df, {
        "source": "yahoo (yfinance)",
        "call": f'yf.download("^NDX", start="{NDX_START}", auto_adjust=True, actions=True)',
    })


def fetch_ixic_daily():
    # Nasdaq Composite: seed candidate (same market as the Nasdaq-100 underlying).
    df = _yf_download("^IXIC", NDX_START, actions=True)
    _write("ixic_daily", df, {
        "source": "yahoo (yfinance)",
        "call": f'yf.download("^IXIC", start="{NDX_START}", auto_adjust=True, actions=True)',
    })


def fetch_gspc_daily():
    # S&P 500: seed candidate (broad-market range-estimator alternative).
    df = _yf_download("^GSPC", NDX_START, actions=True)
    _write("gspc_daily", df, {
        "source": "yahoo (yfinance)",
        "call": f'yf.download("^GSPC", start="{NDX_START}", auto_adjust=True, actions=True)',
    })


def fetch_qqq_stooq():
    from pandas_datareader import data as pdr
    last_exc = None
    for attempt in range(1, 4):
        try:
            df = pdr.DataReader("QQQ.US", "stooq")  # newest-first
            if df is not None and len(df):
                df = df.sort_index().reset_index()
                df.columns = [str(c).strip().lower() for c in df.columns]
                df = df.rename(columns={"index": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                _write("qqq_stooq", df, {
                    "source": "stooq",
                    "call": 'pandas_datareader.data.DataReader("QQQ.US", "stooq")',
                    "url": "https://stooq.com/q/d/l/?s=qqq.us&i=d",
                })
                return
            _log(f"  stooq QQQ: empty (attempt {attempt})")
        except Exception as e:
            last_exc = e
            _log(f"  stooq QQQ: attempt {attempt} error: {str(e)[:100]}")
        time.sleep(5 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("stooq QQQ: empty after retries")


def fetch_dff():
    from pandas_datareader import data as pdr
    last_exc = None
    for attempt in range(1, 4):
        try:
            df = pdr.DataReader("DFF", "fred", start=DFF_START)
            if df is not None and len(df):
                df = df.reset_index()
                df.columns = [str(c).strip().lower() for c in df.columns]
                df = df.rename(columns={"date": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                _write("dff", df, {
                    "source": "FRED",
                    "call": f'pandas_datareader.data.DataReader("DFF", "fred", start="{DFF_START}")',
                    "url": "https://fred.stlouisfed.org/series/DFF",
                })
                return
            _log(f"  FRED DFF: empty (attempt {attempt})")
        except Exception as e:
            last_exc = e
            _log(f"  FRED DFF: attempt {attempt} error: {str(e)[:100]}")
        time.sleep(5 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError("FRED DFF: empty after retries")


FETCHERS = {
    "qqq_daily": fetch_qqq_daily,
    "ndx_daily": fetch_ndx_daily,
    "ixic_daily": fetch_ixic_daily,
    "gspc_daily": fetch_gspc_daily,
    "qqq_stooq": fetch_qqq_stooq,
    "dff": fetch_dff,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", default="", help="comma-separated series to refetch")
    args = ap.parse_args()
    force = set(s for s in args.force.split(",") if s)

    for name, fn in FETCHERS.items():
        if _cached(name) and name not in force:
            _log(f"{name}: cached, skipping (delete data/raw/{name}.parquet to refetch)")
            continue
        _log(f"{name}: fetching ...")
        try:
            fn()
        except Exception as e:
            _log(f"{name}: FAILED -- {type(e).__name__}: {str(e)[:160]}")

    print("\n=== data/raw/ ===", flush=True)
    if os.path.isdir(RAW_DIR):
        for f in sorted(os.listdir(RAW_DIR)):
            p = os.path.join(RAW_DIR, f)
            print(f"  {f:<28} {os.path.getsize(p):>12,} bytes")


if __name__ == "__main__":
    main()

"""
Estimation-seed construction for the RV family.

The RV family (rv, har, rfsv) needs a realized-variance history to fit and to set
its unconditional level via variance targeting. The NDX seed is unusable for
range estimators (open == prior close pre-2000; SPEC.md Section 13), so ^IXIC
(Nasdaq Composite, clean OHLC opens from 1986) supplies the range-estimator seed.

^IXIC is less volatile than the Nasdaq-100 (fewer, less-concentrated names), so
the IXIC realized-variance seed is rescaled by a single scalar

    c = var(NDX close-to-close returns) / mean(IXIC daily RV)

computed over the seed window. This gives the RV family the same unconditional
volatility level that variance targeting pins for the GARCH family (var of NDX cc
returns), so a level difference between two indices is not baked into a
comparison meant to isolate a model difference (SPEC.md Section 13, 2026-08-06).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from volteq.config import load_config, repo_root
from volteq.rv.estimators import daily_proxy_rv
import os

SEED_TAKEOVER = "1999-03-10"   # QQQ realized measures take over here


def _load_raw(symbol_file: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{symbol_file}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def seed_scalar(cfg: dict | None = None) -> float:
    """c = var(NDX cc returns) / mean(IXIC daily RV) over the seed window.

    Includes October 1987 by default (SPEC.md Section 13; do not exclude on own
    initiative). Both terms are computed over the common seed dates.
    """
    cfg = cfg or load_config()
    s = cfg["realized_measures"]["seed"]
    start, end = s["start"], s["end"]

    ndx = _load_raw("ndx_daily").loc[start:end]
    ixic = _load_raw("ixic_daily").loc[start:end]

    ndx_cc = np.log(ndx["close"]).diff().dropna()
    ixic_rv = daily_proxy_rv(ixic)["rv_daily"].dropna()
    common = ndx_cc.index.intersection(ixic_rv.index)
    return float(ndx_cc.loc[common].var(ddof=1) / ixic_rv.loc[common].mean())

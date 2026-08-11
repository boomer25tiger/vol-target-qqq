"""
GARCH-family estimation return series: spliced close-to-close log returns.

Per SPEC.md Section 3.3: NDX daily returns 1985 -> 1999-03-09, then QQQ from
1999-03-10. The bridging session (log QQQ_1999-03-10 / NDX_1999-03-09) crosses
indices and is dropped, so QQQ returns begin 1999-03-11. NDX is estimation-only;
it never enters the *traded* series (that is QQQ, built later). These are the
returns the GARCH family (garch, egarch, gjr, ewma) is fit on.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from volteq.config import load_config, repo_root

SEED_TAKEOVER = "1999-03-10"


def _cc(symbol_file: str) -> pd.Series:
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{symbol_file}.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return np.log(df["close"]).diff()


def build_garch_returns(cfg: dict | None = None) -> pd.DataFrame:
    """Spliced close-to-close log returns with a 'source' label ('ndx'|'qqq')."""
    cfg = cfg or load_config()
    ndx_end = cfg["data"]["seed_source"]["end"]          # 1999-03-09
    ndx = _cc("ndx_daily").loc[:ndx_end].dropna()
    qqq = _cc("qqq_daily").loc[SEED_TAKEOVER:].dropna()  # first valid QQQ cc is 1999-03-11
    out = pd.concat([
        pd.DataFrame({"ret": ndx, "source": "ndx"}),
        pd.DataFrame({"ret": qqq, "source": "qqq"}),
    ]).sort_index()
    # guard: the splice must be clean (NDX strictly before takeover, QQQ from it)
    assert (out.loc[out["source"] == "ndx"].index < pd.Timestamp(SEED_TAKEOVER)).all()
    assert (out.loc[out["source"] == "qqq"].index >= pd.Timestamp(SEED_TAKEOVER)).all()
    return out

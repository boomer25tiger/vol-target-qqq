"""
Realized-measure panel for the daily (fallback) path.

Splices the level-matched ^IXIC range-estimator seed (rescaled by ``c``) onto the
QQQ-derived realized measures at 1999-03-10. Every column is a *trailing* measure
dated t and computable at the close of t - no forward information. The forward
21-day realized variance used for evaluation lives ONLY in ``forward_target.py``.

Columns
-------
  source         'ixic_seed' before 1999-03-10, 'qqq' from 1999-03-10
  overnight_var  squared overnight log return (rescaled by c on the seed)
  rs             Rogers-Satchell daily variance (rescaled by c on the seed)
  rv_daily       daily proxy = overnight_var + rs (rescaled by c on the seed)
  log_rv         log(rv_daily)  -- for HAR-RV lags and the rough-vol series
  yz_21          trailing Yang-Zhang variance over the horizon window (rescaled)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from volteq.config import load_config, repo_root
from volteq.rv.estimators import daily_proxy_rv, yang_zhang
from volteq.rv.seed import seed_scalar, SEED_TAKEOVER, _load_raw

VAR_COLS = ["overnight_var", "rs", "rv_daily", "yz_21"]


def _measures_for(df: pd.DataFrame, n: int) -> pd.DataFrame:
    dp = daily_proxy_rv(df)
    out = pd.DataFrame(index=df.index)
    out["overnight_var"] = dp["overnight_var"]
    out["rs"] = dp["rs"]
    out["rv_daily"] = dp["rv_daily"]
    out["yz_21"] = yang_zhang(df, n)
    return out


def build_panel(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config()
    n = int(cfg["realized_measures"]["seed"]["yz_window_days"])
    horizon = int(cfg["frozen"]["forecast_horizon_days"])
    if n != horizon:
        # the rolling YZ window is tied to the frozen horizon by design
        raise ValueError(f"yz_window_days ({n}) != forecast_horizon_days ({horizon})")

    seed_start = cfg["realized_measures"]["seed"]["start"]
    seed_end = cfg["realized_measures"]["seed"]["end"]

    c = seed_scalar(cfg)

    # seed: IXIC measures over [seed_start, seed_end], rescaled by c
    ixic = _load_raw("ixic_daily").loc[seed_start:seed_end]
    seed = _measures_for(ixic, n)
    seed[VAR_COLS] = seed[VAR_COLS] * c
    seed["source"] = "ixic_seed"

    # traded era: QQQ measures from the takeover date onward (no rescale)
    qqq = _load_raw("qqq_daily").loc[SEED_TAKEOVER:]
    trad = _measures_for(qqq, n)
    trad["source"] = "qqq"

    # splice: seed strictly before takeover, QQQ from takeover
    seed = seed.loc[:pd.Timestamp(SEED_TAKEOVER) - pd.Timedelta(days=1)]
    panel = pd.concat([seed, trad]).sort_index()
    panel["log_rv"] = np.log(panel["rv_daily"])
    panel = panel[["source", "overnight_var", "rs", "rv_daily", "log_rv", "yz_21"]]

    meta = {
        "c_seed_scalar": c,
        "seed_window": [seed_start, seed_end],
        "seed_takeover": SEED_TAKEOVER,
        "yz_window_days": n,
        "rows": int(len(panel)),
        "rows_seed": int((panel["source"] == "ixic_seed").sum()),
        "rows_qqq": int((panel["source"] == "qqq").sum()),
        "date_min": str(panel.index.min().date()),
        "date_max": str(panel.index.max().date()),
        "include_october_1987": True,
    }
    return panel, meta


def cache_path() -> str:
    return os.path.join(repo_root(), "data", "processed", "rv_panel.parquet")


def write_panel(panel: pd.DataFrame, meta: dict) -> str:
    import json
    os.makedirs(os.path.dirname(cache_path()), exist_ok=True)
    p = cache_path()
    panel.reset_index(names="date").to_parquet(p, index=False)
    with open(p.replace(".parquet", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return p


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(cache_path())
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

#!/usr/bin/env python3
"""
G1: egarch daily forecasts for the Layer 1 SECONDARY overlapping-daily sample.

For each frozen-parameter month (params from the most recent rebalance on or
before each day), fix the egarch parameters, filter the daily state, and run one
10,000-path 21-step Monte Carlo per forecast origin via arch's simulation
forecast. No refitting at daily frequency; no look-ahead (the forecast at origin d
conditions only on data through d). Caches data/processed/egarch_daily_secondary.parquet.
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.models.returns import build_garch_returns                     # noqa: E402

SCALE2 = 100.0 ** 2
H = 21


def egarch_daily(cfg, dist="skewt"):
    from arch import arch_model
    seed = int(cfg["meta"]["random_seed"])
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    rets = build_garch_returns(cfg)["ret"]
    r_all = rets * 100.0

    pp = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "param_paths.parquet"))
    e = pp[(pp.model == "egarch") & (pp.dist == dist)].copy()
    e["date"] = pd.to_datetime(e["date"]); e = e.set_index("date").sort_index()
    param_dates = e.index

    eval_days = r_all.loc[eval_start:].index
    # group each eval day by the governing param date (most recent rebalance <= d)
    gov = param_dates[np.searchsorted(param_dates, eval_days.values, side="right") - 1]
    groups = pd.Series(eval_days, index=gov).groupby(level=0)

    out = {}
    t0 = time.time()
    for gi, (r_i, days) in enumerate(groups):
        params = json.loads(e.loc[r_i, "params_json"])
        last = days.max()
        am = arch_model(r_all.loc[:last], mean="Constant", vol="EGARCH",
                        p=1, o=1, q=1, dist=dist)
        res = am.fix(list(params.values()))
        fc = res.forecast(horizon=H, method="simulation", simulations=10000,
                          start=days.min(), random_state=np.random.RandomState(seed + gi),
                          reindex=False)
        V = fc.variance.mean(axis=1) / SCALE2          # per-origin V_t(21), raw
        for d in days:
            out[d] = float(V.loc[d])
    wall = time.time() - t0
    s = pd.Series(out).sort_index()
    s.index.name = "date"; s.name = "egarch_daily"
    return s, wall


def main():
    cfg = load_config()
    s, wall = egarch_daily(cfg)
    p = os.path.join(repo_root(), "data", "processed", "egarch_daily_secondary.parquet")
    s.reset_index().to_parquet(p, index=False)
    print(f"[egarch-daily] {len(s)} daily forecasts {s.index.min().date()}..{s.index.max().date()}")
    print(f"[egarch-daily] wall time {wall:.1f}s ({wall/60:.1f} min); path count 10000; wrote {p}")


if __name__ == "__main__":
    main()

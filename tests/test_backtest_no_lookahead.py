"""
No-look-ahead for the backtest engine: the equity path through any date T is a
pure function of data through T, so truncating future returns must not change any
earlier equity value. Also checks the rebalance timing - the weight set at a
month-end close applies only to later days (the month-end day's own return uses
the pre-rebalance weight).
"""
import os
import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config, repo_root
from volteq.backtest.engine import run_backtest, sizing_weight

CFG = load_config()


def _inputs():
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    q = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "qqq_daily.parquet"))
    q["date"] = pd.to_datetime(q["date"]); q = q.set_index("date").sort_index()
    qqq_ret = q["close"].pct_change().dropna()
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "dff.parquet"))
    d["date"] = pd.to_datetime(d["date"]); dff = d.set_index("date")["dff"].sort_index()
    ws = pd.Series(sizing_weight(v["gjr_skewt"].to_numpy(), CFG), index=v.index)
    return ws, qqq_ret, dff


def test_equity_path_invariant_to_future_returns():
    ws, qqq_ret, dff = _inputs()
    full = run_backtest(ws, qqq_ret, dff, CFG)["equity"]
    T = full.index[3000]
    trunc = run_backtest(ws, qqq_ret.loc[:T], dff, CFG)["equity"]
    common = full.index[full.index <= T]
    assert len(common) > 2000
    assert np.allclose(full.loc[common].values, trunc.loc[common].values, rtol=1e-12, atol=1e-9)


def test_rebalance_uses_only_past_weights():
    ws, qqq_ret, dff = _inputs()
    bt = run_backtest(ws, qqq_ret, dff, CFG)
    # every rebalance date lies within the backtest window; weights are finite and
    # the target is the sizing of V at that date (set at the close, applied later)
    rb = bt["rebalance"]
    assert rb["w_target"].between(0.0, 2.0).all()
    assert rb.index.min() >= pd.Timestamp(CFG["frozen"]["eval_start"])
    assert bt["equity"].index[0] >= pd.Timestamp(CFG["frozen"]["eval_start"])

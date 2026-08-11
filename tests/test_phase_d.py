"""
Phase D verification tests.
- D3 decomposition identity: mean gross excess = w_bar E[x_Q] + Cov(w,x_Q) - drag.
- Financing sign: constant leverage w=1.5 has excess Sharpe strictly below
  buy-and-hold under the 50bp spread.
- Retransformation fixture: exp(mu_hat + s2_hat/2) recovers a known lognormal mean.
"""
import os
import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config, repo_root
from volteq.backtest.engine import run_backtest, sizing_weight, metrics, ANNUAL, DAYCOUNT

CFG = load_config()


def _inputs():
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    q = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "qqq_daily.parquet"))
    q["date"] = pd.to_datetime(q["date"]); qqq_ret = q.set_index("date")["close"].sort_index().pct_change().dropna()
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "dff.parquet"))
    d["date"] = pd.to_datetime(d["date"]); dff = d.set_index("date")["dff"].sort_index()
    return v, qqq_ret, dff


def test_d3_decomposition_identity():
    v, qqq_ret, dff = _inputs()
    spread = CFG["frozen"]["financing"]["borrow_spread"]
    for col in ["gjr_skewt", "ewma", "rv"]:
        ws = pd.Series(sizing_weight(v[col].to_numpy(), CFG), index=v.index)
        bt = run_backtest(ws, qqq_ret, dff, CFG)
        rp, cash = bt["ret"], bt["cash_ret"]
        days = rp.index
        lev, reb = bt["leverage"], bt["rebalance"]
        w0 = float(ws.loc[ws.index < days[0]].iloc[-1])
        w = pd.Series(index=days, dtype=float); w.iloc[0] = w0
        for i in range(1, len(days)):
            p = days[i - 1]
            w.iloc[i] = float(reb.loc[p, "w_target"]) if p in reb.index else float(lev.loc[p])
        xQ = qqq_ret.reindex(days) - cash
        dc = pd.Series(days, index=days).diff().dt.days.fillna(3).astype(float)
        drag = np.maximum(w - 1.0, 0.0) * spread * dc / DAYCOUNT
        gross = w * xQ - drag
        recon = w.mean() * xQ.mean() + np.cov(w.values, xQ.values, ddof=0)[0, 1] - drag.mean()
        assert abs(gross.mean() - recon) < 1e-10, col


def test_constant_leverage_above_one_lowers_excess_sharpe():
    # SYNTHETIC constant leverage (daily-held, no drift) isolates the financing
    # sign: excess = w*x_Q - (w-1)*spread. The borrow spread makes SR(w>1) < SR_BH.
    _, qqq_ret, dff = _inputs()
    eval_start = pd.Timestamp(CFG["frozen"]["eval_start"])
    spread = CFG["frozen"]["financing"]["borrow_spread"]
    days = qqq_ret.loc[eval_start:].index
    rq = qqq_ret.reindex(days)
    dc = pd.Series(days, index=days).diff().dt.days.fillna(3.0).astype(float)
    base = dff.reindex(days, method="ffill") / 100.0
    cash = base * dc / DAYCOUNT
    xQ = rq - cash
    sr_bh = xQ.mean() / xQ.std() * np.sqrt(ANNUAL)               # w = 1
    for w in (1.5, 2.0):
        excess = w * xQ - (w - 1.0) * spread * dc / DAYCOUNT
        sr = excess.mean() / excess.std() * np.sqrt(ANNUAL)
        assert sr < sr_bh, f"w={w}: excess Sharpe {sr:.4f} !< buy-hold {sr_bh:.4f}"


def test_daily_returns_cumulate_to_equity_path():
    # After the E2 fix the turnover cost is in the daily return, so cumulating the
    # daily returns must reproduce the equity path for every strategy.
    v, qqq_ret, dff = _inputs()
    E0 = CFG["frozen"]["starting_capital"]
    cols = list(v.columns) + ["__buyhold__"]
    for c in cols:
        ws = (pd.Series(1.0, index=v.index) if c == "__buyhold__"
              else pd.Series(sizing_weight(v[c].to_numpy(), CFG), index=v.index))
        bt = run_backtest(ws, qqq_ret, dff, CFG)
        recon = E0 * (1.0 + bt["ret"]).cumprod()
        assert float((recon / bt["equity"] - 1.0).abs().max()) < 1e-10, c


def test_lognormal_retransformation_recovers_mean():
    rng = np.random.RandomState(20260806)
    m, s = 0.5, 0.8
    x = rng.lognormal(mean=m, sigma=s, size=3_000_000)  # E[x] = exp(m + s^2/2)
    true_mean = np.exp(m + 0.5 * s ** 2)
    logx = np.log(x)
    recovered = np.exp(logx.mean() + 0.5 * logx.var(ddof=1))
    assert abs(recovered / true_mean - 1.0) < 0.005

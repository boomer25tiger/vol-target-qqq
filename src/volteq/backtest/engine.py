"""
Monthly-rebalanced volatility-targeting backtest engine (SPEC.md Section 8).

Sizing at each rebalance date t: w_t = clip(0.20 / sqrt(252·V_t(21)), 0, 2).
The dollar legs (QQQ notional Q, cash/borrow leg C) are set at the rebalance close
and held for the month, so leverage floats within the month (no intramonth
rebalancing). The cash leg earns the effective fed funds rate (FRED DFF) when
w ≤ 1 and pays DFF + 50bp when w > 1, accrued over calendar days (actual/360).
Turnover at each rebalance is measured against the drifted weight and charged 2bp.

All weights are set from information available at the rebalance close; the weight
active on trading day d comes from the last rebalance strictly before d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252
DAYCOUNT = 360.0   # actual/360 money-market convention for fed funds


def sizing_weight(v_daily_var, cfg: dict):
    fz = cfg["frozen"]
    ann_vol = np.sqrt(ANNUAL * np.asarray(v_daily_var, float))
    return np.clip(fz["target_vol"] / ann_vol, fz["leverage_floor"], fz["leverage_cap"])


def run_backtest(wsched: pd.Series, qqq_ret: pd.Series, dff: pd.Series, cfg: dict) -> dict:
    fz = cfg["frozen"]
    cap0 = float(fz["starting_capital"])
    spread = float(fz["financing"]["borrow_spread"])
    tx = float(fz["transaction_cost"]["round_trip"])
    eval_start = pd.Timestamp(fz["eval_start"])

    reb = wsched.dropna().sort_index()
    days = qqq_ret.loc[eval_start:].index
    first = days[0]
    warm = reb.index[reb.index < first]
    if len(warm) == 0:
        raise ValueError("no rebalance date before eval_start (need warmup weight)")
    prev = warm[-1]
    w0 = float(reb.loc[prev])
    reb_set = set(reb.index)

    E = cap0
    Q, C = w0 * E, (1.0 - w0) * E
    borrow = w0 > 1.0

    idx, equity, dret, cash_ret, fin_pnl, levg = [], [], [], [], [], []
    rb_dt, wt_tar, wt_dr, turn, cost_l = [], [], [], [], []
    breach_months = set()

    for d in days:
        r = float(qqq_ret[d])
        dc = (d - prev).days
        base = float(dff.asof(prev)) / 100.0
        rate = base + (spread if borrow else 0.0)
        fin = rate * dc / DAYCOUNT
        cash_r = base * dc / DAYCOUNT
        E_prev = E
        fin_dollars = C * fin
        Q *= (1.0 + r)
        C *= (1.0 + fin)
        E = Q + C
        lev_drift = Q / E if E > 0 else np.nan     # drifted weight, pre-rebalance
        if E > 0 and E < 0.25 * Q:                 # Reg-T maintenance breach (drifted position)
            breach_months.add(d.to_period("M"))
        if d in reb_set:
            wd = Q / E
            wtar = float(reb[d])
            to = abs(wtar - wd)
            cst = tx * to * E
            E -= cst                               # cost enters the rebalance-day return
            Q, C = wtar * E, (1.0 - wtar) * E
            borrow = wtar > 1.0
            rb_dt.append(d); wt_tar.append(wtar); wt_dr.append(wd)
            turn.append(to); cost_l.append(cst)
        idx.append(d); equity.append(E); dret.append(E / E_prev - 1.0)
        cash_ret.append(cash_r); fin_pnl.append(fin_dollars); levg.append(lev_drift)
        prev = d

    eq = pd.Series(equity, index=idx)
    rp = pd.Series(dret, index=idx)
    return {
        "equity": eq, "ret": rp,
        "cash_ret": pd.Series(cash_ret, index=idx),
        "financing": pd.Series(fin_pnl, index=idx),
        "leverage": pd.Series(levg, index=idx),
        "rebalance": pd.DataFrame({"w_target": wt_tar, "w_drift": wt_dr,
                                   "turnover": turn, "cost": cost_l}, index=rb_dt),
        "regt_breach_months": len(breach_months),
    }


def active_weights(bt: dict, wsched: pd.Series) -> pd.Series:
    """Reconstruct the start-of-day active (drifted) weight w_t such that the daily
    excess return is w_t·x_Q − max(w_t−1,0)·spread. Shared by the D3 Sharpe
    decomposition and the G4 return-gap decomposition so they agree by
    construction."""
    days = bt["ret"].index
    lev, reb = bt["leverage"], bt["rebalance"]
    warm = wsched.loc[wsched.index < days[0]]
    w = pd.Series(index=days, dtype=float)
    w.iloc[0] = float(warm.iloc[-1])
    for i in range(1, len(days)):
        p = days[i - 1]
        w.iloc[i] = float(reb.loc[p, "w_target"]) if p in reb.index else float(lev.loc[p])
    return w


def metrics(bt: dict, cfg: dict) -> dict:
    fz = cfg["frozen"]
    rp, eq = bt["ret"], bt["equity"]
    roll = rp.rolling(21).std() * np.sqrt(ANNUAL)
    excess = rp - bt["cash_ret"]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1.0
    rb = bt["rebalance"]
    total_cost = rb["cost"].sum()
    return {
        "final_equity": float(eq.iloc[-1]),
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
        "realized_vol": float(rp.std() * np.sqrt(ANNUAL)),
        "vol_target_gap": float(rp.std() * np.sqrt(ANNUAL) - fz["target_vol"]),
        "mad_roll_vol_from_target": float((roll - fz["target_vol"]).abs().mean()),
        "vol_of_vol": float(roll.std()),
        "sharpe_excess": float(excess.mean() / excess.std() * np.sqrt(ANNUAL)),
        "max_drawdown": float(dd.min()),
        "skew": float(rp.skew()), "kurtosis": float(rp.kurt()),
        "avg_annual_turnover": float(rb["turnover"].sum() / yrs),
        "cost_drag_annual": float(total_cost / bt["equity"].iloc[0] / yrs),
        "pct_months_at_cap": float((rb["w_target"] >= fz["leverage_cap"] - 1e-9).mean()),
        "mean_leverage": float(rb["w_target"].mean()),
        "financing_total": float(bt["financing"].sum()),
        "regt_breach_months": int(bt["regt_breach_months"]),
    }

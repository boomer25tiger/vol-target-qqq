#!/usr/bin/env python3
"""
Layer 2 part 1 (SPEC.md Section 10.2): target adherence (primary), descriptive
risk/return, return-gap decomposition vs buy-and-hold, and subperiods. Sharpe is
secondary to adherence. No Ledoit-Wolf, no deflated Sharpe (Phase H).

Writes outputs/tables/{layer2_adherence,layer2_riskreturn,return_gap_decomposition,
layer2_subperiods}.csv.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.backtest.engine import (run_backtest, sizing_weight, metrics,  # noqa: E402
                                    active_weights, ANNUAL, DAYCOUNT)

TAB = os.path.join(repo_root(), "outputs", "tables")
TARGET = 0.20
SUBPERIODS = [("2000-2002", "2000-01-01", "2002-12-31"),
              ("2008-2009", "2008-01-01", "2009-12-31"),
              ("2010-2019", "2010-01-01", "2019-12-31"),
              ("2020", "2020-01-01", "2020-12-31"),
              ("2022", "2022-01-01", "2022-12-31")]


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def schedules(cfg, v, qqq_ret):
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    sc = {c: pd.Series(sizing_weight(v[c].to_numpy(), cfg), index=v.index) for c in v.columns}
    sc["bench_buy_hold"] = pd.Series(1.0, index=v.index)
    k = TARGET / (qqq_ret.loc[eval_start:].std() * np.sqrt(ANNUAL))
    sc["bench_const_lev"] = pd.Series(float(k), index=v.index)
    rv_exp = load_panel()["rv_daily"].expanding().mean().reindex(v.index, method="ffill")
    sc["bench_uncond_vol"] = pd.Series(sizing_weight(rv_exp.to_numpy(), cfg), index=v.index)
    return sc


def blocks(x, w=21):
    """Non-overlapping w-day blocks of a daily series -> list of arrays."""
    n = len(x) // w
    return [x[i * w:(i + 1) * w] for i in range(n)]


def adherence(ret):
    rv_full = ret.std() * np.sqrt(ANNUAL)
    roll = ret.rolling(21).std() * np.sqrt(ANNUAL)
    mad_overlap = float((roll - TARGET).abs().mean())
    bvol = np.array([b.std(ddof=1) * np.sqrt(ANNUAL) for b in blocks(ret.values)])
    mad_nonoverlap = float(np.abs(bvol - TARGET).mean())
    vov = float(bvol.std(ddof=1))                     # Harvey-style: std of realized vol
    within10 = float((np.abs(bvol - TARGET) <= 0.10 * TARGET).mean())
    within25 = float((np.abs(bvol - TARGET) <= 0.25 * TARGET).mean())
    return dict(realized_vol=float(rv_full), vol_target_gap=float(rv_full - TARGET),
                mad_roll_overlap=mad_overlap, mad_nonoverlap=mad_nonoverlap,
                vol_of_vol=vov, frac_within_10pct=within10, frac_within_25pct=within25,
                n_blocks=len(bvol))


def riskreturn(bt, cfg):
    ret, cash, eq = bt["ret"], bt["cash_ret"], bt["equity"]
    excess = ret - cash
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    bret = np.array([np.prod(1 + b) - 1 for b in blocks(ret.values)])
    rb = bt["rebalance"]; fin = bt["financing"]
    return dict(
        ann_mean_excess=float(excess.mean() * ANNUAL), ann_vol=float(ret.std() * np.sqrt(ANNUAL)),
        sharpe_excess=float(excess.mean() / excess.std() * np.sqrt(ANNUAL)),
        max_drawdown=float(dd.min()), cagr=float(cagr),
        calmar=float(cagr / abs(dd.min())), skew=float(ret.skew()), excess_kurtosis=float(ret.kurt()),
        monthly_q05=float(np.quantile(bret, 0.05)), monthly_q01=float(np.quantile(bret, 0.01)),
        mean_weight=float(rb["w_target"].mean()),
        w_q05=float(rb["w_target"].quantile(0.05)), w_q50=float(rb["w_target"].median()),
        w_q95=float(rb["w_target"].quantile(0.95)),
        pct_months_at_cap=float((rb["w_target"] >= cfg["frozen"]["leverage_cap"] - 1e-9).mean()),
        pct_months_at_zero=float((rb["w_target"] <= 1e-9).mean()),
        mean_monthly_turnover=float(rb["turnover"].mean()),
        cost_drag_annual=float(rb["cost"].sum() / eq.iloc[0] / yrs),
        cash_credit_usd=float(fin[fin > 0].sum()), borrow_paid_usd=float(fin[fin < 0].sum()),
        days_underwater_total=int((eq < eq.cummax() - 1e-9).sum()),
        days_underwater_longest=_longest_underwater(eq),
    )


def _longest_underwater(eq):
    """Longest consecutive run of trading days below a prior equity peak (M4)."""
    below = (eq < eq.cummax() - 1e-9).to_numpy()
    longest = cur = 0
    for x in below:
        cur = cur + 1 if x else 0
        if cur > longest:
            longest = cur
    return int(longest)


def return_gap(name, bt, ws, qqq_ret, cfg, bh_excess_ann):
    spread = cfg["frozen"]["financing"]["borrow_spread"]
    w = active_weights(bt, ws)
    days = w.index
    xQ = qqq_ret.reindex(days) - bt["cash_ret"]
    dc = pd.Series(days, index=days).diff().dt.days.fillna(3.0).astype(float)
    drag = np.maximum(w - 1.0, 0.0) * spread * dc / DAYCOUNT
    gross = w * xQ - drag
    net = bt["ret"] - bt["cash_ret"]
    cost = (gross - net)                                # turnover cost (E2: in the return)
    market = (w.mean() - 1.0) * xQ.mean() * ANNUAL
    voltiming = np.cov(w.values, xQ.values, ddof=0)[0, 1] * ANNUAL
    financing = -spread * (np.maximum(w - 1.0, 0.0) * dc / DAYCOUNT).mean() * ANNUAL
    txn = -cost.mean() * ANNUAL
    gap = net.mean() * ANNUAL - bh_excess_ann
    recon = market + voltiming + financing + txn
    return dict(strategy=name, gap_ann=float(gap), market_exposure=float(market),
                vol_timing=float(voltiming), financing=float(financing),
                transaction_cost=float(txn), residual=float(gap - recon))


def main():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = _raw("qqq_daily", "close").pct_change().dropna()
    dff = _raw("dff", "dff")
    sc = schedules(cfg, v, qqq_ret)

    bts = {n: run_backtest(ws, qqq_ret, dff, cfg) for n, ws in sc.items()}
    bh_excess_ann = float((bts["bench_buy_hold"]["ret"] - bts["bench_buy_hold"]["cash_ret"]).mean() * ANNUAL)

    # ---- G2 adherence ----
    adh = pd.DataFrame({n: adherence(bt["ret"]) for n, bt in bts.items()}).T
    adh = adh.sort_values("mad_nonoverlap")
    adh.to_csv(os.path.join(TAB, "layer2_adherence.csv"))
    print("=" * 96); print("G2 TARGET ADHERENCE (ranked by non-overlapping MAD from 20%; PRIMARY metric)"); print("=" * 96)
    print(f"  {'strategy':17s} {'realvol':>8} {'MADnov':>7} {'MADroll':>8} {'vov':>6} "
          f"{'w/in10%':>8} {'w/in25%':>8}")
    for n, r in adh.iterrows():
        print(f"  {n:17s} {r['realized_vol']:8.3f} {r['mad_nonoverlap']:7.4f} {r['mad_roll_overlap']:8.4f} "
              f"{r['vol_of_vol']:6.3f} {r['frac_within_10pct']:8.1%} {r['frac_within_25pct']:8.1%}")
    print("\n  vol-of-vol = std of annualized realized vol over non-overlapping 21-day blocks")
    print("  (matches Harvey et al. 2018 'std of realized volatility'; QQQ level exceeds their 4.6/1.8% as QQQ is more volatile)")

    # ---- G3 risk/return ----
    rr = pd.DataFrame({n: riskreturn(bt, cfg) for n, bt in bts.items()}).T
    rr.to_csv(os.path.join(TAB, "layer2_riskreturn.csv"))
    print("\n" + "=" * 96); print("G3 RISK/RETURN (descriptive)"); print("=" * 96)
    show = ["ann_mean_excess", "ann_vol", "sharpe_excess", "max_drawdown", "calmar",
            "skew", "excess_kurtosis", "monthly_q05", "monthly_q01", "mean_weight",
            "pct_months_at_cap", "pct_months_at_zero", "mean_monthly_turnover", "cost_drag_annual"]
    print(rr[show].round(4).to_string())
    print("\n  financing ($ on $100k): " + ", ".join(
        f"{n.split('_')[-1] if n.startswith('bench') else n}={rr.loc[n,'cash_credit_usd']:.0f}/{rr.loc[n,'borrow_paid_usd']:.0f}"
        for n in ["gjr_skewt", "ewma", "rv", "bench_buy_hold", "bench_const_lev"]) + "  (credit/borrow)")

    # ---- G4 return-gap decomposition ----
    gaps = pd.DataFrame([return_gap(n, bts[n], sc[n], qqq_ret, cfg, bh_excess_ann)
                         for n in sc if n != "bench_buy_hold"]).set_index("strategy")
    gaps.to_csv(os.path.join(TAB, "return_gap_decomposition.csv"))
    print("\n" + "=" * 96); print("G4 RETURN-GAP vs BUY-AND-HOLD (annualized; 4 components + residual)"); print("=" * 96)
    print(gaps[["gap_ann", "market_exposure", "vol_timing", "financing", "transaction_cost", "residual"]].round(5).to_string())
    print(f"\n  max |reconstruction residual|: {gaps['residual'].abs().max():.2e}")

    # ---- G5 subperiods ----
    subs = []
    for lab, a, b in SUBPERIODS:
        for n, bt in bts.items():
            r = bt["ret"].loc[a:b]; cash = bt["cash_ret"].loc[a:b]
            if len(r) < 21:
                continue
            ex = r - cash
            bvol = np.array([x.std(ddof=1) * np.sqrt(ANNUAL) for x in blocks(r.values)])
            subs.append({"subperiod": lab, "strategy": n,
                         "n_months": len(pd.period_range(a, min(b, str(r.index[-1].date())), freq="M")),
                         "realized_vol": float(r.std() * np.sqrt(ANNUAL)),
                         "mad_nonoverlap": float(np.abs(bvol - TARGET).mean()) if len(bvol) else np.nan,
                         "sharpe_excess": float(ex.mean() / ex.std() * np.sqrt(ANNUAL))})
    subdf = pd.DataFrame(subs)
    subdf.to_csv(os.path.join(TAB, "layer2_subperiods.csv"), index=False)
    print("\n" + "=" * 96); print("G5 SUBPERIODS (adherence + Sharpe; NO significance tests - samples too short)"); print("=" * 96)
    for lab, _, _ in SUBPERIODS:
        s = subdf[subdf.subperiod == lab]
        nm = int(s["n_months"].iloc[0]) if len(s) else 0
        best = s.loc[s["mad_nonoverlap"].idxmin(), "strategy"] if len(s) else "-"
        print(f"  {lab:10s} ({nm} months): best adherence = {best}; "
              f"buy&hold vol {s[s.strategy=='bench_buy_hold']['realized_vol'].iloc[0]:.2f}, "
              f"gjr vol {s[s.strategy=='gjr_skewt']['realized_vol'].iloc[0]:.2f} "
              f"Sharpe {s[s.strategy=='gjr_skewt']['sharpe_excess'].iloc[0]:+.2f}")
    print(f"\ntables -> {TAB}/{{layer2_adherence,layer2_riskreturn,return_gap_decomposition,layer2_subperiods}}.csv")


if __name__ == "__main__":
    main()

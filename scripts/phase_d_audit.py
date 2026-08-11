#!/usr/bin/env python3
"""
Phase D diagnostics (report only; reads model/forecast/backtest code, no edits).
D1 RFSV retransformation audit; D2 where H stops mattering; D3 Sharpe
decomposition + financing sign. Writes three CSVs under outputs/tables/ and
figures/h_stage_spread.png. No network.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.models.rebalance import rebalance_dates
from volteq.models.direct_h import har_fit_forecast, rfsv_fit_forecast
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL, DAYCOUNT

PROC = os.path.join(repo_root(), "data", "processed")
TAB = os.path.join(repo_root(), "outputs", "tables")
FIG = os.path.join(repo_root(), "figures")
H = 21


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def _ctx():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(PROC, "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel(); rv = panel["rv_daily"]
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], H)
    dates = rebalance_dates(cfg, data_end=panel.index.max(), include_warmup=True)
    qqq_ret = _raw("qqq_daily", "close").pct_change().dropna()
    dff = _raw("dff", "dff")
    return cfg, v, rv, fwd, dates, qqq_ret, dff


# ---------------------------------------------------------------- D1
def d1(cfg, v, rv, fwd, dates):
    print("\n" + "=" * 90); print("D1  RFSV RETRANSFORMATION AUDIT"); print("=" * 90)
    print("exp(.) argument in rfsv_fit_forecast: 'ehat + 0.5*s2', s2 = 0.5*nu2*Delta^(2H)")
    print("  file src/volteq/models/direct_h.py lines 130-133; X=log(rv_daily)=log VARIANCE,")
    print("  so s2 is the conditional variance of LOG VARIANCE, units (log variance)^2.\n")
    bt_metrics = pd.read_csv(os.path.join(TAB, "backtest_metrics.csv"), index_col=0)

    variants = {"rfsv": None, "rfsv_h002": 0.02, "rfsv_h005": 0.05,
                "rfsv_h010": 0.10, "rfsv_h015": 0.15}
    rows = []
    fwd_at = fwd.reindex(dates)
    for name, Hf in variants.items():
        s2m, corr, v21, v0 = [], [], [], []
        for t in dates:
            r = rv.loc[:t].to_numpy()
            f = rfsv_fit_forecast(r, H, H_fixed=Hf)
            s2 = 0.5 * f["nu2"] * np.arange(1, H + 1) ** (2.0 * f["H"] if Hf is None else 2.0 * Hf)
            s2m.append(s2.mean()); corr.append(1 + f["retransform_pct"])
            v21.append(f["V21"]); v0.append(f["V21"] / (1 + f["retransform_pct"]))
        v21 = pd.Series(v21, index=dates); v0 = pd.Series(v0, index=dates)
        j = pd.concat([np.log(fwd_at).rename("y"), np.log(v21).rename("f")], axis=1).dropna()
        emp_var = float((j["y"] - j["f"]).var(ddof=1))       # empirical OOS log fc-error var
        j0 = pd.concat([np.log(fwd_at).rename("y"), np.log(v0).rename("f0")], axis=1).dropna()
        emp_var0 = float((j0["y"] - j0["f0"]).var(ddof=1))
        rr = fwd_at.reindex(v21.index)
        rows.append({
            "model": name, "s2_hat_mean": float(np.mean(s2m)),
            "s_hat": float(np.sqrt(np.mean(s2m))),
            "corr_factor_mean": float(np.mean(corr)), "corr_factor_sd": float(np.std(corr)),
            "corr_pct_mean": float(np.mean(corr) - 1),
            "emp_logfcerr_var_corrected": emp_var,
            "emp_logfcerr_var_uncorr": emp_var0,
            "meanVhat_over_meanRV_corrected": float(v21.mean() / rr.mean()),
            "meanVhat_over_meanRV_uncorr": float(v0.mean() / rr.mean()),
            "bt_realized_vol": float(bt_metrics.loc[name, "realized_vol"]),
            "bt_mean_leverage": float(bt_metrics.loc[name, "mean_leverage"]),
        })
    # HAR control
    s2h, corrh, v21h, v0h = [], [], [], []
    for t in dates:
        r = rv.loc[:t].to_numpy(); f = har_fit_forecast(r, H)
        s2h.append(f["sigma2_resid"]); corrh.append(1 + f["retransform_pct"])
        v21h.append(f["V21"]); v0h.append(f["V21"] / (1 + f["retransform_pct"]))
    v21h = pd.Series(v21h, index=dates); v0h = pd.Series(v0h, index=dates)
    jh = pd.concat([np.log(fwd_at).rename("y"), np.log(v21h).rename("f")], axis=1).dropna()
    rows.append({
        "model": "har", "s2_hat_mean": float(np.mean(s2h)), "s_hat": float(np.sqrt(np.mean(s2h))),
        "corr_factor_mean": float(np.mean(corrh)), "corr_factor_sd": float(np.std(corrh)),
        "corr_pct_mean": float(np.mean(corrh) - 1),
        "emp_logfcerr_var_corrected": float((jh["y"] - jh["f"]).var(ddof=1)),
        "emp_logfcerr_var_uncorr": np.nan,
        "meanVhat_over_meanRV_corrected": float(v21h.mean() / fwd_at.reindex(v21h.index).mean()),
        "meanVhat_over_meanRV_uncorr": float(v0h.mean() / fwd_at.reindex(v0h.index).mean()),
        "bt_realized_vol": float(bt_metrics.loc["har", "realized_vol"]),
        "bt_mean_leverage": float(bt_metrics.loc["har", "mean_leverage"]),
    })
    # gjr level reference (no retransformation)
    gj = v["gjr_skewt"]
    rows.append({"model": "gjr_skewt", "s2_hat_mean": np.nan, "s_hat": np.nan,
                 "corr_factor_mean": np.nan, "corr_factor_sd": np.nan, "corr_pct_mean": np.nan,
                 "emp_logfcerr_var_corrected": float((np.log(fwd_at) - np.log(gj.reindex(dates))).dropna().var(ddof=1)),
                 "emp_logfcerr_var_uncorr": np.nan,
                 "meanVhat_over_meanRV_corrected": float(gj.reindex(dates).mean() / fwd_at.mean()),
                 "meanVhat_over_meanRV_uncorr": np.nan,
                 "bt_realized_vol": float(bt_metrics.loc["gjr_skewt", "realized_vol"]),
                 "bt_mean_leverage": float(bt_metrics.loc["gjr_skewt", "mean_leverage"])})
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(os.path.join(TAB, "rfsv_retransform_audit.csv"))
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.round(4).to_string())
    return df


# ---------------------------------------------------------------- D2
def d2(cfg, rv, fwd, dates, qqq_ret, dff):
    print("\n" + "=" * 90); print("D2  WHERE H STOPS MATTERING (extended grid + Brownian anchor)"); print("=" * 90)
    grid = {"est": None, "0.02": 0.02, "0.05": 0.05, "0.10": 0.10,
            "0.15": 0.15, "0.35": 0.35, "0.50": 0.50}
    V = {k: [] for k in grid}
    for t in dates:
        r = rv.loc[:t].to_numpy()
        for k, Hf in grid.items():
            V[k].append(rfsv_fit_forecast(r, H, H_fixed=Hf)["V21"])
    V = pd.DataFrame(V, index=dates)

    s1 = V
    s2 = np.sqrt(ANNUAL * V)
    s3 = cfg["frozen"]["target_vol"] / s2
    s4 = s3.clip(cfg["frozen"]["leverage_floor"], cfg["frozen"]["leverage_cap"])

    # stage 5: trailing 21d realized portfolio vol per variant
    roll = {}
    clip_hi, clip_lo = {}, {}
    for k in grid:
        ws = pd.Series(sizing_weight(V[k].to_numpy(), cfg), index=V.index)
        bt = run_backtest(ws, qqq_ret, dff, cfg)
        roll[k] = bt["ret"].rolling(21).std() * np.sqrt(ANNUAL)
        clip_hi[k] = float((ws >= cfg["frozen"]["leverage_cap"] - 1e-9).mean())
        clip_lo[k] = float((ws <= cfg["frozen"]["leverage_floor"] + 1e-9).mean())
    s5 = pd.DataFrame(roll)

    def spread(df):
        return ((df.max(axis=1) - df.min(axis=1)) / df.mean(axis=1)).dropna()

    stages = {"1_V": s1, "2_sigma": s2, "3_w_unclipped": s3, "4_w_clipped": s4, "5_realized_vol": s5}
    summ = {}
    for nm, df in stages.items():
        sp = spread(df)
        summ[nm] = {"mean_spread": float(sp.mean()), "p95_spread": float(sp.quantile(.95))}
    out = pd.DataFrame(summ).T
    out.loc["4_w_clipped", "ratio_to_stage3"] = summ["4_w_clipped"]["mean_spread"] / summ["3_w_unclipped"]["mean_spread"]
    out.to_csv(os.path.join(TAB, "h_stage_spread.csv"))
    print(out.round(4).to_string())
    print("\nclip binding fraction per variant (at cap 2.0 / floor 0.0):")
    for k in grid:
        print(f"  {k:>4}: hi {clip_hi[k]:.1%}  lo {clip_lo[k]:.1%}")

    # MZ intercept/slope per variant (extended grid)
    print("\nMincer-Zarnowitz (realized ~ V_hat, HAC lag 5) per H variant:")
    fwd_at = fwd.reindex(dates)
    for k in grid:
        j = pd.concat([fwd_at.rename("y"), V[k].rename("f")], axis=1).dropna()
        res = sm.OLS(j["y"], sm.add_constant(j["f"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        print(f"  H={k:>4}: a={res.params.iloc[0]:+.2e}  b={res.params.iloc[1]:.3f}  R2={res.rsquared:.3f}")

    # figure: five stage spreads over time
    fig, ax = plt.subplots(figsize=(12, 5))
    for nm, df in stages.items():
        ax.plot(spread(df).index, spread(df).values, lw=0.9, label=nm)
    ax.set_yscale("log"); ax.set_title("RFSV H-grid cross-variant spread (max-min)/mean by pipeline stage")
    ax.set_ylabel("relative spread (log)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "h_stage_spread.png"), dpi=130); plt.close(fig)
    print(f"\nfigure -> {FIG}/h_stage_spread.png")
    return out


# ---------------------------------------------------------------- D3
def d3(cfg, v, qqq_ret, dff):
    print("\n" + "=" * 90); print("D3  SHARPE DECOMPOSITION + FINANCING SIGN"); print("=" * 90)
    print("Sharpe as implemented: engine.metrics 'sharpe_excess' = excess.mean()/excess.std()*sqrt(252),")
    print("  excess = ret - cash_ret, cash_ret = DFF*days/360 -> numerator is EXCESS OF DFF.\n")
    spread = float(cfg["frozen"]["financing"]["borrow_spread"])
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])

    scheds = {c: pd.Series(sizing_weight(v[c].to_numpy(), cfg), index=v.index) for c in v.columns}
    scheds["bench_buy_hold"] = pd.Series(1.0, index=v.index)
    k = cfg["frozen"]["target_vol"] / (qqq_ret.loc[eval_start:].std() * np.sqrt(ANNUAL))
    scheds["bench_const_lev"] = pd.Series(float(k), index=v.index)
    panel = load_panel(); rv_exp = panel["rv_daily"].expanding().mean().reindex(v.index, method="ffill")
    scheds["bench_uncond_vol"] = pd.Series(sizing_weight(rv_exp.to_numpy(), cfg), index=v.index)

    rows = []
    for name, ws in scheds.items():
        bt = run_backtest(ws, qqq_ret, dff, cfg)
        rp, cash = bt["ret"], bt["cash_ret"]
        excess = rp - cash
        yrs = (rp.index[-1] - rp.index[0]).days / 365.25
        # reconstruct start-of-day active weight
        days = rp.index
        lev = bt["leverage"]; reb = bt["rebalance"]
        w0 = float(ws.loc[ws.index < days[0]].iloc[-1])
        w = pd.Series(index=days, dtype=float); w.iloc[0] = w0
        for i in range(1, len(days)):
            p = days[i - 1]
            w.iloc[i] = float(reb.loc[p, "w_target"]) if p in reb.index else float(lev.loc[p])
        xQ = qqq_ret.reindex(days) - cash
        dc = pd.Series(days, index=days).diff().dt.days.fillna(3).astype(float)
        drag = np.maximum(w - 1.0, 0.0) * spread * dc / DAYCOUNT
        gross_excess = w * xQ - drag
        wbar = float(w.mean())
        term_wbar = wbar * float(xQ.mean())
        term_cov = float(np.cov(w.values, xQ.values, ddof=0)[0, 1])
        term_drag = float(drag.mean())
        recon = term_wbar + term_cov - term_drag
        resid = float(gross_excess.mean() - recon)
        rows.append({
            "strategy": name,
            "sharpe_excess": float(excess.mean() / excess.std() * np.sqrt(ANNUAL)),
            "sharpe_total": float(rp.mean() / rp.std() * np.sqrt(ANNUAL)),
            "w_bar": wbar,
            "term_wbar_ExQ_ann": term_wbar * ANNUAL,
            "term_cov_ann": term_cov * ANNUAL,
            "term_financing_drag_ann": term_drag * ANNUAL,
            "mean_gross_excess_ann": float(gross_excess.mean()) * ANNUAL,
            "recon_residual": resid,
        })
    df = pd.DataFrame(rows).set_index("strategy")
    df.to_csv(os.path.join(TAB, "sharpe_decomposition.csv"))
    print(df[["sharpe_excess", "sharpe_total", "w_bar", "term_wbar_ExQ_ann",
              "term_cov_ann", "term_financing_drag_ann", "recon_residual"]].round(5).to_string())
    print(f"\nmax |reconstruction residual| across strategies: {df['recon_residual'].abs().max():.2e}")
    return df


def main():
    cfg, v, rv, fwd, dates, qqq_ret, dff = _ctx()
    d1(cfg, v, rv, fwd, dates)
    d2(cfg, rv, fwd, dates, qqq_ret, dff)
    d3(cfg, v, qqq_ret, dff)


if __name__ == "__main__":
    main()

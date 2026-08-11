#!/usr/bin/env python3
"""
Phase H analysis (explanatory/inference; changes no Phase A-G number).
H1 loss-function divergence, H3 Sharpe inference, H4 cross-layer ranks.
Writes outputs/tables/{loss_concentration,sharpe_inference,cross_layer_ranks}.csv.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.models.rebalance import rebalance_dates
from volteq.backtest.engine import run_backtest, sizing_weight, metrics, ANNUAL
from volteq.eval.layer1 import qlike, mse
from volteq.eval.sharpe import ledoit_wolf_sharpe_test, deflated_sharpe, nw_auto_lag

TAB = os.path.join(repo_root(), "outputs", "tables")
H = 21
PRIMARY8 = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv", "trailing_rv21"]
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def _in_crises(idx):
    m = np.zeros(len(idx), bool)
    for a, b in CRISES:
        m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m


def secondary_df(cfg):
    cache = os.path.join(repo_root(), "data", "processed", "secondary_daily_forecasts.parquet")
    if os.path.exists(cache):
        d = pd.read_parquet(cache); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")
    import layer1_secondary as ls
    fc, rv = ls.daily_forecasts(cfg)
    eg = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "egarch_daily_secondary.parquet"))
    eg["date"] = pd.to_datetime(eg["date"])
    fc = fc.join(eg.set_index("date")["egarch_daily"].rename("egarch_skewt"))
    fwd = forward_realized_variance_avg(rv, H)
    df = pd.concat([fwd.rename("RV"), fc], axis=1).dropna()
    df.reset_index().rename(columns={"index": "date"}).to_parquet(cache, index=False)
    return df


def primary_df(cfg):
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    v = v.copy(); v["trailing_rv21"] = panel["yz_21"].reindex(v.index)
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], H)
    df = pd.concat([fwd.reindex(v.index).rename("RV"), v[PRIMARY8]], axis=1)
    return df.loc[df.index >= pd.Timestamp(cfg["frozen"]["eval_start"])].dropna()


# ---------------------------------------------------------------- H1
def h1(cfg):
    print("=" * 92); print("H1 LOSS-FUNCTION DIVERGENCE"); print("=" * 92)
    rows = []
    for sample, df in [("primary", primary_df(cfg)), ("secondary", secondary_df(cfg)[["RV"] + PRIMARY8])]:
        rv = df["RV"].values
        crisis = _in_crises(df.index)
        for m in PRIMARY8:
            ms = mse(rv, df[m].values); ql = qlike(rv, df[m].values)
            k = max(1, int(np.ceil(0.01 * len(ms))))
            rows.append({"sample": sample, "model": m,
                         "mse_full": float(ms.mean()),
                         "mse_excl_crisis": float(ms[~crisis].mean()),
                         "mse_top1pct_share": float(np.sort(ms)[-k:].sum() / ms.sum()),
                         "qlike_full": float(ql.mean()),
                         "qlike_top1pct_share": float(np.sort(ql)[-k:].sum() / ql.sum())})
    t = pd.DataFrame(rows)
    for s in ["primary", "secondary"]:
        sub = t[t["sample"] == s].set_index("model")
        sub["mse_rank_full"] = sub["mse_full"].rank().astype(int)
        sub["mse_rank_excl"] = sub["mse_excl_crisis"].rank().astype(int)
        t.loc[t["sample"] == s, "mse_rank_full"] = sub["mse_rank_full"].values
        t.loc[t["sample"] == s, "mse_rank_excl"] = sub["mse_rank_excl"].values
    t.to_csv(os.path.join(TAB, "loss_concentration.csv"), index=False)
    for s in ["primary", "secondary"]:
        sub = t[t["sample"] == s].set_index("model")
        rho = spearmanr(sub["mse_rank_full"], sub["mse_rank_excl"]).correlation
        reord = int((sub["mse_rank_full"] != sub["mse_rank_excl"]).sum())
        print(f"\n  [{s}] MSE rank full vs crisis-excluded (Spearman {rho:.3f}, {reord}/8 change):")
        print(sub[["mse_rank_full", "mse_rank_excl", "mse_top1pct_share", "qlike_top1pct_share"]].round(3).to_string())


# ---------------------------------------------------------------- H3
def build_strategies(cfg, v, qqq_ret, panel):
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    sc = {c: pd.Series(sizing_weight(v[c].to_numpy(), cfg), index=v.index) for c in v.columns}
    sc["trailing_rv21"] = pd.Series(sizing_weight(panel["yz_21"].reindex(v.index).to_numpy(), cfg), index=v.index)
    sc["bench_buy_hold"] = pd.Series(1.0, index=v.index)
    sc["bench_const_lev"] = pd.Series(float(cfg["frozen"]["target_vol"] /
                                           (qqq_ret.loc[eval_start:].std() * np.sqrt(ANNUAL))), index=v.index)
    rv_exp = panel["rv_daily"].expanding().mean().reindex(v.index, method="ffill")
    sc["bench_uncond_vol"] = pd.Series(sizing_weight(rv_exp.to_numpy(), cfg), index=v.index)
    return sc


def h3(cfg, sc, qqq_ret, dff):
    print("\n" + "=" * 92); print("H3 SHARPE INFERENCE"); print("=" * 92)
    bts = {n: run_backtest(ws, qqq_ret, dff, cfg) for n, ws in sc.items()}
    ex = {n: (bt["ret"] - bt["cash_ret"]) for n, bt in bts.items()}
    names = list(sc.keys())
    n_obs = len(next(iter(ex.values())))
    lag = nw_auto_lag(n_obs)
    print(f"  daily excess returns, n={n_obs}; Ledoit-Wolf HAC (Newey-West Bartlett) lag {lag} "
          f"= floor(4(n/100)^(2/9))")

    # DSR trial count: forecast-model configs searched (14 live + 5 discarded rfsv c=0.5)
    model_cols = [c for c in sc if not c.startswith("bench") and c != "trailing_rv21"]
    N_TRIALS = len(model_cols) + 5
    sr_daily = {n: ex[n].mean() / ex[n].std(ddof=1) for n in model_cols}
    sr_var = float(np.var(list(sr_daily.values()), ddof=1))
    print(f"  Deflated Sharpe: N={N_TRIALS} trials (= {len(model_cols)} live model configs "
          f"+ 5 discarded RFSV c=0.5 variants, per the pre-registered no-tuning rule; passive benchmarks excluded "
          f"from N as reference rungs). Var(SR) across model configs = {sr_var:.4e}")

    rows = []
    for n in names:
        m = metrics(bts[n], cfg)
        d = deflated_sharpe(ex[n].values, N_TRIALS, sr_var)
        rows.append({"strategy": n, "sharpe_excess_ann": m["sharpe_excess"],
                     "realized_vol": m["realized_vol"], "sr_daily": d["sr_daily"],
                     "dsr": d["dsr"], "skew": d["skew"], "kurtosis": d["kurtosis"]})
    dsr_df = pd.DataFrame(rows).set_index("strategy")

    # LW pairwise p-value matrix
    P = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            P.loc[i, j] = 1.0 if i == j else ledoit_wolf_sharpe_test(ex[i].values, ex[j].values, lag)["p"]
    dsr_df.to_csv(os.path.join(TAB, "sharpe_inference.csv"))
    P.to_csv(os.path.join(TAB, "sharpe_inference_lw_pmatrix.csv"))
    print("\n  Sharpe (ann, excess) beside realized vol, and Deflated Sharpe:")
    print(dsr_df[["sharpe_excess_ann", "realized_vol", "dsr"]].round(4).sort_values("sharpe_excess_ann", ascending=False).to_string())
    # LW: which pairs' Sharpe difference is significant at 5%?
    sig = [(i, j, P.loc[i, j]) for ii, i in enumerate(names) for j in names[ii + 1:] if P.loc[i, j] < 0.05]
    print(f"\n  Ledoit-Wolf: pairwise Sharpe differences significant at 5%: {len(sig)}/{len(names)*(len(names)-1)//2}")
    for i, j, p in sorted(sig, key=lambda x: x[2])[:12]:
        print(f"    {i:16s} vs {j:16s} p={p:.3f}")
    return dsr_df, bts


# ---------------------------------------------------------------- H4
def h4(cfg, bts, qqq_ret):
    print("\n" + "=" * 92); print("H4 CROSS-LAYER RANK CORRELATION (8 primary; descriptive, n=8 no power)"); print("=" * 92)
    l1 = pd.read_csv(os.path.join(TAB, "layer1_losses.csv"), index_col=0)
    # L2 adherence + Sharpe for the 8 primary (rv/trailing_rv21 backtested separately)
    def mad_nonoverlap(ret):
        w = 21; n = len(ret) // w
        bvol = np.array([ret.values[i*w:(i+1)*w].std(ddof=1)*np.sqrt(ANNUAL) for i in range(n)])
        return float(np.abs(bvol - cfg["frozen"]["target_vol"]).mean())
    l2 = {}
    for m in PRIMARY8:
        bt = bts[m]
        ex = bt["ret"] - bt["cash_ret"]
        l2[m] = {"adherence_mad": mad_nonoverlap(bt["ret"]),
                 "sharpe": float(ex.mean() / ex.std() * np.sqrt(ANNUAL))}
    l2 = pd.DataFrame(l2).T
    tab = pd.DataFrame({"qlike": l1.loc[PRIMARY8, "qlike_mean"],
                        "adherence_mad": l2["adherence_mad"], "sharpe": l2["sharpe"]})
    tab["qlike_rank"] = tab["qlike"].rank().astype(int)
    tab["adherence_rank"] = tab["adherence_mad"].rank().astype(int)
    tab["sharpe_rank"] = tab["sharpe"].rank(ascending=False).astype(int)  # higher Sharpe = better = rank 1
    tab.to_csv(os.path.join(TAB, "cross_layer_ranks.csv"))
    rho_adh = spearmanr(tab["qlike_rank"], tab["adherence_rank"]).correlation
    rho_shp = spearmanr(tab["qlike_rank"], tab["sharpe_rank"]).correlation
    print(tab[["qlike_rank", "adherence_rank", "sharpe_rank"]].to_string())
    print(f"\n  Spearman(L1 QLIKE order, L2 adherence order) = {rho_adh:.3f}")
    print(f"  Spearman(L1 QLIKE order, L2 Sharpe order)    = {rho_shp:.3f}")
    print("  (n=8: essentially no power; coefficients summarize, do not establish.)")


def main():
    cfg = load_config()
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = _raw("qqq_daily", "close").pct_change().dropna()
    dff = _raw("dff", "dff"); panel = load_panel()
    h1(cfg)
    sc = build_strategies(cfg, v, qqq_ret, panel)
    _, bts = h3(cfg, sc, qqq_ret, dff)
    h4(cfg, bts, qqq_ret)
    print(f"\ntables -> {TAB}/{{loss_concentration,sharpe_inference,sharpe_inference_lw_pmatrix,cross_layer_ranks}}.csv")


if __name__ == "__main__":
    main()

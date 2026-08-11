#!/usr/bin/env python3
"""
Phase I1 - HAR crisis-window robustness read (no model changes; existing forecasts).
Recomputes, crisis-windows-excluded, on primary (monthly) and secondary (daily):
  har QLIKE rank vs full; QLIKE MCS over the 8 primary; har DM vs trailing_rv21;
and har adherence MAD over the 2010-2019 non-crisis subperiod.
Prints only.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.backtest.engine import run_backtest, sizing_weight, ANNUAL
from volteq.eval.layer1 import qlike, diebold_mariano, model_confidence_set, nw_auto_lag

PRIMARY8 = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv", "trailing_rv21"]
CRISES = [("2008-09-01", "2009-06-30"), ("2020-02-01", "2020-06-30")]
SEED = 20260806
SEC_LAG = 25   # secondary daily DM lag (overlap MA(20); SPEC >=25), matches dm_matrix_secondary


def _in_crises(idx):
    m = np.zeros(len(idx), bool)
    for a, b in CRISES:
        m |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    return m


def _raw(name, col):
    d = pd.read_parquet(os.path.join(repo_root(), "data", "raw", f"{name}.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date").sort_index()[col]


def primary_df(cfg):
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    v = v.copy(); v["trailing_rv21"] = panel["yz_21"].reindex(v.index)
    fwd = forward_realized_variance_avg(panel.loc[panel["source"] == "qqq", "rv_daily"], 21)
    df = pd.concat([fwd.reindex(v.index).rename("RV"), v[PRIMARY8]], axis=1)
    return df.loc[df.index >= pd.Timestamp(cfg["frozen"]["eval_start"])].dropna()


def secondary_df():
    d = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "secondary_daily_forecasts.parquet"))
    d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")[["RV"] + PRIMARY8]


def robustness(sample, df, dm_lag):
    print("\n" + "=" * 90); print(f"I1 [{sample}]  n={len(df)}  ({df.index.min().date()}..{df.index.max().date()})"); print("=" * 90)
    rv = df["RV"].values
    crisis = _in_crises(df.index)
    QL = {m: qlike(rv, df[m].values) for m in PRIMARY8}

    def ranks(mask):
        means = pd.Series({m: QL[m][mask].mean() for m in PRIMARY8})
        return means.rank().astype(int), means
    r_full, mean_full = ranks(np.ones(len(df), bool))
    r_excl, mean_excl = ranks(~crisis)
    print(f"  n excluded (crisis): {int(crisis.sum())};  retained: {int((~crisis).sum())}")
    print("  QLIKE rank among the 8 primary (1 = best):")
    tbl = pd.DataFrame({"qlike_full": mean_full, "rank_full": r_full,
                        "qlike_excl": mean_excl, "rank_excl": r_excl}).sort_values("rank_full")
    print(tbl.round(4).to_string())
    print(f"  >>> HAR QLIKE rank: full = {r_full['har']}   crisis-excluded = {r_excl['har']}")

    # MCS on crisis-excluded, 8 primary only, blocks {2,5,10}
    L = np.column_stack([QL[m][~crisis] for m in PRIMARY8])
    print(f"  QLIKE MCS (90%, B=10000, seed {SEED}) on crisis-excluded sample:")
    for blk in (2, 5, 10):
        res = model_confidence_set(L, PRIMARY8, alpha=0.10, B=10000, block=blk, seed=SEED)
        ret = res["retained"]
        mark = "  <-- headline" if blk == 5 else ""
        print(f"    block={blk:2d}: retained ({len(ret)}/8) {sorted(ret)}{mark}")
        if blk == 5:
            print(f"             elimination order: {res['elim_order']}")
            print(f"             har in retained set: {'har' in ret}")

    # DM har vs trailing_rv21 on crisis-excluded
    dm = diebold_mariano(QL["har"][~crisis], QL["trailing_rv21"][~crisis], lag=dm_lag)
    print(f"  DM  har vs trailing_rv21 (crisis-excluded, lag {dm['lag']}): "
          f"dm={dm['dm']:+.3f}  p={dm['p']:.4f}  {'*sig*' if dm['p']<0.05 else 'NOT sig'}  "
          f"(dm<0 => har lower QLIKE)")
    # full-sample DM for reference
    dmf = diebold_mariano(QL["har"], QL["trailing_rv21"], lag=dm_lag)
    print(f"      reference full-sample DM: dm={dmf['dm']:+.3f}  p={dmf['p']:.4f}")
    return r_full["har"], r_excl["har"], dm


def har_adherence_2010s(cfg):
    print("\n" + "=" * 90); print("I1  HAR adherence MAD over 2010-2019 (non-crisis subperiod)"); print("=" * 90)
    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    qqq_ret = _raw("qqq_daily", "close").pct_change().dropna()
    dff = _raw("dff", "dff")
    ws = pd.Series(sizing_weight(v["har"].to_numpy(), cfg), index=v.index)
    bt = run_backtest(ws, qqq_ret, dff, cfg)
    ret = bt["ret"].loc["2010-01-01":"2019-12-31"]
    w = 21; nb = len(ret) // w
    bvol = np.array([ret.values[i*w:(i+1)*w].std(ddof=1)*np.sqrt(ANNUAL) for i in range(nb)])
    mad = float(np.abs(bvol - cfg["frozen"]["target_vol"]).mean())
    print(f"  2010-2019: {len(ret)} days, {nb} non-overlapping 21-day blocks")
    print(f"  har mean realized vol = {bvol.mean():.4f};  MAD from 0.20 = {mad:.4f}")
    print(f"  (G5 full-window best-adherence in 2010-2019 was garch_normal; full-sample har MAD = 0.0493)")
    return mad


def main():
    cfg = load_config()
    robustness("primary-monthly", primary_df(cfg), dm_lag=None)   # None -> nw_auto_lag
    robustness("secondary-daily", secondary_df(), dm_lag=SEC_LAG)
    har_adherence_2010s(cfg)


if __name__ == "__main__":
    main()

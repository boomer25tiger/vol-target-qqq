#!/usr/bin/env python3
"""
Layer 1 forecast-accuracy evaluation (SPEC.md Section 10.1).

Target RV_{t+1:t+21} = forward 21-day mean of the daily realized-variance proxy
(overnight^2 + Rogers-Satchell, the summed daily Yang-Zhang components). Losses
QLIKE and MSE. Primary MCS over the 7 models + the trailing_rv21 benchmark;
grid variants and Gaussian QML rows form a SEPARATE sensitivity MCS. Non-
overlapping monthly sample. Writes outputs/tables/{layer1_losses,dm_matrix,
mcs_results}.csv. No network.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.rv.forward_target import forward_realized_variance_avg        # noqa: E402
from volteq.eval.layer1 import (qlike, mse, diebold_mariano, hac_var_mean,  # noqa: E402
                                nw_auto_lag, model_confidence_set)

TAB = os.path.join(repo_root(), "outputs", "tables")
PRIMARY = ["garch_skewt", "egarch_skewt", "gjr_skewt", "ewma", "rv", "har",
           "rfsv", "trailing_rv21"]
SENSITIVITY = ["rfsv_h002", "rfsv_h005", "rfsv_h010", "rfsv_h015",
               "garch_normal", "gjr_normal", "egarch_normal"]
BLOCKS = [2, 5, 10]


def build():
    cfg = load_config()
    h = int(cfg["frozen"]["forecast_horizon_days"])
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    seed = int(cfg["meta"]["random_seed"])

    v = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet"))
    v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date").sort_index()
    panel = load_panel()
    rv_qqq = panel.loc[panel["source"] == "qqq", "rv_daily"]
    fwd = forward_realized_variance_avg(rv_qqq, h)
    # trailing_rv21 benchmark = trailing 21-day Yang-Zhang (RV_t(21) used directly)
    v = v.copy()
    v["trailing_rv21"] = panel["yz_21"].reindex(v.index)

    cols = PRIMARY + SENSITIVITY
    df = pd.concat([fwd.reindex(v.index).rename("RV"), v[cols]], axis=1)
    df = df.loc[df.index >= eval_start].dropna()          # guard: >= eval_start, target present
    return cfg, h, seed, df, cols


def losses_table(df, cols):
    rows = []
    for c in cols:
        ql = qlike(df["RV"].values, df[c].values)
        ms = mse(df["RV"].values, df[c].values)
        n = len(ql)
        rows.append({
            "model": c, "min_forecast": float(df[c].min()),
            "qlike_mean": float(ql.mean()), "qlike_se": float(np.sqrt(hac_var_mean(ql, nw_auto_lag(n)))),
            "mse_mean": float(ms.mean()), "mse_se": float(np.sqrt(hac_var_mean(ms, nw_auto_lag(n)))),
        })
    t = pd.DataFrame(rows).set_index("model")
    t["qlike_rank"] = t["qlike_mean"].rank().astype(int)
    t["mse_rank"] = t["mse_mean"].rank().astype(int)
    return t


def dm_matrix(df, cols, loss_fn):
    n = len(df)
    L = {c: loss_fn(df["RV"].values, df[c].values) for c in cols}
    recs = []
    for i in cols:
        for j in cols:
            if i >= j:
                continue
            r = diebold_mariano(L[i], L[j])
            recs.append({"a": i, "b": j, "dm": r["dm"], "p": r["p"],
                         "lag": r["lag"], "acf1_diff": r["acf1"]})
    return pd.DataFrame(recs)


def run_mcs(df, cols, loss_fn, seed):
    L = np.column_stack([loss_fn(df["RV"].values, df[c].values) for c in cols])
    out = {}
    for blk in BLOCKS:
        out[blk] = model_confidence_set(L, cols, alpha=0.10, B=10000, block=blk, seed=seed)
    return out


def main():
    cfg, h, seed, df, cols = build()
    os.makedirs(TAB, exist_ok=True)
    n = len(df)
    print(f"Layer 1 | primary non-overlapping monthly | n={n} | "
          f"{df.index.min().date()}..{df.index.max().date()} | NW auto-lag {nw_auto_lag(n)}")
    print(f"positivity guard: all forecasts > 0 : {bool((df[cols] > 0).all().all())}")

    # ---- losses ----
    t = losses_table(df, cols)
    t.to_csv(os.path.join(TAB, "layer1_losses.csv"))
    print("\n" + "=" * 80); print("MEAN LOSS (SE), RANKS  [primary set in the first 8 rows]"); print("=" * 80)
    print(f"  {'model':14s} {'min_fc':>9} {'QLIKE':>9} {'(SE)':>8} {'rk':>3} {'MSE':>10} {'(SE)':>9} {'rk':>3}")
    for c in cols:
        r = t.loc[c]
        tag = "" if c in PRIMARY else "  [sens]"
        print(f"  {c:14s} {r['min_forecast']:9.2e} {r['qlike_mean']:9.4f} {r['qlike_se']:8.4f} "
              f"{int(r['qlike_rank']):3d} {r['mse_mean']:10.3e} {r['mse_se']:9.2e} {int(r['mse_rank']):3d}{tag}")

    # ranking agreement among the primary set
    tp = t.loc[PRIMARY]
    rho = tp["qlike_mean"].rank().corr(tp["mse_mean"].rank(), method="spearman")
    print(f"\n  QLIKE vs MSE rank correlation (primary 8): {rho:.3f}")
    print(f"  QLIKE best: {tp['qlike_mean'].idxmin()}   MSE best: {tp['mse_mean'].idxmin()}")

    # ---- DM matrices ----
    dm_q = dm_matrix(df, PRIMARY, qlike); dm_q["loss"] = "qlike"
    dm_m = dm_matrix(df, PRIMARY, mse); dm_m["loss"] = "mse"
    dmall = pd.concat([dm_q, dm_m], ignore_index=True)
    dmall.to_csv(os.path.join(TAB, "dm_matrix.csv"), index=False)
    print("\n" + "=" * 80); print("DIEBOLD-MARIANO (primary): pairs with p<0.10, per loss"); print("=" * 80)
    for loss, dm in [("QLIKE", dm_q), ("MSE", dm_m)]:
        sig = dm[dm["p"] < 0.10]
        print(f"  [{loss}] loss-diff acf1 range [{dm['acf1_diff'].min():+.2f},{dm['acf1_diff'].max():+.2f}], "
              f"lag {dm['lag'].iloc[0]}; significant pairs {len(sig)}/{len(dm)}:")
        for _, r in sig.iterrows():
            print(f"    {r['a']:13s} vs {r['b']:13s} DM {r['dm']:+6.2f} p {r['p']:.3f}")

    # ---- MCS ----
    print("\n" + "=" * 80); print(f"MODEL CONFIDENCE SET (90%, stationary bootstrap B=10000, seed {seed})"); print("=" * 80)
    mcs_rows = []
    for label, cset in [("primary", PRIMARY), ("sensitivity", PRIMARY + SENSITIVITY)]:
        for loss_name, loss_fn in [("qlike", qlike), ("mse", mse)]:
            res = run_mcs(df, cset, loss_fn, seed)
            for blk in BLOCKS:
                r = res[blk]
                for mname in cset:
                    mcs_rows.append({"set": label, "loss": loss_name, "block": blk,
                                     "model": mname, "mcs_p": r["mcs_p"][mname],
                                     "retained": mname in r["retained"]})
            # print block=5 as headline
            r5 = res[5]
            print(f"  [{label:11s} {loss_name}] block=5 retained ({len(r5['retained'])}/{len(cset)}): "
                  f"{', '.join(r5['retained'])}")
            # stability across blocks
            sets = {blk: set(res[blk]['retained']) for blk in BLOCKS}
            stable = sets[2] == sets[5] == sets[10]
            print(f"       membership stable across blocks {{2,5,10}}: {stable}"
                  + ("" if stable else f"  (b2={len(sets[2])}, b5={len(sets[5])}, b10={len(sets[10])})"))
    pd.DataFrame(mcs_rows).to_csv(os.path.join(TAB, "mcs_results.csv"), index=False)
    print(f"\ntables -> {TAB}/{{layer1_losses,dm_matrix,mcs_results}}.csv")


if __name__ == "__main__":
    main()

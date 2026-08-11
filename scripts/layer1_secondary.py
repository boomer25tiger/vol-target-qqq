#!/usr/bin/env python3
"""
Layer 1 SECONDARY sample (SPEC.md 10.1): overlapping daily forecasts.

Daily V_t(21) is produced by freezing each month's fitted model and rolling its
state forward one day at a time (no look-ahead: day d uses parameters from the
most recent rebalance r<=d and data through d only). egarch is omitted from the
daily set: its aggregation is a 10,000-path Monte Carlo that is not tractable to
re-run every trading day; it stays in the (complete) primary monthly sample.

Overlapping 21-day forecasts make the loss differential an MA(20) process, so the
Newey-West bandwidth is set to h-1 = 20 at minimum (SPEC asks >= 25); we use 25.
Appends the secondary rows to the console report; writes no new files.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root                          # noqa: E402
from volteq.rv.panel import load_panel                                    # noqa: E402
from volteq.rv.forward_target import forward_realized_variance_avg        # noqa: E402
from volteq.models.returns import build_garch_returns                     # noqa: E402
from volteq.models.direct_h import c_fbm, WEEKLY, MONTHLY                  # noqa: E402
from volteq.forecast.aggregate import v_closed_form                       # noqa: E402
from volteq.eval.layer1 import qlike, mse, diebold_mariano, model_confidence_set  # noqa: E402

MODELS = ["garch_skewt", "gjr_skewt", "ewma", "rv", "har", "rfsv", "trailing_rv21"]
NW_LAG = 25
H = 21
TAB = os.path.join(repo_root(), "outputs", "tables")


def _pp():
    p = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "param_paths.parquet"))
    p["date"] = pd.to_datetime(p["date"]); return p


def _asof(param_dates, d):
    i = param_dates.searchsorted(d, side="right") - 1
    return None if i < 0 else i


def daily_forecasts(cfg):
    h = int(cfg["frozen"]["forecast_horizon_days"])
    lam = float(next(m for m in cfg["models"] if m["id"] == "ewma")["lambda"])
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    panel = load_panel(); rv = panel["rv_daily"]; logrv = np.log(rv)
    rets = build_garch_returns(cfg)["ret"]
    pp = _pp()

    def frozen(model, dist):
        s = pp[(pp.model == model) & (pp.dist == dist)].set_index("date").sort_index()
        return s
    g = frozen("garch", "skewt"); j = frozen("gjr", "skewt")
    gd = g.index.values; jd = j.index.values

    days = rv.loc[eval_start:].index
    out = {m: [] for m in MODELS}
    idx = []
    # precompute garch-return resid arrays lazily per day is expensive; do vectorized recursion per day
    rv_v = rv.values; rv_idx = rv.index
    for d in days:
        idx.append(d)
        # --- rv, trailing_rv21 (no params) ---
        pos = rv_idx.get_loc(d)
        out["rv"].append(rv_v[pos - h + 1:pos + 1].mean())
        out["trailing_rv21"].append(panel["yz_21"].iloc[pos])
        # --- ewma (identity), daily recursion on returns through d ---
        r = rets.loc[:d].values * 100.0
        r2 = r ** 2
        x = np.empty_like(r2); x[0] = r2.mean(); x[1:] = (1 - lam) * r2[:-1]
        s2 = lfilter([1.0], [1.0, -lam], x)
        out["ewma"].append((lam * s2[-1] + (1 - lam) * r2[-1]) / 1e4)
        # --- garch / gjr: frozen params, VT recursion on returns through d ---
        resid = r - r.mean(); rsq = resid ** 2
        gi = _asof(gd, np.datetime64(d)); ji = _asof(jd, np.datetime64(d))
        for name, s, ii in [("garch_skewt", g, gi), ("gjr_skewt", j, ji)]:
            if ii is None:
                out[name].append(np.nan); continue
            p = s.iloc[ii]
            a, b = p["alpha"], p["beta"]
            sb = p["sig2bar"] * 1e4                      # sig2bar stored raw -> percent^2
            if name.startswith("gjr"):
                gam = p["gamma"]; phi = a + 0.5 * gam + b
                neg = (resid < 0).astype(float)
                xx = np.empty_like(rsq); xx[0] = sb
                xx[1:] = sb * (1 - phi) + (a + gam * neg[:-1]) * rsq[:-1]
                s2g = lfilter([1.0], [1.0, -b], xx)
                nl = 1.0 if resid[-1] < 0 else 0.0
                s2n = sb * (1 - phi) + (a + gam * nl) * rsq[-1] + b * s2g[-1]
            else:
                phi = a + b
                xx = np.empty_like(rsq); xx[0] = sb; xx[1:] = sb * (1 - phi) + a * rsq[:-1]
                s2g = lfilter([1.0], [1.0, -b], xx)
                s2n = sb * (1 - phi) + a * rsq[-1] + b * s2g[-1]
            V = v_closed_form(sb / 1e4, s2n / 1e4, phi, h)
            out[name].append(V)
        # --- har (frozen coefs) & rfsv (frozen H, nu2): daily inputs ---
        X = logrv.loc[:d].values
        # HAR features
        Rl = rv.loc[:d].values
        dcomp = np.log(Rl[-1]); wcomp = np.log(Rl[-WEEKLY:].mean()); mcomp = np.log(Rl[-MONTHLY:].mean())
        out["har"].append(_har_daily(d, dcomp, wcomp, mcomp))
        out["rfsv"].append(_rfsv_daily(d, X, h))
    df = pd.DataFrame(out, index=pd.DatetimeIndex(idx))
    return df, rv


# frozen HAR / RFSV path tables (module-level, loaded once)
_HARP = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "har_coef_path.parquet"))
_HARP["date"] = pd.to_datetime(_HARP["date"]); _HARP = _HARP.set_index("date").sort_index()
_RFP = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "rfsv_hurst_path.parquet"))
_RFP["date"] = pd.to_datetime(_RFP["date"]); _RFP = _RFP.set_index("date").sort_index()


def _har_daily(d, dcomp, wcomp, mcomp):
    p = _HARP.iloc[_HARP.index.searchsorted(d, side="right") - 1]
    yhat = p["beta0"] + p["beta_d"] * dcomp + p["beta_w"] * wcomp + p["beta_m"] * mcomp
    return float(np.exp(yhat + 0.5 * p["sigma2_resid"]))


def _rfsv_daily(d, X, h, U=1500):
    p = _RFP.iloc[_RFP.index.searchsorted(d, side="right") - 1]
    Hk = float(np.clip(p["H"], 0.02, 0.49)); nu2 = p["nu2"]
    n = len(X); mu = X.mean(); Xc = X - mu; varX = X.var(ddof=1)
    U = min(U, n - 1); uu = np.arange(1, U + 1.0); past = Xc[::-1][:U]
    dd = np.arange(1, h + 1.0); const = np.cos(np.pi * Hk) / np.pi
    K = const * (dd[:, None] ** (Hk + 0.5)) / ((uu[None, :] + dd[:, None]) * uu[None, :] ** (Hk + 0.5))
    ehat = mu + K @ past
    s2 = np.minimum(c_fbm(Hk) * nu2 * dd ** (2 * Hk), varX)
    return float(np.exp(ehat + 0.5 * s2).mean())


def _dm_matrix(df, models, loss_fn):
    L = {m: loss_fn(df["RV"].values, df[m].values) for m in models}
    recs = []
    for i in models:
        for j in models:
            if i >= j:
                continue
            r = diebold_mariano(L[i], L[j], lag=NW_LAG)
            recs.append({"a": i, "b": j, "dm": r["dm"], "p": r["p"], "lag": NW_LAG,
                         "acf1_diff": r["acf1"]})
    return pd.DataFrame(recs)


def main():
    cfg = load_config()
    seed = int(cfg["meta"]["random_seed"])
    fc, rv = daily_forecasts(cfg)
    # G1: include egarch from the cached daily forecasts, if present
    egpath = os.path.join(repo_root(), "data", "processed", "egarch_daily_secondary.parquet")
    models = list(MODELS)
    if os.path.exists(egpath):
        eg = pd.read_parquet(egpath); eg["date"] = pd.to_datetime(eg["date"])
        fc = fc.join(eg.set_index("date")["egarch_daily"].rename("egarch_skewt"))
        models = models + ["egarch_skewt"]
    fwd = forward_realized_variance_avg(rv, H)
    df = pd.concat([fwd.rename("RV"), fc], axis=1).dropna()
    n = len(df)
    print(f"SECONDARY overlapping daily | n={n} | {df.index.min().date()}..{df.index.max().date()}")
    print(f"NW bandwidth: overlap makes the loss differential MA({H-1}); using lag {NW_LAG} (SPEC >=25).")
    print(f"egarch in daily set: {'egarch_skewt' in models}")
    print(f"positivity: all forecasts > 0 : {bool((df[models] > 0).all().all())}\n")

    ql = {m: qlike(df['RV'].values, df[m].values).mean() for m in models}
    ms = {m: mse(df['RV'].values, df[m].values).mean() for m in models}
    qr = pd.Series(ql).rank(); mr = pd.Series(ms).rank()
    print(f"  {'model':14s} {'QLIKE':>9} {'MSE':>11} {'qlike_rk':>8} {'mse_rk':>7}")
    for m in models:
        print(f"  {m:14s} {ql[m]:9.4f} {ms[m]:11.3e} {int(qr[m]):8d} {int(mr[m]):7d}")
    print(f"\n  QLIKE vs MSE rank corr: {qr.corr(mr, method='spearman'):.3f}  | "
          f"QLIKE best {min(ql,key=ql.get)}  MSE best {min(ms,key=ms.get)}")

    # DM matrix (secondary) written to disk
    dq = _dm_matrix(df, models, qlike); dq["loss"] = "qlike"
    dm_ = _dm_matrix(df, models, mse); dm_["loss"] = "mse"
    pd.concat([dq, dm_], ignore_index=True).to_csv(os.path.join(TAB, "dm_matrix_secondary.csv"), index=False)

    # MCS with vs without egarch: does egarch's inclusion change membership?
    mcs_rows = []
    for loss_name, loss_fn in [("qlike", qlike), ("mse", mse)]:
        for setlabel, cset in [("with_egarch", models), ("without_egarch", MODELS)]:
            L = np.column_stack([loss_fn(df['RV'].values, df[m].values) for m in cset])
            res = model_confidence_set(L, cset, alpha=0.10, B=10000, block=5, seed=seed)
            print(f"  MCS[{loss_name}, {setlabel:13s}] retained ({len(res['retained'])}/{len(cset)}): "
                  f"{', '.join(res['retained'])}")
            for m in cset:
                mcs_rows.append({"loss": loss_name, "set": setlabel, "model": m,
                                 "mcs_p": res["mcs_p"][m], "retained": m in res["retained"]})
        # membership change among the shared models
        w = {r["model"] for r in mcs_rows if r["loss"] == loss_name and r["set"] == "with_egarch" and r["retained"] and r["model"] != "egarch_skewt"}
        wo = {r["model"] for r in mcs_rows if r["loss"] == loss_name and r["set"] == "without_egarch" and r["retained"]}
        print(f"       -> egarch inclusion changes other models' membership: {'YES ' + str(w ^ wo) if w != wo else 'no'}")
    pd.DataFrame(mcs_rows).to_csv(os.path.join(TAB, "mcs_secondary.csv"), index=False)
    print(f"\ntables -> {TAB}/{{dm_matrix_secondary,mcs_secondary}}.csv")


if __name__ == "__main__":
    main()

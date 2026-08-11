#!/usr/bin/env python3
"""
E1: RFSV level-shortfall diagnosis (REPORT ONLY; does not modify direct_h.py).

mean(V_hat)/mean(RV)=0.81 for corrected rfsv. Locate the shortfall:
  (a) mean(ehat) vs mean(log rv_daily) on the forward sample -> is the point
      forecast log-unbiased? If yes, the shortfall is in s2.
  (b) truncation: sensitivity of ehat to doubling the history U.
  (c) s2 constant: empirical conditional variance var(log rv_{t+d} - ehat) vs the
      implemented 0.5*nu2*d^{2H}, and the theoretical fBm conditional-variance
      constant c(H) (Gaussian conditioning on the fBm covariance).
  (d) estimated effect of the corrected c(H) on V_t(21), w, realized vol, MZ slope.
Reads model/backtest code; no edits. No network.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gamma as Gamma

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.rv.forward_target import forward_realized_variance_avg
from volteq.models.rebalance import rebalance_dates
from volteq.models.direct_h import rfsv_fit_forecast
from volteq.backtest.engine import run_backtest, sizing_weight, metrics, ANNUAL

H = 21
GRID = {"est": None, "0.02": 0.02, "0.05": 0.05, "0.10": 0.10, "0.15": 0.15}


def _kernel_ehat(X, Hk, U=1500):
    """Reproduce rfsv ehat_d (d=1..H) for log-RV series X (through t)."""
    n = len(X); mu = X.mean(); Xc = X - mu
    U = min(U, n - 1)
    uu = np.arange(1, U + 1, dtype=float)
    past = Xc[::-1][:U]
    d = np.arange(1, H + 1, dtype=float)
    c = np.cos(np.pi * Hk) / np.pi
    K = c * (d[:, None] ** (Hk + 0.5)) / ((uu[None, :] + d[:, None]) * uu[None, :] ** (Hk + 0.5))
    return mu + K @ past          # length H


def fbm_cond_var_const(Hval, N=700):
    """c(H) = Var(W^H_{t+d} | W^H_{t..t-N}) / d^{2H} for standard fBm (numeric,
    Gaussian conditioning). Anchored far from the W_0=0 pinning."""
    T = N
    past = np.arange(T, T - N - 1, -1.0)      # t, t-1, ..., t-N  (all > 0)
    out = []
    for d in range(1, H + 1):
        allt = np.concatenate([[T + d], past])
        A = allt[:, None]; B = allt[None, :]
        C = 0.5 * (np.abs(A) ** (2 * Hval) + np.abs(B) ** (2 * Hval) - np.abs(A - B) ** (2 * Hval))
        Cpp = C[1:, 1:] + 1e-9 * np.eye(C.shape[0] - 1)
        cond = C[0, 0] - C[0, 1:] @ np.linalg.solve(Cpp, C[0, 1:])
        out.append(cond / d ** (2 * Hval))
    return float(np.mean(out))


def c_gamma_closed(Hval):
    """Candidate closed form built from Gamma(3/2-H), Gamma(H+1/2), 2H:
    c(H) = Gamma(3/2-H) / (2H * Gamma(H+1/2) * Gamma(2-2H)).  Reported for cross-check."""
    return Gamma(1.5 - Hval) / (2 * Hval * Gamma(Hval + 0.5) * Gamma(2 - 2 * Hval))


def main():
    cfg = load_config()
    panel = load_panel(); rv = panel["rv_daily"]
    rv_qqq = panel.loc[panel["source"] == "qqq", "rv_daily"]
    logrv = np.log(rv)
    dates = rebalance_dates(cfg, data_end=panel.index.max(), include_warmup=True)
    fwd = forward_realized_variance_avg(rv_qqq, H).reindex(dates)
    dff = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "dff.parquet"))
    dff["date"] = pd.to_datetime(dff["date"]); dff = dff.set_index("date")["dff"].sort_index()
    qqq = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "qqq_daily.parquet"))
    qqq["date"] = pd.to_datetime(qqq["date"]); qqq_ret = qqq.set_index("date")["close"].sort_index().pct_change().dropna()

    # ---- (a) point-forecast log bias + (c) empirical conditional variance (est-H) ----
    print("=" * 88); print("E1(a,c)  ehat log-bias and empirical conditional variance  [rfsv est-H]"); print("=" * 88)
    resid_by_d = {d: [] for d in range(1, H + 1)}
    per_date_avg_resid = []
    nu2_list, Hest_list = [], []
    li = logrv.index
    for t in dates:
        Xt = logrv.loc[:t].to_numpy()
        f = rfsv_fit_forecast(rv.loc[:t].to_numpy(), H)   # for H, nu2
        Hk, nu2 = f["H"], f["nu2"]; nu2_list.append(nu2); Hest_list.append(Hk)
        eh = _kernel_ehat(Xt, float(np.clip(Hk, 0.02, 0.49)))
        pos = li.get_loc(t)
        rr = []
        for d in range(1, H + 1):
            if pos + d < len(li):
                res = logrv.iloc[pos + d] - eh[d - 1]
                resid_by_d[d].append(res); rr.append(res)
        if rr:
            per_date_avg_resid.append(np.mean(rr))
    per_date_avg_resid = np.array(per_date_avg_resid)
    # HAC SE on the per-date average residual
    X1 = np.ones((len(per_date_avg_resid), 1))
    m = sm.OLS(per_date_avg_resid, X1).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    print(f"  mean(log rv_daily_(t+d) - ehat) = {per_date_avg_resid.mean():+.4f}  "
          f"HAC SE {float(np.asarray(m.bse)[0]):.4f}  (=0 places the shortfall entirely in s2)")
    emp_s2 = np.array([np.var(resid_by_d[d], ddof=1) for d in range(1, H + 1)])
    nu2_bar = float(np.mean(nu2_list)); Hbar = float(np.mean(Hest_list))
    impl_s2 = 0.5 * nu2_bar * np.arange(1, H + 1) ** (2 * Hbar)
    print(f"  mean nu^2 {nu2_bar:.3f}, mean H {Hbar:.4f}")
    print(f"  implemented s2 (0.5*nu2*d^2H): mean over d = {impl_s2.mean():.3f}")
    print(f"  EMPIRICAL cond var var(logrv_(t+d)-ehat): mean over d = {emp_s2.mean():.3f}")
    print(f"  empirical c(H) = emp_s2 / (nu2*d^2H): mean = {(emp_s2/(nu2_bar*np.arange(1,H+1)**(2*Hbar))).mean():.3f}")
    # NON-NORMALITY check: is exp(0.5*var) the right retransformation for these residuals?
    from scipy.stats import skew as _sk, kurtosis as _ku
    allr = np.concatenate([np.array(resid_by_d[d]) for d in range(1, H + 1)])
    gauss = np.exp(0.5 * allr.var(ddof=1))
    emp = np.exp(allr).mean()
    print(f"\n  NON-NORMALITY of the log-RV forecast residual r (pooled, n={len(allr)}):")
    print(f"    skew {_sk(allr):+.3f}  excess kurtosis {_ku(allr):+.3f}")
    print(f"    E[exp(r)] empirical {emp:.4f}  vs Gaussian exp(0.5 var) {gauss:.4f}  "
          f"ratio {emp/gauss:.4f}")
    print(f"    -> Gaussian retransformation recovers {gauss/emp:.3f} of E[exp(r)] "
          f"(cf. mean V_hat/RV 0.81)")

    # ---- (b) truncation sensitivity of ehat to doubling U ----
    print("\n" + "=" * 88); print("E1(b)  truncation: history length and ehat sensitivity to doubling U"); print("=" * 88)
    for t in [dates[5], dates[160], dates[-1]]:
        Xt = logrv.loc[:t].to_numpy()
        eh1 = _kernel_ehat(Xt, float(np.clip(Hbar, 0.02, 0.49)), U=1500)
        eh2 = _kernel_ehat(Xt, float(np.clip(Hbar, 0.02, 0.49)), U=3000)
        print(f"  {t.date()}: history {len(Xt)} days | mean ehat U=1500 {eh1.mean():.4f} "
              f"U=3000 {eh2.mean():.4f}  d(V0) {np.exp(eh2).mean()/np.exp(eh1).mean()-1:+.3%}")

    # ---- (c/d) theoretical c(H) and the corrected-constant effect ----
    print("\n" + "=" * 88); print("E1(c,d)  theoretical fBm conditional-variance constant c(H) and effect"); print("=" * 88)
    print(f"  {'H':>6} {'c(H)_numeric':>13} {'c_gamma_form':>13}")
    for k, Hf in GRID.items():
        Hval = Hbar if Hf is None else Hf
        print(f"  {k:>6} {fbm_cond_var_const(Hval):13.4f} {c_gamma_closed(Hval):13.4f}")
    # crossover: H where c(H)=0.5
    Hs = np.linspace(0.02, 0.49, 200)
    cs = np.array([fbm_cond_var_const(h) for h in Hs])
    cross = Hs[np.argmin(np.abs(cs - 0.5))]
    print(f"  c(H) crosses the implemented 0.5 at H ~= {cross:.3f}")

    # effect: rebuild V_hat with s2 = c(H)*nu2*d^2H, size, backtest, MZ
    print("\n  EFFECT of replacing 0.5 with theoretical c(H) (per variant):")
    print(f"  {'variant':>7} {'cH':>5} {'corr_old':>9} {'corr_new':>9} {'VR_old':>7} {'VR_new':>7} "
          f"{'rvol_old':>8} {'rvol_new':>8} {'MZb_old':>7} {'MZb_new':>7}")
    for k, Hf in GRID.items():
        Hval = Hbar if Hf is None else Hf
        cH = fbm_cond_var_const(Hval)
        v_old, v_new = [], []
        for t in dates:
            Xt = logrv.loc[:t].to_numpy()
            f = rfsv_fit_forecast(rv.loc[:t].to_numpy(), H, H_fixed=Hf)
            Hk = float(np.clip(f["H"], 0.02, 0.49)); nu2 = f["nu2"]
            eh = _kernel_ehat(Xt, Hk)
            d = np.arange(1, H + 1); s2raw = nu2 * d ** (2 * Hk)
            varX = float(np.log(rv.loc[:t]).var(ddof=1))
            s2_old = np.minimum(0.5 * s2raw, varX); s2_new = np.minimum(cH * s2raw, varX)
            v_old.append(np.exp(eh + 0.5 * s2_old).mean())
            v_new.append(np.exp(eh + 0.5 * s2_new).mean())
        v_old = pd.Series(v_old, index=dates); v_new = pd.Series(v_new, index=dates)
        rr = fwd
        VR_old = v_old.mean() / rr.mean(); VR_new = v_new.mean() / rr.mean()
        # correction factors vs uncorrected v0 (recompute quickly)
        # realized vol via backtest
        def rvol(vser):
            ws = pd.Series(sizing_weight(vser.to_numpy(), cfg), index=vser.index)
            return metrics(run_backtest(ws, qqq_ret, dff, cfg), cfg)["realized_vol"]
        def mzb(vser):
            j = pd.concat([rr.rename("y"), vser.rename("f")], axis=1).dropna()
            return sm.OLS(j["y"], sm.add_constant(j["f"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).params.iloc[1]
        print(f"  {k:>7} {cH:5.2f} {'-':>9} {'-':>9} {VR_old:7.3f} {VR_new:7.3f} "
              f"{rvol(v_old):8.3f} {rvol(v_new):8.3f} {mzb(v_old):7.3f} {mzb(v_new):7.3f}")


if __name__ == "__main__":
    main()

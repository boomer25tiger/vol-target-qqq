"""
Direct-h realized-variance models. Each maps the daily RV proxy history through t
to V_t(21) directly, with no separate GARCH-style fit/forecast split.

- rv:   trailing realized variance over the prior 21 sessions (no estimation).
- har:  HAR-RV (Corsi 2009) as a direct-h OLS regression of the forward 21-day
        average log RV on daily/weekly/monthly log RV, refit on data through t.
- rfsv: Hurst H estimated from the variogram of log RV (expanding window), then
        the RFSV (Gatheral-Jaisson-Rosenbaum 2018) log-variance prediction formula.

har and rfsv predict log RV, so the variance forecast needs the lognormal
retransformation  V = exp(mu_hat + sigma_hat^2 / 2); omitting it biases variance
low and inflates leverage. All inputs use only data through t (causal).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as _gammafn

WEEKLY, MONTHLY = 5, 22


def c_fbm(Hval: float) -> float:
    """Conditional-variance constant of fractional Brownian motion given the past,
    Var(W^H_{t+d} | past) = c(H) * d^{2H}:

        c(H) = Gamma(3/2 - H) / (Gamma(H + 1/2) * Gamma(2 - 2H)).

    Gripenberg & Norros (1996); Nuzman & Poor (2000); RFSV application
    Gatheral-Jaisson-Rosenbaum (2018). c(0+) = 1/2, c(1/2) = 1 (reduction to
    standard Brownian motion), monotone increasing on (0, 1/2].
    """
    return float(_gammafn(1.5 - Hval) / (_gammafn(Hval + 0.5) * _gammafn(2.0 - 2.0 * Hval)))


# --------------------------------------------------------------------------
# rv: trailing realized variance
# --------------------------------------------------------------------------

def rv_forecast(rv_daily: np.ndarray, h: int) -> float:
    """Trailing mean daily variance over the prior h sessions (through t)."""
    r = np.asarray(rv_daily, float)
    return float(r[-h:].mean())


# --------------------------------------------------------------------------
# har: HAR-RV direct-h regression with lognormal retransformation
# --------------------------------------------------------------------------

def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean of the last w values at each position (NaN until w-1)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def har_fit_forecast(rv_daily: np.ndarray, h: int) -> dict:
    r = np.asarray(rv_daily, float)
    n = len(r)
    logr = np.log(r)
    d = logr
    w = np.log(_roll_mean(r, WEEKLY))
    m = np.log(_roll_mean(r, MONTHLY))
    # forward h-day average log RV target at s: log(mean(r[s+1:s+h+1]))
    fwd = np.full(n, np.nan)
    fm = _roll_mean(r, h)                 # trailing h-mean at position s covers [s-h+1, s]
    fwd[:-h] = np.log(fm[h:])             # shift so target at s uses [s+1, s+h]

    # training rows: features defined (>= MONTHLY-1) and forward observed (<= n-1-h)
    lo, hi = MONTHLY - 1, n - 1 - h
    s = np.arange(lo, hi + 1)
    X = np.column_stack([np.ones(len(s)), d[s], w[s], m[s]])
    y = fwd[s]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(s) - X.shape[1]
    sigma2 = float(resid @ resid / dof)   # residual variance for the retransformation

    xt = np.array([1.0, d[-1], w[-1], m[-1]])
    yhat = float(xt @ beta)
    v0 = np.exp(yhat)                     # uncorrected
    V = np.exp(yhat + 0.5 * sigma2)       # lognormal retransformation
    return {
        "V21": float(V), "beta0": float(beta[0]), "beta_d": float(beta[1]),
        "beta_w": float(beta[2]), "beta_m": float(beta[3]),
        "sigma2_resid": sigma2, "retransform_pct": float(V / v0 - 1.0),
        "n_train": int(len(s)),
    }


# --------------------------------------------------------------------------
# rfsv: variogram Hurst + RFSV prediction formula + retransformation
# --------------------------------------------------------------------------

def hurst_variogram(X: np.ndarray, max_lag: int = 40) -> dict:
    """Estimate H and nu^2 from the log-RV variogram m(d)=E[(X_{t+d}-X_t)^2]=nu^2 d^{2H}."""
    lags = np.arange(1, max_lag + 1)
    m = np.array([np.mean((X[d:] - X[:-d]) ** 2) for d in lags])
    lx, ly = np.log(lags), np.log(m)
    A = np.column_stack([np.ones(len(lx)), lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    dof = len(lx) - 2
    s2 = (resid @ resid) / dof
    se_slope = float(np.sqrt(s2 / np.sum((lx - lx.mean()) ** 2)))
    slope = float(coef[1])
    return {"H": slope / 2.0, "H_se": se_slope / 2.0, "nu2": float(np.exp(coef[0]))}


def _nu2_given_H(X: np.ndarray, H: float, max_lag: int) -> float:
    """nu^2 from the variogram with the slope fixed at 2H (for a fixed-H grid)."""
    lags = np.arange(1, max_lag + 1)
    m = np.array([np.mean((X[d:] - X[:-d]) ** 2) for d in lags])
    return float(np.exp(np.mean(np.log(m) - 2.0 * H * np.log(lags))))


def _rfsv_pointwise(rv_daily: np.ndarray, h: int, H_fixed: float | None,
                    max_lag: int, U: int) -> dict:
    """Per-horizon RFSV point forecasts pt[d] = exp(ehat_d + s2_d/2) for d = 1..h,
    the shared core of the h=21 forecast and the h=1..H term structure. The retransformed
    forecast averaged over d = 1..h is V_t(h); a running mean of pt gives the term structure.
    ehat_d and s2_d depend on d alone, so pt[:k] is identical whether computed with h=k or
    any larger h, which makes the term structure reproduce the point forecast exactly."""
    r = np.asarray(rv_daily, float)
    X = np.log(r)
    n = len(X)
    vg = hurst_variogram(X, max_lag)
    if H_fixed is None:
        H_raw, H_se, nu2 = vg["H"], vg["H_se"], vg["nu2"]
    else:
        H_raw, H_se, nu2 = float(H_fixed), 0.0, _nu2_given_H(X, float(H_fixed), max_lag)
    H = float(np.clip(H_raw, 0.02, 0.49))     # kernel stability; report H_raw
    mu = float(X.mean())
    varX = float(X.var(ddof=1))
    Xc = X - mu

    U = min(U, n - 1)
    uu = np.arange(1, U + 1, dtype=float)
    past = Xc[::-1][:U]                        # past[k] = X_{t-k} - mu, k=0..U-1 (u=k+1)
    deltas = np.arange(1, h + 1, dtype=float)
    const = np.cos(np.pi * H) / np.pi
    # kernel K[d, u] = const * d^{H+1/2} / ((u+d) u^{H+1/2})
    K = const * (deltas[:, None] ** (H + 0.5)) / ((uu[None, :] + deltas[:, None])
                                                  * uu[None, :] ** (H + 0.5))
    ehat = mu + K @ past                       # E[X_{t+d}], d=1..h
    s2_raw = c_fbm(H) * nu2 * deltas ** (2.0 * H)   # c(H) = fBm conditional-var constant
    s2 = np.minimum(s2_raw, varX)              # forecast-error variance (capped)
    cap_binds = int(np.sum(s2_raw >= varX))    # how often the min() guard binds
    return {
        "pt": np.exp(ehat + 0.5 * s2), "pt0": np.exp(ehat),   # retransformed / uncorrected, d=1..h
        "H_raw": float(H_raw), "H_se": float(H_se), "nu2": float(nu2),
        "H_clipped": bool(H != H_raw), "cap_binds": cap_binds, "n": int(n),
    }


def rfsv_fit_forecast(rv_daily: np.ndarray, h: int, H_fixed: float | None = None,
                      max_lag: int = 40, U: int = 1500) -> dict:
    """RFSV V_t(h). H_fixed=None estimates H from the variogram; otherwise H is
    fixed (nu^2 re-estimated with the slope pinned at 2H)."""
    p = _rfsv_pointwise(rv_daily, h, H_fixed, max_lag, U)
    v_corr = float(p["pt"].mean())            # V_t(h) with retransformation
    v0 = float(p["pt0"].mean())               # uncorrected
    return {
        "V21": v_corr, "H": p["H_raw"], "H_se": p["H_se"],
        "nu2": p["nu2"], "H_clipped": p["H_clipped"],
        "cap_binds": p["cap_binds"], "cap_frac": float(p["cap_binds"] / h),
        "retransform_pct": float(v_corr / v0 - 1.0), "n": p["n"],
    }


def rfsv_term_structure(rv_daily: np.ndarray, hmax: int, H_fixed: float | None = None,
                        max_lag: int = 40, U: int = 1500) -> np.ndarray:
    """V_t(h) for h = 1..hmax as a running mean of the per-horizon point forecasts.
    Element h-1 equals rfsv_fit_forecast(rv_daily, h, H_fixed)['V21'] exactly."""
    p = _rfsv_pointwise(rv_daily, hmax, H_fixed, max_lag, U)
    return np.cumsum(p["pt"]) / np.arange(1, hmax + 1)


def har_term_structure(rv_daily: np.ndarray, hmax: int) -> np.ndarray:
    """V_t(h) for h = 1..hmax, each a separate direct-h HAR regression (the forward
    target is horizon-specific, so there is no shared computation to reuse)."""
    return np.array([har_fit_forecast(rv_daily, h)["V21"] for h in range(1, hmax + 1)])

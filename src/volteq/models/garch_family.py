"""
GARCH-family fitters.

- garch, gjr: variance-targeted QMLE (skewed-t or Gaussian). The unconditional
  variance is pinned to the sample second moment (sigma-bar^2 = mean of squared
  residuals) and omega = sigma-bar^2 * (1 - persistence) is profiled out, so only
  the dynamics and shape parameters are optimized (Engle-Mezrich targeting).
- egarch: arch's QMLE (no closed-form unconditional variance; aggregated by MC).
- ewma: RiskMetrics lambda = 0.94, no estimation.

Fitting is done on percent returns (x100) for numerical stability; sigma-bar^2 and
the one-step forecast are returned in raw (fraction^2) daily-variance units.

The GARCH/GJR variance recursion s2[t] = beta*s2[t-1] + x[t] is a first-order IIR
filter, evaluated with scipy.signal.lfilter (C speed) inside the objective.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize
from arch.univariate import SkewStudent, Normal

SCALE = 100.0            # fit in percent
_EPS = 1e-6
_VAR_FLOOR = 1e-12


# --------------------------------------------------------------------------
# variance-targeted recursion (percent units)
# --------------------------------------------------------------------------

def _garch_sigma2(resid_sq, sig2bar, alpha, beta):
    c = sig2bar * (1.0 - alpha - beta)
    x = np.empty_like(resid_sq)
    x[0] = sig2bar
    x[1:] = c + alpha * resid_sq[:-1]
    s2 = lfilter([1.0], [1.0, -beta], x)
    return np.maximum(s2, _VAR_FLOOR)


def _gjr_sigma2(resid, resid_sq, sig2bar, alpha, gamma, beta):
    phi = alpha + 0.5 * gamma + beta
    c = sig2bar * (1.0 - phi)
    neg = (resid < 0.0).astype(float)
    x = np.empty_like(resid_sq)
    x[0] = sig2bar
    x[1:] = c + (alpha + gamma * neg[:-1]) * resid_sq[:-1]
    s2 = lfilter([1.0], [1.0, -beta], x)
    return np.maximum(s2, _VAR_FLOOR)


def _loglik(resid, s2, dist, shape):
    if dist == "normal":
        return float(-0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + resid ** 2 / s2))
    return float(SkewStudent().loglikelihood(shape, resid, s2, individual=False))


# --------------------------------------------------------------------------
# fitters
# --------------------------------------------------------------------------

def _finalize(model, dist, converged, params, phi, sig2bar_pct, s2_next_pct, n):
    return {
        "model": model, "dist": dist, "converged": bool(converged),
        "params": params, "persistence": float(phi),
        "sig2bar": float(sig2bar_pct / SCALE ** 2),        # raw daily variance
        "sigma2_next": float(s2_next_pct / SCALE ** 2),    # raw daily variance
        "n": int(n),
    }


def fit_vt_garch(returns: np.ndarray, dist: str = "skewt") -> dict:
    r = np.asarray(returns, float) * SCALE
    mu = r.mean()
    resid = r - mu
    resid_sq = resid ** 2
    sig2bar = resid_sq.mean()

    def neg_ll(theta):
        alpha, beta = theta[0], theta[1]
        shape = theta[2:] if dist == "skewt" else []
        s2 = _garch_sigma2(resid_sq, sig2bar, alpha, beta)
        return -_loglik(resid, s2, dist, shape)

    if dist == "skewt":
        x0 = [0.05, 0.90, 8.0, -0.10]
        bounds = [(_EPS, 1.0), (_EPS, 1.0), (2.05, 300.0), (-0.99, 0.99)]
    else:
        x0 = [0.05, 0.90]
        bounds = [(_EPS, 1.0), (_EPS, 1.0)]
    cons = [{"type": "ineq", "fun": lambda th: 1.0 - _EPS - (th[0] + th[1])}]
    res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})

    alpha, beta = res.x[0], res.x[1]
    shape = list(res.x[2:]) if dist == "skewt" else []
    s2 = _garch_sigma2(resid_sq, sig2bar, alpha, beta)
    s2_next = sig2bar * (1.0 - alpha - beta) + alpha * resid_sq[-1] + beta * s2[-1]
    params = {"mu": float(mu / SCALE), "alpha": float(alpha), "beta": float(beta)}
    if dist == "skewt":
        params.update({"eta": float(shape[0]), "lambda": float(shape[1])})
    return _finalize("garch", dist, res.success, params, alpha + beta, sig2bar, s2_next, len(r))


def fit_vt_gjr(returns: np.ndarray, dist: str = "skewt") -> dict:
    r = np.asarray(returns, float) * SCALE
    mu = r.mean()
    resid = r - mu
    resid_sq = resid ** 2
    sig2bar = resid_sq.mean()

    def neg_ll(theta):
        alpha, gamma, beta = theta[0], theta[1], theta[2]
        shape = theta[3:] if dist == "skewt" else []
        s2 = _gjr_sigma2(resid, resid_sq, sig2bar, alpha, gamma, beta)
        return -_loglik(resid, s2, dist, shape)

    if dist == "skewt":
        x0 = [0.03, 0.06, 0.90, 8.0, -0.10]
        bounds = [(0.0, 1.0), (-0.5, 1.0), (_EPS, 1.0), (2.05, 300.0), (-0.99, 0.99)]
    else:
        x0 = [0.03, 0.06, 0.90]
        bounds = [(0.0, 1.0), (-0.5, 1.0), (_EPS, 1.0)]
    cons = [
        {"type": "ineq", "fun": lambda th: 1.0 - _EPS - (th[0] + 0.5 * th[1] + th[2])},  # phi<1
        {"type": "ineq", "fun": lambda th: th[0] + th[1]},   # alpha+gamma >= 0
    ]
    res = minimize(neg_ll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})

    alpha, gamma, beta = res.x[0], res.x[1], res.x[2]
    shape = list(res.x[3:]) if dist == "skewt" else []
    phi = alpha + 0.5 * gamma + beta
    s2 = _gjr_sigma2(resid, resid_sq, sig2bar, alpha, gamma, beta)
    neg_last = 1.0 if resid[-1] < 0 else 0.0
    s2_next = sig2bar * (1.0 - phi) + (alpha + gamma * neg_last) * resid_sq[-1] + beta * s2[-1]
    params = {"mu": float(mu / SCALE), "alpha": float(alpha), "gamma": float(gamma),
              "beta": float(beta)}
    if dist == "skewt":
        params.update({"eta": float(shape[0]), "lambda": float(shape[1])})
    return _finalize("gjr", dist, res.success, params, phi, sig2bar, s2_next, len(r))


def garch_state(returns: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
    """Recompute (sig2bar, sigma^2_{t+1|t}) on current data with FIXED alpha,beta
    (raw units). Used to carry a converged parameter set onto a later window when
    a re-fit fails to converge."""
    r = np.asarray(returns, float) * SCALE
    resid = r - r.mean()
    resid_sq = resid ** 2
    sig2bar = resid_sq.mean()
    s2 = _garch_sigma2(resid_sq, sig2bar, alpha, beta)
    s2_next = sig2bar * (1.0 - alpha - beta) + alpha * resid_sq[-1] + beta * s2[-1]
    return sig2bar / SCALE ** 2, s2_next / SCALE ** 2


def gjr_state(returns: np.ndarray, alpha: float, gamma: float, beta: float) -> tuple[float, float]:
    r = np.asarray(returns, float) * SCALE
    resid = r - r.mean()
    resid_sq = resid ** 2
    sig2bar = resid_sq.mean()
    phi = alpha + 0.5 * gamma + beta
    s2 = _gjr_sigma2(resid, resid_sq, sig2bar, alpha, gamma, beta)
    neg_last = 1.0 if resid[-1] < 0 else 0.0
    s2_next = sig2bar * (1.0 - phi) + (alpha + gamma * neg_last) * resid_sq[-1] + beta * s2[-1]
    return sig2bar / SCALE ** 2, s2_next / SCALE ** 2


def egarch_apply(returns: np.ndarray, params: dict, dist: str) -> dict:
    """Filter current data at a FIXED egarch parameter set (arch .fix), for
    carry-forward. Returns the same shape as fit_egarch, with '_res'."""
    from arch import arch_model
    r = np.asarray(returns, float) * SCALE
    d = "skewt" if dist == "skewt" else "normal"
    am = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=d)
    res = am.fix(list(params.values()))
    beta = float(params.get("beta[1]", 0.0))
    fc1 = res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    out = _finalize("egarch", dist, True, dict(params), beta, np.nan, float(fc1), len(r))
    out["_res"] = res
    out["carried"] = True
    return out


def fit_egarch(returns: np.ndarray, dist: str = "skewt") -> dict:
    """EGARCH(1,1,1) via arch. Returns the arch results under '_res' for MC
    aggregation; persistence is the log-variance AR coefficient beta."""
    from arch import arch_model
    r = np.asarray(returns, float) * SCALE
    d = "skewt" if dist == "skewt" else "normal"
    am = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist=d)
    res = am.fit(disp="off", show_warning=False)
    p = res.params
    beta = float(p.get("beta[1]", 0.0))
    fc1 = res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    params = {k: float(v) for k, v in p.items()}
    out = _finalize("egarch", dist, res.convergence_flag == 0, params, beta,
                    sig2bar_pct=np.nan, s2_next_pct=float(fc1), n=len(r))
    out["_res"] = res
    return out


def fit_ewma(returns: np.ndarray, lam: float = 0.94) -> dict:
    """RiskMetrics EWMA (fixed lambda). No estimation; phi = 1 (flat term)."""
    r = np.asarray(returns, float) * SCALE
    r_sq = r ** 2
    s2bar = r_sq.mean()
    x = np.empty_like(r_sq)
    x[0] = s2bar
    x[1:] = (1.0 - lam) * r_sq[:-1]
    s2 = lfilter([1.0], [1.0, -lam], x)
    s2_next = lam * s2[-1] + (1.0 - lam) * r_sq[-1]
    return _finalize("ewma", "none", True, {"lambda": float(lam)}, 1.0,
                     sig2bar_pct=s2bar, s2_next_pct=float(s2_next), n=len(r))

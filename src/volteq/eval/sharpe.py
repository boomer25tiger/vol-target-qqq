"""
Sharpe-ratio inference for Layer 2 (SPEC.md Section 10.2).

- Ledoit-Wolf (2008) pairwise Sharpe-difference test with a HAC (Newey-West)
  standard error via the delta method on (mu_i, mu_j, E[r_i^2], E[r_j^2]).
- Deflated Sharpe ratio (Bailey & Lopez de Prado 2014).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649015329


def nw_auto_lag(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _hac_longrun(M: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West long-run covariance of the per-t moment vector rows of M."""
    n = M.shape[0]
    Mc = M - M.mean(axis=0)
    S = Mc.T @ Mc / n
    for l in range(1, lag + 1):
        G = Mc[l:].T @ Mc[:-l] / n
        S += (1.0 - l / (lag + 1.0)) * (G + G.T)
    return S


def ledoit_wolf_sharpe_test(ri: np.ndarray, rj: np.ndarray, lag: int | None = None):
    """Test H0: SR_i = SR_j (excess returns). Returns dict(diff, z, p, lag)."""
    ri = np.asarray(ri, float); rj = np.asarray(rj, float)
    n = len(ri)
    if lag is None:
        lag = nw_auto_lag(n)
    mi, mj = ri.mean(), rj.mean()
    gi, gj = (ri ** 2).mean(), (rj ** 2).mean()      # uncentered second moments
    si, sj = np.sqrt(gi - mi ** 2), np.sqrt(gj - mj ** 2)
    sr_i, sr_j = mi / si, mj / sj
    diff = sr_i - sr_j
    grad = np.array([gi / si ** 3, -gj / sj ** 3,
                     -mi / (2 * si ** 3), mj / (2 * sj ** 3)])
    M = np.column_stack([ri, rj, ri ** 2, rj ** 2])
    S = _hac_longrun(M, lag)
    var_diff = float(grad @ (S / n) @ grad)
    if var_diff > 0:
        z = diff / np.sqrt(var_diff)
    elif diff == 0.0:
        z = 0.0          # identical series: Sharpes equal, difference exactly null
    else:
        z = np.nan
    return {"diff": float(diff), "z": float(z),
            "p": float(2 * (1 - norm.cdf(abs(z)))), "lag": int(lag)}


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_variance: float) -> dict:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014). `returns` are the
    per-period (daily) excess returns; SR is the per-period Sharpe. `sr_variance`
    is the variance of the SR estimates across the `n_trials` configurations."""
    r = np.asarray(returns, float); n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd
    g3 = float(skew(r)); g4 = float(kurtosis(r, fisher=False))   # full (non-excess) kurtosis
    sr0 = np.sqrt(sr_variance) * ((1 - EULER) * norm.ppf(1 - 1.0 / n_trials)
                                  + EULER * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    denom = np.sqrt(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2)
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return {"sr_daily": float(sr), "sr0": float(sr0), "skew": g3, "kurtosis": g4,
            "dsr": float(norm.cdf(z))}

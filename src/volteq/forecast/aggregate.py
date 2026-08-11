"""
Aggregation of one-step conditional variance to the h = 21 sizing object

    V_t(h) = E_t[ (1/h) * sum_{k=1..h} sigma^2_{t+k} ].

- garch/gjr: closed form  V = sigma-bar^2 + lambda_h * (sigma^2_{t+1|t} - sigma-bar^2),
  lambda_h = (1 - phi^h) / (h * (1 - phi)),  phi = alpha+beta (garch),
  phi = alpha + gamma/2 + beta (gjr, symmetric-innovation convention, SPEC.md 6).
- ewma: identity (phi = 1 -> lambda_h = 1 -> flat term structure): V = sigma^2_{t+1|t}.
- egarch: Monte Carlo via arch's simulation forecast (no closed form in log-variance).
"""
from __future__ import annotations

import numpy as np

SCALE2 = 100.0 ** 2   # percent^2 -> fraction^2


def lambda_h(phi: float, h: int) -> float:
    if abs(phi - 1.0) < 1e-12:
        return 1.0
    return (1.0 - phi ** h) / (h * (1.0 - phi))


def v_closed_form(sig2bar: float, sigma2_next: float, phi: float, h: int) -> float:
    return float(sig2bar + lambda_h(phi, h) * (sigma2_next - sig2bar))


def v_identity(sigma2_next: float) -> float:
    return float(sigma2_next)


def v_egarch_mc_path(res, h: int, n_paths: int, seed: int) -> np.ndarray:
    """E_t[sigma^2_{t+k}] for k=1..h (fraction^2), the per-step MC forecast path.
    v_egarch_mc(res, h, ...) equals this path's mean; the term structure V_t(k) is its
    running mean, cumsum(path)/arange(1, h+1)."""
    fc = res.forecast(horizon=h, method="simulation", simulations=n_paths,
                      random_state=np.random.RandomState(seed), reindex=False)
    return fc.variance.iloc[-1].to_numpy() / SCALE2   # E[sigma^2_{t+k}] fraction^2, k=1..h


def v_egarch_mc(res, h: int, n_paths: int, seed: int) -> float:
    """V_t(h) from arch's simulation forecast (returns raw fraction^2)."""
    return float(v_egarch_mc_path(res, h, n_paths, seed).mean())


def simulate_garch_avg_var(sig2bar: float, sigma2_next: float, alpha: float,
                           beta: float, h: int, n_paths: int, seed: int) -> dict:
    """Monte Carlo the VT-GARCH forward from sigma^2_{t+1|t} and return the mean
    (1/h) sum sigma^2_{t+k} across paths, for checking the closed form. The
    aggregation expectation depends only on E[z^2] = 1, so standardized normal
    innovations are used (any unit-variance innovation gives the same E_t)."""
    rng = np.random.RandomState(seed)
    z = rng.standard_normal((n_paths, h))
    c = sig2bar * (1.0 - alpha - beta)
    s2 = np.empty((n_paths, h))
    s2[:, 0] = sigma2_next                             # sigma^2_{t+1|t} is known
    for k in range(1, h):
        eps2 = s2[:, k - 1] * z[:, k - 1] ** 2
        s2[:, k] = c + alpha * eps2 + beta * s2[:, k - 1]
    avg = s2.mean(axis=1)                              # (1/h) sum over horizon, per path
    return {"mc_mean": float(avg.mean()), "mc_se": float(avg.std(ddof=1) / np.sqrt(n_paths))}

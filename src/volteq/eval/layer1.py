"""
Layer 1 forecast-accuracy machinery (SPEC.md Section 10.1).

Loss functions QLIKE and MSE (both robust to a noisy volatility proxy; Patton
2011). Pairwise Diebold-Mariano with HAC (Newey-West) standard errors and
reported loss-differential autocorrelation. Hansen-Lunde-Nason (2011) Model
Confidence Set via the stationary bootstrap (Politis-Romano 1994). No MAE, no R^2
on volatility.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ losses
def qlike(rv: np.ndarray, vhat: np.ndarray) -> np.ndarray:
    """Robust QLIKE loss (Patton 2011): RV/V - log(RV/V) - 1 >= 0, min at V=RV.
    Undefined for non-positive forecasts."""
    rv = np.asarray(rv, float); vhat = np.asarray(vhat, float)
    if np.any(vhat <= 0):
        raise ValueError("QLIKE undefined for non-positive forecasts")
    r = rv / vhat
    return r - np.log(r) - 1.0


def mse(rv: np.ndarray, vhat: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, float); vhat = np.asarray(vhat, float)
    return (rv - vhat) ** 2


# ------------------------------------------------------------------ HAC / DM
def nw_auto_lag(n: int) -> int:
    """Newey-West automatic bandwidth floor(4 (n/100)^(2/9))."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_var_mean(d: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett) HAC variance of the sample mean of series d."""
    d = np.asarray(d, float); n = len(d)
    dc = d - d.mean()
    g0 = np.dot(dc, dc) / n
    s = g0
    for k in range(1, lag + 1):
        gk = np.dot(dc[k:], dc[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return s / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None):
    """DM test on the loss differential d = loss_a - loss_b. Returns dict with the
    statistic, two-sided p-value, lag used, and acf(1) of d."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if lag is None:
        lag = nw_auto_lag(n)
    var = hac_var_mean(d, lag)
    dm = d.mean() / np.sqrt(var) if var > 0 else np.nan
    dc = d - d.mean()
    acf1 = float(np.dot(dc[1:], dc[:-1]) / np.dot(dc, dc)) if n > 1 else np.nan
    return {"dm": float(dm), "p": float(2 * (1 - norm.cdf(abs(dm)))),
            "lag": int(lag), "acf1": acf1, "mean_diff": float(d.mean())}


# ------------------------------------------------------------------ MCS
def stationary_bootstrap_idx(T: int, B: int, block: int, rng: np.random.RandomState):
    """B x T index matrix from the stationary bootstrap (mean block = `block`)."""
    p = 1.0 / block
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.randint(0, T, B)
    restart = rng.random_sample((B, T)) < p
    fresh = rng.randint(0, T, (B, T))
    for t in range(1, T):
        cont = (idx[:, t - 1] + 1) % T
        idx[:, t] = np.where(restart[:, t], fresh[:, t], cont)
    return idx


def model_confidence_set(losses: np.ndarray, names, alpha: float = 0.10,
                         B: int = 10000, block: int = 5, seed: int = 0):
    """Hansen-Lunde-Nason (2011) MCS with the T_max statistic and the stationary
    bootstrap. Returns retained set at `alpha`, MCS p-value per model, and the
    elimination order. `losses` is T x M (rows = obs, cols = models)."""
    L = np.asarray(losses, float)
    T, M = L.shape
    rng = np.random.RandomState(seed)
    boot = stationary_bootstrap_idx(T, B, block, rng)          # B x T, shared

    surviving = list(range(M))
    elim_order = []
    mcs_p = {}
    running = 0.0
    while len(surviving) > 1:
        Ls = L[:, surviving]                                    # T x m
        m = len(surviving)
        Lbar_t = Ls.mean(axis=1, keepdims=True)                 # T x 1
        d = Ls - Lbar_t                                         # loss - cross-sectional mean
        dbar = d.mean(axis=0)                                   # m
        # bootstrap dbar and its variance
        db = d[boot, :].mean(axis=1)                            # B x m (mean over resampled t)
        var = ((db - dbar) ** 2).mean(axis=0)                   # m, bootstrap var of dbar
        var = np.where(var <= 0, np.inf, var)
        t_i = dbar / np.sqrt(var)                               # m t-stats
        Tmax = t_i.max()
        Tmax_b = ((db - dbar) / np.sqrt(var)).max(axis=1)       # B, null distribution
        pval = float((Tmax_b >= Tmax).mean())
        running = max(running, pval)
        worst = surviving[int(np.argmax(t_i))]
        mcs_p[names[worst]] = running
        elim_order.append(names[worst])
        if pval >= alpha:
            break
        surviving.remove(worst)
    for s in surviving:                                        # survivors: p = 1
        mcs_p[names[s]] = 1.0
    # the MCS at level alpha keeps every model whose MCS p-value >= alpha
    retained = [n for n in names if mcs_p.get(n, 0.0) >= alpha]
    return {"retained": retained, "mcs_p": mcs_p, "elim_order": elim_order,
            "block": block, "B": B}

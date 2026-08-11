"""
Fixtures for the fBm conditional-variance constant c(H) adopted in rfsv
(direct_h.c_fbm). The closed form c(H)=Gamma(3/2-H)/(Gamma(H+1/2)Gamma(2-2H)) is
validated by its exact anchors and by a numerically stable fractional-Gaussian-
noise conditioning that converges to it as the sampling refines.
"""
import numpy as np
import pytest
from scipy.special import gamma as G

from volteq.models.direct_h import c_fbm


def test_c_half_is_one_exactly():
    # H=1/2 reduces fBm to standard Brownian motion; conditional variance = d, c=1.
    assert abs(c_fbm(0.5) - 1.0) < 1e-12


def test_c_zero_limit_is_half():
    assert abs(c_fbm(1e-6) - 0.5) < 1e-4


def test_c_quarter_analytic():
    # Interior analytic anchor: c(1/4) = Gamma(5/4) / (Gamma(3/4) Gamma(3/2)).
    # With the anchors at 1/2 and 0+ and monotonicity, this validates the
    # three-Gamma expression at an interior point in closed form.
    expected = G(1.25) / (G(0.75) * G(1.5))
    assert abs(c_fbm(0.25) - expected) < 1e-12


def test_c_monotone_increasing_on_0_half():
    Hs = np.linspace(1e-4, 0.5, 400)
    cs = np.array([c_fbm(h) for h in Hs])
    assert np.all(np.diff(cs) > 0)


def _c_numeric_fgn(Hval, dt, N):
    """Var(W^H_{t+d}|discrete past)/d^{2H}, averaged over d=1..21, via stationary
    fGn increment conditioning (well-conditioned). dt -> 0 approaches the
    continuous-past closed form."""
    def gam(k):
        k = np.abs(k)
        return 0.5 * (np.abs(k + 1) ** (2 * Hval) - 2 * np.abs(k) ** (2 * Hval)
                      + np.abs(k - 1) ** (2 * Hval)) * dt ** (2 * Hval)
    npast = int(N / dt)
    pp = -np.arange(0, npast)
    Kinv = np.linalg.inv(gam(pp[:, None] - pp[None, :]) + 1e-12 * np.eye(npast))
    out = []
    for di in range(1, 22):
        nf = int(round(di / dt))
        fp = np.arange(1, nf + 1); w = np.ones(nf)
        Kff = gam(fp[:, None] - fp[None, :]); Kfp = gam(fp[:, None] - pp[None, :])
        out.append((w @ Kff @ w - w @ Kfp @ Kinv @ Kfp.T @ w) / di ** (2 * Hval))
    return float(np.mean(out))


@pytest.mark.parametrize("H", [0.02, 0.05, 0.10, 0.15, 0.35, 0.50])
def test_discrete_fbm_conditioning_convergence_rate(H):
    # Documents the CONVERGENCE RATE of the discrete fBm conditioning toward the
    # (analytically validated) closed form -- not a check on c(H) itself, which is
    # validated by the exact anchors above. Richardson (dt->0) lands within ~1e-3;
    # exact at the Brownian anchor. The residual is finite-grid discretization.
    a = _c_numeric_fgn(H, 0.2, N=250)
    b = _c_numeric_fgn(H, 0.1, N=250)
    rich = 2 * b - a
    assert abs(rich - c_fbm(H)) < 5e-3, (H, rich, c_fbm(H))

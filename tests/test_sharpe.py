"""Unit tests for the Layer-2 Sharpe inference (src/volteq/eval/sharpe.py)."""
import numpy as np
import pytest

from volteq.eval.sharpe import (
    nw_auto_lag,
    ledoit_wolf_sharpe_test,
    deflated_sharpe,
)


def test_nw_auto_lag_known_values():
    # floor(4*(n/100)^(2/9))
    assert nw_auto_lag(100) == 4
    assert nw_auto_lag(6688) == 10


def test_lw_self_comparison_is_null():
    """A series tested against itself has zero Sharpe difference and p=1."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 4000)
    out = ledoit_wolf_sharpe_test(r, r, lag=8)
    assert abs(out["diff"]) < 1e-12
    assert abs(out["z"]) < 1e-8
    assert out["p"] == pytest.approx(1.0, abs=1e-8)


def test_lw_antisymmetry():
    """diff and z flip sign when the two arguments are swapped; p is unchanged."""
    rng = np.random.default_rng(1)
    ri = rng.normal(0.0007, 0.011, 5000)
    rj = rng.normal(0.0003, 0.009, 5000)
    a = ledoit_wolf_sharpe_test(ri, rj, lag=6)
    b = ledoit_wolf_sharpe_test(rj, ri, lag=6)
    assert a["diff"] == pytest.approx(-b["diff"], rel=1e-10)
    assert a["z"] == pytest.approx(-b["z"], rel=1e-8)
    assert a["p"] == pytest.approx(b["p"], rel=1e-8)


def test_lw_detects_a_real_difference():
    """A clearly higher-Sharpe series is flagged significant against a zero-mean one."""
    rng = np.random.default_rng(2)
    n = 8000
    hi = rng.normal(0.0010, 0.008, n)   # high Sharpe
    lo = rng.normal(0.0000, 0.008, n)   # ~zero Sharpe
    out = ledoit_wolf_sharpe_test(hi, lo)
    assert out["diff"] > 0
    assert out["p"] < 0.01


def test_dsr_decreases_with_more_trials():
    """More candidate trials raise the deflation hurdle SR0, lowering the DSR."""
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, 5000)
    d_few = deflated_sharpe(r, n_trials=2, sr_variance=1e-6)
    d_many = deflated_sharpe(r, n_trials=200, sr_variance=1e-6)
    assert d_few["sr0"] < d_many["sr0"]
    assert d_few["dsr"] > d_many["dsr"]


def test_dsr_increases_with_strategy_sharpe():
    """A higher realized Sharpe yields a higher DSR at fixed trial count/variance."""
    rng = np.random.default_rng(4)
    weak = rng.normal(0.0002, 0.01, 5000)
    strong = rng.normal(0.0009, 0.01, 5000)
    dw = deflated_sharpe(weak, n_trials=19, sr_variance=1e-6)
    ds = deflated_sharpe(strong, n_trials=19, sr_variance=1e-6)
    assert ds["sr_daily"] > dw["sr_daily"]
    assert ds["dsr"] > dw["dsr"]
    assert 0.0 <= dw["dsr"] <= 1.0 and 0.0 <= ds["dsr"] <= 1.0

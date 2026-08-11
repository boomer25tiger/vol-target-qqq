"""Layer 1 machinery fixtures: hand-computed losses, a known-answer DM, and the
MCS returning the full set when all models are identical."""
import numpy as np
import pytest

from volteq.eval.layer1 import (qlike, mse, diebold_mariano, hac_var_mean,
                                 model_confidence_set)


def test_mse_hand_computed():
    rv = np.array([1.0, 2.0, 3.0]); vhat = np.array([1.5, 1.5, 1.5])
    assert np.allclose(mse(rv, vhat), [0.25, 0.25, 2.25])


def test_qlike_hand_computed():
    # RV=2, V=1 -> 2 - ln2 - 1 = 1 - ln2; RV=1,V=1 -> 0 (minimum at V=RV)
    rv = np.array([2.0, 1.0]); vhat = np.array([1.0, 1.0])
    assert np.allclose(qlike(rv, vhat), [1.0 - np.log(2.0), 0.0])
    assert qlike(np.array([1.0]), np.array([1.0]))[0] == 0.0


def test_qlike_rejects_nonpositive_forecast():
    with pytest.raises(ValueError):
        qlike(np.array([1.0]), np.array([0.0]))


def test_dm_known_answer_lag0():
    # loss_a - loss_b = [2,0,2,0]: mean 1, population var 1, HAC(lag0) var = 1/4,
    # DM = 1 / sqrt(1/4) = 2.0
    a = np.array([2.0, 0.0, 2.0, 0.0]); b = np.zeros(4)
    r = diebold_mariano(a, b, lag=0)
    assert r["dm"] == pytest.approx(2.0, rel=1e-12)
    assert hac_var_mean(a - b, 0) == pytest.approx(0.25, rel=1e-12)


def test_mcs_returns_full_set_when_identical():
    rng = np.random.RandomState(1)
    base = rng.random(300)
    losses = np.column_stack([base, base, base, base])   # 4 identical models
    res = model_confidence_set(losses, ["a", "b", "c", "d"], alpha=0.10,
                               B=500, block=5, seed=7)
    assert set(res["retained"]) == {"a", "b", "c", "d"}


def test_mcs_eliminates_a_dominated_model():
    rng = np.random.RandomState(2)
    T = 400
    good = rng.random(T) * 0.1            # small loss
    bad = good + 0.5 + rng.random(T) * 0.05  # strictly, clearly worse
    losses = np.column_stack([good, good + 1e-4 * rng.random(T), bad])
    res = model_confidence_set(losses, ["g1", "g2", "bad"], alpha=0.10,
                               B=1000, block=5, seed=3)
    assert "bad" not in res["retained"]
    assert "g1" in res["retained"]

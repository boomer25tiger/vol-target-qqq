"""
No-look-ahead invariant: a trailing measure dated t must be identical whether it
is computed from the full history or from history truncated at t. If any measure
peeked at t+1.., truncation would change its value at t.
"""
import os
import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config, repo_root
from volteq.rv.estimators import daily_proxy_rv, yang_zhang


def _qqq():
    df = pd.read_parquet(os.path.join(repo_root(), "data", "raw", "qqq_daily.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()[["open", "high", "low", "close"]]


@pytest.fixture(scope="module")
def qqq():
    return _qqq()


def test_daily_proxy_is_causal(qqq):
    n = int(load_config()["frozen"]["forecast_horizon_days"])
    full = daily_proxy_rv(qqq)["rv_daily"]
    # test at several interior dates spread across the sample
    idx = qqq.index
    for pos in [500, 2000, 4000, len(idx) - 1]:
        t = idx[pos]
        trunc = daily_proxy_rv(qqq.loc[:t])["rv_daily"].iloc[-1]
        assert np.isfinite(full.loc[t]) and np.isfinite(trunc)
        assert full.loc[t] == pytest.approx(trunc, rel=1e-12, abs=1e-15), t


def test_yang_zhang_is_causal(qqq):
    n = int(load_config()["frozen"]["forecast_horizon_days"])
    full = yang_zhang(qqq, n)
    idx = qqq.index
    for pos in [500, 2000, 4000, len(idx) - 1]:
        t = idx[pos]
        trunc = yang_zhang(qqq.loc[:t], n).iloc[-1]
        assert np.isfinite(full.loc[t]) and np.isfinite(trunc)
        assert full.loc[t] == pytest.approx(trunc, rel=1e-12, abs=1e-15), t


def test_adding_future_rows_never_changes_past(qqq):
    """Appending later rows must not perturb any earlier trailing value."""
    n = int(load_config()["frozen"]["forecast_horizon_days"])
    cut = qqq.index[3000]
    past = yang_zhang(qqq.loc[:cut], n)
    full = yang_zhang(qqq, n).loc[:cut]
    aligned = pd.concat([past.rename("past"), full.rename("full")], axis=1).dropna()
    assert len(aligned) > 100
    assert np.allclose(aligned["past"], aligned["full"], rtol=1e-12, atol=1e-15)

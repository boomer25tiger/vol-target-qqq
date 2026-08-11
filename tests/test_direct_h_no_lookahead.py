"""
No-look-ahead + isolation for the direct-h models (rv, har, rfsv). Each is
deterministic, so its cached V_t(21) must be reproduced exactly by recomputing on
rv_daily[:t], and must be invariant to appending future rows.
"""
import os
import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config, repo_root
from volteq.rv.panel import load_panel
from volteq.models.direct_h import rv_forecast, har_fit_forecast, rfsv_fit_forecast

CFG = load_config()
H = int(CFG["frozen"]["forecast_horizon_days"])


@pytest.fixture(scope="module")
def rv():
    return load_panel()["rv_daily"]


def _cached_v21():
    p = os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet")
    df = pd.read_parquet(p); df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def test_direct_h_reproduce_cached_panel(rv):
    cached = _cached_v21()
    for t in cached.index[[10, 120, len(cached) - 2]]:  # -2: last has forward NaN but V exists
        r = rv.loc[:t].to_numpy()
        assert rv_forecast(r, H) == pytest.approx(cached.loc[t, "rv"], rel=1e-9)
        assert har_fit_forecast(r, H)["V21"] == pytest.approx(cached.loc[t, "har"], rel=1e-9)
        assert rfsv_fit_forecast(r, H)["V21"] == pytest.approx(cached.loc[t, "rfsv"], rel=1e-9)


def test_direct_h_invariant_to_future(rv):
    idx = rv.index
    t = idx[4000]
    r_now = rv.loc[:t].to_numpy()
    r_more = rv.iloc[: idx.get_loc(t) + 400].loc[:t].to_numpy()
    assert np.array_equal(r_now, r_more)
    for fn in (lambda x: har_fit_forecast(x, H)["V21"],
               lambda x: rfsv_fit_forecast(x, H)["V21"],
               lambda x: rv_forecast(x, H)):
        assert fn(r_now) == pytest.approx(fn(r_more), rel=1e-12)


def test_direct_h_module_has_no_forward_target():
    import volteq.models.direct_h as m
    src = open(m.__file__).read()
    assert "forward_target" not in src

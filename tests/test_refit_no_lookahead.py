"""
No-look-ahead on the refit loop: each date's forecast must be a pure function of
data through that date. Two checks:
  (1) the deterministic models' cached V_t(21) is reproduced exactly by refitting
      on returns[:t] alone;
  (2) a fit is invariant to appending future returns and slicing back to t.
"""
import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config
from volteq.models.returns import build_garch_returns
from volteq.models.garch_family import fit_vt_garch, fit_vt_gjr, fit_ewma
from volteq.forecast.aggregate import v_closed_form, v_identity

CFG = load_config()
H = int(CFG["frozen"]["forecast_horizon_days"])


@pytest.fixture(scope="module")
def rets():
    return build_garch_returns(CFG)


def _v21_cached():
    import os
    from volteq.config import repo_root
    p = os.path.join(repo_root(), "data", "processed", "forecast_v21.parquet")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def test_deterministic_models_reproduce_cached_panel(rets):
    cached = _v21_cached()
    # sample a few rebalance dates across the sample
    sample = cached.index[[10, 100, 200, len(cached) - 1]]
    for t in sample:
        r = rets.loc[:t, "ret"].to_numpy()
        g = fit_vt_garch(r, "skewt")
        vg = v_closed_form(g["sig2bar"], g["sigma2_next"], g["persistence"], H)
        j = fit_vt_gjr(r, "skewt")
        vj = v_closed_form(j["sig2bar"], j["sigma2_next"], j["persistence"], H)
        e = fit_ewma(r)
        ve = v_identity(e["sigma2_next"])
        assert vg == pytest.approx(cached.loc[t, "garch_skewt"], rel=1e-9)
        assert vj == pytest.approx(cached.loc[t, "gjr_skewt"], rel=1e-9)
        assert ve == pytest.approx(cached.loc[t, "ewma"], rel=1e-9)


def test_fit_invariant_to_future_rows(rets):
    idx = rets.index
    t = idx[3000]
    r_now = rets.loc[:t, "ret"].to_numpy()
    # a strictly longer window, sliced back to t, must give the identical array/fit
    r_more = rets.iloc[: idx.get_loc(t) + 500].loc[:t, "ret"].to_numpy()
    assert np.array_equal(r_now, r_more)
    a = fit_vt_garch(r_now, "skewt")
    b = fit_vt_garch(r_more, "skewt")
    assert a["sigma2_next"] == pytest.approx(b["sigma2_next"], rel=1e-12)
    assert a["persistence"] == pytest.approx(b["persistence"], rel=1e-12)

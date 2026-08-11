"""
No-look-ahead for the egarch daily secondary forecasts. The Monte Carlo forecast
at origin d starts from the filtered conditional variance at d; that state (and
hence the forecast) must be invariant to returns after d. We check the
deterministic state - the conditional volatility at d under the frozen parameters
- is identical whether the model is filtered on data through d or through d+40.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from volteq.config import load_config, repo_root
from volteq.models.returns import build_garch_returns


@pytest.mark.parametrize("dpos", [4000, 5000])
def test_egarch_conditioning_state_invariant_to_future(dpos):
    from arch import arch_model
    cfg = load_config()
    r_all = build_garch_returns(cfg)["ret"] * 100.0
    pp = pd.read_parquet(os.path.join(repo_root(), "data", "processed", "param_paths.parquet"))
    e = pp[(pp.model == "egarch") & (pp.dist == "skewt")].copy()
    e["date"] = pd.to_datetime(e["date"]); e = e.set_index("date").sort_index()
    d = r_all.index[dpos]
    r_gov = e.index[e.index <= d][-1]
    params = list(json.loads(e.loc[r_gov, "params_json"]).values())

    def cvol_at_d(end_pos):
        r = r_all.iloc[: end_pos + 1]
        res = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewt").fix(params)
        return float(res.conditional_volatility.loc[d])

    now = cvol_at_d(dpos)
    future = cvol_at_d(dpos + 40)          # append 40 future returns
    assert now == pytest.approx(future, rel=1e-10)

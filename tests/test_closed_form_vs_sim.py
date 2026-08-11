"""
Closed-form vs simulation agreement for the h=21 aggregation. Simulating the
fitted GARCH forward and averaging (1/h) sum sigma^2_{t+k} must match the
closed-form V_t(h) within Monte Carlo error.
"""
import numpy as np
import pytest

from volteq.config import load_config
from volteq.models.returns import build_garch_returns
from volteq.models.garch_family import fit_vt_garch
from volteq.forecast.aggregate import v_closed_form, simulate_garch_avg_var

CFG = load_config()
H = int(CFG["frozen"]["forecast_horizon_days"])
SEED = int(CFG["meta"]["random_seed"])


def test_closed_form_matches_simulation():
    rets = build_garch_returns(CFG)
    # a couple of dates: one high-vol (2000), one calm (2017)
    for tstr in ["2000-01-31", "2017-06-30"]:
        r = rets.loc[:tstr, "ret"].to_numpy()
        f = fit_vt_garch(r, "normal")
        cf = v_closed_form(f["sig2bar"], f["sigma2_next"], f["persistence"], H)
        sim = simulate_garch_avg_var(f["sig2bar"], f["sigma2_next"],
                                     f["params"]["alpha"], f["params"]["beta"],
                                     H, n_paths=300_000, seed=SEED)
        z = abs(sim["mc_mean"] - cf) / sim["mc_se"]
        assert z < 5.0, f"{tstr}: |mc-cf|={abs(sim['mc_mean']-cf):.3e}, {z:.1f} SE"
        assert abs(sim["mc_mean"] - cf) / cf < 0.01, tstr

"""
FORWARD-LOOKING realized-variance target - EVALUATION USE ONLY.

============================  LOOK-AHEAD LIVES HERE  ============================
Every function in this module uses information from sessions AFTER date t. The
values it returns are NOT computable at the close of t and MUST NOT feed any
model fit, forecast, sizing, or backtest decision. This module exists so that the
one legitimate use of forward information - scoring a forecast against what was
actually realized - is quarantined in a single, obviously-named place.

Nothing under src/volteq/models/, src/volteq/forecast/, or src/volteq/backtest/
may import this module. tests/test_eval_target_isolation.py enforces that.
================================================================================

The forecast object each model produces is V_t(h) = E_t[(1/h) Σ_{k=1..h} σ²_{t+k}]
(SPEC.md Section 6). Its realized counterpart over the holding period is what these
functions return.
"""
from __future__ import annotations

import pandas as pd

# A sentinel other modules can assert against; see the isolation test.
EVALUATION_ONLY = True


def forward_realized_variance_avg(rv_daily: pd.Series, h: int) -> pd.Series:
    """Average daily realized variance over the forward window [t+1, t+h].

    fwd_t = (1/h) * Σ_{k=1..h} rv_daily_{t+k}.  Uses future data by design.
    """
    trailing_mean = rv_daily.rolling(h, min_periods=h).mean()   # mean over [s-h+1, s]
    return trailing_mean.shift(-h).rename(f"fwd_rv_avg_{h}")     # place window [t+1,t+h] at t


def forward_realized_variance_yz(yz_trailing: pd.Series, h: int) -> pd.Series:
    """Yang-Zhang variance over the forward window [t+1, t+h].

    Reuses the trailing YZ over an h-day window: the window ending at t+h is the
    forward window [t+1, t+h]. Uses future data by design.
    """
    return yz_trailing.shift(-h).rename(f"fwd_yz_{h}")

"""
Daily realized-variance estimators for the fallback (daily-frequency) path.

All estimators here are causal: the value dated ``t`` uses only OHLC information
from sessions up to and including ``t``. The rolling Yang-Zhang window looks back
``n`` sessions; it never looks forward. Forward-looking realized variance lives in
``forward_target.py`` and nowhere else.

References
----------
Rogers, L.C.G. and S.E. Satchell (1991). Estimating variance from high, low and
closing prices. Annals of Applied Probability 1(4), 504-512.
Yang, D. and Q. Zhang (2000). Drift-independent volatility estimation based on
high, low, open, and close prices. Journal of Business 73(3), 477-491.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


def _require_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLC frame index must be sorted ascending by date")


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """Per-session Rogers-Satchell variance (drift-independent, open-to-close).

    RS_t = ln(H/O)*(ln(H/O) - ln(C/O)) + ln(L/O)*(ln(L/O) - ln(C/O))
         = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

    Uses only day t's OHLC - no overnight term.
    """
    _require_ohlc(df)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ho = np.log(h / o)
    lo = np.log(l / o)
    co = np.log(c / o)
    return (ho * (ho - co) + lo * (lo - co)).rename("rs")


def overnight_log_return(df: pd.DataFrame) -> pd.Series:
    """Overnight log return ln(O_t / C_{t-1}). NaN on the first session."""
    _require_ohlc(df)
    return np.log(df["open"] / df["close"].shift(1)).rename("ret_on")


def daily_proxy_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback daily variance proxy: squared overnight return + Rogers-Satchell.

    Returns a frame with columns ``overnight_var``, ``rs``, ``rv_daily`` where
    ``rv_daily = overnight_var + rs``. The first session's overnight term is NaN
    (no prior close); ``rv_daily`` there falls back to ``rs`` alone so the series
    has no leading gap, and the substitution is recorded in ``overnight_is_na``.
    """
    _require_ohlc(df)
    rs = rogers_satchell(df)
    on = overnight_log_return(df)
    overnight_var = (on ** 2).rename("overnight_var")
    rv = overnight_var + rs
    on_na = overnight_var.isna()
    rv = rv.where(~on_na, rs)  # first session: RS only
    out = pd.DataFrame({
        "overnight_var": overnight_var,
        "rs": rs,
        "rv_daily": rv.rename("rv_daily"),
        "overnight_is_na": on_na,
    })
    return out


def yang_zhang(df: pd.DataFrame, n: int) -> pd.Series:
    """Rolling Yang-Zhang variance over a trailing window of ``n`` sessions.

    σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS,
        k = 0.34 / (1.34 + (n+1)/(n-1)).

    The value dated t is computed from sessions [t-n+1, t] (plus C_{t-1} for the
    first overnight in the window). Trailing only; requires a full window
    (min_periods = n), so the first n-1 values are NaN.
    """
    _require_ohlc(df)
    if n < 2:
        raise ValueError("Yang-Zhang window must be >= 2")
    o = overnight_log_return(df)                      # ln(O_t / C_{t-1})
    c = np.log(df["close"] / df["open"]).rename("oc")  # open-to-close
    rs = rogers_satchell(df)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_on = o.rolling(n, min_periods=n).var(ddof=1)
    var_oc = c.rolling(n, min_periods=n).var(ddof=1)
    mean_rs = rs.rolling(n, min_periods=n).mean()
    return (var_on + k * var_oc + (1 - k) * mean_rs).rename(f"yz_{n}")


def annualized_vol_from_daily_var(daily_var, periods: int = 252) -> float:
    """sqrt(mean(daily variance) * periods)."""
    m = float(np.nanmean(np.asarray(daily_var, dtype=float)))
    return float(np.sqrt(m * periods))

"""
Monthly rebalance calendar: the last NYSE trading day of each month, from the
evaluation start through the last complete month. Refits happen on these dates,
using data through the date only.
"""
from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

from volteq.config import load_config


def rebalance_dates(cfg: dict | None = None, data_end: str | pd.Timestamp | None = None,
                    include_warmup: bool = False):
    """Last trading day of each month in [eval_start, last complete month <= data_end].

    include_warmup prepends the month-end immediately before eval_start (the date
    whose forecast sizes the position held on eval_start), so a backtest can begin
    its equity curve at eval_start rather than one month in.
    """
    cfg = cfg or load_config()
    eval_start = pd.Timestamp(cfg["frozen"]["eval_start"])
    nyse = mcal.get_calendar("NYSE")
    end = pd.Timestamp(data_end) if data_end is not None else pd.Timestamp.today().normalize()

    start = eval_start
    if include_warmup:
        prev_month_end = (eval_start.to_period("M") - 1).end_time
        wu = nyse.valid_days(start_date=prev_month_end - pd.Timedelta(days=10),
                             end_date=eval_start - pd.Timedelta(days=1))
        start = (wu[0].tz_localize(None) if wu[0].tzinfo else wu[0])

    days = nyse.valid_days(start_date=start, end_date=end)
    days = pd.DatetimeIndex([d.tz_localize(None) if d.tzinfo else d for d in days])
    s = pd.Series(days, index=days)
    # last trading day within each calendar month
    month_ends = s.groupby(s.dt.to_period("M")).max()
    # drop an incomplete trailing month: keep a month only if its true calendar
    # month-end trading day is <= data_end (i.e., the month is fully observed)
    keep = []
    for period, last_td in month_ends.items():
        m_hi = nyse.valid_days(start_date=period.start_time, end_date=period.end_time)
        m_hi = m_hi[-1].tz_localize(None) if m_hi[-1].tzinfo else m_hi[-1]
        if last_td >= m_hi:               # observed through the real month-end
            keep.append(last_td)
    return pd.DatetimeIndex(sorted(keep))

"""
Session-count reconciliation: the realized-measure panel's trading days must
match the NYSE calendar year by year (no missing or spurious sessions beyond a
small tolerance), separately for the IXIC seed era and the QQQ traded era.
"""
import pandas as pd
import pandas_market_calendars as mcal

from volteq.rv.panel import load_panel

TOL = 2  # sessions per year


def _flagged(dates):
    nyse = mcal.get_calendar("NYSE")
    out = []
    lo, hi = dates.min(), dates.max()
    for y in range(lo.year, hi.year + 1):
        y_lo = max(pd.Timestamp(f"{y}-01-01"), lo)
        y_hi = min(pd.Timestamp(f"{y}-12-31"), hi)
        cal = len(nyse.valid_days(start_date=y_lo, end_date=y_hi))
        have = int(((dates >= y_lo) & (dates <= y_hi)).sum())
        if abs(cal - have) > TOL:
            out.append((y, cal, have))
    return out


def test_qqq_era_matches_nyse():
    panel = load_panel()
    qqq_dates = panel.index[panel["source"] == "qqq"]
    flagged = _flagged(qqq_dates)
    assert not flagged, f"QQQ-era years off NYSE by > {TOL}: {flagged}"


def test_seed_era_matches_nyse():
    panel = load_panel()
    seed_dates = panel.index[panel["source"] == "ixic_seed"]
    flagged = _flagged(seed_dates)
    assert not flagged, f"IXIC-seed years off NYSE by > {TOL}: {flagged}"

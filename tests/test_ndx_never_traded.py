"""
NDX-never-traded invariant.

The estimation seed (NDX for the GARCH family, IXIC for the RV family) is burn-in
only and must never enter the traded series. The traded era begins with QQQ on
1999-03-10. This test enforces that no seed-sourced observation appears on or
after the takeover date in the realized-measure panel, and that the config marks
both seed sources estimation-only.
"""
import pandas as pd

from volteq.config import load_config
from volteq.rv.panel import load_panel
from volteq.rv.seed import SEED_TAKEOVER


def test_no_seed_rows_in_traded_era():
    panel = load_panel()
    takeover = pd.Timestamp(SEED_TAKEOVER)
    seed_rows = panel[panel["source"] != "qqq"]
    assert (seed_rows.index < takeover).all(), "seed data leaked into the traded era"
    traded_rows = panel[panel.index >= takeover]
    assert (traded_rows["source"] == "qqq").all(), "traded era must be QQQ only"


def test_seed_symbols_are_estimation_only_in_config():
    cfg = load_config()
    assert "only" in cfg["data"]["seed_source"]["purpose"]         # NDX
    assert "only" in cfg["data"]["range_seed_source"]["purpose"]   # IXIC
    # the RV panel seed is labelled ixic_seed, never a traded symbol
    panel = load_panel()
    assert set(panel["source"].unique()) <= {"ixic_seed", "qqq"}


def test_takeover_is_qqq_first_day():
    cfg = load_config()
    # QQQ (the only traded symbol) begins 1999-03-10 per the daily source
    assert cfg["data"]["daily_source"]["start"] == "1999-03-10"
    assert SEED_TAKEOVER == "1999-03-10"

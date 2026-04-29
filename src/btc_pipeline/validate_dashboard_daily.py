#!/usr/bin/env python3
"""
Daily-workflow dashboard wrapper with isolated output files.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.btc_pipeline import validate_dashboard as dashboard
from src.btc_pipeline.path_config import (
    DAILY_DASHBOARD_MARKET_HOURS_PATH,
    DAILY_DASHBOARD_MARKET_HOURS_REVERSE_PATH,
    DAILY_DASHBOARD_PATH,
    DAILY_DASHBOARD_REVERSE_PATH,
    DAILY_HISTORY_PATH,
    DAILY_LAST_PREDICTION_PATH,
)


dashboard.HISTORY_PATH = DAILY_HISTORY_PATH
dashboard.DASHBOARD_PATH = DAILY_DASHBOARD_PATH
dashboard.LOCAL_LAST_PREDICTION_PATH = DAILY_LAST_PREDICTION_PATH
dashboard.DASHBOARD_TITLE = "BTC Daily Model Validation Dashboard"
dashboard.DASHBOARD_VARIANTS = [
    {
        "path": DAILY_DASHBOARD_PATH,
        "title": "BTC Daily Model Validation Dashboard",
        "subtitle": "",
        "signal_mode": "normal",
        "filter_mode": "all",
    },
    {
        "path": DAILY_DASHBOARD_REVERSE_PATH,
        "title": "BTC Daily Model Reverse Dashboard",
        "subtitle": "Reverse actions from the same hourly predictions",
        "signal_mode": "reverse",
        "filter_mode": "all",
    },
    {
        "path": DAILY_DASHBOARD_MARKET_HOURS_PATH,
        "title": "BTC Daily Model Market Hours Dashboard",
        "subtitle": "Filtered to ET market-hours target candles only",
        "signal_mode": "normal",
        "filter_mode": "market_hours",
    },
    {
        "path": DAILY_DASHBOARD_MARKET_HOURS_REVERSE_PATH,
        "title": "BTC Daily Model Reverse Market Hours Dashboard",
        "subtitle": "Reverse actions filtered to ET market-hours target candles",
        "signal_mode": "reverse",
        "filter_mode": "market_hours",
    },
]
dashboard.configure_tracking = lambda: (None, "", "")


if __name__ == "__main__":
    dashboard.main()

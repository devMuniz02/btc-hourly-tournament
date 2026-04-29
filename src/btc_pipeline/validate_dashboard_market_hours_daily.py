#!/usr/bin/env python3
"""
Validation dashboard wrapper for the isolated ET market-hours daily workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.btc_pipeline import validate_dashboard_market_hours as market_hours_dashboard
from src.btc_pipeline.path_config import (
    MARKET_HOURS_DAILY_DASHBOARD_PATH,
    MARKET_HOURS_DAILY_DASHBOARD_REVERSE_PATH,
    MARKET_HOURS_DAILY_HISTORY_PATH,
    MARKET_HOURS_DAILY_LAST_PREDICTION_PATH,
)


market_hours_dashboard.dashboard.HISTORY_PATH = MARKET_HOURS_DAILY_HISTORY_PATH
market_hours_dashboard.dashboard.DASHBOARD_PATH = MARKET_HOURS_DAILY_DASHBOARD_PATH
market_hours_dashboard.dashboard.LOCAL_LAST_PREDICTION_PATH = MARKET_HOURS_DAILY_LAST_PREDICTION_PATH
market_hours_dashboard.dashboard.DASHBOARD_TITLE = "BTC Market Hours Daily Model Dashboard"
market_hours_dashboard.dashboard.DASHBOARD_VARIANTS = [
    {
        "path": MARKET_HOURS_DAILY_DASHBOARD_PATH,
        "title": "BTC Market Hours Daily Model Dashboard",
        "subtitle": "",
        "signal_mode": "normal",
        "filter_mode": "market_hours",
    },
    {
        "path": MARKET_HOURS_DAILY_DASHBOARD_REVERSE_PATH,
        "title": "BTC Reverse Market Hours Daily Model Dashboard",
        "subtitle": "Reverse actions from the same market-hours daily predictions",
        "signal_mode": "reverse",
        "filter_mode": "market_hours",
    },
]
market_hours_dashboard.dashboard.configure_tracking = lambda: (None, "", "")


if __name__ == "__main__":
    market_hours_dashboard.dashboard.main()

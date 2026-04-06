#!/usr/bin/env python3
"""
Daily-workflow dashboard wrapper with isolated output files.
"""

from __future__ import annotations

from pathlib import Path

import validate_dashboard as dashboard


dashboard.HISTORY_PATH = Path("history_daily.csv")
dashboard.DASHBOARD_PATH = Path("assets/dailydashboard.png")
dashboard.LOCAL_LAST_PREDICTION_PATH = Path("last_prediction_daily.json")
dashboard.DASHBOARD_TITLE = "BTC Daily Model Validation Dashboard"
dashboard.DASHBOARD_VARIANTS = [
    {
        "path": Path("assets/dailydashboard.png"),
        "title": "BTC Daily Model Validation Dashboard",
        "subtitle": "",
        "signal_mode": "normal",
        "filter_mode": "all",
    },
    {
        "path": Path("assets/dailydashboard_reverse.png"),
        "title": "BTC Daily Model Reverse Dashboard",
        "subtitle": "Reverse actions from the same hourly predictions",
        "signal_mode": "reverse",
        "filter_mode": "all",
    },
    {
        "path": Path("assets/dailydashboard_market_hours.png"),
        "title": "BTC Daily Model Market Hours Dashboard",
        "subtitle": "Filtered to ET market-hours target candles only",
        "signal_mode": "normal",
        "filter_mode": "market_hours",
    },
    {
        "path": Path("assets/dailydashboard_market_hours_reverse.png"),
        "title": "BTC Daily Model Reverse Market Hours Dashboard",
        "subtitle": "Reverse actions filtered to ET market-hours target candles",
        "signal_mode": "reverse",
        "filter_mode": "market_hours",
    },
]
dashboard.configure_tracking = lambda: (None, "", "")


if __name__ == "__main__":
    dashboard.main()

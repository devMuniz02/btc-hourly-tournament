#!/usr/bin/env python3
"""
Validation dashboard wrapper for the isolated ET market-hours daily workflow.
"""

from __future__ import annotations

from pathlib import Path

import validate_dashboard_market_hours as market_hours_dashboard


market_hours_dashboard.dashboard.HISTORY_PATH = Path("history_market_hours_daily.csv")
market_hours_dashboard.dashboard.DASHBOARD_PATH = Path("assets/dashboard_market_hours_daily.png")
market_hours_dashboard.dashboard.LOCAL_LAST_PREDICTION_PATH = Path("last_prediction_market_hours_daily.json")
market_hours_dashboard.dashboard.configure_tracking = lambda: (None, "", "")


if __name__ == "__main__":
    market_hours_dashboard.dashboard.main()

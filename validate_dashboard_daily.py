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
dashboard.configure_tracking = lambda: (None, "", "")


if __name__ == "__main__":
    dashboard.main()

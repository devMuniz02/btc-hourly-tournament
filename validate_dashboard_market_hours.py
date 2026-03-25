#!/usr/bin/env python3
"""
Validation dashboard wrapper for the isolated ET market-hours hourly workflow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import validate_dashboard as dashboard
from market_hours_common import is_allowed_prediction_target_timestamp


dashboard.HISTORY_PATH = Path("history_market_hours.csv")
dashboard.DASHBOARD_PATH = Path("assets/dashboard_market_hours.png")
dashboard.LOCAL_LAST_PREDICTION_PATH = Path("last_prediction_market_hours.json")


def ensure_recent_history_slots_market_hours(
    history: pd.DataFrame,
    hours: int = 10,
) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_available_target = now_utc.floor("h") - pd.Timedelta(hours=1)
    expected_timestamps: list[pd.Timestamp] = []
    cursor = latest_available_target
    while len(expected_timestamps) < hours:
        if is_allowed_prediction_target_timestamp(cursor):
            expected_timestamps.append(cursor)
        cursor -= pd.Timedelta(hours=1)

    updated = history.copy()
    existing_timestamps = set()
    if not updated.empty:
        existing_timestamps = {
            pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
            for ts in updated["timestamp"]
        }

    missing_rows: list[dict[str, object]] = []
    for target_ts in expected_timestamps:
        if target_ts in existing_timestamps:
            continue
        missing_rows.append(
            {
                "timestamp": target_ts,
                "predicted": "",
                "actual": "",
                "result": "",
                "failed": 0,
                "status": "missing",
                "reference_open": pd.NA,
                "reference_close": pd.NA,
                "target_open": pd.NA,
                "target_close": pd.NA,
                "model_predictions": "",
                "best_champion_name": "",
                "best_champion_family": "",
                "best_champion_version": "",
                "workflow_name": "",
                "workflow_variant": "",
                "daily_model_refresh": pd.NA,
                "model_refresh_et_date": "",
                "prediction_generated_at": "",
            }
        )

    if not missing_rows:
        return updated

    missing_frame = dashboard.build_history_frame(missing_rows)
    if updated.empty:
        updated = missing_frame
    else:
        updated = pd.concat([dashboard.ensure_history_schema(updated), missing_frame], ignore_index=True)
    updated = updated.sort_values("timestamp").reset_index(drop=True)
    updated.to_csv(dashboard.HISTORY_PATH, index=False)
    return updated


dashboard.ensure_recent_history_slots = ensure_recent_history_slots_market_hours


if __name__ == "__main__":
    dashboard.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.btc_pipeline import validate_dashboard as dashboard


class DashboardHistoryTransformTests(unittest.TestCase):
    def test_skip_missing_slots_env_keeps_history_unchanged(self) -> None:
        history = dashboard.build_history_frame(
            [
                {
                    "timestamp": "2026-08-21T12:00:00+00:00",
                    "status": "validated",
                }
            ]
        )

        with patch.dict("os.environ", {"BTC_DASHBOARD_SKIP_MISSING_SLOTS": "1"}):
            updated = dashboard.ensure_recent_history_slots(history, hours=10)

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated.iloc[0]["timestamp"], history.iloc[0]["timestamp"])

    def test_reverse_dashboard_recomputes_text_label_results(self) -> None:
        history = dashboard.build_history_frame(
            [
                {
                    "timestamp": "2026-08-21T12:00:00+00:00",
                    "predicted": 1,
                    "actual": "DOWN",
                    "result": 0,
                    "failed": 0,
                    "status": "validated",
                    "model_predictions": "{}",
                }
            ]
        )

        transformed = dashboard.transform_history_for_dashboard(
            history,
            reverse=True,
            market_hours_only=False,
        )

        self.assertEqual(int(transformed.iloc[0]["predicted"]), 0)
        self.assertEqual(int(transformed.iloc[0]["result"]), 1)

    def test_text_actual_labels_are_chart_eligible(self) -> None:
        history = dashboard.build_history_frame(
            [
                {
                    "timestamp": "2026-08-21T12:00:00+00:00",
                    "predicted": 1,
                    "actual": "UP",
                    "result": 1,
                    "failed": 0,
                    "status": "validated",
                    "model_predictions": '{"xgb":{"predicted_label":1}}',
                }
            ]
        )
        scored = history.sort_values("timestamp").copy()
        scored["actual_label_for_chart"] = scored["actual"].apply(dashboard.parse_binary_label)
        scored = scored[
            (pd.to_numeric(scored["failed"], errors="coerce").fillna(0) == 0)
            & (scored["status"] != "missing")
            & scored["actual_label_for_chart"].notna()
        ]

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["actual_label_for_chart"], 1)


if __name__ == "__main__":
    unittest.main()

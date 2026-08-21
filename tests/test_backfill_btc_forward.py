from __future__ import annotations

import shutil
import unittest
from argparse import Namespace
from csv import DictWriter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.maintenance import backfill_btc_forward as backfill


ROOT = Path(__file__).resolve().parents[1]


class BackfillManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "temp" / f"backfill-tests-{self._testMethodName}"
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def write_history_rows(self, history_path: Path, rows: list[dict[str, object]]) -> None:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = DictWriter(handle, fieldnames=backfill.HISTORY_COLUMNS)
            writer.writeheader()
            for row in rows:
                payload = {column: "" for column in backfill.HISTORY_COLUMNS}
                payload.update(row)
                writer.writerow(payload)

    def test_manifest_starts_after_latest_non_missing_prediction(self) -> None:
        history_path = self.temp_root / "history.csv"
        self.write_history_rows(
            history_path,
            [
                {
                    "timestamp": "2026-04-28T00:00:00+00:00",
                    "predicted": 1,
                    "actual": "UP",
                    "result": 1,
                    "failed": 0,
                    "status": "validated",
                    "workflow_name": "hourly24",
                    "workflow_variant": "hourly_24h_prediction",
                },
                {
                    "timestamp": "2026-04-28T01:00:00+00:00",
                    "actual": "UP",
                    "failed": 0,
                    "status": "missing",
                    "workflow_name": "hourly24",
                    "workflow_variant": "hourly_24h_prediction",
                },
                {
                    "timestamp": "2026-04-28T03:00:00+00:00",
                    "predicted": 0,
                    "actual": "DOWN",
                    "result": 1,
                    "failed": 0,
                    "status": "validated",
                    "workflow_name": "hourly24",
                    "workflow_variant": "hourly_24h_prediction",
                },
            ],
        )
        variation = backfill.Variation(
            label="Test Hourly",
            history_path=history_path,
            last_prediction_path=None,
            registered_model_name="test-model",
            workflow_name="hourly24",
            workflow_variant="hourly_24h_prediction",
            daily_model_refresh=False,
        )
        args = Namespace(
            from_ts=None,
            to_ts="2026-04-28T05:00:00+00:00",
            only=None,
            newtest_latest_hours=None,
        )

        with patch.object(backfill, "VARIATIONS", (variation,)), patch.dict(
            backfill.OLD_NEW_SPLIT_UTC,
            {},
            clear=True,
        ):
            manifest = backfill.build_manifest(args)

        self.assertEqual(
            [target.isoformat() for target in manifest[variation]],
            [
                "2026-04-28T04:00:00+00:00",
                "2026-04-28T05:00:00+00:00",
            ],
        )

    def test_manifest_ignores_internal_missing_rows_before_latest_prediction(self) -> None:
        history_path = self.temp_root / "history.csv"
        self.write_history_rows(
            history_path,
            [
                {
                    "timestamp": "2026-04-28T00:00:00+00:00",
                    "predicted": 1,
                    "actual": "UP",
                    "result": 1,
                    "failed": 0,
                    "status": "validated",
                    "workflow_name": "daily-hourly",
                    "workflow_variant": "daily_model_hourly_prediction",
                },
                {
                    "timestamp": "2026-04-28T01:00:00+00:00",
                    "actual": "UP",
                    "failed": 0,
                    "status": "missing",
                    "workflow_name": "daily-hourly",
                    "workflow_variant": "daily_model_hourly_prediction",
                },
                {
                    "timestamp": "2026-04-28T02:00:00+00:00",
                    "predicted": 1,
                    "actual": "UP",
                    "result": 1,
                    "failed": 0,
                    "status": "validated",
                    "workflow_name": "daily-hourly",
                    "workflow_variant": "daily_model_hourly_prediction",
                },
            ],
        )
        variation = backfill.Variation(
            label="Test Daily",
            history_path=history_path,
            last_prediction_path=None,
            registered_model_name="test-daily-model",
            workflow_name="daily-hourly",
            workflow_variant="daily_model_hourly_prediction",
            daily_model_refresh=True,
        )
        args = Namespace(
            from_ts=None,
            to_ts="2026-04-28T03:00:00+00:00",
            only=None,
            newtest_latest_hours=None,
        )

        with patch.object(backfill, "VARIATIONS", (variation,)), patch.dict(
            backfill.OLD_NEW_SPLIT_UTC,
            {},
            clear=True,
        ):
            manifest = backfill.build_manifest(args)

        self.assertEqual(
            [target.isoformat() for target in manifest[variation]],
            ["2026-04-28T03:00:00+00:00"],
        )

    def test_limit_manifest_preserves_per_variation_and_global_caps(self) -> None:
        first = backfill.Variation(
            label="First",
            history_path=self.temp_root / "first.csv",
            last_prediction_path=None,
            registered_model_name="first",
            workflow_name="first",
            workflow_variant="first",
            daily_model_refresh=False,
        )
        second = backfill.Variation(
            label="Second",
            history_path=self.temp_root / "second.csv",
            last_prediction_path=None,
            registered_model_name="second",
            workflow_name="second",
            workflow_variant="second",
            daily_model_refresh=False,
        )
        manifest = {
            first: list(
                backfill.iter_hourly_targets(
                    backfill.parse_timestamp("2026-04-28T00:00:00+00:00"),
                    backfill.parse_timestamp("2026-04-28T04:00:00+00:00"),
                )
            ),
            second: list(
                backfill.iter_hourly_targets(
                    backfill.parse_timestamp("2026-04-29T00:00:00+00:00"),
                    backfill.parse_timestamp("2026-04-29T04:00:00+00:00"),
                )
            ),
        }

        limited = backfill.limit_manifest(
            manifest,
            max_hours=3,
            max_total_targets=4,
        )

        self.assertEqual(len(limited[first]), 3)
        self.assertEqual(len(limited[second]), 1)
        self.assertEqual(
            [target.isoformat() for target in limited[second]],
            ["2026-04-29T00:00:00+00:00"],
        )

    def test_iter_manifest_batches_keeps_variation_target_order(self) -> None:
        variation = backfill.Variation(
            label="Batch",
            history_path=self.temp_root / "batch.csv",
            last_prediction_path=None,
            registered_model_name="batch",
            workflow_name="batch",
            workflow_variant="batch",
            daily_model_refresh=False,
        )
        targets = list(
            backfill.iter_hourly_targets(
                backfill.parse_timestamp("2026-04-28T00:00:00+00:00"),
                backfill.parse_timestamp("2026-04-28T02:00:00+00:00"),
            )
        )

        batches = backfill.iter_manifest_batches({variation: targets}, batch_size=2)

        self.assertEqual(len(batches), 2)
        self.assertEqual(
            [target.isoformat() for target in batches[0][variation]],
            [
                "2026-04-28T00:00:00+00:00",
                "2026-04-28T01:00:00+00:00",
            ],
        )
        self.assertEqual(
            [target.isoformat() for target in batches[1][variation]],
            ["2026-04-28T02:00:00+00:00"],
        )

    def test_assert_no_missing_rows_rejects_executed_missing_target(self) -> None:
        history_path = self.temp_root / "history.csv"
        self.write_history_rows(
            history_path,
            [
                {
                    "timestamp": "2026-04-28T00:00:00+00:00",
                    "status": "missing",
                    "workflow_name": "hourly24",
                    "workflow_variant": "hourly_24h_prediction",
                },
            ],
        )
        variation = backfill.Variation(
            label="Missing Audit",
            history_path=history_path,
            last_prediction_path=None,
            registered_model_name="missing",
            workflow_name="hourly24",
            workflow_variant="hourly_24h_prediction",
            daily_model_refresh=False,
        )

        with self.assertRaisesRegex(RuntimeError, "still has missing"):
            backfill.assert_no_missing_rows(
                variation,
                [backfill.parse_timestamp("2026-04-28T00:00:00+00:00")],
            )

    def test_daily_replay_trains_at_midnight_or_when_bundle_missing(self) -> None:
        variation = backfill.Variation(
            label="Daily",
            history_path=self.temp_root / "daily.csv",
            last_prediction_path=None,
            registered_model_name="daily",
            workflow_name="daily",
            workflow_variant="daily",
            daily_model_refresh=True,
            refresh_policy="daily_midnight_et",
        )

        self.assertTrue(
            backfill.should_train_for_replay(
                variation,
                backfill.parse_timestamp("2026-04-28T05:00:00+00:00"),
                bundle_exists=True,
            )
        )
        self.assertTrue(
            backfill.should_train_for_replay(
                variation,
                backfill.parse_timestamp("2026-04-28T12:00:00+00:00"),
                bundle_exists=False,
            )
        )
        self.assertFalse(
            backfill.should_train_for_replay(
                variation,
                backfill.parse_timestamp("2026-04-28T12:00:00+00:00"),
                bundle_exists=True,
            )
        )

    def test_market_hours_daily_replay_trains_only_inside_training_window(self) -> None:
        variation = backfill.Variation(
            label="Market Daily",
            history_path=self.temp_root / "market-daily.csv",
            last_prediction_path=None,
            registered_model_name="market-daily",
            workflow_name="market-daily",
            workflow_variant="market-daily",
            daily_model_refresh=True,
            market_hours_only=True,
            refresh_policy="market_hours_daily",
        )

        self.assertTrue(backfill.is_market_hours_target("2026-04-28T12:00:00+00:00"))
        self.assertTrue(
            backfill.should_train_for_replay(
                variation,
                backfill.parse_timestamp("2026-04-28T12:00:00+00:00"),
                bundle_exists=False,
            )
        )
        self.assertFalse(
            backfill.should_train_for_replay(
                variation,
                backfill.parse_timestamp("2026-04-28T12:00:00+00:00"),
                bundle_exists=True,
            )
        )
        self.assertFalse(backfill.is_market_hours_target("2026-04-28T01:00:00+00:00"))

    def test_consolidated_replay_executes_once_per_target(self) -> None:
        hourly = backfill.Variation(
            label="Consolidated A",
            history_path=self.temp_root / "consolidated.csv",
            last_prediction_path=None,
            registered_model_name="a",
            workflow_name="a",
            workflow_variant="a",
            daily_model_refresh=False,
            refresh_policy="consolidated",
        )
        daily = backfill.Variation(
            label="Consolidated B",
            history_path=self.temp_root / "consolidated.csv",
            last_prediction_path=None,
            registered_model_name="b",
            workflow_name="b",
            workflow_variant="b",
            daily_model_refresh=True,
            refresh_policy="consolidated",
        )
        target = backfill.parse_timestamp("2026-04-28T12:00:00+00:00")

        with patch.object(backfill, "execute_consolidated_replay") as replay_mock:
            backfill.execute_manifest({hourly: [target], daily: [target]})

        replay_mock.assert_called_once_with(target)

    def test_registry_name_resolution_matches_live_variant_env_rules(self) -> None:
        fake_tournament = SimpleNamespace(
            DEFAULT_MODEL_NAME="default-btc-model",
            DEFAULT_EXPERIMENT_PREFIX="btc",
            get_env_str=lambda key: {
                "MLFLOW_MODEL_NAME": "base-model",
                "MLFLOW_DAILY_MODEL_NAME": "",
                "MLFLOW_MARKET_HOURS_MODEL_NAME": "explicit-market",
                "MLFLOW_MARKET_HOURS_DAILY_MODEL_NAME": "",
            }.get(key, ""),
        )
        daily = backfill.Variation(
            label="BTC Daily",
            history_path=self.temp_root / "daily.csv",
            last_prediction_path=None,
            registered_model_name="old-daily",
            workflow_name="daily-hourly",
            workflow_variant="daily",
            daily_model_refresh=True,
        )
        market = backfill.Variation(
            label="BTC Market Hours",
            history_path=self.temp_root / "market.csv",
            last_prediction_path=None,
            registered_model_name="old-market",
            workflow_name="market-hours-hourly",
            workflow_variant="market",
            daily_model_refresh=False,
        )
        market_daily = backfill.Variation(
            label="BTC Market Hours Daily",
            history_path=self.temp_root / "market-daily.csv",
            last_prediction_path=None,
            registered_model_name="old-market-daily",
            workflow_name="market-hours-daily",
            workflow_variant="market-daily",
            daily_model_refresh=True,
        )

        with patch.object(backfill, "require_tournament", return_value=fake_tournament):
            self.assertEqual(
                backfill.resolve_registered_model_name(daily),
                "base-model-daily",
            )
            self.assertEqual(
                backfill.resolve_registered_model_name(market),
                "explicit-market",
            )
            self.assertEqual(
                backfill.resolve_registered_model_name(market_daily),
                "base-model-market-hours-daily",
            )

    def test_unknown_only_family_fails_early(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "No model families matched"):
            backfill.selected_families({"made_up_family"})

    def test_execute_manifest_does_not_force_binance_only_mode(self) -> None:
        hourly = backfill.Variation(
            label="BTC Hourly",
            history_path=self.temp_root / "hourly.csv",
            last_prediction_path=None,
            registered_model_name="hourly",
            workflow_name="hourly24",
            workflow_variant="hourly",
            daily_model_refresh=False,
        )
        target = backfill.parse_timestamp("2026-04-28T00:00:00+00:00")

        with patch.dict("os.environ", {}, clear=True), \
            patch.object(backfill, "fetch_raw_snapshot", return_value=object()), \
            patch.object(backfill, "execute_live_btc_replay", return_value=None), \
            patch.object(backfill, "assert_no_missing_rows"):
            backfill.execute_manifest({hourly: [target]})

        self.assertNotIn("BTC_EXCHANGE_MODE", backfill.os.environ)

    def test_live_daily_prediction_only_reloads_registry_champions(self) -> None:
        variation = backfill.Variation(
            label="BTC Daily",
            history_path=self.temp_root / "daily.csv",
            last_prediction_path=self.temp_root / "last.json",
            registered_model_name="daily",
            workflow_name="daily-hourly",
            workflow_variant="daily",
            daily_model_refresh=True,
            refresh_policy="daily_midnight_et",
        )
        first_target = backfill.parse_timestamp("2026-04-28T05:00:00+00:00")
        second_target = backfill.parse_timestamp("2026-04-28T06:00:00+00:00")
        calls: list[bool] = []

        def fake_daily_execute(*_: object, replay_day_has_models: bool, **__: object) -> dict[str, object]:
            calls.append(replay_day_has_models)
            target = first_target if len(calls) == 1 else second_target
            return {"target_candle_timestamp": target.isoformat(), "predicted_label": 1}

        with patch.object(backfill, "fetch_raw_snapshot", return_value=object()), \
            patch("src.btc_pipeline.daily_main.execute_daily_workflow", side_effect=fake_daily_execute) as execute_mock, \
            patch.object(backfill, "history_row_from_record", return_value={"timestamp": second_target.isoformat()}), \
            patch.object(backfill, "append_history_row"), \
            patch.object(backfill, "write_last_prediction"), \
            patch.object(backfill, "assert_no_missing_rows"):
            backfill.execute_manifest({variation: [first_target, second_target]})

        self.assertEqual(execute_mock.call_count, 2)
        self.assertEqual(calls, [False, True])


if __name__ == "__main__":
    unittest.main()

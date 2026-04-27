#!/usr/bin/env python3
"""
Train BTC challengers once, then compare and optionally promote across four isolated tracks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import pandas as pd
from mlflow import MlflowClient

import daily_main
import main as tournament
import market_hours_common

from pipelines.consolidated import config, io, logging_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated consolidated BTC tournament workflow."
    )
    parser.add_argument(
        "--reset-champion-from-challenger",
        action="store_true",
        help="Ignore current champions and select from the current challengers when promotion is allowed.",
    )
    return parser.parse_args()


def configure_tracking() -> str:
    tournament.DEFAULT_EXPERIMENT_PREFIX = config.resolve_experiment_prefix()
    tournament.configure_tracking()
    return config.resolve_base_registered_model_name()


def fetch_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tournament.log_step("Fetch BTC/USDT market data")
    raw = tournament.fetch_ohlcv(
        limit=tournament.LOOKBACK_HOURS,
        min_candles=5000,
        retry_binanceus=True,
        retry_binanceus_attempts=3,
    )
    tournament.log_step("Build features and dataset splits")
    featured = tournament.add_features(raw)
    train_df, valid_df, future_row = tournament.split_dataset(
        featured,
        tournament.VALIDATION_HOURS,
    )
    return raw, train_df, valid_df, future_row


def target_timestamp_from_future_row(future_row: pd.DataFrame) -> pd.Timestamp:
    reference_timestamp = pd.Timestamp(future_row["timestamp"].iloc[0])
    return reference_timestamp + pd.Timedelta(hours=1)


def champions_trained_for_current_et_day(
    client: MlflowClient,
    track_registered_model_name: str,
    now: pd.Timestamp | None = None,
) -> bool:
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    current_et_day = current.tz_convert(market_hours_common.EASTERN_TZ).date()

    for family in config.MODEL_FAMILIES:
        family_registered_model_name = tournament.registered_model_name_for_family(
            track_registered_model_name,
            family,
        )
        try:
            version = client.get_model_version_by_alias(
                family_registered_model_name,
                tournament.CHAMPION_ALIAS,
            )
        except Exception:
            return False

        creation_timestamp = getattr(version, "creation_timestamp", None)
        if creation_timestamp is None:
            return False

        version_et_day = (
            pd.Timestamp(creation_timestamp, unit="ms", tz="UTC")
            .tz_convert(market_hours_common.EASTERN_TZ)
            .date()
        )
        if version_et_day != current_et_day:
            return False

    return True


def clone_result(result: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(result)
    cloned["candidate"] = result["candidate"]
    return cloned


def serialize_result_optional(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return tournament.serialize_result(result)


def compute_track_state(
    track: config.TrackConfig,
    client: MlflowClient,
    now: pd.Timestamp,
) -> dict[str, Any]:
    prediction_window_open = market_hours_common.should_run_prediction_window(now)
    training_window_open = market_hours_common.should_run_training_window(now)
    target_timestamp = market_hours_common.next_target_timestamp_utc(now)
    track_registered_model_name = config.registered_model_name_for_track(track)

    if track.id == "hourly_24h":
        return {
            "status": "active",
            "evaluate": True,
            "promotion_allowed": True,
            "daily_model_refresh": False,
            "track_registered_model_name": track_registered_model_name,
            "prediction_window_open": prediction_window_open,
            "training_window_open": training_window_open,
            "target_timestamp": target_timestamp,
        }

    if track.id == "hourly_daily":
        midnight_refresh_due = daily_main.should_refresh_daily_models(now)
        current_day_has_models = champions_trained_for_current_et_day(
            client,
            track_registered_model_name,
            now,
        )
        refresh_due = midnight_refresh_due or not current_day_has_models
        return {
            "status": "active",
            "evaluate": True,
            "promotion_allowed": refresh_due,
            "daily_model_refresh": refresh_due,
            "track_registered_model_name": track_registered_model_name,
            "prediction_window_open": prediction_window_open,
            "training_window_open": training_window_open,
            "target_timestamp": target_timestamp,
            "current_day_has_models": current_day_has_models,
            "midnight_refresh_due": midnight_refresh_due,
        }

    if track.id == "market_hours":
        if not prediction_window_open:
            return {
                "status": "skipped",
                "evaluate": False,
                "promotion_allowed": False,
                "daily_model_refresh": False,
                "track_registered_model_name": track_registered_model_name,
                "prediction_window_open": prediction_window_open,
                "training_window_open": training_window_open,
                "target_timestamp": target_timestamp,
                "reason": "next target candle is outside ET market hours",
            }
        return {
            "status": "active",
            "evaluate": True,
            "promotion_allowed": True,
            "daily_model_refresh": False,
            "track_registered_model_name": track_registered_model_name,
            "prediction_window_open": prediction_window_open,
            "training_window_open": training_window_open,
            "target_timestamp": target_timestamp,
        }

    if track.id == "market_hours_daily":
        if not prediction_window_open:
            return {
                "status": "skipped",
                "evaluate": False,
                "promotion_allowed": False,
                "daily_model_refresh": False,
                "track_registered_model_name": track_registered_model_name,
                "prediction_window_open": prediction_window_open,
                "training_window_open": training_window_open,
                "target_timestamp": target_timestamp,
                "reason": "next target candle is outside ET market hours",
            }

        current_day_has_models = champions_trained_for_current_et_day(
            client,
            track_registered_model_name,
            now,
        )
        refresh_due = training_window_open and not current_day_has_models
        if not refresh_due and not current_day_has_models and not training_window_open:
            return {
                "status": "skipped",
                "evaluate": False,
                "promotion_allowed": False,
                "daily_model_refresh": False,
                "track_registered_model_name": track_registered_model_name,
                "prediction_window_open": prediction_window_open,
                "training_window_open": training_window_open,
                "target_timestamp": target_timestamp,
                "reason": "same-day market-hours daily champions are missing outside the ET training window",
            }
        return {
            "status": "active",
            "evaluate": True,
            "promotion_allowed": refresh_due,
            "daily_model_refresh": refresh_due,
            "track_registered_model_name": track_registered_model_name,
            "prediction_window_open": prediction_window_open,
            "training_window_open": training_window_open,
            "target_timestamp": target_timestamp,
            "current_day_has_models": current_day_has_models,
        }

    raise ValueError(f"Unsupported track id: {track.id}")


def build_skipped_track_record(track: config.TrackConfig, state: dict[str, Any]) -> dict[str, Any]:
    target_timestamp = pd.Timestamp(state["target_timestamp"])
    return {
        "status": "skipped",
        "track": track.id,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "target_candle_timestamp": target_timestamp.isoformat(),
        "prediction_generated_at": pd.Timestamp.utcnow().isoformat(),
        "registered_model_name": state["track_registered_model_name"],
        "workflow_name": track.workflow_name,
        "workflow_variant": track.workflow_variant,
        "daily_model_refresh": False,
        "skipped_reason": state.get("reason", "track not eligible for this run"),
        "prediction_window_open": bool(state.get("prediction_window_open", False)),
        "training_window_open": bool(state.get("training_window_open", False)),
    }


def evaluate_track_family(
    *,
    args: argparse.Namespace,
    track_id: str,
    promotion_allowed: bool,
    daily_model_refresh: bool,
    track_registered_model_name: str,
    family: str,
    challenger_source: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
) -> dict[str, Any]:
    challenger_result = clone_result(challenger_source)
    champion_result: dict[str, Any] | None = None
    champion_meta: dict[str, str] | None = None
    family_registered_model_name = tournament.registered_model_name_for_family(
        track_registered_model_name,
        family,
    )

    if not args.reset_champion_from_challenger or not promotion_allowed:
        local_client = MlflowClient()
        champion_candidate, champion_meta = tournament.get_current_champion(
            local_client,
            family_registered_model_name,
            alias=tournament.CHAMPION_ALIAS,
        )
        if champion_candidate is not None and champion_meta is not None:
            champion_result = tournament.evaluate_champion(
                champion_candidate,
                train_df,
                valid_df,
                future_row,
            )
            champion_result["registry_version"] = champion_meta["version"]

    null_model_block = (
        challenger_result["f1"] <= 0.5 or challenger_result["accuracy"] <= 0.5
    )
    challenger_beats_champion = champion_result is None or (
        challenger_result["f1"] > champion_result["f1"]
    )
    deferred_due_to_schedule = (
        not promotion_allowed
        and champion_result is not None
        and challenger_beats_champion
    )

    if promotion_allowed:
        should_promote = champion_result is None or (
            challenger_beats_champion and not null_model_block
        )
    else:
        should_promote = False

    if should_promote:
        active_family_result = challenger_result
    elif champion_result is not None:
        active_family_result = champion_result
    else:
        active_family_result = challenger_result

    return {
        "family": family,
        "track_id": track_id,
        "registered_model_name": family_registered_model_name,
        "promotion_allowed": bool(promotion_allowed),
        "daily_model_refresh": bool(daily_model_refresh),
        "promoted": bool(should_promote),
        "promotion_blocked": bool(null_model_block and challenger_beats_champion),
        "deferred_due_to_schedule": bool(deferred_due_to_schedule),
        "challenger_result": challenger_result,
        "champion_result": champion_result,
        "champion_meta": champion_meta,
        "active_result": active_family_result,
    }


def process_track(
    *,
    args: argparse.Namespace,
    track: config.TrackConfig,
    state: dict[str, Any],
    client: MlflowClient,
    challenger_by_family: dict[str, dict[str, Any]],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
    full_labeled_df: pd.DataFrame,
) -> dict[str, Any]:
    track_registered_model_name = state["track_registered_model_name"]
    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()
    all_results: list[dict[str, Any]] = [clone_result(result) for result in challenger_by_family.values()]
    family_decisions: list[dict[str, Any]] = []
    active_results_by_family: dict[str, dict[str, Any]] = {}

    with mlflow.start_run(run_name=f"btc-directional-consolidated-{track.id}", nested=True):
        mlflow.set_tags(
            {
                "consolidated_workflow": "true",
                "track_id": track.id,
                "workflow_name": track.workflow_name,
                "workflow_variant": track.workflow_variant,
                "daily_model_refresh": str(bool(state["daily_model_refresh"])).lower(),
                "prediction_window_open": str(bool(state["prediction_window_open"])).lower(),
                "training_window_open": str(bool(state["training_window_open"])).lower(),
            }
        )

        if args.reset_champion_from_challenger:
            print(f"[{track.id}] Champion comparison disabled where promotion is allowed.")

        parallel_results: dict[str, dict[str, Any]] = {}
        max_workers = min(8, len(config.MODEL_FAMILIES))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_track_family,
                    args=args,
                    track_id=track.id,
                    promotion_allowed=bool(state["promotion_allowed"]),
                    daily_model_refresh=bool(state["daily_model_refresh"]),
                    track_registered_model_name=track_registered_model_name,
                    family=family,
                    challenger_source=challenger_by_family[family],
                    train_df=train_df,
                    valid_df=valid_df,
                    future_row=future_row,
                ): family
                for family in config.MODEL_FAMILIES
            }
            for future in as_completed(future_map):
                result = future.result()
                parallel_results[result["family"]] = result

        for family in config.MODEL_FAMILIES:
            decision = parallel_results[family]
            challenger_result = decision["challenger_result"]
            champion_result = decision["champion_result"]
            champion_meta = decision["champion_meta"]
            active_family_result = decision["active_result"]
            family_registered_model_name = decision["registered_model_name"]

            if champion_result is not None:
                all_results.append(champion_result)

            if decision["promoted"]:
                new_version = tournament.promote_champion(
                    client=client,
                    registered_model_name=family_registered_model_name,
                    winner=challenger_result,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    feature_rows=full_labeled_df,
                    alias=tournament.CHAMPION_ALIAS,
                )
                active_family_result["registry_version"] = new_version
                active_family_result["source"] = "champion"
                print(
                    f"[{track.id}] {challenger_result['name']} -> promoted to "
                    f"{family_registered_model_name} version {new_version}"
                )
            elif champion_result is not None:
                print(
                    f"[{track.id}] {champion_result['name']} retained for "
                    f"{family_registered_model_name} version {champion_meta['version']}"
                )
            elif not state["promotion_allowed"]:
                print(
                    f"[{track.id}] Missing incumbent champion for {family_registered_model_name}; "
                    "using challenger output for comparison only."
                )

            active_results_by_family[family] = active_family_result
            family_decisions.append(
                {
                    "family": family,
                    "registered_model_name": family_registered_model_name,
                    "promotion_allowed": bool(decision["promotion_allowed"]),
                    "daily_model_refresh": bool(decision["daily_model_refresh"]),
                    "promoted": bool(decision["promoted"]),
                    "promotion_blocked": bool(decision["promotion_blocked"]),
                    "deferred_due_to_schedule": bool(decision["deferred_due_to_schedule"]),
                    "challenger": serialize_result_optional(challenger_result),
                    "champion": serialize_result_optional(champion_result),
                    "active": serialize_result_optional(active_family_result),
                }
            )

        tournament.print_scoreboard(all_results)
        active_result = sorted(active_results_by_family.values(), key=tournament.ranking_key)[0]
        tournament.log_comparison_metrics(
            active_results_by_family=active_results_by_family,
            active_result=active_result,
        )

        prediction_record = tournament.build_prediction_record(
            active_result=active_result,
            active_results_by_family=active_results_by_family,
            future_row=future_row,
            registered_model_name=track_registered_model_name,
        )
        prediction_record.update(
            {
                "track": track.id,
                "workflow_name": track.workflow_name,
                "workflow_variant": track.workflow_variant,
                "daily_model_refresh": bool(state["daily_model_refresh"]),
                "prediction_generated_at": prediction_record.get("generated_at"),
                "promotion_allowed": bool(state["promotion_allowed"]),
                "prediction_window_open": bool(state["prediction_window_open"]),
                "training_window_open": bool(state["training_window_open"]),
            }
        )
        track_summary = {
            "status": "success",
            "track": track.id,
            "registered_model_name": track_registered_model_name,
            "prediction_record": prediction_record,
            "family_decisions": family_decisions,
        }
        mlflow.log_text(
            json.dumps(track_summary, indent=2),
            f"tracks/{track.id}.json",
        )
        return track_summary


def build_history_row(track_payload: dict[str, Any]) -> dict[str, Any]:
    if track_payload["status"] != "success":
        return {
            "timestamp": track_payload["target_candle_timestamp"],
            "track": track_payload["track"],
            "status": track_payload["status"],
            "target_candle_timestamp": track_payload["target_candle_timestamp"],
            "prediction_generated_at": track_payload["prediction_generated_at"],
            "daily_model_refresh": False,
            "best_champion_name": "",
            "best_champion_family": "",
            "best_champion_version": "",
            "predicted_signal": "",
            "probability_up": "",
            "model_accuracy": "",
            "model_f1": "",
            "promoted": False,
            "promotion_blocked": False,
            "registered_model_name": track_payload["registered_model_name"],
            "actual": "",
            "result": "",
            "failed": False,
            "reference_candle_timestamp": "",
            "reference_open": "",
            "reference_close": "",
            "target_open": "",
            "target_close": "",
            "model_predictions": "{}",
            "workflow_name": track_payload["workflow_name"],
            "workflow_variant": track_payload["workflow_variant"],
            "skipped_reason": track_payload.get("skipped_reason", ""),
        }

    record = track_payload["prediction_record"]
    promoted = any(
        bool(entry["promoted"])
        for entry in track_payload.get("family_decisions", [])
    )
    promotion_blocked = any(
        bool(entry["promotion_blocked"])
        for entry in track_payload.get("family_decisions", [])
    )
    return {
        "timestamp": record["target_candle_timestamp"],
        "track": track_payload["track"],
        "status": record["status"],
        "target_candle_timestamp": record["target_candle_timestamp"],
        "prediction_generated_at": record.get("prediction_generated_at", record["generated_at"]),
        "daily_model_refresh": bool(record.get("daily_model_refresh", False)),
        "best_champion_name": record.get("best_champion_name", ""),
        "best_champion_family": record.get("best_champion_family", ""),
        "best_champion_version": record.get("best_champion_version", ""),
        "predicted_signal": record.get("predicted_signal", ""),
        "probability_up": float(record.get("probability_up", 0.0)),
        "model_accuracy": float(record.get("model_accuracy", 0.0)),
        "model_f1": float(record.get("model_f1", 0.0)),
        "promoted": promoted,
        "promotion_blocked": promotion_blocked,
        "registered_model_name": record["registered_model_name"],
        "actual": "",
        "result": "",
        "failed": False,
        "reference_candle_timestamp": record.get("reference_candle_timestamp", ""),
        "reference_open": float(record.get("reference_open", 0.0)),
        "reference_close": float(record.get("reference_close", 0.0)),
        "target_open": "",
        "target_close": "",
        "model_predictions": io.serialize_model_predictions(record.get("model_predictions", {})),
        "workflow_name": record.get("workflow_name", config.WORKFLOW_NAME),
        "workflow_variant": record.get("workflow_variant", config.WORKFLOW_VARIANT),
        "skipped_reason": "",
    }


def build_output_payload(
    *,
    base_registered_model_name: str,
    now: pd.Timestamp,
    track_outputs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_prediction_payload = {
        "status": "success",
        "generated_at": now.isoformat(),
        "workflow_name": config.WORKFLOW_NAME,
        "workflow_variant": config.WORKFLOW_VARIANT,
        "experiment_prefix": config.resolve_experiment_prefix(),
        "registered_model_name": base_registered_model_name,
        "target_candle_timestamp": market_hours_common.next_target_timestamp_utc(now).isoformat(),
        "tracks": track_outputs,
    }

    comparison_summary = {
        "status": "success",
        "generated_at": now.isoformat(),
        "workflow_name": config.WORKFLOW_NAME,
        "workflow_variant": config.WORKFLOW_VARIANT,
        "experiment_prefix": config.resolve_experiment_prefix(),
        "registered_model_name": base_registered_model_name,
        "tracks": track_outputs,
    }
    return last_prediction_payload, comparison_summary


def run() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    tournament.set_seed()
    tournament.log_step("Initialize consolidated BTC workflow")
    print(market_hours_common.describe_window())

    base_registered_model_name = configure_tracking()
    client = MlflowClient()
    now = pd.Timestamp.now(tz="UTC")
    _, train_df, valid_df, future_row = fetch_dataset()

    tournament.log_step("Train challenger zoo once for all consolidated tracks")
    challengers, cv_summary = tournament.train_challengers(train_df, valid_df)
    challenger_results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_labeled_df = pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = tournament.retrain_challengers_on_full_data(full_labeled_df)
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    full_prediction_frame = pd.concat([full_labeled_df, future_row], ignore_index=True)
    challenger_by_family: dict[str, dict[str, Any]] = {}

    for result in challenger_results:
        refit_candidate = refit_by_family[result["family"]]
        updated_result = clone_result(result)
        updated_result["candidate"] = refit_candidate
        updated_result["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, full_prediction_frame)[-1]
        )
        updated_result["next_signal"] = tournament.prediction_to_signal(
            updated_result["next_probability"]
        )
        challenger_by_family[updated_result["family"]] = updated_result

    track_outputs: dict[str, dict[str, Any]] = {}
    with mlflow.start_run(run_name="btc-directional-consolidated"):
        mlflow.set_tags(
            {
                "asset": tournament.SYMBOL,
                "timeframe": tournament.TIMEFRAME,
                "validation_hours": str(tournament.VALIDATION_HOURS),
                "consolidated_workflow": "true",
                "feature_preprocessor": tournament.FEATURE_PREPROCESSOR_NAME,
            }
        )
        mlflow.log_params(
            {
                "lookback_hours": tournament.LOOKBACK_HOURS,
                "validation_hours": tournament.VALIDATION_HOURS,
                "cross_validation_folds": tournament.CROSS_VALIDATION_FOLDS,
                "sequence_length": tournament.SEQUENCE_LENGTH,
                "registered_model_name": base_registered_model_name,
            }
        )
        tournament.log_challenger_summary(list(challenger_by_family.values()))

        for track in config.TRACKS:
            state = compute_track_state(track, client, now)
            if not state["evaluate"]:
                track_outputs[track.id] = build_skipped_track_record(track, state)
                print(f"[{track.id}] Skipped: {track_outputs[track.id]['skipped_reason']}")
                continue

            track_outputs[track.id] = process_track(
                args=args,
                track=track,
                state=state,
                client=client,
                challenger_by_family=challenger_by_family,
                train_df=train_df,
                valid_df=valid_df,
                future_row=future_row,
                full_labeled_df=full_labeled_df,
            )

        last_prediction_payload, comparison_summary = build_output_payload(
            base_registered_model_name=base_registered_model_name,
            now=now,
            track_outputs=track_outputs,
        )
        mlflow.log_text(
            json.dumps(last_prediction_payload, indent=2),
            config.LAST_PREDICTION_PATH.name,
        )
        mlflow.log_text(
            json.dumps(comparison_summary, indent=2),
            config.COMPARISON_SUMMARY_PATH.name,
        )

    io.write_json(config.LAST_PREDICTION_PATH, last_prediction_payload)
    io.write_json(config.COMPARISON_SUMMARY_PATH, comparison_summary)
    history_rows = [build_history_row(track_outputs[track.id]) for track in config.TRACKS]
    io.append_history_rows(history_rows)


def main() -> None:
    config.ensure_output_dirs()
    with logging_utils.tee_output(config.WORKFLOW_LOG_PATH):
        run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "workflow_name": config.WORKFLOW_NAME,
            "workflow_variant": config.WORKFLOW_VARIANT,
            "registered_model_name": config.resolve_base_registered_model_name(),
            "target_candle_timestamp": market_hours_common.next_target_timestamp_utc().isoformat(),
            "error": str(exc),
            "tracks": {},
        }
        config.ensure_output_dirs()
        io.write_json(config.LAST_PREDICTION_PATH, failure_payload)
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise

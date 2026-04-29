#!/usr/bin/env python3
"""
Train BTC challengers once, then compare and optionally promote across four isolated tracks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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

from src.btc_pipeline import daily_main
from src.btc_pipeline import main as tournament
from src.btc_pipeline import market_hours_common

from pipelines.consolidated import config, io, logging_utils


TRACK_DISPLAY_NAMES = {
    "hourly_24h": "Hourly 24h",
    "hourly_daily": "Hourly 24h Daily",
    "market_hours": "Market hours",
    "market_hours_daily": "Market hours Daily",
}


@dataclass
class ConsolidatedExecutionResult:
    base_registered_model_name: str
    now: pd.Timestamp
    track_outputs: dict[str, dict[str, Any]]
    last_prediction_payload: dict[str, Any]
    comparison_summary: dict[str, Any]
    raw_candles: pd.DataFrame | None = None
    pending_publish: "PendingConsolidatedPublish | None" = None


@dataclass
class PendingConsolidatedPublish:
    base_registered_model_name: str
    now: pd.Timestamp
    run_reference_time: pd.Timestamp
    track_states: dict[str, dict[str, Any]]
    track_decisions: dict[str, dict[str, dict[str, Any]]]
    raw_candles: pd.DataFrame
    valid_df: pd.DataFrame
    future_row: pd.DataFrame
    full_labeled_df: pd.DataFrame


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


def configure_tracking(
    reference_time: pd.Timestamp | None = None,
) -> str:
    tournament.DEFAULT_EXPERIMENT_PREFIX = config.resolve_experiment_prefix()
    tournament.configure_tracking(reference_time)
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


def track_display_name(track_id: str) -> str:
    return TRACK_DISPLAY_NAMES.get(track_id, track_id.replace("_", " ").title())


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "-+-".join("-" * widths[index] for index in range(len(headers)))
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def log_workflows_to_evaluate(
    track_states: dict[str, dict[str, Any]],
    now: pd.Timestamp,
) -> list[config.TrackConfig]:
    tournament.log_step("Checking workflows to eval")
    prediction_window_open = market_hours_common.should_run_prediction_window(now)
    if prediction_window_open:
        print("Check time: next target is inside ET market hours.")
    else:
        print("Check time: next target is outside ET market hours.")
        print("Skipping market-hours champion download/eval for workflows gated to ET market hours.")

    tracks_to_evaluate: list[config.TrackConfig] = []
    for track in config.TRACKS:
        state = track_states[track.id]
        if state["evaluate"]:
            tracks_to_evaluate.append(track)
            print(f"{track_display_name(track.id)}: queued for evaluation.")
        else:
            print(
                f"{track_display_name(track.id)}: skipped for evaluation "
                f"({state.get('reason', 'not eligible for this run')})."
            )
    return tracks_to_evaluate


def download_track_family_champion(
    *,
    args: argparse.Namespace,
    track_id: str,
    promotion_allowed: bool,
    track_registered_model_name: str,
    family: str,
    download_root: Path | None = None,
) -> dict[str, Any]:
    family_registered_model_name = tournament.registered_model_name_for_family(
        track_registered_model_name,
        family,
    )
    champion_candidate = None
    champion_meta = None
    comparison_skipped = bool(args.reset_champion_from_challenger and promotion_allowed)
    if not comparison_skipped:
        local_client = MlflowClient()
        champion_candidate, champion_meta = tournament.get_current_champion(
            local_client,
            family_registered_model_name,
            alias=tournament.CHAMPION_ALIAS,
            download_root=download_root,
        )
    return {
        "family": family,
        "track_id": track_id,
        "registered_model_name": family_registered_model_name,
        "comparison_skipped": comparison_skipped,
        "champion_candidate": champion_candidate,
        "champion_meta": champion_meta,
    }


def build_track_family_decision(
    *,
    track_id: str,
    promotion_allowed: bool,
    daily_model_refresh: bool,
    family: str,
    challenger_source: dict[str, Any],
    registered_model_name: str,
    champion_candidate: Any,
    champion_meta: dict[str, str] | None,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
    comparison_skipped: bool,
) -> dict[str, Any]:
    challenger_result = clone_result(challenger_source)
    champion_result: dict[str, Any] | None = None
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
        "registered_model_name": registered_model_name,
        "promotion_allowed": bool(promotion_allowed),
        "daily_model_refresh": bool(daily_model_refresh),
        "promoted": bool(should_promote),
        "promotion_blocked": bool(null_model_block and challenger_beats_champion),
        "deferred_due_to_schedule": bool(deferred_due_to_schedule),
        "comparison_skipped": bool(comparison_skipped),
        "challenger_result": challenger_result,
        "champion_result": champion_result,
        "champion_meta": champion_meta,
        "active_result": active_family_result,
    }


def print_eval_results(
    track_decisions: dict[str, dict[str, dict[str, Any]]],
    tracks_to_evaluate: list[config.TrackConfig],
) -> None:
    tournament.log_step("Results of eval")
    headers = ["Family", "Challenger eval", *[track_display_name(track.id) for track in tracks_to_evaluate]]
    rows: list[list[str]] = []

    for family in config.MODEL_FAMILIES:
        first_decision = next(track_decisions[track.id][family] for track in tracks_to_evaluate)
        row = [
            first_decision["challenger_result"]["name"],
            format_metric(first_decision["challenger_result"]["f1"]),
        ]
        for track in tracks_to_evaluate:
            decision = track_decisions[track.id][family]
            if decision["comparison_skipped"]:
                row.append("reset")
            elif decision["champion_result"] is None:
                row.append("-")
            else:
                row.append(format_metric(decision["champion_result"]["f1"]))
        rows.append(row)

    print_table(headers, rows)


def describe_promotion_status(
    *,
    track: config.TrackConfig,
    state: dict[str, Any],
    decision: dict[str, Any] | None,
) -> str:
    if not state["evaluate"]:
        return f"Skipped ({state.get('reason', 'not eligible for this run')})"

    if track.id in {"hourly_daily", "market_hours_daily"} and not state["promotion_allowed"]:
        if state.get("current_day_has_models"):
            return "Daily model already trained"
        if decision is not None and decision["deferred_due_to_schedule"]:
            return "Waiting for daily refresh window"

    if not state["promotion_allowed"]:
        if decision is not None and decision["deferred_due_to_schedule"]:
            return "Challenger won, promotion deferred"
        if decision is not None and decision["champion_result"] is not None:
            return "Keeping current champion"
        return "No promotion scheduled"

    if decision is None:
        return "No evaluation result"
    if decision["promoted"]:
        return "Promoting new champion"
    if decision["promotion_blocked"]:
        return "Blocked by null-model guard"
    if decision["champion_result"] is not None:
        return "Keeping current champion"
    return "Using challenger as active output"


def print_promotion_summary(
    track_states: dict[str, dict[str, Any]],
    track_decisions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    tournament.log_step("Checking workflows to promote")
    print("Check daily model champion (if daily champion is not available, promote when challenger wins).")
    tournament.log_step("Uploading/promoting models for challenger that defeated champions")

    for family in config.MODEL_FAMILIES:
        family_name = next(
            (
                track_decisions[track.id][family]["challenger_result"]["name"]
                for track in config.TRACKS
                if track.id in track_decisions
            ),
            family,
        )
        print(family_name)
        for track in config.TRACKS:
            decision = track_decisions.get(track.id, {}).get(family)
            status = describe_promotion_status(
                track=track,
                state=track_states[track.id],
                decision=decision,
            )
            print(f"{track_display_name(track.id)}: {status}")


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


def finalize_track(
    *,
    track: config.TrackConfig,
    state: dict[str, Any],
    client: MlflowClient,
    family_decisions_by_family: dict[str, dict[str, Any]],
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
    full_labeled_df: pd.DataFrame,
) -> dict[str, Any]:
    track_registered_model_name = state["track_registered_model_name"]
    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()
    family_decisions: list[dict[str, Any]] = []
    active_results_by_family: dict[str, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []

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

        for family in config.MODEL_FAMILIES:
            decision = family_decisions_by_family[family]
            challenger_result = decision["challenger_result"]
            champion_result = decision["champion_result"]
            champion_meta = decision["champion_meta"]
            active_family_result = decision["active_result"]
            family_registered_model_name = decision["registered_model_name"]

            all_results.append(challenger_result)
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


def build_preview_track_summary(
    *,
    track: config.TrackConfig,
    state: dict[str, Any],
    family_decisions_by_family: dict[str, dict[str, Any]],
    future_row: pd.DataFrame,
) -> dict[str, Any]:
    family_decisions: list[dict[str, Any]] = []
    active_results_by_family: dict[str, dict[str, Any]] = {}
    all_results: list[dict[str, Any]] = []

    for family in config.MODEL_FAMILIES:
        decision = family_decisions_by_family[family]
        challenger_result = decision["challenger_result"]
        champion_result = decision["champion_result"]
        active_family_result = clone_result(decision["active_result"])
        family_registered_model_name = decision["registered_model_name"]

        all_results.append(challenger_result)
        if champion_result is not None:
            all_results.append(champion_result)

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
    prediction_record = tournament.build_prediction_record(
        active_result=active_result,
        active_results_by_family=active_results_by_family,
        future_row=future_row,
        registered_model_name=state["track_registered_model_name"],
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
    return {
        "status": "success",
        "track": track.id,
        "registered_model_name": state["track_registered_model_name"],
        "prediction_record": prediction_record,
        "family_decisions": family_decisions,
    }


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


def execute_consolidated_workflow(
    args: argparse.Namespace,
    champion_download_root_base: Path | None = None,
    defer_promotion: bool = False,
    reference_time: pd.Timestamp | None = None,
) -> ConsolidatedExecutionResult:
    config.ensure_output_dirs()
    tournament.set_seed()
    tournament.log_step("Initialize consolidated BTC workflow")
    print(market_hours_common.describe_window())

    run_reference_time = (
        pd.Timestamp.now(tz="UTC") if reference_time is None else pd.Timestamp(reference_time)
    )
    if run_reference_time.tzinfo is None:
        run_reference_time = run_reference_time.tz_localize("UTC")
    else:
        run_reference_time = run_reference_time.tz_convert("UTC")

    base_registered_model_name = configure_tracking(run_reference_time)
    client = MlflowClient()
    now = run_reference_time
    raw, train_df, valid_df, future_row = fetch_dataset()

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
    pending_publish: PendingConsolidatedPublish | None = None
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
        track_states = {
            track.id: compute_track_state(track, client, now)
            for track in config.TRACKS
        }
        tracks_to_evaluate = log_workflows_to_evaluate(track_states, now)
        track_decisions: dict[str, dict[str, dict[str, Any]]] = {}

        if tracks_to_evaluate:
            tournament.log_step("Downloading all champions for workflows to eval")
            downloaded_champions: dict[str, dict[str, dict[str, Any]]] = {
                track.id: {} for track in tracks_to_evaluate
            }
            max_workers = min(8, len(tracks_to_evaluate) * len(config.MODEL_FAMILIES))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        download_track_family_champion,
                        args=args,
                        track_id=track.id,
                        promotion_allowed=bool(track_states[track.id]["promotion_allowed"]),
                        track_registered_model_name=track_states[track.id]["track_registered_model_name"],
                        family=family,
                        download_root=(
                            champion_download_root_base / track.id / family
                            if champion_download_root_base is not None
                            else None
                        ),
                    ): (track.id, family)
                    for track in tracks_to_evaluate
                    for family in config.MODEL_FAMILIES
                }
                for future in as_completed(future_map):
                    track_id, family = future_map[future]
                    downloaded_champions[track_id][family] = future.result()

            tournament.log_step("Evaluating on val data")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        build_track_family_decision,
                        track_id=track.id,
                        promotion_allowed=bool(track_states[track.id]["promotion_allowed"]),
                        daily_model_refresh=bool(track_states[track.id]["daily_model_refresh"]),
                        family=family,
                        challenger_source=challenger_by_family[family],
                        registered_model_name=downloaded_champions[track.id][family]["registered_model_name"],
                        champion_candidate=downloaded_champions[track.id][family]["champion_candidate"],
                        champion_meta=downloaded_champions[track.id][family]["champion_meta"],
                        train_df=train_df,
                        valid_df=valid_df,
                        future_row=future_row,
                        comparison_skipped=bool(downloaded_champions[track.id][family]["comparison_skipped"]),
                    ): (track.id, family)
                    for track in tracks_to_evaluate
                    for family in config.MODEL_FAMILIES
                }
                for future in as_completed(future_map):
                    track_id, family = future_map[future]
                    track_decisions.setdefault(track_id, {})[family] = future.result()

            print_eval_results(track_decisions, tracks_to_evaluate)

        if defer_promotion:
            for track in config.TRACKS:
                state = track_states[track.id]
                if not state["evaluate"]:
                    track_outputs[track.id] = build_skipped_track_record(track, state)
                    print(f"[{track.id}] Skipped: {track_outputs[track.id]['skipped_reason']}")
                    continue

                track_outputs[track.id] = build_preview_track_summary(
                    track=track,
                    state=state,
                    family_decisions_by_family=track_decisions[track.id],
                    future_row=future_row,
                )
            pending_publish = PendingConsolidatedPublish(
                base_registered_model_name=base_registered_model_name,
                now=now,
                run_reference_time=run_reference_time,
                track_states=track_states,
                track_decisions=track_decisions,
                raw_candles=raw,
                valid_df=valid_df,
                future_row=future_row,
                full_labeled_df=full_labeled_df,
            )
        else:
            print_promotion_summary(track_states, track_decisions)

            for track in config.TRACKS:
                state = track_states[track.id]
                if not state["evaluate"]:
                    track_outputs[track.id] = build_skipped_track_record(track, state)
                    print(f"[{track.id}] Skipped: {track_outputs[track.id]['skipped_reason']}")
                    continue

                track_outputs[track.id] = finalize_track(
                    track=track,
                    state=state,
                    client=client,
                    family_decisions_by_family=track_decisions[track.id],
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

    return ConsolidatedExecutionResult(
        base_registered_model_name=base_registered_model_name,
        now=now,
        track_outputs=track_outputs,
        last_prediction_payload=last_prediction_payload,
        comparison_summary=comparison_summary,
        raw_candles=raw,
        pending_publish=pending_publish,
    )


def finalize_pending_publish(
    pending_publish: PendingConsolidatedPublish,
) -> ConsolidatedExecutionResult:
    config.ensure_output_dirs()
    configured_name = configure_tracking(pending_publish.run_reference_time)
    if configured_name != pending_publish.base_registered_model_name:
        print(
            "Configured consolidated model base differs from deferred publish payload. "
            f"Using payload value '{pending_publish.base_registered_model_name}'."
        )
    client = MlflowClient()
    track_outputs: dict[str, dict[str, Any]] = {}

    with mlflow.start_run(run_name="btc-directional-consolidated"):
        mlflow.set_tags(
            {
                "asset": tournament.SYMBOL,
                "timeframe": tournament.TIMEFRAME,
                "validation_hours": str(tournament.VALIDATION_HOURS),
                "consolidated_workflow": "true",
                "feature_preprocessor": tournament.FEATURE_PREPROCESSOR_NAME,
                "deferred_publish": "true",
            }
        )
        mlflow.log_params(
            {
                "lookback_hours": tournament.LOOKBACK_HOURS,
                "validation_hours": tournament.VALIDATION_HOURS,
                "cross_validation_folds": tournament.CROSS_VALIDATION_FOLDS,
                "sequence_length": tournament.SEQUENCE_LENGTH,
                "registered_model_name": pending_publish.base_registered_model_name,
            }
        )
        challenger_by_family: dict[str, dict[str, Any]] = {}
        for family_decisions in pending_publish.track_decisions.values():
            for family, decision in family_decisions.items():
                challenger_by_family.setdefault(
                    family,
                    clone_result(decision["challenger_result"]),
                )
        unique_challengers = list(challenger_by_family.values())
        if unique_challengers:
            tournament.log_challenger_summary(unique_challengers)

        print_promotion_summary(
            pending_publish.track_states,
            pending_publish.track_decisions,
        )

        for track in config.TRACKS:
            state = pending_publish.track_states[track.id]
            if not state["evaluate"]:
                track_outputs[track.id] = build_skipped_track_record(track, state)
                print(f"[{track.id}] Skipped: {track_outputs[track.id]['skipped_reason']}")
                continue

            track_outputs[track.id] = finalize_track(
                track=track,
                state=state,
                client=client,
                family_decisions_by_family=pending_publish.track_decisions[track.id],
                valid_df=pending_publish.valid_df,
                future_row=pending_publish.future_row,
                full_labeled_df=pending_publish.full_labeled_df,
            )

        last_prediction_payload, comparison_summary = build_output_payload(
            base_registered_model_name=pending_publish.base_registered_model_name,
            now=pending_publish.now,
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

    return ConsolidatedExecutionResult(
        base_registered_model_name=pending_publish.base_registered_model_name,
        now=pending_publish.now,
        track_outputs=track_outputs,
        last_prediction_payload=last_prediction_payload,
        comparison_summary=comparison_summary,
        raw_candles=pending_publish.raw_candles,
        pending_publish=None,
    )


def persist_execution_outputs(execution: ConsolidatedExecutionResult) -> None:
    io.write_json(config.LAST_PREDICTION_PATH, execution.last_prediction_payload)
    io.write_json(config.COMPARISON_SUMMARY_PATH, execution.comparison_summary)
    history_rows = [
        build_history_row(execution.track_outputs[track.id])
        for track in config.TRACKS
    ]
    history = io.append_history_rows(history_rows)
    io.backfill_history_with_candles(history, execution.raw_candles)


def run() -> None:
    args = parse_args()
    execution = execute_consolidated_workflow(args)
    persist_execution_outputs(execution)


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

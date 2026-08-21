#!/usr/bin/env python3
"""Forward-only BTC artifact backfill.

This runner starts after the latest existing non-missing prediction for each
variation and can append validated forward rows up to a safe settled cutoff.
Execution is intentionally bounded by --max-hours because every timestamp trains
the challenger zoo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ModuleNotFoundError:  # Dry-run manifest mode works without the ML stack.
    pd = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.maintenance.generate_btc_model_metrics_report import (
    OLD_NEW_SPLIT_UTC,
    generate_all_reports,
)
from src.btc_pipeline import artifact_sync


HISTORY_COLUMNS = [
    "timestamp",
    "predicted",
    "actual",
    "result",
    "failed",
    "status",
    "reference_open",
    "reference_close",
    "target_open",
    "target_close",
    "model_predictions",
    "best_champion_name",
    "best_champion_family",
    "best_champion_version",
    "workflow_name",
    "workflow_variant",
    "daily_model_refresh",
    "model_refresh_et_date",
    "prediction_generated_at",
]
BACKFILL_MODEL_CACHE_DIR = ROOT / "artifacts" / "backfill_model_cache"
NEWTEST_LOCAL_CHAMPION_DIR = ROOT / "artifacts" / "newtest" / "local_champions"
REPORT_PATHS = (
    ROOT / "BTC_MODEL_METRICS_REPORT.md",
    ROOT / "BTC_MODEL_METRICS_REPORT_ALL.md",
    ROOT / "BTC_MODEL_METRICS_REPORT_NEW.md",
    ROOT / "BTC_MODEL_METRICS_REPORT_NEWTEST.md",
    ROOT / "BTC_MODEL_METRICS_REPORT_OLD.md",
)


@dataclass(frozen=True)
class Variation:
    label: str
    history_path: Path
    last_prediction_path: Path | None
    registered_model_name: str
    workflow_name: str
    workflow_variant: str
    daily_model_refresh: bool
    market_hours_only: bool = False
    refresh_policy: str = "hourly"


@dataclass
class TrainedModelBundle:
    model_refresh_et_date: str
    active_results_by_family: dict[str, dict[str, Any]]
    registered_model_name: str | None = None


@dataclass
class LocalChampion:
    candidate: Any
    version: int
    trained_at_target: str
    model_refresh_et_date: str


@dataclass
class RawSnapshotCache:
    raw: Any | None = None


VARIATIONS = (
    Variation(
        label="BTC Hourly",
        history_path=ROOT / "artifacts/btc/hourly/history.csv",
        last_prediction_path=ROOT / "artifacts/btc/hourly/last_prediction.json",
        registered_model_name="btc-usdt-directional-classifier",
        workflow_name="hourly24",
        workflow_variant="hourly_24h_prediction",
        daily_model_refresh=False,
    ),
    Variation(
        label="BTC Daily",
        history_path=ROOT / "artifacts/btc/daily/history.csv",
        last_prediction_path=ROOT / "artifacts/btc/daily/last_prediction.json",
        registered_model_name="btc-usdt-directional-classifier-daily",
        workflow_name="daily-hourly",
        workflow_variant="daily_model_hourly_prediction",
        daily_model_refresh=True,
        refresh_policy="daily_midnight_et",
    ),
    Variation(
        label="BTC Market Hours",
        history_path=ROOT / "artifacts/btc/market_hours/history.csv",
        last_prediction_path=ROOT / "artifacts/btc/market_hours/last_prediction.json",
        registered_model_name="btc-usdt-directional-classifier-market-hours",
        workflow_name="market-hours-hourly",
        workflow_variant="hourly_train_7am_7pm_et_predict_8am_8pm_et",
        daily_model_refresh=False,
        market_hours_only=True,
        refresh_policy="market_hours_hourly",
    ),
    Variation(
        label="BTC Market Hours Daily",
        history_path=ROOT / "artifacts/btc/market_hours_daily/history.csv",
        last_prediction_path=ROOT / "artifacts/btc/market_hours_daily/last_prediction.json",
        registered_model_name="btc-usdt-directional-classifier-market-hours-daily",
        workflow_name="market-hours-daily",
        workflow_variant="daily_refresh_7am_7pm_et_hourly_predict_8am_8pm_et",
        daily_model_refresh=True,
        market_hours_only=True,
        refresh_policy="market_hours_daily",
    ),
    Variation(
        label="Consolidated Hourly",
        history_path=ROOT / "artifacts/consolidated/history.csv",
        last_prediction_path=None,
        registered_model_name="btc-usdt-directional-classifier-consolidated-hourly-24h",
        workflow_name="consolidated-hourly-24h",
        workflow_variant="hourly_24h_always_refresh",
        daily_model_refresh=False,
        refresh_policy="consolidated",
    ),
    Variation(
        label="Consolidated Daily/Hourly Refresh",
        history_path=ROOT / "artifacts/consolidated/history.csv",
        last_prediction_path=None,
        registered_model_name="btc-usdt-directional-classifier-consolidated-hourly-daily",
        workflow_name="consolidated-hourly-daily",
        workflow_variant="hourly_prediction_daily_refresh_at_midnight_et",
        daily_model_refresh=True,
        refresh_policy="consolidated",
    ),
    Variation(
        label="Consolidated Market Hours",
        history_path=ROOT / "artifacts/consolidated/history.csv",
        last_prediction_path=None,
        registered_model_name="btc-usdt-directional-classifier-consolidated-market-hours",
        workflow_name="consolidated-market-hours",
        workflow_variant="hourly_prediction_market_hours_only",
        daily_model_refresh=False,
        market_hours_only=True,
        refresh_policy="consolidated",
    ),
    Variation(
        label="Consolidated Market Hours Daily",
        history_path=ROOT / "artifacts/consolidated/history.csv",
        last_prediction_path=None,
        registered_model_name="btc-usdt-directional-classifier-consolidated-market-hours-daily",
        workflow_name="consolidated-market-hours-daily",
        workflow_variant="market_hours_prediction_same_day_refresh",
        daily_model_refresh=True,
        market_hours_only=True,
        refresh_policy="consolidated",
    ),
    Variation(
        label="NEWTEST BTC Daily",
        history_path=ROOT / "artifacts/newtest/btc_daily_history.csv",
        last_prediction_path=ROOT / "artifacts/newtest/btc_daily_last_prediction.json",
        registered_model_name="btc-usdt-directional-classifier-newtest-daily",
        workflow_name="newtest-daily-hourly",
        workflow_variant="newtest_daily_model_hourly_prediction",
        daily_model_refresh=True,
        refresh_policy="daily_midnight_et",
    ),
)
EASTERN_TZ = ZoneInfo("America/New_York")


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    if pd is None:
        parsed = parse_datetime(value)
        return None if parsed is None else parsed  # type: ignore[return-value]
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    timestamp = pd.Timestamp(text)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text or text.lower() == "nan":
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "Execution requires project dependencies. Install requirements.txt first."
        )
    return pd


def require_tournament() -> Any:
    try:
        from src.btc_pipeline import main as tournament
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Execution requires project dependencies. Install requirements.txt first."
        ) from exc
    return tournament


def selected_families(only_families: set[str] | None = None) -> tuple[str, ...]:
    families = ("rf", "xgb", "mlp_sklearn", "lstm", "transformer", "nn")
    if not only_families:
        return families
    selected = tuple(family for family in families if family in only_families)
    if not selected:
        raise RuntimeError(
            f"No model families matched: {', '.join(sorted(only_families))}"
        )
    return selected


def uses_local_replay_registry(variation: Variation) -> bool:
    return variation.label.startswith("NEWTEST ")


def configure_registry_for_variation(variation: Variation, reference_time: Any) -> Any:
    tournament = require_tournament()
    resolve_registered_model_name(variation)
    tournament.configure_tracking(reference_time)
    from mlflow import MlflowClient

    return MlflowClient()


def resolve_registered_model_name(variation: Variation) -> str:
    tournament = require_tournament()
    if variation.workflow_name == "daily-hourly":
        tournament.DEFAULT_EXPERIMENT_PREFIX = "btc-daily"
        explicit_name = tournament.get_env_str("MLFLOW_DAILY_MODEL_NAME")
        base_name = tournament.get_env_str("MLFLOW_MODEL_NAME") or tournament.DEFAULT_MODEL_NAME
        return explicit_name or f"{base_name}-daily"
    elif variation.workflow_name == "market-hours-hourly":
        tournament.DEFAULT_EXPERIMENT_PREFIX = "btc-market-hours"
        explicit_name = tournament.get_env_str("MLFLOW_MARKET_HOURS_MODEL_NAME")
        base_name = tournament.get_env_str("MLFLOW_MODEL_NAME") or tournament.DEFAULT_MODEL_NAME
        return explicit_name or f"{base_name}-market-hours"
    elif variation.workflow_name == "market-hours-daily":
        tournament.DEFAULT_EXPERIMENT_PREFIX = "btc-market-hours-daily"
        explicit_name = tournament.get_env_str("MLFLOW_MARKET_HOURS_DAILY_MODEL_NAME")
        base_name = tournament.get_env_str("MLFLOW_MODEL_NAME") or tournament.DEFAULT_MODEL_NAME
        return explicit_name or f"{base_name}-market-hours-daily"
    tournament.DEFAULT_EXPERIMENT_PREFIX = "btc"
    return tournament.get_env_str("MLFLOW_MODEL_NAME") or tournament.DEFAULT_MODEL_NAME


def is_market_hours_target(value: Any) -> bool:
    timestamp = parse_datetime(value)
    if timestamp is None:
        return False
    eastern = timestamp.astimezone(EASTERN_TZ)
    return 8 <= eastern.hour <= 20


def replay_run_time(target_timestamp: Any) -> Any:
    timestamp = parse_timestamp(target_timestamp)
    if timestamp is None:
        raise RuntimeError(f"Invalid replay target timestamp: {target_timestamp!r}")
    return timestamp - pd_timedelta(hours=1)


def replay_et_date(value: Any) -> str:
    timestamp = parse_timestamp(value)
    if timestamp is None:
        raise RuntimeError(f"Invalid replay timestamp: {value!r}")
    return timestamp.tz_convert(EASTERN_TZ).date().isoformat()


def replay_et_hour(value: Any) -> int:
    timestamp = parse_timestamp(value)
    if timestamp is None:
        raise RuntimeError(f"Invalid replay timestamp: {value!r}")
    return int(timestamp.tz_convert(EASTERN_TZ).hour)


def is_consolidated_variation(variation: Variation) -> bool:
    return variation.refresh_policy == "consolidated"


def should_train_for_replay(
    variation: Variation,
    target_timestamp: Any,
    bundle_exists: bool,
) -> bool:
    run_time = replay_run_time(target_timestamp)
    if variation.refresh_policy in {"hourly", "market_hours_hourly"}:
        return True
    if variation.refresh_policy == "daily_midnight_et":
        return replay_et_hour(run_time) == 0 or not bundle_exists
    if variation.refresh_policy == "market_hours_daily":
        training_window_open = 7 <= replay_et_hour(run_time) <= 19
        return training_window_open and not bundle_exists
    if variation.refresh_policy == "consolidated":
        return True
    raise RuntimeError(f"Unknown refresh policy for {variation.label}: {variation.refresh_policy}")


def live_refresh_run_name(variation: Variation) -> str:
    if variation.workflow_name == "daily-hourly":
        return "btc-directional-daily-refresh"
    if variation.workflow_name == "market-hours-hourly":
        return "btc-directional-market-hours-tournament"
    if variation.workflow_name == "market-hours-daily":
        return "btc-directional-market-hours-daily-refresh"
    return "btc-directional-tournament"


def live_prediction_run_name(variation: Variation) -> str:
    if variation.workflow_name == "daily-hourly":
        return "btc-directional-daily-hourly-prediction"
    if variation.workflow_name == "market-hours-daily":
        return "btc-directional-market-hours-daily-hourly-prediction"
    return live_refresh_run_name(variation)


def live_mlflow_tags(variation: Variation, *, daily_model_refresh: bool) -> dict[str, str]:
    tournament = require_tournament()
    tags = {
        "asset": tournament.SYMBOL,
        "timeframe": tournament.TIMEFRAME,
        "validation_hours": str(tournament.VALIDATION_HOURS),
    }
    if variation.workflow_name in {"daily-hourly", "market-hours-daily"}:
        tags["daily_model_refresh"] = str(daily_model_refresh).lower()
    if variation.workflow_name in {"market-hours-hourly", "market-hours-daily"}:
        tags["market_hours_workflow"] = "true"
    return tags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-fill BTC prediction artifacts.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the manifest without writing artifacts.")
    mode.add_argument("--execute", action="store_true", help="Train and append bounded forward rows.")
    parser.add_argument("--from", dest="from_ts", help="Optional UTC target timestamp start.")
    parser.add_argument("--to", dest="to_ts", help="Optional UTC target timestamp end.")
    parser.add_argument(
        "--max-hours",
        type=int,
        help="Maximum target timestamps per variation to execute. Required with --execute.",
    )
    parser.add_argument(
        "--max-total-targets",
        type=int,
        help="Optional maximum target timestamps to execute across all selected variations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional number of target timestamps to execute before regenerating reports/publishing artifacts.",
    )
    parser.add_argument(
        "--only",
        choices=[variation.label for variation in VARIATIONS],
        action="append",
        help="Limit to one or more single-workflow variations.",
    )
    parser.add_argument(
        "--only-family",
        choices=["lstm", "mlp_sklearn", "nn", "rf", "transformer", "xgb"],
        action="append",
        help="Limit execution to one or more model families.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Do not regenerate BTC_MODEL_METRICS_REPORT.md after execution.",
    )
    parser.add_argument(
        "--newtest-latest-hours",
        type=int,
        help="For NEWTEST variations, run only the latest N settled target hours.",
    )
    parser.add_argument(
        "--reset-newtest",
        action="store_true",
        help="Delete isolated NEWTEST history, last prediction, and cached model bundles before running.",
    )
    parser.add_argument(
        "--publish-artifacts",
        action="store_true",
        help="Publish generated backfill artifacts to origin/main after execution.",
    )
    parser.add_argument(
        "--commit-message",
        default="BTC forward backfill artifacts [skip ci]",
        help="Commit message used with --publish-artifacts.",
    )
    return parser.parse_args()


def now_cutoff() -> pd.Timestamp:
    if pd is None:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return now - pd_timedelta(hours=2)  # type: ignore[return-value]
    return pd.Timestamp.now(tz="UTC").floor("h") - pd_timedelta(hours=2)


def pd_timedelta(*, hours: int) -> Any:
    if pd is not None:
        return pd.Timedelta(value=hours, unit="h")
    from datetime import timedelta

    return timedelta(hours=hours)


def load_history(path: Path) -> pd.DataFrame:
    local_pd = require_pandas()
    if not path.exists():
        return local_pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = local_pd.read_csv(path)
    for column in HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[HISTORY_COLUMNS]


def latest_non_missing_prediction(history: pd.DataFrame, variation: Variation) -> pd.Timestamp | None:
    if history.empty:
        return None
    candidate = history
    if "workflow_name" in candidate.columns:
        candidate = candidate[candidate["workflow_name"].fillna("").astype(str).eq(variation.workflow_name)]
    predicted = candidate[
        candidate["predicted"].fillna("").astype(str).str.strip().ne("")
        & candidate["status"].fillna("").astype(str).str.lower().ne("missing")
    ].copy()
    if predicted.empty:
        return None
    predicted["timestamp"] = pd.to_datetime(predicted["timestamp"], utc=True, errors="coerce")
    predicted = predicted.dropna(subset=["timestamp"])
    if predicted.empty:
        return None
    return pd.Timestamp(predicted["timestamp"].max()).tz_convert("UTC")


def latest_non_missing_prediction_csv(variation: Variation) -> Any:
    path = variation.history_path
    if not path.exists():
        return None
    latest = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predicted = str(row.get("predicted", "") or "").strip()
            status = str(row.get("status", "") or "").strip().lower()
            workflow = str(row.get("workflow_name", "") or "").strip()
            if workflow and workflow != variation.workflow_name:
                continue
            if not predicted or status == "missing":
                continue
            timestamp = parse_datetime(row.get("timestamp"))
            if timestamp is not None and (latest is None or timestamp > latest):
                latest = timestamp
    return latest


def latest_existing_timestamp_csv(variation: Variation) -> Any:
    if not variation.history_path.exists():
        return None
    latest = None
    with variation.history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workflow = str(row.get("workflow_name", "") or "").strip()
            if workflow and workflow != variation.workflow_name:
                continue
            timestamp = parse_datetime(row.get("timestamp"))
            if timestamp is not None and (latest is None or timestamp > latest):
                latest = timestamp
    return latest


def backfill_boundary_start(variation: Variation) -> Any:
    split = parse_datetime(OLD_NEW_SPLIT_UTC.get(variation.label))
    if split is None:
        return None
    return split + pd_timedelta(hours=1)


def existing_prediction_timestamps_csv(variation: Variation) -> set[datetime]:
    existing: set[datetime] = set()
    if not variation.history_path.exists():
        return existing
    with variation.history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            workflow = str(row.get("workflow_name", "") or "").strip()
            if workflow != variation.workflow_name:
                continue
            predicted = str(row.get("predicted", "") or "").strip()
            status = str(row.get("status", "") or "").strip().lower()
            if not predicted or status == "missing":
                continue
            timestamp = parse_datetime(row.get("timestamp"))
            if timestamp is not None:
                existing.add(timestamp.replace(minute=0, second=0, microsecond=0))
    return existing


def iter_hourly_targets(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if pd is None:
        targets = []
        current = start
        while current <= end:
            targets.append(current)
            current = current + pd_timedelta(hours=1)
        return targets  # type: ignore[return-value]
    if start > end:
        return []
    return list(pd.date_range(start=start, end=end, freq="h", tz="UTC"))


def build_manifest(args: argparse.Namespace) -> dict[Variation, list[pd.Timestamp]]:
    explicit_start = parse_timestamp(args.from_ts)
    explicit_end = parse_timestamp(args.to_ts)
    end = explicit_end or now_cutoff()
    selected = set(args.only or [])
    manifest: dict[Variation, list[pd.Timestamp]] = {}
    for variation in VARIATIONS:
        if selected and variation.label not in selected:
            continue
        if variation.label.startswith("NEWTEST ") and args.newtest_latest_hours:
            start = explicit_start or (end - pd_timedelta(hours=args.newtest_latest_hours - 1))
        else:
            boundary_start = backfill_boundary_start(variation)
            latest = latest_non_missing_prediction_csv(variation) or latest_existing_timestamp_csv(variation)
            start = explicit_start or boundary_start or ((latest + pd_timedelta(hours=1)) if latest is not None else None)
        if start is None:
            manifest[variation] = []
            continue
        if pd is None:
            start_hour = start.replace(minute=0, second=0, microsecond=0)
            end_hour = end.replace(minute=0, second=0, microsecond=0)
        else:
            start_hour = pd.Timestamp(start).floor("h")
            end_hour = pd.Timestamp(end).floor("h")
        targets = iter_hourly_targets(start_hour, end_hour)
        if variation.market_hours_only:
            targets = [
                target
                for target in targets
                if is_market_hours_target(target)
            ]
        existing_timestamps = existing_prediction_timestamps_csv(variation)
        targets = [
            target
            for target in targets
            if parse_datetime(target) not in existing_timestamps
        ]
        manifest[variation] = targets
    return manifest


def fetch_raw_snapshot(
    target_timestamp: pd.Timestamp,
    cache: RawSnapshotCache | None = None,
) -> pd.DataFrame:
    local_pd = require_pandas()
    tournament = require_tournament()
    reference_timestamp = target_timestamp - local_pd.Timedelta(value=1, unit="h")
    raw = None if cache is None else cache.raw
    if raw is not None and not raw.empty:
        raw = raw.copy()
        raw["timestamp"] = local_pd.to_datetime(raw["timestamp"], utc=True)
        reusable = (
            raw["timestamp"].min() <= reference_timestamp
            and raw["timestamp"].max() >= reference_timestamp
            and len(raw[raw["timestamp"] <= reference_timestamp]) >= tournament.LOOKBACK_HOURS
        )
        if not reusable:
            raw = None
    if raw is None:
        current_reference = local_pd.Timestamp.now(tz="UTC").floor("h") - local_pd.Timedelta(
            value=1,
            unit="h",
        )
        extra_hours = max(
            0,
            int((current_reference - reference_timestamp).total_seconds() // 3600),
        )
        fetch_limit = tournament.LOOKBACK_HOURS + extra_hours
        raw = tournament.fetch_ohlcv(
            limit=fetch_limit,
            min_candles=tournament.LOOKBACK_HOURS,
            retry_binanceus=True,
            retry_binanceus_attempts=3,
        )
        raw["timestamp"] = local_pd.to_datetime(raw["timestamp"], utc=True)
        raw = (
            raw.drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if cache is not None:
            cache.raw = raw.copy()
        print(
            f"Fetched {len(raw)} candles, including {extra_hours} extra replay-gap hours.",
            flush=True,
        )
    else:
        print(f"Reusing cached {len(raw)} candle snapshot.", flush=True)
    replay_raw = raw[raw["timestamp"] <= reference_timestamp].tail(tournament.LOOKBACK_HOURS)
    if len(replay_raw) < tournament.VALIDATION_HOURS + tournament.SEQUENCE_LENGTH + 2:
        raise RuntimeError(
            f"Only {len(replay_raw)} candles are available before {reference_timestamp.isoformat()}, "
            "not enough to build train/validation/future splits."
        )
    print(
        f"Using {len(replay_raw)} candles through {reference_timestamp.isoformat()}."
    )
    return replay_raw.reset_index(drop=True)


def build_replay_frames(raw: pd.DataFrame, target_timestamp: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tournament = require_tournament()
    reference_timestamp = target_timestamp - pd_timedelta(hours=1)
    training_raw = raw[raw["timestamp"] <= reference_timestamp].copy()
    featured = tournament.add_features(training_raw)
    return tournament.split_dataset(featured, tournament.VALIDATION_HOURS)


def train_prediction_record(
    *,
    variation: Variation,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    only_families: set[str] | None = None,
) -> dict[str, Any]:
    local_pd = require_pandas()
    tournament = require_tournament()
    train_df, valid_df, future_row = build_replay_frames(raw, target_timestamp)
    tournament.set_seed()
    challengers, cv_summary = tournament.train_challengers(
        train_df,
        valid_df,
        only_families=only_families,
    )
    if only_families and not challengers:
        raise RuntimeError(
            f"No challenger families matched: {', '.join(sorted(only_families))}"
        )
    results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_labeled_df = local_pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = tournament.retrain_challengers_on_full_data(
        full_labeled_df,
        only_families=only_families,
    )
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    prediction_frame = local_pd.concat([full_labeled_df, future_row], ignore_index=True)
    active_results_by_family: dict[str, dict[str, Any]] = {}
    for result in results:
        updated = dict(result)
        refit_candidate = refit_by_family[result["family"]]
        updated["candidate"] = refit_candidate
        updated["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, prediction_frame)[-1]
        )
        updated["next_signal"] = tournament.prediction_to_signal(updated["next_probability"])
        active_results_by_family[updated["family"]] = updated

    active_result = sorted(active_results_by_family.values(), key=tournament.ranking_key)[0]
    record = tournament.build_prediction_record(
        active_result=active_result,
        active_results_by_family=active_results_by_family,
        future_row=future_row,
        registered_model_name=variation.registered_model_name,
    )
    record.update(
        {
            "generated_at": replay_run_time(target_timestamp).isoformat(),
            "workflow_name": variation.workflow_name,
            "workflow_variant": variation.workflow_variant,
            "daily_model_refresh": bool(variation.daily_model_refresh),
            "model_refresh_et_date": replay_et_date(replay_run_time(target_timestamp)),
            "prediction_generated_at": replay_run_time(target_timestamp).isoformat(),
        }
    )
    return record


def train_model_bundle(
    *,
    variation: Variation,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    only_families: set[str] | None = None,
) -> TrainedModelBundle:
    local_pd = require_pandas()
    tournament = require_tournament()
    train_df, valid_df, future_row = build_replay_frames(raw, target_timestamp)
    tournament.set_seed()
    challengers, cv_summary = tournament.train_challengers(
        train_df,
        valid_df,
        only_families=only_families,
    )
    if only_families and not challengers:
        raise RuntimeError(
            f"No challenger families matched: {', '.join(sorted(only_families))}"
        )
    results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_labeled_df = local_pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = tournament.retrain_challengers_on_full_data(
        full_labeled_df,
        only_families=only_families,
    )
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    full_prediction_frame = local_pd.concat([full_labeled_df, future_row], ignore_index=True)
    active_results_by_family = {}
    for result in results:
        challenger_result = dict(result)
        refit_candidate = refit_by_family[result["family"]]
        challenger_result["candidate"] = refit_candidate
        challenger_result["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, full_prediction_frame)[-1]
        )
        challenger_result["next_signal"] = tournament.prediction_to_signal(
            challenger_result["next_probability"]
        )
        if variation.label.startswith("NEWTEST "):
            active_results_by_family[challenger_result["family"]] = select_local_champion(
                variation=variation,
                challenger_result=challenger_result,
                train_df=train_df,
                valid_df=valid_df,
                future_row=future_row,
                target_timestamp=target_timestamp,
            )
        else:
            active_results_by_family[challenger_result["family"]] = challenger_result
    return TrainedModelBundle(
        model_refresh_et_date=replay_et_date(replay_run_time(target_timestamp)),
        active_results_by_family=active_results_by_family,
        registered_model_name=variation.registered_model_name,
    )


def train_registry_model_bundle(
    *,
    variation: Variation,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    only_families: set[str] | None = None,
) -> TrainedModelBundle:
    local_pd = require_pandas()
    tournament = require_tournament()
    reference_time = replay_run_time(target_timestamp)
    client = configure_registry_for_variation(variation, reference_time)
    registered_model_name = resolve_registered_model_name(variation)
    train_df, valid_df, future_row = build_replay_frames(raw, target_timestamp)
    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()
    tournament.set_seed()
    challengers, cv_summary = tournament.train_challengers(
        train_df,
        valid_df,
        only_families=only_families,
    )
    results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_labeled_df = local_pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = tournament.retrain_challengers_on_full_data(
        full_labeled_df,
        only_families=only_families,
    )
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    full_prediction_frame = local_pd.concat([full_labeled_df, future_row], ignore_index=True)
    challenger_by_family: dict[str, dict[str, Any]] = {}
    for result in results:
        updated = dict(result)
        refit_candidate = refit_by_family[result["family"]]
        updated["candidate"] = refit_candidate
        updated["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, full_prediction_frame)[-1]
        )
        updated["next_signal"] = tournament.prediction_to_signal(updated["next_probability"])
        challenger_by_family[updated["family"]] = updated

    all_results = list(challenger_by_family.values())
    active_results_by_family: dict[str, dict[str, Any]] = {}
    family_decisions: list[dict[str, Any]] = []
    for family in selected_families(only_families):
        challenger_result = challenger_by_family[family]
        family_registered_model_name = tournament.registered_model_name_for_family(
            registered_model_name,
            family,
        )
        champion_candidate, champion_meta = tournament.get_current_champion(
            client,
            family_registered_model_name,
            alias=tournament.CHAMPION_ALIAS,
        )
        champion_result = None
        if champion_candidate is not None and champion_meta is not None:
            champion_result = tournament.evaluate_champion(
                champion_candidate,
                train_df,
                valid_df,
                future_row,
            )
            champion_result["registry_version"] = champion_meta["version"]
            all_results.append(champion_result)
        null_model_block = (
            challenger_result["f1"] <= 0.5 or challenger_result["accuracy"] <= 0.5
        )
        if champion_result is None:
            should_promote = True
            active_family_result = challenger_result
        else:
            should_promote = (
                challenger_result["f1"] > champion_result["f1"] and not null_model_block
            )
            active_family_result = challenger_result if should_promote else champion_result
        active_results_by_family[family] = active_family_result
        family_decisions.append(
            {
                "family": family,
                "registered_model_name": family_registered_model_name,
                "challenger": challenger_result,
                "champion": champion_result,
                "champion_meta": champion_meta,
                "should_promote": should_promote,
                "null_model_block": null_model_block,
                "active_result": active_family_result,
            }
        )

    active_result = sorted(active_results_by_family.values(), key=tournament.ranking_key)[0]
    import mlflow

    with mlflow.start_run(run_name=live_refresh_run_name(variation)):
        tournament.print_scoreboard(all_results)
        mlflow.set_tags(
            live_mlflow_tags(
                variation,
                daily_model_refresh=variation.daily_model_refresh,
            )
        )
        mlflow.log_params(
            {
                "lookback_hours": tournament.LOOKBACK_HOURS,
                "validation_hours": tournament.VALIDATION_HOURS,
                "cross_validation_folds": tournament.CROSS_VALIDATION_FOLDS,
                "sequence_length": tournament.SEQUENCE_LENGTH,
                "rf_estimators": 400,
                "xgb_estimators": 500,
                "lstm_epochs": 40,
                "transformer_epochs": 36,
                "nn_epochs": 48,
            }
        )
        promotion_feature_rows = full_labeled_df
        for decision in family_decisions:
            challenger_result = decision["challenger"]
            champion_result = decision["champion"]
            if decision["null_model_block"] and champion_result is not None:
                print(
                    f"Promotion blocked for {challenger_result['name']}: "
                    f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                )
            elif decision["null_model_block"] and champion_result is None:
                print(
                    f"Bootstrapping missing {decision['registered_model_name']} because no incumbent champion exists. "
                    "The null-model guard only blocks replacing an existing champion: "
                    f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                )
            if decision["should_promote"]:
                new_version = tournament.promote_champion(
                    client=client,
                    registered_model_name=decision["registered_model_name"],
                    winner=challenger_result,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    feature_rows=promotion_feature_rows,
                    alias=tournament.CHAMPION_ALIAS,
                )
                decision["active_result"]["registry_version"] = new_version
                decision["active_result"]["source"] = "champion"
                print(
                    f"{challenger_result['name']} -> promoted to "
                    f"{decision['registered_model_name']} version {new_version}"
                )
            elif champion_result is not None:
                decision["active_result"]["registry_version"] = decision["champion_meta"]["version"]
                print(
                    f"{champion_result['name']} -> retained as {decision['registered_model_name']} "
                    f"version {decision['champion_meta']['version']}"
                )
            else:
                print(
                    f"{challenger_result['name']} -> no existing "
                    f"{decision['registered_model_name']} and not promoted"
                )
        best_registered_result = next(
            (
                result
                for result in sorted(active_results_by_family.values(), key=tournament.ranking_key)
                if result.get("registry_version") is not None
            ),
            None,
        )
        if best_registered_result is not None:
            active_result["best_overall_registry_version"] = best_registered_result["registry_version"]
        tournament.log_challenger_summary(list(challenger_by_family.values()))
        tournament.log_comparison_metrics(
            active_results_by_family=active_results_by_family,
            active_result=active_result,
        )
    return TrainedModelBundle(
        model_refresh_et_date=replay_et_date(reference_time),
        active_results_by_family=active_results_by_family,
        registered_model_name=registered_model_name,
    )


def load_registry_model_bundle(
    *,
    variation: Variation,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    only_families: set[str] | None = None,
) -> TrainedModelBundle:
    tournament = require_tournament()
    reference_time = replay_run_time(target_timestamp)
    client = configure_registry_for_variation(variation, reference_time)
    registered_model_name = resolve_registered_model_name(variation)
    train_df, valid_df, future_row = build_replay_frames(raw, target_timestamp)
    active_results_by_family: dict[str, dict[str, Any]] = {}
    for family in selected_families(only_families):
        family_registered_model_name = tournament.registered_model_name_for_family(
            registered_model_name,
            family,
        )
        champion_candidate, champion_meta = tournament.get_current_champion(
            client,
            family_registered_model_name,
            alias=tournament.CHAMPION_ALIAS,
        )
        if champion_candidate is None or champion_meta is None:
            raise RuntimeError(
                "Prediction-only replay requires an existing champion for every family. "
                f"Missing champion alias for {family_registered_model_name}."
            )
        champion_result = tournament.evaluate_champion(
            champion_candidate,
            train_df,
            valid_df,
            future_row,
        )
        champion_result["registry_version"] = champion_meta["version"]
        active_results_by_family[family] = champion_result
    return TrainedModelBundle(
        model_refresh_et_date=replay_et_date(reference_time),
        active_results_by_family=active_results_by_family,
        registered_model_name=registered_model_name,
    )


def local_champion_path(variation: Variation, family: str) -> Path:
    safe_label = (
        variation.label.lower()
        .replace("/", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )
    return NEWTEST_LOCAL_CHAMPION_DIR / safe_label / f"{family}.pkl"


def load_local_champion(variation: Variation, family: str) -> LocalChampion | None:
    path = local_champion_path(variation, family)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        champion = pickle.load(handle)
    if not isinstance(champion, LocalChampion):
        raise RuntimeError(f"Unexpected local champion payload in {path}.")
    return champion


def save_local_champion(variation: Variation, family: str, champion: LocalChampion) -> None:
    path = local_champion_path(variation, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(champion, handle)
    print(
        f"Saved NEWTEST local champion {family} v{champion.version}: "
        f"{path.relative_to(ROOT)}"
    )


def select_local_champion(
    *,
    variation: Variation,
    challenger_result: dict[str, Any],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
    target_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    tournament = require_tournament()
    family = challenger_result["family"]
    champion = load_local_champion(variation, family)
    champion_result = None
    if champion is not None:
        champion_result = tournament.evaluate_champion(
            champion.candidate,
            train_df,
            valid_df,
            future_row,
        )
        champion_result["registry_version"] = f"local-v{champion.version}"

    null_model_block = challenger_result["f1"] <= 0.5 or challenger_result["accuracy"] <= 0.5
    should_promote = (
        champion_result is None
        or (challenger_result["f1"] > champion_result["f1"] and not null_model_block)
    )

    if should_promote:
        next_version = 1 if champion is None else champion.version + 1
        saved = LocalChampion(
            candidate=challenger_result["candidate"],
            version=next_version,
            trained_at_target=target_timestamp.isoformat(),
            model_refresh_et_date=replay_et_date(replay_run_time(target_timestamp)),
        )
        save_local_champion(variation, family, saved)
        active_result = dict(challenger_result)
        active_result["source"] = "champion"
        active_result["registry_version"] = f"local-v{next_version}"
        if champion is None:
            print(f"{family}: bootstrapped NEWTEST local champion v{next_version}.")
        else:
            print(f"{family}: promoted NEWTEST local champion to v{next_version}.")
        return active_result

    if champion_result is None:
        return challenger_result
    if null_model_block:
        print(
            f"{family}: retained NEWTEST local champion v{champion.version}; "
            f"challenger failed null-model guard."
        )
    else:
        print(f"{family}: retained NEWTEST local champion v{champion.version}.")
    return champion_result


def bundle_cache_path(
    variation: Variation,
    model_day: str,
    only_families: set[str] | None,
) -> Path:
    family_part = "all" if not only_families else "-".join(sorted(only_families))
    safe_label = (
        variation.label.lower()
        .replace("/", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )
    return BACKFILL_MODEL_CACHE_DIR / safe_label / family_part / f"{model_day}.pkl"


def load_model_bundle(
    variation: Variation,
    model_day: str,
    only_families: set[str] | None,
) -> TrainedModelBundle | None:
    path = bundle_cache_path(variation, model_day, only_families)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, TrainedModelBundle):
        raise RuntimeError(f"Unexpected bundle payload in {path}.")
    return bundle


def save_model_bundle(
    variation: Variation,
    model_day: str,
    only_families: set[str] | None,
    bundle: TrainedModelBundle,
) -> None:
    path = bundle_cache_path(variation, model_day, only_families)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    print(f"Saved daily model bundle cache: {path.relative_to(ROOT)}")


def reset_newtest_artifacts() -> None:
    for variation in VARIATIONS:
        if not variation.label.startswith("NEWTEST "):
            continue
        for path in (variation.history_path, variation.last_prediction_path):
            if path is not None and path.exists():
                path.unlink()
                print(f"Deleted {path.relative_to(ROOT)}")
        cache_root = bundle_cache_path(variation, "*", None).parents[1]
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print(f"Deleted {cache_root.relative_to(ROOT)}")
        champion_root = local_champion_path(variation, "*").parent
        if champion_root.exists():
            shutil.rmtree(champion_root)
            print(f"Deleted {champion_root.relative_to(ROOT)}")


def prediction_record_from_bundle(
    *,
    variation: Variation,
    bundle: TrainedModelBundle,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    daily_model_refresh: bool,
) -> dict[str, Any]:
    local_pd = require_pandas()
    tournament = require_tournament()
    train_df, valid_df, future_row = build_replay_frames(raw, target_timestamp)
    prediction_frame = local_pd.concat([train_df, valid_df, future_row], ignore_index=True)
    active_results_by_family = {}
    for family, result in bundle.active_results_by_family.items():
        updated = dict(result)
        updated["next_probability"] = float(
            tournament.predict_candidate_probabilities(
                updated["candidate"],
                prediction_frame,
            )[-1]
        )
        updated["next_signal"] = tournament.prediction_to_signal(updated["next_probability"])
        active_results_by_family[family] = updated

    active_result = sorted(active_results_by_family.values(), key=tournament.ranking_key)[0]
    record = tournament.build_prediction_record(
        active_result=active_result,
        active_results_by_family=active_results_by_family,
        future_row=future_row,
        registered_model_name=bundle.registered_model_name or variation.registered_model_name,
    )
    record.update(
        {
            "generated_at": replay_run_time(target_timestamp).isoformat(),
            "workflow_name": variation.workflow_name,
            "workflow_variant": variation.workflow_variant,
            "daily_model_refresh": bool(daily_model_refresh),
            "model_refresh_et_date": bundle.model_refresh_et_date,
            "prediction_generated_at": replay_run_time(target_timestamp).isoformat(),
        }
    )
    return record


def log_prediction_only_record(
    *,
    variation: Variation,
    record: dict[str, Any],
    bundle: TrainedModelBundle,
) -> None:
    if uses_local_replay_registry(variation):
        return
    tournament = require_tournament()
    active_result = sorted(
        bundle.active_results_by_family.values(),
        key=tournament.ranking_key,
    )[0]
    import mlflow

    with mlflow.start_run(run_name=live_prediction_run_name(variation)):
        tournament.print_scoreboard(list(bundle.active_results_by_family.values()))
        mlflow.set_tags(live_mlflow_tags(variation, daily_model_refresh=False))
        tournament.log_comparison_metrics(
            active_results_by_family=bundle.active_results_by_family,
            active_result=active_result,
        )
        if variation.last_prediction_path is not None:
            mlflow.log_text(
                json.dumps(record, indent=2),
                variation.last_prediction_path.name,
            )


def actual_for_target(raw: pd.DataFrame, target_timestamp: pd.Timestamp) -> tuple[int, float, float, float] | None:
    local_pd = require_pandas()
    reference_timestamp = target_timestamp - pd_timedelta(hours=1)
    prior_timestamp = target_timestamp - local_pd.Timedelta(value=2, unit="h")
    reference = raw[raw["timestamp"] == reference_timestamp]
    if reference.empty:
        return None
    reference_row = reference.iloc[-1]
    prior = raw[raw["timestamp"] == prior_timestamp]
    reference_open = (
        float(prior.iloc[-1]["close"])
        if not prior.empty
        else float(reference_row["open"])
    )
    reference_close = float(reference_row["close"])
    target_open = reference_close
    label = int(reference_close > reference_open)
    return label, reference_open, reference_close, target_open


def history_row_from_record(record: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any] | None:
    target_timestamp = parse_timestamp(record["target_candle_timestamp"])
    if target_timestamp is None:
        return None
    actual = actual_for_target(raw, target_timestamp)
    if actual is None:
        return None
    actual_label, reference_open, reference_close, target_open = actual
    predicted_label = int(record["predicted_label"])
    return {
        "timestamp": target_timestamp.isoformat(),
        "predicted": predicted_label,
        "actual": "UP" if actual_label else "DOWN",
        "result": int(predicted_label == actual_label),
        "failed": 0,
        "status": "validated",
        "reference_open": reference_open,
        "reference_close": reference_close,
        "target_open": target_open,
        "target_close": "",
        "model_predictions": json.dumps(record.get("model_predictions", {}), separators=(",", ":"), sort_keys=True),
        "best_champion_name": record.get("best_champion_name", ""),
        "best_champion_family": record.get("best_champion_family", ""),
        "best_champion_version": record.get("best_champion_version", ""),
        "workflow_name": record.get("workflow_name", ""),
        "workflow_variant": record.get("workflow_variant", ""),
        "daily_model_refresh": record.get("daily_model_refresh", ""),
        "model_refresh_et_date": record.get("model_refresh_et_date", ""),
        "prediction_generated_at": record.get("prediction_generated_at", ""),
    }


def enrich_live_replay_record(
    variation: Variation,
    record: dict[str, Any],
    target_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    """Stamp live workflow output with replay metadata needed by artifact history."""
    updated = dict(record)
    expected_target = parse_timestamp(target_timestamp)
    actual_target = parse_timestamp(updated.get("target_candle_timestamp"))
    if expected_target is None or actual_target is None:
        raise RuntimeError(
            f"{variation.label} live replay returned an invalid target timestamp."
        )
    if actual_target != expected_target:
        raise RuntimeError(
            f"{variation.label} live replay returned target {actual_target.isoformat()} "
            f"for manifest target {expected_target.isoformat()}."
        )
    reference_time = replay_run_time(target_timestamp)
    updated["workflow_name"] = variation.workflow_name
    updated["workflow_variant"] = variation.workflow_variant
    updated["generated_at"] = reference_time.isoformat()
    updated["prediction_generated_at"] = reference_time.isoformat()
    updated.setdefault("daily_model_refresh", bool(variation.daily_model_refresh))
    updated.setdefault("model_refresh_et_date", replay_et_date(reference_time))
    return updated


def append_history_row(path: Path, row: dict[str, Any]) -> None:
    local_pd = require_pandas()
    history = load_history(path)
    extra = local_pd.DataFrame([row], columns=HISTORY_COLUMNS)
    if history.empty:
        combined = extra.copy()
    else:
        combined = local_pd.concat(
            [
                history.dropna(axis=1, how="all"),
                extra.dropna(axis=1, how="all"),
            ],
            ignore_index=True,
        )
        for column in HISTORY_COLUMNS:
            if column not in combined.columns:
                combined[column] = ""
    try:
        combined["timestamp_key"] = local_pd.to_datetime(
            combined["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except TypeError:
        combined["timestamp_key"] = local_pd.to_datetime(
            combined["timestamp"],
            utc=True,
            errors="coerce",
        )
    combined["workflow_key"] = combined["workflow_name"].fillna("").astype(str)
    combined = (
        combined.sort_values("timestamp_key")
        .drop_duplicates(subset=["timestamp_key", "workflow_key"], keep="last")
        .drop(columns=["timestamp_key", "workflow_key"])
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined[HISTORY_COLUMNS].to_csv(path, index=False)


def assert_no_missing_rows(
    variation: Variation,
    targets: list[pd.Timestamp],
) -> None:
    if not targets:
        return
    local_pd = require_pandas()
    history = load_history(variation.history_path)
    if history.empty:
        raise RuntimeError(
            f"{variation.label} has no history after backfill execution."
        )
    history = history.copy()
    try:
        history["timestamp_key"] = local_pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except TypeError:
        history["timestamp_key"] = local_pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="coerce",
        )
    history["workflow_key"] = history["workflow_name"].fillna("").astype(str)
    expected = {
        local_pd.Timestamp(target).floor("h").tz_convert("UTC")
        for target in targets
    }
    scoped = history[
        history["workflow_key"].eq(variation.workflow_name)
        & history["timestamp_key"].isin(expected)
    ]
    valid = scoped[
        scoped["predicted"].fillna("").astype(str).str.strip().ne("")
        & scoped["status"].fillna("").astype(str).str.lower().ne("missing")
    ]
    valid_targets = set(valid["timestamp_key"])
    missing_targets = sorted(expected - valid_targets)
    if missing_targets:
        formatted = ", ".join(target.isoformat() for target in missing_targets[:10])
        suffix = "" if len(missing_targets) <= 10 else f", ... ({len(missing_targets)} total)"
        raise RuntimeError(
            f"{variation.label} still has missing/unvalidated rows in the executed backfill range: "
            f"{formatted}{suffix}"
        )


def write_last_prediction(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def limit_manifest(
    manifest: dict[Variation, list[pd.Timestamp]],
    *,
    max_hours: int | None = None,
    max_total_targets: int | None = None,
) -> dict[Variation, list[pd.Timestamp]]:
    remaining = max_total_targets
    limited: dict[Variation, list[pd.Timestamp]] = {}
    for variation, targets in manifest.items():
        selected = targets[: max_hours or len(targets)]
        if remaining is not None:
            selected = selected[:remaining]
            remaining -= len(selected)
        limited[variation] = selected
    return limited


def iter_manifest_batches(
    manifest: dict[Variation, list[pd.Timestamp]],
    batch_size: int | None,
) -> list[dict[Variation, list[pd.Timestamp]]]:
    flattened = [
        (variation, target)
        for variation, targets in manifest.items()
        for target in targets
    ]
    if not flattened:
        return []
    size = batch_size or len(flattened)
    batches: list[dict[Variation, list[pd.Timestamp]]] = []
    for index in range(0, len(flattened), size):
        batch: dict[Variation, list[pd.Timestamp]] = {}
        for variation, target in flattened[index : index + size]:
            batch.setdefault(variation, []).append(target)
        batches.append(batch)
    return batches


def print_manifest(manifest: dict[Variation, list[pd.Timestamp]], max_hours: int | None = None) -> None:
    print(f"Safe default cutoff: {now_cutoff().isoformat()}")
    total = 0
    for variation, targets in manifest.items():
        shown = targets[: max_hours or len(targets)]
        total += len(shown)
        start = shown[0].isoformat() if shown else "n/a"
        end = shown[-1].isoformat() if shown else "n/a"
        suffix = "" if len(shown) == len(targets) else f" (limited from {len(targets)})"
        print(f"{variation.label}: {len(shown)} target hours{suffix}; {start} -> {end}")
    print(f"Total target hours to execute: {total}")


def execute_manifest(
    manifest: dict[Variation, list[pd.Timestamp]],
    only_families: set[str] | None = None,
    raw_cache: RawSnapshotCache | None = None,
) -> None:
    daily_bundles: dict[tuple[str, str, tuple[str, ...]], TrainedModelBundle] = {}
    live_replay_days_with_models: set[tuple[str, str]] = set()
    consolidated_targets: set[Any] = set()
    for variation, targets in manifest.items():
        if is_consolidated_variation(variation):
            consolidated_targets.update(targets)
            continue
        for target in targets:
            print(f"\n=== {variation.label}: {target.isoformat()} ===", flush=True)
            raw = fetch_raw_snapshot(target, cache=raw_cache)
            if not uses_local_replay_registry(variation):
                record = execute_live_btc_replay(
                    variation=variation,
                    raw=raw,
                    target_timestamp=target,
                    replay_days_with_models=live_replay_days_with_models,
                    only_families=only_families,
                )
                if record is None:
                    continue
                record = enrich_live_replay_record(variation, record, target)
            elif variation.daily_model_refresh:
                model_day = replay_et_date(replay_run_time(target))
                family_key = tuple(sorted(only_families or []))
                bundle_key = (variation.label, model_day, family_key)
                bundle = daily_bundles.get(bundle_key)
                trained_this_target = False
                if bundle is None and uses_local_replay_registry(variation):
                    bundle = load_model_bundle(variation, model_day, only_families)
                    if bundle is not None:
                        print(f"Loaded daily model bundle cache for {variation.label} ET day {model_day}.")
                replay_day_has_champions = (
                    bundle is not None
                    if uses_local_replay_registry(variation)
                    else bundle_key in daily_bundles
                )
                should_train = should_train_for_replay(
                    variation,
                    target,
                    bundle_exists=replay_day_has_champions,
                )
                if should_train:
                    trained_this_target = True
                    print(f"Training daily model bundle for {variation.label} ET day {model_day}.")
                    if uses_local_replay_registry(variation):
                        bundle = train_model_bundle(
                            variation=variation,
                            raw=raw,
                            target_timestamp=target,
                            only_families=only_families,
                        )
                        save_model_bundle(variation, model_day, only_families, bundle)
                    else:
                        bundle = train_registry_model_bundle(
                            variation=variation,
                            raw=raw,
                            target_timestamp=target,
                            only_families=only_families,
                        )
                elif uses_local_replay_registry(variation) and bundle is not None:
                    print(f"Reusing daily model bundle for {variation.label} ET day {model_day}.")
                elif not uses_local_replay_registry(variation):
                    print(
                        f"Loading registry champions for {variation.label} ET day {model_day} "
                        "using the live prediction-only path."
                    )
                    bundle = load_registry_model_bundle(
                        variation=variation,
                        raw=raw,
                        target_timestamp=target,
                        only_families=only_families,
                    )
                else:
                    raise RuntimeError(
                        f"{variation.label} has no same-day replay champions for {model_day} "
                        "and live workflow gating would not train at this target."
                    )
                daily_bundles[bundle_key] = bundle
                record = prediction_record_from_bundle(
                    variation=variation,
                    bundle=bundle,
                    raw=raw,
                    target_timestamp=target,
                    daily_model_refresh=trained_this_target,
                )
                if not trained_this_target:
                    log_prediction_only_record(
                        variation=variation,
                        record=record,
                        bundle=bundle,
                    )
            else:
                if uses_local_replay_registry(variation):
                    record = train_prediction_record(
                        variation=variation,
                        raw=raw,
                        target_timestamp=target,
                        only_families=only_families,
                    )
                else:
                    bundle = train_registry_model_bundle(
                        variation=variation,
                        raw=raw,
                        target_timestamp=target,
                        only_families=only_families,
                    )
                    record = prediction_record_from_bundle(
                        variation=variation,
                        bundle=bundle,
                        raw=raw,
                        target_timestamp=target,
                        daily_model_refresh=False,
                    )
            row = history_row_from_record(record, raw)
            if row is None:
                raise RuntimeError(f"No completed target candle available for {target.isoformat()}.")
            append_history_row(variation.history_path, row)
            if variation.last_prediction_path is not None:
                write_last_prediction(variation.last_prediction_path, record)
        assert_no_missing_rows(variation, targets)
    for target in sorted(consolidated_targets):
        execute_consolidated_replay(target)


def execute_live_btc_replay(
    *,
    variation: Variation,
    raw: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    replay_days_with_models: set[tuple[str, str]],
    only_families: set[str] | None = None,
) -> dict[str, Any] | None:
    if only_families:
        raise RuntimeError(
            "--only-family is not live-equivalent for BTC workflow replay. "
            "Run without --only-family when exact live semantics are required."
        )
    args = SimpleNamespace(reset_champion_from_challenger=False, force_refresh=False)
    reference_time = replay_run_time(target_timestamp)
    model_day = replay_et_date(reference_time)
    day_key = (variation.workflow_name, model_day)

    if variation.workflow_name == "hourly24":
        from src.btc_pipeline import main as hourly

        return hourly.execute_hourly_workflow(
            args,
            raw=raw,
            reference_time=reference_time,
        )

    if variation.workflow_name == "daily-hourly":
        from src.btc_pipeline import daily_main

        should_refresh = should_train_for_replay(
            variation,
            target_timestamp,
            bundle_exists=day_key in replay_days_with_models,
        )
        record = daily_main.execute_daily_workflow(
            args,
            raw=raw,
            reference_time=reference_time,
            replay_day_has_models=(day_key in replay_days_with_models),
        )
        if should_refresh and record is not None:
            replay_days_with_models.add(day_key)
        return record

    if variation.workflow_name == "market-hours-hourly":
        from src.btc_pipeline import market_hours_main

        return market_hours_main.execute_market_hours_workflow(
            args,
            raw=raw,
            reference_time=reference_time,
        )

    if variation.workflow_name == "market-hours-daily":
        from src.btc_pipeline import market_hours_daily_main

        should_refresh = should_train_for_replay(
            variation,
            target_timestamp,
            bundle_exists=day_key in replay_days_with_models,
        )
        record = market_hours_daily_main.execute_market_hours_daily_workflow(
            args,
            raw=raw,
            reference_time=reference_time,
            replay_day_has_models=(day_key in replay_days_with_models),
        )
        if should_refresh and record is not None:
            replay_days_with_models.add(day_key)
        return record

    raise RuntimeError(f"No live BTC replay executor for {variation.label}.")


def execute_consolidated_replay(target_timestamp: Any) -> None:
    local_pd = require_pandas()
    from pipelines.consolidated import main as consolidated

    reference_time = replay_run_time(target_timestamp)
    reference_time = local_pd.Timestamp(reference_time)
    print(
        f"\n=== Consolidated replay: target={local_pd.Timestamp(target_timestamp).isoformat()} "
        f"reference_time={reference_time.isoformat()} ===",
        flush=True,
    )
    args = SimpleNamespace(reset_champion_from_challenger=False)
    execution = consolidated.execute_consolidated_workflow(
        args,
        reference_time=reference_time,
    )
    consolidated.persist_execution_outputs(execution)


def collect_artifact_files() -> list[Path]:
    explicit_paths = [
        ROOT / "artifacts/btc/hourly/last_prediction.json",
        ROOT / "artifacts/btc/hourly/history.csv",
        ROOT / "artifacts/btc/daily/last_prediction.json",
        ROOT / "artifacts/btc/daily/history.csv",
        ROOT / "artifacts/btc/market_hours/last_prediction.json",
        ROOT / "artifacts/btc/market_hours/history.csv",
        ROOT / "artifacts/btc/market_hours_daily/last_prediction.json",
        ROOT / "artifacts/btc/market_hours_daily/history.csv",
        ROOT / "artifacts/consolidated/last_prediction.json",
        ROOT / "artifacts/consolidated/history.csv",
        ROOT / "artifacts/consolidated/workflow.log",
        ROOT / "artifacts/consolidated/comparison_summary.json",
        *REPORT_PATHS,
    ]
    glob_roots = [
        ROOT / "assets/btc",
        ROOT / "assets/consolidated",
        ROOT / "artifacts/backfill_model_cache",
        ROOT / "artifacts/newtest",
    ]
    files = [path for path in explicit_paths if path.exists()]
    for root in glob_roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    deduped: dict[str, Path] = {}
    for path in files:
        deduped[artifact_sync.normalize_git_path(str(path.relative_to(ROOT)))] = path
    return [deduped[key] for key in sorted(deduped)]


def publish_backfill_artifacts(commit_message: str) -> None:
    artifact_files = collect_artifact_files()
    print(f"Publishing {len(artifact_files)} backfill artifact files.")
    artifact_sync.publish_artifacts_to_origin(
        repo_root=ROOT,
        artifact_files=artifact_files,
        commit_message=commit_message,
        max_attempts=5,
    )


def finish_batch(*, skip_report: bool, publish_artifacts: bool, commit_message: str) -> None:
    if not skip_report:
        for report_path in generate_all_reports():
            print(f"Regenerated {report_path.relative_to(ROOT)}")
    if publish_artifacts:
        publish_backfill_artifacts(commit_message)


def main() -> int:
    args = parse_args()
    if args.execute and not args.max_hours:
        raise SystemExit("--execute requires --max-hours to bound training work.")
    if args.max_total_targets is not None and args.max_total_targets < 1:
        raise SystemExit("--max-total-targets must be at least 1 when provided.")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1 when provided.")
    if args.reset_newtest:
        reset_newtest_artifacts()
    manifest = build_manifest(args)
    should_limit_manifest = args.execute or args.max_hours is not None or args.max_total_targets is not None
    limited_manifest = limit_manifest(
        manifest,
        max_hours=args.max_hours if should_limit_manifest else None,
        max_total_targets=args.max_total_targets if should_limit_manifest else None,
    )
    print_manifest(limited_manifest if should_limit_manifest else manifest)
    if not args.execute:
        return 0
    batches = iter_manifest_batches(limited_manifest, args.batch_size)
    raw_cache = RawSnapshotCache()
    for batch_index, batch in enumerate(batches, start=1):
        print(f"\n=== Execute backfill batch {batch_index}/{len(batches)} ===", flush=True)
        execute_manifest(batch, set(args.only_family or []), raw_cache=raw_cache)
        finish_batch(
            skip_report=args.skip_report,
            publish_artifacts=args.publish_artifacts,
            commit_message=args.commit_message,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

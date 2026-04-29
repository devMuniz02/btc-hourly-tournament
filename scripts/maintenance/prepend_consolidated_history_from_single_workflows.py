#!/usr/bin/env python3
"""
Prepend older single-workflow BTC history rows into the consolidated history CSV.

This script backfills only historical rows that are older than the earliest
existing consolidated entry for the same track and are not already present in the
consolidated file. The generated rows are normalized to the consolidated CSV
schema so downstream dashboards and the trading simulator can consume them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.consolidated import config as consolidated_config
from pipelines.consolidated import io as consolidated_io
from src.btc_pipeline import main as tournament


@dataclass(frozen=True)
class SourceWorkflow:
    track: str
    source_path: Path
    workflow_name: str
    workflow_variant: str
    daily_model_refresh: bool


SOURCE_WORKFLOWS: tuple[SourceWorkflow, ...] = (
    SourceWorkflow(
        track="hourly_24h",
        source_path=ROOT / "artifacts" / "btc" / "hourly" / "history.csv",
        workflow_name="consolidated-hourly-24h",
        workflow_variant="hourly_24h_always_refresh",
        daily_model_refresh=False,
    ),
    SourceWorkflow(
        track="hourly_daily",
        source_path=ROOT / "artifacts" / "btc" / "daily" / "history.csv",
        workflow_name="consolidated-hourly-daily",
        workflow_variant="hourly_prediction_daily_refresh_at_midnight_et",
        daily_model_refresh=True,
    ),
    SourceWorkflow(
        track="market_hours",
        source_path=ROOT / "artifacts" / "btc" / "market_hours" / "history.csv",
        workflow_name="consolidated-market-hours",
        workflow_variant="hourly_prediction_market_hours_only",
        daily_model_refresh=False,
    ),
    SourceWorkflow(
        track="market_hours_daily",
        source_path=ROOT / "artifacts" / "btc" / "market_hours_daily" / "history.csv",
        workflow_name="consolidated-market-hours-daily",
        workflow_variant="market_hours_prediction_same_day_refresh",
        daily_model_refresh=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepend older single-workflow rows into artifacts/consolidated/history.csv."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview how many rows would be prepended without writing the consolidated CSV.",
    )
    parser.add_argument(
        "--skip-candle-refresh",
        action="store_true",
        help="Skip fetching BTC candles and leave reference/target/actual fields unchanged.",
    )
    return parser.parse_args()


def normalize_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    ts = pd.Timestamp(text)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalize_signal(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if text in {"UP", "DOWN"}:
        return text
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return ""
    return "UP" if int(float(numeric)) == 1 else "DOWN"


def normalize_result(value: Any) -> str:
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if text in {"WIN", "LOSS", "FAILED"}:
        return "LOSS" if text == "FAILED" else text
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return ""
    return "WIN" if int(float(numeric)) == 1 else "LOSS"


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def normalize_status(source_status: Any, predicted_signal: str, failed: bool) -> str:
    status = str(source_status).strip().lower()
    if failed:
        return "failed"
    if status in {"missing", "skipped"} or not predicted_signal:
        return "skipped"
    if status in {"validated", "success"}:
        return "success"
    return status or "success"


def maybe_float(value: Any) -> float | str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return float(numeric)


def serialize_model_predictions(value: Any) -> str:
    return consolidated_io.serialize_model_predictions(value if str(value).strip() else {})


def build_backfill_row(source: SourceWorkflow, raw: pd.Series) -> dict[str, Any] | None:
    target_ts = normalize_timestamp(raw.get("timestamp"))
    if target_ts is None:
        return None

    predicted_signal = normalize_signal(raw.get("predicted"))
    actual_signal = normalize_signal(raw.get("actual"))
    result = normalize_result(raw.get("result"))
    failed = normalize_bool(raw.get("failed"))
    status = normalize_status(raw.get("status"), predicted_signal, failed)

    reference_open = maybe_float(raw.get("reference_open"))
    reference_close = maybe_float(raw.get("reference_close"))
    target_open = maybe_float(raw.get("target_open"))
    target_close = maybe_float(raw.get("target_close"))
    reference_ts = (target_ts - pd.Timedelta(hours=1)).isoformat()

    return {
        "timestamp": target_ts.isoformat(),
        "track": source.track,
        "status": status,
        "target_candle_timestamp": target_ts.isoformat(),
        "prediction_generated_at": str(raw.get("prediction_generated_at", "")).strip(),
        "daily_model_refresh": source.daily_model_refresh,
        "best_champion_name": str(raw.get("best_champion_name", "")).strip(),
        "best_champion_family": str(raw.get("best_champion_family", "")).strip(),
        "best_champion_version": str(raw.get("best_champion_version", "")).strip(),
        "predicted_signal": predicted_signal,
        "probability_up": "",
        "model_accuracy": "",
        "model_f1": "",
        "promoted": False,
        "promotion_blocked": False,
        "registered_model_name": "",
        "actual": actual_signal,
        "result": result,
        "failed": failed,
        "reference_candle_timestamp": reference_ts,
        "reference_open": reference_open,
        "reference_close": reference_close,
        "target_open": target_open,
        "target_close": target_close,
        "model_predictions": serialize_model_predictions(raw.get("model_predictions", "{}")),
        "workflow_name": source.workflow_name,
        "workflow_variant": source.workflow_variant,
        "skipped_reason": "" if status != "skipped" else str(raw.get("status", "")).strip().lower(),
    }


def build_existing_key_map(history: pd.DataFrame) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for _, row in history.iterrows():
        track = str(row.get("track", "")).strip()
        target_ts = normalize_timestamp(row.get("target_candle_timestamp") or row.get("timestamp"))
        if track and target_ts is not None:
            keys.add((track, target_ts.isoformat()))
    return keys


def earliest_timestamp_by_track(history: pd.DataFrame) -> dict[str, pd.Timestamp]:
    result: dict[str, pd.Timestamp] = {}
    for _, row in history.iterrows():
        track = str(row.get("track", "")).strip()
        target_ts = normalize_timestamp(row.get("target_candle_timestamp") or row.get("timestamp"))
        if not track or target_ts is None:
            continue
        current = result.get(track)
        if current is None or target_ts < current:
            result[track] = target_ts
    return result


def collect_backfill_rows(history: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
    existing_keys = build_existing_key_map(history)
    earliest_by_track = earliest_timestamp_by_track(history)
    rows_to_add: list[dict[str, Any]] = []
    counts_by_track: dict[str, int] = {}

    for source in SOURCE_WORKFLOWS:
        if not source.source_path.exists():
            print(f"Skipping missing source file: {source.source_path}")
            continue

        source_frame = pd.read_csv(source.source_path)
        earliest_target = earliest_by_track.get(source.track)
        added = 0

        for _, raw in source_frame.iterrows():
            backfill_row = build_backfill_row(source, raw)
            if backfill_row is None:
                continue
            target_iso = backfill_row["target_candle_timestamp"]
            target_ts = normalize_timestamp(target_iso)
            if target_ts is None:
                continue
            if earliest_target is not None and target_ts >= earliest_target:
                continue
            key = (source.track, target_iso)
            if key in existing_keys:
                continue
            rows_to_add.append(backfill_row)
            existing_keys.add(key)
            added += 1

        counts_by_track[source.track] = added

    return rows_to_add, counts_by_track


def compute_candle_lookback_hours(history: pd.DataFrame) -> int:
    if history.empty:
        return 48
    target_timestamps = history["target_candle_timestamp"].apply(normalize_timestamp).dropna()
    if target_timestamps.empty:
        target_timestamps = history["timestamp"].apply(normalize_timestamp).dropna()
    if target_timestamps.empty:
        return 48
    oldest_target = min(target_timestamps)
    now_utc = pd.Timestamp.now(tz="UTC")
    lookback_hours = int(max((now_utc - oldest_target).total_seconds() // 3600 + 6, 48))
    return min(max(lookback_hours, 48), tournament.LOOKBACK_HOURS)


def refresh_with_btc_candles(history: pd.DataFrame) -> pd.DataFrame:
    lookback_hours = compute_candle_lookback_hours(history)
    min_candles = min(max(lookback_hours, 48), tournament.LOOKBACK_HOURS)
    print(f"Fetching BTC candles for {lookback_hours} hour(s) of lookback...")
    candles = tournament.fetch_ohlcv(limit=lookback_hours, min_candles=min_candles)
    refreshed = consolidated_io.backfill_history_with_candles(history, candles)
    return consolidated_io.dedupe_history_rows(refreshed)


def main() -> int:
    args = parse_args()

    history = consolidated_io.load_history()
    rows_to_add, counts_by_track = collect_backfill_rows(history)

    if rows_to_add:
        extra = pd.DataFrame(rows_to_add, columns=consolidated_io.HISTORY_COLUMNS)
        combined = pd.concat([extra, history], ignore_index=True)
        combined = consolidated_io.dedupe_history_rows(combined)
    else:
        print("No older missing consolidated history rows were found to prepend.")
        combined = consolidated_io.dedupe_history_rows(history)

    if args.skip_candle_refresh:
        refreshed = combined
    else:
        refreshed = refresh_with_btc_candles(combined)

    print("Backfill summary:")
    for track in sorted(counts_by_track):
        print(f"  {track}: {counts_by_track[track]} row(s)")
    print(f"  total: {len(rows_to_add)} row(s)")
    print(f"  output: {consolidated_config.HISTORY_PATH}")
    if not args.skip_candle_refresh:
        print("  candle refresh: enabled")

    if args.dry_run:
        print("Dry run requested. No files were written.")
        return 0

    consolidated_config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    refreshed.to_csv(consolidated_config.HISTORY_PATH, index=False)
    print("Consolidated history updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

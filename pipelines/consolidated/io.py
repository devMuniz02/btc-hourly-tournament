#!/usr/bin/env python3
"""
Local file I/O helpers for consolidated workflow outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config


HISTORY_COLUMNS = [
    "timestamp",
    "track",
    "status",
    "target_candle_timestamp",
    "prediction_generated_at",
    "daily_model_refresh",
    "best_champion_name",
    "best_champion_family",
    "best_champion_version",
    "predicted_signal",
    "probability_up",
    "model_accuracy",
    "model_f1",
    "promoted",
    "promotion_blocked",
    "registered_model_name",
    "actual",
    "result",
    "failed",
    "reference_candle_timestamp",
    "reference_open",
    "reference_close",
    "target_open",
    "target_close",
    "model_predictions",
    "workflow_name",
    "workflow_variant",
    "skipped_reason",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def serialize_model_predictions(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, separators=(",", ":"), sort_keys=True)


def ensure_history_schema(history: pd.DataFrame) -> pd.DataFrame:
    updated = history.copy()
    for column in HISTORY_COLUMNS:
        if column not in updated.columns:
            updated[column] = ""
    return updated[HISTORY_COLUMNS]


def load_history() -> pd.DataFrame:
    if not config.HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame = pd.read_csv(config.HISTORY_PATH)
    return ensure_history_schema(frame)


def dedupe_history_rows(history: pd.DataFrame) -> pd.DataFrame:
    updated = ensure_history_schema(history.copy())
    if updated.empty:
        return updated
    deduped = updated.drop_duplicates(subset=["track", "target_candle_timestamp"], keep="last")
    deduped = deduped.sort_values(["timestamp", "track"]).reset_index(drop=True)
    return ensure_history_schema(deduped)


def append_history_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    history = load_history()
    extra = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    combined = pd.concat([history, ensure_history_schema(extra)], ignore_index=True)
    combined = dedupe_history_rows(combined)
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.HISTORY_PATH, index=False)
    return combined


def normalize_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    timestamp = pd.Timestamp(text)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def coerce_numeric(value: Any) -> float | pd._libs.missing.NAType:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return pd.NA
    return float(numeric)


def get_resolved_hour_prices(
    indexed: pd.DataFrame,
    target_ts: pd.Timestamp,
) -> tuple[float, float] | None:
    reference_ts = target_ts - pd.Timedelta(hours=1)
    prior_ts = target_ts - pd.Timedelta(hours=2)
    if reference_ts not in indexed.index:
        return None

    resolved_close = float(indexed.loc[reference_ts, "close"])
    if prior_ts in indexed.index:
        resolved_open = float(indexed.loc[prior_ts, "close"])
    else:
        resolved_open = float(indexed.loc[reference_ts, "open"])
    return resolved_open, resolved_close


def backfill_history_with_candles(
    history: pd.DataFrame,
    candles: pd.DataFrame | None,
) -> pd.DataFrame:
    updated = dedupe_history_rows(history)
    if updated.empty or candles is None or candles.empty:
        return updated

    candle_frame = candles.copy()
    candle_frame["timestamp"] = pd.to_datetime(candle_frame["timestamp"], utc=True)
    indexed = candle_frame.set_index("timestamp")
    changed = False

    for idx, row in updated.iterrows():
        target_ts = normalize_timestamp(row.get("target_candle_timestamp") or row.get("timestamp"))
        if target_ts is None:
            continue

        resolved_prices = get_resolved_hour_prices(indexed, target_ts)
        if resolved_prices is not None:
            reference_open, reference_close = resolved_prices
            current_reference_open = coerce_numeric(row.get("reference_open"))
            current_reference_close = coerce_numeric(row.get("reference_close"))
            if pd.isna(current_reference_open) or float(current_reference_open) != reference_open:
                updated.at[idx, "reference_open"] = reference_open
                changed = True
            if pd.isna(current_reference_close) or float(current_reference_close) != reference_close:
                updated.at[idx, "reference_close"] = reference_close
                changed = True
            current_target_open = coerce_numeric(row.get("target_open"))
            if pd.isna(current_target_open) or float(current_target_open) != reference_close:
                updated.at[idx, "target_open"] = reference_close
                changed = True

        if target_ts in indexed.index:
            target_close = float(indexed.loc[target_ts, "close"])
            current_target_close = coerce_numeric(row.get("target_close"))
            if pd.isna(current_target_close) or float(current_target_close) != target_close:
                updated.at[idx, "target_close"] = target_close
                changed = True

        predicted_signal = str(row.get("predicted_signal", "")).strip().upper()
        if predicted_signal not in {"UP", "DOWN"}:
            continue

        current_reference_open = coerce_numeric(updated.at[idx, "reference_open"])
        current_reference_close = coerce_numeric(updated.at[idx, "reference_close"])
        if pd.isna(current_reference_open) or pd.isna(current_reference_close):
            continue

        actual_signal = "UP" if float(current_reference_close) > float(current_reference_open) else "DOWN"
        current_actual = str(row.get("actual", "")).strip().upper()
        if current_actual != actual_signal:
            updated.at[idx, "actual"] = actual_signal
            changed = True

        resolved_result = "WIN" if predicted_signal == actual_signal else "LOSS"
        current_result = str(row.get("result", "")).strip().upper()
        if current_result != resolved_result:
            updated.at[idx, "result"] = resolved_result
            changed = True

    if changed:
        updated = dedupe_history_rows(updated)
        config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(config.HISTORY_PATH, index=False)
    return updated

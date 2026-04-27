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


def append_history_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    history = load_history()
    extra = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    combined = pd.concat([history, ensure_history_schema(extra)], ignore_index=True)
    combined = ensure_history_schema(combined)
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.HISTORY_PATH, index=False)
    return combined

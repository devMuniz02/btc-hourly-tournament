#!/usr/bin/env python3
"""
Render consolidated dashboards using the shared workflow dashboard layout.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

import main as tournament
from pipelines.consolidated import config, io


DASHBOARD_VARIANTS = (
    {
        "path": config.DASHBOARD_PATH,
        "title": "BTC Consolidated Hourly + Daily Dashboard",
        "subtitle": "Combined view for hourly_24h and hourly_daily tracks",
        "signal_mode": "normal",
        "track_ids": ("hourly_24h", "hourly_daily"),
    },
    {
        "path": config.DASHBOARD_REVERSE_PATH,
        "title": "BTC Consolidated Reverse Hourly + Daily Dashboard",
        "subtitle": "Reverse actions for hourly_24h and hourly_daily tracks",
        "signal_mode": "reverse",
        "track_ids": ("hourly_24h", "hourly_daily"),
    },
    {
        "path": config.DASHBOARD_MARKET_HOURS_PATH,
        "title": "BTC Consolidated Market Hours Dashboard",
        "subtitle": "Combined view for market_hours and market_hours_daily tracks",
        "signal_mode": "normal",
        "track_ids": ("market_hours", "market_hours_daily"),
    },
    {
        "path": config.DASHBOARD_MARKET_HOURS_REVERSE_PATH,
        "title": "BTC Consolidated Reverse Market Hours Dashboard",
        "subtitle": "Reverse actions for market_hours and market_hours_daily tracks",
        "signal_mode": "reverse",
        "track_ids": ("market_hours", "market_hours_daily"),
    },
)

_STANDARD_DASHBOARD: ModuleType | None = None


def load_last_prediction() -> dict[str, Any] | None:
    return io.read_json(config.LAST_PREDICTION_PATH)


def load_standard_dashboard_module() -> ModuleType:
    global _STANDARD_DASHBOARD
    if _STANDARD_DASHBOARD is not None:
        return _STANDARD_DASHBOARD

    module_path = ROOT / "validate_dashboard.py"
    spec = importlib.util.spec_from_file_location("shared_validate_dashboard", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared dashboard module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HISTORY_PATH = config.HISTORY_PATH
    _STANDARD_DASHBOARD = module
    return module


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


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    return bool(value)


def signal_to_label(value: Any) -> int | str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if text == "UP":
        return 1
    if text == "DOWN":
        return 0
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return ""
    return int(float(numeric))


def label_to_signal(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        text = str(value).strip().upper()
        if text in {"UP", "DOWN"}:
            return text
        return ""
    return "UP" if int(float(numeric)) == 1 else "DOWN"


def result_to_int(value: Any) -> int | str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if text == "WIN":
        return 1
    if text == "LOSS":
        return 0
    if text == "FAILED":
        return 0
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return ""
    return int(float(numeric))


def result_to_text(value: Any) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return ""
    return "WIN" if int(float(numeric)) == 1 else "LOSS"


def build_standard_history(history: pd.DataFrame) -> pd.DataFrame:
    dashboard = load_standard_dashboard_module()
    source = io.ensure_history_schema(history)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        timestamp = normalize_timestamp(row.get("target_candle_timestamp") or row.get("timestamp"))
        if timestamp is None:
            continue
        status = str(row.get("status", "")).strip().lower()
        failed = coerce_bool(row.get("failed")) or status == "failed"
        rows.append(
            {
                "timestamp": timestamp,
                "predicted": "FAILED" if failed else signal_to_label(row.get("predicted_signal")),
                "actual": signal_to_label(row.get("actual")),
                "result": result_to_int(row.get("result")),
                "failed": 1 if failed else 0,
                "status": "failed" if failed else ("missing" if status == "skipped" else "validated"),
                "reference_open": coerce_numeric(row.get("reference_open")),
                "reference_close": coerce_numeric(row.get("reference_close")),
                "target_open": coerce_numeric(row.get("target_open")),
                "target_close": coerce_numeric(row.get("target_close")),
                "model_predictions": row.get("model_predictions", "{}"),
                "best_champion_name": row.get("best_champion_name", ""),
                "best_champion_family": row.get("best_champion_family", ""),
                "best_champion_version": row.get("best_champion_version", ""),
                "workflow_name": row.get("workflow_name", ""),
                "workflow_variant": row.get("workflow_variant", ""),
                "daily_model_refresh": coerce_bool(row.get("daily_model_refresh")),
                "model_refresh_et_date": "",
                "prediction_generated_at": row.get("prediction_generated_at", ""),
            }
        )
    frame = pd.DataFrame(rows, columns=dashboard.HISTORY_COLUMNS)
    return dashboard.ensure_history_schema(frame)


def build_history_key(row: pd.Series) -> tuple[str, str, str]:
    timestamp = normalize_timestamp(row.get("target_candle_timestamp") or row.get("timestamp"))
    return (
        "" if timestamp is None else timestamp.isoformat(),
        str(row.get("workflow_name", "")),
        str(row.get("workflow_variant", "")),
    )


def refresh_history_outcomes(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    standard = load_standard_dashboard_module()
    standard_history = build_standard_history(history)
    standard_history = standard.normalize_history_labels(standard_history)

    try:
        candles = fetch_validation_candles(history)
    except Exception as exc:
        print(f"Could not fetch validation candles. Continuing with stored history only: {exc}")
        candles = None

    if candles is not None:
        standard_history = standard.normalize_history_labels(
            standard.backfill_recent_history_prices(
                standard_history,
                candles,
            )
        )

    updated = io.ensure_history_schema(history.copy())
    updated["actual"] = updated["actual"].astype("object")
    updated["result"] = updated["result"].astype("object")
    standard_by_key = {
        build_history_key(row): row
        for _, row in standard_history.iterrows()
    }
    changed = False

    for index, row in updated.iterrows():
        standard_row = standard_by_key.get(build_history_key(row))
        if standard_row is None:
            continue

        for column in ("reference_open", "reference_close", "target_open", "target_close"):
            current_value = coerce_numeric(row.get(column))
            new_value = coerce_numeric(standard_row.get(column))
            if pd.isna(new_value):
                continue
            if pd.isna(current_value) or float(current_value) != float(new_value):
                updated.at[index, column] = float(new_value)
                changed = True

        predicted_present = bool(str(row.get("predicted_signal", "")).strip()) or str(row.get("status", "")).strip().lower() == "success"
        if not predicted_present:
            continue

        actual_signal = label_to_signal(standard_row.get("actual"))
        if actual_signal and str(row.get("actual", "")).strip().upper() != actual_signal:
            updated.at[index, "actual"] = actual_signal
            changed = True

        resolved_result = result_to_text(standard_row.get("result"))
        if resolved_result and str(row.get("result", "")).strip().upper() != resolved_result:
            updated.at[index, "result"] = resolved_result
            changed = True

    if changed:
        config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(config.HISTORY_PATH, index=False)
    return updated


def fetch_validation_candles(history: pd.DataFrame) -> pd.DataFrame | None:
    if history.empty:
        return None
    unresolved = history[
        (history["status"] == "success")
        & (history["actual"].fillna("").astype(str).str.strip() == "")
    ].copy()
    if unresolved.empty:
        return None
    target_timestamps = unresolved["target_candle_timestamp"].apply(normalize_timestamp).dropna()
    if target_timestamps.empty:
        return None
    oldest_target = min(target_timestamps)
    now_utc = pd.Timestamp.now(tz="UTC")
    lookback_hours = int(max((now_utc - oldest_target).total_seconds() // 3600 + 6, 48))
    lookback_hours = min(max(lookback_hours, 48), tournament.LOOKBACK_HOURS)
    return tournament.fetch_ohlcv(limit=lookback_hours, min_candles=min(lookback_hours, 5000))


def filter_history_by_track_ids(
    history: pd.DataFrame,
    track_ids: tuple[str, ...],
) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    filtered = history[history["track"].isin(track_ids)].copy()
    return filtered.sort_values(["timestamp", "track"]).reset_index(drop=True)


def select_prediction_track(
    prediction_payload: dict[str, Any] | None,
    *,
    track_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    if prediction_payload is None:
        return None
    tracks = prediction_payload.get("tracks")
    if not isinstance(tracks, dict):
        return None

    candidates: list[tuple[pd.Timestamp, int, dict[str, Any]]] = []
    for order, track in enumerate(config.TRACKS):
        if track.id not in track_ids:
            continue
        payload = tracks.get(track.id)
        if not isinstance(payload, dict) or payload.get("status") != "success":
            continue
        record = payload.get("prediction_record")
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        target_timestamp = normalize_timestamp(record.get("target_candle_timestamp"))
        if target_timestamp is None:
            continue
        candidates.append((target_timestamp, -order, payload))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def build_standard_prediction_record(
    prediction_payload: dict[str, Any] | None,
    *,
    track_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    selected_track = select_prediction_track(
        prediction_payload,
        track_ids=track_ids,
    )
    if selected_track is None:
        return None

    record = dict(selected_track["prediction_record"])
    return {
        "status": record.get("status", "success"),
        "generated_at": record.get("generated_at", prediction_payload.get("generated_at") if prediction_payload else ""),
        "reference_candle_timestamp": record.get("reference_candle_timestamp"),
        "target_candle_timestamp": record.get("target_candle_timestamp"),
        "reference_open": record.get("reference_open"),
        "reference_close": record.get("reference_close"),
        "target_open": record.get("target_open"),
        "target_close": record.get("target_close"),
        "predicted_label": signal_to_label(record.get("predicted_signal")),
        "predicted_signal": record.get("predicted_signal", ""),
        "probability_up": record.get("probability_up"),
        "model_accuracy": record.get("model_accuracy"),
        "model_f1": record.get("model_f1"),
        "model_predictions": record.get("model_predictions", {}),
        "best_champion_name": record.get("best_champion_name", ""),
        "best_champion_family": record.get("best_champion_family", ""),
        "best_champion_version": record.get("best_champion_version", ""),
        "workflow_name": record.get("workflow_name", config.WORKFLOW_NAME),
        "workflow_variant": record.get("workflow_variant", config.WORKFLOW_VARIANT),
        "daily_model_refresh": bool(record.get("daily_model_refresh", False)),
        "model_refresh_et_date": "",
        "prediction_generated_at": record.get("prediction_generated_at", record.get("generated_at", "")),
    }


def render_dashboard_variants(
    history: pd.DataFrame,
    prediction_payload: dict[str, Any] | None,
) -> None:
    standard = load_standard_dashboard_module()
    standard_history = build_standard_history(history)

    for variant in DASHBOARD_VARIANTS:
        track_ids = tuple(variant.get("track_ids", ()))
        variant_history_source = filter_history_by_track_ids(history, track_ids)
        variant_standard_history = build_standard_history(variant_history_source)
        variant_prediction = build_standard_prediction_record(
            prediction_payload,
            track_ids=track_ids,
        )
        standard_variant = {
            **variant,
            "filter_mode": "all",
        }
        variant_history, stats, variant_prediction = standard.build_dashboard_variant_view(
            variant_standard_history,
            variant_prediction,
            standard_variant,
        )
        standard.render_dashboard(
            variant_history,
            stats,
            variant_prediction,
            dashboard_path=variant["path"],
            dashboard_title=variant.get("title"),
            dashboard_subtitle=variant.get("subtitle"),
        )


def main() -> None:
    history = io.load_history()
    history = refresh_history_outcomes(history)
    prediction_payload = load_last_prediction()
    render_dashboard_variants(history, prediction_payload)


if __name__ == "__main__":
    main()

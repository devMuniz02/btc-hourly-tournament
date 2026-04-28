#!/usr/bin/env python3
"""
Validate the prior BTC directional prediction and render a simple dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from mlflow import MlflowClient

import main as tournament
from market_hours_common import is_allowed_prediction_target_timestamp


HISTORY_PATH = Path("history.csv")
DASHBOARD_PATH = Path("assets/dashboard.png")
LOCAL_LAST_PREDICTION_PATH = Path("last_prediction.json")
DASHBOARD_TITLE = "BTC Directional Bot Validation Dashboard"
DASHBOARD_SUBTITLE = ""
DASHBOARD_VARIANTS = [
    {
        "path": DASHBOARD_PATH,
        "title": DASHBOARD_TITLE,
        "subtitle": DASHBOARD_SUBTITLE,
        "signal_mode": "normal",
        "filter_mode": "all",
    },
    {
        "path": Path("assets/dashboard_reverse.png"),
        "title": "BTC Directional Bot Reverse Dashboard",
        "subtitle": "Reverse actions from the same hourly predictions",
        "signal_mode": "reverse",
        "filter_mode": "all",
    },
    {
        "path": Path("assets/dashboard_market_hours_from_24h.png"),
        "title": "BTC Directional Bot Market Hours Dashboard",
        "subtitle": "Filtered to ET market-hours target candles only",
        "signal_mode": "normal",
        "filter_mode": "market_hours",
    },
    {
        "path": Path("assets/dashboard_market_hours_from_24h_reverse.png"),
        "title": "BTC Directional Bot Reverse Market Hours Dashboard",
        "subtitle": "Reverse actions filtered to ET market-hours target candles",
        "signal_mode": "reverse",
        "filter_mode": "market_hours",
    },
]
EASTERN_TZ = ZoneInfo("America/New_York")
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


def configure_tracking() -> tuple[MlflowClient, str, str]:
    registered_model_name = tournament.configure_tracking()
    experiment_name = tournament.build_experiment_name()
    return MlflowClient(), registered_model_name, experiment_name


def load_last_prediction() -> dict[str, Any] | None:
    if LOCAL_LAST_PREDICTION_PATH.exists():
        return json.loads(LOCAL_LAST_PREDICTION_PATH.read_text(encoding="utf-8"))
    return None


def ensure_history_schema(history: pd.DataFrame) -> pd.DataFrame:
    updated = history.copy()
    for column in HISTORY_COLUMNS:
        if column not in updated.columns:
            updated[column] = (
                ""
                if column.startswith("best_champion")
                or column in {"model_predictions", "workflow_name", "workflow_variant", "model_refresh_et_date", "prediction_generated_at"}
                else pd.NA
            )
    return updated[HISTORY_COLUMNS]


def build_history_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    return ensure_history_schema(frame)


def append_history_frame(history: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    base_records = ensure_history_schema(history).to_dict("records")
    extra_records = ensure_history_schema(extra).to_dict("records")
    if not base_records:
        return build_history_frame(extra_records)
    if not extra_records:
        return build_history_frame(base_records)
    return build_history_frame([*base_records, *extra_records])


def parse_model_predictions(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_model_order(
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
) -> list[str]:
    families: set[str] = set()
    if prediction_record is not None:
        families.update(parse_model_predictions(prediction_record.get("model_predictions")).keys())
    if not history.empty and "model_predictions" in history.columns:
        for value in history["model_predictions"]:
            families.update(parse_model_predictions(value).keys())
    return sorted(families)


def get_model_display_name(
    family: str,
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
) -> str:
    if prediction_record is not None:
        payload = parse_model_predictions(prediction_record.get("model_predictions")).get(family)
        if payload and payload.get("name"):
            return str(payload["name"])
    if not history.empty and "model_predictions" in history.columns:
        for value in history["model_predictions"]:
            payload = parse_model_predictions(value).get(family)
            if payload and payload.get("name"):
                return str(payload["name"])
    return family.upper()


def compute_validation_lookback_hours(
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
) -> int:
    lookback_hours = 48

    if not history.empty:
        oldest_ts = pd.Timestamp(history["timestamp"].min())
        now_utc = pd.Timestamp.utcnow()
        history_hours = int(max((now_utc - oldest_ts).total_seconds() // 3600 + 4, 48))
        lookback_hours = max(lookback_hours, history_hours)

    if prediction_record is not None:
        reference_ts = pd.Timestamp(prediction_record["reference_candle_timestamp"])
        target_ts = pd.Timestamp(prediction_record["target_candle_timestamp"])
        now_utc = pd.Timestamp.utcnow()
        prediction_hours = int(
            max((now_utc - min(reference_ts, target_ts)).total_seconds() // 3600 + 4, 8)
        )
        lookback_hours = max(lookback_hours, prediction_hours)

    return min(lookback_hours, tournament.LOOKBACK_HOURS)


def fetch_validation_candles(
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
) -> pd.DataFrame:
    lookback_hours = compute_validation_lookback_hours(history, prediction_record)
    min_candles = min(max(lookback_hours, 8), tournament.LOOKBACK_HOURS)
    return tournament.fetch_ohlcv(limit=lookback_hours, min_candles=min_candles)


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


def resolve_actual_direction(
    candles: pd.DataFrame,
    prediction_record: dict[str, Any],
) -> tuple[int, pd.Timestamp, float, float, float] | None:
    target_ts = pd.Timestamp(prediction_record["target_candle_timestamp"])
    now_utc = pd.Timestamp.now(tz="UTC")

    # The target hour is resolved as soon as the next hour opens.
    if now_utc < target_ts:
        return None

    candle_frame = candles.copy()
    candle_frame["timestamp"] = pd.to_datetime(candle_frame["timestamp"], utc=True)
    indexed = candle_frame.set_index("timestamp")
    resolved_prices = get_resolved_hour_prices(indexed, target_ts)
    if resolved_prices is None:
        return None

    reference_open, reference_close = resolved_prices
    target_open = reference_close
    actual_label = int(reference_close > reference_open)
    return actual_label, target_ts, reference_open, reference_close, target_open


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.read_csv(HISTORY_PATH)
    if history.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    original_columns = list(history.columns)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    if "failed" not in history.columns:
        history["failed"] = 0
    if "status" not in history.columns:
        history["status"] = "validated"
    history = ensure_history_schema(history)
    if original_columns != HISTORY_COLUMNS:
        history.to_csv(HISTORY_PATH, index=False)
    return history


def get_prediction_timestamp(prediction_record: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(
        prediction_record.get("target_candle_timestamp")
        or prediction_record.get("generated_at")
    )


def prediction_already_recorded(history: pd.DataFrame, prediction_record: dict[str, Any]) -> bool:
    if history.empty:
        return False
    prediction_timestamp = get_prediction_timestamp(prediction_record)
    matching_rows = history[history["timestamp"] == prediction_timestamp]
    if matching_rows.empty:
        return False

    for _, row in matching_rows.iterrows():
        status = str(row.get("status", "")).strip().lower()
        predicted_value = row.get("predicted")
        predicted_text = "" if pd.isna(predicted_value) else str(predicted_value).strip()

        if status == "failed":
            return True
        if status == "validated" and predicted_text != "":
            return True

    return False


def upsert_history_row(history: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    updated = ensure_history_schema(history.copy())
    if not updated.empty:
        updated = updated[updated["timestamp"] != row["timestamp"]]
    row_frame = build_history_frame([row])
    if updated.empty:
        updated = row_frame
    else:
        updated = append_history_frame(updated, row_frame)
    updated = updated.sort_values("timestamp").reset_index(drop=True)
    updated.to_csv(HISTORY_PATH, index=False)
    return updated


def remove_incomplete_validations(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    now_utc = pd.Timestamp.now(tz="UTC")
    keep_mask = ~(
        (history["status"] == "validated")
        & (history["timestamp"] > now_utc)
    )
    cleaned = history.loc[keep_mask].copy()
    if len(cleaned) != len(history):
        cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
        cleaned.to_csv(HISTORY_PATH, index=False)
    return cleaned


def ensure_recent_history_slots(history: pd.DataFrame, hours: int = 10) -> pd.DataFrame:
    now_utc = pd.Timestamp.now(tz="UTC")
    latest_available_target = now_utc.floor("h") - pd.Timedelta(hours=1)
    expected_timestamps = [
        latest_available_target - pd.Timedelta(hours=offset)
        for offset in range(hours)
    ]

    updated = history.copy()
    existing_timestamps = set()
    if not updated.empty:
        existing_timestamps = {
            pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
            for ts in updated["timestamp"]
        }

    missing_rows: list[dict[str, Any]] = []
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

    missing_frame = build_history_frame(missing_rows)
    if updated.empty:
        updated = missing_frame
    else:
        updated = append_history_frame(updated, missing_frame)
    updated = updated.sort_values("timestamp").reset_index(drop=True)
    updated.to_csv(HISTORY_PATH, index=False)
    return updated


def compute_stats(history: pd.DataFrame) -> dict[str, int]:
    if history.empty:
        return {
            "total_predictions": 0,
            "total_correct": 0,
            "total_failed": 0,
            "last_24h_predictions": 0,
            "last_24h_correct": 0,
            "last_24h_failed": 0,
            "total_accuracy_pct": 0.0,
            "last_24h_accuracy_pct": 0.0,
        }

    now = pd.Timestamp.utcnow()
    last_24h_cutoff = now - pd.Timedelta(hours=24)
    counted = history[
        (history["status"] != "missing")
        & (history["predicted"].fillna("").astype(str) != "")
    ]
    recent = counted[counted["timestamp"] >= last_24h_cutoff]
    total_predictions = int(len(counted))
    total_correct = int(pd.to_numeric(counted["result"], errors="coerce").fillna(0).sum())
    total_failed = int(pd.to_numeric(counted["failed"], errors="coerce").fillna(0).sum())
    last_24h_predictions = int(len(recent))
    last_24h_correct = int(pd.to_numeric(recent["result"], errors="coerce").fillna(0).sum())
    last_24h_failed = int(pd.to_numeric(recent["failed"], errors="coerce").fillna(0).sum())
    total_scored = max(total_predictions - total_failed, 0)
    last_24h_scored = max(last_24h_predictions - last_24h_failed, 0)
    return {
        "total_predictions": total_predictions,
        "total_correct": total_correct,
        "total_failed": total_failed,
        "last_24h_predictions": last_24h_predictions,
        "last_24h_correct": last_24h_correct,
        "last_24h_failed": last_24h_failed,
        "total_accuracy_pct": (
            (total_correct / total_scored) * 100 if total_scored else 0.0
        ),
        "last_24h_accuracy_pct": (
            (last_24h_correct / last_24h_scored) * 100 if last_24h_scored else 0.0
        ),
    }


def backfill_recent_history_prices(history: pd.DataFrame, candles: pd.DataFrame | None = None) -> pd.DataFrame:
    if history.empty:
        return history

    if candles is None:
        return history

    candle_frame = candles.copy()
    candle_frame["timestamp"] = pd.to_datetime(candle_frame["timestamp"], utc=True)
    indexed = candle_frame.set_index("timestamp")

    updated = history.copy()
    changed = False
    for idx, row in updated.iterrows():
        target_ts = pd.Timestamp(row["timestamp"])
        resolved_prices = get_resolved_hour_prices(indexed, target_ts)
        if resolved_prices is not None:
            reference_open, reference_close = resolved_prices
            if pd.isna(row["reference_open"]) or float(row["reference_open"]) != reference_open:
                updated.at[idx, "reference_open"] = reference_open
                changed = True
            if pd.isna(row["reference_close"]) or float(row["reference_close"]) != reference_close:
                updated.at[idx, "reference_close"] = reference_close
                changed = True
            target_open = reference_close
            if pd.isna(row["target_open"]) or float(row["target_open"]) != target_open:
                updated.at[idx, "target_open"] = target_open
                changed = True
        if target_ts in indexed.index:
            target_close = float(indexed.loc[target_ts, "close"])
            if pd.isna(row["target_close"]) or float(row["target_close"]) != target_close:
                updated.at[idx, "target_close"] = target_close
                changed = True

    if changed:
        updated = updated.sort_values("timestamp").reset_index(drop=True)
        updated.to_csv(HISTORY_PATH, index=False)
    return updated


def normalize_history_labels(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    updated = history.copy()
    changed = False
    for idx, row in updated.iterrows():
        if int(row.get("failed", 0)) == 1:
            continue
        if pd.isna(row["reference_open"]) or pd.isna(row["reference_close"]):
            continue

        actual_label = int(float(row["reference_close"]) > float(row["reference_open"]))
        predicted_value = row["predicted"]
        predicted_text = "" if pd.isna(predicted_value) else str(predicted_value).strip()
        if pd.isna(row["actual"]) or str(row["actual"]).strip() == "" or int(float(row["actual"])) != actual_label:
            updated.at[idx, "actual"] = actual_label
            changed = True

        if predicted_text == "" or predicted_text.upper() == "FAILED":
            if str(row.get("status", "")) == "validated" and (pd.isna(row["result"]) or str(row["result"]).strip() == ""):
                continue
            if str(row.get("status", "")) == "missing":
                continue
            continue

        predicted_label = int(float(predicted_value))
        result = int(predicted_label == actual_label)

        current_result = row["result"]
        if pd.isna(current_result) or str(current_result).strip() == "" or int(float(current_result)) != result:
            updated.at[idx, "result"] = result
            if str(row.get("status", "")) == "missing":
                updated.at[idx, "status"] = "validated"
            changed = True

    if changed:
        updated.to_csv(HISTORY_PATH, index=False)
    return updated


def normalize_prediction_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    timestamp = pd.Timestamp(text)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def should_include_market_hours_timestamp(value: Any) -> bool:
    timestamp = normalize_prediction_timestamp(value)
    if timestamp is None:
        return False
    return is_allowed_prediction_target_timestamp(timestamp)


def filter_history_for_market_hours(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    filtered = history[
        history["timestamp"].apply(should_include_market_hours_timestamp)
    ].copy()
    return filtered.sort_values("timestamp").reset_index(drop=True)


def invert_binary_label(value: Any) -> Any:
    if value is None or pd.isna(value):
        return value
    text = str(value).strip()
    if not text or text.upper() == "FAILED":
        return value
    try:
        numeric = int(float(text))
    except ValueError:
        return value
    return 1 - numeric


def invert_signal_text(value: Any) -> Any:
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    upper = text.upper()
    if "UP" in upper:
        return "DOWN"
    if "DOWN" in upper:
        return "UP"
    return value


def transform_model_predictions_payload(
    value: Any,
    *,
    reverse: bool,
) -> str:
    payloads = parse_model_predictions(value)
    if not payloads:
        if isinstance(value, str):
            return value
        return json.dumps({}, sort_keys=True)

    transformed: dict[str, dict[str, Any]] = {}
    for family, payload in payloads.items():
        updated = dict(payload)
        if reverse:
            updated["predicted_label"] = invert_binary_label(updated.get("predicted_label"))
            updated["predicted_signal"] = invert_signal_text(updated.get("predicted_signal"))
            probability_up = pd.to_numeric(updated.get("probability_up"), errors="coerce")
            if not pd.isna(probability_up):
                updated["probability_up"] = float(1.0 - float(probability_up))
        transformed[family] = updated
    return json.dumps(transformed, sort_keys=True)


def transform_prediction_record(
    prediction_record: dict[str, Any] | None,
    *,
    reverse: bool,
    market_hours_only: bool,
) -> dict[str, Any] | None:
    if prediction_record is None:
        return None
    target_timestamp = normalize_prediction_timestamp(
        prediction_record.get("target_candle_timestamp")
        or prediction_record.get("generated_at")
    )
    if market_hours_only and not should_include_market_hours_timestamp(target_timestamp):
        return None

    updated = dict(prediction_record)
    if reverse:
        updated["predicted_label"] = invert_binary_label(updated.get("predicted_label"))
        updated["predicted_signal"] = invert_signal_text(updated.get("predicted_signal"))
        probability_up = pd.to_numeric(updated.get("probability_up"), errors="coerce")
        if not pd.isna(probability_up):
            updated["probability_up"] = float(1.0 - float(probability_up))
    updated["model_predictions"] = transform_model_predictions_payload(
        updated.get("model_predictions"),
        reverse=reverse,
    )
    return updated


def transform_history_for_dashboard(
    history: pd.DataFrame,
    *,
    reverse: bool,
    market_hours_only: bool,
) -> pd.DataFrame:
    transformed = ensure_history_schema(history.copy())
    if market_hours_only:
        transformed = filter_history_for_market_hours(transformed)

    if transformed.empty or not reverse:
        return transformed

    predicted_values = transformed["predicted"].copy()
    failed_mask = pd.to_numeric(transformed["failed"], errors="coerce").fillna(0) == 1
    missing_mask = transformed["status"].fillna("").astype(str).str.lower() == "missing"
    reversible_mask = ~failed_mask & ~missing_mask

    transformed.loc[reversible_mask, "predicted"] = predicted_values.loc[reversible_mask].apply(
        invert_binary_label
    )

    actual_numeric = pd.to_numeric(transformed["actual"], errors="coerce")
    predicted_numeric = pd.to_numeric(transformed["predicted"], errors="coerce")
    reversible_result_mask = reversible_mask & actual_numeric.notna() & predicted_numeric.notna()
    transformed.loc[reversible_result_mask, "result"] = (
        predicted_numeric.loc[reversible_result_mask].astype(int)
        == actual_numeric.loc[reversible_result_mask].astype(int)
    ).astype(int)
    transformed["model_predictions"] = transformed["model_predictions"].apply(
        lambda value: transform_model_predictions_payload(value, reverse=True)
    )
    return transformed.sort_values("timestamp").reset_index(drop=True)


def build_dashboard_variant_view(
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any] | None]:
    market_hours_only = variant.get("filter_mode") == "market_hours"
    reverse = variant.get("signal_mode") == "reverse"
    variant_history = transform_history_for_dashboard(
        history,
        reverse=reverse,
        market_hours_only=market_hours_only,
    )
    variant_prediction = transform_prediction_record(
        prediction_record,
        reverse=reverse,
        market_hours_only=market_hours_only,
    )
    stats = compute_stats(variant_history)
    return variant_history, stats, variant_prediction


def format_dual_time(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return "--"
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    eastern = timestamp.tz_convert(EASTERN_TZ)
    return (
        f"{timestamp.strftime('%m-%d %H:%M')} UTC\n"
        f"{eastern.strftime('%m-%d %I:%M %p')} ET"
    )


def format_current_dual_time() -> str:
    now_utc = pd.Timestamp.now(tz="UTC")
    now_et = now_utc.tz_convert(EASTERN_TZ)
    return (
        f"Current Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        f" | {now_et.strftime('%Y-%m-%d %I:%M:%S %p')} ET"
    )


def compute_table_col_widths(
    headers: list[str],
    rows: list[list[Any]],
    *,
    min_width: float = 0.08,
    max_width: float = 0.24,
    header_weight: float = 1.0,
) -> list[float]:
    if not headers:
        return []

    weights: list[float] = []
    for col_idx, header in enumerate(headers):
        max_len = len(str(header)) * header_weight
        for row in rows:
            if col_idx >= len(row):
                continue
            value = str(row[col_idx])
            line_len = max((len(line) for line in value.splitlines()), default=0)
            max_len = max(max_len, line_len)
        weights.append(max(max_len, 4))

    total = float(sum(weights)) or 1.0
    widths = [weight / total for weight in weights]
    widths = [min(max(width, min_width), max_width) for width in widths]
    normalized_total = sum(widths) or 1.0
    return [width / normalized_total for width in widths]


def render_dashboard(
    history: pd.DataFrame,
    stats: dict[str, int],
    prediction_record: dict[str, Any] | None,
    *,
    dashboard_path: Path | None = None,
    dashboard_title: str | None = None,
    dashboard_subtitle: str | None = None,
) -> None:
    history = ensure_history_schema(history)
    model_families = get_model_order(history, prediction_record)
    model_headers = {
        family: get_model_display_name(family, history, prediction_record)
        for family in model_families
    }
    color_cycle = ["#1f3c4d", "#c44536", "#1b7f4a", "#946846", "#6b5b95", "#d17a22"]

    def to_arrow(value: Any) -> str:
        if value == "" or pd.isna(value):
            return "--"
        if isinstance(value, str):
            if value.upper() == "FAILED":
                return "FAILED"
            try:
                value = int(float(value))
            except ValueError:
                return value
        return "^ UP" if int(value) == 1 else "v DOWN"

    def to_result(row: pd.Series) -> str:
        if int(pd.to_numeric(row.get("failed", 0), errors="coerce") or 0) == 1:
            return "FAILED"
        result_value = row.get("result")
        if pd.isna(result_value) or str(result_value).strip() == "":
            return "--"
        return "OK" if int(float(result_value)) == 1 else "MISS"

    fig = plt.figure(figsize=(19, 10), facecolor="#f4efe6")
    grid = fig.add_gridspec(2, 2, width_ratios=[2.6, 1.2], height_ratios=[1.05, 1.55], hspace=0.36, wspace=0.16)
    ax_trend = fig.add_subplot(grid[0, 0])
    ax_chart = fig.add_subplot(grid[1, 0])
    right_grid = grid[:, 1].subgridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.22)
    ax_next = fig.add_subplot(right_grid[0, 0])
    ax_table = fig.add_subplot(right_grid[1, 0])
    for axis in [ax_trend, ax_chart, ax_next, ax_table]:
        axis.set_facecolor("#fbf8f2")

    ax_trend.set_title("Per-Model Accuracy Across Time", fontsize=15, weight="bold", pad=12)
    scored = history.sort_values("timestamp").copy()
    scored = scored[
        (pd.to_numeric(scored["failed"], errors="coerce").fillna(0) == 0)
        & (scored["status"] != "missing")
        & (pd.to_numeric(scored["actual"], errors="coerce").notna())
    ].copy()
    plotted = False
    for index, family in enumerate(model_families):
        correct = 0
        total = 0
        xs: list[pd.Timestamp] = []
        ys: list[float] = []
        for _, row in scored.iterrows():
            model_payload = parse_model_predictions(row.get("model_predictions")).get(family, {})
            predicted_label = model_payload.get("predicted_label")
            actual_label = pd.to_numeric(row.get("actual"), errors="coerce")
            if predicted_label is None or pd.isna(actual_label):
                continue
            total += 1
            correct += int(int(predicted_label) == int(actual_label))
            xs.append(pd.Timestamp(row["timestamp"]))
            ys.append((correct / total) * 100)
        if not xs:
            continue
        plotted = True
        color = color_cycle[index % len(color_cycle)]
        ax_trend.plot(xs, ys, linewidth=2.2, marker="o", markersize=3.8, color=color, label=model_headers[family])

    if not plotted:
        ax_trend.text(0.5, 0.5, "No validated per-model history yet", ha="center", va="center", fontsize=12, color="#6c757d", transform=ax_trend.transAxes)
        ax_trend.set_xticks([])
        ax_trend.set_yticks([])
        for spine in ax_trend.spines.values():
            spine.set_visible(False)
    else:
        ax_trend.axhline(50, color="#c9bba7", linestyle="--", linewidth=1.0)
        ax_trend.set_ylim(0, 110)
        ax_trend.set_ylabel("Accuracy %")
        ax_trend.tick_params(axis="x", rotation=25, labelsize=8)
        ax_trend.tick_params(axis="y", labelsize=9)
        ax_trend.grid(axis="y", alpha=0.18)
        for spine in ax_trend.spines.values():
            spine.set_color("#d8cbb8")
        champion_history = scored[scored["best_champion_name"].fillna("").astype(str).str.strip() != ""].copy()
        if not champion_history.empty:
            champion_history["champion_key"] = (
                champion_history["best_champion_family"].fillna("").astype(str)
                + ":"
                + champion_history["best_champion_name"].fillna("").astype(str)
            )
            changes = champion_history[champion_history["champion_key"] != champion_history["champion_key"].shift(1)]
            for idx, (_, row) in enumerate(changes.iterrows()):
                ts = pd.Timestamp(row["timestamp"])
                ax_trend.axvline(ts, color="#b08b57", linestyle=":", linewidth=1.0, alpha=0.55)
                ax_trend.text(ts, 106 - ((idx % 3) * 6), str(row["best_champion_name"]), rotation=90, va="top", ha="right", fontsize=7, color="#7a5c2e")
        ax_trend.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)

    ax_chart.axis("off")
    ax_chart.set_title("Last 10 Predictions and Best Champion", fontsize=15, weight="bold", pad=18)
    recent_history = history.sort_values("timestamp", ascending=False).head(10).copy()
    if recent_history.empty:
        recent_rows = [["--", "--", "--", "--", "--", "--"]]
        recent_headers = ["Open Time", "Open Price", "Close Price", "Champion", "Actual", "Result"]
    else:
        recent_rows = []
        recent_headers = ["Open Time", "Open Price", "Close Price", *[model_headers[family] for family in model_families], "Champion", "Actual", "Result"]
        for _, row in recent_history.iterrows():
            model_payloads = parse_model_predictions(row.get("model_predictions"))
            row_values = [
                format_dual_time(pd.Timestamp(row["timestamp"]) - pd.Timedelta(hours=1)),
                f"{float(row['reference_open']):,.2f}" if pd.notna(row["reference_open"]) else "--",
                f"{float(row['reference_close']):,.2f}" if pd.notna(row["reference_close"]) else "--",
            ]
            for family in model_families:
                row_values.append(to_arrow(model_payloads.get(family, {}).get("predicted_label", "")))
            row_values.extend([row.get("best_champion_name") or "--", to_arrow(row.get("actual")), to_result(row)])
            recent_rows.append(row_values)

    recent_col_widths = compute_table_col_widths(
        recent_headers,
        recent_rows,
        min_width=0.075,
        max_width=0.17,
    )
    recent_table = ax_chart.table(
        cellText=recent_rows,
        colLabels=recent_headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=recent_col_widths,
    )
    recent_table.scale(1, 2.0)
    recent_table.auto_set_font_size(False)
    recent_table.set_fontsize(8.6 if len(recent_headers) > 8 else 9.6)
    model_col_start = 3
    model_col_end = model_col_start + len(model_families) - 1
    champion_col_idx = len(recent_headers) - 3
    actual_col_idx = len(recent_headers) - 2
    result_col_idx = len(recent_headers) - 1
    for (row_idx, col_idx), cell in recent_table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
            continue
        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        cell.set_text_props(ha="center", va="center")
        text_value = str(cell.get_text().get_text())
        if model_col_start <= col_idx <= model_col_end or col_idx == actual_col_idx:
            if "UP" in text_value:
                cell.set_text_props(color="#1b7f4a", weight="bold", ha="center", va="center")
            elif "DOWN" in text_value:
                cell.set_text_props(color="#c44536", weight="bold", ha="center", va="center")
            elif "FAILED" in text_value:
                cell.set_text_props(color="#6c757d", weight="bold", ha="center", va="center")
        if col_idx == champion_col_idx:
            cell.set_text_props(color="#7a5c2e", weight="bold", ha="center", va="center")
        if col_idx == result_col_idx:
            if text_value == "OK":
                cell.set_facecolor("#d9f2e3")
                cell.set_text_props(color="#1b7f4a", weight="bold", ha="center", va="center")
            elif text_value == "MISS":
                cell.set_facecolor("#f8d7da")
                cell.set_text_props(color="#a12d2f", weight="bold", ha="center", va="center")
            elif text_value == "FAILED":
                cell.set_facecolor("#e9ecef")
                cell.set_text_props(color="#495057", weight="bold", ha="center", va="center")

    ax_next.axis("off")
    ax_next.set_title("Next Prediction by Model (* = best overall)", fontsize=14, weight="bold", pad=6, color="#000000")
    next_payloads = parse_model_predictions(None if prediction_record is None else prediction_record.get("model_predictions"))
    current_best_family = None if prediction_record is None else prediction_record.get("best_champion_family")
    next_headers = ["Field", *[f"{model_headers[family]}{' *' if family == current_best_family else ''}" for family in model_families]]
    if not model_families:
        next_headers = ["Field", "Value"]
    if not prediction_record:
        next_rows = [["Open Time", "--"], ["Target Time", "--"], ["Signal", "--"], ["Probability", "--"], ["Accuracy", "--"], ["F1", "--"], ["Best Overall", "--"]]
    else:
        open_time = format_dual_time(prediction_record.get("reference_candle_timestamp"))
        target_time = format_dual_time(prediction_record.get("target_candle_timestamp"))
        next_rows = [["Open Time"], ["Target Time"], ["Signal"], ["Probability"], ["Accuracy"], ["F1"], ["Best Overall"]]
        for family in model_families:
            payload = next_payloads.get(family, {})
            next_rows[0].append("")
            next_rows[1].append("")
            next_rows[2].append(payload.get("predicted_signal", "--"))
            next_rows[3].append(f"{float(payload.get('probability_up', 0.0)):.1%}" if payload else "--")
            next_rows[4].append(f"{float(payload.get('accuracy', 0.0)):.3f}" if payload else "--")
            next_rows[5].append(f"{float(payload.get('f1', 0.0)):.3f}" if payload else "--")
            next_rows[6].append("BEST" if family == current_best_family else "")
        if not model_families:
            next_rows = [["Open Time", open_time], ["Target Time", target_time], ["Signal", prediction_record.get("predicted_signal", "--")], ["Probability", f"{float(prediction_record.get('probability_up', 0.0)):.1%}"], ["Accuracy", f"{float(prediction_record.get('model_accuracy', 0.0)):.3f}"], ["F1", f"{float(prediction_record.get('model_f1', 0.0)):.3f}"], ["Best Overall", prediction_record.get("best_champion_name", "--")]]

    next_col_widths = compute_table_col_widths(
        next_headers,
        next_rows,
        min_width=0.11 if len(next_headers) > 4 else 0.16,
        max_width=0.26,
        header_weight=1.3,
    )
    next_table = ax_next.table(
        cellText=next_rows,
        colLabels=next_headers,
        bbox=[-0.02, 0.0, 1.06, 0.9],
        cellLoc="center",
        colLoc="center",
        colWidths=next_col_widths,
    )
    next_table.scale(1, 1.95)
    next_table.auto_set_font_size(False)
    next_table.set_fontsize(7.9 if len(next_headers) > 4 else 9.1)
    for (row_idx, col_idx), cell in next_table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
            continue
        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        cell.set_text_props(ha="center", va="center")
        text_value = str(cell.get_text().get_text())
        if row_idx in {1, 2} and len(next_headers) > 2:
            if col_idx == 1:
                cell.visible_edges = "LTB"
                cell.set_text_props(color="#1f3c4d", weight="bold", ha="center", va="center")
            elif col_idx == len(next_headers) - 1:
                cell.visible_edges = "RTB"
                cell.get_text().set_text("")
            elif col_idx > 1:
                cell.visible_edges = "TB"
                cell.get_text().set_text("")
        if row_idx == 3 and col_idx > 0:
            if "UP" in text_value:
                cell.set_facecolor("#e9f5ec")
                cell.set_text_props(color="#1b7f4a", weight="bold", ha="center", va="center")
            elif "DOWN" in text_value:
                cell.set_facecolor("#f8d7da")
                cell.set_text_props(color="#c44536", weight="bold", ha="center", va="center")
        if row_idx == 7 and text_value == "BEST":
            cell.set_facecolor("#f6eee1")
            cell.set_text_props(color="#7a5c2e", weight="bold", ha="center", va="center")

    if prediction_record and len(next_headers) > 2:
        fig.canvas.draw()
        merged_start_col = 1
        merged_end_col = len(next_headers) - 1
        merged_time_rows = [
            (1, open_time),
            (2, target_time),
        ]
        for table_row_idx, label in merged_time_rows:
            start_cell = next_table[(table_row_idx, merged_start_col)]
            end_cell = next_table[(table_row_idx, merged_end_col)]
            x0, y0 = start_cell.get_xy()
            x1 = end_cell.get_xy()[0] + end_cell.get_width()
            y_center = y0 + (start_cell.get_height() / 2.0)
            x_center = (x0 + x1) / 2.0
            ax_next.text(
                x_center,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=8.8 if len(next_headers) > 4 else 9.6,
                color="#1f3c4d",
                weight="bold",
                transform=ax_next.transAxes,
                zorder=10,
            )

    ax_table.axis("off")
    ax_table.set_title("Champion Summary", fontsize=14, weight="bold", pad=6, color="#000000")
    summary_rows = [
        ["Current Best Champion", prediction_record.get("best_champion_name", "--") if prediction_record else "--"],
        ["Best Champion Family", prediction_record.get("best_champion_family", "--") if prediction_record else "--"],
        ["Best Champion Version", prediction_record.get("best_champion_version", "--") if prediction_record else "--"],
        ["Workflow", prediction_record.get("workflow_variant", "--") if prediction_record else "--"],
        ["Daily Refresh Run", str(bool(prediction_record.get("daily_model_refresh"))) if prediction_record and prediction_record.get("daily_model_refresh") is not None else "--"],
        ["Model Day (ET)", prediction_record.get("model_refresh_et_date", "--") if prediction_record else "--"],
        ["Total Predictions", stats["total_predictions"]],
        ["Total Correct", stats["total_correct"]],
        ["Total Accuracy %", f"{stats['total_accuracy_pct']:.1f}%"],
        ["Last 24h Predictions", stats["last_24h_predictions"]],
        ["Last 24h Accuracy %", f"{stats['last_24h_accuracy_pct']:.1f}%"],
    ]
    table = ax_table.table(cellText=summary_rows, colLabels=["Metric", "Value"], bbox=[0, 0.02, 1, 0.9], cellLoc="center", colLoc="center")
    table.scale(1, 1.95)
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold", ha="center", va="center")
            continue
        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        cell.set_text_props(ha="center", va="center")
        if row_idx in {1, 2, 3} and col_idx == 1:
            cell.set_text_props(color="#7a5c2e", weight="bold", ha="center", va="center")
        if col_idx == 1 and "%" in str(cell.get_text().get_text()):
            pct = float(str(cell.get_text().get_text()).replace("%", ""))
            cell.set_text_props(color="#1b7f4a" if pct >= 50 else "#c44536", weight="bold", ha="center", va="center")

    output_path = DASHBOARD_PATH if dashboard_path is None else dashboard_path
    resolved_title = DASHBOARD_TITLE if dashboard_title is None else dashboard_title
    resolved_subtitle = DASHBOARD_SUBTITLE if dashboard_subtitle is None else dashboard_subtitle

    fig.suptitle(resolved_title, fontsize=18, weight="bold", color="#1f3c4d", y=0.98)
    current_time_y = 0.945
    if resolved_subtitle:
        fig.text(0.5, 0.952, resolved_subtitle, ha="center", va="center", fontsize=10.5, color="#7a5c2e")
        current_time_y = 0.93
    fig.text(0.5, current_time_y, format_current_dual_time(), ha="center", va="center", fontsize=10.5, color="#5b5f66")
    fig.text(0.015, 0.02, "Green = UP / correct strength   Red = DOWN / misses   Gray = failed run", fontsize=10, color="#5b5f66")
    fig.subplots_adjust(left=0.04, right=0.985, top=0.88, bottom=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_dashboard_variants(
    history: pd.DataFrame,
    prediction_record: dict[str, Any] | None,
) -> None:
    for variant in DASHBOARD_VARIANTS:
        variant_history, stats, variant_prediction = build_dashboard_variant_view(
            history,
            prediction_record,
            variant,
        )
        render_dashboard(
            variant_history,
            stats,
            variant_prediction,
            dashboard_path=variant["path"],
            dashboard_title=variant.get("title"),
            dashboard_subtitle=variant.get("subtitle"),
        )


def main() -> None:
    try:
        configure_tracking()
    except Exception as exc:
        print(f"Could not configure tracking. Continuing with dashboard refresh only: {exc}")
    prediction_record = load_last_prediction()
    history = ensure_recent_history_slots(load_history())

    candles: pd.DataFrame | None = None
    try:
        candles = fetch_validation_candles(history, prediction_record)
    except Exception as exc:
        print(f"Could not fetch validation candles. Continuing with stored history only: {exc}")

    history = remove_incomplete_validations(
        normalize_history_labels(
            backfill_recent_history_prices(
                history,
                candles,
            )
        )
    )
    if prediction_record is None:
        render_dashboard_variants(history, prediction_record)
        print(
            "No local last_prediction.json found. "
            "Skipping validation and refreshing the dashboard with existing history only."
        )
        return

    if prediction_already_recorded(history, prediction_record):
        render_dashboard_variants(history, prediction_record)
        print(
            f"Prediction for {get_prediction_timestamp(prediction_record).isoformat()} "
            "is already recorded. Skipping duplicate validation."
        )
        return

    prediction_status = prediction_record.get("status", "success")
    if prediction_status == "failed":
        failure_timestamp = get_prediction_timestamp(prediction_record)
        history = upsert_history_row(
            history,
            {
                "timestamp": failure_timestamp,
                "predicted": "FAILED",
                "actual": "",
                "result": 0,
                "failed": 1,
                "status": "failed",
                "reference_open": prediction_record.get("reference_open"),
                "reference_close": prediction_record.get("reference_close"),
                "target_open": pd.NA,
                "target_close": pd.NA,
                "model_predictions": json.dumps(
                    parse_model_predictions(prediction_record.get("model_predictions")),
                    sort_keys=True,
                ),
                "best_champion_name": prediction_record.get("best_champion_name", ""),
                "best_champion_family": prediction_record.get("best_champion_family", ""),
                "best_champion_version": prediction_record.get("best_champion_version", ""),
                "workflow_name": prediction_record.get("workflow_name", ""),
                "workflow_variant": prediction_record.get("workflow_variant", ""),
                "daily_model_refresh": prediction_record.get("daily_model_refresh", pd.NA),
                "model_refresh_et_date": prediction_record.get("model_refresh_et_date", ""),
                "prediction_generated_at": prediction_record.get("prediction_generated_at", ""),
            },
        )
        render_dashboard_variants(history, prediction_record)
        print(
            "Prediction run previously failed:",
            json.dumps(
                {
                    "timestamp": failure_timestamp.isoformat(),
                    "error": prediction_record.get("error"),
                },
                indent=2,
            ),
        )
        return

    if candles is None:
        render_dashboard_variants(history, prediction_record)
        print("Could not fetch validation candles. Skipping validation for now.")
        return
    actual = resolve_actual_direction(candles, prediction_record)
    if actual is None:
        render_dashboard_variants(history, prediction_record)
        print("Target candle is not available yet. Dashboard refreshed without new validation row.")
        return

    actual_label, target_timestamp, reference_open, reference_close, target_open = actual
    predicted_label = int(prediction_record["predicted_label"])
    result = int(predicted_label == actual_label)

    history = upsert_history_row(
        history,
        {
            "timestamp": target_timestamp,
            "predicted": predicted_label,
            "actual": actual_label,
            "result": result,
            "failed": 0,
            "status": "validated",
            "reference_open": reference_open,
            "reference_close": reference_close,
            "target_open": target_open,
            "target_close": pd.NA,
            "model_predictions": json.dumps(
                parse_model_predictions(prediction_record.get("model_predictions")),
                sort_keys=True,
            ),
            "best_champion_name": prediction_record.get("best_champion_name", ""),
            "best_champion_family": prediction_record.get("best_champion_family", ""),
            "best_champion_version": prediction_record.get("best_champion_version", ""),
            "workflow_name": prediction_record.get("workflow_name", ""),
            "workflow_variant": prediction_record.get("workflow_variant", ""),
            "daily_model_refresh": prediction_record.get("daily_model_refresh", pd.NA),
            "model_refresh_et_date": prediction_record.get("model_refresh_et_date", ""),
            "prediction_generated_at": prediction_record.get("prediction_generated_at", ""),
        },
    )
    render_dashboard_variants(history, prediction_record)

    print(
        "Validation complete:",
        json.dumps(
            {
                "champion_version": prediction_record.get("best_champion_version"),
                "model_name": prediction_record.get("model_name"),
                "predicted": predicted_label,
                "actual": actual_label,
                "result": result,
                "reference_open": reference_open,
                "reference_close": reference_close,
                "target_open": target_open,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()

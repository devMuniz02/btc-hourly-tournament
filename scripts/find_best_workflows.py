#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


ROOT = Path(__file__).resolve().parent.parent
try:
    NEW_YORK_TZ = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:
    NEW_YORK_TZ = None

LIVE_PRICE_TRADING_FEE_RATE = 0.072
LIVE_PRICE_MAKER_REBATE_RATE = 0.2

WORKFLOW_DEFS = [
    {"id": "hourly24", "label": "Hourly 24h", "file": "history.csv", "market_hours_native": False},
    {"id": "daily24", "label": "Daily Model + Hourly 24h", "file": "history_daily.csv", "market_hours_native": False},
    {"id": "marketHourly", "label": "Market Hours Hourly", "file": "history_market_hours.csv", "market_hours_native": True},
    {"id": "marketDaily", "label": "Market Hours Daily Model", "file": "history_market_hours_daily.csv", "market_hours_native": True},
]

VARIATIONS = [
    {"id": "original", "label": "Original"},
    {"id": "reverse", "label": "Reverse"},
    {"id": "market-hours", "label": "Market Hours"},
    {"id": "market-hours-reverse", "label": "Market Hours Reverse"},
]

DEFAULTS = {
    "starting_balance": 68.0,
    "position_sizing": "peak-percent",
    "percentage_amount": 5.0,
    "fixed_quantity": 1.0,
    "optimal_metric": "cumulative",
    "optimal_lookback": 5,
    "use_live_price": True,
    "live_price_minute": 10,
    "live_price_order_type": "market",
    "live_price_threshold": 1.0,
    "weekly_deposit": 40.0,
}


@dataclass
class WorkflowStatus:
    id: str
    label: str
    file: str
    market_hours_native: bool
    rows: list[dict[str, Any]]
    base_workflow_id: str | None = None
    model_key: str | None = None
    model_label: str | None = None


@dataclass
class SimulationResult:
    workflow_id: str
    workflow: str
    base_workflow_id: str
    market_hours_native: bool
    model_key: str | None
    model_label: str | None
    strategy: str
    strategy_mode: str
    variation_id: str | None
    variation_label: str | None
    sizing_mode: str
    win_rate: float
    pnl_pct: float


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    sizing_mode: str
    use_optimal: bool


STRATEGY_CONFIGS = [
    StrategyConfig(label="Conservative", sizing_mode="fixed", use_optimal=False),
    StrategyConfig(label="Percentage Based on Max", sizing_mode="peak-percent", use_optimal=False),
    StrategyConfig(label="Optimal - Percentage Based on Max", sizing_mode="peak-percent", use_optimal=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser.parse_args()


def normalize_trader_workflow_id(script_workflow_id: str) -> str:
    mapping = {
        "hourly24": "hourly24",
        "daily24": "daily24",
        "marketHourly": "market-hours-hourly",
        "marketDaily": "market-hours-daily",
    }
    return mapping.get(script_workflow_id, script_workflow_id)


def parse_timestamp(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso_text = text.replace(" ", "T")
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        return None


def to_number(value: Any) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_json_object(value: str | None) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> datetime:
    first_day = datetime(year, month, 1)
    delta = (weekday - first_day.weekday()) % 7
    day = 1 + delta + ((occurrence - 1) * 7)
    return datetime(year, month, day)


def to_eastern_time(date_obj: datetime) -> datetime:
    if NEW_YORK_TZ is not None:
        return date_obj.astimezone(NEW_YORK_TZ)

    utc_value = date_obj.astimezone(timezone.utc)
    year = utc_value.year
    dst_start_local = nth_weekday_of_month(year, 3, 6, 2).replace(hour=2)
    dst_end_local = nth_weekday_of_month(year, 11, 6, 1).replace(hour=2)
    dst_start_utc = (dst_start_local + timedelta(hours=5)).replace(tzinfo=timezone.utc)
    dst_end_utc = (dst_end_local + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    offset_hours = -4 if dst_start_utc <= utc_value < dst_end_utc else -5
    return utc_value + timedelta(hours=offset_hours)


def is_market_hours_timestamp(date_obj: datetime | None) -> bool:
    if date_obj is None:
        return False
    eastern = to_eastern_time(date_obj)
    return 8 <= eastern.hour <= 20


def derive_price(row: dict[str, Any]) -> float | None:
    return row.get("target_close") or row.get("target_open") or row.get("reference_close")


def normalize_row(raw: dict[str, str], index: int) -> dict[str, Any] | None:
    timestamp = parse_timestamp(raw.get("timestamp"))
    if timestamp is None:
        return None
    normalized = {
        "index": index,
        "timestamp": timestamp,
        "timestamp_text": str(raw.get("timestamp", "")).strip(),
        "predicted": to_number(raw.get("predicted")),
        "actual": to_number(raw.get("actual")),
        "result": to_number(raw.get("result")),
        "failed": to_number(raw.get("failed")) or 0.0,
        "status": str(raw.get("status", "")).strip().lower(),
        "reference_open": to_number(raw.get("reference_open")),
        "reference_close": to_number(raw.get("reference_close")),
        "target_open": to_number(raw.get("target_open")),
        "target_close": to_number(raw.get("target_close")),
        "model_predictions": parse_json_object(raw.get("model_predictions")),
        "best_champion_name": str(raw.get("best_champion_name", "")).strip(),
        "best_champion_family": str(raw.get("best_champion_family", "")).strip(),
        "raw": raw,
    }
    normalized["price"] = derive_price(normalized)
    return normalized


def load_workflow(defn: dict[str, Any]) -> WorkflowStatus:
    path = ROOT / defn["file"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, raw in enumerate(reader):
            normalized = normalize_row(raw, index)
            if normalized is not None:
                rows.append(normalized)
    return WorkflowStatus(
        id=defn["id"],
        label=defn["label"],
        file=defn["file"],
        market_hours_native=bool(defn["market_hours_native"]),
        rows=rows,
    )


def get_model_display_label(model_key: str, model_info: dict[str, Any]) -> str:
    preferred = str(model_info.get("name", "")).strip()
    if preferred:
        return preferred
    labels = {
        "lstm": "LSTM",
        "nn": "NN",
        "rf": "RF",
        "xgb": "XGBoost",
        "transformer": "Transformer",
        "mlp_sklearn": "MLP",
    }
    return labels.get(model_key, str(model_key or "Model").upper())


def create_expanded_rows_for_model(rows: list[dict[str, Any]], model_key: str) -> list[dict[str, Any]]:
    expanded_rows = []
    for row in rows:
        model_info = row.get("model_predictions", {}).get(model_key) or {}
        model_predicted = to_number(model_info.get("predicted_label"))
        expanded_row = dict(row)
        expanded_row["predicted"] = model_predicted if model_predicted is not None else None
        expanded_row["expanded_model_key"] = model_key
        expanded_row["expanded_model_label"] = get_model_display_label(model_key, model_info)
        expanded_row["expanded_model_info"] = model_info
        expanded_rows.append(expanded_row)
    return expanded_rows


def build_expanded_workflows(workflows: list[WorkflowStatus]) -> list[WorkflowStatus]:
    expanded: list[WorkflowStatus] = []
    for workflow in workflows:
        model_map: dict[str, str] = {}
        for row in workflow.rows:
            for model_key, model_info in (row.get("model_predictions") or {}).items():
                if model_key and model_key not in model_map:
                    model_map[model_key] = get_model_display_label(model_key, model_info or {})
        for model_key, model_label in model_map.items():
            expanded.append(
                WorkflowStatus(
                    id=f"{workflow.id}__{model_key}",
                    label=f"{workflow.label} ({model_label})",
                    file=workflow.file,
                    market_hours_native=workflow.market_hours_native,
                    rows=create_expanded_rows_for_model(workflow.rows, model_key),
                    base_workflow_id=workflow.id,
                    model_key=model_key,
                    model_label=model_label,
                )
            )
    return expanded


def load_live_price_index() -> dict[tuple[int, int], dict[str, Any]]:
    path = ROOT / "btc_history.csv"
    if not path.exists():
        return {}
    index: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            market_end = parse_timestamp(raw.get("market_end_iso"))
            try:
                minute_value = int(str(raw.get("minutes_from_market_start") or "").strip())
            except ValueError:
                continue
            if market_end is None:
                continue
            key = (int(market_end.timestamp() * 1000), minute_value)
            index[key] = {
                "up_price": to_number(raw.get("up_price")),
                "down_price": to_number(raw.get("down_price")),
            }
    return index


def get_variation_label(variation_id: str) -> str:
    for variation in VARIATIONS:
        if variation["id"] == variation_id:
            return variation["label"]
    return variation_id


def get_variation_descriptor(workflow: WorkflowStatus, variation_id: str) -> dict[str, Any]:
    market_hours_mode = variation_id in {"market-hours", "market-hours-reverse"}
    reverse_mode = variation_id in {"reverse", "market-hours-reverse"}
    effective_market_hours = True if workflow.market_hours_native else market_hours_mode
    return {
        "effective_market_hours": effective_market_hours,
        "effective_reverse": reverse_mode,
    }


def build_variation_trade_row(workflow: WorkflowStatus, variation_id: str, row: dict[str, Any], visible_index: int) -> dict[str, Any]:
    descriptor = get_variation_descriptor(workflow, variation_id)
    if descriptor["effective_market_hours"] and not is_market_hours_timestamp(row["timestamp"]):
        skipped = dict(row)
        skipped.update(
            {
                "view_index": visible_index,
                "variation_id": variation_id,
                "variation_label": get_variation_label(variation_id),
                "transformed_predicted": None,
                "transformed_correct": None,
                "can_trade": False,
            }
        )
        return skipped

    predicted = row.get("predicted")
    if predicted is not None and descriptor["effective_reverse"]:
        predicted = 1 - predicted
    actual = row.get("actual")
    scored = row.get("status") != "missing" and row.get("failed") != 1 and predicted is not None and actual is not None
    correct = predicted == actual if scored else None

    trade_row = dict(row)
    trade_row.update(
        {
            "view_index": visible_index,
            "variation_id": variation_id,
            "variation_label": get_variation_label(variation_id),
            "transformed_predicted": predicted,
            "transformed_correct": correct,
            "can_trade": bool(scored),
        }
    )
    return trade_row


def build_variation_rows(workflow: WorkflowStatus, variation_id: str) -> list[dict[str, Any]]:
    descriptor = get_variation_descriptor(workflow, variation_id)
    rows = workflow.rows
    if descriptor["effective_market_hours"]:
        rows = [row for row in rows if is_market_hours_timestamp(row["timestamp"])]
    return [
        build_variation_trade_row(workflow, variation_id, row, visible_index)
        for visible_index, row in enumerate(rows)
    ]


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def weighted_average(values: list[float]) -> float | None:
    if not values:
        return None
    weighted_sum = 0.0
    weight_total = 0
    for index, value in enumerate(values):
        weight = index + 1
        weighted_sum += value * weight
        weight_total += weight
    return None if weight_total == 0 else weighted_sum / weight_total


def standard_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    mean = average(values)
    if mean is None:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def clamp_score(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0.0, min(100.0, value))


def create_optimal_metric_state() -> dict[str, Any]:
    return {
        "history": [],
        "cumulative_sum": 0.0,
        "cumulative_count": 0,
        "ema": None,
        "rma": None,
        "ewma": None,
        "fast_ema": None,
        "slow_ema": None,
        "kalman_estimate": None,
        "kalman_error": 1.0,
    }


def get_optimal_score_before(state: dict[str, Any], metric: str, period: int) -> float | None:
    history = state["history"]
    recent = history[-period:]
    previous_window = history[-(period * 2):-period] if period > 0 else []

    if metric == "ema":
        return state["ema"]
    if metric == "sma":
        return average(recent)
    if metric == "wma":
        return weighted_average(recent)
    if metric == "rma":
        return state["rma"]
    if metric == "ewma":
        return state["ewma"]
    if metric == "rolling":
        return average(recent)
    if metric == "cumulative":
        return None if state["cumulative_count"] == 0 else state["cumulative_sum"] / state["cumulative_count"]
    if metric == "momentum":
        recent_average = average(recent)
        previous_average = average(previous_window)
        if recent_average is None:
            return None
        if previous_average is None:
            return clamp_score(recent_average)
        return clamp_score(50 + ((recent_average - previous_average) / 2))
    if metric == "macd":
        if state["fast_ema"] is None or state["slow_ema"] is None:
            return None
        return clamp_score(50 + ((state["fast_ema"] - state["slow_ema"]) / 2))
    if metric == "bollinger":
        mean = average(recent)
        deviation = standard_deviation(recent)
        if mean is None:
            return None
        return clamp_score(mean + (deviation or 0))
    if metric == "kalman":
        return state["kalman_estimate"]
    return state["ema"]


def update_optimal_metric_state(state: dict[str, Any], outcome: float, period: int) -> None:
    ema_alpha = 2 / (period + 1)
    rma_alpha = 1 / period
    ewma_alpha = min(0.99, 2 / (math.sqrt(period) + 1))
    fast_period = max(2, math.floor(period / 2))
    fast_alpha = 2 / (fast_period + 1)
    slow_alpha = ema_alpha
    process_noise = 0.01
    measurement_noise = max(0.05, 1 / period)

    state["history"].append(outcome)
    state["cumulative_sum"] += outcome
    state["cumulative_count"] += 1

    state["ema"] = outcome if state["ema"] is None else (outcome * ema_alpha) + (state["ema"] * (1 - ema_alpha))
    state["rma"] = outcome if state["rma"] is None else (outcome * rma_alpha) + (state["rma"] * (1 - rma_alpha))
    state["ewma"] = outcome if state["ewma"] is None else (outcome * ewma_alpha) + (state["ewma"] * (1 - ewma_alpha))
    state["fast_ema"] = outcome if state["fast_ema"] is None else (outcome * fast_alpha) + (state["fast_ema"] * (1 - fast_alpha))
    state["slow_ema"] = outcome if state["slow_ema"] is None else (outcome * slow_alpha) + (state["slow_ema"] * (1 - slow_alpha))

    if state["kalman_estimate"] is None:
        state["kalman_estimate"] = outcome
        state["kalman_error"] = 1.0
        return

    state["kalman_error"] += process_noise
    kalman_gain = state["kalman_error"] / (state["kalman_error"] + measurement_noise)
    state["kalman_estimate"] = state["kalman_estimate"] + (kalman_gain * (outcome - state["kalman_estimate"]))
    state["kalman_error"] = (1 - kalman_gain) * state["kalman_error"]


def build_optimal_rows(workflow: WorkflowStatus) -> list[dict[str, Any]]:
    optimal_period = int(DEFAULTS["optimal_lookback"])
    optimal_metric = str(DEFAULTS["optimal_metric"])
    variation_scores = {variation["id"]: create_optimal_metric_state() for variation in VARIATIONS}
    output_rows: list[dict[str, Any]] = []

    for visible_index, row in enumerate(workflow.rows):
        candidates = []
        for variation in VARIATIONS:
            trade_row = build_variation_trade_row(workflow, variation["id"], row, visible_index)
            score_state = variation_scores[variation["id"]]
            candidate = dict(trade_row)
            candidate["metric_before"] = get_optimal_score_before(score_state, optimal_metric, optimal_period)
            candidates.append(candidate)

        tradeable_candidates = [candidate for candidate in candidates if candidate["can_trade"]]
        chosen_candidate = None
        for candidate in tradeable_candidates:
            if chosen_candidate is None:
                chosen_candidate = candidate
                continue
            best_score = chosen_candidate["metric_before"] if chosen_candidate["metric_before"] is not None else -1
            candidate_score = candidate["metric_before"] if candidate["metric_before"] is not None else -1
            if candidate_score > best_score:
                chosen_candidate = candidate

        for candidate in candidates:
            if not candidate["can_trade"]:
                continue
            score_state = variation_scores[candidate["variation_id"]]
            outcome = 100.0 if candidate["transformed_correct"] else 0.0
            update_optimal_metric_state(score_state, outcome, optimal_period)

        if chosen_candidate is None:
            no_trade = dict(row)
            no_trade.update(
                {
                    "view_index": visible_index,
                    "variation_id": "optimal",
                    "variation_label": "Optimal",
                    "chosen_variation_id": None,
                    "chosen_variation_label": None,
                    "chosen_variation_metric": None,
                    "transformed_predicted": None,
                    "transformed_correct": None,
                    "can_trade": False,
                }
            )
            output_rows.append(no_trade)
            continue

        chosen = dict(chosen_candidate)
        chosen["variation_id"] = "optimal"
        chosen["variation_label"] = "Optimal"
        chosen["chosen_variation_id"] = chosen_candidate["variation_id"]
        chosen["chosen_variation_label"] = chosen_candidate["variation_label"]
        chosen["chosen_variation_metric"] = chosen_candidate["metric_before"]
        output_rows.append(chosen)

    return output_rows


def get_live_price_for_row(row: dict[str, Any], live_price_index: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    minute_key = int(DEFAULTS["live_price_minute"])
    row_timestamp = row.get("timestamp")
    lookup_key = (int(row_timestamp.timestamp() * 1000), minute_key) if row_timestamp is not None else None
    price_row = live_price_index.get(lookup_key) if lookup_key is not None else None
    predicted_side = "up" if row.get("transformed_predicted") == 1 else "down" if row.get("transformed_predicted") == 0 else None
    price = 0.5
    if predicted_side == "up" and price_row and price_row.get("up_price") is not None:
        price = price_row["up_price"]
    elif predicted_side == "down" and price_row and price_row.get("down_price") is not None:
        price = price_row["down_price"]
    if not math.isfinite(price) or price <= 0 or price > 1:
        price = 0.5
    return {
        "live_price": price,
        "price_found": bool(price_row),
    }


def simulate_rows(
    rows: list[dict[str, Any]],
    live_price_index: dict[tuple[int, int], dict[str, Any]],
    *,
    sizing_mode: str | None = None,
) -> dict[str, Any]:
    selected_sizing_mode = sizing_mode or str(DEFAULTS["position_sizing"])
    live_price_mode = bool(DEFAULTS["use_live_price"])
    live_price_threshold = float(DEFAULTS["live_price_threshold"])
    live_price_order_type = str(DEFAULTS["live_price_order_type"])
    percent_risk = float(DEFAULTS["percentage_amount"]) / 100.0
    fixed_quantity = float(DEFAULTS["fixed_quantity"])
    starting_balance = max(0.01, float(DEFAULTS["starting_balance"]) or 10.0)
    weekly_deposit = max(0.0, float(DEFAULTS["weekly_deposit"]))
    balance = starting_balance
    peak_balance = starting_balance
    total_invested = 0.0
    total_deposited = 0.0
    wins = 0
    losses = 0
    first_timestamp = next((row["timestamp"] for row in rows if row.get("timestamp") is not None), None)
    deposit_anchor_time = first_timestamp.timestamp() * 1000 if first_timestamp is not None else None
    next_deposit_time = deposit_anchor_time + (7 * 24 * 60 * 60 * 1000) if deposit_anchor_time is not None else None
    simulated_rows = []

    if weekly_deposit > 0 and first_timestamp is not None and next_deposit_time is not None:
        first_ms = first_timestamp.timestamp() * 1000
        while first_ms >= next_deposit_time:
            balance += weekly_deposit
            total_deposited += weekly_deposit
            peak_balance = max(peak_balance, balance)
            next_deposit_time += 7 * 24 * 60 * 60 * 1000

    for row in rows:
        trade_amount = 0.0
        change_dollar = 0.0
        change_percent = 0.0
        applied_live_price = 0.5
        live_price_found = False
        payout_percent = 100.0
        fee_dollar = 0.0
        maker_rebate_dollar = 0.0
        deposited_this_row = 0.0
        skipped_by_live_price_threshold = False

        if weekly_deposit > 0 and next_deposit_time is not None:
            row_ms = row["timestamp"].timestamp() * 1000
            while row_ms >= next_deposit_time:
                balance += weekly_deposit
                total_deposited += weekly_deposit
                deposited_this_row += weekly_deposit
                peak_balance = max(peak_balance, balance)
                next_deposit_time += 7 * 24 * 60 * 60 * 1000

        if row.get("can_trade"):
            if selected_sizing_mode == "fixed":
                trade_amount = fixed_quantity
            elif selected_sizing_mode == "peak-percent":
                trade_amount = peak_balance * percent_risk
            else:
                trade_amount = balance * percent_risk

            if live_price_mode:
                live_price_details = get_live_price_for_row(row, live_price_index)
                applied_live_price = live_price_details["live_price"]
                live_price_found = live_price_details["price_found"]
                if applied_live_price > live_price_threshold:
                    trade_amount = 0.0
                    change_dollar = 0.0
                    change_percent = 0.0
                    skipped_by_live_price_threshold = True
                else:
                    shares_bought = trade_amount / max(applied_live_price, 0.000001)
                    matched_fee_dollar = (
                        shares_bought
                        * LIVE_PRICE_TRADING_FEE_RATE
                        * applied_live_price
                        * (1 - applied_live_price)
                    )
                    if live_price_order_type == "limit":
                        maker_rebate_dollar = matched_fee_dollar * LIVE_PRICE_MAKER_REBATE_RATE
                        payout_percent = ((shares_bought - trade_amount + maker_rebate_dollar) / max(trade_amount, 0.000001)) * 100
                        change_dollar = (shares_bought - trade_amount + maker_rebate_dollar) if row["transformed_correct"] else (-trade_amount + maker_rebate_dollar)
                    else:
                        fee_dollar = matched_fee_dollar
                        payout_percent = ((shares_bought - trade_amount - fee_dollar) / max(trade_amount, 0.000001)) * 100
                        change_dollar = (shares_bought - trade_amount - fee_dollar) if row["transformed_correct"] else (-trade_amount - fee_dollar)
            else:
                change_dollar = trade_amount if row["transformed_correct"] else -trade_amount

            if not skipped_by_live_price_threshold:
                total_invested += trade_amount
                balance += change_dollar
                peak_balance = max(peak_balance, balance)
                balance_before_change = max(balance - change_dollar, 0.000001)
                change_percent = 0.0 if trade_amount == 0 else (change_dollar / balance_before_change) * 100
                if row["transformed_correct"]:
                    wins += 1
                else:
                    losses += 1

        simulated_row = dict(row)
        simulated_row.update(
            {
                "balance": balance,
                "trade_amount": trade_amount,
                "change_dollar": change_dollar,
                "change_percent": change_percent,
                "applied_live_price": applied_live_price,
                "live_price_found": live_price_found,
                "payout_percent": payout_percent,
                "fee_dollar": fee_dollar,
                "maker_rebate_dollar": maker_rebate_dollar,
                "live_price_order_type": live_price_order_type,
                "deposited_this_row": deposited_this_row,
                "skipped_by_live_price_threshold": skipped_by_live_price_threshold,
                "peak_balance": peak_balance,
                "sizing_mode": selected_sizing_mode,
            }
        )
        simulated_rows.append(simulated_row)

    trade_count = wins + losses
    contribution_base = starting_balance + total_deposited
    net_pnl = balance - contribution_base
    return {
        "final_balance": balance,
        "win_rate": 0.0 if trade_count == 0 else (wins / trade_count) * 100,
        "net_pnl_pct": 0.0 if contribution_base == 0 else (net_pnl / contribution_base) * 100,
        "rows": simulated_rows,
    }


def is_failed_balance(value: float | None) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return abs(value) < 0.005 or value < 0


def slice_rows_to_window(workflows: list[WorkflowStatus], hours: float) -> list[WorkflowStatus]:
    all_timestamps = [row["timestamp"] for workflow in workflows for row in workflow.rows]
    if not all_timestamps:
        return workflows
    max_timestamp = max(all_timestamps)
    cutoff = max_timestamp.timestamp() - (hours * 60 * 60)
    sliced = []
    for workflow in workflows:
        sliced_rows = [row for row in workflow.rows if row["timestamp"].timestamp() >= cutoff]
        sliced.append(
            WorkflowStatus(
                id=workflow.id,
                label=workflow.label,
                file=workflow.file,
                market_hours_native=workflow.market_hours_native,
                rows=sliced_rows,
                base_workflow_id=workflow.base_workflow_id,
                model_key=workflow.model_key,
                model_label=workflow.model_label,
            )
        )
    return sliced


def evaluate_workflow(workflow: WorkflowStatus, live_price_index: dict[tuple[int, int], dict[str, Any]]) -> SimulationResult | None:
    candidates: list[SimulationResult] = []

    for strategy_config in STRATEGY_CONFIGS:
        if strategy_config.use_optimal:
            optimal_rows = build_optimal_rows(workflow)
            optimal_simulation = simulate_rows(
                optimal_rows,
                live_price_index,
                sizing_mode=strategy_config.sizing_mode,
            )
            if not any(is_failed_balance(row.get("balance")) for row in optimal_simulation["rows"]):
                candidates.append(
                    SimulationResult(
                        workflow_id=normalize_trader_workflow_id(
                            workflow.base_workflow_id or workflow.id
                        ),
                        workflow=workflow.label,
                        base_workflow_id=workflow.base_workflow_id or workflow.id,
                        market_hours_native=workflow.market_hours_native,
                        model_key=workflow.model_key,
                        model_label=workflow.model_label,
                        strategy=strategy_config.label,
                        strategy_mode="optimal",
                        variation_id=None,
                        variation_label=None,
                        sizing_mode=strategy_config.sizing_mode,
                        win_rate=optimal_simulation["win_rate"],
                        pnl_pct=optimal_simulation["net_pnl_pct"],
                    )
                )
            continue

        for variation in VARIATIONS:
            strategy_rows = build_variation_rows(workflow, variation["id"])
            simulation = simulate_rows(
                strategy_rows,
                live_price_index,
                sizing_mode=strategy_config.sizing_mode,
            )
            if not any(is_failed_balance(row.get("balance")) for row in simulation["rows"]):
                candidates.append(
                    SimulationResult(
                        workflow_id=normalize_trader_workflow_id(
                            workflow.base_workflow_id or workflow.id
                        ),
                        workflow=workflow.label,
                        base_workflow_id=workflow.base_workflow_id or workflow.id,
                        market_hours_native=workflow.market_hours_native,
                        model_key=workflow.model_key,
                        model_label=workflow.model_label,
                        strategy=f"{strategy_config.label} - {variation['label']}",
                        strategy_mode=(
                            "conservative"
                            if strategy_config.label == "Conservative"
                            else "percentage"
                        ),
                        variation_id=variation["id"],
                        variation_label=variation["label"],
                        sizing_mode=strategy_config.sizing_mode,
                        win_rate=simulation["win_rate"],
                        pnl_pct=simulation["net_pnl_pct"],
                    )
                )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item.pnl_pct, -item.win_rate, item.strategy))
    return candidates[0]


def render_table(rows: list[SimulationResult]) -> str:
    headers = ["workflow", "strategy", "winrate", "pnl%"]
    body = [[row.workflow, row.strategy, f"{row.win_rate:.2f}%", f"{row.pnl_pct:.2f}%"] for row in rows]
    widths = [len(header) for header in headers]
    for item in body:
        for index, value in enumerate(item):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    lines = [format_row(headers)]
    if body:
        lines.append(format_row(["-" * width for width in widths]))
        lines.extend(format_row(item) for item in body)
    return "\n".join(lines)


def serialize_results(rows: list[SimulationResult]) -> list[dict[str, Any]]:
    return [
        {
            "workflow_id": row.workflow_id,
            "workflow": row.workflow,
            "base_workflow_id": row.base_workflow_id,
            "market_hours_native": row.market_hours_native,
            "model_key": row.model_key,
            "model_label": row.model_label,
            "strategy": row.strategy,
            "strategy_mode": row.strategy_mode,
            "variation_id": row.variation_id,
            "variation_label": row.variation_label,
            "sizing_mode": row.sizing_mode,
            "win_rate": row.win_rate,
            "pnl_pct": row.pnl_pct,
        }
        for row in rows
    ]


def main() -> None:
    args = parse_args()
    base_workflows = [load_workflow(defn) for defn in WORKFLOW_DEFS]
    all_workflows = slice_rows_to_window(base_workflows, args.hours)
    expanded_workflows = build_expanded_workflows(all_workflows)
    ranking_candidates = all_workflows + expanded_workflows
    live_price_index = load_live_price_index()

    results = []
    for workflow in ranking_candidates:
        best = evaluate_workflow(workflow, live_price_index)
        if best is not None:
            results.append(best)

    results.sort(key=lambda item: (-item.pnl_pct, -item.win_rate, item.workflow, item.strategy))
    if args.format == "json":
        print(json.dumps(serialize_results(results), indent=2))
        return
    print(render_table(results))


if __name__ == "__main__":
    main()

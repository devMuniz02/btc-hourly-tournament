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


HISTORY_PATH = Path("history.csv")
DASHBOARD_PATH = Path("assets/dashboard.png")
LOCAL_LAST_PREDICTION_PATH = Path("last_prediction.json")
EASTERN_TZ = ZoneInfo("America/New_York")


def configure_tracking() -> tuple[MlflowClient, str, str]:
    registered_model_name = tournament.configure_tracking()
    experiment_name = tournament.get_env_str("MLFLOW_EXPERIMENT") or tournament.DEFAULT_EXPERIMENT
    return MlflowClient(), registered_model_name, experiment_name


def get_current_champion_info(
    client: MlflowClient,
    registered_model_name: str,
) -> dict[str, Any] | None:
    try:
        version = client.get_model_version_by_alias(registered_model_name, "champion")
    except Exception:
        return None
    return {
        "registered_model_name": registered_model_name,
        "version": version.version,
        "run_id": version.run_id,
    }


def load_last_prediction() -> dict[str, Any] | None:
    if LOCAL_LAST_PREDICTION_PATH.exists():
        return json.loads(LOCAL_LAST_PREDICTION_PATH.read_text(encoding="utf-8"))
    return None


def fetch_recent_candles() -> pd.DataFrame:
    return tournament.fetch_ohlcv(limit=8, min_candles=8)


def resolve_actual_direction(
    candles: pd.DataFrame,
    prediction_record: dict[str, Any],
) -> tuple[int, pd.Timestamp, float, float] | None:
    reference_ts = pd.Timestamp(prediction_record["reference_candle_timestamp"])
    target_ts = pd.Timestamp(prediction_record["target_candle_timestamp"])

    candle_frame = candles.copy()
    candle_frame["timestamp"] = pd.to_datetime(candle_frame["timestamp"], utc=True)
    indexed = candle_frame.set_index("timestamp")
    if reference_ts not in indexed.index or target_ts not in indexed.index:
        return None

    reference_close = float(indexed.loc[reference_ts, "close"])
    target_close = float(indexed.loc[target_ts, "close"])
    actual_label = int(target_close > reference_close)
    return actual_label, target_ts, reference_close, target_close


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "predicted",
                "actual",
                "result",
                "failed",
                "status",
                "reference_open",
                "reference_close",
                "target_close",
            ]
        )
    history = pd.read_csv(HISTORY_PATH)
    if history.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "predicted",
                "actual",
                "result",
                "failed",
                "status",
                "reference_open",
                "reference_close",
                "target_close",
            ]
        )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    if "failed" not in history.columns:
        history["failed"] = 0
    if "status" not in history.columns:
        history["status"] = "validated"
    for column in ["reference_open", "reference_close", "target_close"]:
        if column not in history.columns:
            history[column] = pd.NA
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
    return bool((history["timestamp"] == prediction_timestamp).any())


def upsert_history_row(history: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    updated = history.copy()
    if not updated.empty:
        updated = updated[updated["timestamp"] != row["timestamp"]]
    updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
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
    recent = history[history["timestamp"] >= last_24h_cutoff]
    total_predictions = int(len(history))
    total_correct = int(history["result"].sum())
    total_failed = int(history["failed"].sum())
    last_24h_predictions = int(len(recent))
    last_24h_correct = int(recent["result"].sum())
    last_24h_failed = int(recent["failed"].sum())
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


def format_dual_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    eastern = timestamp.tz_convert(EASTERN_TZ)
    return (
        f"{timestamp.strftime('%m-%d %H:%M')} UTC\n"
        f"{eastern.strftime('%m-%d %I:%M %p')} ET"
    )


def render_dashboard(
    history: pd.DataFrame,
    stats: dict[str, int],
    prediction_record: dict[str, Any] | None,
) -> None:
    fig = plt.figure(figsize=(15, 8), facecolor="#f4efe6")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.0, 1],
        height_ratios=[1.0, 1.9],
        hspace=0.42,
        wspace=0.18,
    )
    ax_trend = fig.add_subplot(grid[0, 0])
    ax_chart = fig.add_subplot(grid[1, 0])
    right_grid = grid[:, 1].subgridspec(2, 1, height_ratios=[0.72, 1.28], hspace=0.18)
    ax_next = fig.add_subplot(right_grid[0, 0])
    ax_table = fig.add_subplot(right_grid[1, 0])
    fig.patch.set_facecolor("#f4efe6")
    ax_trend.set_facecolor("#fbf8f2")
    ax_chart.set_facecolor("#fbf8f2")
    ax_next.set_facecolor("#fbf8f2")
    ax_table.set_facecolor("#fbf8f2")

    ax_trend.set_title("Recent Accuracy Trend", fontsize=15, weight="bold", pad=12)
    if history.empty:
        ax_trend.text(
            0.5,
            0.5,
            "No validation history yet",
            ha="center",
            va="center",
            fontsize=12,
            color="#6c757d",
            transform=ax_trend.transAxes,
        )
        ax_trend.set_xticks([])
        ax_trend.set_yticks([])
        for spine in ax_trend.spines.values():
            spine.set_visible(False)
    else:
        trend_history = history.sort_values("timestamp").tail(20).copy()
        scored = trend_history[trend_history["failed"] == 0].copy()
        if scored.empty:
            ax_trend.text(
                0.5,
                0.5,
                "Only failed runs in recent history",
                ha="center",
                va="center",
                fontsize=12,
                color="#6c757d",
                transform=ax_trend.transAxes,
            )
            ax_trend.set_xticks([])
            ax_trend.set_yticks([])
            for spine in ax_trend.spines.values():
                spine.set_visible(False)
        else:
            scored["rolling_accuracy"] = scored["result"].rolling(
                window=min(5, len(scored)),
                min_periods=1,
            ).mean() * 100
            x_labels = [
                pd.Timestamp(ts).strftime("%m-%d %H:%M") for ts in scored["timestamp"]
            ]
            point_colors = [
                "#1b7f4a" if int(result) == 1 else "#c44536"
                for result in scored["result"]
            ]
            ax_trend.plot(
                x_labels,
                scored["rolling_accuracy"],
                color="#1f3c4d",
                linewidth=2.5,
                marker="o",
                markersize=0,
                zorder=2,
            )
            ax_trend.scatter(
                x_labels,
                scored["rolling_accuracy"],
                c=point_colors,
                s=55,
                edgecolors="#fbf8f2",
                linewidths=1.0,
                zorder=3,
            )
            ax_trend.axhline(50, color="#c9bba7", linestyle="--", linewidth=1.2)
            ax_trend.set_ylim(-10, 110)
            ax_trend.set_yticks([0, 20, 40, 60, 80, 100])
            ax_trend.set_ylabel("Accuracy %")
            ax_trend.tick_params(axis="x", rotation=25, labelsize=8, pad=4)
            ax_trend.tick_params(axis="y", labelsize=9)
            ax_trend.grid(axis="y", alpha=0.18)
            for spine in ax_trend.spines.values():
                spine.set_color("#d8cbb8")

    ax_chart.axis("off")
    ax_chart.set_title("10 Most Recent Predictions", fontsize=15, weight="bold", pad=14)

    if history.empty:
        recent_rows = [["--", "--", "--", "--", "--", "--"]]
    else:
        recent_history = history.sort_values("timestamp", ascending=False).head(10).copy()

        def to_arrow(value: Any) -> str:
            if value == "" or pd.isna(value):
                return "--"
            if isinstance(value, str):
                if value.upper() == "FAILED":
                    return "FAILED"
                try:
                    value = int(value)
                except ValueError:
                    return value
            return "▲ UP" if int(value) == 1 else "▼ DOWN"

        def to_result(row: pd.Series) -> str:
            if int(row["failed"]) == 1:
                return "FAILED"
            return "OK" if int(row["result"]) == 1 else "MISS"

        recent_rows = [
            [
                format_dual_time(row["timestamp"]),
                (
                    f"{float(row['reference_open']):,.2f}"
                    if pd.notna(row["reference_open"])
                    else "--"
                ),
                (
                    f"{float(row['reference_close']):,.2f}"
                    if pd.notna(row["reference_close"])
                    else "--"
                ),
                to_arrow(row["predicted"]),
                to_arrow(row["actual"]),
                to_result(row),
            ]
            for _, row in recent_history.iterrows()
        ]

    recent_table = ax_chart.table(
        cellText=recent_rows,
        colLabels=["Time", "Open", "Close", "Pred", "Actual", "Result"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    recent_table.scale(1, 2)
    recent_table.auto_set_font_size(False)
    recent_table.set_fontsize(10.5)

    for (row_idx, col_idx), cell in recent_table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        text_value = cell.get_text().get_text()

        if col_idx in {1, 2}:
            if "UP" in text_value:
                cell.set_text_props(color="#1b7f4a", weight="bold")
            elif "DOWN" in text_value:
                cell.set_text_props(color="#c44536", weight="bold")
            elif "FAILED" in text_value:
                cell.set_text_props(color="#6c757d", weight="bold")

        if col_idx == 3:
            if text_value == "OK":
                cell.set_facecolor("#d9f2e3")
                cell.set_text_props(color="#1b7f4a", weight="bold")
            elif text_value == "MISS":
                cell.set_facecolor("#f8d7da")
                cell.set_text_props(color="#a12d2f", weight="bold")
            elif text_value == "FAILED":
                cell.set_facecolor("#e9ecef")
                cell.set_text_props(color="#495057", weight="bold")

    ax_next.axis("off")
    ax_next.set_title("Next Prediction", fontsize=15, weight="bold", pad=14)

    if not prediction_record:
        next_rows = [
            ["Target Time", "--"],
            ["Signal", "--"],
            ["Probability", "--"],
            ["Model", "--"],
            ["Accuracy", "--"],
            ["F1", "--"],
        ]
        signal_color = "#6c757d"
    else:
        signal = prediction_record.get("predicted_signal", "--")
        signal_color = "#1b7f4a" if signal == "UP" else "#c44536"
        if prediction_record.get("status") == "failed":
            signal = "FAILED"
            signal_color = "#6c757d"
        target_time = prediction_record.get("target_candle_timestamp", "--")
        if target_time != "--":
            target_time = format_dual_time(target_time)
        next_rows = [
            ["Target Time", target_time],
            ["Signal", signal],
            ["Probability", f"{float(prediction_record.get('probability_up', 0.0)):.1%}"],
            ["Model", prediction_record.get("model_name", "--")],
            ["Accuracy", f"{float(prediction_record.get('model_accuracy', 0.0)):.3f}"],
            ["F1", f"{float(prediction_record.get('model_f1', 0.0)):.3f}"],
        ]

    next_table = ax_next.table(
        cellText=next_rows,
        colLabels=["Field", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    next_table.scale(1, 1.65)
    next_table.auto_set_font_size(False)
    next_table.set_fontsize(10.5)

    for (row_idx, col_idx), cell in next_table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold")
            continue
        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        if col_idx == 1:
            value_text = str(cell.get_text().get_text())
            if row_idx == 2:
                cell.set_facecolor("#e9f5ec" if signal_color == "#1b7f4a" else "#f8d7da")
                cell.set_text_props(color=signal_color, weight="bold")
            elif row_idx == 3:
                cell.set_text_props(color="#1f3c4d", weight="bold")
            elif value_text.endswith("%"):
                cell.set_text_props(color=signal_color, weight="bold")

    ax_table.axis("off")
    table_rows = [
        ["Total Predictions", stats["total_predictions"]],
        ["Total Correct", stats["total_correct"]],
        ["Total Accuracy %", f"{stats['total_accuracy_pct']:.1f}%"],
        ["Total Failed", stats["total_failed"]],
        ["Last 24h Predictions", stats["last_24h_predictions"]],
        ["Last 24h Correct", stats["last_24h_correct"]],
        ["Last 24h Accuracy %", f"{stats['last_24h_accuracy_pct']:.1f}%"],
        ["Last 24h Failed", stats["last_24h_failed"]],
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax_table.set_title("Bot Stats", fontsize=15, weight="bold", pad=14)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d8cbb8")
        if row_idx == 0:
            cell.set_facecolor("#1f3c4d")
            cell.set_text_props(color="white", weight="bold")
            continue
        cell.set_facecolor("#fffaf3" if row_idx % 2 else "#f6eee1")
        if col_idx == 1:
            value_text = str(cell.get_text().get_text())
            if "%" in value_text:
                pct = float(value_text.replace("%", ""))
                color = "#1b7f4a" if pct >= 50 else "#c44536"
                cell.set_text_props(color=color, weight="bold")

    fig.suptitle(
        "BTC Directional Bot Validation Dashboard",
        fontsize=18,
        weight="bold",
        color="#1f3c4d",
        y=0.98,
    )
    fig.text(
        0.015,
        0.02,
        "Green = UP / correct strength   Red = DOWN / misses   Gray = failed run",
        fontsize=10,
        color="#5b5f66",
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DASHBOARD_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    client, registered_model_name, experiment_name = configure_tracking()
    champion = get_current_champion_info(client, registered_model_name)
    prediction_record = load_last_prediction()

    history = load_history()
    if prediction_record is None:
        stats = compute_stats(history)
        render_dashboard(history, stats, prediction_record)
        print(
            "No local last_prediction.json found. "
            "Skipping validation and refreshing the dashboard with existing history only."
        )
        return

    if prediction_already_recorded(history, prediction_record):
        stats = compute_stats(history)
        render_dashboard(history, stats, prediction_record)
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
                "target_close": pd.NA,
            },
        )
        stats = compute_stats(history)
        render_dashboard(history, stats, prediction_record)
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

    try:
        candles = fetch_recent_candles()
    except Exception as exc:
        stats = compute_stats(history)
        render_dashboard(history, stats, prediction_record)
        print(f"Could not fetch validation candles. Skipping validation for now: {exc}")
        return
    actual = resolve_actual_direction(candles, prediction_record)
    if actual is None:
        stats = compute_stats(history)
        render_dashboard(history, stats, prediction_record)
        print("Target candle is not available yet. Dashboard refreshed without new validation row.")
        return

    actual_label, target_timestamp, reference_close, target_close = actual
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
            "reference_open": prediction_record.get("reference_open"),
            "reference_close": reference_close,
            "target_close": target_close,
        },
    )
    stats = compute_stats(history)
    render_dashboard(history, stats, prediction_record)

    print(
        "Validation complete:",
        json.dumps(
            {
                "champion_version": None if champion is None else champion["version"],
                "model_name": prediction_record.get("model_name"),
                "predicted": predicted_label,
                "actual": actual_label,
                "result": result,
                "reference_close": reference_close,
                "target_close": target_close,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Validate the prior BTC directional prediction and render a simple dashboard.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import MlflowClient

import main as tournament


HISTORY_PATH = Path("history.csv")
DASHBOARD_PATH = Path("assets/dashboard.png")
LOCAL_LAST_PREDICTION_PATH = Path("last_prediction.json")


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


def load_last_prediction_from_mlflow(client: MlflowClient, experiment_name: str) -> dict[str, Any] | None:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    for run in runs:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                downloaded = client.download_artifacts(
                    run.info.run_id,
                    tournament.LAST_PREDICTION_PATH.name,
                    temp_dir,
                )
                return json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def load_last_prediction(client: MlflowClient, experiment_name: str) -> dict[str, Any] | None:
    if LOCAL_LAST_PREDICTION_PATH.exists():
        return json.loads(LOCAL_LAST_PREDICTION_PATH.read_text(encoding="utf-8"))
    return load_last_prediction_from_mlflow(client, experiment_name)


def fetch_recent_candles() -> pd.DataFrame:
    return tournament.fetch_ohlcv(limit=200)


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
            columns=["timestamp", "predicted", "actual", "result", "failed", "status"]
        )
    history = pd.read_csv(HISTORY_PATH)
    if history.empty:
        return pd.DataFrame(
            columns=["timestamp", "predicted", "actual", "result", "failed", "status"]
        )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    if "failed" not in history.columns:
        history["failed"] = 0
    if "status" not in history.columns:
        history["status"] = "validated"
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
        }

    now = pd.Timestamp.utcnow()
    last_24h_cutoff = now - pd.Timedelta(hours=24)
    recent = history[history["timestamp"] >= last_24h_cutoff]
    return {
        "total_predictions": int(len(history)),
        "total_correct": int(history["result"].sum()),
        "total_failed": int(history["failed"].sum()),
        "last_24h_predictions": int(len(recent)),
        "last_24h_correct": int(recent["result"].sum()),
        "last_24h_failed": int(recent["failed"].sum()),
    }


def render_dashboard(history: pd.DataFrame, stats: dict[str, int]) -> None:
    if history.empty:
        recent_correct = 0
        recent_incorrect = 0
        recent_failed = 0
    else:
        now = pd.Timestamp.utcnow()
        last_24h_cutoff = now - pd.Timedelta(hours=24)
        recent = history[history["timestamp"] >= last_24h_cutoff]
        recent_correct = int(recent["result"].sum())
        recent_failed = int(recent["failed"].sum())
        recent_incorrect = int(len(recent) - recent_correct - recent_failed)

    fig, (ax_chart, ax_table) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    fig.patch.set_facecolor("#f7f3eb")

    ax_chart.bar(
        ["Correct", "Incorrect", "Failed"],
        [recent_correct, recent_incorrect, recent_failed],
        color=["#2d6a4f", "#bc4749", "#6c757d"],
        width=0.55,
    )
    ax_chart.set_title("Last 24 Hours", fontsize=14, weight="bold")
    ax_chart.set_ylabel("Predictions")
    ax_chart.grid(axis="y", alpha=0.25)

    ax_table.axis("off")
    table_rows = [
        ["Total Predictions", stats["total_predictions"]],
        ["Total Correct", stats["total_correct"]],
        ["Total Failed", stats["total_failed"]],
        ["Last 24h Predictions", stats["last_24h_predictions"]],
        ["Last 24h Correct", stats["last_24h_correct"]],
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
    ax_table.set_title("Bot Stats", fontsize=14, weight="bold", pad=12)

    fig.suptitle("BTC Directional Bot Validation Dashboard", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(DASHBOARD_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    client, registered_model_name, experiment_name = configure_tracking()
    champion = get_current_champion_info(client, registered_model_name)
    prediction_record = load_last_prediction(client, experiment_name)

    history = load_history()
    if prediction_record is None:
        stats = compute_stats(history)
        render_dashboard(history, stats)
        print("No prior prediction found. Dashboard refreshed with existing history only.")
        return

    if prediction_already_recorded(history, prediction_record):
        stats = compute_stats(history)
        render_dashboard(history, stats)
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
            },
        )
        stats = compute_stats(history)
        render_dashboard(history, stats)
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

    candles = fetch_recent_candles()
    actual = resolve_actual_direction(candles, prediction_record)
    if actual is None:
        stats = compute_stats(history)
        render_dashboard(history, stats)
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
        },
    )
    stats = compute_stats(history)
    render_dashboard(history, stats)

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

#!/usr/bin/env python3
"""
Render dashboards for the consolidated BTC workflow from a single history CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import main as tournament
from pipelines.consolidated import config, io


DASHBOARD_VARIANTS = (
    {
        "path": config.DASHBOARD_PATH,
        "title": "BTC Consolidated Validation Dashboard",
        "reverse": False,
        "market_hours_only": False,
    },
    {
        "path": config.DASHBOARD_REVERSE_PATH,
        "title": "BTC Consolidated Reverse Dashboard",
        "reverse": True,
        "market_hours_only": False,
    },
    {
        "path": config.DASHBOARD_MARKET_HOURS_PATH,
        "title": "BTC Consolidated Market Hours Dashboard",
        "reverse": False,
        "market_hours_only": True,
    },
    {
        "path": config.DASHBOARD_MARKET_HOURS_REVERSE_PATH,
        "title": "BTC Consolidated Market Hours Reverse Dashboard",
        "reverse": True,
        "market_hours_only": True,
    },
)


def load_last_prediction() -> dict[str, Any] | None:
    return io.read_json(config.LAST_PREDICTION_PATH)


def parse_model_predictions(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fetch_validation_candles(history: pd.DataFrame) -> pd.DataFrame | None:
    if history.empty:
        return None
    unresolved = history[
        (history["status"] == "success")
        & (history["actual"].fillna("").astype(str).str.strip() == "")
    ].copy()
    if unresolved.empty:
        return None
    oldest_target = pd.Timestamp(unresolved["target_candle_timestamp"].min())
    now_utc = pd.Timestamp.now(tz="UTC")
    lookback_hours = int(max((now_utc - oldest_target).total_seconds() // 3600 + 6, 48))
    lookback_hours = min(max(lookback_hours, 48), tournament.LOOKBACK_HOURS)
    return tournament.fetch_ohlcv(limit=lookback_hours, min_candles=min(lookback_hours, 5000))


def resolve_actual(row: pd.Series, candles: pd.DataFrame) -> tuple[str, str, float, float] | None:
    target_ts = pd.Timestamp(row["target_candle_timestamp"])
    if pd.Timestamp.now(tz="UTC") < target_ts:
        return None
    indexed = candles.copy()
    indexed["timestamp"] = pd.to_datetime(indexed["timestamp"], utc=True)
    indexed = indexed.set_index("timestamp")
    if target_ts not in indexed.index:
        return None
    target_open = float(indexed.loc[target_ts, "open"])
    target_close = float(indexed.loc[target_ts, "close"])
    actual = "UP" if target_close >= target_open else "DOWN"
    predicted = str(row.get("predicted_signal", "")).strip().upper()
    result = "WIN" if predicted == actual else "LOSS"
    return actual, result, target_open, target_close


def refresh_history_outcomes(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    candles = fetch_validation_candles(history)
    if candles is None:
        return history

    updated = history.copy()
    for index, row in updated.iterrows():
        if str(row.get("status", "")).strip().lower() != "success":
            continue
        if str(row.get("actual", "")).strip():
            continue
        resolved = resolve_actual(row, candles)
        if resolved is None:
            continue
        actual, result, target_open, target_close = resolved
        updated.at[index, "actual"] = actual
        updated.at[index, "result"] = result
        updated.at[index, "target_open"] = target_open
        updated.at[index, "target_close"] = target_close

    updated = io.ensure_history_schema(updated)
    updated.to_csv(config.HISTORY_PATH, index=False)
    return updated


def reverse_signal(value: str) -> str:
    upper = str(value).strip().upper()
    if upper == "UP":
        return "DOWN"
    if upper == "DOWN":
        return "UP"
    return upper


def filter_history(history: pd.DataFrame, *, market_hours_only: bool) -> pd.DataFrame:
    filtered = history.copy()
    filtered = filtered[filtered["status"] == "success"].copy()
    if market_hours_only:
        filtered = filtered[filtered["track"].isin(["market_hours", "market_hours_daily"])].copy()
    return filtered


def compute_plot_frame(history: pd.DataFrame, *, reverse: bool) -> pd.DataFrame:
    frame = history.copy()
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["predicted_for_plot"] = (
        frame["predicted_signal"].apply(reverse_signal)
        if reverse
        else frame["predicted_signal"].astype(str).str.upper()
    )
    frame["is_resolved"] = frame["actual"].fillna("").astype(str).str.strip() != ""
    frame["plot_result"] = frame.apply(
        lambda row: (
            "WIN"
            if row["is_resolved"] and str(row["predicted_for_plot"]).upper() == str(row["actual"]).upper()
            else ("LOSS" if row["is_resolved"] else "")
        ),
        axis=1,
    )
    frame["win_value"] = frame["plot_result"].map({"WIN": 1.0, "LOSS": 0.0}).fillna(pd.NA)
    return frame.sort_values("timestamp")


def render_dashboard(
    history: pd.DataFrame,
    prediction_payload: dict[str, Any] | None,
    *,
    path: Path,
    title: str,
    reverse: bool,
    market_hours_only: bool,
) -> None:
    filtered = compute_plot_frame(
        filter_history(history, market_hours_only=market_hours_only),
        reverse=reverse,
    )

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.4], width_ratios=[1.7, 1.3])
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_table = fig.add_subplot(grid[1, :])

    fig.suptitle(title, fontsize=18, weight="bold")

    if filtered.empty:
        ax_curve.text(0.5, 0.5, "No consolidated prediction history yet", ha="center", va="center")
        ax_curve.set_axis_off()
        ax_bar.set_axis_off()
        ax_table.set_axis_off()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    resolved = filtered[filtered["is_resolved"]].copy()
    if not resolved.empty:
        for track, group in resolved.groupby("track"):
            curve = group.sort_values("timestamp").copy()
            curve["cum_accuracy"] = curve["win_value"].astype(float).expanding().mean() * 100.0
            ax_curve.plot(curve["timestamp"], curve["cum_accuracy"], marker="o", label=track)
        ax_curve.set_ylabel("Cumulative accuracy (%)")
        ax_curve.set_xlabel("Target timestamp (UTC)")
        ax_curve.set_ylim(0, 100)
        ax_curve.grid(alpha=0.25)
        ax_curve.legend()
    else:
        ax_curve.text(0.5, 0.5, "Waiting for resolved predictions", ha="center", va="center")
        ax_curve.set_axis_off()

    recent = filtered.tail(16).copy()
    ax_bar.bar(
        range(len(recent)),
        recent["probability_up"].astype(float),
        color=["#2f855a" if s == "UP" else "#c53030" for s in recent["predicted_for_plot"]],
    )
    ax_bar.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("Recent probability-up values")
    ax_bar.set_xticks(range(len(recent)))
    ax_bar.set_xticklabels(recent["track"], rotation=45, ha="right")

    ax_table.axis("off")
    latest_rows = filtered.tail(10).copy()
    table_rows: list[list[str]] = []
    for _, row in latest_rows.iterrows():
        table_rows.append(
            [
                pd.Timestamp(row["timestamp"]).strftime("%m-%d %H:%M"),
                str(row["track"]),
                str(row["predicted_for_plot"]),
                str(row.get("actual", "") or "--"),
                str(row.get("plot_result", "") or "--"),
                f"{float(row['probability_up']):.1%}",
                str(row.get("best_champion_name", "") or "--"),
            ]
        )

    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Time", "Track", "Signal", "Actual", "Result", "Prob UP", "Champion"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax_table.set_title("Recent consolidated predictions", pad=12)

    summary_lines = []
    if prediction_payload is not None:
        active_tracks = [
            track_id
            for track_id, payload in prediction_payload.get("tracks", {}).items()
            if payload.get("status") == "success"
        ]
        summary_lines.append(f"Active tracks this run: {', '.join(active_tracks) or 'none'}")
        summary_lines.append(
            f"Target candle: {prediction_payload.get('target_candle_timestamp', '--')}"
        )
    if not resolved.empty:
        total = len(resolved)
        wins = int((resolved["plot_result"] == "WIN").sum())
        summary_lines.append(f"Resolved accuracy: {wins}/{total} ({(wins / total) * 100:.1f}%)")
    fig.text(0.02, 0.02, " | ".join(summary_lines), fontsize=10)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    config.ensure_output_dirs()
    history = io.load_history()
    history = refresh_history_outcomes(history)
    prediction_payload = load_last_prediction()
    for variant in DASHBOARD_VARIANTS:
        render_dashboard(
            history,
            prediction_payload,
            path=variant["path"],
            title=variant["title"],
            reverse=variant["reverse"],
            market_hours_only=variant["market_hours_only"],
        )


if __name__ == "__main__":
    main()

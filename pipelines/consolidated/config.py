#!/usr/bin/env python3
"""
Configuration for the isolated consolidated BTC workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "consolidated"
ASSET_DIR = ROOT / "assets" / "consolidated"

LAST_PREDICTION_PATH = ARTIFACT_DIR / "last_prediction.json"
HISTORY_PATH = ARTIFACT_DIR / "history.csv"
WORKFLOW_LOG_PATH = ARTIFACT_DIR / "workflow.log"
COMPARISON_SUMMARY_PATH = ARTIFACT_DIR / "comparison_summary.json"

DASHBOARD_PATH = ASSET_DIR / "dashboard.png"
DASHBOARD_REVERSE_PATH = ASSET_DIR / "dashboard_reverse.png"
DASHBOARD_MARKET_HOURS_PATH = ASSET_DIR / "dashboard_market_hours.png"
DASHBOARD_MARKET_HOURS_REVERSE_PATH = ASSET_DIR / "dashboard_market_hours_reverse.png"

WORKFLOW_NAME = "consolidated-hourly"
WORKFLOW_VARIANT = "train_once_compare_across_all_tracks"
DEFAULT_EXPERIMENT_PREFIX = "btc-consolidated"
DEFAULT_MODEL_NAME_SUFFIX = "-consolidated"
MODEL_FAMILIES = ("rf", "xgb", "mlp_sklearn", "lstm", "transformer", "nn")


@dataclass(frozen=True)
class TrackConfig:
    id: str
    model_suffix: str
    workflow_name: str
    workflow_variant: str
    market_hours_only: bool = False
    daily_refresh_mode: str = "never"


TRACKS: tuple[TrackConfig, ...] = (
    TrackConfig(
        id="hourly_24h",
        model_suffix="hourly-24h",
        workflow_name="consolidated-hourly-24h",
        workflow_variant="hourly_24h_always_refresh",
    ),
    TrackConfig(
        id="hourly_daily",
        model_suffix="hourly-daily",
        workflow_name="consolidated-hourly-daily",
        workflow_variant="hourly_prediction_daily_refresh_at_midnight_et",
        daily_refresh_mode="midnight_et",
    ),
    TrackConfig(
        id="market_hours",
        model_suffix="market-hours",
        workflow_name="consolidated-market-hours",
        workflow_variant="hourly_prediction_market_hours_only",
        market_hours_only=True,
    ),
    TrackConfig(
        id="market_hours_daily",
        model_suffix="market-hours-daily",
        workflow_name="consolidated-market-hours-daily",
        workflow_variant="market_hours_prediction_same_day_refresh",
        market_hours_only=True,
        daily_refresh_mode="market_hours_day",
    ),
)


def get_env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_experiment_prefix() -> str:
    return get_env_str("MLFLOW_CONSOLIDATED_EXPERIMENT_PREFIX") or DEFAULT_EXPERIMENT_PREFIX


def resolve_base_registered_model_name() -> str:
    explicit = get_env_str("MLFLOW_CONSOLIDATED_MODEL_NAME")
    if explicit:
        return explicit
    base_name = get_env_str("MLFLOW_MODEL_NAME") or "btc-usdt-directional-classifier"
    return f"{base_name}{DEFAULT_MODEL_NAME_SUFFIX}"


def registered_model_name_for_track(track: TrackConfig) -> str:
    return f"{resolve_base_registered_model_name()}-{track.model_suffix}"


def tracked_output_files() -> list[Path]:
    return [
        LAST_PREDICTION_PATH,
        HISTORY_PATH,
        WORKFLOW_LOG_PATH,
        COMPARISON_SUMMARY_PATH,
        DASHBOARD_PATH,
        DASHBOARD_REVERSE_PATH,
        DASHBOARD_MARKET_HOURS_PATH,
        DASHBOARD_MARKET_HOURS_REVERSE_PATH,
    ]


def ensure_output_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

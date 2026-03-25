#!/usr/bin/env python3
"""
Isolated hourly BTC tournament that only trains during the ET day session.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

import main as tournament
from market_hours_common import (
    describe_window,
    next_target_timestamp_utc,
    should_run_prediction_window,
    should_run_training_window,
)


MARKET_HOURS_LAST_PREDICTION_PATH = Path("last_prediction_market_hours.json")
MARKET_HOURS_EXPERIMENT_PREFIX = "btc-market-hours"
MARKET_HOURS_MODEL_NAME_SUFFIX = "-market-hours"
MARKET_HOURS_WORKFLOW_NAME = "market-hours-hourly"
MARKET_HOURS_WORKFLOW_VARIANT = "hourly_train_7am_7pm_et_predict_8am_8pm_et"


def configure_market_hours_paths() -> None:
    tournament.LAST_PREDICTION_PATH = MARKET_HOURS_LAST_PREDICTION_PATH


def resolve_market_hours_registered_model_name() -> str:
    explicit_name = tournament.get_env_str("MLFLOW_MARKET_HOURS_MODEL_NAME")
    if explicit_name:
        return explicit_name
    base_name = tournament.get_env_str("MLFLOW_MODEL_NAME") or tournament.DEFAULT_MODEL_NAME
    return f"{base_name}{MARKET_HOURS_MODEL_NAME_SUFFIX}"


def configure_market_hours_tracking() -> str:
    tournament.DEFAULT_EXPERIMENT_PREFIX = MARKET_HOURS_EXPERIMENT_PREFIX
    registered_model_name = resolve_market_hours_registered_model_name()
    tournament.configure_tracking()
    return registered_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated ET market-hours BTC tournament."
    )
    parser.add_argument(
        "--reset-champion-from-challenger",
        action="store_true",
        help="Ignore the current champion comparison and choose from the current challenger only.",
    )
    return parser.parse_args()


def enrich_prediction_record(prediction_record: dict[str, object]) -> dict[str, object]:
    updated = dict(prediction_record)
    updated["workflow_name"] = MARKET_HOURS_WORKFLOW_NAME
    updated["workflow_variant"] = MARKET_HOURS_WORKFLOW_VARIANT
    return updated


def write_failed_prediction_record(exc: Exception) -> None:
    target_timestamp = next_target_timestamp_utc()
    failure_record = {
        "status": "failed",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "registered_model_name": resolve_market_hours_registered_model_name(),
        "symbol": tournament.SYMBOL,
        "timeframe": tournament.TIMEFRAME,
        "reference_candle_timestamp": (target_timestamp - pd.Timedelta(hours=1)).isoformat(),
        "target_candle_timestamp": target_timestamp.isoformat(),
        "error": str(exc),
        "workflow_name": MARKET_HOURS_WORKFLOW_NAME,
        "workflow_variant": MARKET_HOURS_WORKFLOW_VARIANT,
    }
    MARKET_HOURS_LAST_PREDICTION_PATH.write_text(
        json.dumps(failure_record, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    configure_market_hours_paths()
    tournament.log_step("Initialize ET market-hours BTC pipeline")
    print(describe_window())

    if not should_run_prediction_window() or not should_run_training_window():
        print(
            "Skipping ET market-hours hourly training: "
            "training is only allowed when the current ET hour is 7am-7pm "
            "and the next target candle is within 8am-8pm ET."
        )
        return

    tournament.set_seed()
    registered_model_name = configure_market_hours_tracking()
    client = MlflowClient()

    tournament.log_step("Fetch BTC/USDT market data")
    raw = tournament.fetch_ohlcv(
        limit=tournament.LOOKBACK_HOURS,
        min_candles=5000,
        retry_binanceus=True,
        retry_binanceus_attempts=3,
    )
    tournament.log_step("Build features and dataset splits")
    featured = tournament.add_features(raw)
    train_df, valid_df, future_row = tournament.split_dataset(
        featured,
        tournament.VALIDATION_HOURS,
    )

    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()

    tournament.log_step("Train challenger zoo")
    challengers, cv_summary = tournament.train_challengers(train_df, valid_df)
    challenger_results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )
    full_labeled_df = pd.concat([train_df, valid_df], ignore_index=True)
    refit_challengers = tournament.retrain_challengers_on_full_data(full_labeled_df)
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    full_prediction_frame = pd.concat([full_labeled_df, future_row], ignore_index=True)
    for result in challenger_results:
        refit_candidate = refit_by_family[result["family"]]
        result["candidate"] = refit_candidate
        result["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, full_prediction_frame)[-1]
        )
        result["next_signal"] = tournament.prediction_to_signal(result["next_probability"])

    all_results = list(challenger_results)
    challenger_by_family = {result["family"]: result for result in challenger_results}
    family_decisions: list[dict[str, object]] = []
    active_results_by_family: dict[str, dict[str, object]] = {}

    if args.reset_champion_from_challenger:
        print("Champion comparison disabled. Selecting from the current challenger leaderboard only.")

    for family, challenger_result in sorted(challenger_by_family.items()):
        champion_result: dict[str, object] | None = None
        champion_meta: dict[str, str] | None = None
        family_registered_model_name = tournament.registered_model_name_for_family(
            registered_model_name,
            family,
        )
        if not args.reset_champion_from_challenger:
            champion_candidate, champion_meta = tournament.get_current_champion(
                client,
                family_registered_model_name,
                alias=tournament.CHAMPION_ALIAS,
            )
            if champion_candidate is not None and champion_meta is not None:
                champion_result = tournament.evaluate_champion(
                    champion_candidate,
                    train_df,
                    valid_df,
                    future_row,
                )
                champion_result["registry_version"] = champion_meta["version"]
                all_results.append(champion_result)

        null_model_block = (
            challenger_result["f1"] <= 0.5 or challenger_result["accuracy"] <= 0.5
        )
        if champion_result is None:
            should_promote = True
            active_family_result = challenger_result
        else:
            should_promote = (
                challenger_result["f1"] > champion_result["f1"] and not null_model_block
            )
            active_family_result = challenger_result if should_promote else champion_result

        active_results_by_family[family] = active_family_result
        family_decisions.append(
            {
                "family": family,
                "registered_model_name": family_registered_model_name,
                "challenger": challenger_result,
                "champion": champion_result,
                "champion_meta": champion_meta,
                "should_promote": should_promote,
                "null_model_block": null_model_block,
                "active_result": active_family_result,
            }
        )

    active_result = sorted(active_results_by_family.values(), key=tournament.ranking_key)[0]

    with mlflow.start_run(run_name="btc-directional-market-hours-tournament"):
        tournament.print_scoreboard(all_results)
        mlflow.set_tags(
            {
                "asset": tournament.SYMBOL,
                "timeframe": tournament.TIMEFRAME,
                "validation_hours": str(tournament.VALIDATION_HOURS),
                "market_hours_workflow": "true",
            }
        )
        mlflow.log_params(
            {
                "lookback_hours": tournament.LOOKBACK_HOURS,
                "validation_hours": tournament.VALIDATION_HOURS,
                "cross_validation_folds": tournament.CROSS_VALIDATION_FOLDS,
                "sequence_length": tournament.SEQUENCE_LENGTH,
                "rf_estimators": 400,
                "xgb_estimators": 500,
                "lstm_epochs": 40,
                "transformer_epochs": 36,
                "nn_epochs": 48,
            }
        )
        promotion_feature_rows = full_labeled_df
        for decision in family_decisions:
            challenger_result = decision["challenger"]
            champion_result = decision["champion"]
            if decision["null_model_block"] and champion_result is not None:
                print(
                    f"Promotion blocked for {challenger_result['name']}: "
                    f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                )
            elif decision["null_model_block"] and champion_result is None:
                print(
                    f"Bootstrapping missing {decision['registered_model_name']} despite null-model guard: "
                    f"F1={challenger_result['f1']:.3f}, Accuracy={challenger_result['accuracy']:.3f}"
                )

            if decision["should_promote"]:
                new_version = tournament.promote_champion(
                    client=client,
                    registered_model_name=decision["registered_model_name"],
                    winner=challenger_result,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    feature_rows=promotion_feature_rows,
                    alias=tournament.CHAMPION_ALIAS,
                )
                decision["active_result"]["registry_version"] = new_version
                decision["active_result"]["source"] = "champion"
                print(
                    f"{challenger_result['name']} -> promoted to "
                    f"{decision['registered_model_name']} version {new_version}"
                )
            elif champion_result is not None:
                decision["active_result"]["registry_version"] = decision["champion_meta"]["version"]
                print(
                    f"{champion_result['name']} -> retained as {decision['registered_model_name']} "
                    f"version {decision['champion_meta']['version']}"
                )
            else:
                print(
                    f"{challenger_result['name']} -> no existing "
                    f"{decision['registered_model_name']} and not promoted"
                )

        best_registered_result = next(
            (
                result
                for result in sorted(active_results_by_family.values(), key=tournament.ranking_key)
                if result.get("registry_version") is not None
            ),
            None,
        )
        if best_registered_result is not None:
            active_result["best_overall_registry_version"] = best_registered_result["registry_version"]
            print(
                f"Current best across champions: {best_registered_result['candidate'].name} "
                f"({best_registered_result['family']}) version {best_registered_result['registry_version']}"
            )
        else:
            print("Current best across champions: no registered family champion available yet")

        prediction_record = tournament.build_prediction_record(
            active_result=active_result,
            active_results_by_family=active_results_by_family,
            future_row=future_row,
            registered_model_name=registered_model_name,
        )
        prediction_record = enrich_prediction_record(prediction_record)
        tournament.log_step("Write latest market-hours prediction metadata")
        MARKET_HOURS_LAST_PREDICTION_PATH.write_text(
            json.dumps(prediction_record, indent=2),
            encoding="utf-8",
        )

        tournament.log_step("Log tournament results to MLflow")
        tournament.log_challenger_summary(challenger_results)
        tournament.log_comparison_metrics(
            active_results_by_family=active_results_by_family,
            active_result=active_result,
        )
        mlflow.log_text(
            json.dumps(
                [tournament.serialize_result(row) for row in sorted(all_results, key=tournament.ranking_key)],
                indent=2,
            ),
            "tournament_results.json",
        )
        mlflow.log_text(
            json.dumps(prediction_record, indent=2),
            MARKET_HOURS_LAST_PREDICTION_PATH.name,
        )

    print(
        f"Upcoming hour probability: {active_result['next_probability']:.1%} chance of UP"
    )
    print(f"Final signal: {active_result['next_signal']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_failed_prediction_record(exc)
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise

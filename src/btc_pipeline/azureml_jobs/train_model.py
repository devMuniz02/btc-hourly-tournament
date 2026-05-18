from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.btc_pipeline import main as tournament
from src.btc_pipeline.azureml_jobs.common import ensure_dir, read_dataframe, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and package the best BTC challenger model.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--metrics-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    model_dir = ensure_dir(args.model_dir)

    tournament.set_seed()
    train_df = read_dataframe(input_dir / "train.csv")
    valid_df = read_dataframe(input_dir / "valid.csv")
    future_row = read_dataframe(input_dir / "future.csv")
    full_labeled_df = read_dataframe(input_dir / "full_labeled.csv")
    metadata = read_json(input_dir / "metadata.json")

    challengers, cv_summary = tournament.train_challengers(train_df, valid_df)
    challenger_results = tournament.build_results(
        challengers,
        train_df,
        valid_df,
        future_row,
        cv_summary=cv_summary,
    )

    refit_challengers = tournament.retrain_challengers_on_full_data(full_labeled_df)
    refit_by_family = {candidate.family: candidate for candidate in refit_challengers}
    prediction_frame = pd.concat([full_labeled_df, future_row], ignore_index=True)

    for result in challenger_results:
        refit_candidate = refit_by_family[result["family"]]
        result["candidate"] = refit_candidate
        result["next_probability"] = float(
            tournament.predict_candidate_probabilities(refit_candidate, prediction_frame)[-1]
        )
        result["next_signal"] = tournament.prediction_to_signal(result["next_probability"])

    best_result = sorted(challenger_results, key=tournament.ranking_key)[0]
    tournament.save_candidate_package(best_result["candidate"], model_dir)

    metrics_payload = {
        "best_model_name": best_result["candidate"].name,
        "best_model_family": best_result["family"],
        "best_accuracy": float(best_result["accuracy"]),
        "best_f1": float(best_result["f1"]),
        "best_cv_accuracy": float(best_result.get("cv_accuracy", best_result["accuracy"])),
        "best_cv_f1": float(best_result.get("cv_f1", best_result["f1"])),
        "probability_up": float(best_result["next_probability"]),
        "predicted_signal": best_result["next_signal"],
        "registered_model_basename": metadata.get("registered_model_name", "btc-direction-model"),
        "scoreboard": [
            tournament.serialize_result(result, include_registry_version=False)
            for result in sorted(challenger_results, key=tournament.ranking_key)
        ],
    }
    write_json(metrics_payload, args.metrics_path)
    print(f"Packaged best model in {model_dir.resolve()}")


if __name__ == "__main__":
    main()

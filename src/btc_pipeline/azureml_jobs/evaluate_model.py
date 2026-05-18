from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.btc_pipeline import main as tournament
from src.btc_pipeline.azureml_jobs.common import read_dataframe, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a packaged BTC model on the validation split.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metadata = read_json(input_dir / "metadata.json")
    train_df = read_dataframe(input_dir / "train.csv")
    valid_df = read_dataframe(input_dir / "valid.csv")
    future_row = read_dataframe(input_dir / "future.csv")

    candidate = tournament.load_candidate_package(args.model_dir)
    eval_frame = pd.concat([train_df, valid_df], ignore_index=True)
    prediction_frame = pd.concat([eval_frame, future_row], ignore_index=True)
    probabilities = tournament.predict_candidate_probabilities(candidate, prediction_frame)
    valid_probs = probabilities[len(train_df) : -1]
    metrics = tournament.evaluate_probabilities(
        valid_probs,
        valid_df["target"].to_numpy(dtype="int32"),
    )

    write_json(
        {
            "model_name": candidate.name,
            "model_family": candidate.family,
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "probability_up": float(probabilities[-1]),
            "predicted_signal": tournament.prediction_to_signal(float(probabilities[-1])),
            "validation_start": metadata["validation_start"],
            "validation_end": metadata["validation_end"],
        },
        args.evaluation_path,
    )
    print(f"Evaluation written to {Path(args.evaluation_path).resolve()}")


if __name__ == "__main__":
    main()

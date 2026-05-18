from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.btc_pipeline import main as tournament
from src.btc_pipeline.azureml_jobs.common import read_dataframe, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score the next BTC direction with a packaged model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = tournament.load_candidate_package(args.model_dir)
    full_labeled_df = read_dataframe(Path(args.input_dir) / "full_labeled.csv")
    future_row = read_dataframe(Path(args.input_dir) / "future.csv")
    prediction_frame = pd.concat([full_labeled_df, future_row], ignore_index=True)
    probability = float(tournament.predict_candidate_probabilities(candidate, prediction_frame)[-1])

    write_json(
        {
            "model_name": candidate.name,
            "model_family": candidate.family,
            "probability_up": probability,
            "predicted_signal": tournament.prediction_to_signal(probability),
            "reference_candle_timestamp": future_row["timestamp"].iloc[0].isoformat(),
            "reference_open": float(future_row["open"].iloc[0]),
            "reference_close": float(future_row["close"].iloc[0]),
        },
        args.output_path,
    )
    print(f"Batch scoring output written to {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()

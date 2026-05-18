from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.btc_pipeline import main as tournament
from src.btc_pipeline.azureml_jobs.common import read_dataframe, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a BTC model package in MLflow/Azure ML.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--evaluation-path", required=True)
    parser.add_argument("--registration-path", required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--experiment-name", default="btc-azure-mlops")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = read_json(args.evaluation_path)
    full_labeled_df = read_dataframe(Path(args.input_dir) / "full_labeled.csv")
    candidate = tournament.load_candidate_package(args.model_dir)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    payload = tournament.get_promotion_payload(candidate, full_labeled_df)[0]
    model_code_path = Path(tournament.__file__).resolve().with_name("mlflow_tournament_model.py")
    client = MlflowClient()

    with mlflow.start_run(run_name=f"register-{candidate.family}") as run:
        mlflow.log_metrics(
            {
                "accuracy": float(evaluation["accuracy"]),
                "f1": float(evaluation["f1"]),
                "probability_up": float(evaluation["probability_up"]),
            }
        )
        mlflow.set_tags(
            {
                "model_name": candidate.name,
                "model_family": candidate.family,
                "portfolio_project": "btc-azure-mlops",
            }
        )
        mlflow.pyfunc.log_model(
            artifact_path=tournament.MODEL_ARTIFACT_NAME,
            python_model=str(model_code_path),
            artifacts={"model_dir": str(payload["package_dir"])},
            input_example=payload["input_example"],
            signature=payload["signature"],
            pip_requirements=payload["pip_requirements"],
        )
        model_uri = f"runs:/{run.info.run_id}/{tournament.MODEL_ARTIFACT_NAME}"
        registration = mlflow.register_model(
            model_uri=model_uri,
            name=args.registered_model_name,
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registration.version,
            key="model_family",
            value=candidate.family,
        )
        client.set_model_version_tag(
            name=args.registered_model_name,
            version=registration.version,
            key="portfolio_project",
            value="btc-azure-mlops",
        )

        write_json(
            {
                "run_id": run.info.run_id,
                "registered_model_name": args.registered_model_name,
                "model_name": candidate.name,
                "model_family": candidate.family,
                "model_uri": model_uri,
                "registered_version": registration.version,
                "status": "registered",
            },
            args.registration_path,
        )
        print(f"Registered model {args.registered_model_name} from run {run.info.run_id}")


if __name__ == "__main__":
    main()

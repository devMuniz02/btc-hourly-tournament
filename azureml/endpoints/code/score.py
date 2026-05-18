from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd


MODEL = None
MODEL_SEQUENCE_LENGTH = None


def _resolve_model_path() -> str:
    model_root = Path(os.getenv("AZUREML_MODEL_DIR", ""))
    if not model_root.exists():
        raise FileNotFoundError("AZUREML_MODEL_DIR was not found in the deployment container.")

    mlmodel_files = list(model_root.rglob("MLmodel"))
    if not mlmodel_files:
        raise FileNotFoundError("Could not find an MLmodel file inside AZUREML_MODEL_DIR.")
    return str(mlmodel_files[0].parent)


def init() -> None:
    global MODEL
    global MODEL_SEQUENCE_LENGTH
    model_path = _resolve_model_path()
    MODEL = mlflow.pyfunc.load_model(model_path)
    candidate = getattr(MODEL, "_model_impl", None)
    python_model = getattr(candidate, "python_model", None)
    wrapped_candidate = getattr(python_model, "candidate", None)
    MODEL_SEQUENCE_LENGTH = getattr(wrapped_candidate, "sequence_length", None)


def _parse_payload(raw_data: Any) -> pd.DataFrame:
    if isinstance(raw_data, (bytes, bytearray)):
        raw_data = raw_data.decode("utf-8")
    if isinstance(raw_data, str):
        payload = json.loads(raw_data)
    else:
        payload = raw_data

    if not isinstance(payload, dict) or "input_data" not in payload:
        raise ValueError("Request payload must be a JSON object with an 'input_data' key.")

    input_data = payload["input_data"]
    if isinstance(input_data, dict) and "columns" in input_data and "data" in input_data:
        return pd.DataFrame(input_data["data"], columns=input_data["columns"])
    if isinstance(input_data, list):
        return pd.DataFrame(input_data)

    raise ValueError(
        "Unsupported payload format. Use {'input_data': {'columns': [...], 'data': [...]}}."
    )


def run(raw_data: Any) -> dict[str, Any]:
    if MODEL is None:
        raise RuntimeError("Model is not loaded. The init() function did not complete successfully.")

    frame = _parse_payload(raw_data)
    required_rows = MODEL_SEQUENCE_LENGTH or 24
    if len(frame) < required_rows:
        return {
            "error": (
                f"At least {required_rows} rows are required for inference because this "
                f"BTC model uses a rolling sequence window. Received {len(frame)} rows."
            ),
            "required_rows": required_rows,
            "received_rows": len(frame),
        }

    predictions = MODEL.predict(frame)
    if isinstance(predictions, pd.DataFrame) and predictions.isna().all().all():
        return {
            "error": (
                f"The model returned no valid prediction. Provide at least {required_rows} "
                "feature rows ordered from oldest to newest."
            ),
            "required_rows": required_rows,
            "received_rows": len(frame),
        }
    if isinstance(predictions, pd.DataFrame):
        records = predictions.to_dict(orient="records")
    else:
        records = pd.DataFrame(predictions).to_dict(orient="records")
    return {"predictions": records}

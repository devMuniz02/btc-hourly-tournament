from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def build_sequence_dataset(
    features: np.ndarray,
    labels: np.ndarray | None,
    indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []

    for idx in indices:
        if idx < seq_len - 1:
            continue
        x_seq.append(features[idx - seq_len + 1 : idx + 1])
        if labels is not None:
            y_seq.append(labels[idx])

    x_array = np.asarray(x_seq, dtype=np.float32)
    if labels is None:
        return x_array, None
    return x_array, np.asarray(y_seq, dtype=np.float32)


def flatten_sequence_features(sequence_features: np.ndarray) -> np.ndarray:
    return sequence_features.reshape(sequence_features.shape[0], -1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class SequenceMLPClassifier(nn.Module):
    def __init__(self, seq_len: int, input_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim
        self.network = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.network(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, 24, model_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.input_proj(x) + self.position_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


def torch_predict_proba(model: nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32)
        return torch.sigmoid(model(tensor)).cpu().numpy()


@dataclass
class TournamentCandidate:
    name: str
    family: str
    model: Any
    feature_columns: list[str]
    sequence_length: int
    scaler: Any = None


def load_candidate_package(model_dir: str | Path) -> TournamentCandidate:
    package_dir = Path(model_dir)
    with (package_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    candidate = TournamentCandidate(
        name=config["name"],
        family=config["family"],
        model=None,
        feature_columns=config["feature_columns"],
        sequence_length=config["sequence_length"],
        scaler=None,
    )

    scaler_path = package_dir / "scaler.joblib"
    if scaler_path.exists():
        candidate.scaler = joblib.load(scaler_path)

    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        candidate.model = joblib.load(package_dir / "model.joblib")
        return candidate

    kwargs = config["model_kwargs"]
    if candidate.family == "lstm":
        candidate.model = LSTMClassifier(**kwargs)
    elif candidate.family == "transformer":
        candidate.model = TransformerClassifier(**kwargs)
    elif candidate.family == "nn":
        candidate.model = SequenceMLPClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported packaged family: {candidate.family}")

    state = torch.load(package_dir / "model.pt", map_location="cpu")
    candidate.model.load_state_dict(state["state_dict"])
    candidate.model.eval()
    return candidate


def predict_candidate_probabilities(
    candidate: TournamentCandidate,
    feature_rows: pd.DataFrame,
) -> np.ndarray:
    features = feature_rows[candidate.feature_columns].to_numpy(dtype=np.float32)
    scaled = candidate.scaler.transform(features) if candidate.scaler is not None else features
    sequence_indices = np.arange(len(feature_rows))
    seq_x, _ = build_sequence_dataset(
        features=scaled,
        labels=None,
        indices=sequence_indices,
        seq_len=candidate.sequence_length,
    )
    output = np.full(len(feature_rows), np.nan, dtype=np.float32)
    if len(seq_x) == 0:
        return output
    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        seq_features = flatten_sequence_features(seq_x)
        output[candidate.sequence_length - 1 :] = candidate.model.predict_proba(seq_features)[:, 1]
        return output
    output[candidate.sequence_length - 1 :] = torch_predict_proba(candidate.model, seq_x)
    return output


class TournamentPyFuncModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.candidate = load_candidate_package(context.artifacts["model_dir"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        probabilities = predict_candidate_probabilities(self.candidate, model_input)
        return pd.DataFrame({"prob_up": probabilities})


mlflow.models.set_model(TournamentPyFuncModel())

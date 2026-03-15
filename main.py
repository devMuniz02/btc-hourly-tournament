#!/usr/bin/env python3
"""
Directional BTC/USDT model tournament with MLflow registry promotion on DagsHub.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_HOURS = 2000
VALIDATION_HOURS = 48
SEQUENCE_LENGTH = 24
SEED = 42
DEVICE = torch.device("cpu")
DEFAULT_EXPERIMENT = "btc-directional-tournament"
DEFAULT_MODEL_NAME = "btc-usdt-directional-classifier"
ARTIFACT_SUBDIR = "packaged_model"
BINANCE_CANDIDATES = [
    ("binance", "BTC/USDT", {}),
    ("binance", "BTC/USDT:USDT", {"options": {"defaultType": "future"}}),
    ("binanceus", "BTC/USDT", {}),
]
FALLBACK_EXCHANGE_CANDIDATES = [
    ("kraken", "BTC/USDT", {}),
    ("okx", "BTC/USDT", {}),
    ("kucoin", "BTC/USDT", {}),
    ("bitfinex", "BTC/USDT", {}),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_env_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def configure_tracking() -> str:
    tracking_uri = get_env_str("MLFLOW_TRACKING_URI")
    username = get_env_str("MLFLOW_TRACKING_USERNAME")
    password = get_env_str("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required.")
    if not username:
        raise RuntimeError("MLFLOW_TRACKING_USERNAME is required.")
    if not password:
        raise RuntimeError("MLFLOW_TRACKING_PASSWORD is required.")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = get_env_str("MLFLOW_EXPERIMENT") or DEFAULT_EXPERIMENT
    mlflow.set_experiment(experiment_name)
    return get_env_str("MLFLOW_MODEL_NAME") or DEFAULT_MODEL_NAME


def fetch_ohlcv(limit: int = LOOKBACK_HOURS) -> pd.DataFrame:
    failures: list[str] = []
    candidates = BINANCE_CANDIDATES + FALLBACK_EXCHANGE_CANDIDATES
    for exchange_id, symbol, extra_config in candidates:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "timeout": 30000,
                    **extra_config,
                }
            )
            candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            if len(candles) < 300:
                raise RuntimeError(
                    f"{exchange_id} returned too few candles ({len(candles)})."
                )
            frame = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            print(f"Fetched {len(frame)} candles from {exchange_id} using {symbol}.")
            return frame
        except Exception as exc:
            failures.append(f"{exchange_id}:{symbol}: {exc}")
            print(f"Exchange attempt failed for {exchange_id} {symbol}: {exc}")

    raise RuntimeError(
        "Could not fetch BTC/USDT candles from Binance variants or fallback exchanges. "
        + " | ".join(failures)
    )


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()

    df["rsi_14"] = compute_rsi(close, 14)
    df["macd"] = ema_fast - ema_slow
    df["volume_delta"] = df["volume"].diff()
    df["price_change_1h"] = close.pct_change(1)
    df["price_change_3h"] = close.pct_change(3)
    df["price_change_6h"] = close.pct_change(6)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi_14",
    "macd",
    "volume_delta",
    "price_change_1h",
    "price_change_3h",
    "price_change_6h",
]


def split_dataset(
    df: pd.DataFrame,
    validation_hours: int = VALIDATION_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df.iloc[:-1].copy()
    future_row = df.iloc[[-1]].copy()
    if len(labeled) <= validation_hours + SEQUENCE_LENGTH:
        raise RuntimeError("Not enough samples to build train and validation splits.")

    train_df = labeled.iloc[:-validation_hours].copy()
    valid_df = labeled.iloc[-validation_hours:].copy()
    return train_df, valid_df, future_row


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
        logits = self.head(output[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logits)


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
        return torch.sigmoid(self.network(x).squeeze(-1))


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
            torch.randn(1, SEQUENCE_LENGTH, model_dim) * 0.02
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
        logits = self.head(x[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logits)


def train_torch_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int = 32,
    patience: int = 4,
) -> nn.Module:
    model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32, device=DEVICE)
    valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32, device=DEVICE)

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_probs = model(valid_x_tensor)
            valid_loss = criterion(valid_probs, valid_y_tensor).item()

        if valid_loss < best_loss:
            best_loss = valid_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    return model


def torch_predict_proba(model: nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32)
        return model(tensor).cpu().numpy()


@dataclass
class TournamentCandidate:
    name: str
    family: str
    model: Any
    feature_columns: list[str]
    sequence_length: int
    scaler: StandardScaler | None = None


def prediction_to_signal(probability: float) -> str:
    return "UP" if probability >= 0.5 else "DOWN"


def evaluate_probabilities(probabilities: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(actual, labels)),
        "f1": float(f1_score(actual, labels, zero_division=0)),
    }


def ranking_key(result: dict[str, Any]) -> tuple[float, float]:
    return (-result["f1"], -result["accuracy"])


def predict_candidate_probabilities(
    candidate: TournamentCandidate,
    feature_rows: pd.DataFrame,
) -> np.ndarray:
    features = feature_rows[candidate.feature_columns].to_numpy(dtype=np.float32)

    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        return candidate.model.predict_proba(features)[:, 1]

    if candidate.scaler is None:
        raise RuntimeError(f"{candidate.name} requires a scaler for inference.")

    scaled = candidate.scaler.transform(features)
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
    output[candidate.sequence_length - 1 :] = torch_predict_proba(candidate.model, seq_x)
    return output


def train_challengers(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> list[TournamentCandidate]:
    train_x = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_y = train_df["target"].to_numpy(dtype=np.int32)

    sequence_frame = pd.concat([train_df, valid_df], ignore_index=True)
    seq_scaler = StandardScaler().fit(train_x)
    scaled_sequence_features = seq_scaler.transform(
        sequence_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    )
    sequence_labels = sequence_frame["target"].to_numpy(dtype=np.float32)

    train_indices = np.arange(len(train_df))
    valid_indices = np.arange(len(train_df), len(train_df) + len(valid_df))

    train_seq_x, train_seq_y = build_sequence_dataset(
        features=scaled_sequence_features,
        labels=sequence_labels,
        indices=train_indices,
        seq_len=SEQUENCE_LENGTH,
    )
    valid_seq_x, valid_seq_y = build_sequence_dataset(
        features=scaled_sequence_features,
        labels=sequence_labels,
        indices=valid_indices,
        seq_len=SEQUENCE_LENGTH,
    )

    if len(train_seq_x) == 0 or len(valid_seq_x) == 0:
        raise RuntimeError("Could not build sequence datasets for the DL challengers.")

    challengers = [
        TournamentCandidate(
            name="RandomForest",
            family="rf",
            model=RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=1,
        ),
        TournamentCandidate(
            name="XGBoost",
            family="xgb",
            model=XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=2,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=1,
        ),
        TournamentCandidate(
            name="MLPClassifier",
            family="mlp_sklearn",
            model=MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                batch_size=32,
                learning_rate_init=1e-3,
                max_iter=200,
                random_state=SEED,
                early_stopping=True,
                n_iter_no_change=10,
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=1,
        ),
        TournamentCandidate(
            name="LSTM",
            family="lstm",
            model=LSTMClassifier(input_dim=len(FEATURE_COLUMNS)),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=seq_scaler,
        ),
        TournamentCandidate(
            name="Transformer",
            family="transformer",
            model=TransformerClassifier(input_dim=len(FEATURE_COLUMNS)),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=seq_scaler,
        ),
        TournamentCandidate(
            name="NN",
            family="nn",
            model=SequenceMLPClassifier(
                seq_len=SEQUENCE_LENGTH,
                input_dim=len(FEATURE_COLUMNS),
            ),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=SEQUENCE_LENGTH,
            scaler=seq_scaler,
        ),
    ]

    for candidate in challengers:
        if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
            candidate.model.fit(train_x, train_y)
        elif candidate.family == "lstm":
            candidate.model = train_torch_classifier(
                candidate.model,
                train_seq_x,
                train_seq_y,
                valid_seq_x,
                valid_seq_y,
                epochs=14,
                learning_rate=8e-4,
            )
        elif candidate.family == "transformer":
            candidate.model = train_torch_classifier(
                candidate.model,
                train_seq_x,
                train_seq_y,
                valid_seq_x,
                valid_seq_y,
                epochs=12,
                learning_rate=7e-4,
            )
        elif candidate.family == "nn":
            candidate.model = train_torch_classifier(
                candidate.model,
                train_seq_x,
                train_seq_y,
                valid_seq_x,
                valid_seq_y,
                epochs=16,
                learning_rate=1e-3,
            )
        else:
            raise ValueError(f"Unsupported challenger family: {candidate.family}")

    return challengers


def build_results(
    candidates: list[TournamentCandidate],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
) -> list[dict[str, Any]]:
    eval_frame = pd.concat([train_df, valid_df], ignore_index=True)
    prediction_frame = pd.concat([eval_frame, future_row], ignore_index=True)
    val_start = len(train_df)
    actual_valid = valid_df["target"].to_numpy(dtype=np.int32)

    results = []
    for candidate in candidates:
        full_probs = predict_candidate_probabilities(candidate, eval_frame)
        valid_probs = full_probs[val_start:]
        if np.isnan(valid_probs).any():
            raise RuntimeError(f"{candidate.name} produced incomplete validation output.")
        metrics = evaluate_probabilities(valid_probs, actual_valid)
        next_probability = float(predict_candidate_probabilities(candidate, prediction_frame)[-1])
        results.append(
            {
                "name": candidate.name,
                "source": "challenger",
                "candidate": candidate,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "next_probability": next_probability,
                "next_signal": prediction_to_signal(next_probability),
            }
        )
    return results


def save_candidate_package(candidate: TournamentCandidate, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": candidate.name,
        "family": candidate.family,
        "feature_columns": candidate.feature_columns,
        "sequence_length": candidate.sequence_length,
    }

    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        joblib.dump(candidate.model, output_dir / "model.joblib")
    else:
        if candidate.scaler is None:
            raise RuntimeError(f"{candidate.name} is missing its scaler.")
        joblib.dump(candidate.scaler, output_dir / "scaler.joblib")
        if candidate.family == "lstm":
            config["model_kwargs"] = {
                "input_dim": candidate.model.input_dim,
                "hidden_dim": candidate.model.hidden_dim,
            }
        elif candidate.family == "transformer":
            config["model_kwargs"] = {
                "input_dim": candidate.model.input_dim,
                "model_dim": candidate.model.model_dim,
                "num_heads": candidate.model.num_heads,
                "num_layers": candidate.model.num_layers,
            }
        elif candidate.family == "nn":
            config["model_kwargs"] = {
                "seq_len": candidate.model.seq_len,
                "input_dim": candidate.model.input_dim,
            }
        else:
            raise ValueError(f"Unsupported candidate family: {candidate.family}")
        torch.save({"state_dict": candidate.model.state_dict()}, output_dir / "model.pt")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


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

    if candidate.family in {"rf", "xgb", "mlp_sklearn"}:
        candidate.model = joblib.load(package_dir / "model.joblib")
        return candidate

    candidate.scaler = joblib.load(package_dir / "scaler.joblib")
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


def get_current_champion(
    client: MlflowClient,
    registered_model_name: str,
) -> tuple[TournamentCandidate | None, dict[str, str] | None]:
    try:
        version = client.get_model_version_by_alias(registered_model_name, "champion")
    except Exception:
        return None, None

    local_dir = mlflow.artifacts.download_artifacts(
        model_uri=f"models:/{registered_model_name}@champion"
    )
    candidate = load_candidate_package(Path(local_dir) / "artifacts" / "model_dir")
    metadata = {"version": version.version, "run_id": version.run_id}
    return candidate, metadata


def evaluate_champion(
    champion: TournamentCandidate,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    future_row: pd.DataFrame,
) -> dict[str, Any]:
    eval_frame = pd.concat([train_df, valid_df], ignore_index=True)
    prediction_frame = pd.concat([eval_frame, future_row], ignore_index=True)
    val_start = len(train_df)
    valid_labels = valid_df["target"].to_numpy(dtype=np.int32)

    probs = predict_candidate_probabilities(champion, eval_frame)
    valid_probs = probs[val_start:]
    if np.isnan(valid_probs).any():
        raise RuntimeError("Champion produced incomplete validation output.")

    metrics = evaluate_probabilities(valid_probs, valid_labels)
    next_probability = float(predict_candidate_probabilities(champion, prediction_frame)[-1])
    return {
        "name": champion.name,
        "source": "champion",
        "candidate": champion,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "next_probability": next_probability,
        "next_signal": prediction_to_signal(next_probability),
    }


def log_challenger_runs(
    challenger_results: list[dict[str, Any]],
    validation_start: str,
    validation_end: str,
) -> None:
    for result in challenger_results:
        with mlflow.start_run(run_name=f"challenger-{result['name']}", nested=True):
            mlflow.set_tags(
                {
                    "role": "challenger",
                    "model_name": result["name"],
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": result["accuracy"],
                    "f1": result["f1"],
                    "next_probability": result["next_probability"],
                }
            )


def promote_champion(
    client: MlflowClient,
    registered_model_name: str,
    winner: dict[str, Any],
    validation_start: str,
    validation_end: str,
) -> str:
    candidate: TournamentCandidate = winner["candidate"]
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / ARTIFACT_SUBDIR
        save_candidate_package(candidate, package_dir)

        with mlflow.start_run(run_name=f"promote-{candidate.name}") as run:
            mlflow.set_tags(
                {
                    "role": "champion_candidate",
                    "model_name": candidate.name,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": winner["accuracy"],
                    "f1": winner["f1"],
                    "next_probability": winner["next_probability"],
                }
            )
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=TournamentPyFuncModel(),
                artifacts={"model_dir": str(package_dir)},
                registered_model_name=registered_model_name,
            )

            versions = client.search_model_versions(f"run_id = '{run.info.run_id}'")
            if not versions:
                raise RuntimeError("Registered model version was not found for the promoted run.")

            version = sorted(versions, key=lambda item: int(item.version))[-1]
            client.set_model_version_tag(
                registered_model_name,
                version.version,
                "champion",
                "true",
            )
            client.set_model_version_tag(
                registered_model_name,
                version.version,
                "tournament_model_name",
                candidate.name,
            )
            client.set_registered_model_alias(
                registered_model_name,
                "champion",
                version.version,
            )
            return version.version


def print_scoreboard(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=ranking_key)
    print("Directional tournament scoreboard:")
    for index, result in enumerate(ordered, start=1):
        print(
            f"{index}. {result['name']} [{result['source']}] "
            f"F1={result['f1']:.3f} "
            f"Accuracy={result['accuracy']:.3f} "
            f"Next={result['next_probability']:.3f} {result['next_signal']}"
        )


def main() -> None:
    set_seed()
    registered_model_name = configure_tracking()
    client = MlflowClient()

    raw = fetch_ohlcv(limit=LOOKBACK_HOURS)
    featured = add_features(raw)
    train_df, valid_df, future_row = split_dataset(featured, VALIDATION_HOURS)

    validation_start = valid_df["timestamp"].iloc[0].isoformat()
    validation_end = valid_df["timestamp"].iloc[-1].isoformat()

    challengers = train_challengers(train_df, valid_df)
    challenger_results = build_results(challengers, train_df, valid_df, future_row)

    champion_candidate, champion_meta = get_current_champion(client, registered_model_name)
    all_results = list(challenger_results)
    champion_result: dict[str, Any] | None = None
    if champion_candidate is not None and champion_meta is not None:
        champion_result = evaluate_champion(
            champion_candidate,
            train_df,
            valid_df,
            future_row,
        )
        champion_result["name"] = f"{champion_result['name']} (champion)"
        champion_result["registry_version"] = champion_meta["version"]
        all_results.append(champion_result)

    with mlflow.start_run(run_name="btc-directional-tournament"):
        mlflow.set_tags(
            {
                "asset": SYMBOL,
                "timeframe": TIMEFRAME,
                "validation_hours": str(VALIDATION_HOURS),
            }
        )
        mlflow.log_params(
            {
                "lookback_hours": LOOKBACK_HOURS,
                "validation_hours": VALIDATION_HOURS,
                "sequence_length": SEQUENCE_LENGTH,
            }
        )
        log_challenger_runs(challenger_results, validation_start, validation_end)
        mlflow.log_text(
            json.dumps(
                [
                    {
                        "name": row["name"],
                        "source": row["source"],
                        "accuracy": row["accuracy"],
                        "f1": row["f1"],
                        "next_probability": row["next_probability"],
                        "next_signal": row["next_signal"],
                    }
                    for row in sorted(all_results, key=ranking_key)
                ],
                indent=2,
            ),
            "tournament_results.json",
        )

    print_scoreboard(all_results)

    best_challenger = sorted(challenger_results, key=ranking_key)[0]
    null_model_block = (
        best_challenger["f1"] <= 0.5 or best_challenger["accuracy"] <= 0.5
    )

    if champion_result is None:
        should_promote = not null_model_block
        active_result = best_challenger
    else:
        should_promote = (
            best_challenger["f1"] > champion_result["f1"] and not null_model_block
        )
        active_result = best_challenger if should_promote else champion_result

    if null_model_block:
        print(
            f"Promotion blocked by null-model guard: "
            f"{best_challenger['name']} scored F1={best_challenger['f1']:.3f}, "
            f"Accuracy={best_challenger['accuracy']:.3f}"
        )

    if should_promote:
        new_version = promote_champion(
            client=client,
            registered_model_name=registered_model_name,
            winner=best_challenger,
            validation_start=validation_start,
            validation_end=validation_end,
        )
        print(
            f"Winner this hour: {best_challenger['name']} -> promoted to champion version {new_version}"
        )
        active_result = best_challenger
    elif champion_result is not None:
        print(
            f"Winner this hour: {champion_result['name']} -> champion retained"
        )
    else:
        print(
            f"Winner this hour: {best_challenger['name']} -> not promoted because it failed the null-model check"
        )

    print(
        f"Upcoming hour probability: {active_result['next_probability']:.1%} chance of UP"
    )
    print(f"Final signal: {active_result['next_signal']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise

#!/usr/bin/env python3
"""
predict_next_hour.py
====================
Fetches latest BTC/USDT data, loads saved models, generates predictions,
and saves results to predictions.json for the HTML dashboard.

ALL MODELS NOW USE SEQUENCE DATA: (seq_len=10, n_features=5)
Models loaded with CLEAN NAMES (no timestamps): RF.joblib, LSTM.pt, etc.
Scaler: deploy_scaler.joblib

All times use America/New_York (ET) timezone.

Usage:
    python predict_next_hour.py [--models RF XGB NN LSTM Transformer Ensemble]
                                [--output predictions.json]
                                [--seq-len 10]
                                [--model-dir saved_models]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# =============================================================================
# TIMEZONE UTILITIES
# =============================================================================

def get_et_now():
    """Get current time in America/New_York (ET) timezone"""
    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et_tz = pytz.timezone("America/New_York")
    return datetime.now(et_tz)


def convert_to_et(utc_timestamp):
    """Convert UTC timestamp to ET timezone"""
    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        et_tz = pytz.timezone("America/New_York")
    
    if isinstance(utc_timestamp, (int, float)):
        utc_timestamp = datetime.fromtimestamp(utc_timestamp / 1000, tz=timezone.utc)
    elif isinstance(utc_timestamp, pd.Timestamp):
        if utc_timestamp.tz is None:
            utc_timestamp = utc_timestamp.tz_localize('UTC')
        else:
            utc_timestamp = utc_timestamp.tz_convert('UTC')
    return utc_timestamp.astimezone(et_tz)


def format_hour_12(hour):
    """Format hour (0-23) to 12-hour format with am/pm"""
    hour = int(hour) % 24
    if hour == 0:
        return "12am"
    elif hour < 12:
        return f"{hour}am"
    elif hour == 12:
        return "12pm"
    else:
        return f"{hour - 12}pm"


def get_et_hour_range(et_datetime):
    """Get current and next hour in ET 12-hour format"""
    current_hour = et_datetime.hour
    next_hour = (current_hour + 1) % 24
    return f"{format_hour_12(current_hour)}-{format_hour_12(next_hour)}"


def add_one_hour_et(et_datetime):
    """Add one elapsed hour and keep the ET timezone representation DST-safe."""
    utc_dt = et_datetime.astimezone(timezone.utc) + timedelta(hours=1)
    return convert_to_et(utc_dt)


def format_et_interval(start_et, end_et):
    """Format an ET interval like 'Mar 8, 1-3am'."""
    start_prefix = start_et.strftime('%b %-d') if os.name != 'nt' else start_et.strftime('%b %#d')
    start_label = format_hour_12(start_et.hour)
    end_label = format_hour_12(end_et.hour)
    if start_label[-2:] == end_label[-2:]:
        range_label = f"{start_label[:-2]}-{end_label}"
    else:
        range_label = f"{start_label}-{end_label}"
    return f"{start_prefix}, {range_label}"


def load_last_test_end(test_values_path='test_values.csv'):
    """Read the end timestamp of the last interval covered by the test split."""
    csv_path = Path(test_values_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Test values file not found: {csv_path.resolve()}")

    test_df = pd.read_csv(csv_path)
    if test_df.empty:
        raise ValueError(f"Test values file is empty: {csv_path.resolve()}")

    if 'date_end' in test_df.columns and pd.notna(test_df['date_end'].iloc[-1]):
        return pd.to_datetime(test_df['date_end'].iloc[-1])

    if 'date' not in test_df.columns:
        raise ValueError(f"Test values file must contain 'date' or 'date_end': {csv_path.resolve()}")

    last_start = pd.to_datetime(test_df['date'].iloc[-1])
    if last_start.tzinfo is None:
        last_start = last_start.tz_localize('America/New_York')
    return add_one_hour_et(last_start.to_pydatetime())


# =============================================================================
# MODEL CLASSES (Must match training code exactly)
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = self.norm(out[:, -1, :])
        logits = self.head(last)
        return torch.sigmoid(logits)


class TransformerModel(nn.Module):
    def __init__(self, in_dim=5):
        super().__init__()
        model_dim = 64
        n_heads = 8
        max_seq_len = 512

        self.input_proj = nn.Linear(in_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        seq_len = x.size(1)
        x_proj = self.input_proj(x)
        x_proj = x_proj + self.pos_embed[:, :seq_len, :]
        encoded = self.encoder(x_proj)
        last = self.norm(encoded[:, -1, :])
        logits = self.head(last)
        return torch.sigmoid(logits)


class NNModel(nn.Module):
    """Feedforward neural network that accepts SEQUENCE data (flattened internally)"""
    def __init__(self, in_dim=5, seq_len=10, hidden_dims=None, dropout=0.3, 
                 lr=0.001, epochs=60, batch_size=32, patience=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.flattened_dim = in_dim * seq_len  # 10 * 5 = 50
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self._built = False
    
    def _build_network(self):
        if self._built:
            return
        layers = []
        prev_dim = self.flattened_dim  # Start with flattened dimension
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._built = True
    
    def forward(self, x):
        if not self._built:
            self._build_network()
        # Handle both 2D (batch, flattened) and 3D (batch, seq_len, features) inputs
        if x.dim() == 3:
            # FIX: Use reshape() instead of view() for non-contiguous tensors
            x = x.reshape(x.size(0), -1)  # Flatten: (batch, seq_len * features)
        logits = self.network(x)
        return torch.sigmoid(logits)
    
    def predict_proba(self, X):
        if not self._built:
            self._build_network()
        # Flatten sequence data if 3D: (n_samples, seq_len, features) -> (n_samples, seq_len*features)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self(x_tensor).squeeze(-1).cpu().numpy()
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

class EnsembleModel:
    """Ensemble model that combines predictions from multiple base models (ALL use sequence data)"""
    def __init__(self, model_factories, weights=None, method='average'):
        self.model_factories = model_factories
        self.weights = weights
        self.method = method
        self.models = []
        self.is_fitted = False
        if weights is not None and len(weights) != len(model_factories):
            raise ValueError("weights length must match number of model factories")
    
    def fit(self, X, y, X_val=None, y_val=None):
        self.models = []
        for name, factory in self.model_factories:
            model = factory()
            if hasattr(model, 'fit'):
                try:
                    model.fit(X, y, X_val=X_val, y_val=y_val)
                except TypeError:
                    model.fit(X, y)
            self.models.append((name, model))
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel must be fitted before prediction")
        all_probs = []
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'eval') and hasattr(model, 'parameters'):
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(X, dtype=torch.float32)
                    probs = model(x_tensor).squeeze(-1).cpu().numpy()
            else:
                preds = model.predict(X)
                probs = preds.astype(float)
            all_probs.append(probs)
        all_probs = np.stack(all_probs, axis=1)
        if self.method == 'voting':
            avg_probs = np.mean(all_probs, axis=1)
            preds = (avg_probs > 0.5).astype(int)
            return np.column_stack([1 - preds, preds])
        elif self.method == 'weighted' and self.weights is not None:
            weights_arr = np.array(self.weights)
            weights_arr = weights_arr / weights_arr.sum()
            avg_probs = np.average(all_probs, axis=1, weights=weights_arr)
        else:
            avg_probs = np.mean(all_probs, axis=1)
        return np.column_stack([1 - avg_probs, avg_probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)


class RandomModel:
    def fit(self, X, y): pass
    def predict(self, X): return np.random.randint(0, 2, size=(X.shape[0],))
    def predict_proba(self, X): return np.random.rand(X.shape[0], 2)


# =============================================================================
# MODEL LOADING UTILITIES (UPDATED FOR CLEAN NAMES)
# =============================================================================

def find_model_file(model_dir, model_prefix, extension):
    """
    Find model file - prefers CLEAN NAME (no timestamp) for deployment models
    """
    model_dir = Path(model_dir)
    
    # First try: clean name without timestamp (deployment model)
    clean_file = model_dir / f"{model_prefix}{extension}"
    if clean_file.exists():
        return clean_file
    
    # Fallback: find any file matching prefix
    candidates = list(model_dir.glob(f"{model_prefix}*{extension}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_pytorch_model(filepath, model_class):
    """Load PyTorch model from saved checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    kwargs = {k: v for k, v in checkpoint.get('init_kwargs', {}).items() if v is not None}
    model = model_class(**kwargs)
    # Build network before loading state dict (critical for NNModel)
    if hasattr(model, '_build_network'):
        model._build_network()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def load_ensemble_model(ensemble_dir):
    """Load ensemble model from directory"""
    config_path = Path(ensemble_dir) / "ensemble_config.json"
    if not config_path.exists():
        raise ValueError(f"Ensemble config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    loaded_models = []
    for name, mtype in zip(config['model_names'], config['model_types']):
        if mtype == "pytorch":
            class_map = {'LSTM': LSTMModel, 'Transformer': TransformerModel, 'NN': NNModel}
            model_cls = class_map.get(name, NNModel)
            model_file = Path(ensemble_dir) / f"{name}.pt"
            sub_model = load_pytorch_model(model_file, model_cls)
        else:
            model_file = Path(ensemble_dir) / f"{name}.joblib"
            sub_model = joblib.load(model_file)
        loaded_models.append((name, sub_model))
    
    ensemble = EnsembleModel([], weights=config['weights'], method=config['method'])
    ensemble.models = loaded_models
    ensemble.is_fitted = True
    return ensemble


def load_model_by_name(model_name, model_dir="saved_models"):
    """
    Load a trained model by name from the saved_models directory.
    ALL MODELS EXPECT SEQUENCE INPUT: (seq_len, n_features)
    
    Returns: (model_instance, scaler_or_None, metadata_dict)
    """
    model_dir = Path(model_dir)
    
    # Handle ensemble separately (clean name: Ensemble_ensemble)
    if "Ensemble" in model_name:
        # Try clean name first, then timestamped
        ensemble_dirs = list(model_dir.glob(f"{model_name}_ensemble"))
        if not ensemble_dirs:
            ensemble_dirs = list(model_dir.glob(f"{model_name}_*_ensemble"))
        if not ensemble_dirs:
            raise FileNotFoundError(f"No ensemble directory found for {model_name}")
        ensemble_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ensemble = load_ensemble_model(ensemble_dirs[0])
        
        # Load scaler (prefer clean name: deploy_scaler)
        scaler = None
        scaler_files = list(model_dir.glob("deploy_scaler.joblib"))
        if not scaler_files:
            scaler_files = list(model_dir.glob("deploy_scaler_*.joblib"))
        if scaler_files:
            scaler_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            scaler = joblib.load(scaler_files[0])
        return ensemble, scaler, {}
    
    # Handle PyTorch models (NN, LSTM, Transformer)
    if model_name in ['LSTM', 'Transformer', 'NN']:
        class_map = {'LSTM': LSTMModel, 'Transformer': TransformerModel, 'NN': NNModel}
        model_cls = class_map[model_name]
        model_file = find_model_file(model_dir, model_name, '.pt')
        if not model_file:
            raise FileNotFoundError(f"No .pt file found for {model_name}")
        model = load_pytorch_model(model_file, model_cls)
        
        # Load metadata
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        metadata = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        
        # Load scaler
        scaler = None
        scaler_files = list(model_dir.glob("deploy_scaler.joblib"))
        if not scaler_files:
            scaler_files = list(model_dir.glob("deploy_scaler_*.joblib"))
        if scaler_files:
            scaler_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            scaler = joblib.load(scaler_files[0])
        return model, scaler, metadata
    
    # Handle sklearn models (RF, XGB) - NOW USE SEQUENCE DATA (flattened internally)
    if model_name in ['RF', 'XGB', 'Random']:
        model_file = find_model_file(model_dir, model_name, '.joblib')
        if not model_file:
            raise FileNotFoundError(f"No .joblib file found for {model_name}")
        model = joblib.load(model_file)
        
        # Load metadata
        meta_file = model_file.parent / f"{model_file.stem}_meta.json"
        metadata = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        
        # Load scaler
        scaler = None
        scaler_files = list(model_dir.glob("deploy_scaler.joblib"))
        if not scaler_files:
            scaler_files = list(model_dir.glob("deploy_scaler_*.joblib"))
        if scaler_files:
            scaler_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            scaler = joblib.load(scaler_files[0])
        return model, scaler, metadata
    
    raise ValueError(f"Unknown model type: {model_name}")


# =============================================================================
# DATA FETCHING & PREPROCESSING
# =============================================================================

def fetch_latest_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=20, exclude_current=True):
    """Fetch latest OHLCV data from Binance via ccxt."""
    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if exclude_current and len(df) > 1:
        df = df.iloc[:-1].copy()
    return df


def extract_prediction_sequence(df, features, seq_len):
    """Extract the raw feature window used for one sequence prediction."""
    if len(df) < seq_len:
        raise ValueError(f"Not enough data: got {len(df)} rows, need {seq_len}")
    return df[features].values[-seq_len:].astype(float)


def prepare_sequence_for_model(sequence_raw, scaler=None):
    """Apply the deployment scaler when available, otherwise z-score locally."""
    if scaler is not None:
        return scaler.transform(sequence_raw)
    means = sequence_raw.mean(axis=0)
    stds = sequence_raw.std(axis=0) + 1e-8
    return (sequence_raw - means) / stds


def build_prediction_targets(df, seq_len, last_test_end_et, max_intervals=10):
    """Build recent prediction intervals after the test split, including the next upcoming hour."""
    if len(df) < seq_len:
        raise ValueError(f"Not enough data: got {len(df)} rows, need {seq_len}")

    working_df = df.copy()
    working_df['timestamp_et'] = working_df['timestamp'].apply(convert_to_et)
    targets = []

    for idx in range(seq_len, len(working_df)):
        start_et = working_df.iloc[idx]['timestamp_et']
        if start_et < last_test_end_et:
            continue
        if idx + 1 < len(working_df):
            end_et = working_df.iloc[idx + 1]['timestamp_et']
        else:
            end_et = add_one_hour_et(start_et)
        target_row = working_df.iloc[idx]
        actual_direction = 'up' if float(target_row['close']) > float(target_row['open']) else 'down'
        targets.append({
            'source': 'historical',
            'history_df': working_df.iloc[idx - seq_len:idx].copy(),
            'interval_start_et': start_et,
            'interval_end_et': end_et,
            'actual_direction': actual_direction,
            'actual_open': float(target_row['open']),
            'actual_close': float(target_row['close']),
        })

    upcoming_start_et = add_one_hour_et(working_df.iloc[-1]['timestamp_et'])
    if upcoming_start_et >= last_test_end_et:
        targets.append({
            'source': 'future',
            'history_df': working_df.iloc[-seq_len:].copy(),
            'interval_start_et': upcoming_start_et,
            'interval_end_et': add_one_hour_et(upcoming_start_et),
            'actual_direction': None,
            'actual_open': None,
            'actual_close': None,
        })

    if not targets:
        targets.append({
            'source': 'future',
            'history_df': working_df.iloc[-seq_len:].copy(),
            'interval_start_et': upcoming_start_et,
            'interval_end_et': add_one_hour_et(upcoming_start_et),
            'actual_direction': None,
            'actual_open': None,
            'actual_close': None,
        })

    targets.sort(key=lambda item: item['interval_start_et'])
    return targets[-max_intervals:]


def predict_model_universal(model, X_seq, model_name):
    """
    UNIVERSAL PREDICTION FUNCTION - ALL MODELS RECEIVE SEQUENCE DATA
    
    Args:
        model: Loaded model instance
        X_seq: Sequence data of shape (seq_len, n_features)
        model_name: Name of model for routing
    
    Returns:
        float: Probability of UP (close > open)
    """
    # Add batch dimension for prediction: (1, seq_len, n_features)
    X_batch = np.expand_dims(X_seq, axis=0)
    
    if hasattr(model, 'eval') and hasattr(model, 'parameters'):
        # PyTorch models (NN, LSTM, Transformer)
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_batch, dtype=torch.float32)
            prob = model(x_tensor).squeeze().item()
        return float(prob)
    
    elif hasattr(model, 'predict_proba'):
        # Sklearn models (RF, XGB) - FLATTEN sequence to 1D
        # Shape: (1, seq_len, n_features) -> (1, seq_len * n_features)
        X_flat = X_batch.reshape(1, -1)
        probs = model.predict_proba(X_flat)
        return float(probs[0, 1])  # Return probability of class "1" (UP)
    
    else:
        # Random model fallback
        return 0.5


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def generate_predictions(
    model_names=None,
    features=None,
    seq_len=10,
    model_dir="saved_models",
    output_file="predictions.json",
    test_values_file="test_values.csv",
    max_intervals=10
):
    if model_names is None:
        model_names = ['RF', 'XGB', 'NN', 'LSTM', 'Transformer', 'Ensemble']
    if features is None:
        features = ["open", "high", "low", "close", "volume"]

    print(f"Generating predictions for models: {model_names}")
    print(f"  Model directory: {Path(model_dir).resolve()}")
    print(f"  Input shape for ALL models: ({seq_len}, {len(features)})")

    feature_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df_features = [feature_map.get(f, f) for f in features]

    print("  Fetching latest BTC/USDT data from Binance...")
    df = fetch_latest_ohlcv(limit=max(seq_len + max_intervals + 5, seq_len + 12), exclude_current=True)
    if len(df) < seq_len:
        raise RuntimeError(f"Insufficient data: got {len(df)}, need {seq_len}")

    last_candle = df.iloc[-1]
    last_time_utc = last_candle["timestamp"]
    last_time_et = convert_to_et(last_time_utc)
    current_et = get_et_now()
    last_test_end_et = load_last_test_end(test_values_file)
    targets = build_prediction_targets(df, seq_len, last_test_end_et, max_intervals=max_intervals)

    print(f"  Last test interval ended: {last_test_end_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Last closed candle ET: {last_time_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Current ET time: {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Building {len(targets)} prediction interval(s) beyond the test range")

    loaded_models = {}
    prediction_intervals = []
    errors = []

    for target in targets:
        interval_predictions = []
        sequence_raw = extract_prediction_sequence(target['history_df'], df_features, seq_len)

        for model_name in model_names:
            try:
                if model_name not in loaded_models:
                    loaded_models[model_name] = load_model_by_name(model_name, model_dir)
                model, scaler, metadata = loaded_models[model_name]
                X_scaled = prepare_sequence_for_model(sequence_raw, scaler=scaler)
                prob = predict_model_universal(model, X_scaled, model_name)
                direction = "up" if prob > 0.5 else "down"
                confidence = abs(prob - 0.5) * 2
                actual_direction = target.get('actual_direction')
                interval_predictions.append({
                    "model": model_name,
                    "probability": round(prob, 4),
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    "input_shape": list(X_scaled.shape),
                    "actual_direction": actual_direction,
                    "is_correct": None if actual_direction is None else direction == actual_direction
                })
            except Exception as e:
                error_msg = f"{model_name} @ {target['interval_start_et'].isoformat()}: {str(e)}"
                errors.append(error_msg)
                print(f"    Error: {error_msg}")

        prediction_intervals.append({
            "interval_start_et": target['interval_start_et'].isoformat(),
            "interval_end_et": target['interval_end_et'].isoformat(),
            "interval_label": format_et_interval(target['interval_start_et'], target['interval_end_et']),
            "source": target['source'],
            "actual_direction": target.get('actual_direction'),
            "actual_open": target.get('actual_open'),
            "actual_close": target.get('actual_close'),
            "predictions": interval_predictions,
        })

    latest_interval = prediction_intervals[-1] if prediction_intervals else None
    latest_predictions = latest_interval['predictions'] if latest_interval else []

    result = {
        "generated_at": datetime.now().isoformat(),
        "generated_at_et": current_et.isoformat(),
        "prediction_for_hour": latest_interval['interval_label'] if latest_interval else get_et_hour_range(current_et),
        "last_candle_et": last_time_et.isoformat(),
        "last_candle_utc": last_time_utc.isoformat(),
        "current_price": float(last_candle["close"]),
        "seq_len": seq_len,
        "features": features,
        "predictions": latest_predictions,
        "prediction_intervals": prediction_intervals,
        "errors": errors if errors else None,
        "metadata": {
            "total_candles_used": len(df),
            "model_dir": str(Path(model_dir).resolve()),
            "timezone": "America/New_York (ET)",
            "utc_offset": current_et.strftime('%z'),
            "input_type": "sequence",
            "input_shape": [seq_len, len(df_features)],
            "test_values_file": str(Path(test_values_file).resolve()),
            "last_test_end_et": last_test_end_et.isoformat(),
            "prediction_interval_count": len(prediction_intervals),
            "max_intervals": max_intervals
        }
    }

    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nPredictions saved to: {output_path.resolve()}")
    print(f"Summary: {sum(len(item['predictions']) for item in prediction_intervals)} successful rows across {len(prediction_intervals)} interval(s)")
    if latest_interval:
        print(f"Latest interval: {latest_interval['interval_label']}")

    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate BTC price direction predictions (ALL models use sequence data)")
    parser.add_argument("--models", nargs="+", default=['RF', 'XGB', 'NN', 'LSTM', 'Transformer', 'Ensemble'])
    parser.add_argument("--features", nargs="+", default=["o", "h", "l", "c", "v"])
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default="saved_models")
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--test-values-file", type=str, default="test_values.csv")
    parser.add_argument("--max-intervals", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        result = generate_predictions(
            model_names=args.models,
            features=args.features,
            seq_len=args.seq_len,
            model_dir=args.model_dir,
            output_file=args.output,
            test_values_file=args.test_values_file,
            max_intervals=args.max_intervals
        )
        
        print("\n" + "="*70)
        print(f"PREDICTION SUMMARY through {result['prediction_for_hour']} ET")
        print(f"   Generated: {result['generated_at_et'][:19]} ET")
        print(f"   Input: sequence of {result['seq_len']} hours x {len(result['features'])} features")
        print(f"   Intervals: {len(result.get('prediction_intervals', []))}")
        print("="*70)
        for interval in result.get('prediction_intervals', []):
            print(f"[{interval['interval_label']}] {interval['source']}")
            for pred in interval.get('predictions', []):
                symbol = 'UP' if pred['direction'] == 'up' else 'DOWN'
                print(f"  {pred['model']:<18} {symbol} {pred['direction']:<4} {pred['probability']*100:>6.1f}%  conf {pred['confidence']*100:>5.1f}%")
            print("-"*70)

        if result['errors']:
            print("\n⚠️  Errors:")
            for err in result['errors']:
                print(f"  • {err}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
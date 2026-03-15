#!/usr/bin/env python3
"""
btc_gains_plot.py
=================
Train BTC prediction models and evaluate performance.
ALL MODELS NOW USE SEQUENCE DATA: (seq_len=10, n_features=5)
Models saved with CLEAN NAMES for deployment: RF.joblib, LSTM.pt, etc.
Scaler: deploy_scaler.joblib
"""

import json
import os
import random
import joblib
import shutil
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# =============================================================================
# MODEL CLASSES
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
        prev_dim = self.flattened_dim
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
            x = x.reshape(x.size(0), -1)
        logits = self.network(x)
        return torch.sigmoid(logits)
    
    def fit(self, X, y, X_val=None, y_val=None):
        from torch.utils.data import TensorDataset, DataLoader
        if not self._built:
            self._build_network()
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        train_dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = self(batch_x).squeeze(-1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                    val_out = self(x_val_tensor).squeeze(-1)
                    val_loss = criterion(val_out, y_val_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
        
        if best_state is not None:
            self.load_state_dict(best_state)
        return self
    
    def predict_proba(self, X):
        if not self._built:
            self._build_network()
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
# MODEL PERSISTENCE UTILITIES (CLEAN NAMES FOR DEPLOYMENT)
# =============================================================================

def ensure_model_dir(model_dir="saved_models"):
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_pytorch_model(model, filepath):
    torch.save({
        'state_dict': model.state_dict(),
        'class_name': model.__class__.__name__,
        'init_kwargs': {
            'in_dim': getattr(model, 'in_dim', None),
            'seq_len': getattr(model, 'seq_len', None),
            'hidden_dims': getattr(model, 'hidden_dims', None),
            'dropout': getattr(model, 'dropout', None),
            'lr': getattr(model, 'lr', None),
            'epochs': getattr(model, 'epochs', None),
            'batch_size': getattr(model, 'batch_size', None),
            'patience': getattr(model, 'patience', None),
        }
    }, filepath)


def load_pytorch_model(filepath, model_class):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    kwargs = {k: v for k, v in checkpoint.get('init_kwargs', {}).items() if v is not None}
    model = model_class(**kwargs)
    if hasattr(model, '_build_network'):
        model._build_network()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def save_sklearn_model(model, filepath):
    joblib.dump(model, filepath)


def load_sklearn_model(filepath):
    return joblib.load(filepath)


def save_model(model, model_name, model_dir="saved_models", metadata=None, use_timestamp=False):
    """Save model - use_timestamp=False for deployment (clean names)"""
    model_dir = ensure_model_dir(model_dir)
    
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if isinstance(model, nn.Module):
        filepath = os.path.join(model_dir, f"{model_name}.pt") if not use_timestamp else os.path.join(model_dir, f"{model_name}_{timestamp}.pt")
        save_pytorch_model(model, filepath)
    elif isinstance(model, EnsembleModel):
        filepath = os.path.join(model_dir, f"{model_name}_ensemble") if not use_timestamp else os.path.join(model_dir, f"{model_name}_{timestamp}_ensemble")
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        os.makedirs(filepath, exist_ok=True)
        ensemble_info = {
            'method': model.method, 'weights': model.weights,
            'model_names': [name for name, _ in model.models], 'model_types': []
        }
        for name, sub_model in model.models:
            if isinstance(sub_model, nn.Module):
                save_pytorch_model(sub_model, os.path.join(filepath, f"{name}.pt"))
                ensemble_info['model_types'].append("pytorch")
            else:
                save_sklearn_model(sub_model, os.path.join(filepath, f"{name}.joblib"))
                ensemble_info['model_types'].append("sklearn")
        with open(os.path.join(filepath, "ensemble_config.json"), 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        print(f"✓ Saved {model_name} → {filepath}")
        return filepath
    else:
        filepath = os.path.join(model_dir, f"{model_name}.joblib") if not use_timestamp else os.path.join(model_dir, f"{model_name}_{timestamp}.joblib")
        save_sklearn_model(model, filepath)
    
    if metadata:
        meta_path = filepath.replace('.pt', '_meta.json').replace('.joblib', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Saved {model_name} → {filepath}")
    return filepath


def save_scaler(scaler, scaler_name, model_dir="saved_models"):
    model_dir = ensure_model_dir(model_dir)
    filepath = os.path.join(model_dir, f"{scaler_name}.joblib")
    joblib.dump(scaler, filepath)
    print(f"✓ Saved scaler {scaler_name} → {filepath}")
    return filepath


# =============================================================================
# CORE UTILITIES
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(timeframe, limit=5000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe, limit=limit, params={"paginate": True, "paginationCalls": 5})
    df = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "c", "v"])
    df["target"] = (df["c"] > df["o"]).astype(int)
    return df.dropna().reset_index(drop=True)


def compute_split_index(n_rows, test_min_size=30):
    split = max(int(n_rows * 0.8), n_rows - test_min_size)
    if split <= 0 or split >= n_rows:
        raise ValueError(f"Invalid split={split} for dataset size={n_rows}")
    return split


def build_train_sequences(x_train, y_train, seq_len):
    if len(x_train) <= seq_len:
        raise ValueError("Training split too small for requested seq_len")
    x_seq = np.array([x_train[i : i + seq_len] for i in range(len(x_train) - seq_len)])
    y_seq = y_train[seq_len:]
    return x_seq, y_seq


def build_test_sequences(x_train, x_test, seq_len):
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    n_test = len(x_test)
    if n_test == 0:
        return np.empty((0, seq_len, x_train.shape[1]))
    overlap_len = seq_len
    if len(x_train) < overlap_len:
        raise ValueError("Not enough train rows to create test sequence overlap")
    history = x_train[-overlap_len:]
    base = np.vstack([history, x_test])
    x_test_seq = np.array([base[i : i + seq_len] for i in range(n_test)])
    return x_test_seq


def predict_sequence_probs(model, x_test_seq):
    if hasattr(model, 'eval') and hasattr(model, 'parameters'):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_test_seq, dtype=torch.float32)
            prob_up = model(x_tensor).squeeze(-1).cpu().numpy()
        return np.asarray(prob_up, dtype=float)
    elif hasattr(model, 'predict_proba'):
        if x_test_seq.ndim == 3:
            x_flat = x_test_seq.reshape(x_test_seq.shape[0], -1)
        else:
            x_flat = x_test_seq
        return model.predict_proba(x_flat)[:, 1]
    else:
        return np.random.rand(x_test_seq.shape[0])


def threshold_predictions(prob_up, prob_threshold):
    return (np.asarray(prob_up) > prob_threshold).astype(int)


def build_directional_strategy_preds(base_preds):
    base_preds = np.asarray(base_preds, dtype=int)
    return {
        "ONLY_UP_WHEN_PREDICTED_UP": np.where(base_preds == 1, 1, -1),
        "ONLY_DOWN_WHEN_PREDICTED_DOWN": np.where(base_preds == 0, 0, -1),
    }


def simulate_gains(preds, y_true, min_gain=0.5, transaction_cost=0.0):
    gains = [1]
    max_gain = min_gain + (1 - min_gain) * 2
    tx_multiplier = max(0.0, 1.0 - transaction_cost)
    for pred, y in zip(preds, y_true):
        if int(pred) == -1:
            gains.append(gains[-1])
            continue
        base_mult = max_gain if int(pred) == int(y) else min_gain
        gains.append(gains[-1] * base_mult * tx_multiplier)
    return gains[1:]


class MonteCarloEvaluator:
    def __init__(self, y_test, min_gain=0.5, n_simulations=10000, seed=42, block_size=24, transaction_cost=0.0):
        self.rng = np.random.default_rng(seed)
        self.y_test = np.asarray(y_test)
        self.min_gain = min_gain
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.transaction_cost = transaction_cost
        self.results = {}

    def evaluate_random_baseline(self, model_gains):
        if "random" not in self.results:
            random_final_gains = []
            for _ in range(self.n_simulations):
                random_preds = self.rng.integers(0, 2, size=len(self.y_test))
                gains = simulate_gains(random_preds, self.y_test, min_gain=self.min_gain, transaction_cost=self.transaction_cost)
                random_final_gains.append(gains[-1] if len(gains) > 0 else 1.0)
            self.results["random"] = np.asarray(random_final_gains, dtype=float)
        random_final_gains = self.results["random"]
        model_final = model_gains[-1] if len(model_gains) > 0 else 1.0
        rank = np.searchsorted(np.sort(random_final_gains), model_final, side="right")
        model_percentile = rank / len(random_final_gains) * 100.0
        return {
            "model_final_gain": float(model_final), "random_median": float(np.median(random_final_gains)),
            "random_mean": float(np.mean(random_final_gains)), "random_std": float(np.std(random_final_gains)),
            "p_value": float(np.mean(random_final_gains >= model_final)), "model_percentile": float(model_percentile),
        }

    def bootstrap_ci(self, preds, confidence=0.95):
        n = len(self.y_test)
        block_size = self.block_size
        n_blocks = (n + block_size - 1) // block_size
        n_boot = max(100, self.n_simulations // 10)
        bootstrap_gains = []
        preds = np.asarray(preds)
        for _ in range(n_boot):
            block_starts = self.rng.choice(max(0, n - block_size), size=n_blocks, replace=True)
            indices = np.concatenate([np.arange(start, min(start + block_size, n)) for start in block_starts])[:n]
            boot_preds = preds[indices]
            boot_y = self.y_test[indices]
            boot_gains = simulate_gains(boot_preds, boot_y, min_gain=self.min_gain, transaction_cost=self.transaction_cost)
            bootstrap_gains.append(boot_gains[-1] if len(boot_gains) > 0 else 1.0)
        lower = np.percentile(bootstrap_gains, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_gains, (1 + confidence) / 2 * 100)
        return {"lower": float(lower), "median": float(np.median(bootstrap_gains)), "upper": float(upper)}

    def transaction_cost_sensitivity(self, preds, min_cost=0.0001, max_cost=0.01, n_points=50):
        costs = np.linspace(min_cost, max_cost, n_points)
        profitable_ratios = []
        preds = np.asarray(preds)
        for cost in costs:
            total_cost = self.transaction_cost + cost
            gains = simulate_gains(preds, self.y_test, min_gain=self.min_gain, transaction_cost=total_cost)
            final_gain = gains[-1] if len(gains) > 0 else 1.0
            profitable_ratios.append(1.0 if final_gain > 1.0 else 0.0)
        profitable_ratios = np.asarray(profitable_ratios, dtype=float)
        break_even_idx = np.where(profitable_ratios < 0.5)[0]
        break_even = costs[break_even_idx[0]] if len(break_even_idx) > 0 else costs[-1]
        return {"costs": costs.tolist(), "ratios": profitable_ratios.tolist(), "break_even": float(break_even)}


def train_sequence_model(model, x_train_seq, y_train_seq, epochs=60, lr=0.001, name="SeqModel", x_val_seq=None, y_val_seq=None, patience=10):
    x_tensor = torch.tensor(x_train_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    if x_val_seq is not None:
        x_val_tensor = torch.tensor(x_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x_tensor).squeeze(-1)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        if x_val_seq is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(x_val_tensor).squeeze(-1)
                val_loss = criterion(val_out, y_val_tensor)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            model.train()
    if best_state is not None:
        model.load_state_dict(best_state)


def temporal_train_val_split(x_train_raw, y_train, val_ratio=0.2, min_val_size=50):
    n = len(x_train_raw)
    val_size = max(int(n * val_ratio), min_val_size)
    val_size = min(val_size, n - 1)
    fit_end = n - val_size
    return x_train_raw[:fit_end], y_train[:fit_end], x_train_raw[fit_end:], y_train[fit_end:]


def binary_accuracy(y_true, preds):
    return float(np.mean(np.asarray(y_true) == np.asarray(preds)))


def run_cv5_sequence_universal(model_name, model_factory, x_train_raw, y_train, seq_len, epochs, lr, threshold=0.5):
    """Cross-validation for ALL models using sequence data"""
    splitter = TimeSeriesSplit(n_splits=5)
    scores = []
    for fold_id, (fit_idx, val_idx) in enumerate(splitter.split(x_train_raw), start=1):
        x_fit_raw, x_val_raw = x_train_raw[fit_idx], x_train_raw[val_idx]
        y_fit, y_val = y_train[fit_idx], y_train[val_idx]
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x_fit_raw)
        x_val = scaler.transform(x_val_raw)
        if len(x_fit) <= seq_len:
            continue
        x_fit_seq, y_fit_seq = build_train_sequences(x_fit, y_fit, seq_len)
        x_val_seq = build_test_sequences(x_fit, x_val, seq_len)
        y_val_seq = y_val[-len(x_val_seq):] if len(y_val) > len(x_val_seq) else y_val  # Align labels
        
        model = model_factory()
        if hasattr(model, 'eval') and hasattr(model, 'parameters'):
            if isinstance(model, NNModel) and hasattr(model, '_build_network'):
                model._build_network()
            train_sequence_model(model, x_fit_seq, y_fit_seq, epochs=epochs, lr=lr, name=f"{model_name}-CV{fold_id}", x_val_seq=x_val_seq, y_val_seq=y_val_seq, patience=10)
            probs = predict_sequence_probs(model, x_val_seq)
        elif hasattr(model, 'predict_proba'):
            x_fit_flat = x_fit_seq.reshape(x_fit_seq.shape[0], -1)
            x_val_flat = x_val_seq.reshape(x_val_seq.shape[0], -1)
            model.fit(x_fit_flat, y_fit_seq)
            probs = model.predict_proba(x_val_flat)[:, 1]
        else:
            probs = np.random.rand(len(y_val_seq))
        
        # Align probs length
        if len(probs) != len(y_val_seq):
            if len(probs) < len(y_val_seq):
                probs = np.concatenate([probs, np.full(len(y_val_seq) - len(probs), 0.5)])
            else:
                probs = probs[:len(y_val_seq)]
        
        preds = threshold_predictions(probs, threshold)
        fold_acc = binary_accuracy(y_val_seq, preds)
        scores.append(fold_acc)
    return scores


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    set_seed(42)
    MODEL_DIR = "saved_models"
    
    # Clear old models for clean deployment save
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    ensure_model_dir(MODEL_DIR)

    df = get_data("1h")
    features = ["o", "h", "l", "c", "v"]
    seq_len = 10
    split = compute_split_index(len(df), test_min_size=40)
    if split <= seq_len:
        raise ValueError("Train split must be larger than seq_len")

    x_raw = df[features].values
    y = df["target"].values
    x_train_raw, x_test_raw = x_raw[:split], x_raw[split:]
    y_train, y_test = y[:split], y[split:]

    # Save test values to CSV
    test_df = df.iloc[split:].copy()
    test_df["date"] = pd.to_datetime(test_df["ts"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    test_df["date_end"] = test_df["date"].shift(-1)
    test_df.loc[test_df.index[-1], "date_end"] = test_df.loc[test_df.index[-1], "date"] + pd.Timedelta(hours=1)
    test_df["label"] = test_df["target"].astype(int)
    test_df["direction"] = test_df["target"].map({1: "up", 0: "down"})
    output_df = test_df[["label", "date", "date_end", "direction", "o", "c"]].copy()
    output_df.columns = ["label", "date", "date_end", "direction", "open", "close"]
    output_df.to_csv("test_values.csv", index=False)

    min_gain_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    prob_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    training_config = {
        "val_ratio": 0.2, "val_min_size": 50, "rf_n_estimators": 1200, "xgb_n_estimators": 1400,
        "xgb_learning_rate": 0.02, "xgb_max_depth": 8, "xgb_subsample": 0.95, "xgb_colsample_bytree": 0.95,
        "sequence_epochs": 180, "sequence_cv_epochs": 45, "sequence_lr": 0.0008, "sequence_patience": 18,
        "nn_dropout": 0.25, "nn_batch_size": 32,
        "transaction_fee": 0.0156, "monte_carlo_simulations": 10000, "monte_carlo_seed": 42, "bootstrap_block_size": 12,
    }
    pos_ratio = np.mean(y_train)
    scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0

    # ===== ALL MODEL FACTORIES (ALL USE SEQUENCE DATA) =====
    all_model_factories = [
        ("RF", lambda: RandomForestClassifier(n_estimators=training_config["rf_n_estimators"], random_state=42, n_jobs=-1, class_weight='balanced')),
        ("XGB", lambda: XGBClassifier(n_estimators=training_config["xgb_n_estimators"], learning_rate=training_config["xgb_learning_rate"], max_depth=training_config["xgb_max_depth"], subsample=training_config["xgb_subsample"], colsample_bytree=training_config["xgb_colsample_bytree"], random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=scale_pos_weight)),
        ("NN", lambda: NNModel(in_dim=len(features), seq_len=seq_len, epochs=training_config["sequence_cv_epochs"], lr=training_config["sequence_lr"], patience=training_config["sequence_patience"], batch_size=training_config["nn_batch_size"], dropout=training_config["nn_dropout"])),
        ("Random", lambda: RandomModel()),
    ]
    
    def create_ensemble():
        models = [
            ("RF", lambda: RandomForestClassifier(n_estimators=training_config["rf_n_estimators"], random_state=42, n_jobs=-1, class_weight='balanced')),
            ("XGB", lambda: XGBClassifier(n_estimators=training_config["xgb_n_estimators"], learning_rate=training_config["xgb_learning_rate"], max_depth=training_config["xgb_max_depth"], subsample=training_config["xgb_subsample"], colsample_bytree=training_config["xgb_colsample_bytree"], random_state=42, n_jobs=-1, eval_metric="logloss", scale_pos_weight=scale_pos_weight)),
            ("NN", lambda: NNModel(in_dim=len(features), seq_len=seq_len, epochs=training_config["sequence_cv_epochs"], lr=training_config["sequence_lr"], patience=training_config["sequence_patience"], batch_size=training_config["nn_batch_size"], dropout=training_config["nn_dropout"])),
        ]
        return EnsembleModel(models, method='average')
    all_model_factories.append(("Ensemble", create_ensemble))
    
    sequence_model_factories = [
        ("LSTM", lambda: LSTMModel(in_dim=len(features))),
        ("Transformer", lambda: TransformerModel(in_dim=len(features))),
    ]

    # ===== CROSS-VALIDATION =====
    print("\n📊 Running 5-fold cross-validation (ALL models use sequence data)...")
    cv_scores = {}
    for name, model_factory in all_model_factories:
        scores = run_cv5_sequence_universal(name, model_factory, x_train_raw, y_train, seq_len=seq_len, epochs=training_config["sequence_cv_epochs"], lr=training_config["sequence_lr"], threshold=0.5)
        if scores:
            cv_scores[name] = scores
            print(f"  ✓ {name} CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    for name, model_factory in sequence_model_factories:
        scores = run_cv5_sequence_universal(name, model_factory, x_train_raw, y_train, seq_len=seq_len, epochs=training_config["sequence_cv_epochs"], lr=training_config["sequence_lr"], threshold=0.5)
        if scores:
            cv_scores[name] = scores
            print(f"  ✓ {name} CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    def get_validation_summary(model_label):
        base_label = model_label
        for suffix in ("_UP", "_DOWN"):
            if base_label.endswith(suffix):
                base_label = base_label[:-len(suffix)]
                break
        scores = cv_scores.get(base_label, [])
        if not scores:
            return {"validation_mean": None, "validation_std": None, "validation_folds": 0}
        return {
            "validation_mean": float(np.mean(scores)),
            "validation_std": float(np.std(scores)),
            "validation_folds": int(len(scores)),
        }

    # ===== TEST EVALUATION =====
    print("\n🎯 Evaluating on test split...")
    full_scaler = StandardScaler()
    x_train = full_scaler.fit_transform(x_train_raw)
    x_test = full_scaler.transform(x_test_raw)
    
    x_train_seq, y_train_seq = build_train_sequences(x_train, y_train, seq_len)
    x_test_seq = build_test_sequences(x_train, x_test, seq_len)
    
    # CRITICAL: Align y_test with x_test_seq predictions
    y_test_seq = y_test[-len(x_test_seq):] if len(y_test) > len(x_test_seq) else y_test
    
    model_probabilities = {}
    for name, model_factory in all_model_factories:
        print(f"  → Evaluating {name}...")
        model = model_factory()
        
        if hasattr(model, 'eval') and hasattr(model, 'parameters'):
            if isinstance(model, NNModel) and hasattr(model, '_build_network'):
                model._build_network()
            train_sequence_model(model, x_train_seq, y_train_seq, epochs=training_config["sequence_epochs"], lr=training_config["sequence_lr"], name=f"{name}-TEST", patience=training_config["sequence_patience"])
            probs = predict_sequence_probs(model, x_test_seq)
        else:
            x_train_flat = x_train_seq.reshape(x_train_seq.shape[0], -1)
            x_test_flat = x_test_seq.reshape(x_test_seq.shape[0], -1)
            model.fit(x_train_flat, y_train_seq)
            probs = model.predict_proba(x_test_flat)[:, 1]
        
        # Ensure probs length matches y_test_seq
        if len(probs) != len(y_test_seq):
            print(f"  ⚠️  {name}: Aligning predictions from {len(probs)} to {len(y_test_seq)}")
            if len(probs) < len(y_test_seq):
                probs = np.concatenate([probs, np.full(len(y_test_seq) - len(probs), 0.5)])
            else:
                probs = probs[:len(y_test_seq)]
        
        model_probabilities[name] = probs

    for name, model_factory in sequence_model_factories:
        print(f"  → Evaluating {name}...")
        model = model_factory()
        train_sequence_model(model, x_train_seq, y_train_seq, epochs=training_config["sequence_epochs"], lr=training_config["sequence_lr"], name=f"{name}-TEST", patience=training_config["sequence_patience"])
        probs = predict_sequence_probs(model, x_test_seq)
        
        if len(probs) != len(y_test_seq):
            if len(probs) < len(y_test_seq):
                probs = np.concatenate([probs, np.full(len(y_test_seq) - len(probs), 0.5)])
            else:
                probs = probs[:len(y_test_seq)]
        
        model_probabilities[name] = probs

    # ===== SAVE DEPLOYMENT MODELS (ONCE - CLEAN NAMES) =====
    print("\n💾 Saving DEPLOYMENT models (all data except last hour)...")
    x_deploy_raw = x_raw[:-1]
    y_deploy = y[:-1]
    
    deploy_scaler = StandardScaler()
    x_deploy = deploy_scaler.fit_transform(x_deploy_raw)
    x_deploy_seq, y_deploy_seq = build_train_sequences(x_deploy, y_deploy, seq_len)
    
    # Save scaler with clean name
    save_scaler(deploy_scaler, "deploy_scaler", MODEL_DIR)
    
    for name, model_factory in all_model_factories:
        if name == "Random":
            print("  -> Skipping deployment save for Random baseline")
            continue
        print(f"  → Training deployment: {name}")
        model = model_factory()
        
        if hasattr(model, 'eval') and hasattr(model, 'parameters'):
            if isinstance(model, NNModel) and hasattr(model, '_build_network'):
                model._build_network()
            train_sequence_model(model, x_deploy_seq, y_deploy_seq, epochs=training_config["sequence_epochs"], lr=training_config["sequence_lr"], name=f"{name}-DEPLOY", patience=training_config["sequence_patience"])
        else:
            x_deploy_flat = x_deploy_seq.reshape(x_deploy_seq.shape[0], -1)
            model.fit(x_deploy_flat, y_deploy_seq)
        
        metadata = {
            "features": features, "seq_len": seq_len, "scaler": "deploy_scaler",
            "train_size": len(x_deploy_seq), "trained_at": datetime.now().isoformat(),
            "config": training_config, "cv_accuracy": float(np.mean(cv_scores.get(name, [0]))),
            "model_type": "deployment", "input_type": "sequence"
        }
        # Save with CLEAN NAME (use_timestamp=False)
        save_model(model, name, MODEL_DIR, metadata=metadata, use_timestamp=False)

    for name, model_factory in sequence_model_factories:
        print(f"  → Training deployment: {name}")
        model = model_factory()
        train_sequence_model(model, x_deploy_seq, y_deploy_seq, epochs=training_config["sequence_epochs"], lr=training_config["sequence_lr"], name=f"{name}-DEPLOY", patience=training_config["sequence_patience"])
        metadata = {
            "features": features, "seq_len": seq_len, "scaler": "deploy_scaler",
            "train_size": len(x_deploy_seq), "trained_at": datetime.now().isoformat(),
            "config": training_config, "cv_accuracy": float(np.mean(cv_scores.get(name, [0]))),
            "model_type": "deployment", "input_type": "sequence"
        }
        save_model(model, name, MODEL_DIR, metadata=metadata, use_timestamp=False)

    # ===== MONTE CARLO EVALUATION =====
    results = []
    for min_gain in min_gain_list:
        mc_evaluator = MonteCarloEvaluator(
            y_test_seq,  # Use aligned labels
            min_gain=min_gain,
            n_simulations=training_config["monte_carlo_simulations"],
            seed=training_config["monte_carlo_seed"],
            block_size=training_config["bootstrap_block_size"],
            transaction_cost=training_config["transaction_fee"]
        )
        for prob_threshold in prob_thresholds:
            for strategy_name, preds in {"ALWAYS_UP": np.ones(len(y_test_seq), dtype=int), "ALWAYS_DOWN": np.zeros(len(y_test_seq), dtype=int)}.items():
                gains = simulate_gains(preds, y_test_seq, min_gain=min_gain, transaction_cost=training_config["transaction_fee"])
                random_stats = mc_evaluator.evaluate_random_baseline(gains)
                bootstrap_ci = mc_evaluator.bootstrap_ci(preds)
                tc_sensitivity = mc_evaluator.transaction_cost_sensitivity(preds)
                validation_summary = get_validation_summary(strategy_name)
                results.append({
                    "model": strategy_name, "min_gain": min_gain, "prob_threshold": prob_threshold, "gains": gains,
                    "monte_carlo": {"p_value": random_stats["p_value"], "random_median": random_stats["random_median"],
                                   "random_mean": random_stats["random_mean"], "random_std": random_stats["random_std"],
                                   "model_percentile": random_stats["model_percentile"], "bootstrap_ci": bootstrap_ci,
                                   "break_even_cost": tc_sensitivity["break_even"], **validation_summary}
                })
            
            for model_name, probs in model_probabilities.items():
                base_preds = threshold_predictions(probs, prob_threshold)
                
                # Final alignment check
                if len(base_preds) != len(y_test_seq):
                    print(f"  ⚠️  {model_name}: Final alignment {len(base_preds)}→{len(y_test_seq)}")
                    if len(base_preds) < len(y_test_seq):
                        padding_val = 1 if prob_threshold < 0.5 else 0
                        base_preds = np.concatenate([base_preds, np.full(len(y_test_seq) - len(base_preds), padding_val)])
                    else:
                        base_preds = base_preds[:len(y_test_seq)]
                
                for strat_name, preds in {
                    model_name: base_preds,
                    f"{model_name}_UP": build_directional_strategy_preds(base_preds)["ONLY_UP_WHEN_PREDICTED_UP"],
                    f"{model_name}_DOWN": build_directional_strategy_preds(base_preds)["ONLY_DOWN_WHEN_PREDICTED_DOWN"]
                }.items():
                    gains = simulate_gains(preds, y_test_seq, min_gain=min_gain, transaction_cost=training_config["transaction_fee"])
                    random_stats = mc_evaluator.evaluate_random_baseline(gains)
                    bootstrap_ci = mc_evaluator.bootstrap_ci(preds)
                    tc_sensitivity = mc_evaluator.transaction_cost_sensitivity(preds)
                    validation_summary = get_validation_summary(strat_name)
                    results.append({
                        "model": strat_name, "min_gain": min_gain, "prob_threshold": prob_threshold, "gains": gains,
                        "monte_carlo": {"p_value": random_stats["p_value"], "random_median": random_stats["random_median"],
                                       "random_mean": random_stats["random_mean"], "random_std": random_stats["random_std"],
                                       "model_percentile": random_stats["model_percentile"], "bootstrap_ci": bootstrap_ci,
                                       "break_even_cost": tc_sensitivity["break_even"], **validation_summary}
                    })

    # ===== SAVE JSON OUTPUT WITH LENGTH ALIGNMENT =====
    traces_for_json = []
    test_length = len(y_test_seq)
    
    for res in results:
        final_value = res["gains"][-1] if res["gains"] else 1.0
        trace_name = f"{res['model']} | min_gain={res['min_gain']} | prob={res['prob_threshold']} | final_value={final_value:.2f}"
        
        # CRITICAL: Ensure gains length matches test_length for Plotly alignment
        gains = [float(g) for g in res["gains"]]
        if len(gains) != test_length:
            if len(gains) < test_length:
                gains.extend([gains[-1] if gains else 1.0] * (test_length - len(gains)))
            else:
                gains = gains[:test_length]
        
        traces_for_json.append({
            "name": trace_name, "y": gains, "model": res["model"],
            "min_gain": res["min_gain"], "prob_threshold": res["prob_threshold"],
            "final_value": float(final_value), "monte_carlo": res["monte_carlo"]
        })
    
    traces_for_json.sort(key=lambda t: t["final_value"], reverse=True)
    
    plot_dates = output_df["date"].astype(str).tolist()[-test_length:]
    plot_date_ends = output_df["date_end"].astype(str).tolist()[-test_length:]
    plot_json = {
        "traces": traces_for_json,
        "dates": plot_dates,
        "date_ends": plot_date_ends,
        "layout": {
            "title": "Interactive Model Gains Simulation",
            "xaxis": {"title": "Date / Hour (ET)", "type": "date"},
            "yaxis": {"title": "Gains ($)"},
            "legend": {"title": {"text": "Model | min_gain | prob"}},
            "template": "plotly_white", "height": 700,
        },
        "test_length": test_length,  # Use aligned length
    }
    
    with open("btc_gains_plot.json", "w", encoding="utf-8") as f:
        json.dump(plot_json, f, indent=2)
    
    print(f"\n✅ Deployment models saved to: {os.path.abspath(MODEL_DIR)}")
    print(f"✅ Results saved to: btc_gains_plot.json")
    print(f"✅ Test values saved to: test_values.csv")
    print(f"📊 ALL models use sequence data (shape: {seq_len} x {len(features)})")
    
    return results


if __name__ == "__main__":
    main()

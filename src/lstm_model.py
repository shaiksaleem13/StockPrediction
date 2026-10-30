"""Deep Learning LSTM Model for Stock Price Prediction in EAATS.

Architecture:
Input(60, n_features) -> LSTM(128, return_sequences=True) -> Dropout(0.2)
-> LSTM(64, return_sequences=False) -> Dropout(0.2)
-> Dense(32, relu) -> Dense(1)
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import config
from src.utils import logger, set_seed


class StreamlitProgressCallback(Callback):
    """Keras callback to update a progress callback during training."""

    def __init__(self, total_epochs: int, update_fn: Optional[Callable[[float, str], None]] = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.update_fn = update_fn

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        progress = (epoch + 1) / self.total_epochs
        msg = f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {loss:.5f} - Val Loss: {val_loss:.5f}"
        if self.update_fn:
            self.update_fn(progress, msg)
        logger.debug(msg)


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Constructs the stacked 2-layer LSTM network according to EAATS specification."""
    set_seed(config.RANDOM_SEED)

    cfg = config.LSTM_CONFIG
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(cfg["units_l1"], return_sequences=cfg["return_sequences_l1"]),
            Dropout(cfg["dropout_l1"]),
            LSTM(cfg["units_l2"], return_sequences=cfg["return_sequences_l2"]),
            Dropout(cfg["dropout_l2"]),
            Dense(cfg["dense_units"], activation=cfg["activation"]),
            Dense(1),  # Linear activation for price regression
        ],
        name="EAATS_Stacked_LSTM",
    )

    optimizer = Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, previous_prices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculates regression & directional accuracy metrics on unscaled prices.

    Directional Accuracy: % of days where predicted move sign matches actual move sign.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))) * 100.0)

    # Directional Accuracy
    if previous_prices is None or len(previous_prices) != len(y_true):
        # Fallback: day-to-day directional shift within the test vector
        actual_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    else:
        actual_dir = np.sign(y_true - previous_prices)
        pred_dir = np.sign(y_pred - previous_prices)

    correct_dir = (actual_dir == pred_dir).sum()
    total_dirs = len(actual_dir)
    dir_acc = float((correct_dir / total_dirs * 100.0) if total_dirs > 0 else 0.0)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "Directional_Accuracy": round(dir_acc, 2),
    }


def get_model_path(ticker: str, model_type: str = "lstm", multivariate: bool = False) -> Path:
    """Generates standardized model checkpoint filename."""
    safe_ticker = ticker.replace(".", "_")
    mode = "multi" if multivariate else "uni"
    return config.MODELS_DIR / f"{safe_ticker}_{model_type}_{mode}.keras"


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ticker: str = "TICKER",
    multivariate: bool = False,
    force_retrain: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Sequential, Dict[str, Any]]:
    """Trains the LSTM model with EarlyStopping or loads cached model from disk."""
    model_path = get_model_path(ticker, "lstm", multivariate)

    if not force_retrain and model_path.exists():
        try:
            logger.info(f"Loading cached LSTM model from {model_path}")
            model = load_model(model_path)
            return model, {"cached": True, "epochs_trained": 0}
        except Exception as e:
            logger.warning(f"Could not load cached LSTM ({e}); retraining...")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_lstm_model((seq_len, n_features))

    cfg = config.LSTM_CONFIG
    callbacks = [
        EarlyStopping(
            monitor=cfg["early_stopping_patience"] and "val_loss",
            patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=0,
        ),
        StreamlitProgressCallback(cfg["epochs"], progress_callback),
    ]

    logger.info(f"Training LSTM model for {ticker} ({'multivariate' if multivariate else 'univariate'})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=cfg["validation_split"],
        callbacks=callbacks,
        verbose=0,
        shuffle=False,  # Preserve sequential order in training batches
    )

    model.save(model_path)
    logger.info(f"Saved trained LSTM model to {model_path}")

    return model, {
        "cached": False,
        "epochs_trained": len(history.history["loss"]),
        "history": history.history,
    }


def predict_next_day(
    model: Sequential,
    recent_sequence_scaled: np.ndarray,
    target_scaler: Any,
) -> float:
    """Predicts next-day close price from the most recent 60 scaled days.

    Args:
        model: Trained Keras model.
        recent_sequence_scaled: Array of shape (60, n_features).
        target_scaler: Fitted MinMaxScaler for the target column.

    Returns:
        Predicted real dollar price (float).
    """
    if recent_sequence_scaled.ndim == 2:
        input_seq = np.expand_dims(recent_sequence_scaled, axis=0)  # Shape (1, 60, n_features)
    else:
        input_seq = recent_sequence_scaled

    pred_scaled = model.predict(input_seq, verbose=0)
    pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    return float(pred_real)


def predict_5day_experimental(
    model: Sequential,
    recent_sequence_scaled: np.ndarray,
    target_scaler: Any,
) -> np.ndarray:
    """Generates a 5-day recursive experimental forecast.

    Explicitly labelled experimental due to compounding prediction errors.
    """
    curr_seq = recent_sequence_scaled.copy()
    predictions = []

    for _ in range(5):
        input_seq = np.expand_dims(curr_seq, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        unscaled = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
        predictions.append(unscaled)

        # Slide window forward: append prediction to target column (index 0)
        next_row = curr_seq[-1].copy()
        next_row[0] = pred_scaled
        curr_seq = np.vstack([curr_seq[1:], next_row])

    return np.array(predictions)

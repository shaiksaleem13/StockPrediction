"""Transformer Encoder Model for Stock Trend & Price Forecasting in EAATS.

Architecture:
Input(60, n_features)
-> Positional Encoding
-> 2 x Transformer Encoder Block (MultiHeadAttention 4-heads, FFN 64, LayerNorm, Dropout 0.1)
-> GlobalAveragePooling1D
-> Dense(32, relu)
-> Dense(1)
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import config
from src.lstm_model import StreamlitProgressCallback, calculate_metrics, get_model_path
from src.utils import logger, set_seed


class PositionalEncoding(Layer):
    """Computes sinusoidal positional encodings to inject sequence order awareness."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        # Sin to even indices, cos to odd indices
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, d_model)
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], : tf.shape(inputs)[2]]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


def transformer_encoder_block(inputs, num_heads: int, key_dim: int, ffn_units: int, dropout_rate: float):
    """Constructs a single Transformer encoder sub-layer with multi-head attention and FFN."""
    # Multi-Head Attention sub-layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-Forward Network sub-layer
    d_model = inputs.shape[-1]
    ffn_output = Dense(ffn_units, activation="relu")(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2


def build_transformer_model(input_shape: Tuple[int, int]) -> Model:
    """Constructs the 2-block Transformer encoder network."""
    set_seed(config.RANDOM_SEED)
    cfg = config.TRANSFORMER_CONFIG
    seq_len, n_features = input_shape

    inputs = Input(shape=input_shape, name="transformer_input")

    # Project features into key_dim space if dimensions differ
    x = Dense(cfg["key_dim"])(inputs) if n_features != cfg["key_dim"] else inputs

    # Positional Encoding
    d_model = x.shape[-1]
    x = PositionalEncoding(seq_len=seq_len, d_model=d_model)(x)

    # 2 Transformer Encoder Blocks
    for _ in range(2):
        x = transformer_encoder_block(
            x,
            num_heads=cfg["num_heads"],
            key_dim=cfg["key_dim"],
            ffn_units=cfg["ffn_units"],
            dropout_rate=cfg["dropout_rate"],
        )

    # Global Average Pooling & Dense Output
    x = GlobalAveragePooling1D()(x)
    x = Dense(cfg["dense_units"], activation=cfg["activation"])(x)
    outputs = Dense(1, name="price_prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EAATS_Transformer_Encoder")
    optimizer = Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_transformer_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ticker: str = "TICKER",
    multivariate: bool = False,
    force_retrain: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Model, Dict[str, Any]]:
    """Trains the Transformer model or retrieves cached checkpoint from disk."""
    model_path = get_model_path(ticker, "transformer", multivariate)

    custom_objects = {"PositionalEncoding": PositionalEncoding}

    if not force_retrain and model_path.exists():
        try:
            logger.info(f"Loading cached Transformer model from {model_path}")
            model = load_model(model_path, custom_objects=custom_objects)
            return model, {"cached": True, "epochs_trained": 0}
        except Exception as e:
            logger.warning(f"Could not load cached Transformer ({e}); retraining...")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    model = build_transformer_model((seq_len, n_features))

    cfg = config.TRANSFORMER_CONFIG
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
            verbose=0,
        ),
        StreamlitProgressCallback(cfg["epochs"], progress_callback),
    ]

    logger.info(f"Training Transformer model for {ticker} ({'multivariate' if multivariate else 'univariate'})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_split=cfg["validation_split"],
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    model.save(model_path)
    logger.info(f"Saved trained Transformer model to {model_path}")

    return model, {
        "cached": False,
        "epochs_trained": len(history.history["loss"]),
        "history": history.history,
    }


def compute_model_comparison_table(
    y_test_unscaled: np.ndarray,
    lstm_preds_unscaled: np.ndarray,
    transformer_preds_unscaled: np.ndarray,
    previous_prices: np.ndarray,
) -> pd.DataFrame:
    """Computes rigorous side-by-side benchmark comparison:

    LSTM vs Transformer vs Naive Baseline (Tomorrow = Today).
    Reports honestly whether the baseline outperforms neural predictions.
    """
    # Naive baseline: Tomorrow's price is predicted to equal Today's price
    naive_preds = previous_prices

    lstm_metrics = calculate_metrics(y_test_unscaled, lstm_preds_unscaled, previous_prices)
    trans_metrics = calculate_metrics(y_test_unscaled, transformer_preds_unscaled, previous_prices)
    naive_metrics = calculate_metrics(y_test_unscaled, naive_preds, previous_prices)

    comparison_data = [
        {
            "Model": "Stacked LSTM (128-64)",
            "RMSE ($)": lstm_metrics["RMSE"],
            "MAE ($)": lstm_metrics["MAE"],
            "MAPE (%)": lstm_metrics["MAPE"],
            "Directional Accuracy (%)": lstm_metrics["Directional_Accuracy"],
        },
        {
            "Model": "Transformer Encoder (4-Head)",
            "RMSE ($)": trans_metrics["RMSE"],
            "MAE ($)": trans_metrics["MAE"],
            "MAPE (%)": trans_metrics["MAPE"],
            "Directional Accuracy (%)": trans_metrics["Directional_Accuracy"],
        },
        {
            "Model": "Naive Baseline (Tomorrow = Today)",
            "RMSE ($)": naive_metrics["RMSE"],
            "MAE ($)": naive_metrics["MAE"],
            "MAPE (%)": naive_metrics["MAPE"],
            "Directional Accuracy (%)": naive_metrics["Directional_Accuracy"],
        },
    ]

    return pd.DataFrame(comparison_data)

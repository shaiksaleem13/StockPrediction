"""Phase 2 Verification Script for EAATS.

Trains Stacked LSTM and Transformer Encoder on AAPL sample data,
evaluates test performance, verifies metrics calculation,
compares with Naive Baseline, tests next-day and 5-day forecasts,
and verifies model persistence.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_fetcher import load_csv
from src.preprocessor import MarketDataPreprocessor
from src.lstm_model import (
    train_lstm_model,
    predict_next_day,
    predict_5day_experimental,
    calculate_metrics,
)
from src.transformer_model import (
    train_transformer_model,
    compute_model_comparison_table,
)
from src.utils import logger


def verify_phase2():
    print("=" * 65)
    print("RUNNING PHASE 2 VERIFICATION FOR EAATS (DEEP LEARNING MODELS)")
    print("=" * 65)

    # 1. Load sample dataset
    sample_path = config.SAMPLE_DIR / "AAPL.csv"
    assert sample_path.exists(), f"Missing {sample_path}"
    df = load_csv(sample_path)
    print(f"[DATA] Loaded {len(df)} rows of AAPL data.")

    # 2. Prepare univariate and multivariate datasets
    preprocessor = MarketDataPreprocessor(multivariate=False, seq_length=60, train_split=0.8)
    data = preprocessor.prepare_dataset(df, ticker="AAPL", save_scaler=True)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    target_scaler = data["target_scaler"]
    test_dates = data["test_dates"]

    print(f"[DATA] Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    # Unscaled true test prices and previous-day prices for directional accuracy
    y_test_unscaled = preprocessor.inverse_transform_target(y_test)
    # The previous day's close for test step i is the last day of X_test[i]
    prev_prices_scaled = X_test[:, -1, 0]
    prev_prices_unscaled = preprocessor.inverse_transform_target(prev_prices_scaled)

    # 3. Train Stacked LSTM
    print("\n--- Training Stacked LSTM (128 -> 64) ---")
    lstm_model, lstm_info = train_lstm_model(
        X_train, y_train, ticker="AAPL", multivariate=False, force_retrain=True
    )
    print(f"[LSTM] Training completed. Epochs trained: {lstm_info['epochs_trained']}")

    # Predict on test set
    lstm_preds_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_preds_unscaled = preprocessor.inverse_transform_target(lstm_preds_scaled)
    lstm_metrics = calculate_metrics(y_test_unscaled, lstm_preds_unscaled, prev_prices_unscaled)
    print(f"[LSTM] Test Metrics: {lstm_metrics}")

    # 4. Train Transformer Encoder
    print("\n--- Training Transformer Encoder (MultiHeadAttention) ---")
    transformer_model, trans_info = train_transformer_model(
        X_train, y_train, ticker="AAPL", multivariate=False, force_retrain=True
    )
    print(f"[TRANSFORMER] Training completed. Epochs trained: {trans_info['epochs_trained']}")

    # Predict on test set
    trans_preds_scaled = transformer_model.predict(X_test, verbose=0).flatten()
    trans_preds_unscaled = preprocessor.inverse_transform_target(trans_preds_scaled)
    trans_metrics = calculate_metrics(y_test_unscaled, trans_preds_unscaled, prev_prices_unscaled)
    print(f"[TRANSFORMER] Test Metrics: {trans_metrics}")

    # 5. Side-by-Side Model Comparison Table (LSTM vs Transformer vs Naive Baseline)
    print("\n--- Model Benchmark Comparison Table ---")
    comparison_df = compute_model_comparison_table(
        y_test_unscaled,
        lstm_preds_unscaled,
        trans_preds_unscaled,
        prev_prices_unscaled,
    )
    print(comparison_df.to_string(index=False))

    # 6. Test Next-Day and 5-Day Experimental Forecasts
    last_60_scaled = X_test[-1]  # Most recent 60 days
    last_actual_close = y_test_unscaled[-1]

    next_day_lstm = predict_next_day(lstm_model, last_60_scaled, target_scaler)
    next_day_pct = ((next_day_lstm - last_actual_close) / last_actual_close) * 100.0
    print(f"\n[FORECAST] Last Actual Close: ${last_actual_close:.2f}")
    print(f"[FORECAST] Next-Day LSTM Predicted Close: ${next_day_lstm:.2f} ({next_day_pct:+.2f}%)")

    five_day_forecast = predict_5day_experimental(lstm_model, last_60_scaled, target_scaler)
    print(f"[FORECAST] 5-Day Experimental Trajectory: {[round(p, 2) for p in five_day_forecast]}")

    # 7. Check saved models on disk
    lstm_file = config.MODELS_DIR / "AAPL_lstm_uni.keras"
    trans_file = config.MODELS_DIR / "AAPL_transformer_uni.keras"
    assert lstm_file.exists(), f"Missing {lstm_file}"
    assert trans_file.exists(), f"Missing {trans_file}"
    print(f"\n[PASS] Model checkpoints successfully persisted to {config.MODELS_DIR}")
    print("=" * 65)
    print("ALL PHASE 2 DEEP LEARNING MODEL VERIFICATIONS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    verify_phase2()

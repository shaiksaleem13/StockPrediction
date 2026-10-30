"""Phase 1 Verification Script for EAATS.

Tests Data Fetcher, CSV Ingestion, Technical Indicators,
Chronological Splitting, Leakage-Free Scaling, and Sequence Generation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_fetcher import fetch_stock_data, load_csv
from src.preprocessor import MarketDataPreprocessor, add_technical_indicators, calculate_rsi_wilder
from src.utils import logger


def verify_phase1():
    print("=" * 60)
    print("RUNNING PHASE 1 VERIFICATION FOR EAATS")
    print("=" * 60)

    # 1. Test CSV Loading & Data Fetcher
    sample_file = config.SAMPLE_DIR / "AAPL.csv"
    assert sample_file.exists(), f"Sample file {sample_file} not found"
    df = load_csv(sample_file)
    print(f"[PASS] Successfully loaded {sample_file.name}: {len(df)} rows.")
    print(f"       Columns: {list(df.columns)}")
    print(f"       Date range: {df.index.min().date()} to {df.index.max().date()}")

    # 2. Test Technical Indicators
    df_feat = add_technical_indicators(df)
    expected_indicators = ["SMA_20", "EMA_12", "EMA_26", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", "Pct_Change"]
    for col in expected_indicators:
        assert col in df_feat.columns, f"Missing indicator {col}"
        assert not df_feat[col].isna().any(), f"NaNs detected in indicator {col}"

    # Verify RSI bounds
    rsi_vals = df_feat["RSI_14"]
    assert (rsi_vals >= 0.0).all() and (rsi_vals <= 100.0).all(), "RSI values out of [0, 100] bounds"
    print(f"[PASS] All technical indicators calculated successfully with zero NaNs.")
    print(f"       RSI min: {rsi_vals.min():.2f}, max: {rsi_vals.max():.2f}, mean: {rsi_vals.mean():.2f}")

    # Verify manual SMA calculation on a test window
    manual_sma_20 = df["Close"].iloc[config.INDICATOR_WARMUP_ROWS : config.INDICATOR_WARMUP_ROWS + 20].mean()
    # The SMA_20 at index config.INDICATOR_WARMUP_ROWS + 19 corresponds to rolling mean of those 20
    computed_sma_20 = df_feat["SMA_20"].iloc[19]
    assert np.isclose(manual_sma_20, computed_sma_20, atol=1e-4), "SMA calculation does not match manual expectation"
    print(f"[PASS] SMA math verified against manual average: {computed_sma_20:.4f}")

    # 3. Test Univariate Preprocessor
    prep_uni = MarketDataPreprocessor(multivariate=False, seq_length=60, train_split=0.8)
    data_uni = prep_uni.prepare_dataset(df, ticker="AAPL_test", save_scaler=True)
    X_train_u, y_train_u = data_uni["X_train"], data_uni["y_train"]
    X_test_u, y_test_u = data_uni["X_test"], data_uni["y_test"]

    assert X_train_u.ndim == 3 and X_train_u.shape[1] == 60 and X_train_u.shape[2] == 1, f"Unexpected shape {X_train_u.shape}"
    assert X_test_u.ndim == 3 and X_test_u.shape[1] == 60 and X_test_u.shape[2] == 1, f"Unexpected shape {X_test_u.shape}"
    assert len(X_train_u) == len(y_train_u)
    assert len(X_test_u) == len(y_test_u)
    print(f"[PASS] Univariate sequences constructed:")
    print(f"       X_train: {X_train_u.shape}, y_train: {y_train_u.shape}")
    print(f"       X_test:  {X_test_u.shape}, y_test:  {y_test_u.shape}")

    # 4. Test Multivariate Preprocessor
    prep_multi = MarketDataPreprocessor(multivariate=True, seq_length=60, train_split=0.8)
    data_multi = prep_multi.prepare_dataset(df, ticker="AAPL_test_multi", save_scaler=True)
    X_train_m, y_train_m = data_multi["X_train"], data_multi["y_train"]
    X_test_m, y_test_m = data_multi["X_test"], data_multi["y_test"]

    n_features = len(prep_multi.feature_columns)
    assert X_train_m.shape[1] == 60 and X_train_m.shape[2] == n_features, f"Unexpected multivariate shape {X_train_m.shape}"
    print(f"[PASS] Multivariate sequences constructed with {n_features} features:")
    print(f"       Features: {prep_multi.feature_columns}")
    print(f"       X_train: {X_train_m.shape}, X_test: {X_test_m.shape}")

    # 5. Verify No Data Leakage: Scaler was fitted strictly on train split
    train_split_idx = data_multi["split_idx"]
    raw_train_target = df_feat["Close"].values[:train_split_idx]
    scaler_min = prep_multi.target_scaler.data_min_[0]
    scaler_max = prep_multi.target_scaler.data_max_[0]
    assert np.isclose(scaler_min, raw_train_target.min()), "Data leakage detected: Scaler min != train min"
    assert np.isclose(scaler_max, raw_train_target.max()), "Data leakage detected: Scaler max != train max"
    print(f"[PASS] Zero data leakage confirmed: Scaler min/max match train split exclusively.")
    print(f"       Train min: {scaler_min:.2f}, max: {scaler_max:.2f}")

    # 6. Verify Inverse Scaling
    sample_scaled = np.array([0.5])
    unscaled = prep_multi.inverse_transform_target(sample_scaled)
    expected_unscaled = scaler_min + 0.5 * (scaler_max - scaler_min)
    assert np.isclose(unscaled[0], expected_unscaled), "Inverse scaling mismatch"
    print(f"[PASS] Inverse transform verified: scaled 0.5 -> ${unscaled[0]:.2f}")

    print("=" * 60)
    print("ALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    verify_phase1()

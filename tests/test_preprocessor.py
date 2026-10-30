"""Unit tests for Preprocessor and Technical Indicators in EAATS."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.preprocessor import (
    MarketDataPreprocessor,
    add_technical_indicators,
    calculate_rsi_wilder,
    create_sequences,
)


@pytest.fixture
def sample_ohlcv():
    """Generates synthetic OHLCV time series data for deterministic testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    # Generate random walk prices starting at 100
    noise = np.random.normal(0, 1, 100)
    close = 100 + np.cumsum(noise)
    close = np.maximum(close, 10.0)  # Ensure positive

    df = pd.DataFrame(
        {
            "Open": close + np.random.uniform(-0.5, 0.5, 100),
            "High": close + np.random.uniform(0.5, 1.5, 100),
            "Low": close - np.random.uniform(0.5, 1.5, 100),
            "Close": close,
            "Volume": np.random.randint(100000, 500000, 100),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_rsi_bounds(sample_ohlcv):
    """Verifies that Wilder's RSI produces values strictly between 0 and 100."""
    rsi = calculate_rsi_wilder(sample_ohlcv["Close"], period=14)
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_sma_manual_calculation(sample_ohlcv):
    """Verifies that SMA(20) calculation matches a manual arithmetic rolling mean."""
    df_feat = add_technical_indicators(sample_ohlcv)
    assert "SMA_20" in df_feat.columns

    # Verify manual 20-day mean for a specific index
    test_idx = 30
    # In the original df, rows test_idx-19 to test_idx
    expected_sma = df_feat["Close"].iloc[test_idx - 19 : test_idx + 1].mean()
    actual_sma = df_feat["SMA_20"].iloc[test_idx]
    assert np.isclose(expected_sma, actual_sma, atol=1e-5)


def test_macd_calculation(sample_ohlcv):
    """Verifies MACD line and signal line consistency."""
    df_feat = add_technical_indicators(sample_ohlcv)
    assert "MACD" in df_feat.columns
    assert "MACD_Signal" in df_feat.columns
    assert "MACD_Hist" in df_feat.columns

    # MACD_Hist must equal MACD - MACD_Signal
    expected_hist = df_feat["MACD"] - df_feat["MACD_Signal"]
    assert np.allclose(df_feat["MACD_Hist"], expected_hist, atol=1e-5)


def test_sequence_generator_shapes():
    """Verifies sliding window sequence tensor dimensions: (N - seq_len, seq_len, features)."""
    n_rows, n_features = 80, 5
    data = np.random.randn(n_rows, n_features)
    seq_len = 20

    X, y = create_sequences(data, target_col_idx=0, seq_length=seq_len)
    expected_samples = n_rows - seq_len

    assert X.shape == (expected_samples, seq_len, n_features)
    assert y.shape == (expected_samples,)
    # Target y[0] should match data[seq_len, 0]
    assert np.isclose(y[0], data[seq_len, 0])


def test_zero_data_leakage_scaling(sample_ohlcv):
    """Strictly verifies that MinMaxScaler is fitted solely on the training split."""
    preprocessor = MarketDataPreprocessor(multivariate=False, seq_length=30, train_split=0.8)
    data_dict = preprocessor.prepare_dataset(sample_ohlcv, ticker="TEST", save_scaler=False)

    df_indicators = data_dict["df_indicators"]
    split_idx = data_dict["split_idx"]
    raw_train_close = df_indicators["Close"].iloc[:split_idx].values

    scaler = data_dict["target_scaler"]
    # Scaler min and max must match training data exclusively
    assert np.isclose(scaler.data_min_[0], raw_train_close.min())
    assert np.isclose(scaler.data_max_[0], raw_train_close.max())

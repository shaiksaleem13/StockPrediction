"""Feature Engineering & Data Preprocessing for EAATS.

Calculates technical indicators (SMA, EMA, Wilder's RSI, MACD),
performs non-leaking chronological train/test splitting,
fits MinMaxScaler on training data only, and constructs sliding window sequences.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from src.utils import logger


def calculate_rsi_wilder(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Calculates Relative Strength Index (RSI) using J. Welles Wilder's smoothing technique."""
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing is equivalent to an exponential moving average with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # If avg_loss is 0, RSI is 100; if avg_gain is 0, RSI is 0
    rsi = rsi.fillna(100.0).where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)

    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA(20), EMA(12, 26), Wilder RSI(14), MACD, and Price % Change to DataFrame."""
    df_feat = df.copy()

    close = df_feat["Close"]

    # 1. Simple Moving Average (20)
    df_feat["SMA_20"] = close.rolling(window=config.SMA_PERIOD).mean()

    # 2. Exponential Moving Averages (12, 26)
    df_feat["EMA_12"] = close.ewm(span=config.EMA_SHORT_PERIOD, adjust=False).mean()
    df_feat["EMA_26"] = close.ewm(span=config.EMA_LONG_PERIOD, adjust=False).mean()

    # 3. Wilder's RSI (14)
    df_feat["RSI_14"] = calculate_rsi_wilder(close, period=config.RSI_PERIOD)

    # 4. MACD & Signal Line
    df_feat["MACD"] = df_feat["EMA_12"] - df_feat["EMA_26"]
    df_feat["MACD_Signal"] = df_feat["MACD"].ewm(span=config.MACD_SIGNAL_PERIOD, adjust=False).mean()
    df_feat["MACD_Hist"] = df_feat["MACD"] - df_feat["MACD_Signal"]

    # 5. Daily Percentage Change
    df_feat["Pct_Change"] = close.pct_change() * 100.0

    # Drop warm-up rows where moving averages and RSI are computing initial values
    initial_len = len(df_feat)
    df_feat = df_feat.iloc[config.INDICATOR_WARMUP_ROWS:].copy()
    df_feat = df_feat.dropna()
    logger.debug(f"Computed indicators. Dropped warmup rows: {initial_len - len(df_feat)} rows removed.")

    return df_feat


def create_sequences(
    data: np.ndarray,
    target_col_idx: int = 0,
    seq_length: int = config.SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sliding window sequences for sequence-to-one forecasting.

    Args:
        data: 2D array of shape (N, features).
        target_col_idx: Index of target column (usually Close price, index 0).
        seq_length: Number of lookback days (default: 60).

    Returns:
        X: 3D array of shape (N - seq_length, seq_length, features).
        y: 1D array of shape (N - seq_length,) containing next-step target values.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class MarketDataPreprocessor:
    """Encapsulates data scaling, sequence generation, and train/test splitting

    guaranteeing ZERO data leakage by fitting scalers strictly on the training set.
    """

    def __init__(
        self,
        multivariate: bool = False,
        seq_length: int = config.SEQUENCE_LENGTH,
        train_split: float = config.TRAIN_SPLIT_RATIO,
    ):
        self.multivariate = multivariate
        self.seq_length = seq_length
        self.train_split = train_split
        self.scaler = MinMaxScaler(feature_range=config.SCALE_RANGE)
        self.target_scaler = MinMaxScaler(feature_range=config.SCALE_RANGE)
        self.feature_columns: List[str] = (
            config.FEATURE_COLS_MULTIVARIATE if multivariate else config.FEATURE_COLS_UNIVARIATE
        )
        self.target_col = "Close"
        self.target_idx = 0

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        ticker: str = "TICKER",
        save_scaler: bool = True,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame, pd.DatetimeIndex, float]]:
        """Processes raw stock DataFrame into non-leaking train and test sequence datasets."""
        # 1. Feature Engineering
        df_indicators = add_technical_indicators(df)

        # Ensure all requested feature columns exist
        available_features = [col for col in self.feature_columns if col in df_indicators.columns]
        if self.target_col not in available_features:
            available_features.insert(0, self.target_col)
        self.feature_columns = available_features
        self.target_idx = self.feature_columns.index(self.target_col)

        feature_data = df_indicators[self.feature_columns].values
        target_data = df_indicators[[self.target_col]].values
        dates = df_indicators.index

        total_rows = len(feature_data)
        if total_rows <= self.seq_length + 10:
            raise ValueError(
                f"Dataset too short for {self.seq_length}-day window: only {total_rows} rows available."
            )

        # 2. Chronological Split (Strictly NO SHUFFLING to prevent future data leakage)
        split_idx = int(total_rows * self.train_split)
        if split_idx <= self.seq_length:
            split_idx = self.seq_length + 5

        train_raw = feature_data[:split_idx]
        test_raw = feature_data[split_idx:]

        # 3. Fit Scalers SOLELY on Training Split
        self.scaler.fit(train_raw)
        self.target_scaler.fit(target_data[:split_idx])

        # 4. Transform Both Splits
        train_scaled = self.scaler.transform(train_raw)
        test_scaled = self.scaler.transform(test_raw)

        # To build sequences for the test split without gap, append the last `seq_length` rows of train
        test_input_scaled = np.vstack([train_scaled[-self.seq_length :], test_scaled])

        # 5. Create Sequences
        X_train, y_train = create_sequences(train_scaled, self.target_idx, self.seq_length)
        X_test, y_test = create_sequences(test_input_scaled, self.target_idx, self.seq_length)

        # Corresponding dates for test predictions
        test_dates = dates[split_idx:]

        # 6. Save Scaler
        if save_scaler:
            safe_ticker = ticker.replace(".", "_")
            scaler_filename = f"{safe_ticker}_{'multi' if self.multivariate else 'uni'}_scaler.joblib"
            scaler_path = config.MODELS_DIR / scaler_filename
            joblib.dump({"scaler": self.scaler, "target_scaler": self.target_scaler}, scaler_path)
            logger.info(f"Saved fitted scaler to {scaler_path}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "df_indicators": df_indicators,
            "split_idx": split_idx,
            "test_dates": test_dates,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "target_scaler": self.target_scaler,
        }

    def inverse_transform_target(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Inverses target scaling back to true dollar prices."""
        reshaped = np.array(scaled_targets).reshape(-1, 1)
        return self.target_scaler.inverse_transform(reshaped).flatten()

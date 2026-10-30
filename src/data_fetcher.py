"""Data Layer for EAATS.

Fetches historical and near-real-time stock market data from yfinance,
with optional Alpha Vantage fallback, local disk caching, offline sample fallbacks,
and CSV file ingestion.
"""

from pathlib import Path
from typing import Optional, Union, BinaryIO
import io
import pandas as pd
import requests

import config
from src.utils import logger, timed_execution


def validate_ticker(ticker: str) -> str:
    """Validates and cleans a stock ticker symbol."""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker symbol cannot be empty.")
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Invalid ticker symbol provided.")
    return cleaned


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns, cleans timestamps, sorts, and removes missing values."""
    if df.empty:
        raise ValueError("Stock data is empty.")

    # Flatten multi-index columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names
    col_mapping = {col: col.capitalize() for col in df.columns}
    df = df.rename(columns=col_mapping)

    # Validate required columns
    missing_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Stock data is missing required columns: {missing_cols}")

    # Clean index (must be DatetimeIndex without timezone)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"

    # Select only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Cast to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop NaNs and sort ascending
    df = df.dropna().sort_index(ascending=True)

    if len(df) < 30:
        raise ValueError(f"Insufficient historical data: received only {len(df)} rows.")

    return df


def get_cache_path(ticker: str, period: str, interval: str) -> Path:
    """Returns the local cache file path for a ticker."""
    safe_ticker = ticker.replace(".", "_").replace("^", "_")
    return config.CACHE_DIR / f"{safe_ticker}_{period}_{interval}.parquet"


@timed_execution
def fetch_from_yfinance(ticker: str, period: str = config.DEFAULT_PERIOD, interval: str = config.DEFAULT_INTERVAL) -> pd.DataFrame:
    """Fetches stock data directly using yfinance."""
    import yfinance as yf

    logger.info(f"Fetching {ticker} from yfinance (period={period}, interval={interval})...")
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
    return clean_dataframe(df)


def fetch_from_alphavantage(ticker: str) -> pd.DataFrame:
    """Fallback fetcher from Alpha Vantage if API key is provided."""
    api_key = config.ALPHAVANTAGE_API_KEY
    if not api_key:
        raise ValueError("Alpha Vantage API key not configured in .env.")

    logger.info(f"Attempting Alpha Vantage fallback for {ticker}...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}&datatype=csv"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Date", "adjusted_close": "Close"})
    return clean_dataframe(df)


def load_sample_fallback(ticker: str) -> Optional[pd.DataFrame]:
    """Loads bundled sample CSV data if live fetching fails."""
    sample_files = {
        "AAPL": config.SAMPLE_DIR / "AAPL.csv",
        "TSLA": config.SAMPLE_DIR / "TSLA.csv",
        "RELIANCE.NS": config.SAMPLE_DIR / "RELIANCE.NS.csv",
    }
    sample_path = sample_files.get(ticker)
    if sample_path and sample_path.exists():
        logger.warning(f"Using bundled sample dataset for {ticker} at {sample_path}")
        return load_csv(sample_path)
    return None


def fetch_stock_data(
    ticker: str = config.DEFAULT_TICKER,
    period: str = config.DEFAULT_PERIOD,
    interval: str = config.DEFAULT_INTERVAL,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Main data acquisition function with caching and multi-tier fallbacks."""
    ticker = validate_ticker(ticker)
    cache_file = get_cache_path(ticker, period, interval)

    # 1. Try disk cache if not forcing refresh
    if not force_refresh and cache_file.exists():
        try:
            cache_age = pd.Timestamp.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < config.CACHE_TTL_SECONDS:
                logger.info(f"Loading {ticker} data from valid disk cache ({cache_file.name})")
                df_cached = pd.read_parquet(cache_file)
                return clean_dataframe(df_cached)
        except Exception as e:
            logger.warning(f"Could not read cache for {ticker}: {e}")

    # 2. Try primary source: yfinance
    try:
        df = fetch_from_yfinance(ticker, period, interval)
        try:
            df.to_parquet(cache_file)
        except Exception as cache_err:
            logger.warning(f"Failed to write cache file: {cache_err}")
        return df
    except Exception as yf_err:
        logger.warning(f"yfinance failed for {ticker}: {yf_err}")

    # 3. Try Alpha Vantage fallback if available
    if config.ALPHAVANTAGE_API_KEY:
        try:
            df = fetch_from_alphavantage(ticker)
            df.to_parquet(cache_file)
            return df
        except Exception as av_err:
            logger.warning(f"Alpha Vantage failed for {ticker}: {av_err}")

    # 4. Check if expired cache exists on disk
    if cache_file.exists():
        try:
            logger.warning(f"Live sources failed. Falling back to existing cached data for {ticker}.")
            return clean_dataframe(pd.read_parquet(cache_file))
        except Exception:
            pass

    # 5. Check bundled offline sample
    sample_df = load_sample_fallback(ticker)
    if sample_df is not None:
        return sample_df

    raise RuntimeError(
        f"Unable to fetch stock data for ticker '{ticker}' from any available source. "
        "Please check your internet connection or upload a historical CSV dataset."
    )


def load_csv(file_or_path: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Ingests, validates, and cleans an uploaded or local CSV file."""
    try:
        if isinstance(file_or_path, (str, Path)):
            df = pd.read_csv(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    return clean_dataframe(df)

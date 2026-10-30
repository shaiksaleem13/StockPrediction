"""Downloads offline historical sample datasets for AAPL, TSLA, and RELIANCE.NS.

Saves clean OHLCV CSVs to data/sample/ covering 5 years (including COVID 2020 & 2022 bear market).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
import config
from src.data_fetcher import clean_dataframe
from src.utils import logger


def download_sample_data():
    tickers = ["AAPL", "TSLA", "RELIANCE.NS"]
    for ticker in tickers:
        logger.info(f"Downloading 5-year sample data for {ticker}...")
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5y", interval="1d", auto_adjust=True)
            df_clean = clean_dataframe(df)

            out_path = config.SAMPLE_DIR / f"{ticker}.csv"
            # Reset index so 'Date' is an explicit column in CSV
            df_clean.reset_index().to_csv(out_path, index=False)
            logger.info(f"Successfully saved {len(df_clean)} rows for {ticker} to {out_path}")
        except Exception as e:
            logger.error(f"Failed downloading sample for {ticker}: {e}")


if __name__ == "__main__":
    download_sample_data()

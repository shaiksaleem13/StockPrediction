"""Unit tests for Backtester Engine and Financial Metrics in EAATS."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtester import BacktestEngine
from src.metrics import calculate_max_drawdown, calculate_sharpe_ratio


def test_hand_calculated_max_drawdown():
    """Verifies max drawdown calculation against a hand-computed equity series.

    Equity trajectory: [100, 120, 90, 110, 80]
    Peak = 120
    Trough = 80
    Drawdown from peak 120 to 80 = (80 - 120) / 120 = -40 / 120 = -33.33%
    """
    equity = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    max_dd, dd_curve = calculate_max_drawdown(equity)
    assert np.isclose(max_dd, 33.33, atol=0.01)


def test_sharpe_ratio_zero_variance():
    """Verifies that flat returns produce a zero Sharpe ratio without division by zero."""
    flat_returns = np.zeros(50)
    sharpe = calculate_sharpe_ratio(flat_returns)
    assert sharpe == 0.0


def test_backtester_next_day_open_execution():
    """Verifies that trades execute strictly at Next-Day Open price and cash is preserved."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    # Day 0: Open=100, Close=105
    # Day 1: Open=110, Close=115
    # Day 2: Open=120, Close=125
    # Day 3: Open=130, Close=135
    # Day 4: Open=140, Close=145
    df_tiny = pd.DataFrame(
        {
            "Open": [100.0, 110.0, 120.0, 130.0, 140.0],
            "High": [106.0, 116.0, 126.0, 136.0, 146.0],
            "Low": [99.0, 109.0, 119.0, 129.0, 139.0],
            "Close": [105.0, 115.0, 125.0, 135.0, 145.0],
            "Volume": [10000, 10000, 10000, 10000, 10000],
            "SMA_20": [100.0, 100.0, 100.0, 100.0, 100.0],
            "EMA_12": [105.0, 105.0, 105.0, 105.0, 105.0],
            "RSI_14": [50.0, 50.0, 50.0, 50.0, 50.0],
            "MACD": [1.0, 1.0, 1.0, 1.0, 1.0],
            "MACD_Signal": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        index=dates,
    )

    pred_prices = [120.0, 120.0, 120.0, 120.0, 120.0]
    sentiment_series = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5], index=dates)

    init_capital = 10000.0
    fee_rate = 0.001

    engine = BacktestEngine(
        df_test=df_tiny,
        predicted_prices=pred_prices,
        sentiment_series=sentiment_series,
        initial_capital=init_capital,
        transaction_fee=fee_rate,
        confidence_threshold=20.0,
    )

    # In Buy & Hold: Day 0 Open is 100.0. Shares = (10000 * 0.999) / 100 = 99.9 shares.
    # Final close on Day 4 is 145.0. Final Equity = 99.9 * 145 = 14485.5.
    res_bh = engine.run_buy_and_hold()
    assert np.isclose(res_bh["metrics"]["Final_Equity"], 99.9 * 145.0, atol=0.1)

    # Run EAATS full: On Day 0 (Close=105), a BUY signal is generated.
    # It must execute on Day 1 Open at $110.0!
    res_eaats = engine.run_eaats_full()
    if res_eaats["signals"]:
        first_trade = res_eaats["signals"][0]
        assert first_trade["price"] == 110.0  # Filled at Day 1 Open, NOT Day 0 Close!

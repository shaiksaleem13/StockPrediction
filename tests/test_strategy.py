"""Unit tests for Trading Strategy Decision Engine in EAATS."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.trading_strategy import (
    calculate_lstm_signal,
    calculate_sentiment_signal,
    calculate_technical_signal,
    evaluate_trading_decision,
)


def test_lstm_signal_thresholds():
    """Tests LSTM price change thresholds (+/- 0.5%)."""
    # Predicted move +1.0% -> +1 (Buy)
    sig_buy, _ = calculate_lstm_signal(predicted_price=101.0, current_price=100.0)
    assert sig_buy == 1

    # Predicted move -1.0% -> -1 (Sell)
    sig_sell, _ = calculate_lstm_signal(predicted_price=99.0, current_price=100.0)
    assert sig_sell == -1

    # Predicted move +0.2% -> 0 (Hold)
    sig_hold, _ = calculate_lstm_signal(predicted_price=100.2, current_price=100.0)
    assert sig_hold == 0


def test_sentiment_signal_thresholds():
    """Tests Sentiment thresholds (> +0.05 / < -0.05)."""
    assert calculate_sentiment_signal(0.20)[0] == 1
    assert calculate_sentiment_signal(-0.20)[0] == -1
    assert calculate_sentiment_signal(0.02)[0] == 0
    assert calculate_sentiment_signal(-0.02)[0] == 0


def test_decision_boundary_thresholds():
    """Verifies decision boundaries (> +0.15 BUY, < -0.15 SELL, else HOLD)."""
    # 1. Neutral technical row where EMA > SMA (+0.5) but Price <= SMA (-0.25) and MACD <= Signal (-0.25) -> Tech = 0.0
    neutral_row = pd.Series(
        {
            "Close": 99.0,
            "SMA_20": 100.0,
            "EMA_12": 101.0,  # +0.50
            "RSI_14": 50.0,   # 0.00
            "MACD": 0.0,
            "MACD_Signal": 0.1,  # -0.25, and Price <= SMA is -0.25 -> sum = 0.0
        }
    )

    # Score just above +0.15: LSTM=0 (move < 0.5%), Sent=+1 (0.3), Tech=0 (0.0) -> Final Score = +0.30 -> BUY
    res_buy = evaluate_trading_decision(
        predicted_price=100.2,  # +0.2% move -> LSTM = 0
        current_price=100.0,
        sentiment_score=0.20,   # Sentiment = +1 -> 0.30 contribution
        latest_market_row=neutral_row,
        confidence_threshold=20.0,
    )
    assert res_buy["final_score"] == 0.30
    assert res_buy["final_score"] > config.DECISION_BOUNDARY
    assert res_buy["raw_decision"] == "BUY"

    # Score just below -0.15: LSTM=0, Sent=-1 (-0.30), Tech=0 -> Final Score = -0.30 -> SELL
    res_sell = evaluate_trading_decision(
        predicted_price=100.2,
        current_price=100.0,
        sentiment_score=-0.20,  # Sentiment = -1 -> -0.30 contribution
        latest_market_row=neutral_row,
        confidence_threshold=20.0,
    )
    assert res_sell["final_score"] == -0.30
    assert res_sell["final_score"] < -config.DECISION_BOUNDARY
    assert res_sell["raw_decision"] == "SELL"


def test_confidence_threshold_suppression():
    """Verifies that signals with confidence below the threshold are downgraded to HOLD."""
    dummy_row = pd.Series(
        {
            "Close": 100.0,
            "SMA_20": 95.0,  # Price > SMA (+0.25)
            "EMA_12": 96.0,  # EMA > SMA (+0.50)
            "RSI_14": 50.0,
            "MACD": 0.5,
            "MACD_Signal": 0.1,  # MACD > Signal (+0.25) -> Tech = +1.0
        }
    )
    # Tech gives +1.0 * 0.3 = +0.30. LSTM=0 (move < 0.5%), Sent=0 -> Final Score = +0.30 (Confidence = 30%)
    res = evaluate_trading_decision(
        predicted_price=100.1,  # 0.1% move -> 0 LSTM
        current_price=100.0,
        sentiment_score=0.0,  # 0 Sentiment
        latest_market_row=dummy_row,
        confidence_threshold=45.0,  # Threshold higher than 30%
    )
    assert res["raw_decision"] == "BUY"
    assert res["confidence"] == 30.0
    assert res["final_decision"] == "HOLD"  # Suppressed!
    assert res["is_suppressed"] is True
    assert "Signal suppressed" in res["suppression_reason"]

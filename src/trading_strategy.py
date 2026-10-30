"""Trading Strategy Decision Engine for EAATS.

Weighted voting ensemble combining:
1. Deep Learning LSTM Signal (weight 0.40)
2. Emotion & Sentiment Signal (weight 0.30)
3. Quantitative Technical Signal (weight 0.30)

Calculates confidence rating and applies low-confidence signal suppression.
All parameters and weights strictly reference config.py.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

import config
from src.utils import logger


def calculate_lstm_signal(predicted_price: float, current_price: float) -> Tuple[int, float]:
    """Generates LSTM directional signal based on predicted % move.

    Threshold: >= +0.5% -> +1 (Buy), <= -0.5% -> -1 (Sell), else 0 (Hold).
    """
    if current_price <= 0:
        return 0, 0.0

    pct_change = ((predicted_price - current_price) / current_price) * 100.0

    if pct_change >= config.LSTM_PRICE_CHANGE_THRESHOLD:
        signal = 1
    elif pct_change <= -config.LSTM_PRICE_CHANGE_THRESHOLD:
        signal = -1
    else:
        signal = 0

    return signal, round(pct_change, 2)


def calculate_sentiment_signal(sentiment_score: float) -> Tuple[int, float]:
    """Generates Sentiment signal from composite score.

    Threshold: > +0.05 -> +1, < -0.05 -> -1, else 0.
    """
    if sentiment_score > config.VADER_POSITIVE_THRESHOLD:
        signal = 1
    elif sentiment_score < config.VADER_NEGATIVE_THRESHOLD:
        signal = -1
    else:
        signal = 0

    return signal, round(sentiment_score, 4)


def calculate_technical_signal(row: pd.Series) -> Tuple[float, Dict[str, Any]]:
    """Evaluates technical indicators and generates composite technical score in [-1.0, 1.0].

    Components:
    - EMA(12) > SMA(20): +0.5 (else -0.5)
    - Price > SMA(20): +0.25 (else -0.25)
    - RSI < 30: +0.5 (oversold)
    - RSI > 70: -0.5 (overbought)
    - MACD > Signal: +0.25 (else -0.25)
    Clamped to [-1.0, 1.0].
    """
    score = 0.0
    details = {}

    close = float(row.get("Close", 0.0))
    sma_20 = float(row.get("SMA_20", close))
    ema_12 = float(row.get("EMA_12", close))
    rsi_14 = float(row.get("RSI_14", 50.0))
    macd = float(row.get("MACD", 0.0))
    macd_signal = float(row.get("MACD_Signal", 0.0))

    # 1. EMA vs SMA
    if ema_12 > sma_20:
        score += 0.50
        details["EMA_vs_SMA"] = (+0.50, f"Bullish: EMA12 ({ema_12:.2f}) > SMA20 ({sma_20:.2f})")
    else:
        score -= 0.50
        details["EMA_vs_SMA"] = (-0.50, f"Bearish: EMA12 ({ema_12:.2f}) <= SMA20 ({sma_20:.2f})")

    # 2. Price vs SMA
    if close > sma_20:
        score += 0.25
        details["Price_vs_SMA"] = (+0.25, f"Bullish: Price (${close:.2f}) > SMA20 (${sma_20:.2f})")
    else:
        score -= 0.25
        details["Price_vs_SMA"] = (-0.25, f"Bearish: Price (${close:.2f}) <= SMA20 (${sma_20:.2f})")

    # 3. RSI Overbought / Oversold
    if rsi_14 > config.RSI_OVERBOUGHT:
        score -= 0.50
        details["RSI"] = (-0.50, f"Overbought: RSI14 ({rsi_14:.1f}) > {config.RSI_OVERBOUGHT}")
    elif rsi_14 < config.RSI_OVERSOLD:
        score += 0.50
        details["RSI"] = (+0.50, f"Oversold: RSI14 ({rsi_14:.1f}) < {config.RSI_OVERSOLD}")
    else:
        details["RSI"] = (0.00, f"Neutral: RSI14 ({rsi_14:.1f}) in standard range")

    # 4. MACD vs Signal
    if macd > macd_signal:
        score += 0.25
        details["MACD"] = (+0.25, f"Bullish: MACD ({macd:.3f}) > Signal ({macd_signal:.3f})")
    else:
        score -= 0.25
        details["MACD"] = (-0.25, f"Bearish: MACD ({macd:.3f}) <= Signal ({macd_signal:.3f})")

    clamped_score = float(np.clip(score, -1.0, 1.0))
    return round(clamped_score, 4), details


def evaluate_trading_decision(
    predicted_price: float,
    current_price: float,
    sentiment_score: float,
    latest_market_row: pd.Series,
    confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
    threshold_boost: float = 0.0,
) -> Dict[str, Any]:
    """Generates the unified EAATS ensemble trading decision.

    Weights:
        LSTM: 0.40
        Sentiment: 0.30
        Technical: 0.30

    Decision boundaries:
        > +0.15: BUY
        < -0.15: SELL
        else:    HOLD
    """
    weights = config.STRATEGY_WEIGHTS

    # Compute raw sub-signals
    lstm_sig, lstm_raw_pct = calculate_lstm_signal(predicted_price, current_price)
    sent_sig, sent_raw_score = calculate_sentiment_signal(sentiment_score)
    tech_sig, tech_breakdown = calculate_technical_signal(latest_market_row)

    # Weighted combination
    lstm_contrib = weights["lstm"] * lstm_sig
    sent_contrib = weights["sentiment"] * sent_sig
    tech_contrib = weights["technical"] * tech_sig

    final_score = lstm_contrib + sent_contrib + tech_contrib
    confidence = min(abs(final_score) * 100.0, 100.0)

    # Determine Raw Decision
    if final_score > config.DECISION_BOUNDARY:
        raw_decision = "BUY"
    elif final_score < -config.DECISION_BOUNDARY:
        raw_decision = "SELL"
    else:
        raw_decision = "HOLD"

    # Effective Confidence Threshold (including adaptive risk boost)
    effective_threshold = confidence_threshold + threshold_boost

    # Confidence Threshold Suppression Filter
    is_suppressed = False
    suppression_reason = None
    final_decision = raw_decision

    if raw_decision != "HOLD" and confidence < effective_threshold:
        final_decision = "HOLD"
        is_suppressed = True
        suppression_reason = (
            f"Signal suppressed (low confidence): Raw {raw_decision} signal had "
            f"{confidence:.1f}% confidence, which is below the active threshold of {effective_threshold:.1f}%."
        )

    # Signal Breakdown Table
    breakdown_table = [
        {
            "Signal Name": "Deep Learning (LSTM)",
            "Raw Value": f"{lstm_raw_pct:+.2f}% predicted move",
            "Discrete Score": lstm_sig,
            "Weight": f"{weights['lstm'] * 100:.0f}%",
            "Contribution": f"{lstm_contrib:+.3f}",
        },
        {
            "Signal Name": "Emotion & Sentiment",
            "Raw Value": f"{sent_raw_score:+.3f} composite",
            "Discrete Score": sent_sig,
            "Weight": f"{weights['sentiment'] * 100:.0f}%",
            "Contribution": f"{sent_contrib:+.3f}",
        },
        {
            "Signal Name": "Quantitative Technicals",
            "Raw Value": f"{tech_sig:+.2f} score",
            "Discrete Score": tech_sig,
            "Weight": f"{weights['technical'] * 100:.0f}%",
            "Contribution": f"{tech_contrib:+.3f}",
        },
    ]

    return {
        "final_decision": final_decision,
        "raw_decision": raw_decision,
        "final_score": round(final_score, 4),
        "confidence": round(confidence, 1),
        "is_suppressed": is_suppressed,
        "suppression_reason": suppression_reason,
        "effective_threshold": round(effective_threshold, 1),
        "breakdown_table": pd.DataFrame(breakdown_table),
        "tech_details": tech_breakdown,
        "signals": {
            "lstm": {"signal": lstm_sig, "value": lstm_raw_pct},
            "sentiment": {"signal": sent_sig, "value": sent_raw_score},
            "technical": {"signal": tech_sig, "value": tech_sig},
        },
    }

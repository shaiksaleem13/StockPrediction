"""Adaptive Emotion-Based Risk Engine for EAATS.

Dynamically modulates position sizing, stop-loss protection limits,
and signal entry barriers based on the active market emotion state
and emotional volatility.
"""

from typing import Any, Dict
import pandas as pd

import config
from src.utils import logger


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates Average True Range (ATR) as a measure of recent market volatility."""
    if len(df) < period + 1:
        return 0.0

    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if not pd.isna(atr) else 0.0


def determine_adaptive_risk_profile(
    emotion_state: str,
    emotional_volatility: float = 0.0,
    df_market: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Computes the active risk profile adapting to emotional state.

    Modulates:
    - position_size: fraction of total portfolio capital allocated per trade
    - stop_loss: trailing stop-loss percentage
    - threshold_boost: points added to confidence threshold to filter false breakouts
    - block_buys: whether to forbid new long entries due to extreme emotional instability
    """
    profiles = config.RISK_PROFILES
    profile_data = profiles.get(emotion_state, profiles["NEUTRAL"]).copy()

    position_size = profile_data["position_size"]
    stop_loss = profile_data["stop_loss"]
    threshold_boost = profile_data["threshold_boost"]
    explanation = profile_data["explanation"]

    # Check for extreme emotional volatility override
    block_buys = False
    if emotion_state in ["PANIC", "FEAR"] and emotional_volatility >= 0.45:
        block_buys = True
        position_size = 0.15  # Further defensive haircut
        threshold_boost += 10.0
        explanation += (
            f" [ALERT] Severe emotional volatility ({emotional_volatility:.2f}) detected: "
            "New BUY entries temporarily suspended and position size throttled to 15%."
        )

    # ATR Context if market history is available
    recent_atr = 0.0
    if df_market is not None and not df_market.empty:
        recent_atr = calculate_atr(df_market)

    return {
        "emotion_state": emotion_state,
        "position_size": position_size,
        "position_size_pct": f"{position_size * 100:.0f}%",
        "stop_loss": stop_loss,
        "stop_loss_pct": f"{stop_loss * 100:.1f}%",
        "threshold_boost": threshold_boost,
        "block_buys": block_buys,
        "emotional_volatility": round(emotional_volatility, 3),
        "recent_atr": round(recent_atr, 2),
        "explanation": explanation,
    }

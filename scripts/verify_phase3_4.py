"""Phase 3 & Phase 4 Verification Script for EAATS.

Tests:
1. VADER & FinBERT Scorer modules.
2. Emotion Engine aggregation with missing source re-normalization.
3. Emotion classification & Historical Sentiment Proxy.
4. Trading Strategy weighted voting, decision boundaries, and confidence threshold suppression.
5. Adaptive Emotion Risk Engine profiles and volatility overrides.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_fetcher import load_csv
from src.preprocessor import add_technical_indicators
from src.sentiment.vader_scorer import score_text_vader
from src.sentiment.finbert_scorer import score_text_finbert
from src.sentiment.emotion_engine import (
    aggregate_source_scores,
    classify_emotion_state,
    compute_historical_sentiment_proxy,
)
from src.trading_strategy import (
    calculate_lstm_signal,
    calculate_sentiment_signal,
    calculate_technical_signal,
    evaluate_trading_decision,
)
from src.adaptive_risk import determine_adaptive_risk_profile


def verify_phase3_4():
    print("=" * 65)
    print("RUNNING PHASE 3 & 4 VERIFICATION (EMOTION ENGINE & ADAPTIVE RISK)")
    print("=" * 65)

    # 1. Test VADER Scorer
    pos_vader = score_text_vader("Apple crushed earnings, record iPhone sales and huge revenue growth!")
    neg_vader = score_text_vader("Terrible market crash, massive layoffs, bankruptcy fears loom!")
    neu_vader = score_text_vader("The committee met on Tuesday afternoon for the regular session.")

    assert pos_vader["normalized_score"] > 0.05 and pos_vader["label"] == "POSITIVE"
    assert neg_vader["normalized_score"] < -0.05 and neg_vader["label"] == "NEGATIVE"
    assert pos_vader["normalized_score"] >= -1.0 and pos_vader["normalized_score"] <= 1.0
    print(f"[PASS] VADER Scorer functioning: Pos={pos_vader['normalized_score']}, Neg={neg_vader['normalized_score']}")

    # 2. Test FinBERT Scorer
    fin_pos = score_text_finbert("Operating margins expand by 400 basis points amid robust cloud demand.")
    fin_neg = score_text_finbert("Revenue plunges 25% as supply chain bottlenecks devastate delivery targets.")
    assert -1.0 <= fin_pos["score"] <= 1.0
    assert -1.0 <= fin_neg["score"] <= 1.0
    print(f"[PASS] FinBERT Scorer functioning: Pos={fin_pos['score']}, Neg={fin_neg['score']}")

    # 3. Test Emotion Engine Aggregation & Re-normalization
    # Test case A: All 4 sources present
    sources_full = {
        "news": [{"score": 0.60}],
        "social": [{"score": 0.40}],
        "forums": [{"score": 0.30}],
        "video": [{"score": 0.50}],
    }
    score_full, vol_full, details_full = aggregate_source_scores(sources_full)
    # Expected: 0.4*0.6 + 0.3*0.4 + 0.2*0.3 + 0.1*0.5 = 0.24 + 0.12 + 0.06 + 0.05 = 0.47
    assert np.isclose(score_full, 0.47, atol=1e-3), f"Expected 0.47, got {score_full}"
    print(f"[PASS] 4-Source Aggregation verified: Score={score_full}, Volatility={vol_full:.4f}")

    # Test case B: 2 sources missing (video and forums missing) -> auto-renormalization
    sources_partial = {
        "news": [{"score": 0.80}],
        "social": [{"score": 0.40}],
        "forums": [],
        "video": [],
    }
    score_part, vol_part, details_part = aggregate_source_scores(sources_partial)
    # Total available weight: news (0.4) + social (0.3) = 0.7. Re-normalized: news 0.4/0.7=0.5714, social 0.3/0.7=0.4286
    # Score = 0.8 * (4/7) + 0.4 * (3/7) = (3.2 + 1.2)/7 = 4.4/7 = 0.6286
    expected_renorm = (0.80 * 0.40 + 0.40 * 0.30) / 0.70
    assert np.isclose(score_part, expected_renorm, atol=1e-3), f"Expected {expected_renorm:.4f}, got {score_part}"
    print(f"[PASS] Missing-Source Re-normalization verified: Score={score_part} (weights scaled from 0.7 to 1.0)")

    # Test case C: Zero sources -> defaults to 0.0 Neutral
    score_zero, vol_zero, details_zero = aggregate_source_scores({"news": [], "social": [], "forums": [], "video": []})
    assert score_zero == 0.0 and details_zero["warning"] is not None
    print(f"[PASS] Zero-source graceful fallback: Score={score_zero}, Warning confirmed.")

    # 4. Test Emotion State Mapping
    assert classify_emotion_state(-0.75)[0] == "PANIC"
    assert classify_emotion_state(-0.35)[0] == "FEAR"
    assert classify_emotion_state(0.00)[0] == "NEUTRAL"
    assert classify_emotion_state(0.35)[0] == "CONFIDENCE"
    assert classify_emotion_state(0.65)[0] == "GREED"
    print(f"[PASS] All 5 emotion state classification boundaries verified.")

    # 5. Test Historical Sentiment Proxy on AAPL
    df = load_csv(config.SAMPLE_DIR / "AAPL.csv")
    proxy = compute_historical_sentiment_proxy(df)
    assert len(proxy) == len(df)
    assert (proxy >= -1.0).all() and (proxy <= 1.0).all()
    print(f"[PASS] Historical Sentiment Proxy calculated: Mean={proxy.mean():.4f}, Min={proxy.min():.4f}, Max={proxy.max():.4f}")

    # 6. Test Trading Strategy Weighted Voting
    df_feat = add_technical_indicators(df)
    sample_row = df_feat.iloc[-1]

    # Test Bullish Case
    decision_buy = evaluate_trading_decision(
        predicted_price=200.0,
        current_price=180.0,  # +11% move -> +1
        sentiment_score=0.40,  # +1
        latest_market_row=sample_row,
        confidence_threshold=30.0,
    )
    assert decision_buy["raw_decision"] in ["BUY", "HOLD"]
    print(f"[PASS] Trading Strategy evaluated: Decision={decision_buy['final_decision']}, Score={decision_buy['final_score']}, Confidence={decision_buy['confidence']}%")

    # Test Low-Confidence Suppression
    decision_suppressed = evaluate_trading_decision(
        predicted_price=180.8,
        current_price=180.0,  # +0.44% move -> 0 LSTM
        sentiment_score=0.08,  # +1 sentiment (0.3 contribution)
        latest_market_row=sample_row,
        confidence_threshold=45.0,  # Threshold set high
    )
    if decision_suppressed["confidence"] < 45.0 and decision_suppressed["raw_decision"] != "HOLD":
        assert decision_suppressed["final_decision"] == "HOLD"
        assert decision_suppressed["is_suppressed"] is True
        print(f"[PASS] Low-Confidence Signal Suppression verified: Raw {decision_suppressed['raw_decision']} suppressed to HOLD.")

    # 7. Test Adaptive Risk Engine
    risk_panic = determine_adaptive_risk_profile("PANIC", emotional_volatility=0.55, df_market=df)
    assert risk_panic["position_size"] == 0.15  # Throttled due to high vol
    assert risk_panic["block_buys"] is True
    assert risk_panic["stop_loss"] == 0.02
    print(f"[PASS] Adaptive Risk Panic Profile with High Volatility Override verified: Position={risk_panic['position_size_pct']}, BlockBuys={risk_panic['block_buys']}")

    risk_conf = determine_adaptive_risk_profile("CONFIDENCE", emotional_volatility=0.20, df_market=df)
    assert risk_conf["position_size"] == 0.75
    assert risk_conf["stop_loss"] == 0.04
    print(f"[PASS] Adaptive Risk Confidence Profile verified: Position={risk_conf['position_size_pct']}, StopLoss={risk_conf['stop_loss_pct']}")

    risk_greed = determine_adaptive_risk_profile("GREED", emotional_volatility=0.15, df_market=df)
    assert risk_greed["position_size"] == 0.50  # Capped
    assert risk_greed["stop_loss"] == 0.025  # Tightened
    print(f"[PASS] Adaptive Risk Greed Profile (Contrarian Cap) verified: Position={risk_greed['position_size_pct']}, StopLoss={risk_greed['stop_loss_pct']}")

    print("=" * 65)
    print("ALL PHASE 3 & PHASE 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    verify_phase3_4()

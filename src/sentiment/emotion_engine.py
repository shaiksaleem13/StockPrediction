"""Multi-Source Emotion Detection Engine for EAATS.

Aggregates news (FinBERT), social media (VADER), forum discussions (VADER),
and video transcripts (FinBERT) into a unified market emotion state:
PANIC, FEAR, NEUTRAL, CONFIDENCE, or GREED.

Includes automatic source re-normalization, emotional volatility calculation,
and historical sentiment proxy generation for backtesting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import config
from src.sentiment.finbert_scorer import score_text_finbert
from src.sentiment.news_source import fetch_google_news
from src.sentiment.social_source import fetch_reddit_posts, fetch_stocktwits_stream
from src.sentiment.vader_scorer import score_text_vader
from src.sentiment.video_source import fetch_youtube_transcripts
from src.utils import logger


def aggregate_source_scores(source_items: Dict[str, List[Dict[str, Any]]]) -> Tuple[float, float, Dict[str, Any]]:
    """Aggregates multi-source feeds with dynamic weight re-normalization.

    Returns:
        composite_score: float in [-1.0, 1.0]
        emotional_volatility: float (standard deviation of scores)
        detailed_breakdown: dictionary with per-source metrics
    """
    base_weights = config.SOURCE_WEIGHTS
    valid_sources = {}
    all_individual_scores = []
    source_summaries = {}

    for source_key, items in source_items.items():
        if items and len(items) > 0:
            scores = [item["score"] for item in items]
            mean_score = float(np.mean(scores))
            valid_sources[source_key] = mean_score
            all_individual_scores.extend(scores)
            source_summaries[source_key] = {
                "mean_score": round(mean_score, 4),
                "count": len(items),
                "sample_items": items[:5],
            }
        else:
            source_summaries[source_key] = {
                "mean_score": 0.0,
                "count": 0,
                "sample_items": [],
            }

    # If no textual sources are available, return Neutral with zero volatility
    if not valid_sources:
        logger.warning("No text sentiment sources available. Defaulting emotion state to NEUTRAL (0.0).")
        return 0.0, 0.0, {
            "source_summaries": source_summaries,
            "weights_used": {},
            "all_items": [],
            "warning": "No real-time text sources found. System defaulting to Neutral (0.0).",
        }

    # Dynamic Weight Re-normalization
    total_base_weight = sum(base_weights[k] for k in valid_sources.keys())
    normalized_weights = {k: base_weights[k] / total_base_weight for k in valid_sources.keys()}

    composite_score = sum(valid_sources[k] * normalized_weights[k] for k in valid_sources.keys())
    composite_score = float(np.clip(composite_score, -1.0, 1.0))

    # Emotional Volatility = standard deviation of individual sentiment evaluations
    emotional_volatility = float(np.std(all_individual_scores)) if len(all_individual_scores) > 1 else 0.0

    return (
        round(composite_score, 4),
        round(emotional_volatility, 4),
        {
            "source_summaries": source_summaries,
            "weights_used": {k: round(v, 3) for k, v in normalized_weights.items()},
            "item_count": len(all_individual_scores),
            "warning": None,
        },
    )


def classify_emotion_state(
    score: float,
    previous_scores: Optional[List[float]] = None,
) -> Tuple[str, str]:
    """Maps composite sentiment score [-1, +1] to market emotion state.

    Categories:
        PANIC: <= -0.50 (or sharp drop vs 7-day average)
        FEAR: -0.50 to -0.15
        NEUTRAL: -0.15 to +0.15
        CONFIDENCE: +0.15 to +0.50
        GREED: >= +0.50

    Returns:
        (emotion_label, explanation_text)
    """
    # Detect sudden panic drop if historical sentiment vector is provided
    is_panic_drop = False
    if previous_scores and len(previous_scores) >= 5:
        avg_prev = float(np.mean(previous_scores[-7:]))
        if score - avg_prev < -0.40:
            is_panic_drop = True

    if score <= config.EMOTION_THRESHOLDS["PANIC"] or is_panic_drop:
        return (
            "PANIC",
            "Severe market pessimism or sharp panic sell-off detected across news and public sentiment feeds.",
        )
    elif score <= config.EMOTION_THRESHOLDS["FEAR"]:
        return (
            "FEAR",
            "Heightened uncertainty and negative commentary prevailing in market discussions.",
        )
    elif score < config.EMOTION_THRESHOLDS["NEUTRAL_HIGH"]:
        return (
            "NEUTRAL",
            "Balanced market sentiment with equilibrium between optimistic and cautious opinions.",
        )
    elif score < config.EMOTION_THRESHOLDS["CONFIDENCE"]:
        return (
            "CONFIDENCE",
            "Constructive optimism and positive fundamental/technical outlook observed across feeds.",
        )
    else:
        return (
            "GREED",
            "Elevated euphoria or speculative greed detected; contrarian caution advised.",
        )


def analyze_market_emotion(
    ticker: str,
    enable_social: bool = True,
    enable_video: bool = True,
) -> Dict[str, Any]:
    """Orchestrates multi-source sentiment scraping, scoring, and emotion classification."""
    logger.info(f"Initiating multi-source emotion analysis for {ticker}...")

    # 1. Fetch & Score News (FinBERT)
    news_items = fetch_google_news(ticker)
    scored_news = []
    for item in news_items:
        res = score_text_finbert(item["text"])
        scored_news.append({**item, "score": res["score"], "sentiment_label": res["label"]})

    # 2. Fetch & Score Social (StockTwits & Reddit via VADER)
    scored_social = []
    scored_forums = []
    if enable_social:
        st_items = fetch_stocktwits_stream(ticker)
        for item in st_items:
            res = score_text_vader(item["text"])
            scored_social.append({**item, "score": res["normalized_score"], "sentiment_label": res["label"]})

        reddit_items = fetch_reddit_posts(ticker)
        for item in reddit_items:
            res = score_text_vader(item["text"])
            scored_forums.append({**item, "score": res["normalized_score"], "sentiment_label": res["label"]})

    # 3. Fetch & Score Video Transcripts (FinBERT)
    scored_video = []
    if enable_video:
        video_items = fetch_youtube_transcripts(ticker)
        for item in video_items:
            res = score_text_finbert(item["text"])
            scored_video.append({**item, "score": res["score"], "sentiment_label": res["label"]})

    source_dict = {
        "news": scored_news,
        "social": scored_social,
        "forums": scored_forums,
        "video": scored_video,
    }

    # Aggregate
    composite_score, emotional_volatility, details = aggregate_source_scores(source_dict)
    emotion_label, explanation = classify_emotion_state(composite_score)

    # Flatten all items for UI display
    all_items = scored_news + scored_social + scored_forums + scored_video

    return {
        "ticker": ticker,
        "composite_score": composite_score,
        "emotion_label": emotion_label,
        "explanation": explanation,
        "emotional_volatility": emotional_volatility,
        "details": details,
        "all_items": all_items,
        "counts": {
            "news": len(scored_news),
            "social": len(scored_social),
            "forums": len(scored_forums),
            "video": len(scored_video),
        },
    }


def compute_historical_sentiment_proxy(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """Computes a realistic, academically grounded Fear & Greed sentiment proxy

    for backtesting periods where real-time social feeds cannot be reconstructed.
    Proxy formula uses 14-day rolling return momentum and rolling volatility:
    Proxy = tanh(Momentum / Volatility) in [-1.0, 1.0].
    """
    close = df["Close"]
    returns_window = close.pct_change(periods=window)
    volatility_window = close.pct_change().rolling(window=window).std()

    # Annualized Sharpe-like ratio of recent momentum
    ratio = returns_window / (volatility_window.replace(0, np.nan) * np.sqrt(window))
    ratio = ratio.fillna(0.0)

    # Nonlinear scaling using hyperbolic tangent
    sentiment_proxy = np.tanh(ratio * 0.75)
    sentiment_proxy = sentiment_proxy.clip(-1.0, 1.0)
    sentiment_proxy.name = "Sentiment_Proxy"
    return sentiment_proxy

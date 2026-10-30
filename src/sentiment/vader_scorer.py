"""VADER Sentiment Scorer for Social Media, Forum Posts & Short Text in EAATS."""

from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from src.utils import logger

# Singleton analyzer instance
_analyzer = None


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy loader for VADER SentimentIntensityAnalyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_text_vader(text: str) -> Dict[str, float]:
    """Scores a short text using VADER.

    Returns:
        dict with compound, pos, neg, neu, and normalized_score in [-1.0, 1.0].
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "normalized_score": 0.0, "label": "NEUTRAL"}

    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = float(scores["compound"])

    if compound > config.VADER_POSITIVE_THRESHOLD:
        label = "POSITIVE"
    elif compound < config.VADER_NEGATIVE_THRESHOLD:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "normalized_score": round(compound, 4),  # VADER compound is already in [-1, 1]
        "label": label,
    }

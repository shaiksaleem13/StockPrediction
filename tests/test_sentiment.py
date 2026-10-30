"""Unit tests for Sentiment Scorers and Emotion Engine in EAATS."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.sentiment.vader_scorer import score_text_vader
from src.sentiment.emotion_engine import aggregate_source_scores, classify_emotion_state


def test_vader_scorer_polarities():
    """Tests VADER score bounds and sentiment classification labels."""
    pos = score_text_vader("Incredible profit surge, revolutionary product innovation!")
    neg = score_text_vader("Devastating economic collapse, fraud allegations, massive bankruptcy.")
    neu = score_text_vader("The quarterly report will be filed on the regular schedule.")

    assert pos["label"] == "POSITIVE" and pos["normalized_score"] > 0.05
    assert neg["label"] == "NEGATIVE" and neg["normalized_score"] < -0.05
    assert neu["label"] == "NEUTRAL" and abs(neu["normalized_score"]) <= 0.05
    assert -1.0 <= pos["normalized_score"] <= 1.0


def test_missing_source_dynamic_renormalization():
    """Verifies that missing sentiment sources trigger proportional weight re-normalization."""
    # Only news (weight 0.40) and forums (weight 0.20) available. Total weight = 0.60
    # News re-normalized = 0.40 / 0.60 = 2/3
    # Forums re-normalized = 0.20 / 0.60 = 1/3
    source_items = {
        "news": [{"score": 0.60}],
        "social": [],
        "forums": [{"score": 0.30}],
        "video": [],
    }

    comp_score, vol, details = aggregate_source_scores(source_items)
    expected_score = (0.60 * (0.40 / 0.60)) + (0.30 * (0.20 / 0.60))
    assert np.isclose(comp_score, expected_score, atol=1e-3)
    assert np.isclose(sum(details["weights_used"].values()), 1.0, atol=1e-3)


def test_emotion_state_classification():
    """Verifies all 5 emotion categories: PANIC, FEAR, NEUTRAL, CONFIDENCE, GREED."""
    assert classify_emotion_state(-0.80)[0] == "PANIC"
    assert classify_emotion_state(-0.30)[0] == "FEAR"
    assert classify_emotion_state(0.00)[0] == "NEUTRAL"
    assert classify_emotion_state(0.30)[0] == "CONFIDENCE"
    assert classify_emotion_state(0.70)[0] == "GREED"

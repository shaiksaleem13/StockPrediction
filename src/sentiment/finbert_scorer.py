"""FinBERT Sentiment Scorer for Financial News Headlines & Video Transcripts.

Uses HuggingFace ProsusAI/finbert via PyTorch and Transformers.
Score formulation: Score = P(positive) - P(negative) in [-1.0, 1.0].
Includes automatic fallback to VADER if offline or if model weights are unavailable.
"""

from typing import Dict, List, Optional
import torch

import config
from src.sentiment.vader_scorer import score_text_vader
from src.utils import logger

_finbert_tokenizer = None
_finbert_model = None
_finbert_available = True


def load_finbert():
    """Lazy loads FinBERT tokenizer and model weights with fallback handling."""
    global _finbert_tokenizer, _finbert_model, _finbert_available

    if not _finbert_available:
        return None, None

    if _finbert_tokenizer is not None and _finbert_model is not None:
        return _finbert_tokenizer, _finbert_model

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading FinBERT model ({config.FINBERT_MODEL_NAME})...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL_NAME)
        _finbert_model.eval()
        logger.info("FinBERT successfully loaded.")
        return _finbert_tokenizer, _finbert_model
    except Exception as e:
        logger.warning(f"Could not load FinBERT from HuggingFace ({e}). Falling back to VADER.")
        _finbert_available = False
        return None, None


def score_text_finbert(text: str) -> Dict[str, float]:
    """Scores financial text using FinBERT.

    Formula: score = P(positive) - P(negative) in range [-1.0, 1.0].
    """
    if not text or not isinstance(text, str):
        return {"score": 0.0, "p_pos": 0.0, "p_neg": 0.0, "p_neu": 1.0, "label": "NEUTRAL"}

    tokenizer, model = load_finbert()

    if tokenizer is None or model is None:
        # Graceful fallback to VADER
        vader_res = score_text_vader(text)
        return {
            "score": vader_res["normalized_score"],
            "p_pos": vader_res["pos"],
            "p_neg": vader_res["neg"],
            "p_neu": vader_res["neu"],
            "label": vader_res["label"],
        }

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # FinBERT labels mapping: 0: positive, 1: negative, 2: neutral
        p_pos = float(probs[0].item())
        p_neg = float(probs[1].item())
        p_neu = float(probs[2].item())

        score = p_pos - p_neg  # Range: [-1.0, 1.0]

        if score > config.VADER_POSITIVE_THRESHOLD:
            label = "POSITIVE"
        elif score < config.VADER_NEGATIVE_THRESHOLD:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return {
            "score": round(score, 4),
            "p_pos": round(p_pos, 4),
            "p_neg": round(p_neg, 4),
            "p_neu": round(p_neu, 4),
            "label": label,
        }
    except Exception as e:
        logger.warning(f"FinBERT inference error on '{text[:30]}...': {e}; using VADER fallback.")
        vader_res = score_text_vader(text)
        return {
            "score": vader_res["normalized_score"],
            "p_pos": vader_res["pos"],
            "p_neg": vader_res["neg"],
            "p_neu": vader_res["neu"],
            "label": vader_res["label"],
        }

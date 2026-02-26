"""Financial sentiment scoring using FinBERT."""

from functools import lru_cache

from finsight.config.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_finbert():
    """Lazy-load FinBERT model and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.error("finbert_load_failed", error=str(e))
        return None, None


def score_sentiment(text: str) -> dict:
    """Score text sentiment using FinBERT.

    Returns dict with keys: label (positive/negative/neutral),
    score (confidence 0-1), scores (all three class probabilities).
    """
    tokenizer, model = _load_finbert()

    if tokenizer is None or model is None:
        return _fallback_sentiment(text)

    try:
        import torch

        truncated = text[:512]
        inputs = tokenizer(truncated, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        labels = ["positive", "negative", "neutral"]
        scores = {label: round(prob.item(), 4) for label, prob in zip(labels, probs)}

        best_idx = probs.argmax().item()
        return {
            "label": labels[best_idx],
            "score": round(probs[best_idx].item(), 4),
            "scores": scores,
        }
    except Exception as e:
        logger.error("sentiment_scoring_failed", error=str(e))
        return _fallback_sentiment(text)


def _fallback_sentiment(text: str) -> dict:
    """Simple keyword-based fallback when FinBERT is unavailable."""
    text_lower = text.lower()

    positive_words = {
        "surge", "rally", "gain", "rise", "jump", "soar", "boost", "bull",
        "optimistic", "growth", "beat", "exceed", "strong", "recovery",
        "upgrade", "outperform", "record high", "breakthrough",
    }
    negative_words = {
        "crash", "plunge", "drop", "fall", "decline", "slump", "bear",
        "pessimistic", "recession", "miss", "weak", "downturn", "selloff",
        "downgrade", "underperform", "record low", "crisis", "default",
    }

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return {"label": "neutral", "score": 0.5, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}}

    pos_ratio = pos_count / total
    if pos_ratio > 0.6:
        return {"label": "positive", "score": round(pos_ratio, 4), "scores": {"positive": pos_ratio, "negative": 1 - pos_ratio, "neutral": 0.0}}
    elif pos_ratio < 0.4:
        neg_ratio = neg_count / total
        return {"label": "negative", "score": round(neg_ratio, 4), "scores": {"positive": pos_ratio, "negative": neg_ratio, "neutral": 0.0}}
    else:
        return {"label": "neutral", "score": 0.5, "scores": {"positive": pos_ratio, "negative": 1 - pos_ratio, "neutral": 0.0}}

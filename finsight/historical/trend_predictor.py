"""Predict market trends based on current news + historical patterns.

Combines:
1. Current news context from the live pipeline
2. Historical pattern matches from the pattern matcher
3. Current market data
to generate trend predictions with confidence scores.
"""

import json
import logging
from datetime import datetime
from typing import Any

import ollama as ollama_client

from finsight.config.settings import settings
from finsight.historical.pattern_matcher import (
    find_similar_events,
    get_historical_context_for_prompt,
)

logger = logging.getLogger(__name__)

PREDICTION_PROMPT = """You are FinSight, an AI financial analyst with deep knowledge of historical market patterns.

You have access to:
1. Current news and market data
2. Historical parallels — past events that are similar to what's happening now
3. What actually happened after those historical events

Your task: Generate specific, actionable market trend predictions.

For each major asset class (equities, bonds, commodities, currencies, crypto):
1. Give a directional call: BULLISH / BEARISH / NEUTRAL
2. Assign a confidence score (0-100%)
3. Cite the specific historical parallel that supports your call
4. Provide a 1-week and 1-month outlook

Be specific. Use numbers. If a pattern repeated 4 out of 5 times historically, say that.
Always caveat with risk factors that could invalidate the prediction."""


def predict_trends(
    current_news: str,
    current_market_data: dict | None = None,
    top_parallels: int = 5,
) -> dict[str, Any]:
    """Generate trend predictions based on current context and historical patterns."""
    parallels = find_similar_events(current_news, top_k=top_parallels)
    historical_text = get_historical_context_for_prompt(current_news, top_k=3)

    market_text = ""
    if current_market_data and "rates" in current_market_data:
        lines = []
        for k, v in current_market_data["rates"].items():
            change = current_market_data.get("changes", {}).get(k, 0)
            lines.append(f"  {k}: {v} ({change:+.2f}%)")
        market_text = "\n".join(lines)

    prompt = f"""=== CURRENT NEWS ===
{current_news[:2000]}

=== CURRENT MARKET DATA ===
{market_text}

{historical_text}

Based on the current news, market data, and historical parallels above,
generate your market trend predictions for the next week and month.
"""

    try:
        response = ollama_client.chat(
            model=settings.ollama_llm_model,
            messages=[
                {"role": "system", "content": PREDICTION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_ctx": 8192, "num_predict": 1500},
        )
        prediction_text = response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        prediction_text = _generate_rule_based_prediction(parallels, current_market_data)

    predictions = _extract_structured_predictions(prediction_text, parallels)

    return {
        "prediction_text": prediction_text,
        "predictions": predictions,
        "parallels": [
            {
                "week": p["week_start"],
                "similarity": p["similarity"],
                "summary": p["outcome"][:200],
            }
            for p in parallels
        ],
        "confidence": _calculate_overall_confidence(parallels),
        "generated_at": datetime.now().isoformat(),
    }


def _generate_rule_based_prediction(
    parallels: list[dict],
    market_data: dict | None,
) -> str:
    """Fallback: generate predictions from historical pattern analysis only."""
    if not parallels:
        return "Insufficient historical data for prediction. Monitor key events."

    lines = ["**Market Trend Predictions (Rule-Based)**\n"]

    outcome_texts = [p.get("outcome", "") for p in parallels]
    bullish_count = sum(
        1 for t in outcome_texts
        if any(w in t.lower() for w in ["gained", "rose", "bullish", "rally"])
    )
    bearish_count = sum(
        1 for t in outcome_texts
        if any(w in t.lower() for w in ["fell", "declined", "bearish", "selloff"])
    )

    total = len(parallels)
    if bullish_count > bearish_count:
        direction = "BULLISH"
        confidence = bullish_count / total * 100
    elif bearish_count > bullish_count:
        direction = "BEARISH"
        confidence = bearish_count / total * 100
    else:
        direction = "NEUTRAL"
        confidence = 50

    lines.append(f"**Overall Market Direction: {direction}** "
                 f"(Confidence: {confidence:.0f}%)")
    lines.append(f"\nBased on {total} historical parallels:")
    lines.append(f"- {bullish_count} showed bullish outcomes")
    lines.append(f"- {bearish_count} showed bearish outcomes")
    lines.append(f"- {total - bullish_count - bearish_count} were neutral")

    lines.append("\n**Historical Parallels Used:**")
    for p in parallels[:3]:
        lines.append(
            f"- Week of {p['week_start']} ({p['similarity']:.0%} match): "
            f"{p['outcome'][:150]}..."
        )

    return "\n".join(lines)


def _extract_structured_predictions(
    prediction_text: str,
    parallels: list[dict],
) -> list[dict]:
    """Extract structured predictions from the LLM text output."""
    assets = {
        "S&P 500": ["sp500", "s&p", "equities", "stocks"],
        "NASDAQ": ["nasdaq", "tech"],
        "Gold": ["gold", "xau"],
        "Crude Oil": ["oil", "wti", "crude"],
        "USD (DXY)": ["dollar", "dxy", "usd"],
        "Bitcoin": ["bitcoin", "btc", "crypto"],
        "Treasury 10Y": ["treasury", "10y", "bonds", "yield"],
    }

    predictions = []
    text_lower = prediction_text.lower()

    for asset_name, keywords in assets.items():
        relevant_section = ""
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx >= 0:
                relevant_section = prediction_text[max(0, idx-50):idx+300]
                break

        if not relevant_section:
            continue

        section_lower = relevant_section.lower()
        if any(w in section_lower for w in ["bullish", "buy", "long", "rise", "gain", "rally"]):
            direction = "BULLISH"
        elif any(w in section_lower for w in ["bearish", "sell", "short", "fall", "decline", "drop"]):
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = 50
        if parallels:
            avg_sim = sum(p["similarity"] for p in parallels) / len(parallels)
            confidence = int(avg_sim * 100)

        predictions.append({
            "asset": asset_name,
            "direction": direction,
            "confidence": min(confidence, 95),
            "reasoning": relevant_section[:200].strip(),
        })

    return predictions


def _calculate_overall_confidence(parallels: list[dict]) -> int:
    """Calculate overall prediction confidence based on pattern match quality."""
    if not parallels:
        return 0

    avg_similarity = sum(p["similarity"] for p in parallels) / len(parallels)
    coverage_bonus = min(len(parallels) * 5, 20)

    confidence = int(avg_similarity * 80 + coverage_bonus)
    return min(confidence, 95)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    result = predict_trends(
        "Federal Reserve signals potential rate cut amid cooling inflation. "
        "Tech earnings beat expectations. Geopolitical tensions in Middle East. "
        "Oil prices rising on supply concerns."
    )

    print(f"\nOverall Confidence: {result['confidence']}%")
    print(f"\n{result['prediction_text']}")
    print(f"\nParallels: {len(result['parallels'])}")
    for p in result["parallels"]:
        print(f"  - {p['week']} ({p['similarity']:.0%}): {p['summary'][:100]}...")

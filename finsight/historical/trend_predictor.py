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

PREDICTION_PROMPT = """You are FinSight, an expert financial analyst. You must generate detailed market predictions.

IMPORTANT: You must write a FULL analysis. Do NOT just repeat the instructions. Write at least 500 words.

Structure your response EXACTLY like this:

## Equities (S&P 500, NASDAQ)
Direction: BULLISH/BEARISH/NEUTRAL
Confidence: X%
Analysis: [Your detailed reasoning citing historical parallels]
1-Week Outlook: [Specific prediction]
1-Month Outlook: [Specific prediction]

## Bonds (Treasury 10Y)
Direction: BULLISH/BEARISH/NEUTRAL
Confidence: X%
Analysis: [Your detailed reasoning]

## Commodities (Gold, Oil)
Direction: BULLISH/BEARISH/NEUTRAL
Confidence: X%
Analysis: [Your detailed reasoning]

## Currencies (USD/DXY)
Direction: BULLISH/BEARISH/NEUTRAL
Confidence: X%
Analysis: [Your detailed reasoning]

## Crypto (Bitcoin)
Direction: BULLISH/BEARISH/NEUTRAL
Confidence: X%
Analysis: [Your detailed reasoning]

## Risk Factors
[List key risks that could change these predictions]

Be specific. Use numbers. Reference the historical parallels provided."""


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

    prompt = f"""=== CURRENT NEWS AND EVENTS ===
{current_news[:2000]}

=== CURRENT MARKET DATA ===
{market_text if market_text else "Live market data not available — use historical context."}

{historical_text}

Now write your FULL market trend predictions covering equities, bonds, commodities, currencies, and crypto. Reference the historical parallels above. Be detailed and specific."""

    try:
        response = ollama_client.chat(
            model=settings.ollama_llm_model,
            messages=[
                {"role": "system", "content": PREDICTION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.4, "num_ctx": 8192, "num_predict": 2500},
        )
        prediction_text = response["message"]["content"].strip()
        if len(prediction_text) < 100:
            logger.warning("LLM response too short, using rule-based fallback")
            prediction_text = _generate_rule_based_prediction(parallels, current_market_data)
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        prediction_text = _generate_rule_based_prediction(parallels, current_market_data)

    predictions = _build_predictions_from_parallels(parallels)
    if not predictions:
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


def _parse_asset_movements(outcome_text: str) -> dict[str, list[float]]:
    """Extract per-asset percentage movements from outcome text like 'NASDAQ gained 5.7%'."""
    import re
    asset_map = {
        "S&P 500": ["sp500", "s&p_500", "s&p500"],
        "NASDAQ": ["nasdaq"],
        "Gold": ["gold"],
        "Crude Oil": ["crudeoil_wti", "crudeoil", "crude_oil"],
        "Natural Gas": ["natgas"],
        "Bitcoin": ["bitcoin"],
        "Ethereum": ["ethereum"],
        "Treasury 10Y": ["treasury10y", "treasury_10y"],
        "Treasury 5Y": ["treasury5y", "treasury_5y"],
        "Copper": ["copper"],
        "USD Index": ["dxy", "usd_index"],
    }

    movements: dict[str, list[float]] = {}
    for line in outcome_text.split("\n"):
        line_lower = line.lower().replace(" ", "")
        for display_name, keys in asset_map.items():
            for key in keys:
                if key in line_lower:
                    match = re.search(r"(gained|declined|rose|fell|dropped)\s+([\d.]+)%", line, re.IGNORECASE)
                    if match:
                        pct = float(match.group(2))
                        if match.group(1).lower() in ("declined", "fell", "dropped"):
                            pct = -pct
                        movements.setdefault(display_name, []).append(pct)
                    break
    return movements


def _generate_rule_based_prediction(
    parallels: list[dict],
    market_data: dict | None,
) -> str:
    """Generate predictions by parsing actual price movements from historical outcomes."""
    if not parallels:
        return "Insufficient historical data for prediction. Monitor key events."

    all_movements: dict[str, list[float]] = {}
    for p in parallels:
        outcome = p.get("outcome", "")
        moves = _parse_asset_movements(outcome)
        for asset, pcts in moves.items():
            all_movements.setdefault(asset, []).extend(pcts)

    lines = ["## Market Trend Predictions\n"]
    lines.append(f"*Based on {len(parallels)} historical parallel periods with similar conditions*\n")

    for asset, pcts in sorted(all_movements.items(), key=lambda x: -len(x[1])):
        if len(pcts) < 2:
            continue
        avg = sum(pcts) / len(pcts)
        up_count = sum(1 for p in pcts if p > 0)
        down_count = sum(1 for p in pcts if p < 0)

        if avg > 1.0 and up_count > down_count:
            direction = "BULLISH"
        elif avg < -1.0 and down_count > up_count:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        conf = int(max(up_count, down_count) / len(pcts) * 100)

        lines.append(f"### {asset}")
        lines.append(f"**Direction: {direction}** | Confidence: {conf}%")
        lines.append(f"- Historical average move: {avg:+.1f}%")
        lines.append(f"- Rose in {up_count}/{len(pcts)} similar periods, declined in {down_count}/{len(pcts)}")
        lines.append("")

    if not any(len(v) >= 2 for v in all_movements.values()):
        lines.append("*Limited structured data in historical outcomes. See parallels below for context.*\n")

    lines.append("### Historical Parallels Referenced")
    for p in parallels[:3]:
        lines.append(
            f"- **Week of {p['week_start']}** ({p['similarity']:.0%} match): "
            f"{p['outcome'][:200]}"
        )

    return "\n".join(lines)


def _extract_structured_predictions(
    prediction_text: str,
    parallels: list[dict],
) -> list[dict]:
    """Extract structured predictions from the LLM text output."""
    import re

    assets = {
        "S&P 500": ["sp500", "s&p", "equities", "stocks", "stock market"],
        "NASDAQ": ["nasdaq", "tech stocks"],
        "Gold": ["gold", "xau"],
        "Crude Oil": ["oil", "wti", "crude"],
        "USD (DXY)": ["dollar", "dxy", "usd", "forex"],
        "Bitcoin": ["bitcoin", "btc", "crypto"],
        "Treasury 10Y": ["treasury", "10y", "bonds", "yield", "fixed income"],
    }

    predictions = []
    text_lower = prediction_text.lower()

    base_confidence = 50
    if parallels:
        avg_sim = sum(p["similarity"] for p in parallels) / len(parallels)
        base_confidence = int(avg_sim * 100)

    for asset_name, keywords in assets.items():
        relevant_section = ""
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx >= 0:
                relevant_section = prediction_text[max(0, idx - 50):idx + 400]
                break

        if not relevant_section:
            continue

        section_lower = relevant_section.lower()
        bull_signals = ["bullish", "buy", "long", "rise", "gain", "rally", "upside", "positive"]
        bear_signals = ["bearish", "sell", "short", "fall", "decline", "drop", "downside", "negative"]

        bull_score = sum(1 for w in bull_signals if w in section_lower)
        bear_score = sum(1 for w in bear_signals if w in section_lower)

        if bull_score > bear_score:
            direction = "BULLISH"
        elif bear_score > bull_score:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        conf = base_confidence
        conf_match = re.search(r'(\d{1,3})\s*%', relevant_section)
        if conf_match:
            parsed_conf = int(conf_match.group(1))
            if 10 <= parsed_conf <= 99:
                conf = parsed_conf

        clean_reasoning = relevant_section.strip()
        clean_reasoning = re.sub(r'\[https?://[^\]]+\]', '', clean_reasoning)
        clean_reasoning = clean_reasoning[:250].strip()

        predictions.append({
            "asset": asset_name,
            "direction": direction,
            "confidence": min(conf, 95),
            "reasoning": clean_reasoning,
        })

    if not predictions and parallels:
        predictions.append({
            "asset": "Market Overall",
            "direction": "NEUTRAL",
            "confidence": base_confidence,
            "reasoning": "Based on historical pattern analysis. Check the full analysis text for details.",
        })

    return predictions


def _build_predictions_from_parallels(parallels: list[dict]) -> list[dict]:
    """Build structured predictions directly from historical outcome data."""
    if not parallels:
        return []

    all_movements: dict[str, list[float]] = {}
    for p in parallels:
        moves = _parse_asset_movements(p.get("outcome", ""))
        for asset, pcts in moves.items():
            all_movements.setdefault(asset, []).extend(pcts)

    predictions = []
    for asset, pcts in sorted(all_movements.items(), key=lambda x: -len(x[1])):
        if len(pcts) < 2:
            continue
        avg = sum(pcts) / len(pcts)
        up_count = sum(1 for p in pcts if p > 0)
        down_count = sum(1 for p in pcts if p < 0)

        if avg > 1.0 and up_count > down_count:
            direction = "BULLISH"
        elif avg < -1.0 and down_count > up_count:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        conf = int(max(up_count, down_count) / len(pcts) * 100)

        predictions.append({
            "asset": asset,
            "direction": direction,
            "confidence": min(conf, 95),
            "reasoning": f"In {len(pcts)} similar historical periods: avg move {avg:+.1f}%, "
                         f"rose {up_count}/{len(pcts)} times, declined {down_count}/{len(pcts)} times.",
        })

    return predictions[:7]


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

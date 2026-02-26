"""Build training dataset from historical news + market data.

For each weekly window:
1. Gather news headlines from that week (Wikipedia + GDELT)
2. Gather market data for that week
3. Gather market data for the FOLLOWING week (actual outcomes)
4. Generate analytical training pair using GPT-4o-mini

The model learns what happened historically so it can recognize
similar patterns in current events and predict trends.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openai import OpenAI

from finsight.historical.collectors.yahoo_historical import (
    format_market_snapshot,
    get_weekly_summary,
)
from finsight.historical.collectors.wikipedia_events import load_date_range
from finsight.historical.collectors.gdelt_collector import load_week
from finsight.historical.collectors.fred_data import format_economic_snapshot, get_snapshot

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical")
TRAINING_DIR = DATA_DIR / "training"

ANALYST_PROMPT = """You are an expert financial analyst writing a detailed analysis of market events.
Given the news events, market data, and ACTUAL outcomes for a specific week, write a comprehensive analysis.

Your analysis MUST include:
1. **Key Events & Market Impact**: Identify the 3-5 most market-moving events and explain HOW they affected markets
2. **Actual Market Outcomes**: State what actually happened to major indices, commodities, currencies
3. **Cross-Asset Correlations**: Note any notable correlations (e.g., "stocks fell while gold rose, typical risk-off")
4. **Causal Chains**: Connect events to outcomes (e.g., "CPI surprise → rate hike fears → bond selloff → equity weakness")
5. **Pattern Recognition**: If this resembles other historical periods, mention it
6. **Key Takeaways**: 2-3 actionable lessons from this week

Write as if briefing a portfolio manager. Be specific with numbers. Use the actual outcome data to validate or contradict the initial market narrative.
Keep the analysis between 300-500 words. Be factual and cite specific data points."""


def build_week_context(
    week_start: datetime,
    market_df: pd.DataFrame,
    econ_df: pd.DataFrame | None = None,
) -> dict | None:
    """Build context for a single week including news + market data + outcomes."""
    week_end = week_start + timedelta(days=6)
    next_week_start = week_start + timedelta(days=7)
    next_week_end = next_week_start + timedelta(days=6)

    ws = week_start.strftime("%Y-%m-%d")
    we = week_end.strftime("%Y-%m-%d")
    nws = next_week_start.strftime("%Y-%m-%d")
    nwe = next_week_end.strftime("%Y-%m-%d")

    market_summary = get_weekly_summary(market_df, ws, we)
    outcome_summary = get_weekly_summary(market_df, nws, nwe)

    if not market_summary or not outcome_summary:
        return None

    wiki_events = load_date_range(ws, we)
    gdelt_articles = load_week(week_start)

    econ_snapshot = {}
    if econ_df is not None and not econ_df.empty:
        econ_snapshot = get_snapshot(econ_df, ws)

    news_text = _format_news(wiki_events, gdelt_articles, ws, we)
    market_text = format_market_snapshot(market_summary)
    outcome_text = format_market_snapshot(outcome_summary)
    econ_text = format_economic_snapshot(econ_snapshot) if econ_snapshot else ""

    return {
        "week_start": ws,
        "week_end": we,
        "news_text": news_text,
        "market_text": market_text,
        "outcome_text": outcome_text,
        "econ_text": econ_text,
        "news_count": len(wiki_events) + len(gdelt_articles),
        "market_summary": market_summary,
        "outcome_summary": outcome_summary,
    }


def _format_news(
    wiki_events: list[dict],
    gdelt_articles: list[dict],
    start: str,
    end: str,
) -> str:
    """Combine and format news from multiple sources."""
    lines = [f"NEWS EVENTS ({start} to {end}):"]

    if not wiki_events and not gdelt_articles:
        lines.append("  (Market data only — no news events collected for this period)")
        return "\n".join(lines)

    for ev in wiki_events[:30]:
        date = ev.get("date", "")
        text = ev.get("text", "")
        cats = ", ".join(ev.get("categories", []))
        lines.append(f"  [{date}] [{cats}] {text}")

    gdelt_by_theme = {}
    for art in gdelt_articles[:50]:
        theme = art.get("theme", "general")
        if theme not in gdelt_by_theme:
            gdelt_by_theme[theme] = []
        gdelt_by_theme[theme].append(art)

    for theme, articles in gdelt_by_theme.items():
        for art in articles[:8]:
            title = art.get("title", "").strip()
            source = art.get("source", "")
            date = art.get("date", "")[:10]
            if title:
                lines.append(f"  [{date}] [{source}] {title}")

    return "\n".join(lines)


def generate_analysis(context: dict, client: OpenAI) -> str | None:
    """Use GPT-4o-mini to generate expert analysis from historical data."""
    user_msg = f"""Analyze this week in markets:

{context['news_text']}

=== MARKET DATA (This Week) ===
{context['market_text']}

{context['econ_text']}

=== ACTUAL OUTCOMES (Following Week) ===
{context['outcome_text']}

Write your expert analysis of what happened and why, using the actual outcome data.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ANALYST_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT analysis generation failed: {e}")
        return None


def generate_analysis_local(context: dict) -> str:
    """Generate analysis without LLM using templates and actual data.

    Fallback when OpenAI API is unavailable.
    """
    market = context["market_summary"]
    outcome = context["outcome_summary"]
    ws = context["week_start"]

    biggest_movers = sorted(
        market.items(),
        key=lambda x: abs(x[1].get("change_pct", 0)),
        reverse=True,
    )[:5]

    biggest_outcomes = sorted(
        outcome.items(),
        key=lambda x: abs(x[1].get("change_pct", 0)),
        reverse=True,
    )[:5]

    lines = [f"**Market Analysis for week of {ws}:**\n"]
    lines.append("**This Week's Biggest Moves:**")
    for name, data in biggest_movers:
        direction = "gained" if data["change_pct"] > 0 else "declined"
        lines.append(
            f"- {name} {direction} {abs(data['change_pct']):.1f}% "
            f"(closed at {data['close']})"
        )

    lines.append("\n**Following Week Outcomes:**")
    for name, data in biggest_outcomes:
        direction = "rose" if data["change_pct"] > 0 else "fell"
        lines.append(
            f"- {name} {direction} {abs(data['change_pct']):.1f}% "
            f"(closed at {data['close']})"
        )

    sp500_now = market.get("SP500", {}).get("change_pct", 0)
    sp500_next = outcome.get("SP500", {}).get("change_pct", 0)
    gold_now = market.get("Gold", {}).get("change_pct", 0)
    gold_next = outcome.get("Gold", {}).get("change_pct", 0)

    lines.append("\n**Cross-Asset Analysis:**")
    if sp500_now * gold_now < 0:
        lines.append("- Risk-off pattern: stocks and gold moved in opposite directions")
    if abs(sp500_next) > 2:
        lines.append(
            f"- Significant S&P 500 move of {sp500_next:+.1f}% the following week "
            "suggests this week's events had lasting impact"
        )

    lines.append(
        "\n**Key Takeaway:** "
        f"Markets moved {abs(sp500_now):.1f}% this week and "
        f"{abs(sp500_next):.1f}% the next, "
        f"{'continuing' if sp500_now * sp500_next > 0 else 'reversing'} the trend."
    )

    return "\n".join(lines)


def build_training_pair(context: dict, analysis: str) -> dict:
    """Create a single training example in instruction/input/output format."""
    instruction = (
        "Analyze the following news events and market data. "
        "Identify the key market-moving events, explain their impact, "
        "and describe what happened to markets in the following period."
    )

    input_text = f"""=== WEEK OF {context['week_start']} ===

{context['news_text']}

=== MARKET DATA ===
{context['market_text']}

{context['econ_text']}"""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": analysis,
    }


def build_prediction_pair(context: dict) -> dict:
    """Create a prediction-style training example.

    Input: news + market data
    Output: what markets did next (for the model to learn prediction patterns)
    """
    instruction = (
        "Based on the following news and market data, predict what will happen "
        "to key markets over the next week. Provide specific directional calls "
        "with reasoning."
    )

    input_text = f"""=== CURRENT DATE: {context['week_start']} ===

{context['news_text']}

=== CURRENT MARKET DATA ===
{context['market_text']}

{context['econ_text']}"""

    outcome = context["outcome_summary"]
    output_lines = ["**Market Predictions (Next Week):**\n"]

    for name, data in sorted(outcome.items()):
        direction = "BULLISH" if data["change_pct"] > 0.5 else (
            "BEARISH" if data["change_pct"] < -0.5 else "NEUTRAL"
        )
        output_lines.append(
            f"- {name}: {direction} — moved {data['change_pct']:+.1f}% "
            f"to {data['close']}"
        )

    sp = outcome.get("SP500", {})
    gold = outcome.get("Gold", {})
    oil = outcome.get("CrudeOil_WTI", {})

    output_lines.append("\n**Key Drivers:**")
    if sp:
        output_lines.append(
            f"- Equities: S&P 500 {'gained' if sp.get('change_pct', 0) > 0 else 'lost'} "
            f"{abs(sp.get('change_pct', 0)):.1f}%"
        )
    if gold:
        output_lines.append(
            f"- Safe havens: Gold {'rose' if gold.get('change_pct', 0) > 0 else 'fell'} "
            f"{abs(gold.get('change_pct', 0)):.1f}%"
        )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": "\n".join(output_lines),
    }


def build_dataset(
    start_date: str = "2016-01-01",
    end_date: str = "2026-02-01",
    use_gpt: bool = True,
    output_dir: Path | None = None,
) -> int:
    """Build the complete historical training dataset.

    Args:
        start_date: First week to include
        end_date: Last week to include
        use_gpt: If True, use GPT-4o-mini for high-quality analyses
        output_dir: Where to save the JSONL file
    """
    out = output_dir or TRAINING_DIR
    out.mkdir(parents=True, exist_ok=True)

    market_csv = DATA_DIR / "market" / "daily_prices.csv"
    econ_csv = DATA_DIR / "market" / "economic_indicators.csv"

    if not market_csv.exists():
        logger.error(f"Market data not found at {market_csv}. Run collectors first.")
        return 0

    logger.info("Loading market data...")
    market_df = pd.read_csv(market_csv)
    market_df["Date"] = pd.to_datetime(market_df["Date"])

    econ_df = None
    if econ_csv.exists():
        logger.info("Loading economic indicators...")
        econ_df = pd.read_csv(econ_csv)
        econ_df["Date"] = pd.to_datetime(econ_df["Date"])

    openai_client = None
    if use_gpt:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            logger.info("Using GPT-4o-mini for analysis generation")
        else:
            logger.warning("No OPENAI_API_KEY, falling back to template analysis")
            use_gpt = False

    pairs_file = out / "historical_pairs.jsonl"
    prediction_file = out / "prediction_pairs.jsonl"

    existing_weeks = set()
    if pairs_file.exists():
        with open(pairs_file) as f:
            for line in f:
                data = json.loads(line)
                week = data.get("metadata", {}).get("week_start", "")
                if week:
                    existing_weeks.add(week)
        logger.info(f"Resuming: {len(existing_weeks)} weeks already processed")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    total = 0
    skipped = 0

    with open(pairs_file, "a") as pf, open(prediction_file, "a") as predf:
        while current < end:
            ws = current.strftime("%Y-%m-%d")

            if ws in existing_weeks:
                current += timedelta(days=7)
                skipped += 1
                continue

            context = build_week_context(current, market_df, econ_df)
            if context is None:
                logger.debug(f"No data for week of {ws}")
                current += timedelta(days=7)
                continue

            if use_gpt and openai_client:
                analysis = generate_analysis(context, openai_client)
                if not analysis:
                    analysis = generate_analysis_local(context)
                time.sleep(0.5)  # rate limit
            else:
                analysis = generate_analysis_local(context)

            analysis_pair = build_training_pair(context, analysis)
            analysis_pair["metadata"] = {
                "week_start": ws,
                "news_count": context["news_count"],
                "type": "historical_analysis",
            }
            pf.write(json.dumps(analysis_pair) + "\n")

            pred_pair = build_prediction_pair(context)
            pred_pair["metadata"] = {
                "week_start": ws,
                "type": "prediction",
            }
            predf.write(json.dumps(pred_pair) + "\n")

            total += 2
            if total % 20 == 0:
                logger.info(f"Generated {total} pairs ({skipped} skipped)")
                pf.flush()
                predf.flush()

            current += timedelta(days=7)

    logger.info(
        f"Dataset build complete: {total} new pairs, "
        f"{skipped} skipped, saved to {out}"
    )
    return total


def combine_datasets(output_dir: Path | None = None) -> Path:
    """Combine historical pairs with existing financial QA into final dataset."""
    out = output_dir or TRAINING_DIR
    combined_path = out / "combined_dataset.jsonl"

    all_pairs = []

    historical = out / "historical_pairs.jsonl"
    if historical.exists():
        with open(historical) as f:
            for line in f:
                data = json.loads(line)
                data.pop("metadata", None)
                all_pairs.append(data)
        logger.info(f"Loaded {len(all_pairs)} historical pairs")

    predictions = out / "prediction_pairs.jsonl"
    pred_count = 0
    if predictions.exists():
        with open(predictions) as f:
            for line in f:
                data = json.loads(line)
                data.pop("metadata", None)
                all_pairs.append(data)
                pred_count += 1
        logger.info(f"Loaded {pred_count} prediction pairs")

    with open(combined_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Combined dataset: {len(all_pairs)} total pairs → {combined_path}")
    return combined_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()

    count = build_dataset("2020-01-01", "2024-12-31", use_gpt=True)
    print(f"\nGenerated {count} training pairs")

    combined = combine_datasets()
    print(f"Combined dataset: {combined}")

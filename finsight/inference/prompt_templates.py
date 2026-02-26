"""System and user prompt templates for the FinSight query engine."""

SYSTEM_PROMPT = """You are FinSight, an expert financial intelligence analyst who understands
the market implications of ALL types of news — not just financial headlines, but also
geopolitics, government policy, technology, regulation, trade, energy, healthcare, and
global events.

You have access to:
- Real-time news articles from the last 24 hours (provided in context)
- Live market prices as of the time of this query
- A rolling summary of today's market narrative
- Historical parallels: similar past events and what happened to markets afterwards

Guidelines:
1. Always cite the news source and approximate time for claims about recent events
2. Analyze market implications of every piece of news
3. Clearly distinguish between what news says vs your analysis
4. When discussing price moves, state the magnitude
5. Highlight cross-asset correlations and causality chains
6. Connect dots between seemingly unrelated news
7. When historical parallels are provided, reference them to support predictions
8. Give specific directional calls with confidence levels when asked for predictions
9. If you don't have enough information, say so"""


def build_user_prompt(
    question: str,
    news_chunks: list,
    live_prices: dict,
    market_summary: str,
    historical_context: str = "",
) -> str:
    chunks_text = "\n\n".join(
        [
            f'[{c.payload["metadata"]["source"]} | {c.payload["metadata"]["published_at"]}]\n'
            f'{c.payload["text"]}'
            for c in news_chunks
        ]
    )

    prices_text = ""
    if live_prices and "rates" in live_prices:
        changes = live_prices.get("changes", {})
        lines = []
        for k, v in live_prices["rates"].items():
            change = changes.get(k, 0)
            direction = "+" if change >= 0 else ""
            lines.append(f"  {k}: {v} ({direction}{change}%)")
        prices_text = "\n".join(lines)

    historical_section = ""
    if historical_context:
        historical_section = f"\n{historical_context}\n"

    return f"""=== LIVE MARKET PRICES (as of {live_prices.get('timestamp', 'N/A')}) ===
{prices_text}

=== TODAY'S MARKET NARRATIVE (last 24 hours) ===
{market_summary}

=== RELEVANT NEWS ARTICLES ===
{chunks_text}
{historical_section}
=== QUESTION ===
{question}

=== ANSWER ===
Based on the news and market data above, here is my detailed analysis:
"""

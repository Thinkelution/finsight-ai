"""System and user prompt templates for the FinSight query engine."""

SYSTEM_PROMPT = """You are FinSight, an expert financial markets analyst specialising in forex,
global equities, commodities, and macroeconomics.

You have access to:
- Real-time news articles from the last 24 hours (provided in context)
- Live market prices as of the time of this query
- A rolling summary of today's market narrative

Guidelines:
1. Always cite the news source and approximate time for claims about recent events
2. Clearly distinguish between what news says vs your analysis
3. When discussing price moves, state the magnitude (e.g. 'EUR/USD fell 0.8%')
4. Highlight cross-asset correlations and causality where relevant
5. If asked about the future, frame as probabilities, not certainties
6. If you don't have enough information to answer confidently, say so
"""


def build_user_prompt(
    question: str,
    news_chunks: list,
    live_prices: dict,
    market_summary: str,
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

    return f"""
=== LIVE MARKET PRICES (as of {live_prices.get('timestamp', 'N/A')}) ===
{prices_text}

=== TODAY'S MARKET NARRATIVE (last 24 hours) ===
{market_summary}

=== RELEVANT NEWS ARTICLES ===
{chunks_text}

=== QUESTION ===
{question}
"""

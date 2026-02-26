"""Rolling 24-hour market summary that refreshes every 30 minutes."""

from datetime import datetime

from redis import Redis

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

SUMMARY_KEY = "finsight:market_summary"
SUMMARY_TTL = 3600  # 1 hour fallback


class MarketSummariser:
    def __init__(self, redis_client: Redis | None = None):
        try:
            self.redis = redis_client or Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            self.redis.ping()
            self._use_redis = True
        except Exception:
            logger.warning("redis_unavailable_for_summariser")
            self._use_redis = False
            self._memory_summary = ""
            self._memory_updated = None

    def get_rolling_summary(self) -> str:
        """Get the current 24h market narrative summary."""
        if self._use_redis:
            summary = self.redis.get(SUMMARY_KEY)
            return summary or "No market summary available yet."
        return self._memory_summary or "No market summary available yet."

    def update_summary(self, summary_text: str) -> None:
        """Store an updated market summary."""
        timestamp = datetime.utcnow().isoformat()
        full = f"[Updated: {timestamp}]\n{summary_text}"

        if self._use_redis:
            self.redis.setex(SUMMARY_KEY, SUMMARY_TTL, full)
        else:
            self._memory_summary = full
            self._memory_updated = timestamp

        logger.info("summary_updated", length=len(summary_text))

    def generate_summary(self, recent_chunks: list[dict], live_prices: dict) -> str:
        """Generate a 24h market narrative from recent chunks and prices.

        Uses Ollama to synthesize a summary if available, otherwise
        builds a simple concatenation.
        """
        if not recent_chunks:
            return "Markets are quiet. No significant news in the last 24 hours."

        news_snippets = []
        for chunk in recent_chunks[:15]:
            source = chunk.get("metadata", {}).get("source", "unknown")
            title = chunk.get("metadata", {}).get("title", "")
            sentiment = chunk.get("metadata", {}).get("sentiment", {}).get("label", "neutral")
            text_preview = chunk.get("text", "")[:200]
            news_snippets.append(f"[{source}] ({sentiment}) {title}: {text_preview}")

        price_lines = []
        if live_prices and "rates" in live_prices:
            changes = live_prices.get("changes", {})
            for symbol, price in list(live_prices["rates"].items())[:20]:
                change = changes.get(symbol, 0)
                direction = "+" if change >= 0 else ""
                price_lines.append(f"  {symbol}: {price} ({direction}{change}%)")

        try:
            import ollama as ollama_client
            prompt = (
                "Summarize the following market news and price data into a concise "
                "24-hour market narrative (3-5 paragraphs). Focus on major moves, "
                "themes, and cross-asset correlations.\n\n"
                "NEWS:\n" + "\n".join(news_snippets) + "\n\n"
                "PRICES:\n" + "\n".join(price_lines)
            )
            response = ollama_client.chat(
                model=settings.ollama_llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_ctx": 4096},
            )
            summary = response["message"]["content"]
        except Exception as e:
            logger.warning("llm_summary_failed_using_fallback", error=str(e))
            summary = self._build_simple_summary(news_snippets, price_lines)

        self.update_summary(summary)
        return summary

    @staticmethod
    def _build_simple_summary(news_snippets: list[str], price_lines: list[str]) -> str:
        parts = ["24-Hour Market Summary\n"]
        if price_lines:
            parts.append("Key Prices:")
            parts.extend(price_lines[:10])
            parts.append("")
        if news_snippets:
            parts.append("Top Stories:")
            parts.extend(news_snippets[:5])
        return "\n".join(parts)

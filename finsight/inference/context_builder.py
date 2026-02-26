"""Assemble live prices, news chunks, summary, and historical parallels into query context."""

from finsight.config.logging import get_logger
from finsight.ingestion.market_data import MarketDataFetcher
from finsight.storage.summariser import MarketSummariser

logger = get_logger(__name__)


class ContextBuilder:
    def __init__(self):
        self._market = MarketDataFetcher()
        self._summariser = MarketSummariser()
        self._cached_prices: dict | None = None
        self._pattern_matcher = None

    def _get_pattern_matcher(self):
        if self._pattern_matcher is None:
            try:
                from finsight.historical.pattern_matcher import get_historical_context_for_prompt
                self._pattern_matcher = get_historical_context_for_prompt
            except Exception as e:
                logger.warning("pattern_matcher_unavailable", error=str(e))
                self._pattern_matcher = lambda x, **kw: ""
        return self._pattern_matcher

    def get_live_prices(self) -> dict:
        try:
            self._cached_prices = self._market.get_live_prices()
            return self._cached_prices
        except Exception as e:
            logger.error("live_prices_failed", error=str(e))
            return self._cached_prices or {"rates": {}, "changes": {}, "timestamp": "N/A"}

    def get_market_summary(self) -> str:
        return self._summariser.get_rolling_summary()

    def get_historical_context(self, news_text: str) -> str:
        """Find historical parallels for the current news context."""
        try:
            fn = self._get_pattern_matcher()
            return fn(news_text, top_k=3)
        except Exception as e:
            logger.warning("historical_context_failed", error=str(e))
            return ""

    def build_context(self, news_chunks: list) -> dict:
        """Build the full context dict for a query, including historical parallels."""
        live_prices = self.get_live_prices()
        market_summary = self.get_market_summary()

        source_urls = []
        news_texts = []
        for chunk in news_chunks:
            url = chunk.payload.get("metadata", {}).get("url", "")
            if url and url not in source_urls:
                source_urls.append(url)
            news_texts.append(chunk.payload.get("text", "")[:200])

        combined_news = " ".join(news_texts[:5])
        historical_context = self.get_historical_context(combined_news) if combined_news else ""

        return {
            "news_chunks": news_chunks,
            "live_prices": live_prices,
            "market_summary": market_summary,
            "source_urls": source_urls,
            "historical_context": historical_context,
        }

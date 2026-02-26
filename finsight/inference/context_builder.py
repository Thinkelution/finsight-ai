"""Assemble live prices, news chunks, and summary into query context."""

from finsight.config.logging import get_logger
from finsight.ingestion.market_data import MarketDataFetcher
from finsight.storage.summariser import MarketSummariser

logger = get_logger(__name__)


class ContextBuilder:
    def __init__(self):
        self._market = MarketDataFetcher()
        self._summariser = MarketSummariser()
        self._cached_prices: dict | None = None

    def get_live_prices(self) -> dict:
        try:
            self._cached_prices = self._market.get_live_prices()
            return self._cached_prices
        except Exception as e:
            logger.error("live_prices_failed", error=str(e))
            return self._cached_prices or {"rates": {}, "changes": {}, "timestamp": "N/A"}

    def get_market_summary(self) -> str:
        return self._summariser.get_rolling_summary()

    def build_context(self, news_chunks: list) -> dict:
        """Build the full context dict for a query."""
        live_prices = self.get_live_prices()
        market_summary = self.get_market_summary()

        source_urls = []
        for chunk in news_chunks:
            url = chunk.payload.get("metadata", {}).get("url", "")
            if url and url not in source_urls:
                source_urls.append(url)

        return {
            "news_chunks": news_chunks,
            "live_prices": live_prices,
            "market_summary": market_summary,
            "source_urls": source_urls,
        }

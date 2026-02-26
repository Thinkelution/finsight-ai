"""Live market data fetcher using yfinance (free tier)."""

from datetime import datetime
from pathlib import Path

import yaml
import yfinance as yf

from finsight.config.logging import get_logger

logger = get_logger(__name__)

SOURCES_PATH = Path(__file__).parent / "sources.yaml"


class MarketDataFetcher:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with open(SOURCES_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("market_data", {})

    def get_live_prices(self) -> dict:
        """Fetch current prices for all configured symbols."""
        all_symbols = []
        all_symbols.extend(self.config.get("forex_pairs", []))
        all_symbols.extend(self.config.get("indices", []))
        all_symbols.extend(self.config.get("commodities", []))
        all_symbols.extend(self.config.get("crypto", []))

        logger.info("fetching_market_data", symbols_count=len(all_symbols))

        rates = {}
        changes = {}
        errors = []

        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                price = getattr(info, "last_price", None)
                prev_close = getattr(info, "previous_close", None)

                if price is not None:
                    rates[symbol] = round(price, 4)
                    if prev_close and prev_close > 0:
                        pct = ((price - prev_close) / prev_close) * 100
                        changes[symbol] = round(pct, 2)
            except Exception as e:
                errors.append(symbol)
                logger.warning("price_fetch_failed", symbol=symbol, error=str(e))

        if errors:
            logger.warning("price_fetch_errors", failed=errors)

        return {
            "rates": rates,
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat(),
            "symbols_fetched": len(rates),
            "symbols_failed": len(errors),
        }

    def get_forex_rates(self) -> dict:
        symbols = self.config.get("forex_pairs", [])
        return self._fetch_batch(symbols)

    def get_index_levels(self) -> dict:
        symbols = self.config.get("indices", [])
        return self._fetch_batch(symbols)

    def get_commodity_prices(self) -> dict:
        symbols = self.config.get("commodities", [])
        return self._fetch_batch(symbols)

    def get_crypto_prices(self) -> dict:
        symbols = self.config.get("crypto", [])
        return self._fetch_batch(symbols)

    def _fetch_batch(self, symbols: list[str]) -> dict:
        rates = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                price = getattr(info, "last_price", None)
                if price is not None:
                    rates[symbol] = round(price, 4)
            except Exception as e:
                logger.warning("batch_price_failed", symbol=symbol, error=str(e))
        return {
            "rates": rates,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_price_history(self, symbol: str, period: str = "1d", interval: str = "5m") -> list[dict]:
        """Get intraday price history for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            return [
                {
                    "timestamp": idx.isoformat(),
                    "open": round(row["Open"], 4),
                    "high": round(row["High"], 4),
                    "low": round(row["Low"], 4),
                    "close": round(row["Close"], 4),
                    "volume": int(row["Volume"]),
                }
                for idx, row in hist.iterrows()
            ]
        except Exception as e:
            logger.error("history_fetch_failed", symbol=symbol, error=str(e))
            return []

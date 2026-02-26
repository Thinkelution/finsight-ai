"""GET /market/live endpoint for current market prices."""

from fastapi import APIRouter

from finsight.api.schemas import MarketPricesResponse
from finsight.config.logging import get_logger
from finsight.ingestion.market_data import MarketDataFetcher

logger = get_logger(__name__)

router = APIRouter()

_fetcher: MarketDataFetcher | None = None


def _get_fetcher() -> MarketDataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = MarketDataFetcher()
    return _fetcher


@router.get("/market/live", response_model=MarketPricesResponse)
async def live_prices():
    fetcher = _get_fetcher()
    data = fetcher.get_live_prices()
    return MarketPricesResponse(**data)


@router.get("/market/forex")
async def forex_prices():
    fetcher = _get_fetcher()
    return fetcher.get_forex_rates()


@router.get("/market/indices")
async def index_levels():
    fetcher = _get_fetcher()
    return fetcher.get_index_levels()


@router.get("/market/commodities")
async def commodity_prices():
    fetcher = _get_fetcher()
    return fetcher.get_commodity_prices()


@router.get("/market/crypto")
async def crypto_prices():
    fetcher = _get_fetcher()
    return fetcher.get_crypto_prices()


@router.get("/market/history/{symbol}")
async def price_history(symbol: str, period: str = "1d", interval: str = "5m"):
    fetcher = _get_fetcher()
    return fetcher.get_price_history(symbol, period=period, interval=interval)

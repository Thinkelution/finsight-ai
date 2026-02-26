"""GET/POST /predictions endpoints for trend predictions based on historical patterns."""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/predictions")


class PredictionRequest(BaseModel):
    context: str = ""
    top_parallels: int = 5


@router.get("")
async def get_predictions():
    """Generate trend predictions based on current news + historical patterns."""
    try:
        from finsight.storage.qdrant_store import get_qdrant_client
        from finsight.historical.trend_predictor import predict_trends

        client = get_qdrant_client()
        results = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=20,
            with_payload=True,
            with_vectors=False,
        )

        news_texts = []
        for point in results[0]:
            payload = point.payload or {}
            text = payload.get("text", "")
            title = payload.get("metadata", {}).get("title", "")
            if title:
                news_texts.append(title)
            elif text:
                news_texts.append(text[:200])

        if not news_texts:
            return {
                "predictions": [],
                "parallels": [],
                "prediction_text": "No recent news available for prediction.",
                "confidence": 0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        current_context = "\n".join(news_texts[:15])

        market_data = None
        try:
            from finsight.ingestion.market_data import MarketDataFetcher
            fetcher = MarketDataFetcher()
            market_data = fetcher.get_live_prices()
        except Exception:
            pass

        result = predict_trends(current_context, market_data)
        return result

    except Exception as e:
        logger.error("prediction_endpoint_error", error=str(e))
        return {
            "predictions": [],
            "parallels": [],
            "prediction_text": f"Prediction engine error: {str(e)}",
            "confidence": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


@router.post("")
async def post_predictions(req: PredictionRequest):
    """Generate predictions for a specific context."""
    try:
        from finsight.historical.trend_predictor import predict_trends

        if not req.context:
            return {
                "predictions": [],
                "parallels": [],
                "prediction_text": "Please provide context for prediction.",
                "confidence": 0,
            }

        result = predict_trends(
            req.context,
            top_parallels=req.top_parallels,
        )
        return result

    except Exception as e:
        logger.error("prediction_post_error", error=str(e))
        return {
            "predictions": [],
            "parallels": [],
            "prediction_text": f"Error: {str(e)}",
            "confidence": 0,
            "error": str(e),
        }


@router.get("/parallels")
async def get_parallels(query: str = "", limit: int = 5):
    """Search historical parallels for a given query."""
    try:
        from finsight.historical.pattern_matcher import find_similar_events

        if not query:
            return {"parallels": [], "query": query}

        parallels = find_similar_events(query, top_k=limit)
        return {"parallels": parallels, "query": query}

    except Exception as e:
        logger.error("parallels_endpoint_error", error=str(e))
        return {"parallels": [], "query": query, "error": str(e)}


@router.get("/status")
async def get_historical_status():
    """Return status of the historical data pipeline."""
    from pathlib import Path

    data_dir = Path("data/historical")
    status = {
        "market_data": False,
        "economic_data": False,
        "wikipedia_events": False,
        "gdelt_articles": False,
        "training_pairs": 0,
        "indexed_patterns": 0,
    }

    market_csv = data_dir / "market" / "daily_prices.csv"
    if market_csv.exists():
        status["market_data"] = True
        try:
            import pandas as pd
            df = pd.read_csv(market_csv)
            status["market_data_rows"] = len(df)
            status["market_date_range"] = f"{df['Date'].min()} to {df['Date'].max()}"
        except Exception:
            pass

    econ_csv = data_dir / "market" / "economic_indicators.csv"
    if econ_csv.exists():
        status["economic_data"] = True

    wiki_dir = data_dir / "news" / "wikipedia"
    if wiki_dir.exists():
        wiki_files = list(wiki_dir.glob("*.jsonl"))
        status["wikipedia_events"] = len(wiki_files) > 0
        status["wikipedia_months"] = len(wiki_files)

    gdelt_dir = data_dir / "news" / "gdelt"
    if gdelt_dir.exists():
        gdelt_files = list(gdelt_dir.glob("*.jsonl"))
        status["gdelt_articles"] = len(gdelt_files) > 0
        status["gdelt_weeks"] = len(gdelt_files)

    pairs_file = data_dir / "training" / "historical_pairs.jsonl"
    if pairs_file.exists():
        status["training_pairs"] = sum(1 for _ in open(pairs_file))

    try:
        from finsight.historical.pattern_matcher import get_qdrant_client, COLLECTION
        client = get_qdrant_client()
        count = client.count(collection_name=COLLECTION).count
        status["indexed_patterns"] = count
    except Exception:
        pass

    return status

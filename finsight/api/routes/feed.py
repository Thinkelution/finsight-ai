"""GET /data endpoints for dashboard: recent news, ingestion stats, analysis results."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/data")


@router.get("/feed")
async def get_news_feed(limit: int = 50, hours_back: int = 24, category: str = "all"):
    """Return recently ingested news chunks for the dashboard.

    category: 'all', 'finance', 'geopolitical', 'tech', 'world'
    """
    try:
        from finsight.storage.qdrant_store import get_qdrant_client

        client = get_qdrant_client()
        results = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=min(limit * 3, 200),
            with_payload=True,
            with_vectors=False,
        )

        items = []
        for point in results[0]:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            asset_classes = meta.get("asset_classes", [])
            geo_tags = meta.get("geopolitical_tags", [])

            if category == "finance" and not any(
                ac in asset_classes for ac in ["equities", "forex", "commodities"]
            ):
                continue
            elif category == "geopolitical" and not geo_tags and "geopolitical" not in asset_classes:
                continue
            elif category == "tech" and "google_news_technology" not in meta.get("source", ""):
                if not any(t in (meta.get("title", "") or "").lower() for t in ["ai ", "tech", "chip", "software", "apple", "google", "nvidia"]):
                    continue

            items.append({
                "id": str(point.id),
                "text": (payload.get("text", ""))[:500],
                "title": meta.get("title", "Untitled"),
                "source": meta.get("source", "unknown"),
                "url": meta.get("url", ""),
                "published_at": meta.get("published_at", ""),
                "sentiment_score": meta.get("sentiment_score", 0),
                "sentiment_label": meta.get("sentiment_label", "neutral"),
                "entities": meta.get("entities", []),
                "geopolitical_tags": geo_tags,
                "asset_classes": asset_classes,
            })

            if len(items) >= limit:
                break

        items.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return {"items": items, "total": len(items)}

    except Exception as e:
        logger.error("feed_endpoint_error", error=str(e))
        return {"items": [], "total": 0, "error": str(e)}


@router.get("/stats")
async def get_pipeline_stats():
    """Return ingestion statistics."""
    try:
        from finsight.storage.qdrant_store import get_qdrant_client

        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection)

        return {
            "total_chunks": info.points_count,
            "collection": settings.qdrant_collection,
            "llm_model": settings.ollama_llm_model,
            "embed_model": settings.ollama_embed_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "collection": settings.qdrant_collection,
            "llm_model": settings.ollama_llm_model,
            "embed_model": settings.ollama_embed_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


@router.get("/analysis-history")
async def get_analysis_history():
    """Return recent analysis queries from Redis."""
    try:
        from redis import Redis
        r = Redis.from_url(settings.redis_url)
        keys = r.keys("finsight:analysis:*")
        history = []
        for key in keys[:20]:
            data = r.hgetall(key)
            if data:
                history.append({
                    k.decode(): v.decode() for k, v in data.items()
                })
        return {"history": history}
    except Exception:
        return {"history": []}

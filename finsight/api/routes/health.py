"""GET /health endpoint for service health checks."""

from fastapi import APIRouter

from finsight.api.schemas import HealthResponse
from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    status = "healthy"
    qdrant_status = None
    redis_status = None
    ollama_status = None
    chunks_count = None

    try:
        from finsight.storage.qdrant_store import get_qdrant_client
        client = get_qdrant_client()
        info = client.get_collection(settings.qdrant_collection)
        qdrant_status = "connected"
        chunks_count = info.points_count
    except Exception as e:
        qdrant_status = f"error: {str(e)[:100]}"
        status = "degraded"

    try:
        from redis import Redis
        r = Redis.from_url(settings.redis_url)
        r.ping()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)[:100]}"
        status = "degraded"

    try:
        import ollama as ollama_client
        ollama_client.list()
        ollama_status = "connected"
    except Exception as e:
        ollama_status = f"error: {str(e)[:100]}"
        status = "degraded"

    return HealthResponse(
        status=status,
        qdrant_status=qdrant_status,
        redis_status=redis_status,
        ollama_status=ollama_status,
        chunks_count=chunks_count,
    )

"""Insert processed chunks with embeddings into Qdrant."""

import uuid

from qdrant_client.models import PointStruct

from finsight.config.logging import get_logger
from finsight.config.settings import settings
from finsight.storage.qdrant_store import get_qdrant_client

logger = get_logger(__name__)

BATCH_SIZE = 100


def index_chunks(payloads: list[dict], client=None) -> int:
    """Insert chunk payloads into Qdrant.

    Each payload must have 'embedding' (list[float]) and 'metadata' (dict).
    Returns number of points inserted.
    """
    if not payloads:
        return 0

    client = client or get_qdrant_client()
    collection = settings.qdrant_collection

    points = []
    for payload in payloads:
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=payload["embedding"],
                payload={
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                },
            )
        )

    inserted = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)
        inserted += len(batch)
        logger.info("indexed_batch", batch_size=len(batch), total=inserted)

    logger.info("indexing_complete", total_points=inserted)
    return inserted


def delete_expired_chunks(client=None, max_age_days: int | None = None) -> int:
    """Delete chunks older than max_age_days from Qdrant."""
    from datetime import datetime, timedelta

    client = client or get_qdrant_client()
    max_age = max_age_days or settings.news_expiry_days
    cutoff = (datetime.utcnow() - timedelta(days=max_age)).isoformat()

    from qdrant_client.models import Filter, FieldCondition, Range

    result = client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.published_at",
                    range=Range(lt=cutoff),
                )
            ]
        ),
    )

    logger.info("expired_chunks_deleted", cutoff=cutoff)
    return 0  # Qdrant delete doesn't return count directly

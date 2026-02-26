"""Qdrant connection management and collection setup."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def ensure_collection(client: QdrantClient | None = None) -> QdrantClient:
    """Create the finsight_chunks collection if it doesn't exist."""
    client = client or get_qdrant_client()
    collection = settings.qdrant_collection

    collections = [c.name for c in client.get_collections().collections]
    if collection in collections:
        logger.info("collection_exists", collection=collection)
        return client

    logger.info("creating_collection", collection=collection, dim=settings.embed_dim)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=settings.embed_dim,
            distance=Distance.COSINE,
        ),
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="metadata.published_at",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="metadata.asset_classes",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="metadata.source",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    logger.info("collection_created", collection=collection)
    return client

"""Qdrant connection management and collection setup."""

import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

_qdrant_client: QdrantClient | None = None

QDRANT_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "qdrant_storage",
)


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client, preferring the HTTP server on localhost:6333."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        _qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=5,
        )
        _qdrant_client.get_collections()
        logger.info("qdrant_server_connected", host=settings.qdrant_host, port=settings.qdrant_port)
    except Exception as e:
        logger.warning("qdrant_server_unavailable", error=str(e))
        logger.info("qdrant_using_local_storage", path=QDRANT_LOCAL_PATH)
        os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)

    return _qdrant_client


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

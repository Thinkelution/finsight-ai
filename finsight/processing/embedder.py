"""Embed text chunks via Ollama's nomic-embed-text model."""

import ollama as ollama_client

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)


def embed_text(text: str) -> list[float]:
    """Embed a single text string, returning a 768-dim vector."""
    try:
        response = ollama_client.embeddings(
            model=settings.ollama_embed_model,
            prompt=text,
        )
        return response["embedding"]
    except Exception as e:
        logger.error("embedding_failed", error=str(e), text_len=len(text))
        raise


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    embeddings = []
    for text in texts:
        emb = embed_text(text)
        embeddings.append(emb)
    return embeddings

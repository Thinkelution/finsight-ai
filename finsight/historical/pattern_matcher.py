"""Find historical parallels for current news events.

Embeds historical event summaries and searches for similar
past events when new news arrives. Uses Qdrant for vector search
with a dedicated 'historical_patterns' collection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    Range,
)

from finsight.config.settings import settings

logger = logging.getLogger(__name__)

COLLECTION = "historical_patterns"
DATA_DIR = Path("data/historical")


def get_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        return client
    except Exception:
        return QdrantClient(path="data/qdrant_historical")


def ensure_collection(client: QdrantClient):
    """Create the historical patterns collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {COLLECTION}")


def embed_text(text: str) -> list[float]:
    """Generate embedding using the same model as the main pipeline."""
    response = ollama_client.embed(
        model=settings.ollama_embed_model,
        input=text,
    )
    return response["embeddings"][0]


def index_historical_patterns(
    training_file: Path | None = None,
    batch_size: int = 50,
) -> int:
    """Index historical training pairs into Qdrant for similarity search."""
    tf = training_file or (DATA_DIR / "training" / "historical_pairs.jsonl")
    if not tf.exists():
        logger.error(f"Training file not found: {tf}")
        return 0

    client = get_qdrant_client()
    ensure_collection(client)

    existing = client.count(collection_name=COLLECTION).count
    logger.info(f"Existing patterns in collection: {existing}")

    points = []
    indexed = 0

    with open(tf) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            metadata = data.get("metadata", {})
            week_start = metadata.get("week_start", f"unknown_{i}")

            input_text = data.get("input", "")
            output_text = data.get("output", "")

            summary = input_text[:500] + "\n" + output_text[:300]

            try:
                embedding = embed_text(summary)
            except Exception as e:
                logger.warning(f"Embedding failed for {week_start}: {e}")
                continue

            point = PointStruct(
                id=i + existing,
                vector=embedding,
                payload={
                    "week_start": week_start,
                    "input_text": input_text[:2000],
                    "output_text": output_text[:2000],
                    "news_count": metadata.get("news_count", 0),
                    "type": metadata.get("type", "historical_analysis"),
                    "indexed_at": datetime.now().isoformat(),
                },
            )
            points.append(point)

            if len(points) >= batch_size:
                client.upsert(collection_name=COLLECTION, points=points)
                indexed += len(points)
                points = []
                logger.info(f"Indexed {indexed} patterns")

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        indexed += len(points)

    logger.info(f"Total indexed: {indexed} historical patterns")
    return indexed


def find_similar_events(
    current_context: str,
    top_k: int = 5,
    min_score: float = 0.3,
) -> list[dict]:
    """Find historical events most similar to the current news context."""
    client = get_qdrant_client()
    ensure_collection(client)

    try:
        embedding = embed_text(current_context[:1000])
    except Exception as e:
        logger.error(f"Failed to embed current context: {e}")
        return []

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )
        results = response.points if hasattr(response, "points") else []
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

    parallels = []
    for point in results:
        score = point.score if hasattr(point, "score") else 0
        if score < min_score:
            continue

        payload = point.payload or {}
        parallels.append({
            "week_start": payload.get("week_start", "unknown"),
            "similarity": round(float(score), 3),
            "context": payload.get("input_text", "")[:500],
            "outcome": payload.get("output_text", "")[:500],
            "type": payload.get("type", ""),
        })

    return parallels


def get_historical_context_for_prompt(
    current_news: str,
    top_k: int = 3,
) -> str:
    """Generate a formatted historical parallels section for the LLM prompt."""
    parallels = find_similar_events(current_news, top_k=top_k)

    if not parallels:
        return ""

    lines = ["=== HISTORICAL PARALLELS ==="]
    lines.append("The following past events showed similar patterns:\n")

    for i, p in enumerate(parallels, 1):
        lines.append(f"--- Parallel {i} (Week of {p['week_start']}, "
                     f"{p['similarity']*100:.0f}% match) ---")
        lines.append(f"Context: {p['context'][:300]}...")
        lines.append(f"Outcome: {p['outcome'][:300]}...")
        lines.append("")

    lines.append(
        "Use these historical parallels to inform your analysis of current events. "
        "Note any recurring patterns."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    count = index_historical_patterns()
    print(f"Indexed {count} patterns")

    results = find_similar_events(
        "Federal Reserve raises interest rates, inflation data higher than expected"
    )
    for r in results:
        print(f"\n--- {r['week_start']} ({r['similarity']:.0%} match) ---")
        print(r["outcome"][:200])

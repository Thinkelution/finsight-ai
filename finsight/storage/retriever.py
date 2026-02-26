"""Time-weighted similarity search over Qdrant."""

import math
from datetime import datetime, timedelta

from qdrant_client.models import FieldCondition, Filter, MatchValue

from finsight.config.logging import get_logger
from finsight.config.settings import settings
from finsight.storage.qdrant_store import get_qdrant_client

logger = get_logger(__name__)


class TimeWeightedRetriever:
    def __init__(self, client=None):
        self.client = client or get_qdrant_client()
        self.collection = settings.qdrant_collection

    def retrieve(
        self,
        query_embedding: list[float],
        k: int = 8,
        asset_class: str | None = None,
        hours_back: int = 24,
    ) -> list:
        """Retrieve top-k chunks with time-decay re-ranking.

        Over-fetches 3x then re-ranks by a blend of semantic similarity
        and recency (exponential decay with ~7hr half-life).
        """
        filters = self._build_filters(asset_class)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=filters,
            limit=k * 3,
        )

        if not results:
            logger.info("no_retrieval_results")
            return []

        reranked = sorted(results, key=self._time_score, reverse=True)
        top_k = reranked[:k]

        logger.info(
            "retrieval_complete",
            candidates=len(results),
            returned=len(top_k),
            asset_class=asset_class,
        )
        return top_k

    @staticmethod
    def _build_filters(asset_class: str | None) -> Filter | None:
        if not asset_class:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.asset_classes",
                    match=MatchValue(value=asset_class),
                )
            ]
        )

    @staticmethod
    def _time_score(result) -> float:
        """Blend semantic score (70%) with time-decay score (30%).

        Decay: exp(-0.1 * age_hours), giving a half-life of ~7 hours.
        """
        try:
            pub_time = result.payload.get("metadata", {}).get("published_at", "")
            if pub_time:
                age_hours = (
                    datetime.utcnow() - datetime.fromisoformat(pub_time)
                ).total_seconds() / 3600
            else:
                age_hours = 48.0
        except (ValueError, TypeError):
            age_hours = 48.0

        decay = math.exp(-0.1 * max(age_hours, 0))
        return result.score * 0.7 + decay * 0.3

    def retrieve_by_source(
        self,
        query_embedding: list[float],
        source: str,
        k: int = 5,
    ) -> list:
        """Retrieve chunks filtered by source name."""
        source_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source),
                )
            ]
        )
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            query_filter=source_filter,
            limit=k,
        )

    def get_collection_stats(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.name,
        }

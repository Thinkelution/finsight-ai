"""Tests for the storage/retrieval layer."""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from finsight.storage.retriever import TimeWeightedRetriever


class MockResult:
    def __init__(self, score: float, published_at: str, asset_classes: list[str] | None = None):
        self.score = score
        self.payload = {
            "text": "test chunk",
            "metadata": {
                "published_at": published_at,
                "source": "test",
                "asset_classes": asset_classes or ["equities"],
            },
        }


class TestTimeWeightedRetriever:
    def test_time_score_recent_higher(self):
        now = datetime.utcnow()
        recent = MockResult(score=0.8, published_at=now.isoformat())
        old = MockResult(
            score=0.8,
            published_at=(now - timedelta(hours=24)).isoformat(),
        )

        recent_score = TimeWeightedRetriever._time_score(recent)
        old_score = TimeWeightedRetriever._time_score(old)
        assert recent_score > old_score

    def test_time_score_blend(self):
        now = datetime.utcnow()
        result = MockResult(score=1.0, published_at=now.isoformat())
        score = TimeWeightedRetriever._time_score(result)
        # At age 0: 1.0 * 0.7 + exp(0) * 0.3 = 0.7 + 0.3 = 1.0
        assert abs(score - 1.0) < 0.01

    def test_time_score_decay(self):
        now = datetime.utcnow()
        result = MockResult(
            score=1.0,
            published_at=(now - timedelta(hours=7)).isoformat(),
        )
        score = TimeWeightedRetriever._time_score(result)
        # At 7 hours: 1.0 * 0.7 + exp(-0.7) * 0.3 ≈ 0.7 + 0.149 = 0.849
        assert 0.8 < score < 0.9

    def test_time_score_invalid_date(self):
        result = MockResult(score=0.5, published_at="invalid")
        score = TimeWeightedRetriever._time_score(result)
        # Falls back to 48h age
        assert score > 0

    def test_build_filters_none(self):
        f = TimeWeightedRetriever._build_filters(None)
        assert f is None

    def test_build_filters_asset_class(self):
        f = TimeWeightedRetriever._build_filters("forex")
        assert f is not None

    @patch("finsight.storage.retriever.get_qdrant_client")
    def test_retrieve_returns_reranked(self, mock_client):
        now = datetime.utcnow()
        mock_results = [
            MockResult(0.9, (now - timedelta(hours=12)).isoformat()),
            MockResult(0.7, now.isoformat()),
            MockResult(0.85, (now - timedelta(hours=6)).isoformat()),
        ]

        client = MagicMock()
        client.search.return_value = mock_results
        mock_client.return_value = client

        retriever = TimeWeightedRetriever(client=client)
        results = retriever.retrieve(
            query_embedding=[0.1] * 768,
            k=2,
        )

        assert len(results) == 2
        scores = [TimeWeightedRetriever._time_score(r) for r in results]
        assert scores[0] >= scores[1]


class TestRetrieverIntegration:
    """These tests require a running Qdrant instance — skip in CI."""

    @pytest.mark.skipif(True, reason="Requires running Qdrant")
    def test_end_to_end_retrieval(self):
        pass

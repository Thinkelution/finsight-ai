"""Tests for the ingestion layer."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from finsight.ingestion.deduplicator import Deduplicator
from finsight.ingestion.rss_fetcher import RSSFetcher


class TestDeduplicator:
    def test_hash_content(self):
        text = "Hello, world!"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert Deduplicator.hash_content(text) == expected

    def test_in_memory_dedup(self):
        dedup = Deduplicator.__new__(Deduplicator)
        dedup._use_redis = False
        dedup._fallback = set()

        h = "abc123"
        assert not dedup.is_duplicate(h)
        dedup.mark_seen(h)
        assert dedup.is_duplicate(h)

    def test_is_duplicate_text(self):
        dedup = Deduplicator.__new__(Deduplicator)
        dedup._use_redis = False
        dedup._fallback = set()

        text = "This is a test article about forex markets."
        assert not dedup.is_duplicate_text(text)
        dedup.mark_seen_text(text)
        assert dedup.is_duplicate_text(text)

    def test_different_texts_not_duplicate(self):
        dedup = Deduplicator.__new__(Deduplicator)
        dedup._use_redis = False
        dedup._fallback = set()

        dedup.mark_seen_text("Article A")
        assert not dedup.is_duplicate_text("Article B")


class TestRSSFetcher:
    def test_load_feeds(self):
        with patch.object(RSSFetcher, "__init__", lambda self, **kw: None):
            fetcher = RSSFetcher()
            fetcher.dedup = Deduplicator.__new__(Deduplicator)
            fetcher.dedup._use_redis = False
            fetcher.dedup._fallback = set()
            fetcher._http = MagicMock()
            fetcher.feeds = fetcher._load_feeds()
            assert isinstance(fetcher.feeds, list)
            assert len(fetcher.feeds) > 0

    def test_process_entry_skips_short_text(self):
        with patch.object(RSSFetcher, "__init__", lambda self, **kw: None):
            fetcher = RSSFetcher()
            fetcher.dedup = Deduplicator.__new__(Deduplicator)
            fetcher.dedup._use_redis = False
            fetcher.dedup._fallback = set()

            with patch.object(fetcher, "_extract_full_text", return_value="too short"):
                entry = MagicMock()
                entry.get = lambda key, default="": {
                    "link": "https://example.com/article",
                    "title": "Test",
                    "summary": "too short",
                }.get(key, default)

                result = fetcher._process_entry(
                    entry,
                    {"name": "test", "asset_classes": ["equities"], "regions": ["us"]},
                )
                assert result is None

    def test_process_entry_deduplicates(self):
        with patch.object(RSSFetcher, "__init__", lambda self, **kw: None):
            fetcher = RSSFetcher()
            fetcher.dedup = Deduplicator.__new__(Deduplicator)
            fetcher.dedup._use_redis = False
            fetcher.dedup._fallback = set()

            long_text = "A" * 200

            with patch.object(fetcher, "_extract_full_text", return_value=long_text):
                entry = MagicMock()
                entry.get = lambda key, default="": {
                    "link": "https://example.com/article",
                    "title": "Test Article",
                    "summary": long_text,
                }.get(key, default)

                config = {"name": "test", "asset_classes": ["forex"], "regions": ["global"]}

                result1 = fetcher._process_entry(entry, config)
                assert result1 is not None
                assert result1["source"] == "test"

                result2 = fetcher._process_entry(entry, config)
                assert result2 is None  # duplicate

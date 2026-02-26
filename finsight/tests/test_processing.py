"""Tests for the processing pipeline."""

import pytest

from finsight.processing.chunker import chunk_text
from finsight.processing.cleaner import clean_text, extract_headline
from finsight.processing.ner import extract_entities
from finsight.processing.sentiment import _fallback_sentiment


class TestCleaner:
    def test_strip_html(self):
        text = "<p>Hello <b>world</b></p>"
        result = clean_text(text)
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strip_urls(self):
        text = "Visit https://example.com for more info."
        result = clean_text(text)
        assert "https://" not in result
        assert "Visit" in result

    def test_remove_boilerplate(self):
        text = "Great article. Subscribe to our newsletter for updates."
        result = clean_text(text)
        assert "subscribe" not in result.lower() or "newsletter" not in result.lower()

    def test_normalize_whitespace(self):
        text = "Hello    world\n\n\n\nGoodbye"
        result = clean_text(text)
        assert "    " not in result

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_extract_headline(self):
        text = "Breaking: Markets crash. More details to follow."
        headline = extract_headline(text)
        assert headline.startswith("Breaking")


class TestChunker:
    def test_short_text_single_chunk(self):
        text = "Short text that fits in one chunk."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0

    def test_long_text_multiple_chunks(self):
        words = ["word"] * 1000
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i

    def test_overlap(self):
        words = ["word"] * 200
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        # Second chunk should start before first chunk ends
        assert chunks[1]["start_token"] < chunks[0]["end_token"]

    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []


class TestNER:
    def test_extract_fx_pairs(self):
        text = "EUR/USD fell 0.8% while GBP/JPY rallied."
        entities = extract_entities(text)
        assert "EUR/USD" in entities["fx_pairs"]
        assert "GBP/JPY" in entities["fx_pairs"]

    def test_extract_tickers(self):
        text = "AAPL and MSFT reported strong earnings, while TSLA disappointed."
        entities = extract_entities(text)
        assert "AAPL" in entities["tickers"]
        assert "MSFT" in entities["tickers"]
        assert "TSLA" in entities["tickers"]

    def test_filters_common_words(self):
        text = "THE market WAS strong BUT NOT for ALL stocks."
        entities = extract_entities(text)
        for ticker in entities["tickers"]:
            assert ticker not in {"THE", "WAS", "BUT", "NOT", "ALL"}


class TestSentiment:
    def test_positive_fallback(self):
        text = "Markets surge and rally on strong growth data, beating expectations."
        result = _fallback_sentiment(text)
        assert result["label"] == "positive"

    def test_negative_fallback(self):
        text = "Markets crash and plunge in a massive selloff amid recession fears and crisis."
        result = _fallback_sentiment(text)
        assert result["label"] == "negative"

    def test_neutral_fallback(self):
        text = "The weather is nice today and I had coffee."
        result = _fallback_sentiment(text)
        assert result["label"] == "neutral"

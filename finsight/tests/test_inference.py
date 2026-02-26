"""Tests for the inference layer."""

from unittest.mock import MagicMock, patch

import pytest

from finsight.inference.alerter import Alert, AlertType, MarketAlerter
from finsight.inference.prompt_templates import SYSTEM_PROMPT, build_user_prompt


class MockChunk:
    def __init__(self, text, source, published_at, url=""):
        self.payload = {
            "text": text,
            "metadata": {
                "source": source,
                "published_at": published_at,
                "url": url,
            },
        }


class TestPromptTemplates:
    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 100
        assert "FinSight" in SYSTEM_PROMPT

    def test_build_user_prompt(self):
        chunks = [
            MockChunk(
                "EUR/USD fell 0.8% on hawkish Fed.",
                "reuters",
                "2025-01-15T10:00:00",
                "https://reuters.com/article1",
            ),
        ]
        prices = {
            "rates": {"EURUSD=X": 1.0850, "^GSPC": 5900},
            "changes": {"EURUSD=X": -0.8, "^GSPC": 0.5},
            "timestamp": "2025-01-15T12:00:00",
        }

        prompt = build_user_prompt(
            question="Why did EUR/USD fall?",
            news_chunks=chunks,
            live_prices=prices,
            market_summary="Markets mixed today.",
        )

        assert "EUR/USD" in prompt
        assert "reuters" in prompt
        assert "Why did EUR/USD fall?" in prompt
        assert "1.0850" in prompt
        assert "Markets mixed today." in prompt


class TestAlerter:
    def setup_method(self):
        self.alerts_fired: list[Alert] = []

        def capture(alert):
            self.alerts_fired.append(alert)

        self.alerter = MarketAlerter.__new__(MarketAlerter)
        self.alerter.threshold = 0.015
        self.alerter.on_alert = capture
        self.alerter._use_redis = False
        self.alerter._alert_cooldowns = {}

    def test_price_spike_detected(self):
        alert = self.alerter.check_price_move("EURUSD=X", 1.10, 1.08)
        # 1.85% move > 1.5% threshold
        assert alert is not None
        assert alert.alert_type == AlertType.PRICE_SPIKE
        assert "surged" in alert.message

    def test_price_move_below_threshold(self):
        alert = self.alerter.check_price_move("EURUSD=X", 1.081, 1.08)
        # 0.09% move < 1.5% threshold
        assert alert is None

    def test_price_alert_cooldown(self):
        alert1 = self.alerter.check_price_move("EURUSD=X", 1.10, 1.08)
        assert alert1 is not None

        alert2 = self.alerter.check_price_move("EURUSD=X", 1.12, 1.08)
        assert alert2 is None  # cooldown active

    def test_breaking_news_high_confidence(self):
        article = {
            "title": "Fed raises rates by 100bps in emergency session",
            "source": "reuters",
            "asset_classes": ["macro", "forex"],
            "url": "https://reuters.com/fed",
        }
        sentiment = {"label": "negative", "score": 0.95}

        alert = self.alerter.check_breaking_news(article, sentiment)
        assert alert is not None
        assert alert.alert_type == AlertType.BREAKING_NEWS
        assert "high" == alert.severity

    def test_breaking_news_low_confidence_ignored(self):
        article = {"title": "Minor update", "source": "test", "asset_classes": []}
        sentiment = {"label": "neutral", "score": 0.4}

        alert = self.alerter.check_breaking_news(article, sentiment)
        assert alert is None

    def test_cross_asset_correlation(self):
        changes = {
            "EURUSD=X": 1.2,
            "GC=F": 1.5,
            "USDJPY=X": 0.8,
            "^GSPC": 0.6,
            "CL=F": -2.0,
            "USDCAD=X": 1.0,
        }
        alerts = self.alerter.check_cross_asset_correlation(changes)
        assert isinstance(alerts, list)

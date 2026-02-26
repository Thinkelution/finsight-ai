"""Event detection and proactive alerts for unusual market moves and breaking news."""

from datetime import datetime, timedelta
from typing import Callable

from redis import Redis

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

ALERT_KEY_PREFIX = "finsight:alert:"
ALERT_HISTORY_KEY = "finsight:alert_history"
ALERT_COOLDOWN = timedelta(minutes=15)
PRICE_WINDOW_KEY = "finsight:price_window:"


class AlertType:
    PRICE_SPIKE = "price_spike"
    BREAKING_NEWS = "breaking_news"
    SENTIMENT_SHIFT = "sentiment_shift"
    CORRELATION = "cross_asset_correlation"


class Alert:
    def __init__(
        self,
        alert_type: str,
        symbol: str,
        message: str,
        severity: str = "info",
        data: dict | None = None,
    ):
        self.alert_type = alert_type
        self.symbol = symbol
        self.message = message
        self.severity = severity
        self.data = data or {}
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "symbol": self.symbol,
            "message": self.message,
            "severity": self.severity,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class MarketAlerter:
    """Monitors for unusual price moves, breaking news, and sentiment shifts."""

    def __init__(self, on_alert: Callable[[Alert], None] | None = None):
        self.threshold = settings.alert_price_move_threshold
        self.on_alert = on_alert or self._default_handler
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}

        try:
            self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
            self.redis.ping()
            self._use_redis = True
        except Exception:
            self._use_redis = False
            self._alert_cooldowns: dict[str, datetime] = {}

    def check_price_move(self, symbol: str, current_price: float, previous_price: float) -> Alert | None:
        """Detect unusual price moves exceeding the configured threshold."""
        if previous_price <= 0:
            return None

        pct_change = abs(current_price - previous_price) / previous_price
        if pct_change < self.threshold:
            return None

        if self._is_in_cooldown(f"price_{symbol}"):
            return None

        direction = "surged" if current_price > previous_price else "plunged"
        pct_str = f"{pct_change * 100:.2f}%"

        alert = Alert(
            alert_type=AlertType.PRICE_SPIKE,
            symbol=symbol,
            message=f"{symbol} {direction} {pct_str} — from {previous_price:.4f} to {current_price:.4f}",
            severity="high" if pct_change > self.threshold * 2 else "medium",
            data={
                "current_price": current_price,
                "previous_price": previous_price,
                "pct_change": round(pct_change * 100, 2),
                "direction": "up" if current_price > previous_price else "down",
            },
        )

        self._set_cooldown(f"price_{symbol}")
        self.on_alert(alert)
        return alert

    def check_breaking_news(self, article: dict, sentiment: dict) -> Alert | None:
        """Alert on high-impact breaking news with strong sentiment."""
        score = sentiment.get("score", 0)
        label = sentiment.get("label", "neutral")

        if label == "neutral" or score < 0.75:
            return None

        source = article.get("source", "unknown")
        title = article.get("title", "")
        cooldown_key = f"news_{title[:50]}"

        if self._is_in_cooldown(cooldown_key):
            return None

        alert = Alert(
            alert_type=AlertType.BREAKING_NEWS,
            symbol=",".join(article.get("asset_classes", [])),
            message=f"Breaking ({label}, confidence {score:.0%}): {title}",
            severity="high" if score > 0.9 else "medium",
            data={
                "title": title,
                "source": source,
                "sentiment_label": label,
                "sentiment_score": score,
                "url": article.get("url", ""),
            },
        )

        self._set_cooldown(cooldown_key)
        self.on_alert(alert)
        return alert

    def check_sentiment_shift(
        self,
        asset_class: str,
        recent_sentiments: list[dict],
        window_hours: int = 4,
    ) -> Alert | None:
        """Detect when sentiment flips from positive to negative (or vice versa)
        within a time window."""
        if len(recent_sentiments) < 5:
            return None

        half = len(recent_sentiments) // 2
        early = recent_sentiments[:half]
        late = recent_sentiments[half:]

        def avg_sentiment(items):
            scores = []
            for s in items:
                if s.get("label") == "positive":
                    scores.append(1.0)
                elif s.get("label") == "negative":
                    scores.append(-1.0)
                else:
                    scores.append(0.0)
            return sum(scores) / len(scores) if scores else 0

        early_avg = avg_sentiment(early)
        late_avg = avg_sentiment(late)
        shift = late_avg - early_avg

        if abs(shift) < 0.5:
            return None

        cooldown_key = f"sentiment_{asset_class}"
        if self._is_in_cooldown(cooldown_key):
            return None

        direction = "bullish" if shift > 0 else "bearish"
        alert = Alert(
            alert_type=AlertType.SENTIMENT_SHIFT,
            symbol=asset_class,
            message=f"Sentiment shift to {direction} for {asset_class} (delta: {shift:+.2f})",
            severity="medium",
            data={
                "early_avg": round(early_avg, 2),
                "late_avg": round(late_avg, 2),
                "shift": round(shift, 2),
                "sample_size": len(recent_sentiments),
            },
        )

        self._set_cooldown(cooldown_key)
        self.on_alert(alert)
        return alert

    def check_cross_asset_correlation(
        self,
        price_changes: dict[str, float],
    ) -> list[Alert]:
        """Detect notable cross-asset correlations in current price moves."""
        alerts = []

        correlations = [
            {
                "pair": ("EURUSD=X", "GC=F"),
                "name": "EUR/USD vs Gold",
                "desc": "USD weakness driving both EUR and gold higher",
            },
            {
                "pair": ("USDJPY=X", "^GSPC"),
                "name": "USD/JPY vs S&P 500",
                "desc": "Risk-on sentiment lifting both equities and USD/JPY",
            },
            {
                "pair": ("GC=F", "^GSPC"),
                "name": "Gold vs S&P 500",
                "desc": "Unusual same-direction move in gold and equities",
                "inverse": True,
            },
            {
                "pair": ("CL=F", "USDCAD=X"),
                "name": "Oil vs USD/CAD",
                "desc": "Oil move impacting CAD (petrocurrency)",
                "inverse": True,
            },
        ]

        for corr in correlations:
            sym_a, sym_b = corr["pair"]
            chg_a = price_changes.get(sym_a)
            chg_b = price_changes.get(sym_b)

            if chg_a is None or chg_b is None:
                continue

            both_significant = abs(chg_a) > 0.5 and abs(chg_b) > 0.5
            if not both_significant:
                continue

            is_inverse = corr.get("inverse", False)
            same_dir = (chg_a > 0) == (chg_b > 0)

            if (is_inverse and same_dir) or (not is_inverse and same_dir):
                cooldown_key = f"corr_{sym_a}_{sym_b}"
                if self._is_in_cooldown(cooldown_key):
                    continue

                alert = Alert(
                    alert_type=AlertType.CORRELATION,
                    symbol=f"{sym_a},{sym_b}",
                    message=f"Cross-asset: {corr['name']} — {corr['desc']}",
                    severity="info",
                    data={
                        "symbol_a": sym_a,
                        "change_a": chg_a,
                        "symbol_b": sym_b,
                        "change_b": chg_b,
                    },
                )
                self._set_cooldown(cooldown_key)
                self.on_alert(alert)
                alerts.append(alert)

        return alerts

    def _is_in_cooldown(self, key: str) -> bool:
        if self._use_redis:
            return bool(self.redis.exists(f"{ALERT_KEY_PREFIX}{key}"))
        now = datetime.utcnow()
        last = self._alert_cooldowns.get(key)
        return last is not None and (now - last) < ALERT_COOLDOWN

    def _set_cooldown(self, key: str) -> None:
        if self._use_redis:
            self.redis.setex(
                f"{ALERT_KEY_PREFIX}{key}",
                int(ALERT_COOLDOWN.total_seconds()),
                "1",
            )
        else:
            self._alert_cooldowns[key] = datetime.utcnow()

    @staticmethod
    def _default_handler(alert: Alert):
        logger.info(
            "alert_triggered",
            type=alert.alert_type,
            symbol=alert.symbol,
            severity=alert.severity,
            message=alert.message,
        )

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Get recent alert history from Redis."""
        if not self._use_redis:
            return []
        import json
        raw = self.redis.lrange(ALERT_HISTORY_KEY, 0, limit - 1)
        return [json.loads(r) for r in raw]

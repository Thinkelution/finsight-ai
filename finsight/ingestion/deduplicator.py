import hashlib
from datetime import timedelta

from redis import Redis

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

DEDUP_KEY_PREFIX = "finsight:dedup:"
DEDUP_TTL = timedelta(days=7)


class Deduplicator:
    """SHA256 hash-based deduplication backed by Redis.

    Hashes expire after 7 days so we don't accumulate unbounded state.
    Falls back to an in-memory set when Redis is unavailable.
    """

    def __init__(self, redis_client: Redis | None = None):
        self._fallback: set[str] = set()
        try:
            self.redis = redis_client or Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            self.redis.ping()
            self._use_redis = True
        except Exception:
            logger.warning("redis_unavailable_for_dedup, using in-memory fallback")
            self._use_redis = False

    @staticmethod
    def hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, content_hash: str) -> bool:
        if self._use_redis:
            return bool(self.redis.exists(f"{DEDUP_KEY_PREFIX}{content_hash}"))
        return content_hash in self._fallback

    def mark_seen(self, content_hash: str) -> None:
        if self._use_redis:
            self.redis.setex(
                f"{DEDUP_KEY_PREFIX}{content_hash}",
                DEDUP_TTL,
                "1",
            )
        else:
            self._fallback.add(content_hash)

    def is_duplicate_text(self, text: str) -> bool:
        return self.is_duplicate(self.hash_content(text))

    def mark_seen_text(self, text: str) -> str:
        h = self.hash_content(text)
        self.mark_seen(h)
        return h

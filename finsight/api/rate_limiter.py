"""Simple rate limiter for API endpoints using Redis or in-memory fallback."""

import time
from collections import defaultdict

from fastapi import HTTPException, Request

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

RATE_LIMIT_REQUESTS = 30  # max requests
RATE_LIMIT_WINDOW = 60  # per N seconds


class RateLimiter:
    def __init__(self):
        self._use_redis = False
        self._memory_store: dict[str, list[float]] = defaultdict(list)

        try:
            from redis import Redis
            self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
            self.redis.ping()
            self._use_redis = True
        except Exception:
            logger.warning("rate_limiter_using_memory_fallback")

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request) -> None:
        """Raise 429 if client exceeds rate limit."""
        client_id = self._get_client_id(request)

        if self._use_redis:
            self._check_redis(client_id)
        else:
            self._check_memory(client_id)

    def _check_redis(self, client_id: str):
        key = f"finsight:ratelimit:{client_id}"
        current = self.redis.incr(key)
        if current == 1:
            self.redis.expire(key, RATE_LIMIT_WINDOW)

        if current > RATE_LIMIT_REQUESTS:
            ttl = self.redis.ttl(key)
            logger.warning("rate_limited", client=client_id, count=current)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {ttl}s.",
            )

    def _check_memory(self, client_id: str):
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        self._memory_store[client_id] = [
            t for t in self._memory_store[client_id] if t > window_start
        ]

        if len(self._memory_store[client_id]) >= RATE_LIMIT_REQUESTS:
            logger.warning("rate_limited", client=client_id)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s).",
            )

        self._memory_store[client_id].append(now)


rate_limiter = RateLimiter()

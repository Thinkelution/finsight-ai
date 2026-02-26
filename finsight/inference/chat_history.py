"""Multi-turn conversation history management."""

from datetime import datetime, timedelta
from typing import Any

from redis import Redis

from finsight.config.logging import get_logger
from finsight.config.settings import settings

logger = get_logger(__name__)

HISTORY_KEY_PREFIX = "finsight:chat:"
HISTORY_TTL = timedelta(hours=24)
MAX_TURNS = 10


class ChatHistory:
    """Manages multi-turn conversation context per session."""

    def __init__(self, session_id: str, redis_client: Redis | None = None):
        self.session_id = session_id
        self._messages: list[dict] = []

        try:
            self.redis = redis_client or Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            self.redis.ping()
            self._use_redis = True
            self._load_from_redis()
        except Exception:
            self._use_redis = False

    def _redis_key(self) -> str:
        return f"{HISTORY_KEY_PREFIX}{self.session_id}"

    def _load_from_redis(self):
        import json
        raw = self.redis.get(self._redis_key())
        if raw:
            self._messages = json.loads(raw)

    def _save_to_redis(self):
        import json
        self.redis.setex(
            self._redis_key(),
            int(HISTORY_TTL.total_seconds()),
            json.dumps(self._messages),
        )

    def add_user_message(self, content: str):
        self._messages.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._trim()
        if self._use_redis:
            self._save_to_redis()

    def add_assistant_message(self, content: str):
        self._messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._trim()
        if self._use_redis:
            self._save_to_redis()

    def get_messages_for_llm(self) -> list[dict]:
        """Return messages in the format expected by Ollama/LLM APIs."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def get_context_summary(self) -> str:
        """Build a summary of previous conversation for context injection."""
        if not self._messages:
            return ""
        lines = []
        for m in self._messages[-6:]:
            role = "User" if m["role"] == "user" else "FinSight"
            lines.append(f"{role}: {m['content'][:200]}")
        return "\n".join(lines)

    def _trim(self):
        if len(self._messages) > MAX_TURNS * 2:
            self._messages = self._messages[-(MAX_TURNS * 2):]

    def clear(self):
        self._messages = []
        if self._use_redis:
            self.redis.delete(self._redis_key())

    @property
    def turn_count(self) -> int:
        return len([m for m in self._messages if m["role"] == "user"])

"""
Solace-AI Memory Service - Redis Cache.
Redis-based cache for working memory, session state, and fast access patterns.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class RedisSettings(BaseSettings):
    """Redis connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: str = Field(default="")
    ssl: bool = Field(default=False)
    socket_timeout: int = Field(default=5)
    connection_pool_size: int = Field(default=10)
    working_memory_ttl: int = Field(default=3600)
    session_ttl: int = Field(default=86400)
    cache_ttl: int = Field(default=300)
    key_prefix: str = Field(default="solace:memory:")
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")


@dataclass
class CachedWorkingMemory:
    """Cached working memory state."""
    user_id: UUID
    session_id: UUID
    items: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 8000
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {"user_id": str(self.user_id), "session_id": str(self.session_id),
                "items": self.items, "total_tokens": self.total_tokens, "max_tokens": self.max_tokens,
                "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedWorkingMemory:
        return cls(user_id=UUID(data["user_id"]), session_id=UUID(data["session_id"]),
                   items=data.get("items", []), total_tokens=data.get("total_tokens", 0),
                   max_tokens=data.get("max_tokens", 8000), created_at=datetime.fromisoformat(data["created_at"]),
                   updated_at=datetime.fromisoformat(data["updated_at"]))


@dataclass
class CachedSessionState:
    """Cached session state."""
    session_id: UUID
    user_id: UUID
    session_number: int = 1
    status: str = "active"
    message_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    topics_detected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": str(self.session_id), "user_id": str(self.user_id),
                "session_number": self.session_number, "status": self.status, "message_count": self.message_count,
                "started_at": self.started_at.isoformat(), "last_activity": self.last_activity.isoformat(),
                "topics_detected": self.topics_detected, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedSessionState:
        return cls(session_id=UUID(data["session_id"]), user_id=UUID(data["user_id"]),
                   session_number=data.get("session_number", 1), status=data.get("status", "active"),
                   message_count=data.get("message_count", 0), started_at=datetime.fromisoformat(data["started_at"]),
                   last_activity=datetime.fromisoformat(data["last_activity"]),
                   topics_detected=data.get("topics_detected", []), metadata=data.get("metadata", {}))


class RedisCache:
    """Redis cache for working memory and session state."""

    def __init__(self, settings: RedisSettings | None = None) -> None:
        self._settings = settings or RedisSettings()
        self._client: Any = None
        self._initialized = False
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0}

    async def initialize(self, max_retries: int = 3) -> None:
        """Initialize Redis connection with exponential backoff retry."""
        try:
            import redis.asyncio as redis
        except ImportError:
            logger.error("redis_not_installed", hint="pip install redis")
            return

        for attempt in range(max_retries + 1):
            try:
                self._client = redis.Redis(
                    host=self._settings.host, port=self._settings.port, db=self._settings.db,
                    password=self._settings.password if self._settings.password else None,
                    ssl=self._settings.ssl, socket_timeout=self._settings.socket_timeout,
                    decode_responses=True,
                )
                await self._client.ping()
                self._initialized = True
                logger.info("redis_initialized", host=self._settings.host, db=self._settings.db)
                return
            except Exception as e:
                if attempt < max_retries:
                    import asyncio
                    delay = min(2 ** attempt, 30.0)
                    logger.warning("redis_connect_retry", attempt=attempt + 1,
                                   max_retries=max_retries, delay_seconds=delay, error=str(e))
                    await asyncio.sleep(delay)
                else:
                    logger.error("redis_init_failed", error=str(e), attempts=max_retries + 1)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
        logger.info("redis_closed")

    def _key(self, *parts: str) -> str:
        return self._settings.key_prefix + ":".join(parts)

    async def set_working_memory(self, memory: CachedWorkingMemory) -> bool:
        """Store working memory state."""
        if not self._initialized:
            return False
        self._stats["sets"] += 1
        try:
            key = self._key("working", str(memory.user_id), str(memory.session_id))
            memory.updated_at = datetime.now(timezone.utc)
            await self._client.setex(key, self._settings.working_memory_ttl, json.dumps(memory.to_dict()))
            return True
        except Exception as e:
            logger.error("working_memory_cache_failed", error=str(e))
            return False

    async def get_working_memory(self, user_id: UUID, session_id: UUID) -> CachedWorkingMemory | None:
        """Get working memory state."""
        if not self._initialized:
            return None
        try:
            key = self._key("working", str(user_id), str(session_id))
            data = await self._client.get(key)
            if data:
                self._stats["hits"] += 1
                return CachedWorkingMemory.from_dict(json.loads(data))
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.error("working_memory_get_failed", error=str(e))
            return None

    async def delete_working_memory(self, user_id: UUID, session_id: UUID) -> bool:
        """Delete working memory state."""
        if not self._initialized:
            return False
        self._stats["deletes"] += 1
        try:
            await self._client.delete(self._key("working", str(user_id), str(session_id)))
            return True
        except Exception as e:
            logger.error("working_memory_delete_failed", error=str(e))
            return False

    async def set_session_state(self, state: CachedSessionState) -> bool:
        """Store session state."""
        if not self._initialized:
            return False
        self._stats["sets"] += 1
        try:
            key = self._key("session", str(state.session_id))
            user_key = self._key("user_session", str(state.user_id))
            state.last_activity = datetime.now(timezone.utc)
            await self._client.setex(key, self._settings.session_ttl, json.dumps(state.to_dict()))
            await self._client.setex(user_key, self._settings.session_ttl, str(state.session_id))
            return True
        except Exception as e:
            logger.error("session_state_cache_failed", error=str(e))
            return False

    async def get_session_state(self, session_id: UUID) -> CachedSessionState | None:
        """Get session state."""
        if not self._initialized:
            return None
        try:
            data = await self._client.get(self._key("session", str(session_id)))
            if data:
                self._stats["hits"] += 1
                return CachedSessionState.from_dict(json.loads(data))
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.error("session_state_get_failed", error=str(e))
            return None

    async def get_active_session(self, user_id: UUID) -> CachedSessionState | None:
        """Get active session for a user."""
        if not self._initialized:
            return None
        try:
            session_id_str = await self._client.get(self._key("user_session", str(user_id)))
            return await self.get_session_state(UUID(session_id_str)) if session_id_str else None
        except Exception as e:
            logger.error("active_session_get_failed", error=str(e))
            return None

    async def delete_session_state(self, session_id: UUID, user_id: UUID) -> bool:
        """Delete session state."""
        if not self._initialized:
            return False
        self._stats["deletes"] += 1
        try:
            await self._client.delete(self._key("session", str(session_id)), self._key("user_session", str(user_id)))
            return True
        except Exception as e:
            logger.error("session_state_delete_failed", error=str(e))
            return False

    async def cache_context(self, user_id: UUID, context_key: str, context: str, ttl: int | None = None) -> bool:
        """Cache assembled context for reuse."""
        if not self._initialized:
            return False
        self._stats["sets"] += 1
        try:
            await self._client.setex(self._key("context", str(user_id), context_key), ttl or self._settings.cache_ttl, context)
            return True
        except Exception as e:
            logger.error("context_cache_failed", error=str(e))
            return False

    async def get_cached_context(self, user_id: UUID, context_key: str) -> str | None:
        """Get cached context."""
        if not self._initialized:
            return None
        try:
            data = await self._client.get(self._key("context", str(user_id), context_key))
            if data:
                self._stats["hits"] += 1
                return data
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.error("context_get_failed", error=str(e))
            return None

    async def cache_embedding(self, content_hash: str, embedding: list[float], ttl: int = 3600) -> bool:
        """Cache an embedding for reuse."""
        if not self._initialized:
            return False
        self._stats["sets"] += 1
        try:
            await self._client.setex(self._key("embedding", content_hash), ttl, json.dumps(embedding))
            return True
        except Exception as e:
            logger.error("embedding_cache_failed", error=str(e))
            return False

    async def get_cached_embedding(self, content_hash: str) -> list[float] | None:
        """Get cached embedding."""
        if not self._initialized:
            return None
        try:
            data = await self._client.get(self._key("embedding", content_hash))
            if data:
                self._stats["hits"] += 1
                return json.loads(data)
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.error("embedding_get_failed", error=str(e))
            return None

    async def increment_session_message_count(self, session_id: UUID) -> int:
        """Increment and return session message count."""
        if not self._initialized:
            return 0
        try:
            state = await self.get_session_state(session_id)
            if state:
                state.message_count += 1
                state.last_activity = datetime.now(timezone.utc)
                await self.set_session_state(state)
                return state.message_count
            return 0
        except Exception as e:
            logger.error("message_count_increment_failed", error=str(e))
            return 0

    async def add_session_topic(self, session_id: UUID, topic: str) -> bool:
        """Add a detected topic to session."""
        if not self._initialized:
            return False
        try:
            state = await self.get_session_state(session_id)
            if state and topic not in state.topics_detected:
                state.topics_detected.append(topic)
                await self.set_session_state(state)
                return True
            return False
        except Exception as e:
            logger.error("topic_add_failed", error=str(e))
            return False

    async def get_user_session_count(self, user_id: UUID) -> int:
        """Get count of sessions for user."""
        if not self._initialized:
            return 0
        try:
            count = await self._client.get(self._key("session_count", str(user_id)))
            return int(count) if count else 0
        except Exception as e:
            logger.error("session_count_get_failed", error=str(e))
            return 0

    async def increment_user_session_count(self, user_id: UUID) -> int:
        """Increment user session count."""
        if not self._initialized:
            return 0
        try:
            return await self._client.incr(self._key("session_count", str(user_id)))
        except Exception as e:
            logger.error("session_count_incr_failed", error=str(e))
            return 0

    async def delete_user_cache(self, user_id: UUID) -> int:
        """Delete all cached data for a user."""
        if not self._initialized:
            return 0
        deleted = 0
        try:
            pattern, cursor = self._key("*", str(user_id), "*"), 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info("user_cache_deleted", user_id=str(user_id), deleted=deleted)
        except Exception as e:
            logger.error("user_cache_delete_failed", error=str(e))
        return deleted

    async def health_check(self) -> bool:
        """Check Redis health."""
        if not self._initialized:
            return False
        try:
            await self._client.ping()
            return True
        except Exception as e:
            logger.warning("redis_health_check_failed", error=str(e))
            return False

    def is_initialized(self) -> bool:
        return self._initialized

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {**self._stats, "hit_rate": round(hit_rate, 4), "initialized": self._initialized,
                "host": self._settings.host, "db": self._settings.db}

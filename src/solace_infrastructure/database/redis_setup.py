"""Solace-AI Redis Setup - Namespace configuration and initialization.

Provides enterprise-grade Redis setup with:
- Namespace definitions for data isolation
- TTL policies for memory tiers
- Key pattern conventions
- Initialization and verification
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from solace_infrastructure.redis import RedisClient, RedisSettings
from solace_common.exceptions import CacheError

logger = structlog.get_logger(__name__)


class RedisNamespace(str, Enum):
    """Redis key namespaces for Solace-AI."""
    SESSION = "session"
    WORKING_MEMORY = "wm"
    CONTEXT = "ctx"
    CACHE = "cache"
    LOCK = "lock"
    RATE_LIMIT = "rl"
    PUBSUB = "ps"
    QUEUE = "q"


class MemoryTier(str, Enum):
    """Memory tier identifiers for TTL policies."""
    BUFFER = "buffer"
    WORKING = "working"
    SESSION = "session"
    TEMP = "temp"


@dataclass
class TTLPolicy:
    """TTL policy for a memory tier."""

    tier: MemoryTier
    ttl_seconds: int
    description: str


class RedisSetupSettings(BaseSettings):
    """Redis setup configuration from environment."""

    buffer_ttl_seconds: int = Field(default=300, ge=60)
    working_memory_ttl_seconds: int = Field(default=3600, ge=300)
    session_ttl_seconds: int = Field(default=86400, ge=3600)
    cache_ttl_seconds: int = Field(default=3600, ge=60)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    verify_on_init: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_prefix="REDIS_SETUP_",
        env_file=".env",
        extra="ignore",
    )


class RedisKeyBuilder:
    """Builds consistent Redis keys following Solace-AI conventions."""

    def __init__(self, prefix: str = "solace:") -> None:
        self._prefix = prefix

    def session_key(self, session_id: str) -> str:
        """Build session state key."""
        return f"{self._prefix}{RedisNamespace.SESSION.value}:{session_id}"

    def working_memory_key(self, user_id: str, session_id: str) -> str:
        """Build working memory key."""
        return f"{self._prefix}{RedisNamespace.WORKING_MEMORY.value}:{user_id}:{session_id}"

    def context_key(self, user_id: str) -> str:
        """Build user context cache key."""
        return f"{self._prefix}{RedisNamespace.CONTEXT.value}:{user_id}"

    def cache_key(self, namespace: str, key: str) -> str:
        """Build generic cache key."""
        return f"{self._prefix}{RedisNamespace.CACHE.value}:{namespace}:{key}"

    def lock_key(self, resource: str) -> str:
        """Build distributed lock key."""
        return f"{self._prefix}{RedisNamespace.LOCK.value}:{resource}"

    def rate_limit_key(self, user_id: str, endpoint: str) -> str:
        """Build rate limit counter key."""
        return f"{self._prefix}{RedisNamespace.RATE_LIMIT.value}:{user_id}:{endpoint}"

    def pubsub_channel(self, channel: str) -> str:
        """Build pub/sub channel name."""
        return f"{self._prefix}{RedisNamespace.PUBSUB.value}:{channel}"

    def queue_key(self, queue_name: str) -> str:
        """Build queue key."""
        return f"{self._prefix}{RedisNamespace.QUEUE.value}:{queue_name}"


class RedisSetupManager:
    """Manages Redis initialization and configuration for Solace-AI."""

    def __init__(
        self,
        client: RedisClient,
        settings: RedisSetupSettings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or RedisSetupSettings()
        self._key_builder = RedisKeyBuilder(client.key_prefix)
        self._ttl_policies = self._build_ttl_policies()

    def _build_ttl_policies(self) -> list[TTLPolicy]:
        """Build TTL policies from settings."""
        return [
            TTLPolicy(
                tier=MemoryTier.BUFFER,
                ttl_seconds=self._settings.buffer_ttl_seconds,
                description="Short-lived buffer for current message processing",
            ),
            TTLPolicy(
                tier=MemoryTier.WORKING,
                ttl_seconds=self._settings.working_memory_ttl_seconds,
                description="Working memory for active LLM context",
            ),
            TTLPolicy(
                tier=MemoryTier.SESSION,
                ttl_seconds=self._settings.session_ttl_seconds,
                description="Full session transcript and state",
            ),
            TTLPolicy(
                tier=MemoryTier.TEMP,
                ttl_seconds=self._settings.cache_ttl_seconds,
                description="Temporary cache for computed values",
            ),
        ]

    @property
    def key_builder(self) -> RedisKeyBuilder:
        """Get the key builder instance."""
        return self._key_builder

    def get_ttl(self, tier: MemoryTier) -> int:
        """Get TTL seconds for a memory tier."""
        for policy in self._ttl_policies:
            if policy.tier == tier:
                return policy.ttl_seconds
        return self._settings.cache_ttl_seconds

    async def initialize(self) -> bool:
        """Initialize Redis with verification."""
        try:
            if self._settings.verify_on_init:
                health = await self._client.check_health()
                if health.get("status") != "healthy":
                    raise CacheError(f"Redis not healthy: {health.get('error')}")
            logger.info("redis_setup_initialized", policies=len(self._ttl_policies))
            return True
        except Exception as e:
            logger.error("redis_setup_failed", error=str(e))
            raise

    async def verify_connection(self) -> dict[str, Any]:
        """Verify Redis connection and return status."""
        return await self._client.check_health()

    async def flush_namespace(self, namespace: RedisNamespace) -> int:
        """Flush all keys in a namespace (use with caution)."""
        pattern = f"{self._client.key_prefix}{namespace.value}:*"
        deleted = 0
        client = self._client._ensure_connected()
        async for key in client.scan_iter(match=pattern, count=100):
            await client.delete(key)
            deleted += 1
        logger.warning("redis_namespace_flushed", namespace=namespace.value, deleted=deleted)
        return deleted

    async def get_namespace_stats(self) -> dict[str, int]:
        """Get key counts per namespace."""
        stats: dict[str, int] = {}
        client = self._client._ensure_connected()
        for ns in RedisNamespace:
            pattern = f"{self._client.key_prefix}{ns.value}:*"
            count = 0
            async for _ in client.scan_iter(match=pattern, count=100):
                count += 1
            stats[ns.value] = count
        return stats


async def setup_redis(
    client: RedisClient,
    settings: RedisSetupSettings | None = None,
) -> RedisSetupManager:
    """Initialize Redis setup for Solace-AI."""
    manager = RedisSetupManager(client, settings)
    await manager.initialize()
    return manager

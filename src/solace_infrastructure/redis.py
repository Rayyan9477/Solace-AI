"""Solace-AI Redis Client - Async caching, pub/sub, and session management."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio.client import PubSub

from solace_common.exceptions import CacheError

logger = structlog.get_logger(__name__)


class RedisMode(str, Enum):
    """Redis deployment modes."""
    STANDALONE = "standalone"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


class RedisSettings(BaseSettings):
    """Redis connection settings from environment."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: SecretStr | None = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    mode: RedisMode = Field(default=RedisMode.STANDALONE)
    sentinel_master: str = Field(default="mymaster")
    sentinel_nodes: str = Field(default="")
    cluster_nodes: str = Field(default="")
    ssl: bool = Field(default=False)
    ssl_ca_certs: str | None = Field(default=None)
    max_connections: int = Field(default=50, ge=1, le=1000)
    socket_timeout: float = Field(default=5.0, gt=0)
    socket_connect_timeout: float = Field(default=5.0, gt=0)
    retry_on_timeout: bool = Field(default=True)
    health_check_interval: int = Field(default=30, ge=0)
    decode_responses: bool = Field(default=True)
    key_prefix: str = Field(default="solace:")
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    def get_url(self) -> str:
        """Build Redis connection URL."""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class CacheEntry(BaseModel):
    """Wrapper for cached values with metadata."""
    value: Any
    created_at: float
    ttl_seconds: int | None = None


class RedisClient:
    """Async Redis client with caching, pub/sub, and distributed locking."""

    def __init__(self, settings: RedisSettings | None = None) -> None:
        self._settings = settings or RedisSettings()
        self._client: redis.Redis | None = None
        self._pubsub_handlers: dict[str, list[Callable[[str, Any], Awaitable[None]]]] = {}
        self._pubsub_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def key_prefix(self) -> str:
        return self._settings.key_prefix

    def _make_key(self, key: str) -> str:
        """Apply key prefix for namespace isolation."""
        if key.startswith(self._settings.key_prefix):
            return key
        return f"{self._settings.key_prefix}{key}"

    async def connect(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        """Initialize Redis connection with retry on transient errors."""
        if self._client is not None:
            return
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                if self._settings.mode == RedisMode.CLUSTER:
                    self._client = await self._create_cluster_client()
                elif self._settings.mode == RedisMode.SENTINEL:
                    self._client = await self._create_sentinel_client()
                else:
                    self._client = await self._create_standalone_client()
                await self._client.ping()
                logger.info("redis_connected", host=self._settings.host, mode=self._settings.mode.value)
                return
            except (TimeoutError, redis.RedisError, OSError) as e:
                last_error = e
                self._client = None
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "redis_connect_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise CacheError(
            f"Failed to connect to Redis after {max_retries} attempts: {last_error}",
            cause=last_error,
        )

    async def _create_standalone_client(self) -> redis.Redis:
        """Create standalone Redis client."""
        return redis.Redis(
            host=self._settings.host, port=self._settings.port,
            password=self._settings.password.get_secret_value() if self._settings.password else None,
            db=self._settings.db, decode_responses=self._settings.decode_responses,
            socket_timeout=self._settings.socket_timeout,
            socket_connect_timeout=self._settings.socket_connect_timeout,
            retry_on_timeout=self._settings.retry_on_timeout,
            health_check_interval=self._settings.health_check_interval,
            max_connections=self._settings.max_connections,
        )

    async def _create_cluster_client(self) -> redis.Redis:
        """Create Redis cluster client."""
        from redis.asyncio.cluster import RedisCluster
        nodes = self._settings.cluster_nodes.split(",")
        startup_nodes = [{"host": n.split(":")[0], "port": int(n.split(":")[1])} for n in nodes if n]
        return RedisCluster(startup_nodes=startup_nodes, decode_responses=self._settings.decode_responses)

    async def _create_sentinel_client(self) -> redis.Redis:
        """Create Redis Sentinel client."""
        from redis.asyncio.sentinel import Sentinel
        nodes = self._settings.sentinel_nodes.split(",")
        sentinel_list = [(n.split(":")[0], int(n.split(":")[1])) for n in nodes if n]
        sentinel = Sentinel(sentinel_list, socket_timeout=self._settings.socket_timeout)
        return sentinel.master_for(self._settings.sentinel_master, decode_responses=self._settings.decode_responses)

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("redis_disconnected")

    async def __aenter__(self) -> RedisClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        await self.disconnect()

    def _ensure_connected(self) -> redis.Redis:
        if not self._client:
            raise CacheError("Redis client not connected")
        return self._client

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        client = self._ensure_connected()
        try:
            value = await client.get(self._make_key(key))
            if value is None:
                return None
            return self._deserialize(value)
        except redis.RedisError as e:
            logger.warning("redis_get_error", key=key, error=str(e))
            raise CacheError(f"Failed to get key: {e}", cause=e) from e

    async def set(self, key: str, value: Any, ttl: int | timedelta | None = None) -> bool:
        """Set value in cache with optional TTL."""
        client = self._ensure_connected()
        try:
            serialized = self._serialize(value)
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            await client.set(self._make_key(key), serialized, ex=ttl)
            return True
        except redis.RedisError as e:
            logger.warning("redis_set_error", key=key, error=str(e))
            raise CacheError(f"Failed to set key: {e}", cause=e) from e

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        client = self._ensure_connected()
        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            return await client.delete(*prefixed_keys)
        except redis.RedisError as e:
            raise CacheError(f"Failed to delete keys: {e}", cause=e) from e

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        client = self._ensure_connected()
        prefixed_keys = [self._make_key(k) for k in keys]
        return await client.exists(*prefixed_keys)

    async def expire(self, key: str, ttl: int | timedelta) -> bool:
        """Set expiration on a key."""
        client = self._ensure_connected()
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
        return await client.expire(self._make_key(key), ttl)

    async def ttl(self, key: str) -> int:
        """Get remaining TTL in seconds."""
        client = self._ensure_connected()
        return await client.ttl(self._make_key(key))

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        client = self._ensure_connected()
        return await client.incrby(self._make_key(key), amount)

    async def hget(self, name: str, key: str) -> Any | None:
        """Get hash field value."""
        client = self._ensure_connected()
        value = await client.hget(self._make_key(name), key)
        return self._deserialize(value) if value else None

    async def hset(self, name: str, mapping: dict[str, Any]) -> int:
        """Set hash fields."""
        client = self._ensure_connected()
        serialized = {k: self._serialize(v) for k, v in mapping.items()}
        return await client.hset(self._make_key(name), mapping=serialized)

    async def hgetall(self, name: str) -> dict[str, Any]:
        """Get all hash fields."""
        client = self._ensure_connected()
        data = await client.hgetall(self._make_key(name))
        return {k: self._deserialize(v) for k, v in data.items()}

    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        client = self._ensure_connected()
        return await client.hdel(self._make_key(name), *keys)

    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to list head."""
        client = self._ensure_connected()
        serialized = [self._serialize(v) for v in values]
        return await client.lpush(self._make_key(key), *serialized)

    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to list tail."""
        client = self._ensure_connected()
        serialized = [self._serialize(v) for v in values]
        return await client.rpush(self._make_key(key), *serialized)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        """Get list range."""
        client = self._ensure_connected()
        values = await client.lrange(self._make_key(key), start, end)
        return [self._deserialize(v) for v in values]

    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        client = self._ensure_connected()
        serialized = self._serialize(message)
        count = await client.publish(self._make_key(channel), serialized)
        logger.debug("redis_publish", channel=channel, subscriber_count=count)
        return count

    async def subscribe(self, channel: str, handler: Callable[[str, Any], Awaitable[None]]) -> None:
        """Subscribe to channel with async handler."""
        prefixed = self._make_key(channel)
        if prefixed not in self._pubsub_handlers:
            self._pubsub_handlers[prefixed] = []
        self._pubsub_handlers[prefixed].append(handler)
        if self._pubsub_task is None:
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())

    async def _pubsub_listener(self) -> None:
        """Background task to listen for pub/sub messages."""
        client = self._ensure_connected()
        pubsub: PubSub = client.pubsub()
        try:
            await pubsub.subscribe(*self._pubsub_handlers.keys())
            async for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = self._deserialize(message["data"])
                    for handler in self._pubsub_handlers.get(channel, []):
                        await handler(channel, data)
        except asyncio.CancelledError:
            await pubsub.unsubscribe()
        finally:
            await pubsub.aclose()

    @asynccontextmanager
    async def lock(self, name: str, timeout: int = 10, blocking: bool = True
                   ) -> AsyncIterator[bool]:
        """Distributed lock context manager."""
        client = self._ensure_connected()
        lock_key = self._make_key(f"lock:{name}")
        lock = client.lock(lock_key, timeout=timeout, blocking=blocking)
        acquired = await lock.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                await lock.release()

    async def check_health(self) -> dict[str, Any]:
        """Check Redis health status."""
        try:
            client = self._ensure_connected()
            info = await client.info("server")
            return {"status": "healthy", "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"), "mode": self._settings.mode.value}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)

    def _deserialize(self, value: str | bytes) -> Any:
        """Deserialize stored value."""
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value


async def create_redis_client(settings: RedisSettings | None = None) -> RedisClient:
    """Factory function to create and connect a Redis client."""
    client = RedisClient(settings)
    await client.connect()
    return client

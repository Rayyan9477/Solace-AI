"""Unit tests for Redis setup manager."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solace_infrastructure.database.redis_setup import (
    RedisSetupManager,
    RedisSetupSettings,
    RedisKeyBuilder,
    RedisNamespace,
    MemoryTier,
    TTLPolicy,
    setup_redis,
)


class TestRedisNamespace:
    """Tests for RedisNamespace enum."""

    def test_session_namespace(self) -> None:
        """Test session namespace value."""
        assert RedisNamespace.SESSION.value == "session"

    def test_working_memory_namespace(self) -> None:
        """Test working memory namespace value."""
        assert RedisNamespace.WORKING_MEMORY.value == "wm"

    def test_context_namespace(self) -> None:
        """Test context namespace value."""
        assert RedisNamespace.CONTEXT.value == "ctx"

    def test_cache_namespace(self) -> None:
        """Test cache namespace value."""
        assert RedisNamespace.CACHE.value == "cache"

    def test_lock_namespace(self) -> None:
        """Test lock namespace value."""
        assert RedisNamespace.LOCK.value == "lock"

    def test_rate_limit_namespace(self) -> None:
        """Test rate limit namespace value."""
        assert RedisNamespace.RATE_LIMIT.value == "rl"

    def test_pubsub_namespace(self) -> None:
        """Test pubsub namespace value."""
        assert RedisNamespace.PUBSUB.value == "ps"

    def test_queue_namespace(self) -> None:
        """Test queue namespace value."""
        assert RedisNamespace.QUEUE.value == "q"


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_buffer_tier(self) -> None:
        """Test buffer tier value."""
        assert MemoryTier.BUFFER.value == "buffer"

    def test_working_tier(self) -> None:
        """Test working tier value."""
        assert MemoryTier.WORKING.value == "working"

    def test_session_tier(self) -> None:
        """Test session tier value."""
        assert MemoryTier.SESSION.value == "session"

    def test_temp_tier(self) -> None:
        """Test temp tier value."""
        assert MemoryTier.TEMP.value == "temp"


class TestRedisSetupSettings:
    """Tests for RedisSetupSettings."""

    def test_default_buffer_ttl(self) -> None:
        """Test default buffer TTL."""
        settings = RedisSetupSettings()
        assert settings.buffer_ttl_seconds == 300

    def test_default_working_memory_ttl(self) -> None:
        """Test default working memory TTL."""
        settings = RedisSetupSettings()
        assert settings.working_memory_ttl_seconds == 3600

    def test_default_session_ttl(self) -> None:
        """Test default session TTL."""
        settings = RedisSetupSettings()
        assert settings.session_ttl_seconds == 86400

    def test_default_cache_ttl(self) -> None:
        """Test default cache TTL."""
        settings = RedisSetupSettings()
        assert settings.cache_ttl_seconds == 3600

    def test_verify_on_init_enabled(self) -> None:
        """Test verify on init is enabled by default."""
        settings = RedisSetupSettings()
        assert settings.verify_on_init is True


class TestTTLPolicy:
    """Tests for TTLPolicy dataclass."""

    def test_ttl_policy_creation(self) -> None:
        """Test creating a TTL policy."""
        policy = TTLPolicy(
            tier=MemoryTier.BUFFER,
            ttl_seconds=300,
            description="Buffer tier policy",
        )
        assert policy.tier == MemoryTier.BUFFER
        assert policy.ttl_seconds == 300
        assert policy.description == "Buffer tier policy"


class TestRedisKeyBuilder:
    """Tests for RedisKeyBuilder class."""

    def test_default_prefix(self) -> None:
        """Test default prefix is 'solace:'."""
        builder = RedisKeyBuilder()
        assert builder._prefix == "solace:"

    def test_custom_prefix(self) -> None:
        """Test custom prefix."""
        builder = RedisKeyBuilder(prefix="custom:")
        assert builder._prefix == "custom:"

    def test_session_key(self) -> None:
        """Test session key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.session_key("session123")
        assert key == "solace:session:session123"

    def test_working_memory_key(self) -> None:
        """Test working memory key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.working_memory_key("user123", "session456")
        assert key == "solace:wm:user123:session456"

    def test_context_key(self) -> None:
        """Test context key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.context_key("user789")
        assert key == "solace:ctx:user789"

    def test_cache_key(self) -> None:
        """Test cache key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.cache_key("profiles", "user123")
        assert key == "solace:cache:profiles:user123"

    def test_lock_key(self) -> None:
        """Test lock key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.lock_key("migration")
        assert key == "solace:lock:migration"

    def test_rate_limit_key(self) -> None:
        """Test rate limit key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.rate_limit_key("user123", "/api/chat")
        assert key == "solace:rl:user123:/api/chat"

    def test_pubsub_channel(self) -> None:
        """Test pubsub channel generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        channel = builder.pubsub_channel("notifications")
        assert channel == "solace:ps:notifications"

    def test_queue_key(self) -> None:
        """Test queue key generation."""
        builder = RedisKeyBuilder(prefix="solace:")
        key = builder.queue_key("tasks")
        assert key == "solace:q:tasks"


class TestRedisSetupManager:
    """Tests for RedisSetupManager class."""

    def test_manager_initialization(self) -> None:
        """Test RedisSetupManager can be initialized."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        manager = RedisSetupManager(mock_client)
        assert manager is not None

    def test_manager_with_settings(self) -> None:
        """Test RedisSetupManager with custom settings."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        settings = RedisSetupSettings(buffer_ttl_seconds=600)
        manager = RedisSetupManager(mock_client, settings)
        assert manager._settings.buffer_ttl_seconds == 600

    def test_key_builder_property(self) -> None:
        """Test key_builder property returns RedisKeyBuilder."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        manager = RedisSetupManager(mock_client)
        assert isinstance(manager.key_builder, RedisKeyBuilder)

    def test_get_ttl_buffer(self) -> None:
        """Test get_ttl for buffer tier."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        manager = RedisSetupManager(mock_client)
        ttl = manager.get_ttl(MemoryTier.BUFFER)
        assert ttl == 300

    def test_get_ttl_working(self) -> None:
        """Test get_ttl for working tier."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        manager = RedisSetupManager(mock_client)
        ttl = manager.get_ttl(MemoryTier.WORKING)
        assert ttl == 3600

    def test_get_ttl_session(self) -> None:
        """Test get_ttl for session tier."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        manager = RedisSetupManager(mock_client)
        ttl = manager.get_ttl(MemoryTier.SESSION)
        assert ttl == 86400


class TestSetupRedis:
    """Tests for setup_redis function."""

    @pytest.mark.asyncio
    async def test_setup_returns_manager(self) -> None:
        """Test setup returns RedisSetupManager."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        mock_client.check_health = AsyncMock(return_value={"status": "healthy"})

        manager = await setup_redis(mock_client)
        assert isinstance(manager, RedisSetupManager)

    @pytest.mark.asyncio
    async def test_setup_with_settings(self) -> None:
        """Test setup with custom settings."""
        mock_client = MagicMock()
        mock_client.key_prefix = "solace:"
        mock_client.check_health = AsyncMock(return_value={"status": "healthy"})
        settings = RedisSetupSettings(cache_ttl_seconds=7200)

        manager = await setup_redis(mock_client, settings)
        assert manager._settings.cache_ttl_seconds == 7200

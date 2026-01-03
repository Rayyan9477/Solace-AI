"""Unit tests for Redis client module."""
from __future__ import annotations
import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from solace_infrastructure.redis import (
    RedisClient,
    RedisSettings,
    RedisMode,
    CacheEntry,
    create_redis_client,
)


class TestRedisSettings:
    """Tests for RedisSettings configuration."""

    def test_default_settings(self):
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.db == 0
        assert settings.mode == RedisMode.STANDALONE
        assert settings.key_prefix == "solace:"

    def test_get_url_no_auth(self):
        settings = RedisSettings()
        url = settings.get_url()
        assert url == "redis://localhost:6379/0"

    def test_get_url_with_ssl(self):
        settings = RedisSettings(ssl=True)
        url = settings.get_url()
        assert url.startswith("rediss://")

    def test_custom_settings(self):
        settings = RedisSettings(
            host="redis.example.com", port=6380, db=5,
            mode=RedisMode.CLUSTER, key_prefix="custom:"
        )
        assert settings.host == "redis.example.com"
        assert settings.port == 6380
        assert settings.db == 5
        assert settings.mode == RedisMode.CLUSTER


class TestRedisMode:
    """Tests for RedisMode enum."""

    def test_modes(self):
        assert RedisMode.STANDALONE.value == "standalone"
        assert RedisMode.SENTINEL.value == "sentinel"
        assert RedisMode.CLUSTER.value == "cluster"


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_cache_entry_creation(self):
        entry = CacheEntry(value={"key": "value"}, created_at=1234567890.0, ttl_seconds=3600)
        assert entry.value == {"key": "value"}
        assert entry.ttl_seconds == 3600


class TestRedisClient:
    """Tests for RedisClient operations."""

    @pytest.fixture
    def mock_redis(self):
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        redis_mock.get = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.delete = AsyncMock()
        redis_mock.exists = AsyncMock()
        redis_mock.expire = AsyncMock()
        redis_mock.ttl = AsyncMock()
        redis_mock.incrby = AsyncMock()
        redis_mock.hget = AsyncMock()
        redis_mock.hset = AsyncMock()
        redis_mock.hgetall = AsyncMock()
        redis_mock.hdel = AsyncMock()
        redis_mock.lpush = AsyncMock()
        redis_mock.rpush = AsyncMock()
        redis_mock.lrange = AsyncMock()
        redis_mock.publish = AsyncMock()
        redis_mock.info = AsyncMock()
        redis_mock.aclose = AsyncMock()
        return redis_mock

    @pytest.fixture
    def client(self):
        return RedisClient(RedisSettings())

    def test_initial_state(self, client):
        assert not client.is_connected
        assert client._client is None

    def test_key_prefix(self, client):
        assert client.key_prefix == "solace:"

    def test_make_key(self, client):
        key = client._make_key("test_key")
        assert key == "solace:test_key"

    def test_make_key_already_prefixed(self, client):
        key = client._make_key("solace:test_key")
        assert key == "solace:test_key"

    @pytest.mark.asyncio
    async def test_connect_standalone(self, client, mock_redis):
        with patch("redis.asyncio.Redis", return_value=mock_redis):
            await client.connect()
            assert client.is_connected
            mock_redis.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_redis):
        client._client = mock_redis
        await client.disconnect()
        assert not client.is_connected
        mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.get.return_value = '{"key": "value"}'
        result = await client.get("test")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_none(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.get.return_value = None
        result = await client.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set(self, client, mock_redis):
        client._client = mock_redis
        result = await client.set("test", {"key": "value"}, ttl=3600)
        assert result is True
        mock_redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_with_timedelta(self, client, mock_redis):
        client._client = mock_redis
        result = await client.set("test", "value", ttl=timedelta(hours=1))
        assert result is True

    @pytest.mark.asyncio
    async def test_delete(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.delete.return_value = 1
        result = await client.delete("key1", "key2")
        assert result == 1

    @pytest.mark.asyncio
    async def test_exists(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.exists.return_value = 2
        result = await client.exists("key1", "key2")
        assert result == 2

    @pytest.mark.asyncio
    async def test_expire(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.expire.return_value = True
        result = await client.expire("test", 3600)
        assert result is True

    @pytest.mark.asyncio
    async def test_expire_timedelta(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.expire.return_value = True
        result = await client.expire("test", timedelta(hours=1))
        assert result is True

    @pytest.mark.asyncio
    async def test_ttl(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.ttl.return_value = 1800
        result = await client.ttl("test")
        assert result == 1800

    @pytest.mark.asyncio
    async def test_incr(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.incrby.return_value = 5
        result = await client.incr("counter", 5)
        assert result == 5

    @pytest.mark.asyncio
    async def test_hget(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.hget.return_value = '"field_value"'
        result = await client.hget("hash", "field")
        assert result == "field_value"

    @pytest.mark.asyncio
    async def test_hget_none(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.hget.return_value = None
        result = await client.hget("hash", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_hset(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.hset.return_value = 2
        result = await client.hset("hash", {"field1": "value1", "field2": "value2"})
        assert result == 2

    @pytest.mark.asyncio
    async def test_hgetall(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.hgetall.return_value = {"field1": '"value1"', "field2": '"value2"'}
        result = await client.hgetall("hash")
        assert result["field1"] == "value1"

    @pytest.mark.asyncio
    async def test_hdel(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.hdel.return_value = 1
        result = await client.hdel("hash", "field")
        assert result == 1

    @pytest.mark.asyncio
    async def test_lpush(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.lpush.return_value = 3
        result = await client.lpush("list", "a", "b", "c")
        assert result == 3

    @pytest.mark.asyncio
    async def test_rpush(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.rpush.return_value = 3
        result = await client.rpush("list", "a", "b", "c")
        assert result == 3

    @pytest.mark.asyncio
    async def test_lrange(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.lrange.return_value = ['"a"', '"b"', '"c"']
        result = await client.lrange("list", 0, -1)
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_publish(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.publish.return_value = 2
        result = await client.publish("channel", {"event": "test"})
        assert result == 2

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, client, mock_redis):
        client._client = mock_redis
        mock_redis.info.return_value = {"redis_version": "7.0.0", "connected_clients": "10"}
        health = await client.check_health()
        assert health["status"] == "healthy"
        assert health["redis_version"] == "7.0.0"

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, client):
        client._client = None
        health = await client.check_health()
        assert health["status"] == "unhealthy"


class TestSerialization:
    """Tests for serialization methods."""

    @pytest.fixture
    def client(self):
        return RedisClient()

    def test_serialize_string(self, client):
        result = client._serialize("test")
        assert result == "test"

    def test_serialize_dict(self, client):
        result = client._serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_serialize_list(self, client):
        result = client._serialize([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_deserialize_json(self, client):
        result = client._deserialize('{"key": "value"}')
        assert result == {"key": "value"}

    def test_deserialize_string(self, client):
        result = client._deserialize("plain string")
        assert result == "plain string"

    def test_deserialize_bytes(self, client):
        result = client._deserialize(b'{"key": "value"}')
        assert result == {"key": "value"}


class TestFactoryFunction:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_redis_client(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        with patch("redis.asyncio.Redis", return_value=mock_redis):
            client = await create_redis_client(RedisSettings())
            assert client.is_connected
            mock_redis.ping.assert_awaited_once()

"""Unit tests for PostgreSQL client module."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
from solace_infrastructure.postgres import (
    PostgresClient,
    PostgresSettings,
    PostgresRepository,
    QueryResult,
    IsolationLevel,
    create_postgres_client,
    _truncate_query,
    _json_encoder,
    _json_decoder,
)


class TestPostgresSettings:
    """Tests for PostgresSettings configuration."""

    def test_default_settings(self):
        settings = PostgresSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "solace"
        assert settings.min_pool_size == 5
        assert settings.max_pool_size == 20

    def test_get_dsn(self):
        settings = PostgresSettings(host="db.example.com", port=5433, database="testdb", user="admin")
        dsn = settings.get_dsn()
        assert "db.example.com:5433" in dsn
        assert "testdb" in dsn
        assert "admin" in dsn

    def test_custom_settings(self):
        settings = PostgresSettings(
            host="custom-host", port=5434, database="custom_db",
            min_pool_size=10, max_pool_size=50, command_timeout=120.0
        )
        assert settings.host == "custom-host"
        assert settings.port == 5434
        assert settings.min_pool_size == 10
        assert settings.command_timeout == 120.0


class TestQueryResult:
    """Tests for QueryResult container."""

    def test_empty_result(self):
        result = QueryResult()
        assert result.rows == []
        assert result.row_count == 0
        assert result.first is None
        assert result.scalar is None

    def test_result_with_rows(self):
        result = QueryResult(
            rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            row_count=2,
            execution_time_ms=15.5
        )
        assert len(result.rows) == 2
        assert result.first == {"id": 1, "name": "Alice"}
        assert result.row_count == 2

    def test_scalar_extraction(self):
        result = QueryResult(rows=[{"count": 42}], row_count=1)
        assert result.scalar == 42


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_isolation_levels(self):
        assert IsolationLevel.READ_COMMITTED.value == "read_committed"
        assert IsolationLevel.REPEATABLE_READ.value == "repeatable_read"
        assert IsolationLevel.SERIALIZABLE.value == "serializable"


class MockAsyncContextManager:
    """Helper mock for async context managers."""
    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestPostgresClient:
    """Tests for PostgresClient operations."""

    @pytest.fixture
    def mock_conn(self):
        conn = AsyncMock()
        return conn

    @pytest.fixture
    def mock_pool(self, mock_conn):
        pool = MagicMock()
        pool.get_size.return_value = 10
        pool.get_idle_size.return_value = 5
        pool.get_min_size.return_value = 5
        pool.get_max_size.return_value = 20
        pool.close = AsyncMock()
        pool.acquire.return_value = MockAsyncContextManager(mock_conn)
        return pool

    @pytest.fixture
    def client(self):
        return PostgresClient(PostgresSettings())

    def test_initial_state(self, client):
        assert not client.is_connected
        assert client._pool is None

    @pytest.mark.asyncio
    async def test_connect(self, client, mock_pool):
        with patch("solace_infrastructure.postgres.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
            await client.connect()
            assert client.is_connected
            assert client._pool is mock_pool

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, client, mock_pool):
        client._pool = mock_pool
        with patch("solace_infrastructure.postgres.asyncpg.create_pool") as mock_create:
            await client.connect()
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_pool):
        client._pool = mock_pool
        client._connected = True
        await client.disconnect()
        assert not client.is_connected
        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.fetch.return_value = [{"id": 1, "name": "Test"}]
        result = await client.fetch("SELECT * FROM users WHERE id = $1", 1)
        assert result.row_count == 1
        assert result.first["name"] == "Test"

    @pytest.mark.asyncio
    async def test_fetch_one(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.fetchrow.return_value = {"id": 1, "name": "Single"}
        result = await client.fetch_one("SELECT * FROM users WHERE id = $1", 1)
        assert result["name"] == "Single"

    @pytest.mark.asyncio
    async def test_fetch_one_not_found(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.fetchrow.return_value = None
        result = await client.fetch_one("SELECT * FROM users WHERE id = $1", 999)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_val(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.fetchval.return_value = 42
        result = await client.fetch_val("SELECT COUNT(*) FROM users")
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.execute.return_value = "INSERT 0 1"
        result = await client.execute("INSERT INTO users (name) VALUES ($1)", "Alice")
        assert result == "INSERT 0 1"

    @pytest.mark.asyncio
    async def test_execute_many(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        args_list = [("Alice",), ("Bob",), ("Charlie",)]
        await client.execute_many("INSERT INTO users (name) VALUES ($1)", args_list)
        mock_conn.executemany.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, client, mock_pool, mock_conn):
        client._pool = mock_pool
        client._connected = True
        mock_conn.fetchrow.return_value = {"health": 1, "server_time": datetime.now(timezone.utc)}
        health = await client.check_health()
        assert health["status"] == "healthy"
        assert "pool_size" in health

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, client):
        client._pool = None
        health = await client.check_health()
        assert health["status"] == "unhealthy"


class TestPostgresRepository:
    """Tests for PostgresRepository base class."""

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=PostgresClient)
        return client

    @pytest.fixture
    def repository(self, mock_client):
        return PostgresRepository(mock_client, "users", "public")

    def test_qualified_table(self, repository):
        assert repository.qualified_table == "public.users"

    @pytest.mark.asyncio
    async def test_find_by_id(self, repository, mock_client):
        entity_id = uuid4()
        mock_client.fetch_one.return_value = {"id": entity_id, "name": "Test"}
        result = await repository.find_by_id(entity_id)
        assert result["name"] == "Test"

    @pytest.mark.asyncio
    async def test_find_all(self, repository, mock_client):
        mock_client.fetch.return_value = QueryResult(
            rows=[{"id": 1}, {"id": 2}], row_count=2
        )
        result = await repository.find_all(limit=10, offset=0)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_exists(self, repository, mock_client):
        entity_id = uuid4()
        mock_client.fetch_val.return_value = True
        result = await repository.exists(entity_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_count(self, repository, mock_client):
        mock_client.fetch_val.return_value = 100
        result = await repository.count()
        assert result == 100

    @pytest.mark.asyncio
    async def test_count_with_where(self, repository, mock_client):
        mock_client.fetch_val.return_value = 5
        result = await repository.count("status = $1", "active")
        assert result == 5

    @pytest.mark.asyncio
    async def test_insert(self, repository, mock_client):
        data = {"name": "New User", "email": "new@example.com"}
        mock_client.fetch_one.return_value = {"id": uuid4(), **data}
        result = await repository.insert(data)
        assert result["name"] == "New User"

    @pytest.mark.asyncio
    async def test_update(self, repository, mock_client):
        entity_id = uuid4()
        mock_client.fetch_one.return_value = {"id": entity_id, "name": "Updated"}
        result = await repository.update(entity_id, {"name": "Updated"})
        assert result["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete(self, repository, mock_client):
        entity_id = uuid4()
        mock_client.execute.return_value = "DELETE 1"
        result = await repository.delete(entity_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_soft_delete(self, repository, mock_client):
        entity_id = uuid4()
        mock_client.execute.return_value = "UPDATE 1"
        result = await repository.soft_delete(entity_id)
        assert result is True


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_query_short(self):
        query = "SELECT * FROM users"
        assert _truncate_query(query) == query

    def test_truncate_query_long(self):
        query = "SELECT " + "a, " * 100 + "b FROM very_long_table"
        result = _truncate_query(query, max_length=50)
        assert len(result) <= 53
        assert result.endswith("...")

    def test_json_encoder(self):
        data = {"key": "value", "date": datetime(2024, 1, 1)}
        result = _json_encoder(data)
        assert '"key": "value"' in result

    def test_json_decoder(self):
        json_str = '{"key": "value", "count": 42}'
        result = _json_decoder(json_str)
        assert result["key"] == "value"
        assert result["count"] == 42


class TestFactoryFunction:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_postgres_client(self):
        mock_pool = MagicMock()
        mock_pool.get_size.return_value = 5
        with patch("solace_infrastructure.postgres.asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
            client = await create_postgres_client(PostgresSettings())
            assert client.is_connected

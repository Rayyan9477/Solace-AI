"""Unit tests for Solace-AI Testing Library - Fixtures module."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from solace_testing.fixtures import (
    DatabaseFixture,
    FixtureConfig,
    FixtureContext,
    HTTPClientFixture,
    KafkaFixture,
    MockPostgresConnection,
    ObservabilityFixture,
    PostgresFixture,
    RedisFixture,
    WeaviateFixture,
)


class TestFixtureConfig:
    """Tests for FixtureConfig."""

    def test_default_values(self) -> None:
        config = FixtureConfig()
        assert config.postgres_dsn == "postgresql://test:test@localhost:5432/test_db"
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.cleanup_on_teardown is True

    def test_custom_values(self) -> None:
        config = FixtureConfig(
            postgres_dsn="postgresql://custom:pass@db:5432/custom",
            use_testcontainers=True,
        )
        assert "custom" in config.postgres_dsn
        assert config.use_testcontainers is True


class TestFixtureContext:
    """Tests for FixtureContext."""

    def test_create_context(self) -> None:
        context = FixtureContext.create("test_example")
        assert context.test_name == "test_example"
        assert context.test_id is not None
        assert context.correlation_id.startswith("test-")
        assert context.started_at <= datetime.now(timezone.utc)

    def test_create_with_config(self) -> None:
        config = FixtureConfig(use_testcontainers=True)
        context = FixtureContext.create("test_with_config", config)
        assert context.config.use_testcontainers is True


class TestDatabaseFixture:
    """Tests for DatabaseFixture."""

    def test_fixture_creation(self) -> None:
        context = FixtureContext.create("test_db")
        fixture = DatabaseFixture(context)
        assert fixture._in_transaction is False
        assert len(fixture._savepoints) == 0

    @pytest.mark.asyncio
    async def test_transaction_management(self) -> None:
        context = FixtureContext.create("test_transaction")
        fixture = DatabaseFixture(context)
        await fixture.begin_transaction()
        assert fixture._in_transaction is True
        await fixture.rollback_transaction()
        assert fixture._in_transaction is False

    @pytest.mark.asyncio
    async def test_savepoints(self) -> None:
        context = FixtureContext.create("test_savepoint")
        fixture = DatabaseFixture(context)
        await fixture.begin_transaction()
        await fixture.begin_transaction()
        assert len(fixture._savepoints) == 1
        await fixture.rollback_transaction()
        assert len(fixture._savepoints) == 0


class TestPostgresFixture:
    """Tests for PostgresFixture."""

    @pytest.mark.asyncio
    async def test_setup_teardown(self) -> None:
        context = FixtureContext.create("test_postgres")
        fixture = PostgresFixture(context)
        await fixture.setup()
        assert fixture._connection is not None
        await fixture.teardown()
        assert fixture._connection is None

    @pytest.mark.asyncio
    async def test_execute_query(self) -> None:
        context = FixtureContext.create("test_query")
        fixture = PostgresFixture(context)
        await fixture.setup()
        result = await fixture.execute("SELECT * FROM test")
        assert isinstance(result, list)
        await fixture.teardown()


class TestMockPostgresConnection:
    """Tests for MockPostgresConnection."""

    @pytest.mark.asyncio
    async def test_execute(self) -> None:
        context = FixtureContext.create("test_mock_conn")
        conn = MockPostgresConnection(context)
        result = await conn.execute("SELECT 1")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        context = FixtureContext.create("test_close")
        conn = MockPostgresConnection(context)
        await conn.close()
        with pytest.raises(RuntimeError, match="closed"):
            await conn.execute("SELECT 1")


class TestRedisFixture:
    """Tests for RedisFixture."""

    @pytest.mark.asyncio
    async def test_get_set(self) -> None:
        context = FixtureContext.create("test_redis")
        fixture = RedisFixture(context)
        await fixture.setup()
        await fixture.set("key1", "value1")
        result = await fixture.get("key1")
        assert result == "value1"
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        context = FixtureContext.create("test_delete")
        fixture = RedisFixture(context)
        await fixture.setup()
        await fixture.set("key1", "value1")
        deleted = await fixture.delete("key1")
        assert deleted is True
        result = await fixture.get("key1")
        assert result is None
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        context = FixtureContext.create("test_exists")
        fixture = RedisFixture(context)
        await fixture.setup()
        await fixture.set("key1", "value1")
        assert await fixture.exists("key1") is True
        assert await fixture.exists("key2") is False
        await fixture.teardown()


class TestWeaviateFixture:
    """Tests for WeaviateFixture."""

    @pytest.mark.asyncio
    async def test_create_collection(self) -> None:
        context = FixtureContext.create("test_weaviate")
        fixture = WeaviateFixture(context)
        await fixture.setup()
        await fixture.create_collection("TestCollection", [{"name": "title", "type": "string"}])
        assert "TestCollection" in fixture._collections
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_insert_search(self) -> None:
        context = FixtureContext.create("test_search")
        fixture = WeaviateFixture(context)
        await fixture.setup()
        await fixture.create_collection("Test", [])
        obj_id = await fixture.insert("Test", {"title": "test"}, [0.1, 0.2, 0.3])
        assert obj_id is not None
        results = await fixture.search("Test", [0.1, 0.2, 0.3], limit=5)
        assert len(results) == 1
        await fixture.teardown()


class TestKafkaFixture:
    """Tests for KafkaFixture."""

    @pytest.mark.asyncio
    async def test_produce_consume(self) -> None:
        context = FixtureContext.create("test_kafka")
        fixture = KafkaFixture(context)
        await fixture.setup()
        offset = await fixture.produce("test-topic", {"key": "value"})
        assert offset == 0
        messages = await fixture.consume("test-topic", "test-group")
        assert len(messages) == 1
        assert messages[0]["value"]["key"] == "value"
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_get_messages(self) -> None:
        context = FixtureContext.create("test_messages")
        fixture = KafkaFixture(context)
        await fixture.setup()
        await fixture.produce("topic1", {"msg": 1})
        await fixture.produce("topic1", {"msg": 2})
        messages = fixture.get_messages("topic1")
        assert len(messages) == 2
        await fixture.teardown()


class TestHTTPClientFixture:
    """Tests for HTTPClientFixture."""

    @pytest.mark.asyncio
    async def test_mock_response(self) -> None:
        context = FixtureContext.create("test_http")
        fixture = HTTPClientFixture(context)
        await fixture.setup()
        fixture.mock_response("GET", "/api/test", status=200, json={"result": "ok"})
        response = await fixture.request("GET", "/api/test")
        assert response["status"] == 200
        assert response["json"]["result"] == "ok"
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_record_requests(self) -> None:
        context = FixtureContext.create("test_requests")
        fixture = HTTPClientFixture(context)
        await fixture.setup()
        await fixture.request("POST", "/api/create", json={"name": "test"})
        requests = fixture.get_requests()
        assert len(requests) == 1
        assert requests[0]["method"] == "POST"
        await fixture.teardown()


class TestObservabilityFixture:
    """Tests for ObservabilityFixture."""

    @pytest.mark.asyncio
    async def test_logging(self) -> None:
        context = FixtureContext.create("test_obs")
        fixture = ObservabilityFixture(context)
        await fixture.setup()
        fixture.log("info", "Test message", key="value")
        logs = fixture.get_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "Test message"
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_metrics(self) -> None:
        context = FixtureContext.create("test_metrics")
        fixture = ObservabilityFixture(context)
        await fixture.setup()
        fixture.record_metric("test.counter", 1.0)
        fixture.record_metric("test.counter", 2.0)
        values = fixture.get_metric_values("test.counter")
        assert values == [1.0, 2.0]
        await fixture.teardown()

    @pytest.mark.asyncio
    async def test_spans(self) -> None:
        context = FixtureContext.create("test_spans")
        fixture = ObservabilityFixture(context)
        await fixture.setup()
        span_id = fixture.start_span("test-operation")
        assert span_id is not None
        fixture.end_span(span_id)
        await fixture.teardown()

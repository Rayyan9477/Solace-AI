"""Solace-AI Testing Library - Common pytest fixtures."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, TypeVar

import pytest
import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class FixtureConfig(BaseModel):
    """Configuration for test fixtures."""
    postgres_dsn: str = "postgresql://test:test@localhost:5432/test_db"
    redis_url: str = "redis://localhost:6379/0"
    weaviate_url: str = "http://localhost:8080"
    kafka_bootstrap: str = "localhost:9092"
    use_testcontainers: bool = False
    cleanup_on_teardown: bool = True
    isolation_level: str = "READ_COMMITTED"


class FixtureContext(BaseModel):
    """Context passed to test fixtures."""
    test_id: str
    test_name: str
    correlation_id: str
    started_at: datetime
    config: FixtureConfig
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def create(cls, test_name: str, config: FixtureConfig | None = None) -> FixtureContext:
        return cls(
            test_id=str(uuid.uuid4()),
            test_name=test_name,
            correlation_id=f"test-{uuid.uuid4().hex[:8]}",
            started_at=datetime.now(timezone.utc),
            config=config or FixtureConfig(),
        )


class DatabaseFixture:
    """Base class for database fixtures with transaction management."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._in_transaction = False
        self._savepoints: list[str] = []

    async def begin_transaction(self) -> None:
        if self._in_transaction:
            savepoint = f"sp_{uuid.uuid4().hex[:8]}"
            self._savepoints.append(savepoint)
            logger.debug("Created savepoint", savepoint=savepoint, test=self.context.test_name)
        else:
            self._in_transaction = True
            logger.debug("Began transaction", test=self.context.test_name)

    async def rollback_transaction(self) -> None:
        if self._savepoints:
            savepoint = self._savepoints.pop()
            logger.debug("Rolled back to savepoint", savepoint=savepoint)
        elif self._in_transaction:
            self._in_transaction = False
            logger.debug("Rolled back transaction", test=self.context.test_name)

    async def commit_transaction(self) -> None:
        if self._savepoints:
            self._savepoints.pop()
        elif self._in_transaction:
            self._in_transaction = False
            logger.debug("Committed transaction", test=self.context.test_name)


class PostgresFixture(DatabaseFixture):
    """PostgreSQL database fixture with connection pooling."""

    def __init__(self, context: FixtureContext) -> None:
        super().__init__(context)
        self._pool: Any | None = None
        self._connection: Any | None = None

    async def get_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("PostgresFixture not initialized. Call setup() first.")
        return self._connection

    async def setup(self) -> None:
        logger.info("Setting up PostgreSQL fixture", dsn=self.context.config.postgres_dsn, test=self.context.test_name)
        self._connection = MockPostgresConnection(self.context)
        await self.begin_transaction()

    async def teardown(self) -> None:
        await self.rollback_transaction()
        if self._connection:
            await self._connection.close()
            self._connection = None
        logger.info("Tore down PostgreSQL fixture", test=self.context.test_name)

    async def execute(self, query: str, *args: Any) -> list[dict[str, Any]]:
        conn = await self.get_connection()
        return await conn.execute(query, *args)

    async def execute_many(self, query: str, args_list: list[tuple[Any, ...]]) -> int:
        conn = await self.get_connection()
        return await conn.execute_many(query, args_list)


class MockPostgresConnection:
    """Mock PostgreSQL connection for testing."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._closed = False

    async def execute(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Connection is closed")
        logger.debug("Mock execute", query=query[:100], args=args)
        return []

    async def execute_many(self, query: str, args_list: list[tuple[Any, ...]]) -> int:
        if self._closed:
            raise RuntimeError("Connection is closed")
        return len(args_list)

    async def close(self) -> None:
        self._closed = True


class RedisFixture:
    """Redis cache fixture with automatic cleanup."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._data: dict[str, Any] = {}
        self._ttls: dict[str, float] = {}
        self._prefix = f"test:{context.test_id}:"

    async def setup(self) -> None:
        logger.info("Setting up Redis fixture", test=self.context.test_name)

    async def teardown(self) -> None:
        self._data.clear()
        self._ttls.clear()
        logger.info("Tore down Redis fixture", test=self.context.test_name)

    async def get(self, key: str) -> Any | None:
        return self._data.get(f"{self._prefix}{key}")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        full_key = f"{self._prefix}{key}"
        self._data[full_key] = value
        if ttl:
            self._ttls[full_key] = ttl

    async def delete(self, key: str) -> bool:
        full_key = f"{self._prefix}{key}"
        if full_key in self._data:
            del self._data[full_key]
            self._ttls.pop(full_key, None)
            return True
        return False

    async def exists(self, key: str) -> bool:
        return f"{self._prefix}{key}" in self._data

    async def flush(self) -> None:
        keys = [k for k in self._data if k.startswith(self._prefix)]
        for key in keys:
            del self._data[key]
            self._ttls.pop(key, None)


class WeaviateFixture:
    """Weaviate vector database fixture."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._collections: dict[str, list[dict[str, Any]]] = {}

    async def setup(self) -> None:
        logger.info("Setting up Weaviate fixture", test=self.context.test_name)

    async def teardown(self) -> None:
        self._collections.clear()
        logger.info("Tore down Weaviate fixture", test=self.context.test_name)

    async def create_collection(self, name: str, properties: list[dict[str, Any]]) -> None:
        self._collections[name] = []
        logger.debug("Created collection", name=name, properties=len(properties))

    async def insert(self, collection: str, obj: dict[str, Any], vector: list[float]) -> str:
        obj_id = str(uuid.uuid4())
        self._collections.setdefault(collection, []).append({"id": obj_id, "properties": obj, "vector": vector})
        return obj_id

    async def search(self, collection: str, vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
        return self._collections.get(collection, [])[:limit]

    async def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)


class KafkaFixture:
    """Kafka event bus fixture."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._topics: dict[str, list[dict[str, Any]]] = {}
        self._consumer_offsets: dict[str, int] = {}

    async def setup(self) -> None:
        logger.info("Setting up Kafka fixture", test=self.context.test_name)

    async def teardown(self) -> None:
        self._topics.clear()
        self._consumer_offsets.clear()
        logger.info("Tore down Kafka fixture", test=self.context.test_name)

    async def produce(self, topic: str, message: dict[str, Any], key: str | None = None) -> int:
        self._topics.setdefault(topic, []).append({"key": key, "value": message})
        offset = len(self._topics[topic]) - 1
        logger.debug("Produced message", topic=topic, offset=offset)
        return offset

    async def consume(self, topic: str, group_id: str, timeout: float = 1.0) -> list[dict[str, Any]]:
        offset_key = f"{group_id}:{topic}"
        current_offset = self._consumer_offsets.get(offset_key, 0)
        messages = self._topics.get(topic, [])[current_offset:]
        self._consumer_offsets[offset_key] = current_offset + len(messages)
        return messages

    async def create_topic(self, name: str, partitions: int = 1) -> None:
        self._topics.setdefault(name, [])

    def get_messages(self, topic: str) -> list[dict[str, Any]]:
        return self._topics.get(topic, [])


class HTTPClientFixture:
    """HTTP client fixture for API testing."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._responses: dict[str, dict[str, Any]] = {}
        self._requests: list[dict[str, Any]] = []

    async def setup(self) -> None:
        logger.info("Setting up HTTP client fixture", test=self.context.test_name)

    async def teardown(self) -> None:
        self._responses.clear()
        self._requests.clear()
        logger.info("Tore down HTTP client fixture", test=self.context.test_name)

    def mock_response(self, method: str, url: str, status: int = 200,
                      json: dict[str, Any] | None = None, text: str | None = None) -> None:
        self._responses[f"{method.upper()}:{url}"] = {"status": status, "json": json, "text": text}

    async def request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        key = f"{method.upper()}:{url}"
        self._requests.append({"method": method, "url": url, **kwargs})
        return self._responses.get(key, {"status": 404, "json": None, "text": "Not Found"})

    def get_requests(self) -> list[dict[str, Any]]:
        return self._requests.copy()


class ObservabilityFixture:
    """Observability fixture for logging, metrics, and tracing."""

    def __init__(self, context: FixtureContext) -> None:
        self.context = context
        self._logs: list[dict[str, Any]] = []
        self._metrics: dict[str, list[float]] = {}
        self._spans: list[dict[str, Any]] = []

    async def setup(self) -> None:
        logger.info("Setting up observability fixture", test=self.context.test_name)

    async def teardown(self) -> None:
        self._logs.clear()
        self._metrics.clear()
        self._spans.clear()
        logger.info("Tore down observability fixture", test=self.context.test_name)

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        self._logs.append({"level": level, "message": message, **kwargs})

    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self._metrics.setdefault(name, []).append(value)

    def start_span(self, name: str, parent_id: str | None = None) -> str:
        span_id = str(uuid.uuid4())
        self._spans.append({"id": span_id, "name": name, "parent_id": parent_id, "ended": False})
        return span_id

    def end_span(self, span_id: str) -> None:
        for span in self._spans:
            if span["id"] == span_id:
                span["ended"] = True
                break

    def get_logs(self, level: str | None = None) -> list[dict[str, Any]]:
        if level:
            return [log for log in self._logs if log["level"] == level]
        return self._logs.copy()

    def get_metric_values(self, name: str) -> list[float]:
        return self._metrics.get(name, [])


@asynccontextmanager
async def fixture_scope(context: FixtureContext) -> AsyncIterator[dict[str, Any]]:
    """Context manager for managing fixture lifecycle."""
    fixtures: dict[str, Any] = {}
    try:
        yield fixtures
    finally:
        for name, fixture in reversed(list(fixtures.items())):
            if hasattr(fixture, "teardown"):
                try:
                    await fixture.teardown()
                except Exception as e:
                    logger.error("Fixture teardown failed", fixture=name, error=str(e))


@contextmanager
def sync_fixture_scope(context: FixtureContext) -> Iterator[dict[str, Any]]:
    """Synchronous context manager for fixture lifecycle."""
    fixtures: dict[str, Any] = {}
    try:
        yield fixtures
    finally:
        for name, fixture in reversed(list(fixtures.items())):
            if hasattr(fixture, "sync_teardown"):
                try:
                    fixture.sync_teardown()
                except Exception as e:
                    logger.error("Fixture teardown failed", fixture=name, error=str(e))


def pytest_fixture_factory(fixture_class: type[T], scope: str = "function") -> Any:
    """Factory for creating pytest fixtures from fixture classes."""
    @pytest.fixture(scope=scope)
    async def _fixture(request: Any) -> AsyncIterator[T]:
        context = FixtureContext.create(test_name=request.node.name, config=getattr(request, "param", None))
        fixture = fixture_class(context)
        await fixture.setup()
        try:
            yield fixture
        finally:
            await fixture.teardown()
    return _fixture

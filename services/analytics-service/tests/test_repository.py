"""
Unit tests for analytics repository.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4

from repository import (
    ClickHouseConfig,
    AnalyticsRepository,
    ClickHouseRepository,
    InMemoryRepository,
    RepositoryError,
    ConnectionError,
    QueryError,
    create_repository,
)
from models import (
    TableName,
    AnalyticsEvent,
    MetricRecord,
    SessionRecord,
    AggregationRecord,
)


class TestClickHouseConfig:
    """Tests for ClickHouseConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClickHouseConfig()

        assert config.host == "localhost"
        assert config.port == 8123
        assert config.database == "solace_analytics"
        assert config.username == "default"
        assert config.password == ""
        assert config.secure is False
        assert config.verify is True
        assert config.connect_timeout == 10.0
        assert config.query_timeout == 300.0
        assert config.max_connections == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClickHouseConfig(
            host="clickhouse.example.com",
            port=8443,
            database="analytics",
            username="admin",
            password="secret",
            secure=True,
        )

        assert config.host == "clickhouse.example.com"
        assert config.port == 8443
        assert config.secure is True


class TestTableName:
    """Tests for TableName enum."""

    def test_all_tables_exist(self):
        """Test all expected table names exist."""
        assert TableName.EVENTS == "analytics_events"
        assert TableName.METRICS == "analytics_metrics"
        assert TableName.SESSIONS == "analytics_sessions"
        assert TableName.AGGREGATIONS == "analytics_aggregations"


class TestAnalyticsEvent:
    """Tests for AnalyticsEvent model."""

    def test_create_event(self):
        """Test creating analytics event."""
        event_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()
        correlation_id = uuid4()
        timestamp = datetime.now(timezone.utc)

        event = AnalyticsEvent(
            event_id=event_id,
            event_type="session.started",
            category="session",
            user_id=user_id,
            session_id=session_id,
            timestamp=timestamp,
            correlation_id=correlation_id,
            source_service="orchestrator-service",
            payload={"channel": "web"},
        )

        assert event.event_id == event_id
        assert event.event_type == "session.started"
        assert event.category == "session"
        assert event.user_id == user_id
        assert event.session_id == session_id
        assert event.payload == {"channel": "web"}


class TestMetricRecord:
    """Tests for MetricRecord model."""

    def test_create_metric(self):
        """Test creating metric record."""
        metric_id = uuid4()
        timestamp = datetime.now(timezone.utc)
        window_start = timestamp.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=1)

        metric = MetricRecord(
            metric_id=metric_id,
            metric_name="test.counter",
            value=Decimal("42.5"),
            labels={"service": "test"},
            timestamp=timestamp,
            window_start=window_start,
            window_end=window_end,
            window_type="hour",
        )

        assert metric.metric_id == metric_id
        assert metric.metric_name == "test.counter"
        assert metric.value == Decimal("42.5")
        assert metric.labels == {"service": "test"}
        assert metric.window_type == "hour"


class TestSessionRecord:
    """Tests for SessionRecord model."""

    def test_create_session(self):
        """Test creating session record."""
        session_id = uuid4()
        user_id = uuid4()
        started_at = datetime.now(timezone.utc)

        session = SessionRecord(
            session_id=session_id,
            user_id=user_id,
            started_at=started_at,
            message_count=10,
            channel="mobile",
        )

        assert session.session_id == session_id
        assert session.user_id == user_id
        assert session.message_count == 10
        assert session.channel == "mobile"


class TestAggregationRecord:
    """Tests for AggregationRecord model."""

    def test_create_aggregation(self):
        """Test creating aggregation record."""
        aggregation_id = uuid4()
        window_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=1)

        aggregation = AggregationRecord(
            aggregation_id=aggregation_id,
            metric_name="test.metric",
            window_type="hour",
            window_start=window_start,
            window_end=window_end,
            count=100,
            sum_value=Decimal("1000"),
            min_value=Decimal("1"),
            max_value=Decimal("50"),
            avg_value=Decimal("10"),
        )

        assert aggregation.aggregation_id == aggregation_id
        assert aggregation.count == 100
        assert aggregation.sum_value == Decimal("1000")
        assert aggregation.avg_value == Decimal("10")


class TestInMemoryRepository:
    """Tests for InMemoryRepository."""

    @pytest.fixture
    def repository(self):
        """Create in-memory repository."""
        return InMemoryRepository()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, repository):
        """Test connection lifecycle."""
        await repository.connect()
        assert await repository.health_check() is True

        await repository.disconnect()
        assert await repository.health_check() is False

    @pytest.mark.asyncio
    async def test_insert_and_query_events(self, repository):
        """Test inserting and querying events."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        user_id = uuid4()

        events = [
            AnalyticsEvent(
                event_id=uuid4(),
                event_type="session.started",
                category="session",
                user_id=user_id,
                timestamp=now - timedelta(minutes=i),
                correlation_id=uuid4(),
                source_service="test",
            )
            for i in range(5)
        ]

        count = await repository.insert_events_batch(events)
        assert count == 5

        results = await repository.query_events(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_events_with_filters(self, repository):
        """Test querying events with filters."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        user1 = uuid4()
        user2 = uuid4()

        events = [
            AnalyticsEvent(
                event_id=uuid4(),
                event_type="session.started",
                category="session",
                user_id=user1,
                timestamp=now,
                correlation_id=uuid4(),
                source_service="test",
            ),
            AnalyticsEvent(
                event_id=uuid4(),
                event_type="session.ended",
                category="session",
                user_id=user1,
                timestamp=now,
                correlation_id=uuid4(),
                source_service="test",
            ),
            AnalyticsEvent(
                event_id=uuid4(),
                event_type="session.started",
                category="session",
                user_id=user2,
                timestamp=now,
                correlation_id=uuid4(),
                source_service="test",
            ),
        ]
        await repository.insert_events_batch(events)

        results = await repository.query_events(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
            event_type="session.started",
        )
        assert len(results) == 2

        results = await repository.query_events(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
            user_id=user1,
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_insert_and_query_metrics(self, repository):
        """Test inserting and querying metrics."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        window_start = now.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=1)

        metrics = [
            MetricRecord(
                metric_id=uuid4(),
                metric_name="test.metric",
                value=Decimal(str(i * 10)),
                timestamp=now,
                window_start=window_start,
                window_end=window_end,
                window_type="hour",
            )
            for i in range(1, 6)
        ]

        count = await repository.insert_metrics_batch(metrics)
        assert count == 5

        results = await repository.query_metrics(
            metric_name="test.metric",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_store_and_get_aggregations(self, repository):
        """Test storing and getting aggregations."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        window_start = now.replace(minute=0, second=0, microsecond=0)
        window_end = window_start + timedelta(hours=1)

        aggregation = AggregationRecord(
            aggregation_id=uuid4(),
            metric_name="test.metric",
            window_type="hour",
            window_start=window_start,
            window_end=window_end,
            count=100,
            sum_value=Decimal("1000"),
        )

        await repository.store_aggregation(aggregation)

        results = await repository.get_aggregations(
            metric_name="test.metric",
            window_type="hour",
            start_time=window_start - timedelta(hours=1),
            end_time=window_end + timedelta(hours=1),
        )
        assert len(results) == 1
        assert results[0].count == 100

    @pytest.mark.asyncio
    async def test_insert_single_event(self, repository):
        """Test inserting single event."""
        await repository.connect()

        event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="test.event",
            category="test",
            user_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
        )

        await repository.insert_event(event)

        results = await repository.query_events(
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_insert_single_metric(self, repository):
        """Test inserting single metric."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        metric = MetricRecord(
            metric_id=uuid4(),
            metric_name="single.metric",
            value=Decimal("42"),
            timestamp=now,
            window_start=now.replace(minute=0, second=0, microsecond=0),
            window_end=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            window_type="hour",
        )

        await repository.insert_metric(metric)

        results = await repository.query_metrics(
            metric_name="single.metric",
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_with_limit(self, repository):
        """Test query with limit."""
        await repository.connect()

        now = datetime.now(timezone.utc)
        events = [
            AnalyticsEvent(
                event_id=uuid4(),
                event_type="test.event",
                category="test",
                user_id=uuid4(),
                timestamp=now - timedelta(minutes=i),
                correlation_id=uuid4(),
                source_service="test",
            )
            for i in range(10)
        ]
        await repository.insert_events_batch(events)

        results = await repository.query_events(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(minutes=1),
            limit=3,
        )
        assert len(results) == 3


class TestCreateRepository:
    """Tests for create_repository factory."""

    def test_create_inmemory_repository(self):
        """Test creating in-memory repository."""
        repo = create_repository()
        assert isinstance(repo, InMemoryRepository)

    def test_create_clickhouse_repository(self):
        """Test creating ClickHouse repository."""
        config = ClickHouseConfig()
        repo = create_repository(config)
        assert isinstance(repo, ClickHouseRepository)


class TestClickHouseRepository:
    """Tests for ClickHouseRepository (without actual connection)."""

    def test_create_repository(self):
        """Test creating ClickHouse repository."""
        config = ClickHouseConfig(
            host="localhost",
            port=8123,
            database="test_db",
        )
        repo = ClickHouseRepository(config)

        assert repo._config == config
        assert repo._connected is False

    @pytest.mark.asyncio
    async def test_operations_require_connection(self):
        """Test that operations require connection."""
        config = ClickHouseConfig()
        repo = ClickHouseRepository(config)

        event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="test",
            category="test",
            user_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
        )

        with pytest.raises(ConnectionError):
            await repo.insert_event(event)

        with pytest.raises(ConnectionError):
            await repo.query_events(
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                end_time=datetime.now(timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_health_check_when_not_connected(self):
        """Test health check returns false when not connected."""
        config = ClickHouseConfig()
        repo = ClickHouseRepository(config)

        result = await repo.health_check()
        assert result is False

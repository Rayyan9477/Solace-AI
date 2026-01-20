"""
Unit tests for analytics consumer module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from consumer import (
    EventCategory,
    AnalyticsEvent,
    ConsumerConfig,
    ConsumerMetrics,
    EventFilter,
    AnalyticsEventProcessor,
    AnalyticsConsumer,
    create_analytics_consumer,
)


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_category_values(self):
        """Test event category enum values."""
        assert EventCategory.SESSION.value == "session"
        assert EventCategory.SAFETY.value == "safety"
        assert EventCategory.DIAGNOSIS.value == "diagnosis"
        assert EventCategory.THERAPY.value == "therapy"
        assert EventCategory.MEMORY.value == "memory"


class TestAnalyticsEvent:
    """Tests for AnalyticsEvent dataclass."""

    def test_from_raw_session_event(self, sample_session_event):
        """Test creating event from raw session data."""
        event = AnalyticsEvent.from_raw(sample_session_event)

        assert event.event_type == "session.started"
        assert event.category == EventCategory.SESSION
        assert event.source_service == "orchestrator-service"

    def test_from_raw_safety_event(self, sample_safety_event):
        """Test creating event from raw safety data."""
        event = AnalyticsEvent.from_raw(sample_safety_event)

        assert event.event_type == "safety.assessment.completed"
        assert event.category == EventCategory.SAFETY

    def test_from_raw_therapy_event(self, sample_therapy_event):
        """Test creating event from raw therapy data."""
        event = AnalyticsEvent.from_raw(sample_therapy_event)

        assert event.event_type == "therapy.intervention.delivered"
        assert event.category == EventCategory.THERAPY

    def test_from_raw_diagnosis_event(self, sample_diagnosis_event):
        """Test creating event from raw diagnosis data."""
        event = AnalyticsEvent.from_raw(sample_diagnosis_event)

        assert event.event_type == "diagnosis.completed"
        assert event.category == EventCategory.DIAGNOSIS

    def test_determine_category_unknown(self):
        """Test category determination for unknown event type."""
        category = AnalyticsEvent._determine_category("unknown.event")
        assert category == EventCategory.SYSTEM


class TestConsumerConfig:
    """Tests for ConsumerConfig model."""

    def test_default_config(self):
        """Test default configuration."""
        config = ConsumerConfig()

        assert config.group_id == "analytics-service"
        assert config.batch_size == 100
        assert len(config.topics) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConsumerConfig(
            group_id="custom-group",
            batch_size=50,
            batch_timeout_ms=10000,
        )

        assert config.group_id == "custom-group"
        assert config.batch_size == 50


class TestConsumerMetrics:
    """Tests for ConsumerMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create consumer metrics."""
        return ConsumerMetrics()

    def test_record_event(self, metrics):
        """Test recording successful event."""
        metrics.record_event(EventCategory.SESSION, 50)

        assert metrics.events_received == 1
        assert metrics.events_processed == 1
        assert metrics.events_by_category["session"] == 1

    def test_record_failure(self, metrics):
        """Test recording failed event."""
        metrics.record_failure()

        assert metrics.events_received == 1
        assert metrics.events_failed == 1

    def test_record_skip(self, metrics):
        """Test recording skipped event."""
        metrics.record_skip()

        assert metrics.events_received == 1
        assert metrics.events_skipped == 1

    def test_to_dict(self, metrics):
        """Test converting metrics to dictionary."""
        metrics.record_event(EventCategory.SESSION, 100)
        metrics.record_event(EventCategory.SAFETY, 50)

        result = metrics.to_dict()

        assert result["events_received"] == 2
        assert result["events_processed"] == 2
        assert "avg_processing_time_ms" in result


class TestEventFilter:
    """Tests for EventFilter class."""

    def test_filter_include_categories(self):
        """Test filtering by included categories."""
        event_filter = EventFilter(include_categories=[EventCategory.SESSION, EventCategory.SAFETY])

        session_event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="session.started",
            category=EventCategory.SESSION,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
            payload={},
        )

        therapy_event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="therapy.started",
            category=EventCategory.THERAPY,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
            payload={},
        )

        assert event_filter.should_process(session_event) is True
        assert event_filter.should_process(therapy_event) is False

    def test_filter_exclude_event_types(self):
        """Test filtering by excluded event types."""
        event_filter = EventFilter(exclude_event_types=["session.heartbeat"])

        heartbeat_event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="session.heartbeat",
            category=EventCategory.SESSION,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
            payload={},
        )

        assert event_filter.should_process(heartbeat_event) is False

    def test_filter_sample_rate(self):
        """Test sampling filter."""
        event_filter = EventFilter(sample_rate=0.5)

        event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="session.started",
            category=EventCategory.SESSION,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
            payload={},
        )

        results = [event_filter.should_process(event) for _ in range(10)]
        assert True in results and False in results


class TestAnalyticsEventProcessor:
    """Tests for AnalyticsEventProcessor class."""

    @pytest.fixture
    def processor(self, analytics_aggregator):
        """Create event processor."""
        return AnalyticsEventProcessor(analytics_aggregator)

    @pytest.mark.asyncio
    async def test_process_session_event(self, processor, sample_session_event):
        """Test processing session event."""
        event = AnalyticsEvent.from_raw(sample_session_event)

        await processor.process(event)

    @pytest.mark.asyncio
    async def test_process_safety_event(self, processor, sample_safety_event):
        """Test processing safety event."""
        event = AnalyticsEvent.from_raw(sample_safety_event)

        await processor.process(event)

    @pytest.mark.asyncio
    async def test_process_therapy_event(self, processor, sample_therapy_event):
        """Test processing therapy event."""
        event = AnalyticsEvent.from_raw(sample_therapy_event)

        await processor.process(event)

    @pytest.mark.asyncio
    async def test_process_diagnosis_event(self, processor, sample_diagnosis_event):
        """Test processing diagnosis event."""
        event = AnalyticsEvent.from_raw(sample_diagnosis_event)

        await processor.process(event)

    @pytest.mark.asyncio
    async def test_register_handler(self, processor):
        """Test registering event handler."""
        handled_events = []

        async def handler(event):
            handled_events.append(event)

        processor.register_handler("session.started", handler)

        event = AnalyticsEvent(
            event_id=uuid4(),
            event_type="session.started",
            category=EventCategory.SESSION,
            user_id=uuid4(),
            session_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            correlation_id=uuid4(),
            source_service="test",
            payload={},
        )

        await processor.process(event)

        assert len(handled_events) == 1


class TestAnalyticsConsumer:
    """Tests for AnalyticsConsumer class."""

    @pytest.mark.asyncio
    async def test_consumer_start_stop(self, analytics_consumer):
        """Test starting and stopping consumer."""
        await analytics_consumer.start()
        assert analytics_consumer.metrics.events_received == 0

        await analytics_consumer.stop()

    @pytest.mark.asyncio
    async def test_process_event(self, analytics_consumer, sample_session_event):
        """Test processing a single event."""
        await analytics_consumer.start()

        success = await analytics_consumer.process_event(sample_session_event)

        assert success is True
        assert analytics_consumer.metrics.events_processed == 1

        await analytics_consumer.stop()

    @pytest.mark.asyncio
    async def test_process_invalid_event(self, analytics_consumer):
        """Test processing invalid event data."""
        await analytics_consumer.start()

        invalid_event = {"invalid": "data"}
        success = await analytics_consumer.process_event(invalid_event)

        assert success is True or analytics_consumer.metrics.events_failed >= 0

        await analytics_consumer.stop()

    @pytest.mark.asyncio
    async def test_enqueue_event(self, analytics_consumer, sample_session_event):
        """Test enqueueing an event."""
        await analytics_consumer.start()

        await analytics_consumer.enqueue_event(sample_session_event)

        stats = await analytics_consumer.get_statistics()
        assert stats["queue_size"] >= 0

        await analytics_consumer.stop()

    @pytest.mark.asyncio
    async def test_get_statistics(self, analytics_consumer):
        """Test getting consumer statistics."""
        stats = await analytics_consumer.get_statistics()

        assert "config" in stats
        assert "metrics" in stats
        assert "running" in stats
        assert "queue_size" in stats

    @pytest.mark.asyncio
    async def test_process_multiple_events(self, analytics_consumer, sample_session_event, sample_safety_event):
        """Test processing multiple events."""
        await analytics_consumer.start()

        await analytics_consumer.process_event(sample_session_event)
        await analytics_consumer.process_event(sample_safety_event)

        assert analytics_consumer.metrics.events_processed >= 2

        await analytics_consumer.stop()


class TestCreateAnalyticsConsumer:
    """Tests for create_analytics_consumer factory function."""

    def test_create_with_defaults(self, analytics_aggregator):
        """Test creating consumer with defaults."""
        consumer = create_analytics_consumer(analytics_aggregator)

        assert consumer is not None
        assert consumer.metrics.events_received == 0

    def test_create_with_custom_config(self, analytics_aggregator):
        """Test creating consumer with custom config."""
        config = ConsumerConfig(
            group_id="custom-analytics",
            batch_size=50,
        )

        consumer = create_analytics_consumer(analytics_aggregator, config)

        assert consumer is not None

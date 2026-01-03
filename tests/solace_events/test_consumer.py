"""Unit tests for Solace-AI Event Consumer."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from solace_events.config import ConsumerSettings, KafkaSettings, SolaceTopic
from solace_events.consumer import (
    ConsumerMetrics,
    EventConsumer,
    MockKafkaConsumerAdapter,
    OffsetTracker,
    ProcessingResult,
    ProcessingStatus,
    create_consumer,
)
from solace_events.schemas import SessionStartedEvent


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_success_result(self) -> None:
        """Test successful processing result."""
        event_id = uuid4()
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            event_id=event_id,
        )

        assert result.status == ProcessingStatus.SUCCESS
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed processing result."""
        event_id = uuid4()
        result = ProcessingResult(
            status=ProcessingStatus.FAILED,
            event_id=event_id,
            error="Test error",
        )

        assert result.status == ProcessingStatus.FAILED
        assert result.error == "Test error"


class TestConsumerMetrics:
    """Tests for ConsumerMetrics."""

    def test_initial_values(self) -> None:
        """Test initial metrics values."""
        metrics = ConsumerMetrics()

        assert metrics.messages_received == 0
        assert metrics.messages_processed == 0
        assert metrics.messages_failed == 0

    def test_record_success(self) -> None:
        """Test recording successful processing."""
        metrics = ConsumerMetrics()

        metrics.record_success(100)

        assert metrics.messages_received == 1
        assert metrics.messages_processed == 1
        assert metrics.processing_time_ms_total == 100

    def test_record_failure(self) -> None:
        """Test recording failed processing."""
        metrics = ConsumerMetrics()

        metrics.record_failure()

        assert metrics.messages_received == 1
        assert metrics.messages_failed == 1

    def test_avg_processing_time(self) -> None:
        """Test average processing time calculation."""
        metrics = ConsumerMetrics()

        metrics.record_success(100)
        metrics.record_success(200)

        assert metrics.avg_processing_time_ms == 150.0

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        metrics = ConsumerMetrics()

        metrics.record_success(100)
        metrics.record_success(100)
        metrics.record_failure()

        assert metrics.success_rate == pytest.approx(66.67, rel=0.01)

    def test_success_rate_no_messages(self) -> None:
        """Test success rate with no messages."""
        metrics = ConsumerMetrics()

        assert metrics.success_rate == 100.0


class TestOffsetTracker:
    """Tests for OffsetTracker."""

    @pytest.fixture
    def tracker(self) -> OffsetTracker:
        """Create fresh tracker for each test."""
        return OffsetTracker()

    @pytest.mark.asyncio
    async def test_track_received(self, tracker: OffsetTracker) -> None:
        """Test tracking received messages."""
        await tracker.track_received("topic", 0, 1)
        await tracker.track_received("topic", 0, 2)

        # No commits yet
        committed = await tracker.get_committed("topic", 0)
        assert committed is None

    @pytest.mark.asyncio
    async def test_mark_processed_in_order(self, tracker: OffsetTracker) -> None:
        """Test marking processed in order."""
        await tracker.track_received("topic", 0, 1)
        await tracker.track_received("topic", 0, 2)

        committable = await tracker.mark_processed("topic", 0, 1)
        # Can't commit yet - 2 still pending
        assert committable is None

        committable = await tracker.mark_processed("topic", 0, 2)
        # Now can commit offset 3
        assert committable == 3

    @pytest.mark.asyncio
    async def test_mark_processed_out_of_order(self, tracker: OffsetTracker) -> None:
        """Test marking processed out of order."""
        await tracker.track_received("topic", 0, 1)
        await tracker.track_received("topic", 0, 2)
        await tracker.track_received("topic", 0, 3)

        # Process 2 first
        committable = await tracker.mark_processed("topic", 0, 2)
        assert committable is None

        # Process 1
        committable = await tracker.mark_processed("topic", 0, 1)
        # Can commit up to lowest pending (3)
        assert committable == 3

    @pytest.mark.asyncio
    async def test_get_all_committed(self, tracker: OffsetTracker) -> None:
        """Test getting all committed offsets."""
        await tracker.track_received("topic1", 0, 1)
        await tracker.track_received("topic2", 0, 1)

        await tracker.mark_processed("topic1", 0, 1)
        await tracker.mark_processed("topic2", 0, 1)

        all_committed = tracker.get_all_committed()

        assert len(all_committed) == 2
        assert all_committed[("topic1", 0)] == 2
        assert all_committed[("topic2", 0)] == 2


class TestMockKafkaConsumerAdapter:
    """Tests for MockKafkaConsumerAdapter."""

    @pytest.fixture
    def consumer(self) -> MockKafkaConsumerAdapter:
        """Create mock consumer."""
        return MockKafkaConsumerAdapter()

    @pytest.mark.asyncio
    async def test_start_stop(self, consumer: MockKafkaConsumerAdapter) -> None:
        """Test consumer lifecycle."""
        await consumer.start()
        await consumer.stop()

    @pytest.mark.asyncio
    async def test_subscribe(self, consumer: MockKafkaConsumerAdapter) -> None:
        """Test subscribing to topics."""
        await consumer.start()
        await consumer.subscribe(["topic1", "topic2"])

        assert consumer._subscribed_topics == ["topic1", "topic2"]

    @pytest.mark.asyncio
    async def test_poll(self, consumer: MockKafkaConsumerAdapter) -> None:
        """Test polling for messages."""
        await consumer.start()
        consumer.add_message("topic", 0, 1, {"event_type": "test"})

        messages = await consumer.poll(timeout_ms=100)

        assert len(messages) == 1
        assert messages[0][0] == "topic"
        assert messages[0][2] == 1

    @pytest.mark.asyncio
    async def test_commit(self, consumer: MockKafkaConsumerAdapter) -> None:
        """Test committing offsets."""
        await consumer.start()
        await consumer.commit({("topic", 0): 5})

        committed = consumer.get_committed_offsets()

        assert committed[("topic", 0)] == 5

    def test_clear(self, consumer: MockKafkaConsumerAdapter) -> None:
        """Test clearing state."""
        consumer.add_message("topic", 0, 1, {})
        consumer.clear()

        assert len(consumer._messages) == 0


class TestEventConsumer:
    """Tests for EventConsumer."""

    @pytest.fixture
    def consumer(self) -> EventConsumer:
        """Create consumer with mock adapter."""
        adapter = MockKafkaConsumerAdapter()
        return EventConsumer(adapter)

    @pytest.mark.asyncio
    async def test_start_stop(self, consumer: EventConsumer) -> None:
        """Test consumer lifecycle."""
        await consumer.start([SolaceTopic.SESSIONS])
        await consumer.stop()

    @pytest.mark.asyncio
    async def test_register_handler(self, consumer: EventConsumer) -> None:
        """Test registering event handler."""
        handled_events = []

        async def handler(event):
            handled_events.append(event)

        consumer.register_handler("session.started", handler)

        assert "session.started" in consumer._handlers

    @pytest.mark.asyncio
    async def test_register_default_handler(self, consumer: EventConsumer) -> None:
        """Test registering default handler."""
        async def handler(event):
            pass

        consumer.register_default_handler(handler)

        assert len(consumer._default_handlers) == 1

    @pytest.mark.asyncio
    async def test_process_message(self, consumer: EventConsumer) -> None:
        """Test processing a message."""
        handled_events = []

        async def handler(event):
            handled_events.append(event)

        consumer.register_handler("session.started", handler)

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        result = await consumer._process_message(
            "solace.sessions", 0, 1, event.to_dict()
        )

        assert result.status == ProcessingStatus.SUCCESS
        assert len(handled_events) == 1

    @pytest.mark.asyncio
    async def test_process_message_failure(self, consumer: EventConsumer) -> None:
        """Test handling processing failure."""
        async def failing_handler(event):
            raise ValueError("Test error")

        consumer.register_handler("session.started", failing_handler)

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        result = await consumer._process_message(
            "solace.sessions", 0, 1, event.to_dict()
        )

        assert result.status == ProcessingStatus.FAILED
        assert "Test error" in result.error

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, consumer: EventConsumer) -> None:
        """Test metrics are tracked."""
        async def handler(event):
            pass

        consumer.register_handler("session.started", handler)

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        await consumer._process_message(
            "solace.sessions", 0, 1, event.to_dict()
        )

        assert consumer.metrics.messages_processed == 1

    @pytest.mark.asyncio
    async def test_dead_letter_handler(self, consumer: EventConsumer) -> None:
        """Test dead letter handler is called on failure."""
        dead_letters = []

        async def dlq_handler(event, error):
            dead_letters.append((event, error))

        async def failing_handler(event):
            raise ValueError("Test error")

        consumer.register_handler("session.started", failing_handler)
        consumer.set_dead_letter_handler(dlq_handler)

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        await consumer._process_message(
            "solace.sessions", 0, 1, event.to_dict()
        )

        assert len(dead_letters) == 1


class TestCreateConsumer:
    """Tests for create_consumer factory."""

    def test_create_mock_consumer(self) -> None:
        """Test creating mock consumer."""
        consumer = create_consumer("test-group", use_mock=True)

        assert consumer is not None

    def test_create_with_settings(self) -> None:
        """Test creating consumer with custom settings."""
        kafka_settings = KafkaSettings(bootstrap_servers="custom:9092")
        consumer_settings = ConsumerSettings(
            group_id="my-group",
            max_poll_records=50,
        )

        consumer = create_consumer(
            "my-group",
            kafka_settings=kafka_settings,
            consumer_settings=consumer_settings,
            use_mock=True,
        )

        assert consumer is not None

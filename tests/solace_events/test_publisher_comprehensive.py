"""Comprehensive unit tests for Solace-AI Event Publisher.

This module provides exhaustive coverage for:
- Outbox pattern functionality
- Retry logic and failure handling
- Concurrent publishing
- Edge cases and error scenarios
- Producer adapter behavior
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from solace_events.publisher import (
    OutboxStatus,
    OutboxRecord,
    InMemoryOutboxStore,
    KafkaProducerAdapter,
    MockKafkaProducerAdapter,
    EventPublisher,
    OutboxPoller,
    create_publisher,
)
from solace_events.config import KafkaSettings, ProducerSettings
from solace_events.schemas import SessionStartedEvent, CrisisDetectedEvent, CrisisLevel


class TestOutboxStatus:
    """Tests for OutboxStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Ensure all expected statuses exist."""
        expected = {"PENDING", "PUBLISHED", "FAILED"}
        actual = {s.value for s in OutboxStatus}
        assert actual == expected

    def test_status_values_uppercase(self) -> None:
        """Ensure all status values are uppercase."""
        for status in OutboxStatus:
            assert status.value == status.value.upper()


class TestOutboxRecord:
    """Tests for OutboxRecord model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={"key": "value"},
            aggregate_id=uuid4(),
            topic="test-topic",
            partition_key="key",
        )
        assert record.id is not None
        assert record.created_at is not None
        assert record.published_at is None
        assert record.status == OutboxStatus.PENDING
        assert record.retry_count == 0
        assert record.last_error is None

    def test_from_event(self) -> None:
        """Test creating outbox record from event."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        record = OutboxRecord.from_event(event)
        assert record.event_id == event.metadata.event_id
        assert record.event_type == "session.started"
        assert record.aggregate_id == event.user_id
        assert record.partition_key == str(event.user_id)
        assert record.topic == "solace.sessions"

    def test_from_event_with_custom_topic(self) -> None:
        """Test creating outbox record with custom topic."""
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        record = OutboxRecord.from_event(event, topic="custom-topic")
        assert record.topic == "custom-topic"

    def test_record_not_frozen(self) -> None:
        """Test record is mutable (frozen=False)."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
        )
        record.status = OutboxStatus.PUBLISHED
        assert record.status == OutboxStatus.PUBLISHED

    def test_retry_count_non_negative(self) -> None:
        """Test retry count cannot be negative."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            OutboxRecord(
                event_id=uuid4(),
                event_type="test",
                event_payload={},
                aggregate_id=uuid4(),
                topic="topic",
                partition_key="key",
                retry_count=-1,
            )


class TestInMemoryOutboxStore:
    """Tests for InMemoryOutboxStore."""

    @pytest.fixture
    def store(self) -> InMemoryOutboxStore:
        return InMemoryOutboxStore()

    @pytest.mark.asyncio
    async def test_save_and_retrieve_pending(self, store: InMemoryOutboxStore) -> None:
        """Test saving and retrieving pending records."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
        )
        await store.save(record)
        pending = await store.get_pending()
        assert len(pending) == 1
        assert pending[0].id == record.id

    @pytest.mark.asyncio
    async def test_get_pending_limit(self, store: InMemoryOutboxStore) -> None:
        """Test pending records respects limit."""
        for _ in range(10):
            record = OutboxRecord(
                event_id=uuid4(),
                event_type="test",
                event_payload={},
                aggregate_id=uuid4(),
                topic="topic",
                partition_key="key",
            )
            await store.save(record)
        pending = await store.get_pending(limit=5)
        assert len(pending) == 5

    @pytest.mark.asyncio
    async def test_get_pending_sorted_by_created_at(self, store: InMemoryOutboxStore) -> None:
        """Test pending records sorted by created_at."""
        older_record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        newer_record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
            created_at=datetime.now(timezone.utc),
        )
        await store.save(newer_record)
        await store.save(older_record)
        pending = await store.get_pending()
        assert pending[0].id == older_record.id

    @pytest.mark.asyncio
    async def test_mark_published(self, store: InMemoryOutboxStore) -> None:
        """Test marking record as published."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
        )
        await store.save(record)
        await store.mark_published(record.id)
        pending = await store.get_pending()
        assert len(pending) == 0
        assert store._records[record.id].status == OutboxStatus.PUBLISHED
        assert store._records[record.id].published_at is not None

    @pytest.mark.asyncio
    async def test_mark_failed(self, store: InMemoryOutboxStore) -> None:
        """Test marking record as failed."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
        )
        await store.save(record)
        await store.mark_failed(record.id, "Connection error")
        assert store._records[record.id].status == OutboxStatus.FAILED
        assert store._records[record.id].last_error == "Connection error"

    @pytest.mark.asyncio
    async def test_increment_retry(self, store: InMemoryOutboxStore) -> None:
        """Test incrementing retry count."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test",
            event_payload={},
            aggregate_id=uuid4(),
            topic="topic",
            partition_key="key",
        )
        await store.save(record)
        count1 = await store.increment_retry(record.id)
        count2 = await store.increment_retry(record.id)
        assert count1 == 1
        assert count2 == 2

    @pytest.mark.asyncio
    async def test_increment_retry_nonexistent_returns_zero(self, store: InMemoryOutboxStore) -> None:
        """Test increment retry for nonexistent record returns 0."""
        count = await store.increment_retry(uuid4())
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear(self, store: InMemoryOutboxStore) -> None:
        """Test clearing all records."""
        for _ in range(5):
            record = OutboxRecord(
                event_id=uuid4(),
                event_type="test",
                event_payload={},
                aggregate_id=uuid4(),
                topic="topic",
                partition_key="key",
            )
            await store.save(record)
        store.clear()
        pending = await store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_published_nonexistent_silent(self, store: InMemoryOutboxStore) -> None:
        """Test marking nonexistent record as published is silent."""
        await store.mark_published(uuid4())  # Should not raise

    @pytest.mark.asyncio
    async def test_mark_failed_nonexistent_silent(self, store: InMemoryOutboxStore) -> None:
        """Test marking nonexistent record as failed is silent."""
        await store.mark_failed(uuid4(), "error")  # Should not raise

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, store: InMemoryOutboxStore) -> None:
        """Test concurrent saves work correctly."""
        async def save_record():
            record = OutboxRecord(
                event_id=uuid4(),
                event_type="test",
                event_payload={},
                aggregate_id=uuid4(),
                topic="topic",
                partition_key="key",
            )
            await store.save(record)
            return record.id

        ids = await asyncio.gather(*[save_record() for _ in range(10)])
        assert len(set(ids)) == 10
        pending = await store.get_pending()
        assert len(pending) == 10


class TestMockKafkaProducerAdapter:
    """Tests for MockKafkaProducerAdapter."""

    @pytest.fixture
    def producer(self) -> MockKafkaProducerAdapter:
        return MockKafkaProducerAdapter()

    @pytest.mark.asyncio
    async def test_start_stop(self, producer: MockKafkaProducerAdapter) -> None:
        """Test producer lifecycle."""
        assert producer._started is False
        await producer.start()
        assert producer._started is True
        await producer.stop()
        assert producer._started is False

    @pytest.mark.asyncio
    async def test_send_stores_message(self, producer: MockKafkaProducerAdapter) -> None:
        """Test send stores message in memory."""
        await producer.start()
        await producer.send("topic", "key", {"data": "value"})
        messages = producer.get_messages()
        assert len(messages) == 1
        assert messages[0] == ("topic", "key", {"data": "value"})

    @pytest.mark.asyncio
    async def test_send_multiple_messages(self, producer: MockKafkaProducerAdapter) -> None:
        """Test sending multiple messages."""
        await producer.start()
        for i in range(5):
            await producer.send(f"topic-{i}", f"key-{i}", {"index": i})
        messages = producer.get_messages()
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_send_without_start_raises(self, producer: MockKafkaProducerAdapter) -> None:
        """Test send without start raises error."""
        with pytest.raises(RuntimeError, match="not started"):
            await producer.send("topic", "key", {})

    @pytest.mark.asyncio
    async def test_clear_removes_messages(self, producer: MockKafkaProducerAdapter) -> None:
        """Test clear removes all messages."""
        await producer.start()
        await producer.send("topic", "key", {})
        producer.clear()
        assert len(producer.get_messages()) == 0

    @pytest.mark.asyncio
    async def test_get_messages_returns_copy(self, producer: MockKafkaProducerAdapter) -> None:
        """Test get_messages returns a copy."""
        await producer.start()
        await producer.send("topic", "key", {})
        messages1 = producer.get_messages()
        messages1.append(("fake", "fake", {}))
        messages2 = producer.get_messages()
        assert len(messages2) == 1


class TestEventPublisher:
    """Tests for EventPublisher."""

    @pytest.fixture
    def mock_producer(self) -> MockKafkaProducerAdapter:
        return MockKafkaProducerAdapter()

    @pytest.fixture
    def outbox_store(self) -> InMemoryOutboxStore:
        return InMemoryOutboxStore()

    @pytest.fixture
    def publisher(self, mock_producer: MockKafkaProducerAdapter, outbox_store: InMemoryOutboxStore) -> EventPublisher:
        return EventPublisher(mock_producer, outbox_store, use_outbox=True)

    @pytest.mark.asyncio
    async def test_start_stop(self, publisher: EventPublisher) -> None:
        """Test publisher lifecycle."""
        assert publisher._started is False
        await publisher.start()
        assert publisher._started is True
        await publisher.stop()
        assert publisher._started is False

    @pytest.mark.asyncio
    async def test_publish_without_start_raises(self, publisher: EventPublisher) -> None:
        """Test publish without start raises error."""
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        with pytest.raises(RuntimeError, match="not started"):
            await publisher.publish(event)

    @pytest.mark.asyncio
    async def test_publish_via_outbox(self, publisher: EventPublisher, outbox_store: InMemoryOutboxStore) -> None:
        """Test publishing via outbox pattern."""
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        event_id = await publisher.publish(event)
        assert event_id == event.metadata.event_id
        pending = await outbox_store.get_pending()
        assert len(pending) == 1
        assert pending[0].event_type == "session.started"

    @pytest.mark.asyncio
    async def test_publish_direct(self, mock_producer: MockKafkaProducerAdapter) -> None:
        """Test direct publishing without outbox."""
        publisher = EventPublisher(mock_producer, use_outbox=False)
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        event_id = await publisher.publish(event)
        assert event_id == event.metadata.event_id
        messages = mock_producer.get_messages()
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_publish_with_custom_topic(self, publisher: EventPublisher, outbox_store: InMemoryOutboxStore) -> None:
        """Test publishing with custom topic."""
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        await publisher.publish(event, topic="custom-topic")
        pending = await outbox_store.get_pending()
        assert pending[0].topic == "custom-topic"

    @pytest.mark.asyncio
    async def test_publish_batch(self, publisher: EventPublisher, outbox_store: InMemoryOutboxStore) -> None:
        """Test batch publishing."""
        await publisher.start()
        events = [
            SessionStartedEvent(user_id=uuid4(), session_number=i+1)
            for i in range(5)
        ]
        ids = await publisher.publish_batch(events)
        assert len(ids) == 5
        pending = await outbox_store.get_pending()
        assert len(pending) == 5

    @pytest.mark.asyncio
    async def test_flush_outbox(self, publisher: EventPublisher, mock_producer: MockKafkaProducerAdapter, outbox_store: InMemoryOutboxStore) -> None:
        """Test flushing outbox to Kafka."""
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        await publisher.publish(event)
        published = await publisher.flush_outbox()
        assert published == 1
        messages = mock_producer.get_messages()
        assert len(messages) == 1
        pending = await outbox_store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_flush_outbox_when_disabled(self, mock_producer: MockKafkaProducerAdapter) -> None:
        """Test flush_outbox returns 0 when outbox disabled."""
        publisher = EventPublisher(mock_producer, use_outbox=False)
        await publisher.start()
        count = await publisher.flush_outbox()
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_outbox_with_batch_size(self, publisher: EventPublisher, outbox_store: InMemoryOutboxStore) -> None:
        """Test flush_outbox respects batch size."""
        await publisher.start()
        for i in range(10):
            event = SessionStartedEvent(user_id=uuid4(), session_number=i+1)
            await publisher.publish(event)
        published = await publisher.flush_outbox(batch_size=3)
        assert published == 3
        pending = await outbox_store.get_pending()
        assert len(pending) == 7

    @pytest.mark.asyncio
    async def test_flush_outbox_retries_on_failure(self, outbox_store: InMemoryOutboxStore) -> None:
        """Test flush_outbox retries on failure."""
        failing_producer = MockKafkaProducerAdapter()
        publisher = EventPublisher(failing_producer, outbox_store, max_retries=3)
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        await publisher.publish(event)
        # Make producer fail
        async def fail_send(*args):
            raise Exception("Connection failed")
        failing_producer.send = fail_send
        await publisher.flush_outbox()
        pending = await outbox_store.get_pending()
        assert len(pending) == 1
        assert pending[0].retry_count == 1

    @pytest.mark.asyncio
    async def test_flush_outbox_marks_failed_after_max_retries(self, outbox_store: InMemoryOutboxStore) -> None:
        """Test record marked failed after max retries."""
        failing_producer = MockKafkaProducerAdapter()
        publisher = EventPublisher(failing_producer, outbox_store, max_retries=2)
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        await publisher.publish(event)
        # Pre-increment retry count
        record_id = (await outbox_store.get_pending())[0].id
        await outbox_store.increment_retry(record_id)
        await outbox_store.increment_retry(record_id)
        # Make producer fail
        async def fail_send(*args):
            raise Exception("Connection failed")
        failing_producer.send = fail_send
        await publisher.flush_outbox()
        # Record should be marked failed
        pending = await outbox_store.get_pending()
        assert len(pending) == 0
        assert outbox_store._records[record_id].status == OutboxStatus.FAILED


class TestOutboxPoller:
    """Tests for OutboxPoller."""

    @pytest.fixture
    def mock_producer(self) -> MockKafkaProducerAdapter:
        return MockKafkaProducerAdapter()

    @pytest.fixture
    def publisher(self, mock_producer: MockKafkaProducerAdapter) -> EventPublisher:
        return EventPublisher(mock_producer, use_outbox=True)

    @pytest.mark.asyncio
    async def test_start_stop(self, publisher: EventPublisher) -> None:
        """Test poller lifecycle."""
        await publisher.start()
        poller = OutboxPoller(publisher, poll_interval_ms=100)
        await poller.start()
        assert poller._running is True
        assert poller._task is not None
        await poller.stop()
        assert poller._running is False
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_poller_flushes_outbox(self, publisher: EventPublisher, mock_producer: MockKafkaProducerAdapter) -> None:
        """Test poller flushes outbox automatically."""
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        await publisher.publish(event)
        poller = OutboxPoller(publisher, poll_interval_ms=50, batch_size=10)
        await poller.start()
        for _ in range(30):  # Poll up to 30 times (50ms interval → 1.5s max)
            await asyncio.sleep(0.05)
            if mock_producer.get_messages():
                break
        await poller.stop()
        await publisher.stop()
        messages = mock_producer.get_messages()
        assert len(messages) >= 1

    @pytest.mark.asyncio
    async def test_poller_handles_errors(self, mock_producer: MockKafkaProducerAdapter) -> None:
        """Test poller handles errors gracefully."""
        publisher = EventPublisher(mock_producer, use_outbox=True)
        await publisher.start()
        # Make flush_outbox fail
        publisher.flush_outbox = AsyncMock(side_effect=Exception("Test error"))
        poller = OutboxPoller(publisher, poll_interval_ms=50)
        await poller.start()
        await asyncio.sleep(0.1)  # Let it run with errors
        await poller.stop()  # Should not raise
        await publisher.stop()


class TestCreatePublisher:
    """Tests for create_publisher factory function."""

    def test_create_mock_publisher(self) -> None:
        """Test creating mock publisher."""
        publisher = create_publisher(use_mock=True)
        assert publisher is not None
        assert isinstance(publisher._producer, MockKafkaProducerAdapter)

    def test_create_publisher_with_settings(self) -> None:
        """Test creating publisher with custom settings."""
        kafka_settings = KafkaSettings(bootstrap_servers="custom:9092")
        producer_settings = ProducerSettings(acks="1")
        publisher = create_publisher(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_mock=True,
        )
        assert publisher is not None

    def test_create_publisher_with_outbox_store(self) -> None:
        """Test creating publisher with custom outbox store."""
        custom_store = InMemoryOutboxStore()
        publisher = create_publisher(outbox_store=custom_store, use_mock=True)
        assert publisher._outbox_store is custom_store

    def test_create_publisher_without_outbox(self) -> None:
        """Test creating publisher without outbox."""
        publisher = create_publisher(use_outbox=False, use_mock=True)
        assert publisher._use_outbox is False


class TestPublisherEventTypes:
    """Tests for different event types."""

    @pytest.fixture
    def publisher(self) -> EventPublisher:
        producer = MockKafkaProducerAdapter()
        return EventPublisher(producer, use_outbox=True)

    @pytest.mark.asyncio
    async def test_publish_session_event(self, publisher: EventPublisher) -> None:
        """Test publishing session event."""
        await publisher.start()
        event = SessionStartedEvent(user_id=uuid4(), session_number=1)
        event_id = await publisher.publish(event)
        assert event_id is not None
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_crisis_event(self, publisher: EventPublisher) -> None:
        """Test publishing crisis event."""
        await publisher.start()
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            crisis_level=CrisisLevel.CRITICAL,
            trigger_indicators=["indicator1"],
            detection_layer=1,
            confidence=Decimal("0.95"),
            escalation_action="immediate",
        )
        event_id = await publisher.publish(event)
        assert event_id is not None
        await publisher.stop()


class TestPublisherConcurrency:
    """Concurrency tests for EventPublisher."""

    @pytest.mark.asyncio
    async def test_concurrent_publishes(self) -> None:
        """Test concurrent publishes work correctly."""
        producer = MockKafkaProducerAdapter()
        publisher = EventPublisher(producer, use_outbox=False)
        await publisher.start()

        async def publish_event(i: int):
            event = SessionStartedEvent(user_id=uuid4(), session_number=i+1)
            return await publisher.publish(event)

        ids = await asyncio.gather(*[publish_event(i) for i in range(20)])
        assert len(set(ids)) == 20
        messages = producer.get_messages()
        assert len(messages) == 20
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_concurrent_outbox_publishes(self) -> None:
        """Test concurrent outbox publishes work correctly."""
        producer = MockKafkaProducerAdapter()
        store = InMemoryOutboxStore()
        publisher = EventPublisher(producer, store, use_outbox=True)
        await publisher.start()

        async def publish_event(i: int):
            event = SessionStartedEvent(user_id=uuid4(), session_number=i+1)
            return await publisher.publish(event)

        ids = await asyncio.gather(*[publish_event(i) for i in range(20)])
        assert len(set(ids)) == 20
        pending = await store.get_pending()
        assert len(pending) == 20
        await publisher.stop()

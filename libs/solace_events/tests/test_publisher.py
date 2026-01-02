"""Unit tests for Solace-AI Event Publisher."""

import pytest
from uuid import uuid4

from solace_events.src.config import KafkaSettings, ProducerSettings
from solace_events.src.publisher import (
    EventPublisher,
    InMemoryOutboxStore,
    MockKafkaProducerAdapter,
    OutboxPoller,
    OutboxRecord,
    OutboxStatus,
    create_publisher,
)
from solace_events.src.schemas import SessionStartedEvent


class TestOutboxRecord:
    """Tests for OutboxRecord."""

    def test_default_values(self) -> None:
        """Test default outbox record values."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={"key": "value"},
            aggregate_id=uuid4(),
            topic="test.topic",
            partition_key="key",
        )

        assert record.status == OutboxStatus.PENDING
        assert record.retry_count == 0
        assert record.published_at is None

    def test_from_event(self) -> None:
        """Test creating record from event."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        record = OutboxRecord.from_event(event)

        assert record.event_id == event.metadata.event_id
        assert record.event_type == "session.started"
        assert record.topic == "solace.sessions"
        assert record.partition_key == str(event.user_id)

    def test_from_event_custom_topic(self) -> None:
        """Test creating record with custom topic."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        record = OutboxRecord.from_event(event, topic="custom.topic")

        assert record.topic == "custom.topic"


class TestInMemoryOutboxStore:
    """Tests for InMemoryOutboxStore."""

    @pytest.fixture
    def store(self) -> InMemoryOutboxStore:
        """Create fresh store for each test."""
        return InMemoryOutboxStore()

    @pytest.mark.asyncio
    async def test_save_and_get_pending(self, store: InMemoryOutboxStore) -> None:
        """Test saving and retrieving pending records."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={},
            aggregate_id=uuid4(),
            topic="test.topic",
            partition_key="key",
        )

        await store.save(record)
        pending = await store.get_pending()

        assert len(pending) == 1
        assert pending[0].id == record.id

    @pytest.mark.asyncio
    async def test_mark_published(self, store: InMemoryOutboxStore) -> None:
        """Test marking record as published."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={},
            aggregate_id=uuid4(),
            topic="test.topic",
            partition_key="key",
        )

        await store.save(record)
        await store.mark_published(record.id)
        pending = await store.get_pending()

        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_failed(self, store: InMemoryOutboxStore) -> None:
        """Test marking record as failed."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={},
            aggregate_id=uuid4(),
            topic="test.topic",
            partition_key="key",
        )

        await store.save(record)
        await store.mark_failed(record.id, "Test error")
        pending = await store.get_pending()

        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_increment_retry(self, store: InMemoryOutboxStore) -> None:
        """Test incrementing retry count."""
        record = OutboxRecord(
            event_id=uuid4(),
            event_type="test.event",
            event_payload={},
            aggregate_id=uuid4(),
            topic="test.topic",
            partition_key="key",
        )

        await store.save(record)
        new_count = await store.increment_retry(record.id)

        assert new_count == 1


class TestMockKafkaProducerAdapter:
    """Tests for MockKafkaProducerAdapter."""

    @pytest.fixture
    def producer(self) -> MockKafkaProducerAdapter:
        """Create mock producer."""
        return MockKafkaProducerAdapter()

    @pytest.mark.asyncio
    async def test_start_stop(self, producer: MockKafkaProducerAdapter) -> None:
        """Test producer lifecycle."""
        await producer.start()
        await producer.stop()

    @pytest.mark.asyncio
    async def test_send(self, producer: MockKafkaProducerAdapter) -> None:
        """Test sending messages."""
        await producer.start()
        await producer.send("test.topic", "key", {"data": "value"})

        messages = producer.get_messages()

        assert len(messages) == 1
        assert messages[0][0] == "test.topic"
        assert messages[0][1] == "key"
        assert messages[0][2]["data"] == "value"

    @pytest.mark.asyncio
    async def test_send_not_started(self, producer: MockKafkaProducerAdapter) -> None:
        """Test error when sending without starting."""
        with pytest.raises(RuntimeError):
            await producer.send("test.topic", "key", {})

    @pytest.mark.asyncio
    async def test_clear(self, producer: MockKafkaProducerAdapter) -> None:
        """Test clearing messages."""
        await producer.start()
        await producer.send("test.topic", "key", {})

        producer.clear()

        assert len(producer.get_messages()) == 0


class TestEventPublisher:
    """Tests for EventPublisher."""

    @pytest.fixture
    def publisher(self) -> EventPublisher:
        """Create publisher with mock producer."""
        producer = MockKafkaProducerAdapter()
        return EventPublisher(producer, use_outbox=False)

    @pytest.fixture
    def publisher_with_outbox(self) -> EventPublisher:
        """Create publisher with outbox."""
        producer = MockKafkaProducerAdapter()
        outbox = InMemoryOutboxStore()
        return EventPublisher(producer, outbox, use_outbox=True)

    @pytest.mark.asyncio
    async def test_publish_direct(self, publisher: EventPublisher) -> None:
        """Test direct publishing without outbox."""
        await publisher.start()

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        event_id = await publisher.publish(event)

        assert event_id == event.metadata.event_id
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_publish_via_outbox(self, publisher_with_outbox: EventPublisher) -> None:
        """Test publishing via outbox."""
        await publisher_with_outbox.start()

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        event_id = await publisher_with_outbox.publish(event)

        assert event_id == event.metadata.event_id
        await publisher_with_outbox.stop()

    @pytest.mark.asyncio
    async def test_publish_not_started(self, publisher: EventPublisher) -> None:
        """Test error when publishing without starting."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        with pytest.raises(RuntimeError):
            await publisher.publish(event)

    @pytest.mark.asyncio
    async def test_publish_batch(self, publisher: EventPublisher) -> None:
        """Test publishing batch of events."""
        await publisher.start()

        events = [
            SessionStartedEvent(user_id=uuid4(), session_number=i)
            for i in range(1, 4)
        ]

        event_ids = await publisher.publish_batch(events)

        assert len(event_ids) == 3
        await publisher.stop()

    @pytest.mark.asyncio
    async def test_flush_outbox(self, publisher_with_outbox: EventPublisher) -> None:
        """Test flushing outbox records."""
        await publisher_with_outbox.start()

        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        await publisher_with_outbox.publish(event)

        published = await publisher_with_outbox.flush_outbox()

        assert published == 1
        await publisher_with_outbox.stop()


class TestCreatePublisher:
    """Tests for create_publisher factory."""

    def test_create_mock_publisher(self) -> None:
        """Test creating mock publisher."""
        publisher = create_publisher(use_mock=True)

        assert publisher is not None

    def test_create_with_settings(self) -> None:
        """Test creating publisher with custom settings."""
        kafka_settings = KafkaSettings(bootstrap_servers="custom:9092")
        producer_settings = ProducerSettings(acks="1")

        publisher = create_publisher(
            kafka_settings=kafka_settings,
            producer_settings=producer_settings,
            use_mock=True,
        )

        assert publisher is not None

    def test_create_without_outbox(self) -> None:
        """Test creating publisher without outbox."""
        publisher = create_publisher(use_mock=True, use_outbox=False)

        assert publisher is not None


class TestOutboxPoller:
    """Tests for OutboxPoller."""

    @pytest.mark.asyncio
    async def test_poller_lifecycle(self) -> None:
        """Test poller start and stop."""
        producer = MockKafkaProducerAdapter()
        publisher = EventPublisher(producer, use_outbox=True)
        poller = OutboxPoller(publisher, poll_interval_ms=100)

        await publisher.start()
        await poller.start()

        import asyncio
        await asyncio.sleep(0.2)

        await poller.stop()
        await publisher.stop()

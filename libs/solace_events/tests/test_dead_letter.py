"""Unit tests for Solace-AI Dead Letter Queue."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from solace_events.src.config import SolaceTopic
from solace_events.src.dead_letter import (
    DeadLetterHandler,
    DeadLetterRecord,
    DeadLetterStore,
    RetryPolicy,
    RetryStrategy,
    create_dead_letter_handler,
    get_dlq_topic,
)
from solace_events.src.schemas import SessionStartedEvent


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_values(self) -> None:
        """Test default retry policy values."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.initial_delay_ms == 1000
        assert policy.strategy == RetryStrategy.EXPONENTIAL

    def test_fixed_strategy_delay(self) -> None:
        """Test fixed strategy delay calculation."""
        policy = RetryPolicy(
            strategy=RetryStrategy.FIXED,
            initial_delay_ms=1000,
            jitter_percent=0,
        )

        assert policy.get_delay_ms(1) == 1000
        assert policy.get_delay_ms(2) == 1000
        assert policy.get_delay_ms(3) == 1000

    def test_linear_strategy_delay(self) -> None:
        """Test linear strategy delay calculation."""
        policy = RetryPolicy(
            strategy=RetryStrategy.LINEAR,
            initial_delay_ms=1000,
            jitter_percent=0,
        )

        assert policy.get_delay_ms(1) == 1000
        assert policy.get_delay_ms(2) == 2000
        assert policy.get_delay_ms(3) == 3000

    def test_exponential_strategy_delay(self) -> None:
        """Test exponential strategy delay calculation."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay_ms=1000,
            multiplier=2.0,
            jitter_percent=0,
        )

        assert policy.get_delay_ms(1) == 1000
        assert policy.get_delay_ms(2) == 2000
        assert policy.get_delay_ms(3) == 4000

    def test_max_delay_cap(self) -> None:
        """Test delay is capped at max."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay_ms=1000,
            max_delay_ms=5000,
            multiplier=2.0,
            jitter_percent=0,
        )

        assert policy.get_delay_ms(10) == 5000

    def test_jitter_applied(self) -> None:
        """Test jitter is applied."""
        policy = RetryPolicy(
            strategy=RetryStrategy.FIXED,
            initial_delay_ms=1000,
            jitter_percent=0.1,
        )

        delays = [policy.get_delay_ms(1) for _ in range(10)]
        unique_delays = set(delays)

        # With jitter, we should have some variation
        assert len(unique_delays) > 1


class TestDeadLetterRecord:
    """Tests for DeadLetterRecord."""

    def test_default_values(self) -> None:
        """Test default DLQ record values."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={"key": "value"},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Test error",
            error_type="ValueError",
        )

        assert record.retry_count == 0
        assert record.is_retriable is True
        assert record.resolved is False

    def test_from_failed_event(self) -> None:
        """Test creating record from failed event."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        error = ValueError("Test error")

        record = DeadLetterRecord.from_failed_event(
            event, "solace.sessions", "test-group", error
        )

        assert record.event_id == event.metadata.event_id
        assert record.event_type == "session.started"
        assert record.dlq_topic == "solace.sessions.dlq"
        assert record.error_message == "Test error"
        assert record.error_type == "ValueError"

    def test_increment_retry_allows_more(self) -> None:
        """Test increment retry when more retries allowed."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
            retry_count=0,
        )

        policy = RetryPolicy(max_retries=3)
        can_retry = record.increment_retry(ValueError("New error"), policy)

        assert can_retry is True
        assert record.retry_count == 1
        assert record.next_retry_at is not None

    def test_increment_retry_exhausted(self) -> None:
        """Test increment retry when exhausted."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
            retry_count=2,
        )

        policy = RetryPolicy(max_retries=3)
        can_retry = record.increment_retry(ValueError("New error"), policy)

        assert can_retry is False
        assert record.is_retriable is False
        assert record.next_retry_at is None


class TestDeadLetterStore:
    """Tests for DeadLetterStore."""

    @pytest.fixture
    def store(self) -> DeadLetterStore:
        """Create fresh store for each test."""
        return DeadLetterStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store: DeadLetterStore) -> None:
        """Test saving and retrieving record."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
        )

        await store.save(record)
        retrieved = await store.get(record.id)

        assert retrieved is not None
        assert retrieved.id == record.id

    @pytest.mark.asyncio
    async def test_get_retriable(self, store: DeadLetterStore) -> None:
        """Test getting retriable records."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
            is_retriable=True,
        )

        await store.save(record)
        retriable = await store.get_retriable()

        assert len(retriable) == 1

    @pytest.mark.asyncio
    async def test_get_retriable_respects_next_retry_at(self, store: DeadLetterStore) -> None:
        """Test retriable records respect next_retry_at."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
            is_retriable=True,
            next_retry_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        await store.save(record)
        retriable = await store.get_retriable()

        assert len(retriable) == 0

    @pytest.mark.asyncio
    async def test_get_by_topic(self, store: DeadLetterStore) -> None:
        """Test getting records by DLQ topic."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
        )

        await store.save(record)
        records = await store.get_by_topic("test.topic.dlq")

        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_mark_resolved(self, store: DeadLetterStore) -> None:
        """Test marking record as resolved."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
        )

        await store.save(record)
        await store.mark_resolved(record.id, "Fixed manually")

        retrieved = await store.get(record.id)
        assert retrieved.resolved is True
        assert retrieved.resolution_notes == "Fixed manually"

    @pytest.mark.asyncio
    async def test_delete(self, store: DeadLetterStore) -> None:
        """Test deleting record."""
        record = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
        )

        await store.save(record)
        await store.delete(record.id)

        retrieved = await store.get(record.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_count_by_topic(self, store: DeadLetterStore) -> None:
        """Test counting records by topic."""
        for i in range(3):
            record = DeadLetterRecord(
                original_topic="topic1",
                dlq_topic="topic1.dlq",
                original_event={},
                event_type="test.event",
                event_id=uuid4(),
                user_id=uuid4(),
                consumer_group="test-group",
                error_message="Error",
                error_type="Error",
            )
            await store.save(record)

        for i in range(2):
            record = DeadLetterRecord(
                original_topic="topic2",
                dlq_topic="topic2.dlq",
                original_event={},
                event_type="test.event",
                event_id=uuid4(),
                user_id=uuid4(),
                consumer_group="test-group",
                error_message="Error",
                error_type="Error",
            )
            await store.save(record)

        counts = await store.count_by_topic()

        assert counts["topic1.dlq"] == 3
        assert counts["topic2.dlq"] == 2

    @pytest.mark.asyncio
    async def test_count_unresolved(self, store: DeadLetterStore) -> None:
        """Test counting unresolved records."""
        record1 = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
        )
        record2 = DeadLetterRecord(
            original_topic="test.topic",
            dlq_topic="test.topic.dlq",
            original_event={},
            event_type="test.event",
            event_id=uuid4(),
            user_id=uuid4(),
            consumer_group="test-group",
            error_message="Error",
            error_type="Error",
            resolved=True,
        )

        await store.save(record1)
        await store.save(record2)

        count = await store.count_unresolved()

        assert count == 1


class TestDeadLetterHandler:
    """Tests for DeadLetterHandler."""

    @pytest.fixture
    def handler(self) -> DeadLetterHandler:
        """Create handler for each test."""
        return DeadLetterHandler(consumer_group="test-group")

    @pytest.mark.asyncio
    async def test_handle_failure(self, handler: DeadLetterHandler) -> None:
        """Test handling event failure."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        error = ValueError("Test error")

        record = await handler.handle_failure(
            event, "solace.sessions", error
        )

        assert record.event_id == event.metadata.event_id
        assert record.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_record_no_handler(self, handler: DeadLetterHandler) -> None:
        """Test retry fails without registered handler."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        error = ValueError("Test error")

        record = await handler.handle_failure(
            event, "solace.sessions", error
        )
        success = await handler.retry_record(record)

        assert success is False

    @pytest.mark.asyncio
    async def test_retry_record_success(self, handler: DeadLetterHandler) -> None:
        """Test successful retry."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        error = ValueError("Test error")

        async def retry_handler(e):
            return True

        handler.register_retry_handler("session.started", retry_handler)
        record = await handler.handle_failure(
            event, "solace.sessions", error
        )
        success = await handler.retry_record(record)

        assert success is True

    @pytest.mark.asyncio
    async def test_retry_record_failure(self, handler: DeadLetterHandler) -> None:
        """Test retry that fails."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        error = ValueError("Test error")

        async def retry_handler(e):
            raise RuntimeError("Still failing")

        handler.register_retry_handler("session.started", retry_handler)
        record = await handler.handle_failure(
            event, "solace.sessions", error
        )
        success = await handler.retry_record(record)

        assert success is True  # Can retry more
        assert record.retry_count == 2

    @pytest.mark.asyncio
    async def test_process_retriable(self, handler: DeadLetterHandler) -> None:
        """Test processing retriable records."""
        from datetime import timedelta
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )

        async def retry_handler(e):
            return True

        handler.register_retry_handler("session.started", retry_handler)
        record = await handler.handle_failure(event, "solace.sessions", ValueError("Error"))

        # Set next_retry_at to past to make immediately retriable
        record.next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await handler.store.save(record)

        success_count = await handler.process_retriable()

        assert success_count == 1

    @pytest.mark.asyncio
    async def test_get_dlq_stats(self, handler: DeadLetterHandler) -> None:
        """Test getting DLQ statistics."""
        event = SessionStartedEvent(
            user_id=uuid4(),
            session_number=1,
        )
        await handler.handle_failure(event, "solace.sessions", ValueError("Error"))

        stats = await handler.get_dlq_stats()

        assert stats["total_unresolved"] == 1
        assert "by_topic" in stats
        assert "retry_policy" in stats


class TestDLQHelpers:
    """Tests for DLQ helper functions."""

    def test_get_dlq_topic_from_enum(self) -> None:
        """Test getting DLQ topic from enum."""
        assert get_dlq_topic(SolaceTopic.SAFETY) == "solace.safety.dlq"

    def test_get_dlq_topic_from_string(self) -> None:
        """Test getting DLQ topic from string."""
        assert get_dlq_topic("custom.topic") == "custom.topic.dlq"


class TestCreateDeadLetterHandler:
    """Tests for create_dead_letter_handler factory."""

    def test_create_handler(self) -> None:
        """Test creating handler."""
        handler = create_dead_letter_handler("test-group")

        assert handler is not None

    def test_create_with_custom_policy(self) -> None:
        """Test creating handler with custom policy."""
        policy = RetryPolicy(max_retries=5)
        handler = create_dead_letter_handler("test-group", retry_policy=policy)

        assert handler._retry_policy.max_retries == 5

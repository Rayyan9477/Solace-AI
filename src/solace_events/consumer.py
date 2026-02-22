"""Solace-AI Event Consumer - Consumer group management with offset tracking."""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
import structlog

from .config import ConsumerSettings, KafkaSettings, SolaceTopic
from .schemas import BaseEvent, deserialize_event

logger = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseEvent)
EventHandler = Callable[[BaseEvent], Awaitable[None]]


class ProcessingStatus(str, Enum):
    """Status of event processing."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRY = "RETRY"
    SKIP = "SKIP"


@dataclass
class ProcessingResult:
    """Result of processing an event."""

    status: ProcessingStatus
    event_id: UUID
    error: str | None = None
    retry_after_ms: int | None = None


@dataclass
class ConsumerMetrics:
    """Metrics for consumer monitoring."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    last_message_at: datetime | None = None
    processing_time_ms_total: int = 0
    current_lag: int = 0

    def record_success(self, processing_time_ms: int) -> None:
        """Record successful processing."""
        self.messages_received += 1
        self.messages_processed += 1
        self.processing_time_ms_total += processing_time_ms
        self.last_message_at = datetime.now(timezone.utc)

    def record_failure(self) -> None:
        """Record failed processing."""
        self.messages_received += 1
        self.messages_failed += 1
        self.last_message_at = datetime.now(timezone.utc)

    def record_retry(self) -> None:
        """Record retry attempt."""
        self.messages_retried += 1

    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time in milliseconds."""
        return (
            0.0
            if self.messages_processed == 0
            else self.processing_time_ms_total / self.messages_processed
        )

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (
            100.0
            if self.messages_received == 0
            else (self.messages_processed / self.messages_received) * 100
        )


class OffsetTracker:
    """Tracks committed offsets per partition."""

    def __init__(self) -> None:
        self._offsets: dict[tuple[str, int], int] = {}
        self._pending: dict[tuple[str, int], list[int]] = {}
        self._lock = asyncio.Lock()

    async def track_received(self, topic: str, partition: int, offset: int) -> None:
        """Track received message offset."""
        async with self._lock:
            key = (topic, partition)
            if key not in self._pending:
                self._pending[key] = []
            self._pending[key].append(offset)

    async def mark_processed(
        self, topic: str, partition: int, offset: int
    ) -> int | None:
        """Mark offset as processed, return committable offset if available."""
        async with self._lock:
            key = (topic, partition)
            if key not in self._pending:
                return None
            pending = self._pending[key]
            if offset in pending:
                pending.remove(offset)
            if not pending:
                committable = offset + 1
                self._offsets[key] = committable
                return committable
            min_pending = min(pending)
            if offset + 1 < min_pending:
                committable = min_pending
                self._offsets[key] = committable
                return committable
            return None

    async def get_committed(self, topic: str, partition: int) -> int | None:
        """Get last committed offset for partition."""
        async with self._lock:
            return self._offsets.get((topic, partition))

    def get_all_committed(self) -> dict[tuple[str, int], int]:
        """Get all committed offsets."""
        return self._offsets.copy()


class KafkaConsumerAdapter(ABC):
    """Abstract Kafka consumer adapter."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def subscribe(self, topics: list[str]) -> None: ...

    @abstractmethod
    async def poll(
        self, timeout_ms: int = 1000
    ) -> list[tuple[str, int, int, dict[str, Any]]]: ...

    @abstractmethod
    async def commit(self, offsets: dict[tuple[str, int], int]) -> None: ...


class AIOKafkaConsumerAdapter(KafkaConsumerAdapter):
    """aiokafka consumer implementation."""

    def __init__(
        self, kafka_settings: KafkaSettings, consumer_settings: ConsumerSettings
    ) -> None:
        self._kafka_settings = kafka_settings
        self._consumer_settings = consumer_settings
        self._consumer: Any = None

    async def start(self) -> None:
        """Start the consumer."""
        from aiokafka import AIOKafkaConsumer
        import json

        params = {
            **self._kafka_settings.get_connection_params(),
            **self._consumer_settings.to_consumer_params(),
            "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
        }
        self._consumer = AIOKafkaConsumer(**params)
        await self._consumer.start()
        logger.info("Kafka consumer started", group_id=self._consumer_settings.group_id)

    async def stop(self) -> None:
        """Stop the consumer."""
        if self._consumer:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped")

    async def subscribe(self, topics: list[str]) -> None:
        """Subscribe to topics."""
        if self._consumer:
            self._consumer.subscribe(topics)
            logger.info("Subscribed to topics", topics=topics)

    async def poll(
        self, timeout_ms: int = 1000
    ) -> list[tuple[str, int, int, dict[str, Any]]]:
        """Poll for messages."""
        if not self._consumer:
            return []
        messages: list[tuple[str, int, int, dict[str, Any]]] = []
        data = await self._consumer.getmany(timeout_ms=timeout_ms)
        for tp, records in data.items():
            for record in records:
                messages.append((tp.topic, tp.partition, record.offset, record.value))
        return messages

    async def commit(self, offsets: dict[tuple[str, int], int]) -> None:
        """Commit offsets."""
        if not self._consumer or not offsets:
            return
        from aiokafka import TopicPartition

        tp_offsets = {TopicPartition(t, p): o for (t, p), o in offsets.items()}
        await self._consumer.commit(tp_offsets)
        logger.debug("Offsets committed", count=len(offsets))


class MockKafkaConsumerAdapter(KafkaConsumerAdapter):
    """Mock consumer for testing without Kafka."""

    def __init__(self) -> None:
        self._messages: list[tuple[str, int, int, dict[str, Any]]] = []
        self._subscribed_topics: list[str] = []
        self._committed_offsets: dict[tuple[str, int], int] = {}
        self._started = False
        self._poll_index = 0

    async def start(self) -> None:
        self._started = True
        logger.info("Mock Kafka consumer started")

    async def stop(self) -> None:
        self._started = False
        logger.info("Mock Kafka consumer stopped")

    async def subscribe(self, topics: list[str]) -> None:
        self._subscribed_topics = topics
        logger.info("Mock subscribed to topics", topics=topics)

    async def poll(
        self, timeout_ms: int = 1000
    ) -> list[tuple[str, int, int, dict[str, Any]]]:
        if not self._started or self._poll_index >= len(self._messages):
            await asyncio.sleep(timeout_ms / 1000)
            return []
        batch_end = min(self._poll_index + 10, len(self._messages))
        messages = self._messages[self._poll_index : batch_end]
        self._poll_index = batch_end
        return messages

    async def commit(self, offsets: dict[tuple[str, int], int]) -> None:
        self._committed_offsets.update(offsets)
        logger.debug("Mock offsets committed", count=len(offsets))

    def add_message(
        self, topic: str, partition: int, offset: int, value: dict[str, Any]
    ) -> None:
        """Add message to mock queue."""
        self._messages.append((topic, partition, offset, value))

    def get_committed_offsets(self) -> dict[tuple[str, int], int]:
        """Get committed offsets."""
        return self._committed_offsets.copy()

    def clear(self) -> None:
        """Clear all messages and reset state."""
        self._messages.clear()
        self._committed_offsets.clear()
        self._poll_index = 0


class EventConsumer:
    """Event consumer with handler registration and offset management."""

    def __init__(
        self,
        consumer: KafkaConsumerAdapter,
        commit_interval_ms: int = 5000,
        max_poll_records: int = 100,
    ) -> None:
        self._consumer = consumer
        self._commit_interval_ms = commit_interval_ms
        self._max_poll_records = max_poll_records
        self._handlers: dict[str, list[EventHandler]] = {}
        self._default_handlers: list[EventHandler] = []
        self._offset_tracker = OffsetTracker()
        self._metrics = ConsumerMetrics()
        self._running = False
        self._last_commit = datetime.now(timezone.utc)
        self._dead_letter_handler: (
            Callable[[BaseEvent, str], Awaitable[None]] | None
        ) = None

    @property
    def metrics(self) -> ConsumerMetrics:
        """Get consumer metrics."""
        return self._metrics

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register handler for specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info("Handler registered", event_type=event_type)

    def register_default_handler(self, handler: EventHandler) -> None:
        """Register default handler for unmatched events."""
        self._default_handlers.append(handler)
        logger.info("Default handler registered")

    def set_dead_letter_handler(
        self, handler: Callable[[BaseEvent, str], Awaitable[None]]
    ) -> None:
        """Set handler for dead letter events."""
        self._dead_letter_handler = handler
        logger.info("Dead letter handler set")

    async def start(self, topics: list[str] | list[SolaceTopic]) -> None:
        """Start consuming from topics."""
        await self._consumer.start()
        topic_names = [t.value if isinstance(t, SolaceTopic) else t for t in topics]
        await self._consumer.subscribe(topic_names)
        self._running = True
        logger.info("Event consumer started", topics=topic_names)

    async def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        await self._commit_offsets()
        await self._consumer.stop()
        logger.info("Event consumer stopped")

    async def consume_loop(self) -> None:
        """Main consumption loop."""
        while self._running:
            try:
                messages = await self._consumer.poll(timeout_ms=1000)
                for topic, partition, offset, value in messages:
                    await self._offset_tracker.track_received(topic, partition, offset)
                    result = await self._process_message(
                        topic, partition, offset, value
                    )
                    # Advance offset for all terminal statuses (SUCCESS, FAILED,
                    # SKIP). Only RETRY keeps the offset pending for reprocessing.
                    # Without this, a single failed message blocks the entire
                    # partition indefinitely (S-C1).
                    if result.status != ProcessingStatus.RETRY:
                        committable = await self._offset_tracker.mark_processed(
                            topic, partition, offset
                        )
                        if committable:
                            await self._maybe_commit({(topic, partition): committable})
                await self._maybe_commit_periodic()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Consumer loop error", error=str(e))
                await asyncio.sleep(1)

    async def _process_message(
        self, topic: str, partition: int, offset: int, value: dict[str, Any]
    ) -> ProcessingResult:
        """Process a single message."""
        start_time = datetime.now(timezone.utc)
        try:
            event = deserialize_event(value)
            if event is None:
                logger.warning(
                    "unknown_event_type_skipped",
                    topic=topic,
                    partition=partition,
                    offset=offset,
                    raw_type=value.get("event_type", "missing"),
                )
                return ProcessingResult(
                    status=ProcessingStatus.SKIP,
                    event_id=self._get_fallback_event_id(value, topic, partition, offset),
                )
            handlers = (
                self._handlers.get(event.event_type, []) or self._default_handlers
            )
            for handler in handlers:
                await handler(event)
            processing_time = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            self._metrics.record_success(processing_time)
            logger.debug(
                "Event processed",
                event_type=event.event_type,
                event_id=str(event.metadata.event_id),
                time_ms=processing_time,
            )
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS, event_id=event.metadata.event_id
            )
        except Exception as e:
            self._metrics.record_failure()
            logger.error(
                "Event processing failed",
                topic=topic,
                partition=partition,
                offset=offset,
                error=str(e),
            )
            if self._dead_letter_handler:
                try:
                    dlq_event = deserialize_event(value)
                    if dlq_event is not None:
                        await self._dead_letter_handler(dlq_event, str(e))
                    else:
                        logger.warning("dlq_skip_unknown_event", topic=topic, offset=offset)
                except Exception as dlq_err:
                    logger.error("dlq_handler_failed", error=str(dlq_err), topic=topic, offset=offset)
            # Generate unique fallback UUID to avoid duplicate tracking issues
            # Use offset/partition/topic hash to create deterministic but unique ID for failed events
            fallback_event_id = self._get_fallback_event_id(
                value, topic, partition, offset
            )
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                event_id=fallback_event_id,
                error=str(e),
            )

    def _get_fallback_event_id(
        self, value: dict[str, Any], topic: str, partition: int, offset: int
    ) -> UUID:
        """Generate a deterministic fallback event ID for failed events.

        Uses topic/partition/offset to create a unique but reproducible UUID,
        avoiding duplicate tracking issues from using UUID(int=0).
        """
        # Try to extract event_id from the message first
        event_id_str = value.get("metadata", {}).get("event_id")
        if event_id_str:
            try:
                return (
                    UUID(event_id_str)
                    if isinstance(event_id_str, str)
                    else event_id_str
                )
            except (ValueError, TypeError):
                pass

        # Generate deterministic UUID from message location
        import hashlib

        location_str = f"{topic}:{partition}:{offset}"
        hash_bytes = hashlib.sha256(location_str.encode()).digest()[:16]
        return UUID(bytes=hash_bytes)

    async def _maybe_commit(self, offsets: dict[tuple[str, int], int]) -> None:
        """Commit offsets if any."""
        if offsets:
            await self._consumer.commit(offsets)

    async def _maybe_commit_periodic(self) -> None:
        """Commit periodically based on interval."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self._last_commit).total_seconds() * 1000
        if elapsed >= self._commit_interval_ms:
            all_offsets = self._offset_tracker.get_all_committed()
            if all_offsets:
                await self._consumer.commit(all_offsets)
                self._last_commit = now

    async def _commit_offsets(self) -> None:
        """Commit all pending offsets."""
        all_offsets = self._offset_tracker.get_all_committed()
        if all_offsets:
            await self._consumer.commit(all_offsets)


def create_consumer(
    group_id: str,
    kafka_settings: KafkaSettings | None = None,
    consumer_settings: ConsumerSettings | None = None,
    use_mock: bool = False,
) -> EventConsumer:
    """Factory function to create event consumer."""
    settings = consumer_settings or ConsumerSettings(group_id=group_id)
    if use_mock:
        adapter: KafkaConsumerAdapter = MockKafkaConsumerAdapter()
    else:
        adapter = AIOKafkaConsumerAdapter(kafka_settings or KafkaSettings(), settings)
    return EventConsumer(adapter)

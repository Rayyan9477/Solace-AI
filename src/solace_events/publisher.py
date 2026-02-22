"""Solace-AI Event Publisher - Transactional publishing with outbox pattern."""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
import structlog

from .config import KafkaSettings, ProducerSettings, SolaceTopic
from .schemas import BaseEvent, get_topic_for_event

logger = structlog.get_logger(__name__)


class OutboxStatus(str, Enum):
    """Status of outbox record."""

    PENDING = "PENDING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class OutboxRecord(BaseModel):
    """Record in the transactional outbox."""

    id: UUID = Field(default_factory=uuid4)
    event_id: UUID
    event_type: str
    event_payload: dict[str, Any]
    aggregate_id: UUID
    topic: str
    partition_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: datetime | None = Field(default=None)
    status: OutboxStatus = Field(default=OutboxStatus.PENDING)
    retry_count: int = Field(default=0, ge=0)
    last_error: str | None = Field(default=None)

    model_config = ConfigDict(frozen=False)

    @classmethod
    def from_event(cls, event: BaseEvent, topic: str | None = None) -> OutboxRecord:
        """Create outbox record from event."""
        resolved_topic = topic or get_topic_for_event(event)
        return cls(
            event_id=event.metadata.event_id,
            event_type=event.event_type,
            event_payload=event.to_dict(),
            aggregate_id=event.user_id,
            topic=resolved_topic,
            partition_key=str(event.user_id),
        )


class OutboxStore(Protocol):
    """Protocol for outbox persistence."""

    async def save(self, record: OutboxRecord) -> None:
        """Save outbox record."""
        ...

    async def get_pending(self, limit: int = 100) -> list[OutboxRecord]:
        """Get pending records for publishing."""
        ...

    async def mark_published(self, record_id: UUID) -> None:
        """Mark record as published."""
        ...

    async def mark_failed(self, record_id: UUID, error: str) -> None:
        """Mark record as failed."""
        ...

    async def increment_retry(self, record_id: UUID) -> int:
        """Increment retry count and return new count."""
        ...


class InMemoryOutboxStore:
    """In-memory outbox store for testing and development."""

    def __init__(self) -> None:
        self._records: dict[UUID, OutboxRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, record: OutboxRecord) -> None:
        """Save outbox record."""
        async with self._lock:
            self._records[record.id] = record
            logger.debug("Outbox record saved", record_id=str(record.id))

    async def get_pending(self, limit: int = 100) -> list[OutboxRecord]:
        """Get pending records for publishing."""
        async with self._lock:
            pending = [
                r for r in self._records.values() if r.status == OutboxStatus.PENDING
            ]
            pending.sort(key=lambda r: r.created_at)
            return pending[:limit]

    async def mark_published(self, record_id: UUID) -> None:
        """Mark record as published."""
        async with self._lock:
            if record_id in self._records:
                self._records[record_id].status = OutboxStatus.PUBLISHED
                self._records[record_id].published_at = datetime.now(timezone.utc)

    async def mark_failed(self, record_id: UUID, error: str) -> None:
        """Mark record as failed."""
        async with self._lock:
            if record_id in self._records:
                self._records[record_id].status = OutboxStatus.FAILED
                self._records[record_id].last_error = error

    async def increment_retry(self, record_id: UUID) -> int:
        """Increment retry count and return new count."""
        async with self._lock:
            if record_id in self._records:
                self._records[record_id].retry_count += 1
                return self._records[record_id].retry_count
            return 0

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()


class KafkaProducerAdapter(ABC):
    """Abstract Kafka producer adapter."""

    @abstractmethod
    async def start(self) -> None:
        """Start the producer."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the producer."""
        ...

    @abstractmethod
    async def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        """Send message to topic."""
        ...


class AIOKafkaProducerAdapter(KafkaProducerAdapter):
    """aiokafka producer implementation."""

    def __init__(
        self, kafka_settings: KafkaSettings, producer_settings: ProducerSettings
    ) -> None:
        self._kafka_settings = kafka_settings
        self._producer_settings = producer_settings
        self._producer: Any = None

    async def start(self) -> None:
        """Start the producer."""
        from aiokafka import AIOKafkaProducer

        params = {
            **self._kafka_settings.get_connection_params(),
            **self._producer_settings.to_producer_params(),
            "value_serializer": lambda v: json.dumps(v, default=str).encode("utf-8"),
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
        }
        self._producer = AIOKafkaProducer(**params)
        await self._producer.start()
        logger.info(
            "Kafka producer started", bootstrap=self._kafka_settings.bootstrap_servers
        )

    async def stop(self) -> None:
        """Stop the producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        """Send message to topic."""
        if not self._producer:
            raise RuntimeError("Producer not started")
        await self._producer.send_and_wait(topic, value=value, key=key)
        logger.debug("Message sent", topic=topic, key=key)


class MockKafkaProducerAdapter(KafkaProducerAdapter):
    """Mock producer for testing without Kafka."""

    def __init__(self) -> None:
        self._messages: list[tuple[str, str, dict[str, Any]]] = []
        self._started = False

    async def start(self) -> None:
        """Start the producer."""
        self._started = True
        logger.info("Mock Kafka producer started")

    async def stop(self) -> None:
        """Stop the producer."""
        self._started = False
        logger.info("Mock Kafka producer stopped")

    async def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        """Send message to topic (stores in memory)."""
        if not self._started:
            raise RuntimeError("Producer not started")
        self._messages.append((topic, key, value))
        logger.debug("Mock message sent", topic=topic, key=key)

    def get_messages(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all sent messages."""
        return self._messages.copy()

    def clear(self) -> None:
        """Clear stored messages."""
        self._messages.clear()


class EventPublisher:
    """Transactional event publisher with outbox pattern support."""

    def __init__(
        self,
        producer: KafkaProducerAdapter,
        outbox_store: OutboxStore | None = None,
        use_outbox: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._producer = producer
        self._outbox_store = outbox_store or InMemoryOutboxStore()
        self._use_outbox = use_outbox
        self._max_retries = max_retries
        self._started = False

    async def start(self) -> None:
        """Start the publisher and ensure outbox tables exist."""
        await self._producer.start()
        if hasattr(self._outbox_store, "ensure_table"):
            await self._outbox_store.ensure_table()
        self._started = True
        logger.info("Event publisher started", use_outbox=self._use_outbox)

    async def stop(self) -> None:
        """Stop the publisher."""
        await self._producer.stop()
        self._started = False
        logger.info("Event publisher stopped")

    async def publish(self, event: BaseEvent, topic: str | None = None) -> UUID:
        """Publish event to Kafka."""
        if not self._started:
            raise RuntimeError("Publisher not started")
        resolved_topic = topic or get_topic_for_event(event)
        if self._use_outbox:
            return await self._publish_via_outbox(event, resolved_topic)
        return await self._publish_direct(event, resolved_topic)

    async def _publish_direct(self, event: BaseEvent, topic: str) -> UUID:
        """Publish event directly to Kafka."""
        partition_key = str(event.user_id)
        await self._producer.send(topic, partition_key, event.to_dict())
        logger.info(
            "Event published directly",
            event_id=str(event.metadata.event_id),
            event_type=event.event_type,
            topic=topic,
        )
        return event.metadata.event_id

    async def _publish_via_outbox(self, event: BaseEvent, topic: str) -> UUID:
        """Publish event via outbox pattern."""
        record = OutboxRecord.from_event(event, topic)
        await self._outbox_store.save(record)
        logger.info(
            "Event queued in outbox",
            event_id=str(event.metadata.event_id),
            event_type=event.event_type,
            outbox_id=str(record.id),
        )
        return event.metadata.event_id

    async def publish_batch(
        self, events: list[BaseEvent], topic: str | None = None
    ) -> list[UUID]:
        """Publish multiple events in parallel for better performance."""
        if not events:
            return []
        # Use asyncio.gather for parallel publishing
        tasks = [self.publish(event, topic) for event in events]
        return list(await asyncio.gather(*tasks))

    async def flush_outbox(self, batch_size: int = 100) -> int:
        """Flush pending outbox records to Kafka."""
        if not self._use_outbox:
            return 0
        pending = await self._outbox_store.get_pending(batch_size)
        published_count = 0
        for record in pending:
            try:
                await self._producer.send(
                    record.topic, record.partition_key, record.event_payload
                )
                await self._outbox_store.mark_published(record.id)
                published_count += 1
                logger.debug("Outbox record published", record_id=str(record.id))
            except Exception as e:
                retry_count = await self._outbox_store.increment_retry(record.id)
                if retry_count >= self._max_retries:
                    await self._outbox_store.mark_failed(record.id, str(e))
                    logger.error(
                        "Outbox publish failed permanently",
                        record_id=str(record.id),
                        error=str(e),
                    )
                else:
                    logger.warning(
                        "Outbox publish failed, will retry",
                        record_id=str(record.id),
                        retry=retry_count,
                    )
        return published_count


class OutboxPoller:
    """Background service that polls and publishes outbox records."""

    def __init__(
        self,
        publisher: EventPublisher,
        poll_interval_ms: int = 100,
        batch_size: int = 100,
    ) -> None:
        self._publisher = publisher
        self._poll_interval_ms = poll_interval_ms
        self._batch_size = batch_size
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the poller."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Outbox poller started", interval_ms=self._poll_interval_ms)

    async def stop(self) -> None:
        """Stop the poller."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Outbox poller stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                published = await self._publisher.flush_outbox(self._batch_size)
                if published > 0:
                    logger.debug("Outbox flush completed", published=published)
            except Exception as e:
                logger.error("Outbox poll error", error=str(e))
            await asyncio.sleep(self._poll_interval_ms / 1000)


def create_publisher(
    kafka_settings: KafkaSettings | None = None,
    producer_settings: ProducerSettings | None = None,
    outbox_store: OutboxStore | None = None,
    use_outbox: bool = True,
    use_mock: bool = False,
    postgres_pool: Any = None,
) -> EventPublisher:
    """Factory function to create event publisher.

    Args:
        kafka_settings: Kafka connection settings.
        producer_settings: Producer tuning settings.
        outbox_store: Explicit outbox store override.
        use_outbox: Whether to use outbox pattern.
        use_mock: Use mock producer (for testing).
        postgres_pool: asyncpg connection pool. When provided and no explicit
            outbox_store is given, uses PostgresOutboxStore for durable persistence.
    """
    if use_mock:
        producer: KafkaProducerAdapter = MockKafkaProducerAdapter()
    else:
        producer = AIOKafkaProducerAdapter(
            kafka_settings or KafkaSettings(),
            producer_settings or ProducerSettings(),
        )

    if outbox_store is None and use_outbox:
        if postgres_pool is not None:
            from .postgres_stores import PostgresOutboxStore
            outbox_store = PostgresOutboxStore(postgres_pool)
            logger.info("publisher_using_postgres_outbox")
        else:
            if os.environ.get("ENVIRONMENT", "").lower() == "production":
                raise RuntimeError(
                    "postgres_pool is required for event outbox in production. "
                    "In-memory outbox loses events on restart."
                )
            outbox_store = InMemoryOutboxStore()
            logger.warning(
                "publisher_using_in_memory_outbox",
                hint="Pass postgres_pool for durable persistence",
            )

    return EventPublisher(producer, outbox_store, use_outbox)

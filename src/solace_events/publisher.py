"""Solace-AI Event Publisher - Transactional publishing with outbox pattern."""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
import structlog

from .config import KafkaSettings, ProducerSettings, SolaceTopic
from .schemas import BaseEvent, get_topic_for_event

logger = structlog.get_logger(__name__)


# Default window after which a record left in PUBLISHING (a poller that crashed
# mid-send) is considered stale and re-claimable by another poller. Long enough to
# comfortably exceed a normal send+ack, short enough to bound delivery latency.
DEFAULT_CLAIM_STALE_SECONDS = 300.0

# P1-6: retry budget + per-record exponential backoff.
# The old default of 3 attempts meant a brief (seconds-long) broker/network blip
# could exhaust the budget and terminally FAIL a CRISIS event. Raise it so a
# transient outage of many minutes (with backoff) cannot silently drop an event
# before the durable DLQ recovery path even engages.
DEFAULT_MAX_RETRIES = 10
# Backoff base/cap for a failed outbox send. next_retry_at = now + base * 2**(n-1),
# capped at DEFAULT_RETRY_MAX_SECONDS, so a flapping producer does not re-attempt
# the same record on every tight poll and burn the retry budget in one loop.
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_RETRY_MAX_SECONDS = 300.0


def compute_backoff_seconds(retry_count: int, base: float, cap: float) -> float:
    """Exponential backoff delay (seconds) for the ``retry_count``-th attempt.

    ``base * 2**(retry_count - 1)`` capped at ``cap``. A non-positive ``base``
    disables the delay (0s), which callers use to exercise the immediate
    release->retry state machine deterministically in tests.
    """
    if base <= 0 or retry_count <= 0:
        return 0.0
    delay = base * (2 ** (retry_count - 1))
    return float(min(delay, cap))


class OutboxStatus(str, Enum):
    """Status of outbox record."""

    PENDING = "PENDING"
    PUBLISHING = "PUBLISHING"  # atomically claimed by a poller, send in flight
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class OutboxRecord(BaseModel):
    """Record in the transactional outbox."""

    id: UUID = Field(default_factory=uuid4)
    # MVP-133: the service whose poller owns (claims and publishes) this record;
    # event_outbox.source_service. None until a store stamps it with its own name on save.
    source_service: str | None = Field(default=None)
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
    # Set when a poller atomically claims the row (status -> PUBLISHING); used to
    # detect and reclaim stale claims left by a crashed poller.
    claimed_at: datetime | None = Field(default=None)
    # P1-6: earliest time a PENDING record may be re-claimed. Set to now+backoff
    # after a failed send so the record is not re-attempted on the very next poll.
    # None means immediately claimable.
    next_retry_at: datetime | None = Field(default=None)

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

    async def get_pending(
        self, limit: int = 100, stale_after_seconds: float = DEFAULT_CLAIM_STALE_SECONDS
    ) -> list[OutboxRecord]:
        """Atomically CLAIM up to ``limit`` pending records for publishing.

        Claiming flips PENDING -> PUBLISHING (recording ``claimed_at``) in the same
        atomic step that selects the rows, so two pollers never claim the SAME row
        in the same instant. A PENDING record whose ``next_retry_at`` is in the
        future (backoff after a failed send) is skipped until the window elapses.
        Records left in PUBLISHING longer than ``stale_after_seconds`` (a poller
        that crashed mid-send) are reclaimed.

        NOTE: this is AT-LEAST-ONCE, not exactly-once. The stale-claim reclaim can
        re-publish a record whose owner sent it but died before ``mark_published``,
        so downstream consumers must dedupe on ``event.metadata.event_id`` (see
        ``IdempotencyStore``). ``mark_published`` is claim-conditional so a poller
        whose claim was stolen does not finalize a row it no longer owns.
        """
        ...

    async def mark_published(self, record_id: UUID) -> bool:
        """Finalize a record THIS poller still owns (PUBLISHING -> PUBLISHED).

        Conditional on the row still being PUBLISHING: returns True if it was
        marked, False if the claim had been stolen/reset (row no longer owned),
        so the caller does not count a publish another poller will finalize.
        """
        ...

    async def mark_failed(self, record_id: UUID, error: str) -> None:
        """Mark record as failed."""
        ...

    async def increment_retry(self, record_id: UUID) -> int:
        """Increment retry count and return new count."""
        ...

    async def reset_to_pending(
        self, record_id: UUID, next_retry_at: datetime | None = None
    ) -> None:
        """Release a claimed record back to PENDING for retry.

        ``next_retry_at`` (when given) delays the next claim to enforce backoff;
        None makes it immediately claimable again.
        """
        ...


class IdempotencyStore(Protocol):
    """Consumer-side dedup SEAM for at-least-once delivery (P2-8).

    The outbox is at-least-once: the stale-claim reclaim can re-publish a crisis
    event whose owning poller died between send and ``mark_published``. Consumers
    MUST guard side effects (e.g. firing a crisis escalation) behind a
    processed-event check keyed on ``event.metadata.event_id`` so a duplicate
    delivery cannot double-fire. ``mark_processed`` is the atomic check-and-set:
    it returns True exactly once per (event_id, consumer_group) — the first
    caller processes, every later duplicate gets False and skips.
    """

    async def mark_processed(self, event_id: UUID, consumer_group: str) -> bool:
        """Atomically record (event_id, consumer_group). True if first-seen."""
        ...

    async def was_processed(self, event_id: UUID, consumer_group: str) -> bool:
        """Whether this (event_id, consumer_group) was already processed."""
        ...


class InMemoryIdempotencyStore:
    """In-memory idempotency store for tests/dev (mirrors the Postgres seam)."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()
        self._lock = asyncio.Lock()

    async def mark_processed(self, event_id: UUID, consumer_group: str) -> bool:
        async with self._lock:
            key = (str(event_id), consumer_group)
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    async def was_processed(self, event_id: UUID, consumer_group: str) -> bool:
        async with self._lock:
            return (str(event_id), consumer_group) in self._seen

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._seen.clear()


class InMemoryOutboxStore:
    """In-memory outbox store for testing and development.

    Mirrors ``PostgresOutboxStore``'s per-service ownership (MVP-133): ``save`` stamps a
    record that has no ``source_service`` with this store's, and ``get_pending`` claims
    only records whose ``source_service`` matches.
    """

    def __init__(self, source_service: str = "local") -> None:
        self._source_service = source_service
        self._records: dict[UUID, OutboxRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, record: OutboxRecord) -> None:
        """Save outbox record, stamping an unset ``source_service`` with this store's."""
        async with self._lock:
            if record.source_service is None:
                record.source_service = self._source_service
            self._records[record.id] = record
            logger.debug("Outbox record saved", record_id=str(record.id))

    async def get_pending(
        self, limit: int = 100, stale_after_seconds: float = DEFAULT_CLAIM_STALE_SECONDS
    ) -> list[OutboxRecord]:
        """Atomically claim pending (and reclaim stale) records for publishing.

        Mirrors the Postgres store's claim semantics so unit tests exercise the
        real contract: a claimed record flips to PUBLISHING and is not returned to
        a second concurrent caller; a PUBLISHING record whose ``claimed_at`` is
        older than ``stale_after_seconds`` is reclaimed. Only this store's service's
        records are claimed (MVP-133).
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            stale_cutoff = now - timedelta(seconds=stale_after_seconds)

            def _claimable(r: OutboxRecord) -> bool:
                if r.source_service != self._source_service:
                    return False
                if r.status == OutboxStatus.PENDING:
                    # Respect the backoff window: a record scheduled for a future
                    # retry is not yet claimable (P1-6).
                    return r.next_retry_at is None or r.next_retry_at <= now
                return (
                    r.status == OutboxStatus.PUBLISHING
                    and r.claimed_at is not None
                    and r.claimed_at < stale_cutoff
                )

            claimable = [r for r in self._records.values() if _claimable(r)]
            claimable.sort(key=lambda r: r.created_at)
            claimed = claimable[:limit]
            for r in claimed:
                r.status = OutboxStatus.PUBLISHING
                r.claimed_at = now
            return claimed

    async def mark_published(self, record_id: UUID) -> bool:
        """Finalize a record only while THIS poller still owns it (PUBLISHING).

        Conditional so a poller whose claim was stolen by the stale-reclaim sweep
        does not clobber a row another poller now owns (P2-8). Returns whether the
        record was actually marked PUBLISHED.
        """
        async with self._lock:
            record = self._records.get(record_id)
            if record is None or record.status != OutboxStatus.PUBLISHING:
                return False
            record.status = OutboxStatus.PUBLISHED
            record.published_at = datetime.now(timezone.utc)
            record.claimed_at = None
            return True

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

    async def reset_to_pending(
        self, record_id: UUID, next_retry_at: datetime | None = None
    ) -> None:
        """Release a claimed record back to PENDING, optionally after a backoff."""
        async with self._lock:
            record = self._records.get(record_id)
            if record is not None:
                record.status = OutboxStatus.PENDING
                record.claimed_at = None
                record.next_retry_at = next_retry_at

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
        max_retries: int = DEFAULT_MAX_RETRIES,
        dlq_store: Any = None,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
    ) -> None:
        self._producer = producer
        self._outbox_store = outbox_store or InMemoryOutboxStore()
        self._use_outbox = use_outbox
        self._max_retries = max_retries
        # P1-6: durable recovery path for a permanently-failed event. When set,
        # a record that exhausts its retry budget is routed here (a retriable DLQ
        # record) instead of dying in a terminal FAILED outbox row with only a log.
        self._dlq_store = dlq_store
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._started = False

    async def start(self) -> None:
        """Start the publisher and verify the outbox tables (slot 010; no DDL, MVP-266)."""
        await self._producer.start()
        if hasattr(self._outbox_store, "ensure_table"):
            await self._outbox_store.ensure_table()
        # P1-6: the DLQ store is the durable recovery path for a permanently-FAILED
        # (crisis) event — its table must be verified too, else routing a failed
        # record to the DLQ is dead-on-arrival (no table to write to).
        if self._dlq_store is not None and hasattr(self._dlq_store, "ensure_table"):
            await self._dlq_store.ensure_table()
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
                marked = await self._outbox_store.mark_published(record.id)
                if marked:
                    published_count += 1
                    logger.debug("Outbox record published", record_id=str(record.id))
                else:
                    # Our claim was stolen mid-send (stale-reclaim). Another poller
                    # owns this row now and will finalize it — do not count it.
                    # Delivery is at-least-once; consumers dedupe on event_id.
                    logger.warning(
                        "outbox_mark_published_race",
                        record_id=str(record.id),
                        event_id=str(record.event_id),
                    )
            except Exception as e:
                retry_count = await self._outbox_store.increment_retry(record.id)
                if retry_count >= self._max_retries:
                    # P1-6: route to the durable DLQ (retriable) BEFORE flipping the
                    # outbox row terminal, so a permanently-failed event is never
                    # lost with only a log line.
                    await self._route_to_dlq(record, e, retry_count)
                    await self._outbox_store.mark_failed(record.id, str(e))
                    logger.error(
                        "outbox_publish_failed_permanently",
                        record_id=str(record.id),
                        event_id=str(record.event_id),
                        retry=retry_count,
                        routed_to_dlq=self._dlq_store is not None,
                        error=str(e),
                    )
                else:
                    # Release the claim with an exponential-backoff window so a
                    # flapping producer does not re-attempt on every poll and burn
                    # the retry budget in one loop.
                    delay = compute_backoff_seconds(
                        retry_count, self._retry_base_seconds, self._retry_max_seconds
                    )
                    next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                    await self._outbox_store.reset_to_pending(
                        record.id, next_retry_at=next_retry_at
                    )
                    logger.warning(
                        "outbox_publish_failed_will_retry",
                        record_id=str(record.id),
                        retry=retry_count,
                        backoff_seconds=delay,
                    )
        return published_count

    async def _route_to_dlq(
        self, record: OutboxRecord, error: Exception, retry_count: int
    ) -> None:
        """Persist a permanently-failed outbox record into the durable DLQ (P1-6).

        The DLQ record is marked retriable with a ``next_retry_at`` so the DLQ
        reaper (PostgresDLQStore.get_retriable) can re-drive it, and it remains a
        durable, alertable record of the failed event. If no DLQ store is wired we
        log loudly — a failed crisis event must never disappear silently.
        """
        if self._dlq_store is None:
            logger.error(
                "outbox_permanent_failure_no_dlq_store",
                record_id=str(record.id),
                event_id=str(record.event_id),
                event_type=record.event_type,
                hint="wire a dlq_store so failed events are recoverable/alertable",
            )
            return
        try:
            from .dead_letter import DeadLetterRecord, get_dlq_topic

            delay = compute_backoff_seconds(
                retry_count, self._retry_base_seconds, self._retry_max_seconds
            )
            dlq_record = DeadLetterRecord(
                original_topic=record.topic,
                dlq_topic=get_dlq_topic(record.topic),
                original_event=record.event_payload,
                event_type=record.event_type,
                event_id=record.event_id,
                user_id=record.aggregate_id,
                consumer_group="outbox-publisher",
                error_message=str(error),
                error_type=type(error).__name__,
                retry_count=retry_count,
                is_retriable=True,
                next_retry_at=datetime.now(timezone.utc) + timedelta(seconds=delay),
            )
            await self._dlq_store.save(dlq_record)
            logger.error(
                "outbox_event_routed_to_dlq",
                record_id=str(record.id),
                event_id=str(record.event_id),
                dlq_topic=dlq_record.dlq_topic,
            )
        except Exception as dlq_error:  # pragma: no cover - defensive
            # Never let a DLQ-write failure mask the original error or crash the
            # poll loop; surface it loudly for alerting.
            logger.error(
                "outbox_dlq_route_failed",
                record_id=str(record.id),
                event_id=str(record.event_id),
                error=str(dlq_error),
            )


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
    dlq_store: Any = None,
    source_service: str | None = None,
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
            Requires ``source_service``.
        dlq_store: Explicit DLQ store override for the permanent-failure recovery
            path (P1-6). When omitted and ``postgres_pool`` is given, a
            PostgresDLQStore is wired automatically so a permanently-failed event
            lands in a durable, retriable DLQ record instead of a terminal log.
        source_service: The calling service's name (MVP-133). The outbox store
            writes it into every record and claims only that service's records, so
            each service's poller publishes its own outbox rows only. Required with
            ``postgres_pool``; without one it scopes the in-memory store
            (default ``"local"``).

    Raises:
        ValueError: ``postgres_pool`` is given without ``source_service``.
    """
    if postgres_pool is not None and source_service is None:
        raise ValueError(
            "create_publisher: source_service is required with postgres_pool, so the "
            "service's outbox poller claims only its own event_outbox rows (MVP-133)"
        )
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
            outbox_store = PostgresOutboxStore(postgres_pool, source_service=source_service)
            logger.info("publisher_using_postgres_outbox", source_service=source_service)
        else:
            if os.environ.get("ENVIRONMENT", "").lower() == "production":
                raise RuntimeError(
                    "postgres_pool is required for event outbox in production. "
                    "In-memory outbox loses events on restart."
                )
            outbox_store = (
                InMemoryOutboxStore()
                if source_service is None
                else InMemoryOutboxStore(source_service=source_service)
            )
            logger.warning(
                "publisher_using_in_memory_outbox",
                hint="Pass postgres_pool for durable persistence",
            )

    # P1-6: wire a durable DLQ recovery path so a permanently-failed event (e.g. a
    # crisis event) is preserved and retriable, not dropped after the retry budget.
    if dlq_store is None and use_outbox and postgres_pool is not None:
        from .postgres_stores import PostgresDLQStore
        dlq_store = PostgresDLQStore(postgres_pool)
        logger.info("publisher_using_postgres_dlq_recovery")

    return EventPublisher(producer, outbox_store, use_outbox, dlq_store=dlq_store)

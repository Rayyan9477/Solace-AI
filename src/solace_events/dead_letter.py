"""Solace-AI Dead Letter Queue - DLQ handling with retry policies."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
import structlog

from .config import SolaceTopic
from .schemas import BaseEvent, deserialize_event

logger = structlog.get_logger(__name__)


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    FIXED = "FIXED"
    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"
    DECORRELATED = "DECORRELATED"


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_ms: int = Field(default=1000, ge=100)
    max_delay_ms: int = Field(default=30000, ge=1000)
    strategy: RetryStrategy = Field(default=RetryStrategy.EXPONENTIAL)
    multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    jitter_percent: float = Field(default=0.1, ge=0.0, le=0.5)

    model_config = ConfigDict(frozen=True)

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for given attempt number."""
        if attempt <= 0:
            return self.initial_delay_ms
        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay_ms
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay_ms * attempt
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay_ms * (self.multiplier ** (attempt - 1))
        else:
            import random
            delay = random.uniform(self.initial_delay_ms, min(
                self.max_delay_ms,
                self.initial_delay_ms * (self.multiplier ** attempt)
            ))
        delay = min(delay, self.max_delay_ms)
        # DECORRELATED already incorporates randomness — skip additional jitter
        if self.jitter_percent > 0 and self.strategy != RetryStrategy.DECORRELATED:
            import random
            jitter = delay * self.jitter_percent
            delay += random.uniform(-jitter, jitter)
        return int(max(delay, self.initial_delay_ms))


class DeadLetterRecord(BaseModel):
    """Record of a failed event in DLQ."""

    id: UUID = Field(default_factory=uuid4)
    original_topic: str
    dlq_topic: str
    original_event: dict[str, Any]
    event_type: str
    event_id: UUID
    user_id: UUID
    consumer_group: str
    error_message: str
    error_type: str
    stack_trace: str | None = Field(default=None)
    retry_count: int = Field(default=0, ge=0)
    first_failure_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_failure_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dlq_entry_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    next_retry_at: datetime | None = Field(default=None)
    is_retriable: bool = Field(default=True)
    resolved: bool = Field(default=False)
    resolution_notes: str | None = Field(default=None)

    model_config = ConfigDict(frozen=False)

    @classmethod
    def from_failed_event(
        cls,
        event: BaseEvent,
        original_topic: str,
        consumer_group: str,
        error: Exception,
        retry_count: int = 0,
    ) -> DeadLetterRecord:
        """Create DLQ record from failed event."""
        import traceback
        dlq_topic = f"{original_topic}.dlq"
        return cls(
            original_topic=original_topic,
            dlq_topic=dlq_topic,
            original_event=event.to_dict(),
            event_type=event.event_type,
            event_id=event.metadata.event_id,
            user_id=event.user_id,
            consumer_group=consumer_group,
            error_message=str(error),
            error_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            retry_count=retry_count,
        )

    def increment_retry(self, error: Exception, policy: RetryPolicy) -> bool:
        """Increment retry count. Returns True if more retries allowed."""
        self.retry_count += 1
        self.last_failure_at = datetime.now(timezone.utc)
        self.error_message = str(error)
        if self.retry_count >= policy.max_retries:
            self.is_retriable = False
            self.next_retry_at = None
            return False
        delay_ms = policy.get_delay_ms(self.retry_count)
        from datetime import timedelta
        self.next_retry_at = datetime.now(timezone.utc) + timedelta(milliseconds=delay_ms)
        return True


class DLQStore(Protocol):
    """Protocol for DLQ persistence backends."""

    async def save(self, record: DeadLetterRecord) -> None: ...
    async def get(self, record_id: UUID) -> DeadLetterRecord | None: ...
    async def get_retriable(self, topic: str | None = None, limit: int = 100) -> list[DeadLetterRecord]: ...
    async def get_by_topic(self, dlq_topic: str, limit: int = 100) -> list[DeadLetterRecord]: ...
    async def mark_resolved(self, record_id: UUID, notes: str | None = None) -> None: ...
    async def delete(self, record_id: UUID) -> None: ...
    async def count_by_topic(self) -> dict[str, int]: ...
    async def count_unresolved(self) -> int: ...


class DeadLetterStore:
    """In-memory store for dead letter records (for testing and development)."""

    def __init__(self) -> None:
        self._records: dict[UUID, DeadLetterRecord] = {}
        self._by_topic: dict[str, list[UUID]] = {}
        self._lock = asyncio.Lock()

    async def save(self, record: DeadLetterRecord) -> None:
        """Save DLQ record."""
        async with self._lock:
            self._records[record.id] = record
            if record.dlq_topic not in self._by_topic:
                self._by_topic[record.dlq_topic] = []
            if record.id not in self._by_topic[record.dlq_topic]:
                self._by_topic[record.dlq_topic].append(record.id)
            logger.info("DLQ record saved", record_id=str(record.id),
                        event_type=record.event_type, topic=record.dlq_topic)

    async def get(self, record_id: UUID) -> DeadLetterRecord | None:
        """Get DLQ record by ID."""
        async with self._lock:
            return self._records.get(record_id)

    async def get_retriable(self, topic: str | None = None, limit: int = 100) -> list[DeadLetterRecord]:
        """Get retriable records ready for retry."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            records = list(self._records.values())
            if topic:
                records = [r for r in records if r.original_topic == topic]
            retriable = [
                r for r in records
                if r.is_retriable and not r.resolved
                and (r.next_retry_at is None or r.next_retry_at <= now)
            ]
            retriable.sort(key=lambda r: r.next_retry_at or r.dlq_entry_at)
            return retriable[:limit]

    async def get_by_topic(self, dlq_topic: str, limit: int = 100) -> list[DeadLetterRecord]:
        """Get records for a specific DLQ topic."""
        async with self._lock:
            record_ids = self._by_topic.get(dlq_topic, [])[:limit]
            return [self._records[rid] for rid in record_ids if rid in self._records]

    async def mark_resolved(self, record_id: UUID, notes: str | None = None) -> None:
        """Mark record as resolved."""
        async with self._lock:
            if record_id in self._records:
                self._records[record_id].resolved = True
                self._records[record_id].resolution_notes = notes
                logger.info("DLQ record resolved", record_id=str(record_id))

    async def delete(self, record_id: UUID) -> None:
        """Delete DLQ record."""
        async with self._lock:
            if record_id in self._records:
                record = self._records.pop(record_id)
                if record.dlq_topic in self._by_topic:
                    self._by_topic[record.dlq_topic] = [
                        rid for rid in self._by_topic[record.dlq_topic] if rid != record_id
                    ]

    async def count_by_topic(self) -> dict[str, int]:
        """Count records by DLQ topic."""
        async with self._lock:
            return {topic: len(ids) for topic, ids in self._by_topic.items()}

    async def count_unresolved(self) -> int:
        """Count unresolved records."""
        async with self._lock:
            return sum(1 for r in self._records.values() if not r.resolved)

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._by_topic.clear()


RetryHandler = Callable[[BaseEvent], Awaitable[bool]]


class DeadLetterHandler:
    """Handler for dead letter queue operations."""

    def __init__(
        self,
        store: DLQStore | None = None,
        retry_policy: RetryPolicy | None = None,
        consumer_group: str = "unknown",
    ) -> None:
        self._store: DLQStore = store or DeadLetterStore()
        self._retry_policy = retry_policy or RetryPolicy()
        self._consumer_group = consumer_group
        self._retry_handlers: dict[str, RetryHandler] = {}

    @property
    def store(self) -> DLQStore:
        """Get the DLQ store."""
        return self._store

    def register_retry_handler(self, event_type: str, handler: RetryHandler) -> None:
        """Register handler for retrying specific event type."""
        self._retry_handlers[event_type] = handler
        logger.info("Retry handler registered", event_type=event_type)

    async def handle_failure(
        self,
        event: BaseEvent,
        original_topic: str,
        error: Exception,
        retry_count: int = 0,
    ) -> DeadLetterRecord:
        """Handle a failed event by creating/updating DLQ record."""
        record = DeadLetterRecord.from_failed_event(
            event, original_topic, self._consumer_group, error, retry_count
        )
        can_retry = record.increment_retry(error, self._retry_policy)
        if not can_retry:
            logger.warning("Event exhausted retries, moving to DLQ permanently",
                           event_id=str(event.metadata.event_id),
                           event_type=event.event_type, retries=retry_count)
        await self._store.save(record)
        return record

    async def retry_record(self, record: DeadLetterRecord) -> bool:
        """Attempt to retry a DLQ record."""
        if not record.is_retriable or record.resolved:
            return False
        handler = self._retry_handlers.get(record.event_type)
        if not handler:
            logger.warning("No retry handler for event type", event_type=record.event_type)
            return False
        try:
            event = deserialize_event(record.original_event)
            success = await handler(event)
            if success:
                await self._store.mark_resolved(record.id, "Retry successful")
                logger.info("DLQ record retry successful", record_id=str(record.id))
                return True
            can_retry = record.increment_retry(
                Exception("Retry handler returned False"), self._retry_policy
            )
            await self._store.save(record)
            return can_retry
        except Exception as e:
            can_retry = record.increment_retry(e, self._retry_policy)
            await self._store.save(record)
            logger.error("DLQ retry failed", record_id=str(record.id), error=str(e))
            return can_retry

    async def process_retriable(self, topic: str | None = None, limit: int = 10) -> int:
        """Process retriable records. Returns count of successful retries."""
        records = await self._store.get_retriable(topic, limit)
        success_count = 0
        for record in records:
            if await self.retry_record(record):
                success_count += 1
        return success_count

    async def get_dlq_stats(self) -> dict[str, Any]:
        """Get DLQ statistics."""
        counts = await self._store.count_by_topic()
        unresolved = await self._store.count_unresolved()
        return {
            "total_unresolved": unresolved,
            "by_topic": counts,
            "retry_policy": {
                "max_retries": self._retry_policy.max_retries,
                "strategy": self._retry_policy.strategy.value,
            },
        }


def get_dlq_topic(topic: str | SolaceTopic) -> str:
    """Get DLQ topic name for a given topic."""
    if isinstance(topic, SolaceTopic):
        return topic.dlq_topic
    return f"{topic}.dlq"


def create_dead_letter_handler(
    consumer_group: str,
    retry_policy: RetryPolicy | None = None,
    store: DLQStore | None = None,
    postgres_pool: Any = None,
) -> DeadLetterHandler:
    """Factory function to create dead letter handler.

    Args:
        consumer_group: Name of the consumer group for tracking.
        retry_policy: Retry configuration.
        store: Explicit DLQ store override.
        postgres_pool: asyncpg connection pool. When provided and no explicit
            store is given, uses PostgresDLQStore for durable persistence.
    """
    if store is None:
        if postgres_pool is not None:
            from .postgres_stores import PostgresDLQStore
            store = PostgresDLQStore(postgres_pool)
            logger.info("dlq_using_postgres_store", consumer_group=consumer_group)
        else:
            if os.environ.get("ENVIRONMENT", "").lower() == "production":
                raise RuntimeError(
                    "postgres_pool is required for DLQ in production. "
                    "In-memory DLQ loses failed events on restart."
                )
            store = DeadLetterStore()
            logger.warning(
                "dlq_using_in_memory_store",
                consumer_group=consumer_group,
                hint="Pass postgres_pool for durable persistence",
            )
    return DeadLetterHandler(
        store=store,
        retry_policy=retry_policy,
        consumer_group=consumer_group,
    )

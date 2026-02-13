"""
Solace-AI Persistent Event Stores - PostgreSQL-backed outbox and DLQ stores.

Replaces in-memory stores with durable PostgreSQL persistence for production use.
Events survive service restarts, enabling reliable transactional outbox and DLQ patterns.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from .publisher import OutboxRecord, OutboxStatus
from .dead_letter import DeadLetterRecord

logger = structlog.get_logger(__name__)

# SQL for creating tables (run during migration or startup)
OUTBOX_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS event_outbox (
    id UUID PRIMARY KEY,
    event_id UUID NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_payload JSONB NOT NULL,
    aggregate_id UUID NOT NULL,
    topic VARCHAR(255) NOT NULL,
    partition_key VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    retry_count INT NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status_created ON event_outbox (status, created_at)
    WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_outbox_topic ON event_outbox (topic);
"""

DLQ_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS event_dead_letter (
    id UUID PRIMARY KEY,
    original_topic VARCHAR(255) NOT NULL,
    dlq_topic VARCHAR(255) NOT NULL,
    original_event JSONB NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_id UUID NOT NULL,
    user_id UUID NOT NULL,
    consumer_group VARCHAR(255) NOT NULL,
    error_message TEXT NOT NULL,
    error_type VARCHAR(255) NOT NULL,
    stack_trace TEXT,
    retry_count INT NOT NULL DEFAULT 0,
    first_failure_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_failure_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dlq_entry_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    next_retry_at TIMESTAMPTZ,
    is_retriable BOOLEAN NOT NULL DEFAULT TRUE,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolution_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_dlq_retriable ON event_dead_letter (is_retriable, next_retry_at)
    WHERE NOT resolved AND is_retriable;
CREATE INDEX IF NOT EXISTS idx_dlq_topic ON event_dead_letter (dlq_topic);
CREATE INDEX IF NOT EXISTS idx_dlq_resolved ON event_dead_letter (resolved) WHERE NOT resolved;
"""


class PostgresOutboxStore:
    """PostgreSQL-backed outbox store for transactional event publishing.

    Implements the OutboxStore protocol from publisher.py.
    Requires an asyncpg connection pool.
    """

    def __init__(self, pool: Any) -> None:
        """Initialize with an asyncpg connection pool.

        Args:
            pool: asyncpg.Pool instance
        """
        self._pool = pool

    async def ensure_table(self) -> None:
        """Create the outbox table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(OUTBOX_TABLE_DDL)
            logger.info("outbox_table_ensured")

    async def save(self, record: OutboxRecord) -> None:
        """Save outbox record to PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO event_outbox
                   (id, event_id, event_type, event_payload, aggregate_id,
                    topic, partition_key, created_at, published_at, status,
                    retry_count, last_error)
                   VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12)
                   ON CONFLICT (id) DO NOTHING""",
                record.id, record.event_id, record.event_type,
                json.dumps(record.event_payload, default=str),
                record.aggregate_id, record.topic, record.partition_key,
                record.created_at, record.published_at, record.status.value,
                record.retry_count, record.last_error,
            )
            logger.debug("outbox_record_saved", record_id=str(record.id))

    async def get_pending(self, limit: int = 100) -> list[OutboxRecord]:
        """Get pending records ordered by creation time."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM event_outbox
                   WHERE status = 'PENDING'
                   ORDER BY created_at ASC
                   LIMIT $1
                   FOR UPDATE SKIP LOCKED""",
                limit,
            )
            return [self._row_to_record(row) for row in rows]

    async def mark_published(self, record_id: UUID) -> None:
        """Mark record as published."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE event_outbox
                   SET status = 'PUBLISHED', published_at = $2
                   WHERE id = $1""",
                record_id, datetime.now(timezone.utc),
            )

    async def mark_failed(self, record_id: UUID, error: str) -> None:
        """Mark record as failed."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE event_outbox
                   SET status = 'FAILED', last_error = $2
                   WHERE id = $1""",
                record_id, error,
            )

    async def increment_retry(self, record_id: UUID) -> int:
        """Increment retry count and return new count."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE event_outbox
                   SET retry_count = retry_count + 1
                   WHERE id = $1
                   RETURNING retry_count""",
                record_id,
            )
            return row["retry_count"] if row else 0

    @staticmethod
    def _row_to_record(row: Any) -> OutboxRecord:
        """Convert database row to OutboxRecord."""
        payload = row["event_payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return OutboxRecord(
            id=row["id"], event_id=row["event_id"], event_type=row["event_type"],
            event_payload=payload, aggregate_id=row["aggregate_id"],
            topic=row["topic"], partition_key=row["partition_key"],
            created_at=row["created_at"], published_at=row["published_at"],
            status=OutboxStatus(row["status"]), retry_count=row["retry_count"],
            last_error=row["last_error"],
        )


class PostgresDLQStore:
    """PostgreSQL-backed dead letter queue store.

    Implements the same interface as DeadLetterStore from dead_letter.py.
    Requires an asyncpg connection pool.
    """

    def __init__(self, pool: Any) -> None:
        """Initialize with an asyncpg connection pool.

        Args:
            pool: asyncpg.Pool instance
        """
        self._pool = pool

    async def ensure_table(self) -> None:
        """Create the DLQ table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(DLQ_TABLE_DDL)
            logger.info("dlq_table_ensured")

    async def save(self, record: DeadLetterRecord) -> None:
        """Save DLQ record to PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO event_dead_letter
                   (id, original_topic, dlq_topic, original_event, event_type,
                    event_id, user_id, consumer_group, error_message, error_type,
                    stack_trace, retry_count, first_failure_at, last_failure_at,
                    dlq_entry_at, next_retry_at, is_retriable, resolved, resolution_notes)
                   VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10,
                           $11, $12, $13, $14, $15, $16, $17, $18, $19)
                   ON CONFLICT (id) DO UPDATE SET
                       retry_count = EXCLUDED.retry_count,
                       last_failure_at = EXCLUDED.last_failure_at,
                       next_retry_at = EXCLUDED.next_retry_at,
                       is_retriable = EXCLUDED.is_retriable,
                       error_message = EXCLUDED.error_message,
                       resolved = EXCLUDED.resolved,
                       resolution_notes = EXCLUDED.resolution_notes""",
                record.id, record.original_topic, record.dlq_topic,
                json.dumps(record.original_event, default=str),
                record.event_type, record.event_id, record.user_id,
                record.consumer_group, record.error_message, record.error_type,
                record.stack_trace, record.retry_count,
                record.first_failure_at, record.last_failure_at, record.dlq_entry_at,
                record.next_retry_at, record.is_retriable, record.resolved,
                record.resolution_notes,
            )
            logger.info(
                "dlq_record_saved", record_id=str(record.id),
                event_type=record.event_type, topic=record.dlq_topic,
            )

    async def get(self, record_id: UUID) -> DeadLetterRecord | None:
        """Get DLQ record by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM event_dead_letter WHERE id = $1", record_id,
            )
            return self._row_to_record(row) if row else None

    async def get_retriable(self, topic: str | None = None, limit: int = 100) -> list[DeadLetterRecord]:
        """Get retriable records ready for retry."""
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            if topic:
                rows = await conn.fetch(
                    """SELECT * FROM event_dead_letter
                       WHERE is_retriable AND NOT resolved
                       AND original_topic = $1
                       AND (next_retry_at IS NULL OR next_retry_at <= $2)
                       ORDER BY COALESCE(next_retry_at, dlq_entry_at) ASC
                       LIMIT $3""",
                    topic, now, limit,
                )
            else:
                rows = await conn.fetch(
                    """SELECT * FROM event_dead_letter
                       WHERE is_retriable AND NOT resolved
                       AND (next_retry_at IS NULL OR next_retry_at <= $1)
                       ORDER BY COALESCE(next_retry_at, dlq_entry_at) ASC
                       LIMIT $2""",
                    now, limit,
                )
            return [self._row_to_record(row) for row in rows]

    async def get_by_topic(self, dlq_topic: str, limit: int = 100) -> list[DeadLetterRecord]:
        """Get records for a specific DLQ topic."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM event_dead_letter
                   WHERE dlq_topic = $1
                   ORDER BY dlq_entry_at DESC
                   LIMIT $2""",
                dlq_topic, limit,
            )
            return [self._row_to_record(row) for row in rows]

    async def mark_resolved(self, record_id: UUID, notes: str | None = None) -> None:
        """Mark record as resolved."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE event_dead_letter
                   SET resolved = TRUE, resolution_notes = $2
                   WHERE id = $1""",
                record_id, notes,
            )
            logger.info("dlq_record_resolved", record_id=str(record_id))

    async def delete(self, record_id: UUID) -> None:
        """Delete DLQ record."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM event_dead_letter WHERE id = $1", record_id,
            )

    async def count_by_topic(self) -> dict[str, int]:
        """Count records by DLQ topic."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT dlq_topic, COUNT(*) as cnt
                   FROM event_dead_letter
                   GROUP BY dlq_topic""",
            )
            return {row["dlq_topic"]: row["cnt"] for row in rows}

    async def count_unresolved(self) -> int:
        """Count unresolved records."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM event_dead_letter WHERE NOT resolved",
            )
            return row["cnt"] if row else 0

    @staticmethod
    def _row_to_record(row: Any) -> DeadLetterRecord:
        """Convert database row to DeadLetterRecord."""
        event = row["original_event"]
        if isinstance(event, str):
            event = json.loads(event)
        return DeadLetterRecord(
            id=row["id"], original_topic=row["original_topic"],
            dlq_topic=row["dlq_topic"], original_event=event,
            event_type=row["event_type"], event_id=row["event_id"],
            user_id=row["user_id"], consumer_group=row["consumer_group"],
            error_message=row["error_message"], error_type=row["error_type"],
            stack_trace=row["stack_trace"], retry_count=row["retry_count"],
            first_failure_at=row["first_failure_at"],
            last_failure_at=row["last_failure_at"],
            dlq_entry_at=row["dlq_entry_at"],
            next_retry_at=row["next_retry_at"],
            is_retriable=row["is_retriable"], resolved=row["resolved"],
            resolution_notes=row["resolution_notes"],
        )

"""
Solace-AI Persistent Event Stores - PostgreSQL-backed outbox, idempotency and DLQ stores.

Replaces in-memory stores with durable PostgreSQL persistence for production use.
Events survive service restarts, enabling reliable transactional outbox and DLQ patterns.

The tables come only from Alembic slot ``010_event_plane`` (D-01, MVP-266). Services connect
as ``solace_app``, which holds DML and no DDL (D-02), so no store creates or alters a table at
runtime: ``ensure_table()`` verifies the slot's columns with :func:`verify_event_tables` and
raises :class:`~solace_events.errors.EventStoreSchemaError` when they are absent.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from .dead_letter import DeadLetterRecord
from .errors import EventStoreSchemaError, OutboxOwnershipError
from .publisher import DEFAULT_CLAIM_STALE_SECONDS, OutboxRecord, OutboxStatus

logger = structlog.get_logger(__name__)

EVENT_TABLES: tuple[str, ...] = ("event_outbox", "processed_events", "event_dead_letter")

# The columns slot 010_event_plane creates, per table. A store refuses to start without them.
REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "event_outbox": frozenset({
        "id", "source_service", "event_id", "event_type", "event_payload", "aggregate_id", "topic",
        "partition_key", "created_at", "published_at", "status", "retry_count", "last_error",
        "claimed_at", "claim_token", "next_retry_at",
    }),
    # P2-8: consumer-side idempotency ledger keyed on (event_id, consumer_group).
    "processed_events": frozenset({"event_id", "consumer_group", "processed_at"}),
    "event_dead_letter": frozenset({
        "id", "consumer_group", "original_topic", "dlq_topic", "kafka_partition", "kafka_offset",
        "original_event", "raw_payload", "event_type", "event_id", "user_id", "error_message",
        "error_type", "stack_trace", "retry_count", "redrive_count", "first_failure_at",
        "last_failure_at", "dlq_entry_at", "next_retry_at", "is_retriable", "resolved",
        "resolved_at", "resolution_notes",
    }),
}

# One catalog read, no DDL. information_schema.columns lists only the columns the connected
# role holds a privilege on, so a table solace_app cannot use is reported as missing too.
_COLUMNS_SQL = (
    "SELECT table_name, column_name FROM information_schema.columns "
    "WHERE table_schema = current_schema() AND table_name = ANY($1::text[])"
)

# OutboxRecord.source_service is stored in event_outbox.source_service VARCHAR(64) NOT NULL.
_SOURCE_SERVICE_MAX_LENGTH = 64


async def verify_event_tables(conn: Any, tables: Iterable[str] = EVENT_TABLES) -> None:
    """Check that ``tables`` exist in the current schema with every column slot 010 creates.

    Runs a single ``information_schema.columns`` query and never DDL. Raises
    ``EventStoreSchemaError`` mapping each incomplete table to its missing columns (an absent
    table lists all of them), and ``ValueError`` for a table that is not an event table.
    """
    wanted = list(dict.fromkeys(tables))
    unknown = [table for table in wanted if table not in REQUIRED_COLUMNS]
    if unknown:
        raise ValueError(f"verify_event_tables: not an event table: {', '.join(unknown)}")
    rows = await conn.fetch(_COLUMNS_SQL, wanted)
    found: dict[str, set[str]] = {}
    for row in rows:
        found.setdefault(row["table_name"], set()).add(row["column_name"])
    missing = {
        table: sorted(REQUIRED_COLUMNS[table] - found.get(table, set())) for table in wanted
    }
    missing = {table: columns for table, columns in missing.items() if columns}
    if missing:
        logger.error("event_store_schema_missing", tables=sorted(missing))
        raise EventStoreSchemaError(missing)


def _affected_rows(command_tag: Any) -> int:
    """Rows affected from an asyncpg command tag like ``"UPDATE 1"`` / ``"INSERT 0 1"``."""
    try:
        return int(str(command_tag).split()[-1])
    except (ValueError, IndexError, AttributeError):
        return 0


def _checked_source_service(source_service: Any) -> str:
    if not isinstance(source_service, str):
        raise TypeError("source_service must be the owning service's name (a str)")
    if (not source_service or source_service != source_service.strip()
            or len(source_service) > _SOURCE_SERVICE_MAX_LENGTH):
        raise ValueError(
            "source_service must be a non-blank service name without surrounding whitespace, "
            f"at most {_SOURCE_SERVICE_MAX_LENGTH} characters"
        )
    return source_service


class PostgresOutboxStore:
    """PostgreSQL-backed outbox store for transactional event publishing.

    Implements the OutboxStore protocol from publisher.py. Requires an asyncpg
    connection pool and the name of the service that owns it (MVP-133): ``save``
    writes that name into ``event_outbox.source_service`` and ``get_pending`` claims
    only that service's rows, so one service's poller never publishes another's.
    """

    def __init__(self, pool: Any, *, source_service: str) -> None:
        """Initialize with an asyncpg connection pool.

        Args:
            pool: asyncpg.Pool instance
            source_service: the owning service's name (``event_outbox.source_service``)
        """
        self._pool = pool
        self._source_service = _checked_source_service(source_service)

    async def ensure_table(self) -> None:
        """Verify slot 010's ``event_outbox`` (no DDL); raise EventStoreSchemaError if incomplete."""
        async with self._pool.acquire() as conn:
            await verify_event_tables(conn, ("event_outbox",))
        logger.info("outbox_table_verified", source_service=self._source_service)

    async def save(self, record: OutboxRecord) -> None:
        """Save outbox record to PostgreSQL, always as this store's service (MVP-133).

        ``source_service`` is what scopes every later claim, so the write side enforces the
        same ownership: a record without one is stamped with this store's service, and a
        record owned by another service, or by a blank or whitespace-padded owner, raises
        ``OutboxOwnershipError`` and nothing is written. This is the store's only insert path.
        """
        if record.source_service is not None and record.source_service != self._source_service:
            logger.error(
                "outbox_record_foreign_owner_refused",
                store_service=self._source_service,
                record_service=record.source_service,
            )
            raise OutboxOwnershipError(
                store_service=self._source_service, record_service=record.source_service
            )
        record.source_service = self._source_service
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO event_outbox
                   (id, source_service, event_id, event_type, event_payload, aggregate_id,
                    topic, partition_key, created_at, published_at, status,
                    retry_count, last_error, next_retry_at)
                   VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                   ON CONFLICT (id) DO NOTHING""",
                record.id, self._source_service, record.event_id, record.event_type,
                json.dumps(record.event_payload, default=str),
                record.aggregate_id, record.topic, record.partition_key,
                record.created_at, record.published_at, record.status.value,
                record.retry_count, record.last_error, record.next_retry_at,
            )
            logger.debug("outbox_record_saved", record_id=str(record.id))

    async def get_pending(
        self, limit: int = 100, stale_after_seconds: float = DEFAULT_CLAIM_STALE_SECONDS
    ) -> list[OutboxRecord]:
        """Atomically CLAIM pending (and reclaim stale) records for publishing.

        REV-39: a bare ``SELECT ... FOR UPDATE SKIP LOCKED`` released its row locks
        the instant the connection returned to the pool (autocommit), so two
        pollers could claim the same rows and double-publish. Here the claim is a
        single atomic ``UPDATE ... WHERE id IN (SELECT ... FOR UPDATE SKIP LOCKED)
        RETURNING`` that flips PENDING -> PUBLISHING and stamps ``claimed_at`` in
        one committed statement — two pollers do not claim the same row in the
        same instant.

        A PENDING row whose ``next_retry_at`` is in the future (backoff after a
        failed send, P1-6) is skipped until its window elapses. Records stuck in
        PUBLISHING longer than ``stale_after_seconds`` (a poller that crashed after
        claiming but before publishing) are reclaimed here so a crash never
        permanently strands an event.

        NOTE: delivery is AT-LEAST-ONCE, NOT exactly-once. The stale-reclaim can
        re-publish a record whose owner sent it but died before ``mark_published``
        (P2-8), so downstream consumers MUST dedupe on ``event.metadata.event_id``
        (see ``PostgresIdempotencyStore``). ``mark_published`` is claim-conditional
        so a poller whose claim was stolen does not finalize a row it no longer owns.

        MVP-133: the claim is scoped to this store's ``source_service`` ($3), which
        applies to both branches (a due PENDING row and a stale PUBLISHING reclaim), so
        a service never claims, backs off or reclaims another service's rows. The
        ``status IN`` term matches the partial index ``ix_event_outbox_claimable``.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """UPDATE event_outbox
                   SET status = 'PUBLISHING', claimed_at = NOW()
                   WHERE id IN (
                       SELECT id FROM event_outbox
                       WHERE source_service = $3
                         AND status IN ('PENDING', 'PUBLISHING')
                         AND ((status = 'PENDING'
                               AND (next_retry_at IS NULL OR next_retry_at <= NOW()))
                              OR (status = 'PUBLISHING'
                                  AND claimed_at < NOW() - make_interval(secs => $2)))
                       ORDER BY created_at ASC
                       LIMIT $1
                       FOR UPDATE SKIP LOCKED
                   )
                   RETURNING *""",
                limit, float(stale_after_seconds), self._source_service,
            )
            return [self._row_to_record(row) for row in rows]

    async def mark_published(self, record_id: UUID) -> bool:
        """Finalize a record only while THIS poller still owns it (P2-8).

        Conditional on the row still being PUBLISHING, so a poller whose claim was
        stolen by the stale-reclaim sweep does not mark a row another poller now
        owns. Returns True if it marked the row PUBLISHED, False otherwise.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE event_outbox
                   SET status = 'PUBLISHED', published_at = $2, claimed_at = NULL
                   WHERE id = $1 AND status = 'PUBLISHING'""",
                record_id, datetime.now(timezone.utc),
            )
            marked = _affected_rows(result) > 0
            if not marked:
                logger.warning(
                    "outbox_mark_published_no_owned_row", record_id=str(record_id)
                )
            return marked

    async def reset_to_pending(
        self, record_id: UUID, next_retry_at: datetime | None = None
    ) -> None:
        """Release a claimed record back to PENDING, optionally after a backoff."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE event_outbox
                   SET status = 'PENDING', claimed_at = NULL, next_retry_at = $2
                   WHERE id = $1""",
                record_id, next_retry_at,
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
            id=row["id"], source_service=row.get("source_service"),
            event_id=row["event_id"], event_type=row["event_type"],
            event_payload=payload, aggregate_id=row["aggregate_id"],
            topic=row["topic"], partition_key=row["partition_key"],
            created_at=row["created_at"], published_at=row["published_at"],
            status=OutboxStatus(row["status"]), retry_count=row["retry_count"],
            last_error=row["last_error"],
            # asyncpg.Record.get() returns None if the column is absent (a row
            # from before the claimed_at migration); `in` on a Record checks
            # values, not column names, so .get is the correct guard here.
            claimed_at=row.get("claimed_at"),
            next_retry_at=row.get("next_retry_at"),
        )


class PostgresIdempotencyStore:
    """PostgreSQL-backed processed-event ledger for consumer idempotency (P2-8).

    Implements the ``IdempotencyStore`` protocol from publisher.py. Because the
    outbox is at-least-once, a consumer wraps its side effects behind
    ``mark_processed`` so a re-published (duplicate) event cannot double-fire a
    handler (e.g. a duplicate crisis escalation). The check-and-set is a single
    atomic ``INSERT ... ON CONFLICT DO NOTHING RETURNING`` — exactly one caller
    per (event_id, consumer_group) gets True.
    """

    def __init__(self, pool: Any) -> None:
        """Initialize with an asyncpg connection pool."""
        self._pool = pool

    async def ensure_table(self) -> None:
        """Verify slot 010's ``processed_events`` (no DDL); raise EventStoreSchemaError if incomplete."""
        async with self._pool.acquire() as conn:
            await verify_event_tables(conn, ("processed_events",))
        logger.info("processed_events_table_verified")

    async def mark_processed(self, event_id: UUID, consumer_group: str) -> bool:
        """Atomically record (event_id, consumer_group). True if first-seen.

        Returns True exactly once (this delivery should be processed); a duplicate
        delivery of the same event to the same consumer group returns False and
        should be skipped.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO processed_events (event_id, consumer_group)
                   VALUES ($1, $2)
                   ON CONFLICT (event_id, consumer_group) DO NOTHING
                   RETURNING event_id""",
                event_id, consumer_group,
            )
            return row is not None

    async def was_processed(self, event_id: UUID, consumer_group: str) -> bool:
        """Whether this (event_id, consumer_group) was already processed."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT 1 FROM processed_events
                   WHERE event_id = $1 AND consumer_group = $2""",
                event_id, consumer_group,
            )
            return row is not None


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
        """Verify slot 010's ``event_dead_letter`` (no DDL); raise EventStoreSchemaError if incomplete."""
        async with self._pool.acquire() as conn:
            await verify_event_tables(conn, ("event_dead_letter",))
        logger.info("dlq_table_verified")

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

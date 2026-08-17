"""REV-39: transactional-outbox atomic-claim + stale-reclaim regression tests.

The old poller ran ``SELECT ... FOR UPDATE SKIP LOCKED`` in autocommit, so the row
locks released the instant the connection returned to the pool — two replicas'
pollers could claim the SAME pending rows and double-publish. The fix atomically
CLAIMS records (PENDING -> PUBLISHING, stamping ``claimed_at``) so concurrent
pollers get disjoint sets, reclaims records left stuck in PUBLISHING by a crashed
poller, and releases a failed-send claim back to PENDING for a prompt retry.

These exercise the in-memory store (which mirrors the Postgres claim contract) and
assert the Postgres store issues the atomic ``UPDATE ... RETURNING`` claim SQL.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest

from solace_events.publisher import (
    DEFAULT_CLAIM_STALE_SECONDS,
    EventPublisher,
    InMemoryOutboxStore,
    MockKafkaProducerAdapter,
    OutboxRecord,
    OutboxStatus,
)
from solace_events.schemas import SessionStartedEvent


def _record() -> OutboxRecord:
    return OutboxRecord(
        event_id=uuid4(),
        event_type="test.event",
        event_payload={"k": "v"},
        aggregate_id=uuid4(),
        topic="test.topic",
        partition_key="key",
    )


# ---------------------------------------------------------------------------
# In-memory store: claim / no-double-claim / stale reclaim / reset
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_pending_claims_and_prevents_double_claim() -> None:
    """A claimed record flips to PUBLISHING and is NOT handed to a second poller."""
    store = InMemoryOutboxStore()
    r1, r2 = _record(), _record()
    await store.save(r1)
    await store.save(r2)

    first = await store.get_pending()
    assert {r.id for r in first} == {r1.id, r2.id}
    assert all(r.status == OutboxStatus.PUBLISHING for r in first)
    assert all(r.claimed_at is not None for r in first)

    # A concurrent poller sees nothing to claim (both are freshly PUBLISHING).
    second = await store.get_pending()
    assert second == []


@pytest.mark.asyncio
async def test_stale_publishing_record_is_reclaimed_but_fresh_is_not() -> None:
    """A PUBLISHING record older than the stale window is reclaimed; a fresh one is not."""
    store = InMemoryOutboxStore()
    stale, fresh = _record(), _record()
    await store.save(stale)
    await store.save(fresh)

    # Claim both, then backdate only `stale`'s claim well beyond the window.
    await store.get_pending()
    store._records[stale.id].claimed_at = datetime.now(timezone.utc) - timedelta(seconds=999)

    reclaimed = await store.get_pending(stale_after_seconds=300)
    assert [r.id for r in reclaimed] == [stale.id]
    assert store._records[fresh.id].status == OutboxStatus.PUBLISHING


@pytest.mark.asyncio
async def test_reset_to_pending_makes_record_claimable_again() -> None:
    store = InMemoryOutboxStore()
    r = _record()
    await store.save(r)
    await store.get_pending()  # claim -> PUBLISHING
    assert store._records[r.id].status == OutboxStatus.PUBLISHING

    await store.reset_to_pending(r.id)
    assert store._records[r.id].status == OutboxStatus.PENDING
    assert store._records[r.id].claimed_at is None

    again = await store.get_pending()
    assert [x.id for x in again] == [r.id]


# ---------------------------------------------------------------------------
# flush_outbox: a failed send releases the claim (or fails after max retries)
# ---------------------------------------------------------------------------
class _FailingProducer(MockKafkaProducerAdapter):
    async def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        raise RuntimeError("kafka unavailable")


@pytest.mark.asyncio
async def test_flush_failure_releases_claim_then_next_flush_publishes() -> None:
    """A transient send failure must NOT leave the record stuck in PUBLISHING —
    it returns to PENDING and a later flush (working producer) publishes it once."""
    outbox = InMemoryOutboxStore()
    failing = _FailingProducer()
    pub = EventPublisher(failing, outbox, use_outbox=True, max_retries=3)
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))

    published = await pub.flush_outbox()
    assert published == 0
    # Exactly one record, released back to PENDING (not stranded in PUBLISHING).
    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.PENDING
    assert rec.retry_count == 1
    assert rec.claimed_at is None

    # Swap in a working producer; the released record publishes exactly once.
    working = MockKafkaProducerAdapter()
    await working.start()
    pub._producer = working
    published2 = await pub.flush_outbox()
    assert published2 == 1
    assert list(outbox._records.values())[0].status == OutboxStatus.PUBLISHED
    assert len(working.get_messages()) == 1


@pytest.mark.asyncio
async def test_flush_failure_marks_failed_after_max_retries() -> None:
    outbox = InMemoryOutboxStore()
    pub = EventPublisher(_FailingProducer(), outbox, use_outbox=True, max_retries=2)
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))

    # First failure: released to PENDING (retry_count 1 < 2).
    await pub.flush_outbox()
    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.PENDING and rec.retry_count == 1

    # Second failure hits the retry ceiling: terminal FAILED, not re-claimable.
    await pub.flush_outbox()
    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.FAILED
    assert await outbox.get_pending() == []


# ---------------------------------------------------------------------------
# Postgres store issues the atomic claim SQL (fake pool, no live infra)
# ---------------------------------------------------------------------------
class _FakeConn:
    def __init__(self) -> None:
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> "_FakeConn":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.fetched.append((sql, args))
        return []


class _FakePool:
    def __init__(self) -> None:
        self.conn = _FakeConn()

    def acquire(self) -> _FakeConn:
        return self.conn


@pytest.mark.asyncio
async def test_postgres_get_pending_issues_atomic_claim_sql() -> None:
    from solace_events.postgres_stores import PostgresOutboxStore

    pool = _FakePool()
    store = PostgresOutboxStore(pool)
    await store.get_pending(limit=50, stale_after_seconds=120)

    sql, args = pool.conn.fetched[-1]
    normalized = " ".join(sql.split())
    # Atomic claim: single UPDATE flips PENDING -> PUBLISHING and RETURNS the rows.
    assert normalized.startswith("UPDATE event_outbox")
    assert "SET status = 'PUBLISHING'" in normalized
    assert "claimed_at = NOW()" in normalized
    assert "FOR UPDATE SKIP LOCKED" in normalized
    assert "RETURNING *" in normalized
    # Stale-reclaim branch present, and params are (limit, stale_seconds).
    assert "status = 'PUBLISHING'" in normalized
    assert args == (50, 120.0)

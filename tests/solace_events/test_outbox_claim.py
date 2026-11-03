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
    # retry_base_seconds=0 disables the exponential-backoff delay so this test
    # can assert the immediate release->republish state machine (backoff timing
    # is covered separately by test_transient_failure_schedules_future_backoff_retry).
    pub = EventPublisher(
        failing, outbox, use_outbox=True, max_retries=3, retry_base_seconds=0.0
    )
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
    pub = EventPublisher(
        _FailingProducer(), outbox, use_outbox=True, max_retries=2,
        retry_base_seconds=0.0,
    )
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
    store = PostgresOutboxStore(pool, source_service="svc-test")
    await store.get_pending(limit=50, stale_after_seconds=120)

    sql, args = pool.conn.fetched[-1]
    normalized = " ".join(sql.split())
    # Atomic claim: single UPDATE flips PENDING -> PUBLISHING and RETURNS the rows.
    assert normalized.startswith("UPDATE event_outbox")
    assert "SET status = 'PUBLISHING'" in normalized
    assert "claimed_at = NOW()" in normalized
    assert "FOR UPDATE SKIP LOCKED" in normalized
    assert "RETURNING *" in normalized
    # Stale-reclaim branch present, and params are (limit, stale_seconds, source_service).
    assert "status = 'PUBLISHING'" in normalized
    assert args == (50, 120.0, "svc-test")


# ===========================================================================
# P1-6: exponential backoff + DLQ recovery so a permanently-failed CRISIS event
#       is never lost to a terminal FAILED row with only a log line.
# ===========================================================================

from solace_events.dead_letter import DeadLetterStore
from solace_events.publisher import (
    DEFAULT_MAX_RETRIES,
    InMemoryIdempotencyStore,
)


@pytest.mark.asyncio
async def test_transient_failure_schedules_future_backoff_retry() -> None:
    """A transient send failure schedules the retry in the FUTURE (exponential
    backoff); the record is not immediately re-claimable, so a flapping producer
    cannot burn the whole retry budget inside one tight poll loop."""
    outbox = InMemoryOutboxStore()
    pub = EventPublisher(
        _FailingProducer(), outbox, use_outbox=True, max_retries=10,
        retry_base_seconds=30.0,
    )
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))

    await pub.flush_outbox()
    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.PENDING
    assert rec.retry_count == 1
    assert rec.next_retry_at is not None
    assert rec.next_retry_at > datetime.now(timezone.utc)  # scheduled ahead
    assert rec.claimed_at is None

    # An immediate poll must NOT reclaim it (backoff window not elapsed).
    assert await outbox.get_pending() == []

    # Once the backoff window elapses, it becomes claimable again.
    outbox._records[rec.id].next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    again = await outbox.get_pending()
    assert [r.id for r in again] == [rec.id]


@pytest.mark.asyncio
async def test_backoff_grows_exponentially_with_retry_count() -> None:
    """Successive failures push next_retry_at further out (exponential growth)."""
    outbox = InMemoryOutboxStore()
    pub = EventPublisher(
        _FailingProducer(), outbox, use_outbox=True, max_retries=10,
        retry_base_seconds=10.0, retry_max_seconds=10_000.0,
    )
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))
    (rec,) = list(outbox._records.values())

    delays: list[float] = []
    for _ in range(3):
        before = datetime.now(timezone.utc)
        await pub.flush_outbox()
        nxt = outbox._records[rec.id].next_retry_at
        assert nxt is not None
        delays.append((nxt - before).total_seconds())
        # Make it claimable again for the next iteration without waiting.
        outbox._records[rec.id].next_retry_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    # 10 * 2^0, 10 * 2^1, 10 * 2^2 -> strictly increasing.
    assert delays[0] < delays[1] < delays[2]
    assert delays[0] >= 9.0  # ~10s, allow scheduling slack


@pytest.mark.asyncio
async def test_permanent_failure_routes_event_to_dlq_not_silently_lost() -> None:
    """P1-6: exhausting retries must route the event into a durable, RETRIABLE DLQ
    record (so a crisis event can still be alerted / re-driven) rather than dying
    in a terminal FAILED outbox row with only a log line."""
    outbox = InMemoryOutboxStore()
    dlq = DeadLetterStore()
    pub = EventPublisher(
        _FailingProducer(), outbox, use_outbox=True, max_retries=2,
        retry_base_seconds=0.0, dlq_store=dlq,
    )
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))

    await pub.flush_outbox()  # retry 1 -> released to PENDING
    await pub.flush_outbox()  # retry 2 hits ceiling -> FAILED + routed to DLQ

    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.FAILED

    # The event is durably preserved and still RETRIABLE in the DLQ (not lost).
    retriable = await dlq.get_retriable()
    assert len(retriable) == 1
    dlq_rec = retriable[0]
    assert dlq_rec.event_id == rec.event_id
    assert dlq_rec.user_id == rec.aggregate_id
    assert dlq_rec.is_retriable is True
    assert dlq_rec.original_event == rec.event_payload
    assert await dlq.count_unresolved() == 1


@pytest.mark.asyncio
async def test_default_max_retries_is_raised_above_legacy_three() -> None:
    """A transient outage must not exhaust the retry budget after only 3 tries."""
    pub = EventPublisher(MockKafkaProducerAdapter())
    assert pub._max_retries == DEFAULT_MAX_RETRIES
    assert pub._max_retries >= 10


# ===========================================================================
# P2-8: mark_published is conditional on the caller still OWNING the claim, and a
#       documented idempotency SEAM lets consumers dedupe at-least-once delivery.
# ===========================================================================
@pytest.mark.asyncio
async def test_mark_published_noops_when_claim_was_stolen() -> None:
    """If a stale-reclaim released the row back to PENDING (another poller owns it
    now), the original poller's mark_published must NOT clobber it to PUBLISHED."""
    store = InMemoryOutboxStore()
    r = _record()
    await store.save(r)
    await store.get_pending()  # this poller claims -> PUBLISHING

    # A concurrent reclaim releases it back to PENDING (claim stolen).
    await store.reset_to_pending(r.id)

    marked = await store.mark_published(r.id)
    assert marked is False
    assert store._records[r.id].status == OutboxStatus.PENDING  # not clobbered


@pytest.mark.asyncio
async def test_mark_published_marks_and_reports_true_when_owned() -> None:
    store = InMemoryOutboxStore()
    r = _record()
    await store.save(r)
    await store.get_pending()  # claim -> PUBLISHING

    marked = await store.mark_published(r.id)
    assert marked is True
    assert store._records[r.id].status == OutboxStatus.PUBLISHED
    assert store._records[r.id].published_at is not None


@pytest.mark.asyncio
async def test_flush_does_not_double_count_when_claim_stolen_mid_send() -> None:
    """A poller whose claim was stolen while its send was in flight must not count
    the publish (the owning poller finalizes it); at-least-once, deduped downstream."""

    class _StealDuringSendProducer(MockKafkaProducerAdapter):
        def __init__(self, store: InMemoryOutboxStore) -> None:
            super().__init__()
            self._store = store

        async def send(self, topic, key, value):  # type: ignore[override]
            # Simulate another poller reclaiming this row mid-send by releasing it.
            for rec in list(self._store._records.values()):
                await self._store.reset_to_pending(rec.id)
            await super().send(topic, key, value)

    outbox = InMemoryOutboxStore()
    producer = _StealDuringSendProducer(outbox)
    pub = EventPublisher(producer, outbox, use_outbox=True)
    await pub.start()
    await pub.publish(SessionStartedEvent(user_id=uuid4(), session_number=1))

    published = await pub.flush_outbox()
    # Send happened, but the row is no longer owned -> not counted as published.
    assert published == 0
    (rec,) = list(outbox._records.values())
    assert rec.status == OutboxStatus.PENDING


@pytest.mark.asyncio
async def test_idempotency_store_processes_first_and_skips_duplicate() -> None:
    idem = InMemoryIdempotencyStore()
    event_id = uuid4()
    first = await idem.mark_processed(event_id, "crisis-consumer")
    dup = await idem.mark_processed(event_id, "crisis-consumer")
    assert first is True   # first delivery -> process it
    assert dup is False    # duplicate delivery -> skip (no double crisis fire)
    # Same event_id under a DIFFERENT consumer group is independent.
    other = await idem.mark_processed(event_id, "audit-consumer")
    assert other is True
    assert await idem.was_processed(event_id, "crisis-consumer") is True
    assert await idem.was_processed(uuid4(), "crisis-consumer") is False


# ---------------------------------------------------------------------------
# Postgres store: conditional mark_published SQL, backoff-aware get_pending SQL,
# and the processed-event idempotency INSERT ... ON CONFLICT DO NOTHING seam.
# ---------------------------------------------------------------------------
class _RecordingConn:
    def __init__(self, *, fetchrow_result: Any = None, execute_result: str = "UPDATE 1") -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._fetchrow_result = fetchrow_result
        self._execute_result = execute_result

    async def __aenter__(self) -> "_RecordingConn":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return self._fetchrow_result

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return self._execute_result


class _RecordingPool:
    def __init__(self, conn: _RecordingConn) -> None:
        self._conn = conn

    def acquire(self) -> _RecordingConn:
        return self._conn


@pytest.mark.asyncio
async def test_postgres_mark_published_is_conditional_on_publishing() -> None:
    from solace_events.postgres_stores import PostgresOutboxStore

    conn = _RecordingConn(execute_result="UPDATE 1")
    store = PostgresOutboxStore(_RecordingPool(conn), source_service="svc-test")
    result = await store.mark_published(uuid4())

    sql, _args = conn.calls[-1]
    normalized = " ".join(sql.split())
    assert "SET status = 'PUBLISHED'" in normalized
    # The claim-ownership guard: only finalize a row THIS poller still holds.
    assert "WHERE id = $1 AND status = 'PUBLISHING'" in normalized
    assert result is True


@pytest.mark.asyncio
async def test_postgres_mark_published_reports_false_when_no_row_owned() -> None:
    from solace_events.postgres_stores import PostgresOutboxStore

    conn = _RecordingConn(execute_result="UPDATE 0")  # 0 rows -> claim was stolen
    store = PostgresOutboxStore(_RecordingPool(conn), source_service="svc-test")
    result = await store.mark_published(uuid4())
    assert result is False


@pytest.mark.asyncio
async def test_postgres_get_pending_respects_next_retry_at_backoff() -> None:
    from solace_events.postgres_stores import PostgresOutboxStore

    conn = _RecordingConn()
    store = PostgresOutboxStore(_RecordingPool(conn), source_service="svc-test")
    await store.get_pending(limit=10, stale_after_seconds=60)

    sql, _args = conn.calls[-1]
    normalized = " ".join(sql.split())
    # A PENDING row is only claimable once its backoff window has elapsed.
    assert "next_retry_at IS NULL OR next_retry_at <= NOW()" in normalized


@pytest.mark.asyncio
async def test_postgres_idempotency_insert_on_conflict_do_nothing() -> None:
    from solace_events.postgres_stores import PostgresIdempotencyStore

    conn = _RecordingConn(fetchrow_result={"event_id": uuid4()})
    store = PostgresIdempotencyStore(_RecordingPool(conn))
    processed_first = await store.mark_processed(uuid4(), "crisis-consumer")

    sql, _args = conn.calls[-1]
    normalized = " ".join(sql.split())
    assert normalized.startswith("INSERT INTO processed_events")
    assert "ON CONFLICT (event_id, consumer_group) DO NOTHING" in normalized
    assert "RETURNING event_id" in normalized
    assert processed_first is True  # row returned -> first time -> process

    # A conflicting insert returns no row -> duplicate -> skip.
    conn2 = _RecordingConn(fetchrow_result=None)
    store2 = PostgresIdempotencyStore(_RecordingPool(conn2))
    processed_dup = await store2.mark_processed(uuid4(), "crisis-consumer")
    assert processed_dup is False

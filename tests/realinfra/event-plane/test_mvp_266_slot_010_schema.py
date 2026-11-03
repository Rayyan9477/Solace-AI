"""MVP-266 / MVP-133 (WS-06.1) on real PostgreSQL: slot 010 is the only source of the event tables, solace_app gets
DML and no DDL on them, the stores verify instead of creating, and each service claims only its own outbox rows.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest

from solace_events.postgres_stores import (
    PostgresDLQStore,
    PostgresIdempotencyStore,
    PostgresOutboxStore,
)
from solace_events.publisher import OutboxRecord

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

EXPECTED = {
    "event_outbox": {"id", "source_service", "event_id", "event_type", "event_payload", "aggregate_id", "topic",
                     "partition_key", "created_at", "published_at", "status", "retry_count", "last_error",
                     "claimed_at", "claim_token", "next_retry_at"},
    "processed_events": {"event_id", "consumer_group", "processed_at"},
    "event_dead_letter": {"id", "consumer_group", "original_topic", "dlq_topic", "kafka_partition", "kafka_offset",
                          "original_event", "raw_payload", "event_type", "event_id", "user_id", "error_message",
                          "error_type", "stack_trace", "retry_count", "redrive_count", "first_failure_at",
                          "last_failure_at", "dlq_entry_at", "next_retry_at", "is_retriable", "resolved",
                          "resolved_at", "resolution_notes"},
}


def _outbox_record() -> OutboxRecord:
    return OutboxRecord(event_id=uuid4(), event_type="test.event", event_payload={"k": "v"}, aggregate_id=uuid4(),
                        topic="solace.test", partition_key="k")


async def test_mvp_266_slot_010_creates_event_tables(pg_migrated):
    rows = await pg_migrated.superuser_fetch(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = ANY($1::text[])", list(EXPECTED))
    got: dict[str, set[str]] = {}
    for r in rows:
        got.setdefault(r["table_name"], set()).add(r["column_name"])
    assert got == EXPECTED


async def test_mvp_266_solace_app_has_dml_but_no_ddl(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        assert await conn.fetchval("SELECT current_user") == "solace_app"
        rid = uuid4()
        await conn.execute(
            "INSERT INTO event_outbox (id, source_service, event_id, event_type, event_payload, aggregate_id, topic,"
            " partition_key) VALUES ($1, 'svc-266', $2, 'test.event', $3, $4, 'solace.test', 'k')",
            rid, uuid4(), {"k": "v"}, uuid4())
        assert await conn.execute("UPDATE event_outbox SET status = 'PUBLISHED' WHERE id = $1", rid) == "UPDATE 1"
        assert await conn.execute("DELETE FROM event_outbox WHERE id = $1", rid) == "DELETE 1"
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("ALTER TABLE event_outbox ADD COLUMN x INT")


async def test_mvp_266_stores_verify_as_solace_app(pg_migrated):
    for store in (PostgresOutboxStore(pg_migrated.app_pool, source_service="svc-266"),
                  PostgresDLQStore(pg_migrated.app_pool), PostgresIdempotencyStore(pg_migrated.app_pool)):
        await store.ensure_table()


async def test_mvp_266_store_verification_fails_loud_before_slot_010(pg_migrated):
    from solace_events.errors import EventStoreSchemaError

    db = await pg_migrated.create_database(f"ev266_{uuid4().hex[:8]}", migrate_to="009_platform_roles_rls")
    with pytest.raises(EventStoreSchemaError) as exc:
        await PostgresOutboxStore(db.app_pool, source_service="svc-266").ensure_table()
    assert "event_outbox" in exc.value.missing


async def test_mvp_133_each_service_claims_only_its_own_rows(pg_migrated):
    a = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-a-{uuid4().hex[:6]}")
    b = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-b-{uuid4().hex[:6]}")
    rec_a = [_outbox_record() for _ in range(3)]
    rec_b = [_outbox_record() for _ in range(3)]
    for r in rec_a: await a.save(r)
    for r in rec_b: await b.save(r)
    assert {r.id for r in await a.get_pending(limit=100)} == {r.id for r in rec_a}
    rows = await pg_migrated.superuser_fetch("SELECT status FROM event_outbox WHERE id = ANY($1::uuid[])", [r.id for r in rec_b])
    assert sorted(r["status"] for r in rows) == ["PENDING"] * 3


# --- WS-06.1 additions --------------------------------------------------------------------------------------------

# index name -> (pg_get_indexdef column list, partial?)
EXPECTED_INDEXES = {
    "event_outbox_pkey": ("(id)", False),
    "ix_event_outbox_claimable": ("(source_service, created_at)", True),
    "ix_event_outbox_published_at": ("(published_at)", True),
    "ix_event_outbox_aggregate": ("(aggregate_id)", False),
    "processed_events_pkey": ("(event_id, consumer_group)", False),
    "ix_processed_events_processed_at": ("(processed_at)", False),
    "event_dead_letter_pkey": ("(id)", False),
    "ix_event_dead_letter_redrive": ("(consumer_group, next_retry_at)", True),
    "ix_event_dead_letter_unresolved": ("(consumer_group)", True),
    "ix_event_dead_letter_user": ("(user_id)", False),
    "ix_event_dead_letter_resolved_at": ("(resolved_at)", True),
}


async def test_mvp_266_slot_010_indexes_owner_grants_and_no_rls(pg_migrated):
    tables = list(EXPECTED)
    idx = {r["indexname"]: r["indexdef"] for r in await pg_migrated.superuser_fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = ANY($1::text[])",
        tables)}
    assert set(idx) == set(EXPECTED_INDEXES)
    for name, (columns, partial) in EXPECTED_INDEXES.items():
        assert f"USING btree {columns}" in idx[name], idx[name]
        assert (" WHERE " in idx[name]) is partial, idx[name]
    rels = await pg_migrated.superuser_fetch(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, pg_get_userbyid(c.relowner) AS owner "
        "FROM pg_class c WHERE c.relnamespace = 'public'::regnamespace AND c.relname = ANY($1::text[])", tables)
    assert {r["relname"]: (r["relrowsecurity"], r["relforcerowsecurity"], r["owner"]) for r in rels} == {
        t: (False, False, "solace_migrator") for t in tables}
    privs = await pg_migrated.superuser_fetch(
        "SELECT t, p, has_table_privilege('solace_app', t, p) AS ok FROM unnest($1::text[]) AS t, "
        "unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER']) AS p", tables)
    granted = {(r["t"], r["p"]) for r in privs if r["ok"]}
    assert granted == {(t, p) for t in tables for p in ("SELECT", "INSERT", "UPDATE", "DELETE")}


async def test_mvp_266_outbox_rejects_unknown_status_and_missing_service(pg_migrated):
    sql = ("INSERT INTO event_outbox (id, source_service, event_id, event_type, event_payload, aggregate_id, topic,"
           " partition_key, status) VALUES ($1, $2, $3, 'test.event', $4, $5, 'solace.test', 'k', $6)")
    async with pg_migrated.app_pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(sql, uuid4(), "svc-266", uuid4(), {"k": "v"}, uuid4(), "SENT")
        with pytest.raises(asyncpg.NotNullViolationError):
            await conn.execute(sql, uuid4(), None, uuid4(), {"k": "v"}, uuid4(), "PENDING")


async def test_mvp_133_stale_reclaim_and_backoff_stay_per_service(pg_migrated):
    """Both claim branches (a due backoff retry and a stale PUBLISHING claim) are scoped to the claiming service."""
    a = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-a-{uuid4().hex[:6]}")
    b = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-b-{uuid4().hex[:6]}")
    stale, retry_due = _outbox_record(), _outbox_record()
    await b.save(stale)
    await b.save(retry_due)
    assert {r.id for r in await b.get_pending(limit=100)} == {stale.id, retry_due.id}
    # b's poller died holding `stale`; `retry_due` was released with a backoff window that has already passed.
    await pg_migrated.superuser_fetch(
        "UPDATE event_outbox SET claimed_at = now() - interval '1 hour' WHERE id = $1 RETURNING id", stale.id)
    await b.reset_to_pending(retry_due.id, next_retry_at=datetime.now(UTC) - timedelta(seconds=5))
    assert await a.get_pending(limit=100, stale_after_seconds=60) == []
    rows = await pg_migrated.superuser_fetch(
        "SELECT id, status FROM event_outbox WHERE id = ANY($1::uuid[])", [stale.id, retry_due.id])
    assert {r["id"]: r["status"] for r in rows} == {stale.id: "PUBLISHING", retry_due.id: "PENDING"}
    reclaimed = await b.get_pending(limit=100, stale_after_seconds=60)
    assert {r.id for r in reclaimed} == {stale.id, retry_due.id}
    assert {r.source_service for r in reclaimed} == {b._source_service}


# --- WS-06.1 fix round 1 (review finding WS-06.1-R1) ---------------------------------------------------------------


async def test_mvp_133_r1_save_writes_no_foreign_or_blank_owned_row(pg_migrated):
    """A store writes only its own service's rows: a record owned by another service (which that service's poller
    would then publish) or by a blank or whitespace owner (which no poller could ever claim) inserts nothing."""
    a = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-a-{uuid4().hex[:6]}")
    b = PostgresOutboxStore(pg_migrated.app_pool, source_service=f"svc-b-{uuid4().hex[:6]}")
    owners = [b._source_service, "", "   ", f" {a._source_service}"]
    refused_records = [_outbox_record() for _ in owners]
    raised: list[BaseException] = []
    for rec, owner in zip(refused_records, owners, strict=True):
        rec.source_service = owner
        try:
            await a.save(rec)
        except ValueError as exc:
            raised.append(exc)
    rows = await pg_migrated.superuser_fetch(
        "SELECT source_service FROM event_outbox WHERE id = ANY($1::uuid[])", [r.id for r in refused_records])
    assert sorted(r["source_service"] for r in rows) == []  # nothing was written for any refused owner
    assert await b.get_pending(limit=100) == []  # so b's poller has nothing of a's to publish
    from solace_events.errors import OutboxOwnershipError

    assert [type(e) for e in raised] == [OutboxOwnershipError] * len(owners)
    assert [e.record_service for e in raised] == owners
    own = _outbox_record()
    own.source_service = a._source_service
    await a.save(own)
    (row,) = await pg_migrated.superuser_fetch("SELECT source_service FROM event_outbox WHERE id = $1", own.id)
    assert row["source_service"] == a._source_service
    assert [r.id for r in await a.get_pending(limit=100)] == [own.id]

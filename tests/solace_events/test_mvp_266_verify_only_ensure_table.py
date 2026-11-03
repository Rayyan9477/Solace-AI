"""MVP-266 / MVP-133 (WS-06.1): event tables come only from slot 010; stores verify, never create, and each
service's outbox store writes and claims only its own rows.

D-01: Alembic is the only schema of record, so ``ensure_table()`` runs no DDL at runtime (services connect as
``solace_app``, which holds DML only, D-02). It checks ``information_schema.columns`` and raises
``EventStoreSchemaError`` naming the missing tables and columns. IF-05: ``event_outbox.source_service`` scopes
every claim, so one service's poller never publishes another service's rows.
"""
from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from solace_events.postgres_stores import (
    PostgresDLQStore,
    PostgresIdempotencyStore,
    PostgresOutboxStore,
)
from solace_events.publisher import InMemoryOutboxStore, OutboxRecord, create_publisher

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


class _SchemaConn:
    """Answers information_schema.columns queries from a {table: columns} dict and records every SQL text."""

    def __init__(self, schema: dict[str, set[str]]) -> None:
        self.schema = schema
        self.sql: list[str] = []
        self.args: list[tuple[Any, ...]] = []

    async def __aenter__(self) -> _SchemaConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, str]]:
        self.sql.append(sql)
        self.args.append(args)
        if "information_schema.columns" not in sql:
            return []
        tables = args[0] if args else list(self.schema)
        return [{"table_name": t, "column_name": c} for t in tables for c in sorted(self.schema.get(t, ()))]

    async def execute(self, sql: str, *args: Any) -> str:
        self.sql.append(sql)
        self.args.append(args)
        return "OK"

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self.sql.append(sql)
        self.args.append(args)
        return None


class _SchemaPool:
    def __init__(self, conn: _SchemaConn) -> None:
        self.conn = conn

    def acquire(self) -> _SchemaConn:
        return self.conn


class _RecordingConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> _RecordingConn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return None

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return "INSERT 0 1"


class _RecordingPool:
    def __init__(self, conn: _RecordingConn) -> None:
        self.conn = conn

    def acquire(self) -> _RecordingConn:
        return self.conn


def _record() -> OutboxRecord:
    from uuid import uuid4

    return OutboxRecord(event_id=uuid4(), event_type="test.event", event_payload={"k": "v"}, aggregate_id=uuid4(),
                        topic="test.topic", partition_key="key")


async def test_mvp_266_dlq_and_ledger_ensure_table_execute_no_ddl():
    conn = _SchemaConn(EXPECTED)
    await PostgresDLQStore(_SchemaPool(conn)).ensure_table()
    await PostgresIdempotencyStore(_SchemaPool(conn)).ensure_table()
    assert not [s for s in conn.sql if "CREATE" in s.upper() or "ALTER" in s.upper()]


async def test_mvp_266_missing_columns_raise_event_store_schema_error():
    from solace_events.errors import EventStoreSchemaError
    partial = {**EXPECTED, "event_dead_letter": EXPECTED["event_dead_letter"] - {"raw_payload", "redrive_count"}}
    with pytest.raises(EventStoreSchemaError) as exc:
        await PostgresDLQStore(_SchemaPool(_SchemaConn(partial))).ensure_table()
    assert sorted(exc.value.missing["event_dead_letter"]) == ["raw_payload", "redrive_count"]


async def test_mvp_133_postgres_claim_filters_on_source_service():
    conn = _RecordingConn()
    await PostgresOutboxStore(_RecordingPool(conn), source_service="svc-a").get_pending(limit=10, stale_after_seconds=60)
    sql, args = conn.calls[-1]
    assert "source_service = $3" in " ".join(sql.split())
    assert args == (10, 60.0, "svc-a")


def test_mvp_133_create_publisher_requires_service_name_with_pool():
    with pytest.raises(ValueError):
        create_publisher(use_mock=True, postgres_pool=_RecordingPool(_RecordingConn()))
    pub = create_publisher(use_mock=True, postgres_pool=_RecordingPool(_RecordingConn()), source_service="svc-a")
    assert pub._outbox_store._source_service == "svc-a"


async def test_mvp_133_in_memory_store_claims_only_its_service():
    store = InMemoryOutboxStore(source_service="svc-a")
    mine, other = _record(), _record()
    other.source_service = "svc-b"
    await store.save(mine); await store.save(other)
    assert [r.id for r in await store.get_pending()] == [mine.id]


# --- WS-06.1 additions --------------------------------------------------------------------------------------------


def test_mvp_266_required_columns_pin_the_slot_010_schema():
    from solace_events import postgres_stores

    assert {t: set(cols) for t, cols in postgres_stores.REQUIRED_COLUMNS.items()} == EXPECTED
    assert all(isinstance(cols, frozenset) for cols in postgres_stores.REQUIRED_COLUMNS.values())
    for name in ("OUTBOX_TABLE_DDL", "PROCESSED_EVENTS_TABLE_DDL", "DLQ_TABLE_DDL"):
        assert not hasattr(postgres_stores, name), name


async def test_mvp_266_outbox_ensure_table_runs_one_schema_query_and_no_ddl():
    conn = _SchemaConn(EXPECTED)
    await PostgresOutboxStore(_SchemaPool(conn), source_service="svc-a").ensure_table()
    (sql,) = conn.sql
    normalized = " ".join(sql.split())
    assert normalized == ("SELECT table_name, column_name FROM information_schema.columns "
                          "WHERE table_schema = current_schema() AND table_name = ANY($1::text[])")
    assert conn.args == [(["event_outbox"],)]


async def test_mvp_266_missing_table_lists_all_of_its_columns_and_only_names():
    from solace_events.errors import EventStoreSchemaError
    from solace_events.postgres_stores import verify_event_tables

    schema = {"processed_events": EXPECTED["processed_events"]}
    with pytest.raises(EventStoreSchemaError) as exc:
        await verify_event_tables(_SchemaConn(schema))
    assert exc.value.missing == {"event_dead_letter": sorted(EXPECTED["event_dead_letter"]),
                                 "event_outbox": sorted(EXPECTED["event_outbox"])}
    assert isinstance(exc.value, RuntimeError)
    message = str(exc.value)
    assert "event_outbox" in message and "event_dead_letter" in message and "claim_token" in message
    assert "processed_events" not in message
    await verify_event_tables(_SchemaConn(EXPECTED))  # a complete schema passes
    await verify_event_tables(_SchemaConn(schema), tables=("processed_events",))


async def test_mvp_266_verify_rejects_a_table_it_does_not_know():
    from solace_events.postgres_stores import verify_event_tables

    with pytest.raises(ValueError):
        await verify_event_tables(_SchemaConn(EXPECTED), tables=("users",))


async def test_mvp_266_outbox_store_reports_its_own_missing_table():
    from solace_events.errors import EventStoreSchemaError

    schema = {t: cols for t, cols in EXPECTED.items() if t != "event_outbox"}
    with pytest.raises(EventStoreSchemaError) as exc:
        await PostgresOutboxStore(_SchemaPool(_SchemaConn(schema)), source_service="svc-a").ensure_table()
    assert list(exc.value.missing) == ["event_outbox"]


def test_mvp_266_event_store_schema_error_survives_pickling():
    import pickle

    from solace_events.errors import EventStoreSchemaError

    err = EventStoreSchemaError({"event_outbox": ["claim_token", "source_service"]})
    again = pickle.loads(pickle.dumps(err))  # noqa: S301 - round-trips the test's own object
    assert again.missing == err.missing and str(again) == str(err)


async def test_mvp_133_postgres_save_writes_its_service_name():
    conn = _RecordingConn()
    rec = _record()
    await PostgresOutboxStore(_RecordingPool(conn), source_service="svc-a").save(rec)
    sql, args = conn.calls[-1]
    normalized = " ".join(sql.split())
    assert normalized.startswith("INSERT INTO event_outbox (id, source_service,")
    assert args[:2] == (rec.id, "svc-a")
    assert rec.source_service == "svc-a"


@pytest.mark.parametrize("bad", ["", "   ", " svc-a", "svc-a ", "s" * 65, None, 7])
def test_mvp_133_postgres_store_requires_a_usable_service_name(bad):
    with pytest.raises((TypeError, ValueError), match="source_service must be"):
        PostgresOutboxStore(_RecordingPool(_RecordingConn()), source_service=bad)


def test_mvp_133_postgres_store_has_no_default_service_name():
    with pytest.raises(TypeError):
        PostgresOutboxStore(_RecordingPool(_RecordingConn()))  # type: ignore[call-arg]


def test_mvp_133_create_publisher_without_pool_scopes_the_in_memory_store():
    assert create_publisher(use_mock=True)._outbox_store._source_service == "local"
    assert create_publisher(use_mock=True, source_service="svc-a")._outbox_store._source_service == "svc-a"


async def test_mvp_133_in_memory_store_stamps_unset_records_only():
    store = InMemoryOutboxStore(source_service="svc-a")
    mine, other = _record(), _record()
    other.source_service = "svc-b"
    await store.save(mine)
    await store.save(other)
    assert (mine.source_service, other.source_service) == ("svc-a", "svc-b")
    default = InMemoryOutboxStore()
    rec = _record()
    await default.save(rec)
    assert rec.source_service == "local" and [r.id for r in await default.get_pending()] == [rec.id]


# --- WS-06.1 fix round 1 (review finding WS-06.1-R1) ---------------------------------------------------------------
# The write side enforces the same per-service ownership as the claim side: PostgresOutboxStore.save, the store's
# only insert path, writes its own source_service and refuses a record owned by another service or by a blank or
# whitespace owner (which no store could ever claim), writing nothing.

FOREIGN_OR_BLANK_OWNERS = ["svc-b", "", "   ", " svc-a", "svc-a ", "\tsvc-a"]


@pytest.mark.parametrize("owner", FOREIGN_OR_BLANK_OWNERS)
async def test_mvp_133_r1_postgres_save_refuses_a_foreign_or_blank_owner_and_writes_nothing(owner):
    conn = _RecordingConn()
    rec = _record()
    rec.source_service = owner
    with pytest.raises(ValueError) as exc:
        await PostgresOutboxStore(_RecordingPool(conn), source_service="svc-a").save(rec)
    assert conn.calls == []
    assert rec.source_service == owner  # a refused record is left as it was
    from solace_events.errors import OutboxOwnershipError

    assert isinstance(exc.value, OutboxOwnershipError)
    assert (exc.value.store_service, exc.value.record_service) == ("svc-a", owner)
    message = str(exc.value)
    assert repr("svc-a") in message and repr(owner) in message
    for leaked in (str(rec.id), str(rec.event_id), str(rec.aggregate_id), rec.event_type, rec.topic):
        assert leaked not in message


async def test_mvp_133_r1_postgres_save_always_writes_its_own_service():
    conn = _RecordingConn()
    unset, own = _record(), _record()
    own.source_service = "svc-a"
    store = PostgresOutboxStore(_RecordingPool(conn), source_service="svc-a")
    await store.save(unset)
    await store.save(own)
    assert [args[:2] for _sql, args in conn.calls] == [(unset.id, "svc-a"), (own.id, "svc-a")]
    assert (unset.source_service, own.source_service) == ("svc-a", "svc-a")


def test_mvp_133_r1_outbox_ownership_error_pickles_with_both_names():
    import pickle

    from solace_events.errors import OutboxOwnershipError

    err = OutboxOwnershipError(store_service="svc-a", record_service="svc-b")
    again = pickle.loads(pickle.dumps(err))  # noqa: S301 - round-trips the test's own object
    assert (again.store_service, again.record_service, str(again)) == ("svc-a", "svc-b", str(err))


# --- WS-06.1 fix round 2 (review finding WS061F1-CLO-1) ------------------------------------------------------------
# The single-write-path guard, robust to spelling. Every module of the event plane (src/solace_events) and of the
# services (services/*/src) is scanned, so a second way to write event_outbox turns it red until that path goes
# through PostgresOutboxStore.save's ownership check and has its own test. Covered spellings: any case, whitespace
# or newlines, a quoted or unquoted schema qualifier, implicit string concatenation, f-strings, COPY and asyncpg's
# copy helpers, and any other module. Out of static reach: a table name assembled at run time from pieces, or one
# imported from solace_events into a service.

_REPO = Path(__file__).resolve().parents[2]
_SAVE_SITE = ("src/solace_events/postgres_stores.py", "PostgresOutboxStore.save")
_QUALIFIER = r'(?:"?\w+"?\s*\.\s*)?'  # optional schema qualifier: public.  "public".
_OUTBOX_WRITE_RE = re.compile(
    r"\binsert\s+into\s+" + _QUALIFIER + r'"?event_outbox\b'
    r"|\bcopy\s+" + _QUALIFIER + r'"?event_outbox\b'
    r"""|\bcopy_(?:records_)?to_table\s*\(\s*(?:table_name\s*=\s*)?['"]event_outbox\b""",
    re.IGNORECASE,
)
_INSERT_OR_COPY_RE = re.compile(r"\binsert\s+into\b|\bcopy\b", re.IGNORECASE)
_OUTBOX_NAME_RE = re.compile(r"\bevent_outbox\b", re.IGNORECASE)
# An INSERT whose target table is filled in at run time: the literal stops at the table position (an f-string
# field or a concatenation follows), or a str.format / %-style placeholder stands there.
_DANGLING_INSERT_RE = re.compile(r"\binsert\s+into(?:\s+" + _QUALIFIER + r'"?)?\s*$', re.IGNORECASE)
_PLACEHOLDER_INSERT_RE = re.compile(r"\binsert\s+into\s+" + _QUALIFIER + r'"?(?:\{[^}]*\}|%s|%\(\w+\)s)',
                                    re.IGNORECASE)
_COPY_HELPERS = frozenset({"copy_records_to_table", "copy_to_table"})


def _scanned_modules(root: Path) -> list[tuple[str, ast.Module, str]]:
    """(repo-relative path, AST, source) of every module under src/solace_events and services/*/src."""
    paths = list((root / "src" / "solace_events").rglob("*.py"))
    for service_src in (root / "services").glob("*/src"):
        paths += service_src.rglob("*.py")
    modules = []
    for path in sorted(p for p in paths if "__pycache__" not in p.parts):
        text = path.read_text(encoding="utf-8")
        modules.append((path.relative_to(root).as_posix(), ast.parse(text), text))
    return modules


def _qualname_at(tree: ast.Module, lineno: int) -> str:
    """Dotted name of the innermost class or function enclosing ``lineno``; ``<module>`` when there is none."""
    found = "<module>"

    def visit(node: ast.AST, prefix: str) -> None:
        nonlocal found
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if child.lineno <= lineno <= (child.end_lineno or child.lineno):
                    found = f"{prefix}{child.name}"
                    visit(child, f"{found}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def _string_literals(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """(node, text) for each f-string (its literal parts joined) and each other str constant, docstrings excluded."""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
            yield node, "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            yield node, node.value


def _outbox_write_sites(root: Path) -> dict[str, list[tuple[str, str]]]:
    """Where event_outbox is written: regex matches over each module's source and over its string literals (which
    also joins implicit concatenation), each located by its enclosing class and function."""
    by_source: list[tuple[str, str]] = []
    by_literal: list[tuple[str, str]] = []
    for rel, tree, text in _scanned_modules(root):
        for m in _OUTBOX_WRITE_RE.finditer(text):
            by_source.append((rel, _qualname_at(tree, text.count("\n", 0, m.start()) + 1)))
        for node, literal in _string_literals(tree):
            if _OUTBOX_WRITE_RE.search(literal):
                by_literal.append((rel, _qualname_at(tree, node.lineno)))
    return {"source": by_source, "string literals": by_literal}


def _outbox_bypass_findings(root: Path) -> list[tuple[str, int, str]]:
    """Ways to write event_outbox that the regexes cannot see, as (path, line, what)."""
    findings: list[tuple[str, int, str]] = []
    for rel, tree, _text in _scanned_modules(root):
        event_plane = rel.startswith("src/solace_events/")
        for node, literal in _string_literals(tree):
            if isinstance(node, ast.JoinedStr) and _INSERT_OR_COPY_RE.search(literal) and _OUTBOX_NAME_RE.search(literal):
                findings.append((rel, node.lineno, "f-string INSERT/COPY into event_outbox"))
            if event_plane:
                parts = node.values if isinstance(node, ast.JoinedStr) else []
                if any(isinstance(a, ast.Constant) and isinstance(a.value, str) and _DANGLING_INSERT_RE.search(a.value)
                       and isinstance(b, ast.FormattedValue) for a, b in pairwise(parts)):
                    findings.append((rel, node.lineno, "f-string INSERT with a formatted table"))
                if isinstance(node, ast.Constant) and (_DANGLING_INSERT_RE.search(literal)
                                                       or _PLACEHOLDER_INSERT_RE.search(literal)):
                    findings.append((rel, node.lineno, "INSERT whose table is concatenated or formatted in"))
            elif _OUTBOX_NAME_RE.search(literal):  # services reach the outbox only through solace_events
                findings.append((rel, node.lineno, "service string literal naming event_outbox"))
        if event_plane:
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None))
                        in _COPY_HELPERS):
                    table = node.args[0] if node.args else next(
                        (k.value for k in node.keywords if k.arg == "table_name"), None)
                    if not (isinstance(table, ast.Constant) and isinstance(table.value, str)):
                        findings.append((rel, node.lineno, "COPY helper with a run-time table"))
    return findings


def test_mvp_133_r1_guard_event_outbox_is_written_only_by_postgres_outbox_store_save():
    """Exactly one place writes event_outbox, PostgresOutboxStore.save, which enforces per-service ownership."""
    assert _outbox_write_sites(_REPO) == {"source": [_SAVE_SITE], "string literals": [_SAVE_SITE]}


def test_mvp_133_r1_guard_no_outbox_write_hides_from_the_scan():
    """No f-string spells an INSERT/COPY into event_outbox; the event plane builds no INSERT or COPY whose table is
    chosen at run time; and no service string literal names event_outbox (services have their own f-string
    repository inserts, so there the table name itself is what must not appear)."""
    assert _outbox_bypass_findings(_REPO) == []


def test_mvp_133_r1_guard_scans_the_event_plane_and_every_service():
    rels = [rel for rel, _tree, _text in _scanned_modules(_REPO)]
    assert "src/solace_events/postgres_stores.py" in rels and "src/solace_events/publisher.py" in rels
    services = {rel.split("/")[1] for rel in rels if rel.startswith("services/")}
    assert {"orchestrator_service", "safety_service", "therapy_service", "diagnosis_service", "personality_service",
            "user-service", "notification-service"} <= services

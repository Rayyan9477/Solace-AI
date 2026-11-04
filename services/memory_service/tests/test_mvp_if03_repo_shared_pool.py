"""IF-03: memory's repository uses the one shared pool (WS-17.2)."""
from __future__ import annotations

import inspect
from decimal import Decimal
from uuid import uuid4

from services.memory_service.src.domain.models import MemoryRecord
from services.memory_service.src.infrastructure import postgres_repo
from services.memory_service.src.infrastructure.postgres_repo import PostgresRepository


def test_if03_repository_builds_no_private_engine() -> None:
    source = inspect.getsource(postgres_repo)
    assert "create_async_engine" not in source
    assert "async_sessionmaker" not in source


async def test_if03_writes_run_in_a_connection_pool_manager_transaction(recording_cpm) -> None:
    record = MemoryRecord(user_id=uuid4(), session_id=uuid4(), content="synthetic note",
                          importance_score=Decimal("0.5"))
    await PostgresRepository().store_memory_record(record)
    assert recording_cpm.pool_names == ["default"]
    sql, args = recording_cpm.connection.executed[0]
    assert sql.startswith("INSERT INTO memory_records")
    assert "$1" in sql and record.record_id in args


async def test_if03_initialize_probes_and_close_leaves_the_runtime_pool_open(recording_cpm) -> None:
    repo = PostgresRepository()
    await repo.initialize()
    await repo.close()
    assert recording_cpm.connection.fetched[0][0] == "SELECT 1"
    assert recording_cpm.closed == []


async def test_if03_bind_errors_never_carry_bound_values(recording_cpm) -> None:
    """W172-SEC-1: asyncpg's DataError embeds repr(value)[:40]; the repository must not surface it."""
    import asyncpg

    marker = "SYNTHETIC-PHI-MARKER-7f3a"

    async def _raise_data_error(sql: str, *args: object) -> str:
        raise asyncpg.exceptions.DataError(
            f"invalid input for query argument $4: '{marker}...' (expected str, got dict)"
        )

    recording_cpm.connection.execute = _raise_data_error
    record = MemoryRecord(user_id=uuid4(), session_id=uuid4(), content=marker)
    try:
        await PostgresRepository().store_memory_record(record)
    except Exception as exc:  # the whole chain is inspected below
        raised = exc
    else:
        raise AssertionError("expected the bind error to propagate")
    chain, seen = [], raised
    while seen is not None and len(chain) < 10:
        chain.append(seen)
        seen = seen.__cause__ or seen.__context__
    for err in chain:
        assert marker not in str(err) and marker not in repr(err.args)
    assert "$4" in str(raised) and "DataError" in str(raised)

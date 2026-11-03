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

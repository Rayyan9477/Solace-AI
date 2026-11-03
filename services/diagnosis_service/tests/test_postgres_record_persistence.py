"""REV-04 (Sprint A.4): diagnosis save_record must supply NOT-NULL encryption_key_id.

The diagnosis_records table declares encryption_key_id NOT NULL (no server
default), but the raw-SQL INSERT omitted it — so every record insert failed and
the error was swallowed as a warning. This test captures the raw INSERT via a
fake connection and asserts the column + a non-empty value are provided.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.entities import DiagnosisRecordEntity
from services.diagnosis_service.src.infrastructure import postgres_repository as pr
from services.diagnosis_service.src.infrastructure.postgres_repository import (
    PostgresDiagnosisRepository,
)


class _CapturingConn:
    def __init__(self) -> None:
        self.calls: list = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return "INSERT 0 1"


class _FakeAcquire:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.asyncio
async def test_save_record_provides_encryption_key_id(monkeypatch) -> None:
    # Force the legacy client path (avoid ConnectionPoolManager) to capture the
    # raw INSERT.
    monkeypatch.setattr(pr, "FeatureFlags", None)
    conn = _CapturingConn()
    repo = PostgresDiagnosisRepository(_FakeClient(conn))

    record = DiagnosisRecordEntity(user_id=uuid4(), primary_diagnosis="GAD")
    await repo.save_record(record)

    assert conn.calls, "no INSERT executed"
    query, args = conn.calls[0]
    assert "encryption_key_id" in query, (
        "INSERT omits the NOT-NULL encryption_key_id column — every insert fails"
    )
    # The value bound for the NOT-NULL column must be non-empty.
    assert args[-1], "encryption_key_id value must be non-empty"

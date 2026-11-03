"""Shared fixtures for memory-service tests (WS-17.2).

``recording_cpm`` replaces the shared ConnectionPoolManager's ``acquire`` /
``transaction`` / ``close_pool`` / ``close_all_pools`` with recorders, so the
repository can be exercised without a real Postgres.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from solace_infrastructure.database.connection_manager import ConnectionPoolManager


class RecordingConnection:
    """Stand-in for an asyncpg connection that records every call."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetched: list[tuple[str, tuple]] = []
        self.rows: list[dict[str, Any]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return "INSERT 0 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.fetched.append((sql, args))
        return list(self.rows)

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.fetched.append((sql, args))
        return self.rows[0] if self.rows else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.fetched.append((sql, args))
        return None


class RecordingPoolManager:
    """What the ``recording_cpm`` fixture exposes to tests."""

    def __init__(self) -> None:
        self.connection = RecordingConnection()
        self.pool_names: list[str] = []
        self.closed: list[str] = []


@pytest.fixture
def recording_cpm(monkeypatch: pytest.MonkeyPatch) -> RecordingPoolManager:
    recorder = RecordingPoolManager()

    @asynccontextmanager
    async def _acquire(pool_name: str = "default", timeout: float = 10.0) -> AsyncIterator[RecordingConnection]:
        recorder.pool_names.append(pool_name)
        yield recorder.connection

    @asynccontextmanager
    async def _transaction(
        pool_name: str = "default", isolation: str = "read_committed"
    ) -> AsyncIterator[RecordingConnection]:
        recorder.pool_names.append(pool_name)
        yield recorder.connection

    async def _close_pool(name: str) -> None:
        recorder.closed.append(name)

    async def _close_all_pools() -> None:
        recorder.closed.append("*")

    monkeypatch.setattr(ConnectionPoolManager, "acquire", _acquire)
    monkeypatch.setattr(ConnectionPoolManager, "transaction", _transaction)
    monkeypatch.setattr(ConnectionPoolManager, "close_pool", _close_pool)
    monkeypatch.setattr(ConnectionPoolManager, "close_all_pools", _close_all_pools)
    return recorder

"""C.2 — the DB layer sets the RLS GUC (app.current_user_id) on acquired connections.

Verified against a mock asyncpg connection (RLS itself is a no-op on SQLite and needs
real Postgres, deferred to B.3). Asserts: the GUC is set to the request user on acquire
and reset on release; nothing is set when no user is in context.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import pytest

from solace_infrastructure.database.connection_manager import ConnectionPoolManager
from solace_common.request_context import reset_current_user_id, set_current_user_id


class _FakeConn:
    def __init__(self):
        self.executed: list[tuple] = []

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"


class _FakeAcquireCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self, timeout=None):
        return _FakeAcquireCM(self._conn)

    def get_size(self):
        return 1

    def get_idle_size(self):
        return 1


def _patch_pool(monkeypatch, conn):
    async def fake_get_pool(name="default"):
        return _FakePool(conn)

    monkeypatch.setattr(ConnectionPoolManager, "get_pool", fake_get_pool)


@pytest.mark.asyncio
async def test_acquire_sets_and_resets_guc_for_request_user(monkeypatch):
    conn = _FakeConn()
    _patch_pool(monkeypatch, conn)
    tok = set_current_user_id("user-abc")
    try:
        async with ConnectionPoolManager.acquire() as c:
            assert c is conn
    finally:
        reset_current_user_id(tok)

    # First statement sets the GUC to the user; the last resets it.
    assert conn.executed[0] == (
        "SELECT set_config('app.current_user_id', $1, false)",
        ("user-abc",),
    )
    assert conn.executed[-1] == (
        "SELECT set_config('app.current_user_id', '', false)",
        (),
    )


@pytest.mark.asyncio
async def test_acquire_sets_no_guc_when_no_user_in_context(monkeypatch):
    conn = _FakeConn()
    _patch_pool(monkeypatch, conn)
    reset_current_user_id()
    async with ConnectionPoolManager.acquire():
        pass
    assert conn.executed == []  # no GUC set/reset when unauthenticated

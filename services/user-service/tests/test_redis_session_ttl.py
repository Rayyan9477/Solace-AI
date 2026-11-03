"""REV-31: RedisSessionManager must use the redis.asyncio `ex=` expiry kwarg.

The manager previously called ``redis.set(key, value, ttl=...)``. The
``redis.asyncio`` client's ``set`` takes ``ex=`` (seconds), not ``ttl=``, so
every session write raised ``TypeError`` — meaning **every login failed** the
moment a real Redis was wired in (unit tests using loose mocks never caught it).

These tests use a fake whose ``set`` signature matches the real client (accepts
``ex=``, rejects ``ttl=``), so a regression to ``ttl=`` fails here immediately.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.auth import RedisSessionManager, SessionConfig


class _RedisAccurateFake:
    """Mimics redis.asyncio.Redis: set() accepts ex=/px=, NOT ttl=."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.last_ex: int | None = None

    async def set(self, name, value, ex=None, px=None, nx=False, xx=False, keepttl=False):
        self.store[name] = value
        self.last_ex = ex
        return True

    async def get(self, name):
        return self.store.get(name)

    async def delete(self, *names):
        for n in names:
            self.store.pop(n, None)
        return len(names)


@pytest.fixture
def redis_fake() -> _RedisAccurateFake:
    return _RedisAccurateFake()


@pytest.fixture
def manager(redis_fake: _RedisAccurateFake) -> RedisSessionManager:
    return RedisSessionManager(
        redis_fake,
        SessionConfig(session_timeout_minutes=30, max_sessions_per_user=5),
    )


@pytest.mark.asyncio
async def test_create_session_does_not_raise_typeerror(
    manager: RedisSessionManager, redis_fake: _RedisAccurateFake
) -> None:
    """create_session must succeed against a redis-accurate client (no ttl= kwarg)."""
    user = SimpleNamespace(user_id=uuid4())
    session = await manager.create_session(user, ip_address="10.0.0.1")

    assert session is not None
    # Session stored with the correct expiry kwarg (ex=), value round-trips.
    got = await manager.get_session(session.session_id)
    assert got is not None
    assert got.session_id == session.session_id
    assert redis_fake.last_ex == 30 * 60  # session_timeout_minutes * 60


@pytest.mark.asyncio
async def test_update_activity_uses_ex_expiry(
    manager: RedisSessionManager, redis_fake: _RedisAccurateFake
) -> None:
    """update_activity re-writes the session with an ex= TTL, not ttl=."""
    user = SimpleNamespace(user_id=uuid4())
    session = await manager.create_session(user, ip_address="10.0.0.1")
    redis_fake.last_ex = None

    await manager.update_activity(session.session_id)
    assert redis_fake.last_ex == 30 * 60


@pytest.mark.asyncio
async def test_user_session_index_persists(
    manager: RedisSessionManager, redis_fake: _RedisAccurateFake
) -> None:
    """The per-user session-id index is written (also via ex=), not lost to a TypeError."""
    uid = uuid4()
    user = SimpleNamespace(user_id=uid)
    session = await manager.create_session(user, ip_address="10.0.0.1")

    user_key = manager._user_key(uid)
    stored = json.loads(redis_fake.store[user_key])
    assert str(session.session_id) in stored


class _TtlTrackingFake:
    """Fake redis that records the ex= expiry set on each key (per-key)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}

    async def set(self, name, value, ex=None, px=None, nx=False, xx=False, keepttl=False):
        self.store[name] = value
        self.ttl[name] = ex
        return True

    async def get(self, name):
        return self.store.get(name)

    async def delete(self, *names):
        for n in names:
            self.store.pop(n, None)
            self.ttl.pop(n, None)
        return len(names)


@pytest.mark.asyncio
async def test_revoke_session_preserves_index_ttl() -> None:
    """M1 gate P2: revoking one of several sessions must not clear the index TTL.

    A bare ``SET`` (no ex=) on the rewritten user_sessions index cleared its TTL,
    leaking the key permanently in Redis.
    """
    fake = _TtlTrackingFake()
    manager = RedisSessionManager(
        fake, SessionConfig(session_timeout_minutes=30, max_sessions_per_user=5)
    )
    user = SimpleNamespace(user_id=uuid4())
    s1 = await manager.create_session(user)
    await manager.create_session(user)
    user_key = manager._user_key(user.user_id)
    assert fake.ttl[user_key] == 30 * 60  # created with a TTL

    await manager.revoke_session(s1.session_id)  # one session remains
    assert fake.ttl[user_key] == 30 * 60, "revoke_session cleared the index TTL"

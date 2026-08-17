"""M1 gate round-2 (P0): RedisTokenBlacklist.add must use the redis.asyncio `ex=` kwarg.

The blacklist `add` (called on logout / force-revoke) did `redis.set(key, "1", ttl=...)`.
The `redis.asyncio` client's `set` takes `ex=` (seconds), not `ttl=`, so every add raised
TypeError in production — the JTI was NEVER stored, defeating ALL token revocation
(REV-06 / REV-08 / REV-33 all rely on the blacklist being populated).

Uses a fake whose `set` signature matches the real client (accepts `ex=`, rejects `ttl=`).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from solace_security.auth import RedisTokenBlacklist


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


@pytest.mark.asyncio
async def test_blacklist_add_stores_jti_with_ex_expiry() -> None:
    bl = RedisTokenBlacklist(_RedisAccurateFake())
    expires = datetime.now(timezone.utc) + timedelta(hours=1)

    # Was: TypeError('unexpected keyword argument ttl') → JTI never stored.
    await bl.add("jti-abc", expires)

    assert await bl.is_blacklisted("jti-abc") is True
    assert bl._redis.last_ex is not None and bl._redis.last_ex > 0  # a TTL was set


@pytest.mark.asyncio
async def test_blacklist_add_ttl_is_bounded_to_remaining_lifetime() -> None:
    bl = RedisTokenBlacklist(_RedisAccurateFake())
    await bl.add("jti-xyz", datetime.now(timezone.utc) + timedelta(seconds=120))
    assert 1 <= bl._redis.last_ex <= 120

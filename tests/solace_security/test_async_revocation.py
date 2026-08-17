"""REV-06 (Sprint A.5): request auth must detect token revocation via an async
(Redis-style) blacklist, not only the sync best-effort path.

RedisTokenBlacklist implements only the async ``is_blacklisted``; the sync
``is_blacklisted_sync`` falls back to the base (returns False). The FastAPI
dependency ``get_current_user`` used the sync ``validate_access_token``, so
revoked tokens sailed through in production. ``validate_access_token_async``
routes through the async revocation check.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from solace_security.auth import AuthSettings, JWTManager, TokenBlacklist


class _AsyncOnlyBlacklist(TokenBlacklist):
    """Simulates RedisTokenBlacklist: async check works, sync check does not."""

    def __init__(self) -> None:
        self._revoked: set[str] = set()

    async def add(self, jti: str, expires_at: datetime) -> None:
        self._revoked.add(jti)

    async def is_blacklisted(self, jti: str) -> bool:
        return jti in self._revoked

    # is_blacklisted_sync inherited from base → returns False (like Redis).


@pytest.mark.asyncio
async def test_async_validation_detects_revocation_sync_does_not() -> None:
    secret = "test-secret-key-32-bytes-long!!!"
    blacklist = _AsyncOnlyBlacklist()
    manager = JWTManager(AuthSettings(secret_key=secret), token_blacklist=blacklist)

    token = manager.create_access_token("user123")
    jti = manager.decode_token_sync(token).payload.jti
    await blacklist.add(jti, datetime.now(timezone.utc) + timedelta(hours=1))

    # Sync path MISSES the revocation (documents the vulnerability).
    assert manager.validate_access_token(token).success is True

    # Async path DETECTS it — this is what get_current_user must use.
    result = await manager.validate_access_token_async(token)
    assert result.success is False
    assert result.error_code == "TOKEN_REVOKED"

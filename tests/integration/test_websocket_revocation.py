"""M1 gate finding (P0): revoked tokens must be rejected on the WebSocket path.

REV-06 routed HTTP auth through the async (Redis-aware) revocation check, but the
orchestrator WebSocket auth paths were left on the SYNC validator. Because
``RedisTokenBlacklist`` implements only the async ``is_blacklisted`` (the sync
``is_blacklisted_sync`` falls back to the base → returns False), a revoked /
logged-out token was still accepted on ``/ws/{session_id}`` in the multi-worker
(Redis blacklist) production config — defeating revocation on the real-time chat
surface.

This test drives ``ConnectionManager.authenticate_token`` with a Redis-style
blacklist that has revoked the token; it must return None (rejected).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from solace_security.auth import AuthSettings, JWTManager, TokenBlacklist

from services.orchestrator_service.src.websocket import ConnectionManager, WebSocketSettings


class _AsyncOnlyBlacklist(TokenBlacklist):
    """Simulates RedisTokenBlacklist: async check works, sync check does not."""

    def __init__(self) -> None:
        self._revoked: set[str] = set()

    async def add(self, jti: str, expires_at: datetime) -> None:
        self._revoked.add(jti)

    async def is_blacklisted(self, jti: str) -> bool:
        return jti in self._revoked

    # is_blacklisted_sync inherited from base → returns False (like Redis).


def test_websocket_auth_rejects_revoked_token() -> None:
    async def _run() -> None:
        secret = "test-secret-key-32-bytes-long!!!"
        blacklist = _AsyncOnlyBlacklist()
        manager = JWTManager(AuthSettings(secret_key=secret), token_blacklist=blacklist)

        token = manager.create_access_token("user123")
        jti = manager.decode_token_sync(token).payload.jti
        await blacklist.add(jti, datetime.now(timezone.utc) + timedelta(hours=1))

        cm = ConnectionManager(settings=WebSocketSettings(), jwt_manager=manager)

        # A valid (non-revoked) token still authenticates.
        good_token = manager.create_access_token("user456")
        assert await cm.authenticate_token(good_token) is not None

        # The revoked token must be rejected (was: accepted via sync validator).
        assert await cm.authenticate_token(token) is None

    asyncio.run(_run())

"""P2-2: refresh-token rotation + reuse detection on /auth/refresh.

Each refresh must revoke the CONSUMED refresh token and issue a fresh pair, so a
replayed old refresh token is rejected after one rotation. Replaying an already-rotated
token additionally revokes the whole family (all the user's sessions).
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api_auth import RefreshRequest, refresh_token
from src.infrastructure.jwt_service import (
    JWTConfig,
    JWTService,
    InMemoryTokenBlacklist,
)

SECRET = "test-secret-key-for-refresh-rotation-32b"


class _FakeSessionManager:
    def __init__(self) -> None:
        self.revoked_users: list = []
        self.active = True

    async def get_user_sessions(self, user_id, active_only: bool = True):
        return ["sess-1"] if self.active else []

    async def revoke_all_user_sessions(self, user_id):
        self.revoked_users.append(user_id)
        return 1


def _fake_request():
    svc = SimpleNamespace(increment_stat=lambda *_: None)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(service=svc)))


def _service() -> JWTService:
    return JWTService(JWTConfig(secret_key=SECRET, algorithm="HS256"), blacklist=InMemoryTokenBlacklist())


@pytest.mark.asyncio
async def test_replayed_refresh_token_rejected_after_one_rotation():
    svc = _service()
    sessions = _FakeSessionManager()
    uid = uuid4()
    pair = svc.generate_token_pair(uid, "u@example.com", "user")

    # First rotation succeeds and returns a fresh pair.
    resp = await refresh_token(
        RefreshRequest(refresh_token=pair.refresh_token),
        _fake_request(),
        jwt_service=svc,
        session_manager=sessions,
    )
    assert resp.refresh_token and resp.access_token
    assert resp.refresh_token != pair.refresh_token

    # Replaying the ORIGINAL (now consumed) refresh token is rejected...
    with pytest.raises(HTTPException) as exc:
        await refresh_token(
            RefreshRequest(refresh_token=pair.refresh_token),
            _fake_request(),
            jwt_service=svc,
            session_manager=sessions,
        )
    assert exc.value.status_code == 401
    # ...and the reuse triggers a family revoke of all the user's sessions.
    assert uid in sessions.revoked_users


@pytest.mark.asyncio
async def test_rotated_refresh_token_still_works_once():
    svc = _service()
    sessions = _FakeSessionManager()
    uid = uuid4()
    pair = svc.generate_token_pair(uid, "u@example.com", "user")

    resp1 = await refresh_token(
        RefreshRequest(refresh_token=pair.refresh_token),
        _fake_request(), jwt_service=svc, session_manager=sessions,
    )
    # The freshly-issued refresh token is itself usable for the next rotation.
    resp2 = await refresh_token(
        RefreshRequest(refresh_token=resp1.refresh_token),
        _fake_request(), jwt_service=svc, session_manager=sessions,
    )
    assert resp2.refresh_token and resp2.refresh_token != resp1.refresh_token
    assert sessions.revoked_users == []  # no reuse, no family revoke

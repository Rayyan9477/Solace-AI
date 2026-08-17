"""C.2 — the auth dependency publishes the authenticated user into the request context
so the DB layer can set the RLS GUC.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager
from solace_common.request_context import get_current_user_id, reset_current_user_id


class _FakeURL:
    path = "/api/v1/x"


class _FakeRequest:
    url = _FakeURL()


@pytest.mark.asyncio
async def test_get_current_user_sets_request_context(monkeypatch):
    reset_current_user_id()
    jm = JWTManager(
        AuthSettings(secret_key="test-secret-key-32-bytes-long!!!"),
        token_blacklist=InMemoryTokenBlacklist(),
    )
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: jm)

    token = jm.create_access_token("user-xyz", roles=["user"])
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    user = await mw.get_current_user(_FakeRequest(), creds)

    assert user.user_id == "user-xyz"
    assert get_current_user_id() == "user-xyz"  # published for the DB/RLS layer
    reset_current_user_id()

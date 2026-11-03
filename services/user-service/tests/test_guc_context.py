"""P1-3: user-service get_current_user must set the RLS GUC request context.

The user-service's own get_current_user dependency never published the validated
user id into solace_common.request_context, so the DB layer never set the Postgres
app.current_user_id GUC — breaking RLS on the consent/preferences tables (migration
006) on real Postgres. It must set the context from the token and reset it per-request.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.api import get_current_user
from src.infrastructure.jwt_service import JWTConfig, JWTService
from solace_common.request_context import get_current_user_id

SECRET = "test-secret-key-for-guc-context-only-32b"


def _request_with_service(svc: JWTService):
    state = SimpleNamespace(jwt_service=svc)
    app = SimpleNamespace(state=SimpleNamespace(service=state))
    return SimpleNamespace(app=app)


@pytest.mark.asyncio
async def test_get_current_user_sets_and_resets_guc_context():
    svc = JWTService(JWTConfig(secret_key=SECRET, algorithm="HS256"))
    uid = uuid4()
    pair = svc.generate_token_pair(uid, "u@example.com", "user")
    request = _request_with_service(svc)

    agen = get_current_user(request, authorization=f"Bearer {pair.access_token}")
    payload = await agen.__anext__()
    # While the dependency is live, the context holds this user's id.
    assert payload.user_id == uid
    assert get_current_user_id() == str(uid)

    # Closing the dependency (request teardown) resets the context.
    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()
    assert get_current_user_id() is None

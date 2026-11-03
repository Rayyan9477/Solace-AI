"""C.2 / A2: per-user RLS GUC propagation for the personality service.

RLS enforcement itself is verified on real Postgres (B.3 / staging); these tests
verify the GUC-SETTING behavior the enforcement depends on:

  * the service-auth endpoints (``get_current_service`` sets no user GUC) publish
    ``request.user_id`` into the request context so the DB layer sets
    ``app.current_user_id`` — else RLS would admit/return nothing;
  * the GDPR erase path scopes the GUC to the TARGET user for the delete and
    restores the caller's context afterward — else an admin erasing another user
    (request GUC = admin) would be RLS-filtered to the admin's own rows and
    silently erase nothing.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from solace_common.request_context import (
    get_current_user_id,
    reset_current_user_id,
    set_current_user_id,
)

from fastapi import HTTPException

from services.personality_service.src.api import (
    adapt_response,
    detect_personality,
    get_style,
)
from services.personality_service.src.schemas import (
    AdaptResponseRequest,
    DetectPersonalityRequest,
    GetStyleRequest,
)
from services.personality_service.src.domain.service import (
    PersonalityOrchestrator,
    PersonalityServiceSettings,
)
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository

TARGET_USER_ID = "00000000-0000-4000-a000-000000000099"
CALLER_USER_ID = "00000000-0000-4000-a000-000000000042"


@pytest.mark.asyncio
async def test_detect_endpoint_sets_guc_to_request_user() -> None:
    """The /detect service-to-service endpoint sets the per-user GUC from the body."""
    captured: dict[str, str | None] = {}

    class _CapturingOrchestrator:
        async def detect_personality(self, request):  # noqa: ANN001
            captured["guc"] = get_current_user_id()
            return "ok"

    reset_current_user_id()  # start unset (no user token on a service call)
    req = DetectPersonalityRequest(user_id=UUID(TARGET_USER_ID), text="I feel great today!")
    # `service` is only a Depends marker; the handler body never references it.
    await detect_personality(req, _CapturingOrchestrator(), service=None)

    assert captured["guc"] == TARGET_USER_ID


@pytest.mark.asyncio
async def test_erase_scopes_guc_to_target_and_restores_caller() -> None:
    """delete_user_data runs the DELETEs with the GUC = TARGET (not the caller),
    then restores the caller's request context."""
    captured: dict[str, str | None] = {}

    class _CapturingRepo(InMemoryPersonalityRepository):
        async def delete_user_data(self, user_id):  # noqa: ANN001
            captured["guc"] = get_current_user_id()
            return await super().delete_user_data(user_id)

    orchestrator = PersonalityOrchestrator(
        settings=PersonalityServiceSettings(enable_llm_detection=False),
        llm_client=None,
        repository=_CapturingRepo(),
    )

    # Simulate an admin caller's request context (GUC = admin, not the target).
    token = set_current_user_id(CALLER_USER_ID)
    try:
        await orchestrator.delete_user_data(UUID(TARGET_USER_ID))
        after = get_current_user_id()
    finally:
        reset_current_user_id(token)

    # The delete ran scoped to the TARGET user (so RLS admits the target's rows)...
    assert captured["guc"] == TARGET_USER_ID
    # ...and the caller's context was restored once the erase completed.
    assert after == CALLER_USER_ID


# --- P2-7: request-scoped GUC must be RESET after the handler --------------------
# The service-auth endpoints (/detect, /style, /adapt) set the per-user GUC from the
# request body but must reset it in a finally so isolation does not rely solely on
# asyncio task boundaries (a reused task / thread would otherwise leak the GUC).

PRIOR_USER_ID = "00000000-0000-4000-a000-000000000001"


class _OkOrchestrator:
    async def detect_personality(self, request):  # noqa: ANN001
        return "detect-ok"

    async def get_style(self, request):  # noqa: ANN001
        return "style-ok"

    async def adapt_response(self, request):  # noqa: ANN001
        return "adapt-ok"


class _FailingOrchestrator:
    async def detect_personality(self, request):  # noqa: ANN001
        raise ValueError("boom")

    async def get_style(self, request):  # noqa: ANN001
        raise RuntimeError("boom")

    async def adapt_response(self, request):  # noqa: ANN001
        raise ValueError("boom")


@pytest.mark.asyncio
async def test_detect_endpoint_resets_guc_to_prior_value() -> None:
    token = set_current_user_id(PRIOR_USER_ID)  # a prior request-context value
    try:
        req = DetectPersonalityRequest(user_id=UUID(TARGET_USER_ID), text="I feel great today!")
        await detect_personality(req, _OkOrchestrator(), service=None)
        assert get_current_user_id() == PRIOR_USER_ID  # restored, not left at TARGET
    finally:
        reset_current_user_id(token)


@pytest.mark.asyncio
async def test_style_endpoint_resets_guc_to_prior_value() -> None:
    token = set_current_user_id(PRIOR_USER_ID)
    try:
        req = GetStyleRequest(user_id=UUID(TARGET_USER_ID))
        await get_style(req, _OkOrchestrator(), service=None)
        assert get_current_user_id() == PRIOR_USER_ID
    finally:
        reset_current_user_id(token)


@pytest.mark.asyncio
async def test_adapt_endpoint_resets_guc_to_prior_value() -> None:
    token = set_current_user_id(PRIOR_USER_ID)
    try:
        req = AdaptResponseRequest(user_id=UUID(TARGET_USER_ID), base_response="hello")
        await adapt_response(req, _OkOrchestrator(), service=None)
        assert get_current_user_id() == PRIOR_USER_ID
    finally:
        reset_current_user_id(token)


@pytest.mark.asyncio
async def test_endpoints_reset_guc_even_on_error() -> None:
    """The finally-reset must run on the error path too, so a failed request does not
    leak the target user's GUC into a reused task."""
    reset_current_user_id()  # start unset

    with pytest.raises(HTTPException):
        await detect_personality(
            DetectPersonalityRequest(user_id=UUID(TARGET_USER_ID), text="I feel great today!"),
            _FailingOrchestrator(),
            service=None,
        )
    assert get_current_user_id() is None

    with pytest.raises(HTTPException):
        await get_style(
            GetStyleRequest(user_id=UUID(TARGET_USER_ID)),
            _FailingOrchestrator(),
            service=None,
        )
    assert get_current_user_id() is None

    with pytest.raises(HTTPException):
        await adapt_response(
            AdaptResponseRequest(user_id=UUID(TARGET_USER_ID), base_response="hello"),
            _FailingOrchestrator(),
            service=None,
        )
    assert get_current_user_id() is None

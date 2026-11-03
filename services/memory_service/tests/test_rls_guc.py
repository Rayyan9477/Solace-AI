"""C.2 / P1-2: per-user RLS GUC scoping on the memory GDPR erase path.

The memory Postgres tables are RLS-protected (migration 006). The erase endpoint
authenticates the CALLER, so the request's ``app.current_user_id`` GUC is the
caller's id. When an admin (or any user other than the target) triggers erasure,
an unscoped erase would run every DELETE under the CALLER's GUC — RLS then filters
those DELETEs to the caller's OWN rows and the target's PHI is silently NOT
deleted while the erasure is attested complete.

RLS enforcement itself is verified on real Postgres (B.3 / staging); these tests
verify the GUC-SETTING behavior the enforcement depends on:

  * ``MemoryService.delete_user_data`` scopes the GUC to the TARGET user for the
    duration of the erase (so RLS admits the target's rows across every store),
    then restores the caller's request context afterward.

Existing memory GDPR tests (``test_gdpr_erasure.py``) still hold: the in-memory
tier clearing and the fail-loud / HTTP-204 contract are unchanged.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from solace_common.request_context import (
    get_current_user_id,
    reset_current_user_id,
    set_current_user_id,
)

from services.memory_service.src.domain.models import MemoryServiceSettings
from services.memory_service.src.domain.service import MemoryService

TARGET_USER_ID = "00000000-0000-4000-a000-000000000099"
CALLER_USER_ID = "00000000-0000-4000-a000-000000000042"


class _CapturingPostgres:
    """Fake memory Postgres repo capturing the GUC seen during the DELETE."""

    def __init__(self) -> None:
        self.guc_during_delete: str | None = None
        self.deleted: list = []

    async def delete_user_data(self, user_id):  # noqa: ANN001
        self.guc_during_delete = get_current_user_id()
        self.deleted.append(user_id)
        return (2, 1, 3, 0)  # records, summaries, facts, events


class _CapturingWeaviate:
    def __init__(self) -> None:
        self.guc_during_delete: str | None = None

    async def delete_user_data(self, user_id):  # noqa: ANN001
        self.guc_during_delete = get_current_user_id()
        return 3


class _CapturingRedis:
    def __init__(self) -> None:
        self.guc_during_delete: str | None = None

    async def delete_user_cache(self, user_id):  # noqa: ANN001
        self.guc_during_delete = get_current_user_id()
        return 5


@pytest.mark.asyncio
async def test_erase_scopes_guc_to_target_and_restores_caller() -> None:
    """delete_user_data runs every store DELETE with the GUC = TARGET (not the
    caller), then restores the caller's request context."""
    pg, wv, rd = _CapturingPostgres(), _CapturingWeaviate(), _CapturingRedis()
    service = MemoryService(
        settings=MemoryServiceSettings(),
        postgres_repo=pg,
        weaviate_repo=wv,
        redis_cache=rd,
    )
    target = uuid4()
    service._tier_3_session[target] = ["seed"]

    # Simulate an admin caller's request context (GUC = admin, not the target).
    token = set_current_user_id(CALLER_USER_ID)
    try:
        await service.delete_user_data(user_id=target)
        after = get_current_user_id()
    finally:
        reset_current_user_id(token)

    # Every durable store's DELETE ran scoped to the TARGET user (so RLS admits
    # the target's rows), NOT the admin caller.
    assert pg.guc_during_delete == str(target)
    assert wv.guc_during_delete == str(target)
    assert rd.guc_during_delete == str(target)
    # The caller's request context was restored once the erase completed.
    assert after == CALLER_USER_ID


@pytest.mark.asyncio
async def test_erase_restores_caller_guc_even_when_a_store_raises() -> None:
    """The caller's GUC is restored even if the erase fails mid-way (finally)."""

    class _RaisingPostgres:
        async def delete_user_data(self, user_id):  # noqa: ANN001
            raise RuntimeError("db pool exhausted")

    from solace_infrastructure.gdpr import ErasureIncomplete

    service = MemoryService(
        settings=MemoryServiceSettings(),
        postgres_repo=_RaisingPostgres(),
        weaviate_repo=_CapturingWeaviate(),
        redis_cache=_CapturingRedis(),
    )
    target = uuid4()

    token = set_current_user_id(CALLER_USER_ID)
    try:
        with pytest.raises(ErasureIncomplete):
            await service.delete_user_data(user_id=target)
        # Context restored to the caller even on the fail-loud path.
        assert get_current_user_id() == CALLER_USER_ID
    finally:
        reset_current_user_id(token)


@pytest.mark.asyncio
async def test_erase_without_stores_leaves_caller_guc_untouched() -> None:
    """With no durable stores registered the erase is a no-op for the GUC — the
    caller's request context is left exactly as it was."""
    service = MemoryService(settings=MemoryServiceSettings())
    target = uuid4()

    token = set_current_user_id(CALLER_USER_ID)
    try:
        await service.delete_user_data(user_id=target)
        assert get_current_user_id() == CALLER_USER_ID
    finally:
        reset_current_user_id(token)

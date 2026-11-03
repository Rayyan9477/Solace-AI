"""REV-17 GDPR right-to-erasure tests for the Personality Service.

Acceptance criteria encoded here (RED first):
  (a) DELETE /user/{user_id} erases the user from the persistent (Postgres) store
      via a fake repo AND clears the in-memory ProfileStore cache, self-or-admin
      gated (a non-admin cannot erase another user's data).
  (b) The erasure factory registers exactly the persistent store(s) the service
      owns, under a stable name.
  (c) A failing store deleter is fail-loud: erasure raises ErasureIncomplete
      carrying the failing store (a partial erasure is never reported complete).
"""
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_infrastructure.gdpr import ErasureIncomplete

from services.personality_service.src.api import router
from services.personality_service.src.domain.service import (
    PersonalityOrchestrator,
    PersonalityServiceSettings,
    PersonalityProfile as ServicePersonalityProfile,
    build_user_data_erasure,
)
from services.personality_service.src.domain.entities import (
    PersonalityProfile as EntityProfile,
)
from services.personality_service.tests.fixtures import InMemoryPersonalityRepository

from solace_security.middleware import (
    AuthenticatedUser,
    get_current_user,
)
from solace_security.auth import TokenType

TEST_USER_ID = "00000000-0000-4000-a000-000000000099"
OTHER_USER_ID = "00000000-0000-4000-a000-000000000042"


def _make_orchestrator(repository) -> PersonalityOrchestrator:
    return PersonalityOrchestrator(
        settings=PersonalityServiceSettings(enable_llm_detection=False),
        llm_client=None,
        repository=repository,
    )


def _build_app(orchestrator, *, user_id: str, roles=None):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personality")
    app.state.personality_orchestrator = orchestrator
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        user_id=user_id, token_type=TokenType.ACCESS, roles=roles or ["user"],
    )
    return app


# --------------------------------------------------------------------------- #
# (b) registry store name(s)
# --------------------------------------------------------------------------- #
class TestErasureFactory:
    def test_factory_registers_personality_postgres_store(self) -> None:
        repo = InMemoryPersonalityRepository()
        erasure = build_user_data_erasure(repo)
        assert erasure.store_names == ("personality_postgres",)


# --------------------------------------------------------------------------- #
# (a) endpoint deletes via a fake repo + clears in-memory profile
# --------------------------------------------------------------------------- #
class TestDeleteUserEndpoint:
    def test_delete_erases_postgres_and_clears_in_memory(self) -> None:
        repo = InMemoryPersonalityRepository()
        uid = UUID(TEST_USER_ID)
        # Seed the persistent (fake postgres) store.
        asyncio.run(repo.save_profile(EntityProfile(user_id=uid)))
        assert asyncio.run(repo.get_profile_by_user(uid)) is not None

        orchestrator = _make_orchestrator(repo)
        # Seed the in-memory cache directly (white-box; avoids binding the
        # ProfileStore's asyncio.Lock to a throwaway seeding loop).
        orchestrator._profile_store._profiles[uid] = ServicePersonalityProfile(user_id=uid)
        assert orchestrator._profile_store.exists(uid) is True

        app = _build_app(orchestrator, user_id=TEST_USER_ID)
        client = TestClient(app)
        resp = client.delete(f"/api/v1/personality/user/{TEST_USER_ID}")

        assert resp.status_code == 204
        # In-memory cache purged.
        assert orchestrator._profile_store.get(uid) is None
        assert orchestrator._profile_store.exists(uid) is False
        # Persistent store purged.
        assert asyncio.run(repo.get_profile_by_user(uid)) is None

    def test_delete_other_user_forbidden_for_non_admin(self) -> None:
        repo = InMemoryPersonalityRepository()
        target = UUID(TEST_USER_ID)
        asyncio.run(repo.save_profile(EntityProfile(user_id=target)))

        orchestrator = _make_orchestrator(repo)
        # Authenticated as a DIFFERENT, non-admin user.
        app = _build_app(orchestrator, user_id=OTHER_USER_ID, roles=["user"])
        client = TestClient(app)
        resp = client.delete(f"/api/v1/personality/user/{TEST_USER_ID}")

        assert resp.status_code == 403
        # Data must NOT have been touched.
        assert asyncio.run(repo.get_profile_by_user(target)) is not None

    def test_admin_may_delete_other_users_data(self) -> None:
        repo = InMemoryPersonalityRepository()
        target = UUID(TEST_USER_ID)
        asyncio.run(repo.save_profile(EntityProfile(user_id=target)))

        orchestrator = _make_orchestrator(repo)
        app = _build_app(orchestrator, user_id=OTHER_USER_ID, roles=["admin"])
        client = TestClient(app)
        resp = client.delete(f"/api/v1/personality/user/{TEST_USER_ID}")

        assert resp.status_code == 204
        assert asyncio.run(repo.get_profile_by_user(target)) is None


# --------------------------------------------------------------------------- #
# (c) fail-loud on repo error
# --------------------------------------------------------------------------- #
class _RaisingRepository(InMemoryPersonalityRepository):
    async def delete_user_data(self, user_id: UUID) -> int:  # type: ignore[override]
        raise RuntimeError("postgres unavailable")


class TestErasureFailLoud:
    async def test_repo_error_raises_erasure_incomplete(self) -> None:
        repo = _RaisingRepository()
        orchestrator = _make_orchestrator(repo)
        uid = UUID(TEST_USER_ID)

        with pytest.raises(ErasureIncomplete) as exc_info:
            await orchestrator.delete_user_data(uid)

        # The failing store is named and never reported as complete.
        assert "personality_postgres" in exc_info.value.failed
        assert exc_info.value.report.complete is False

    async def test_endpoint_returns_500_on_incomplete_erasure(self) -> None:
        repo = _RaisingRepository()
        orchestrator = _make_orchestrator(repo)
        app = _build_app(orchestrator, user_id=TEST_USER_ID)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete(f"/api/v1/personality/user/{TEST_USER_ID}")
        assert resp.status_code == 500

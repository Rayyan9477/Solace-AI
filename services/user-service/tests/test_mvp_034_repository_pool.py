"""MVP-034 / R-02.8: repositories use the one default pool; registration writes preferences as the new user."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from solace_common.request_context import get_current_user_id
from solace_infrastructure.database.connection_manager import ConnectionPoolManager
import src.infrastructure.postgres_repository as pg_repo
from src.domain.service import UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository


@pytest.mark.asyncio
async def test_mvp_034_repositories_acquire_from_the_default_pool(monkeypatch) -> None:
    seen: list[str] = []

    @asynccontextmanager
    async def _acquire(pool_name="default", timeout=10.0):
        seen.append(pool_name)
        yield object()

    monkeypatch.setattr(ConnectionPoolManager, "acquire", _acquire)
    for cls in (pg_repo.PostgresUserRepository, pg_repo.PostgresUserPreferencesRepository,
                pg_repo.PostgresConsentRepository):
        async with cls()._acquire():
            pass
    assert seen == ["default", "default", "default"]


@pytest.mark.asyncio
async def test_mvp_034_registration_saves_preferences_as_the_new_user() -> None:
    class _RecordingPrefs(InMemoryUserPreferencesRepository):
        def __init__(self) -> None:
            super().__init__()
            self.contexts: list[str | None] = []

        async def save(self, preferences):
            self.contexts.append(get_current_user_id())
            return await super().save(preferences)

    prefs = _RecordingPrefs()
    svc = UserService(user_repository=InMemoryUserRepository(), preferences_repository=prefs)
    result = await svc.create_user(email="ctx@example.com", password_hash="h", display_name="Ctx")
    assert result.success is True
    assert prefs.contexts == [str(result.user.user_id)]
    assert get_current_user_id() is None


@pytest.mark.asyncio
async def test_mvp_034_default_preferences_are_created_as_their_owner() -> None:
    """The default-create branch of update_preferences writes the new row as its owner, then restores."""
    from uuid import uuid4

    class _RecordingPrefs(InMemoryUserPreferencesRepository):
        def __init__(self) -> None:
            super().__init__()
            self.contexts: list[str | None] = []

        async def save(self, preferences):
            self.contexts.append(get_current_user_id())
            return await super().save(preferences)

    prefs, owner = _RecordingPrefs(), uuid4()
    svc = UserService(user_repository=InMemoryUserRepository(), preferences_repository=prefs)
    result = await svc.update_preferences(owner, theme="dark")
    assert result.success is True and result.preferences.theme == "dark"
    assert prefs.contexts == [str(owner)]
    assert get_current_user_id() is None

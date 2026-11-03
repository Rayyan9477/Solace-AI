"""Phase B / B-2 (P1): personality _persist_profile must not silently swallow a
save failure and return 200 (REV-36/44 class).

Other services (crisis/therapy adaptation) rely on the persisted profile, so a
silent write-loss leaves them reading a stale/missing profile. /detect's primary
value is the returned scores (persistence is a side effect), so we surface the
failure loudly in production (re-raise) while degrading gracefully in dev/test.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from services.personality_service.src.domain.service import PersonalityOrchestrator


def _orch_with_failing_repo() -> PersonalityOrchestrator:
    repo = MagicMock()
    repo.save_profile = AsyncMock(side_effect=RuntimeError("DB connection lost"))
    orch = PersonalityOrchestrator(repository=repo)
    # Bypass full entity conversion — we only exercise the persist + failure path.
    orch._profile_to_entity = lambda profile: object()
    return orch


@pytest.mark.asyncio
async def test_persist_reraises_in_production(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    orch = _orch_with_failing_repo()
    with pytest.raises(Exception):
        await orch._persist_profile(SimpleNamespace(user_id=uuid4()))


@pytest.mark.asyncio
async def test_persist_tolerates_failure_in_dev(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    orch = _orch_with_failing_repo()
    # Must not raise — /detect stays available in dev/test/demo on a transient blip.
    await orch._persist_profile(SimpleNamespace(user_id=uuid4()))

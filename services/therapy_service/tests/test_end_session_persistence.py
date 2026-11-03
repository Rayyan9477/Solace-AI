"""M1 gate round-2 (P1): therapy end_session must not swallow a persistence failure.

Unfixed sibling of REV-36 (diagnosis). end_session wrapped
``self._unit_of_work.sessions.save(entity)`` in a try/except that logged a warning
and continued — then deleted the in-memory session and returned a successful
SessionEndResult. The caller saw a completed session while the clinical session
record never persisted. The write must fail loud so the caller can retry and the
session isn't dropped from memory on a lie.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from services.therapy_service.src.domain.service import TherapyOrchestrator
from services.therapy_service.src.domain.session_manager import SessionManager
from services.therapy_service.src.domain.technique_selector import TechniqueSelector


@pytest.mark.asyncio
async def test_end_session_propagates_persistence_failure() -> None:
    selector = TechniqueSelector()
    manager = SessionManager()
    uow = MagicMock()
    uow.sessions.save = AsyncMock(side_effect=RuntimeError("DB pool exhausted"))

    orchestrator = TherapyOrchestrator(
        technique_selector=selector,
        session_manager=manager,
        unit_of_work=uow,
    )
    await orchestrator.initialize()
    user_id = uuid4()
    plan_id = uuid4()
    start = await orchestrator.start_session(user_id, plan_id, {})

    # A persistence failure must NOT be swallowed into a successful SessionEndResult.
    with pytest.raises(Exception):
        await orchestrator.end_session(start.session_id, user_id)

    # The in-memory session must survive a failed persist (not deleted on a lie).
    assert manager.get_session(start.session_id) is not None

    await orchestrator.shutdown()

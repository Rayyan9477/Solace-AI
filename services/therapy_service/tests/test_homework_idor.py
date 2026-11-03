"""REV-07 (Sprint A.5): therapy assign_homework must enforce session ownership.

assign_homework loaded the session by session_id and updated its state with no
check that the session belongs to the caller — a user could assign homework into
another user's therapy session (IDOR). The session records its owner.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from services.therapy_service.src.domain.service import TherapyOrchestrator
from services.therapy_service.src.domain.session_manager import SessionManager
from services.therapy_service.src.domain.technique_selector import TechniqueSelector


@pytest.mark.asyncio
async def test_assign_homework_rejects_non_owner() -> None:
    orchestrator = TherapyOrchestrator(
        technique_selector=TechniqueSelector(),
        session_manager=SessionManager(),
    )
    await orchestrator.initialize()
    owner = uuid4()
    started = await orchestrator.start_session(
        owner, uuid4(), {"diagnosis": "Depression"}
    )

    homework = MagicMock()
    homework.homework_id = uuid4()

    attacker = uuid4()
    denied = await orchestrator.assign_homework(
        started.session_id, homework, user_id=attacker
    )
    assert denied is False, (
        "attacker assigned homework to another user's session (IDOR)"
    )

    # The owner can still assign homework to their own session.
    allowed = await orchestrator.assign_homework(
        started.session_id, homework, user_id=owner
    )
    assert allowed is True
    await orchestrator.shutdown()

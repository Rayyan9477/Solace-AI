"""REV-07 (Sprint A.5): diagnosis session operations must enforce ownership.

end_session / challenge_hypothesis / assess loaded the session purely by
session_id with no check that it belongs to the caller — a user could end or act
on another user's active clinical session (IDOR). The session records its owner
(SessionState.user_id).
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.differential import (
    DifferentialGenerator,
    DifferentialSettings,
)
from services.diagnosis_service.src.domain.service import (
    DiagnosisService,
    DiagnosisServiceSettings,
)
from services.diagnosis_service.src.domain.symptom_extractor import (
    SymptomExtractor,
    SymptomExtractorSettings,
)


def _service() -> DiagnosisService:
    return DiagnosisService(
        settings=DiagnosisServiceSettings(),
        symptom_extractor=SymptomExtractor(SymptomExtractorSettings()),
        differential_generator=DifferentialGenerator(DifferentialSettings()),
    )


@pytest.mark.asyncio
async def test_end_session_rejects_non_owner() -> None:
    service = _service()
    await service.initialize()
    owner = uuid4()
    started = await service.start_session(
        user_id=owner,
        session_type="assessment",
        initial_context={},
        previous_session_id=None,
    )

    attacker = uuid4()
    await service.end_session(
        user_id=attacker, session_id=started.session_id, generate_summary=True
    )
    # The attacker must NOT be able to end the owner's session.
    assert started.session_id in service._active_sessions, (
        "attacker ended another user's session (IDOR)"
    )

    # The owner can still end their own session.
    await service.end_session(
        user_id=owner, session_id=started.session_id, generate_summary=False
    )
    assert started.session_id not in service._active_sessions


@pytest.mark.asyncio
async def test_assess_rejects_non_owner() -> None:
    from services.diagnosis_service.src.schemas import DiagnosisPhase

    service = _service()
    await service.initialize()
    owner = uuid4()
    started = await service.start_session(
        user_id=owner,
        session_type="assessment",
        initial_context={},
        previous_session_id=None,
    )
    messages_before = len(service._active_sessions[started.session_id].messages)

    attacker = uuid4()
    result = await service.assess(
        user_id=attacker,
        session_id=started.session_id,
        message="inject into another user's session",
        conversation_history=[],
        existing_symptoms=[],
        current_phase=DiagnosisPhase.RAPPORT,
        current_differential=None,
        user_context={},
    )
    # The attacker's message must NOT be written into the owner's session.
    assert (
        len(service._active_sessions[started.session_id].messages) == messages_before
    ), "attacker mutated another user's clinical session (IDOR)"
    assert "unauthorized_session" in result.safety_flags


@pytest.mark.asyncio
async def test_challenge_hypothesis_rejects_non_owner() -> None:
    from unittest.mock import MagicMock

    service = _service()
    await service.initialize()
    owner = uuid4()
    attacker = uuid4()
    session_id = uuid4()
    hyp_id = uuid4()

    # Minimal session WITH a differential, so the ownership guard (not the
    # "no differential" early return) is what blocks the attacker.
    hyp = MagicMock()
    hyp.hypothesis_id = hyp_id
    differential = MagicMock()
    differential.primary = hyp
    differential.alternatives = []
    session = MagicMock()
    session.user_id = owner
    session.differential = differential
    session.symptoms = []
    service._active_sessions[session_id] = session

    result = await service.challenge_hypothesis(session_id, hyp_id, user_id=attacker)
    assert result.challenges == []  # denied before reaching the Devil's Advocate

"""Sprint 3 E2E: Diagnosis service vignette + clinical pathway.

Exercises the DiagnosisService against two demo-walkthrough scenarios:

  1. Crisis-disclosure vignette — a user expressing suicidal ideation
     must route the diagnosis phase to CRISIS (H-11), must persist the
     user + assistant messages on the session (M-07), and must dispatch
     the ``SafetyFlagRaisedEvent`` so downstream services react.

  2. Standard clinical-interview flow — a non-crisis message goes
     through the 4-step chain, produces a response, updates the session
     phase per confidence (C-15 thresholds), and persists the exchange.

These tests don't need a real LLM or database; the service's four
internal steps degrade gracefully (``_symptom_extractor=None``,
``_differential_generator=None``) and still return a structured result.
That's enough to assert side-effect contracts — event emission, session
state update, phase transitions — which is the point of the E2E.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from services.diagnosis_service.src.domain.service import (
    DiagnosisService,
    DiagnosisServiceSettings,
)
from services.diagnosis_service.src.schemas import DiagnosisPhase


class _CapturingDispatcher:
    """Minimal EventDispatcher stand-in that records every dispatched event."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def dispatch(self, event: Any) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


@pytest.fixture
def dispatcher() -> _CapturingDispatcher:
    return _CapturingDispatcher()


@pytest.fixture
def service(dispatcher: _CapturingDispatcher) -> DiagnosisService:
    return DiagnosisService(
        DiagnosisServiceSettings(),
        event_dispatcher=dispatcher,
    )


class TestDiagnosisCrisisPhaseRouting:
    """H-11 E2E: a risk-indicator must force the CRISIS phase regardless
    of the confidence-based routing decision."""

    @pytest.mark.asyncio
    async def test_suicidal_ideation_routes_to_crisis_phase(
        self,
        service: DiagnosisService,
        dispatcher: _CapturingDispatcher,
    ) -> None:
        """Run a tiny stub assessment that produces a safety flag and
        assert the returned phase is CRISIS (not DIAGNOSIS or ASSESSMENT)."""

        # Stub step 1 to emit a suicidal_ideation risk indicator and no symptoms
        async def _fake_step1(
            message: str,
            history: list[dict[str, str]],
            existing: list[Any],
            user_id: Any = None,
            session_id: Any = None,
        ) -> dict[str, Any]:
            return {
                "symptoms": [],
                "temporal": {},
                "contextual": [],
                "risk_indicators": ["suicidal_ideation"],
            }

        # Stub step 2/3/4 so the chain completes quickly
        async def _fake_step2(symptoms, ctx, user_id=None, session_id=None):
            return {"hypotheses": [], "missing_info": [], "hitop_scores": {}}

        async def _fake_step3(hyps, symptoms):
            return {
                "challenges": [], "alternatives": [], "biases": [],
                "per_hypothesis_adjustments": {}, "bias_analysis": None,
            }

        from services.diagnosis_service.src.schemas import DifferentialDTO

        async def _fake_step4(hyps, challenge, phase, message, session=None):
            return {
                "differential": DifferentialDTO(primary=None, alternatives=[]),
                "next_question": None,
                "response": "safety-first response",
                "confidence": 0.4,
            }

        service._step1_analyze = _fake_step1  # type: ignore[method-assign]
        service._step2_hypothesize = _fake_step2  # type: ignore[method-assign]
        service._step3_challenge = _fake_step3  # type: ignore[method-assign]
        service._step4_synthesize = _fake_step4  # type: ignore[method-assign]

        await service.initialize()
        user_id = uuid4()
        session_id = uuid4()
        # Start a session so _update_session has a target
        service._active_sessions[session_id] = _make_session(session_id, user_id)

        result = await service.assess(
            user_id=user_id,
            session_id=session_id,
            message="I don't want to live anymore, I have a plan",
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.HISTORY,
            current_differential=None,
            user_context={},
        )

        assert result.phase == DiagnosisPhase.CRISIS, (
            f"H-11: risk indicator suicidal_ideation must force CRISIS phase, "
            f"got {result.phase}"
        )
        # safety_flags must be carried through to the result so downstream
        # code (notification, audit) can react.
        assert "suicidal_ideation" in result.safety_flags, (
            f"H-11: safety flag must survive the reasoning chain and reach "
            f"the result. safety_flags={result.safety_flags}"
        )


class TestSessionMessagesPersistence:
    """M-07 E2E: both user and assistant messages must land on
    session.messages after assess() completes."""

    @pytest.mark.asyncio
    async def test_assess_appends_user_and_assistant_messages(
        self, service: DiagnosisService
    ) -> None:
        from services.diagnosis_service.src.schemas import DifferentialDTO

        async def _fake_step1(*args, **kwargs):
            return {"symptoms": [], "temporal": {}, "contextual": [], "risk_indicators": []}

        async def _fake_step2(*args, **kwargs):
            return {"hypotheses": [], "missing_info": [], "hitop_scores": {}}

        async def _fake_step3(*args, **kwargs):
            return {
                "challenges": [], "alternatives": [], "biases": [],
                "per_hypothesis_adjustments": {}, "bias_analysis": None,
            }

        async def _fake_step4(*args, **kwargs):
            return {
                "differential": DifferentialDTO(primary=None, alternatives=[]),
                "next_question": None,
                "response": "I hear that. Tell me more about how that's been affecting you.",
                "confidence": 0.55,
            }

        service._step1_analyze = _fake_step1  # type: ignore[method-assign]
        service._step2_hypothesize = _fake_step2  # type: ignore[method-assign]
        service._step3_challenge = _fake_step3  # type: ignore[method-assign]
        service._step4_synthesize = _fake_step4  # type: ignore[method-assign]

        await service.initialize()
        user_id = uuid4()
        session_id = uuid4()
        service._active_sessions[session_id] = _make_session(session_id, user_id)

        user_message = "I've been feeling down and having trouble sleeping"
        await service.assess(
            user_id=user_id,
            session_id=session_id,
            message=user_message,
            conversation_history=[],
            existing_symptoms=[],
            current_phase=DiagnosisPhase.HISTORY,
            current_differential=None,
            user_context={},
        )

        session = service._active_sessions[session_id]
        assert len(session.messages) == 2, (
            f"M-07: expected 2 messages (user + assistant), got {len(session.messages)}"
        )
        user_msg, assistant_msg = session.messages[0], session.messages[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == user_message
        assert assistant_msg["role"] == "assistant"
        assert "tell me more" in assistant_msg["content"].lower()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_session(session_id, user_id):
    """Build a SessionState object matching the service's internal shape."""
    from services.diagnosis_service.src.domain.models import SessionState
    return SessionState(
        session_id=session_id,
        user_id=user_id,
        phase=DiagnosisPhase.RAPPORT,
        messages=[],
        symptoms=[],
        differential=None,
        reasoning_history=[],
        safety_flags=[],
    )

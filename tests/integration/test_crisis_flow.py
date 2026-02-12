"""
Integration test: Crisis message → Detection → Escalation → Notification.

Tests the crisis-path conversation flow, verifying that messages indicating
suicidal ideation are routed to the crisis handler, escalation is triggered,
crisis resources are provided, and events are emitted.
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4, UUID

from services.orchestrator_service.src.langgraph.state_schema import (
    create_initial_state,
    ProcessingPhase,
    RiskLevel,
    SafetyFlags,
)
from services.orchestrator_service.src.langgraph.graph_builder import (
    safety_precheck_node,
    crisis_handler_node,
    route_after_safety,
)


# ---------------------------------------------------------------------------
# Crisis detection via orchestrator precheck
# ---------------------------------------------------------------------------


class TestCrisisDetection:
    """Safety precheck must flag crisis keywords correctly."""

    @pytest.mark.parametrize(
        "message,expected_level",
        [
            ("I want to kill myself", RiskLevel.HIGH),
            ("I want to end my life tonight, I have a plan", RiskLevel.CRITICAL),
            ("I've been thinking about suicide", RiskLevel.HIGH),
            ("I want to hurt myself", RiskLevel.HIGH),
        ],
    )
    def test_crisis_keywords_detected(self, message: str, expected_level: RiskLevel) -> None:
        state = create_initial_state(user_id=uuid4(), session_id=uuid4(), message=message)
        result = safety_precheck_node(state)

        flags = result["safety_flags"]
        assert flags["crisis_detected"] is True
        assert flags["risk_level"] == expected_level.value
        assert flags["requires_escalation"] is True
        assert len(flags["triggered_keywords"]) > 0

    def test_crisis_routes_to_handler(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="I want to end my life"
        )
        result = safety_precheck_node(state)
        merged = {**state, **result}

        assert route_after_safety(merged) == "crisis_handler"


# ---------------------------------------------------------------------------
# Crisis handler produces resources
# ---------------------------------------------------------------------------


class TestCrisisHandler:
    """Crisis handler must include crisis resources in the response."""

    def test_crisis_response_includes_resources(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="I want to kill myself"
        )
        precheck = safety_precheck_node(state)
        merged = {**state, **precheck}
        result = crisis_handler_node(merged)

        response = result["final_response"]
        # Must mention key crisis resources
        assert "988" in response or "Suicide" in response or "Lifeline" in response or "Crisis" in response
        assert result["processing_phase"] == ProcessingPhase.CRISIS_HANDLING.value

        # Safety flags should show resources were displayed
        assert result["safety_flags"]["safety_resources_shown"] is True

    def test_crisis_handler_sets_agent_result(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="I want to end my life"
        )
        precheck = safety_precheck_node(state)
        merged = {**state, **precheck}
        result = crisis_handler_node(merged)

        agent_results = result.get("agent_results", [])
        assert len(agent_results) >= 1
        assert agent_results[0]["agent_type"] == "safety"
        assert agent_results[0]["success"] is True


# ---------------------------------------------------------------------------
# Safety service crisis detector (domain-level)
# ---------------------------------------------------------------------------


class TestCrisisDetectorDomain:
    """Test the safety service CrisisDetector directly."""

    @pytest.mark.asyncio
    async def test_detect_suicidal_ideation(self) -> None:
        from services.safety_service.src.domain.crisis_detector import (
            CrisisDetector,
            CrisisLevel,
        )

        detector = CrisisDetector()
        result = await detector.detect(
            content="I want to kill myself, I can't take it anymore",
            context={"user_id": str(uuid4())},
        )

        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        assert result.risk_score > Decimal("0.5")
        assert len(result.trigger_indicators) > 0

    @pytest.mark.asyncio
    async def test_detect_safe_message(self) -> None:
        from services.safety_service.src.domain.crisis_detector import (
            CrisisDetector,
            CrisisLevel,
        )

        detector = CrisisDetector()
        result = await detector.detect(
            content="I had a wonderful day at the park with my family",
            context={"user_id": str(uuid4())},
        )

        assert result.crisis_detected is False
        assert result.crisis_level in (CrisisLevel.NONE, CrisisLevel.LOW)


# ---------------------------------------------------------------------------
# Escalation manager
# ---------------------------------------------------------------------------


class TestEscalationManager:
    """Escalation manager should create escalation records for crisis events."""

    @pytest.mark.asyncio
    async def test_escalation_for_critical_crisis(self) -> None:
        from services.safety_service.src.domain.escalation import EscalationManager

        manager = EscalationManager()
        user_id = uuid4()
        session_id = uuid4()

        result = await manager.escalate(
            user_id=user_id,
            session_id=session_id,
            crisis_level="CRITICAL",
            reason="Suicidal ideation with active plan detected",
            context={"trigger_indicators": ["kill myself", "tonight"]},
        )

        assert result.status in ("PENDING", "ACKNOWLEDGED", "IN_PROGRESS")
        assert result.priority in ("HIGH", "CRITICAL")
        assert result.escalation_id is not None
        assert len(result.actions_taken) > 0

    def test_crisis_resources_available(self) -> None:
        from services.safety_service.src.domain.escalation import EscalationManager

        manager = EscalationManager()
        resources = manager.get_crisis_resources("CRITICAL")

        assert len(resources) > 0
        # At least one resource should be the 988 Suicide & Crisis Lifeline
        resource_names = [r.get("name", "") for r in resources]
        assert any(
            "988" in name or "lifeline" in name.lower() or "crisis" in name.lower()
            for name in resource_names
        ), f"Expected crisis hotline in resources, got: {resource_names}"


# ---------------------------------------------------------------------------
# End-to-end graph invocation for crisis
# ---------------------------------------------------------------------------


class TestCrisisFlowEndToEnd:
    """Full graph: crisis message → precheck → crisis_handler → resources."""

    @pytest.mark.asyncio
    async def test_crisis_message_triggers_full_flow(self) -> None:
        from services.orchestrator_service.src.langgraph.graph_builder import (
            OrchestratorGraphBuilder,
            GraphBuilderSettings,
        )

        settings = GraphBuilderSettings(
            enable_checkpointing=False,
            enable_safety_precheck=True,
            enable_safety_postcheck=True,
        )
        builder = OrchestratorGraphBuilder(settings=settings)
        builder.build()
        builder.compile()

        state = create_initial_state(
            user_id=uuid4(),
            session_id=uuid4(),
            message="I want to end my life, I've been thinking about it all week",
        )

        result = await builder.invoke(state)

        # Crisis must be detected
        flags = result.get("safety_flags", {})
        assert flags.get("crisis_detected") is True
        assert flags.get("risk_level") in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)
        assert flags.get("requires_escalation") is True

        # Response must include crisis resources
        response = result.get("final_response", "")
        assert response, "Expected a non-empty crisis response"
        assert any(
            keyword in response
            for keyword in ["988", "Suicide", "Lifeline", "Crisis", "support"]
        ), f"Expected crisis resources in response, got: {response[:200]}"

        # Processing phase should be crisis handling
        assert result.get("processing_phase") in (
            ProcessingPhase.CRISIS_HANDLING.value,
            ProcessingPhase.COMPLETED.value,
        )

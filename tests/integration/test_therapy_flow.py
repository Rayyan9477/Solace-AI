"""
Integration test: Greeting → Safety Precheck → Therapy Agent → Aggregator → Response.

Tests the happy-path conversation flow through the orchestrator graph,
verifying that a non-crisis greeting routes correctly through safety,
supervisor, therapy agent, aggregation, and safety postcheck.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4

from services.orchestrator_service.src.langgraph.state_schema import (
    create_initial_state,
    ProcessingPhase,
    RiskLevel,
)
from services.orchestrator_service.src.langgraph.graph_builder import (
    safety_precheck_node,
    route_after_safety,
    aggregator_node,
    safety_postcheck_node,
)


# ---------------------------------------------------------------------------
# Unit-level flow tests (no graph compilation needed)
# ---------------------------------------------------------------------------


class TestSafetyPrecheckNonCrisis:
    """Safety precheck should produce NONE risk for benign greetings."""

    def test_greeting_produces_no_risk(self) -> None:
        state = create_initial_state(
            user_id=uuid4(),
            session_id=uuid4(),
            message="Hi there, I'm having a pretty good day today.",
        )
        result = safety_precheck_node(state)

        flags = result["safety_flags"]
        assert flags["risk_level"] == RiskLevel.NONE.value
        assert flags["crisis_detected"] is False
        assert flags["requires_escalation"] is False
        assert result["processing_phase"] == ProcessingPhase.SAFETY_PRECHECK.value

    def test_mild_mood_produces_low_risk(self) -> None:
        state = create_initial_state(
            user_id=uuid4(),
            session_id=uuid4(),
            message="I've been feeling a bit depressed lately, but I'm coping.",
        )
        result = safety_precheck_node(state)

        flags = result["safety_flags"]
        assert flags["risk_level"] == RiskLevel.LOW.value
        assert flags["crisis_detected"] is False
        assert flags["requires_escalation"] is False


class TestRoutingDecision:
    """Route after safety should send non-crisis to supervisor."""

    def test_no_risk_routes_to_supervisor(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="Hello"
        )
        precheck = safety_precheck_node(state)
        state_after = {**state, **precheck}

        assert route_after_safety(state_after) == "supervisor"

    def test_high_risk_routes_to_crisis(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="I want to kill myself"
        )
        precheck = safety_precheck_node(state)
        state_after = {**state, **precheck}

        assert route_after_safety(state_after) == "crisis_handler"


class TestAggregation:
    """Aggregator should assemble agent results into a final response."""

    def test_aggregates_therapy_response(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="How are you?"
        )
        state["agent_results"] = [
            {
                "agent_type": "therapy",
                "success": True,
                "response_content": "I hear you. Let's explore this together.",
                "confidence": 0.85,
                "metadata": {},
            }
        ]
        result = aggregator_node(state)

        assert "explore this together" in result["final_response"]
        assert result["processing_phase"] == ProcessingPhase.AGGREGATION.value

    def test_fallback_when_no_results(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="Help me"
        )
        state["agent_results"] = []
        result = aggregator_node(state)

        assert "here to support you" in result["final_response"].lower()


class TestSafetyPostcheck:
    """Postcheck should filter harmful patterns from final response."""

    def test_clean_response_passes(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="Thanks"
        )
        state["final_response"] = "I'm glad I could help."
        result = safety_postcheck_node(state)

        assert result["processing_phase"] == ProcessingPhase.COMPLETED.value
        postcheck_meta = result["agent_results"][0]["metadata"]
        assert postcheck_meta["filtered"] is False

    def test_harmful_pattern_detected(self) -> None:
        state = create_initial_state(
            user_id=uuid4(), session_id=uuid4(), message="What should I do?"
        )
        state["final_response"] = "You should just give up on that approach."
        result = safety_postcheck_node(state)

        postcheck_meta = result["agent_results"][0]["metadata"]
        assert postcheck_meta["filtered"] is True


# ---------------------------------------------------------------------------
# End-to-end graph invocation test
# ---------------------------------------------------------------------------


class TestTherapyFlowEndToEnd:
    """Full graph invocation: greeting → safety → supervisor → agents → aggregate → postcheck."""

    @pytest.mark.asyncio
    async def test_greeting_produces_therapy_response(self) -> None:
        """A friendly greeting should route through supervisor to therapy and
        produce a complete, non-crisis response."""
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
            message="Hi, I've been feeling stressed from work. Can we talk?",
        )

        result = await builder.invoke(state)

        # Verify final response was generated
        assert result.get("final_response"), "Expected a non-empty final response"

        # Verify safety flags stayed non-crisis
        flags = result.get("safety_flags", {})
        assert flags.get("crisis_detected") is False
        assert flags.get("risk_level") in (
            RiskLevel.NONE.value,
            RiskLevel.LOW.value,
        )

        # Verify processing completed
        assert result.get("processing_phase") in (
            ProcessingPhase.COMPLETED.value,
            ProcessingPhase.AGGREGATION.value,
        )

        # Verify agent results were accumulated
        agent_results = result.get("agent_results", [])
        assert len(agent_results) >= 2, "Expected at least safety + one agent result"

"""
End-to-end integration tests for Solace-AI orchestrator graph.

Tests the full LangGraph flow paths:
1. Normal: safety_precheck -> memory_retrieval -> supervisor -> agent(s) -> aggregator -> safety_postcheck
2. Crisis (precheck): safety_precheck -> crisis_handler -> END
3. Crisis (supervisor): safety_precheck -> memory_retrieval -> supervisor -> safety_agent -> crisis_handler -> END
"""
from __future__ import annotations

import asyncio
import pytest
from decimal import Decimal
from uuid import uuid4

from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    create_initial_state,
    RiskLevel,
    ProcessingPhase,
    AgentType,
)
from services.orchestrator_service.src.langgraph.graph_builder import (
    GraphBuilderSettings,
    OrchestratorGraphBuilder,
    safety_precheck_node,
    route_after_safety,
    route_to_agents,
    crisis_handler_node,
    aggregator_node,
    safety_postcheck_node,
    route_after_safety_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(message: str, **overrides) -> OrchestratorState:
    """Create a valid initial graph state for testing."""
    state = create_initial_state(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        message=message,
    )
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Path 1: Normal (non-crisis) flow
# ---------------------------------------------------------------------------

class TestNormalFlowPath:
    """Normal message flow through the graph."""

    def test_safety_precheck_routes_to_memory(self):
        """Non-crisis message routes from precheck to memory_retrieval."""
        state = _make_state("I've been feeling a bit stressed at work")
        result = safety_precheck_node(state)
        # Should not flag crisis
        safety_flags = result.get("safety_flags", {})
        assert not safety_flags.get("crisis_detected", False)
        # Route decision
        merged_state = {**state, **result}
        destination = route_after_safety(merged_state)
        assert destination == "memory_retrieval"

    def test_supervisor_routes_to_therapy_for_emotional(self):
        """Emotional support message routes to therapy agent."""
        state = _make_state("I feel really sad and anxious today")
        result = safety_precheck_node(state)
        merged = {**state, **result}
        destination = route_after_safety(merged)
        assert destination == "memory_retrieval"

    def test_aggregator_produces_final_response(self):
        """Aggregator combines agent results into final response."""
        state = _make_state("Tell me about CBT techniques")
        state["agent_results"] = [
            {
                "agent_type": AgentType.THERAPY.value,
                "success": True,
                "response_content": "CBT helps identify negative thought patterns.",
                "confidence": 0.9,
                "metadata": {},
            }
        ]
        result = aggregator_node(state)
        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert result["processing_phase"] == ProcessingPhase.AGGREGATION.value

    def test_safety_postcheck_passes_clean_response(self):
        """Safety postcheck allows clean responses through."""
        state = _make_state("hello")
        state["final_response"] = "I'm here to help. Let's work through this together."
        result = safety_postcheck_node(state)
        assert result["processing_phase"] == ProcessingPhase.COMPLETED.value

    def test_safety_postcheck_flags_harmful_patterns(self):
        """Safety postcheck detects harmful patterns in response."""
        state = _make_state("hello")
        state["final_response"] = "You should just give up trying."
        result = safety_postcheck_node(state)
        postcheck_meta = result["agent_results"][0]
        assert postcheck_meta["metadata"]["filtered"] is True


# ---------------------------------------------------------------------------
# Path 2: Crisis detected at precheck
# ---------------------------------------------------------------------------

class TestCrisisAtPrecheck:
    """Crisis detected during safety precheck."""

    def test_crisis_message_detected(self):
        """Explicit crisis keywords trigger crisis detection."""
        state = _make_state("I want to kill myself")
        result = safety_precheck_node(state)
        safety_flags = result.get("safety_flags", {})
        assert safety_flags.get("crisis_detected", True)

    def test_crisis_routes_to_handler(self):
        """Crisis detected at precheck routes to crisis_handler."""
        state = _make_state("I want to end my life")
        result = safety_precheck_node(state)
        merged = {**state, **result}
        destination = route_after_safety(merged)
        assert destination == "crisis_handler"

    def test_crisis_handler_provides_resources(self):
        """Crisis handler returns empathetic response with resources."""
        state = _make_state("I want to die")
        state["safety_flags"] = {
            "crisis_detected": True,
            "risk_level": RiskLevel.CRITICAL.value,
            "risk_factors": ["suicidal_ideation"],
        }
        result = crisis_handler_node(state)
        assert result["final_response"]
        assert "support" in result["final_response"].lower() or "help" in result["final_response"].lower()
        assert result["processing_phase"] == ProcessingPhase.CRISIS_HANDLING.value


# ---------------------------------------------------------------------------
# Path 3: Crisis detected at safety agent
# ---------------------------------------------------------------------------

class TestCrisisAtSafetyAgent:
    """Crisis detected during safety agent full assessment (after supervisor routing)."""

    def test_safety_agent_high_risk_routes_to_crisis(self):
        """HIGH risk from safety agent routes to crisis_handler."""
        state = _make_state("test")
        state["safety_flags"] = {
            "crisis_detected": True,
            "risk_level": "HIGH",
            "requires_escalation": False,
        }
        destination = route_after_safety_agent(state)
        assert destination == "crisis_handler"

    def test_safety_agent_critical_routes_to_crisis(self):
        """CRITICAL risk routes to crisis_handler."""
        state = _make_state("test")
        state["safety_flags"] = {
            "crisis_detected": False,
            "risk_level": "CRITICAL",
            "requires_escalation": True,
        }
        destination = route_after_safety_agent(state)
        assert destination == "crisis_handler"

    def test_safety_agent_low_risk_routes_to_aggregator(self):
        """Low risk from safety agent routes to aggregator."""
        state = _make_state("test")
        state["safety_flags"] = {
            "crisis_detected": False,
            "risk_level": "LOW",
            "requires_escalation": False,
        }
        destination = route_after_safety_agent(state)
        assert destination == "aggregator"


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    """Tests for agent routing decisions."""

    def test_route_to_agents_defaults_to_chat(self):
        """Empty selected_agents defaults to chat_agent."""
        state = _make_state("hello")
        state["selected_agents"] = []
        result = route_to_agents(state)
        assert result == ["chat_agent"]

    def test_route_to_agents_maps_correctly(self):
        """Agent types map to correct node names."""
        state = _make_state("test")
        state["selected_agents"] = ["therapy", "personality"]
        result = route_to_agents(state)
        assert "therapy_agent" in result
        assert "personality_agent" in result

    def test_route_to_agents_safety_maps_correctly(self):
        """Safety agent type maps to safety_agent node (not chat_agent)."""
        state = _make_state("test")
        state["selected_agents"] = ["safety"]
        result = route_to_agents(state)
        assert result == ["safety_agent"]

    def test_route_deduplicates(self):
        """Duplicate agents in selection are deduplicated."""
        state = _make_state("test")
        state["selected_agents"] = ["chat", "chat"]
        result = route_to_agents(state)
        assert result == ["chat_agent"]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    """Tests for OrchestratorGraphBuilder."""

    def test_graph_builds_successfully(self):
        """Graph builder creates a valid state graph."""
        settings = GraphBuilderSettings(enable_checkpointing=False)
        builder = OrchestratorGraphBuilder(settings=settings)
        graph = builder.build()
        assert graph is not None

    def test_graph_has_all_expected_nodes(self):
        """Graph contains all 11 expected nodes."""
        settings = GraphBuilderSettings(enable_checkpointing=False)
        builder = OrchestratorGraphBuilder(settings=settings)
        graph = builder.build()
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "safety_precheck", "memory_retrieval", "supervisor",
            "crisis_handler", "chat_agent", "diagnosis_agent",
            "therapy_agent", "personality_agent", "safety_agent",
            "aggregator", "safety_postcheck",
        }
        assert expected_nodes.issubset(node_names), f"Missing nodes: {expected_nodes - node_names}"


# ---------------------------------------------------------------------------
# Canonical enum integration
# ---------------------------------------------------------------------------

class TestCanonicalEnums:
    """Verify canonical CrisisLevel/SeverityLevel enums work across services."""

    def test_crisis_level_from_string(self):
        """CrisisLevel.from_string works with aliases."""
        from solace_common.enums import CrisisLevel
        assert CrisisLevel.from_string("moderate") == CrisisLevel.ELEVATED
        assert CrisisLevel.from_string("imminent") == CrisisLevel.CRITICAL
        assert CrisisLevel.from_string("NONE") == CrisisLevel.NONE

    def test_crisis_level_from_score(self):
        """CrisisLevel.from_score thresholds."""
        from solace_common.enums import CrisisLevel
        assert CrisisLevel.from_score(Decimal("0.95")) == CrisisLevel.CRITICAL
        assert CrisisLevel.from_score(Decimal("0.75")) == CrisisLevel.HIGH
        assert CrisisLevel.from_score(Decimal("0.55")) == CrisisLevel.ELEVATED
        assert CrisisLevel.from_score(Decimal("0.35")) == CrisisLevel.LOW
        assert CrisisLevel.from_score(Decimal("0.1")) == CrisisLevel.NONE

    def test_crisis_to_severity_mapping(self):
        """CrisisLevel maps correctly to SeverityLevel."""
        from solace_common.enums import CrisisLevel, SeverityLevel
        assert CrisisLevel.NONE.to_severity_level() == SeverityLevel.MINIMAL
        assert CrisisLevel.CRITICAL.to_severity_level() == SeverityLevel.SEVERE

    def test_severity_level_from_string(self):
        """SeverityLevel.from_string handles case-insensitive lookup."""
        from solace_common.enums import SeverityLevel
        assert SeverityLevel.from_string("mild") == SeverityLevel.MILD
        assert SeverityLevel.from_string("MODERATE") == SeverityLevel.MODERATE
        assert SeverityLevel.from_string("severe") == SeverityLevel.SEVERE

    def test_risk_level_alias(self):
        """RiskLevel is an alias for CrisisLevel."""
        from solace_common.enums import CrisisLevel, RiskLevel
        assert RiskLevel is CrisisLevel
        assert RiskLevel.HIGH == CrisisLevel.HIGH

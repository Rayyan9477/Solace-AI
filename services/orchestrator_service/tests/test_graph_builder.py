"""
Unit tests for Orchestrator Service Graph Builder.
Tests LangGraph state machine construction and execution.
"""
from __future__ import annotations
from uuid import uuid4
import pytest

from services.orchestrator_service.src.langgraph.graph_builder import (
    GraphBuilderSettings,
    OrchestratorGraphBuilder,
    safety_precheck_node,
    crisis_handler_node,
    chat_agent_node,
    diagnosis_agent_node,
    therapy_agent_node,
    personality_agent_node,
    aggregator_node,
    safety_postcheck_node,
    route_after_safety,
    route_to_agents,
)
from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    create_initial_state,
    RiskLevel,
    ProcessingPhase,
    AgentType,
)


class TestGraphBuilderSettings:
    """Tests for GraphBuilderSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default graph builder settings."""
        settings = GraphBuilderSettings()
        assert settings.enable_checkpointing is True
        assert settings.enable_safety_precheck is True
        assert settings.enable_safety_postcheck is True
        assert settings.enable_parallel_processing is True
        assert settings.max_iterations == 10

    def test_custom_settings(self) -> None:
        """Test custom graph builder settings."""
        settings = GraphBuilderSettings(
            enable_checkpointing=False,
            max_iterations=5,
            safety_timeout_ms=3000,
        )
        assert settings.enable_checkpointing is False
        assert settings.max_iterations == 5
        assert settings.safety_timeout_ms == 3000


class TestSafetyPrecheckNode:
    """Tests for safety precheck node."""

    def test_safe_message(self) -> None:
        """Test safety precheck with safe message."""
        state = create_initial_state("user-1", "session-1", "Hello, how are you?")
        updates = safety_precheck_node(state)
        assert updates["processing_phase"] == ProcessingPhase.SAFETY_PRECHECK.value
        assert updates["safety_flags"]["crisis_detected"] is False
        assert updates["safety_flags"]["risk_level"] == "none"

    def test_crisis_message(self) -> None:
        """Test safety precheck with crisis message."""
        state = create_initial_state("user-1", "session-1", "I want to kill myself")
        updates = safety_precheck_node(state)
        assert updates["safety_flags"]["crisis_detected"] is True
        assert updates["safety_flags"]["risk_level"] in ("high", "critical")
        assert "kill myself" in updates["safety_flags"]["triggered_keywords"]

    def test_critical_crisis_message(self) -> None:
        """Test safety precheck with critical crisis indicators."""
        state = create_initial_state("user-1", "session-1", "I'm going to end my life tonight")
        updates = safety_precheck_node(state)
        assert updates["safety_flags"]["risk_level"] == "critical"
        assert updates["safety_flags"]["requires_escalation"] is True

    def test_mild_distress_message(self) -> None:
        """Test safety precheck with mild distress indicators."""
        state = create_initial_state("user-1", "session-1", "I've been feeling depressed lately")
        updates = safety_precheck_node(state)
        assert updates["safety_flags"]["crisis_detected"] is False
        assert updates["safety_flags"]["risk_level"] == "low"

    def test_adds_agent_result(self) -> None:
        """Test that safety precheck adds agent result."""
        state = create_initial_state("user-1", "session-1", "Hello")
        updates = safety_precheck_node(state)
        assert len(updates["agent_results"]) == 1
        assert updates["agent_results"][0]["agent_type"] == "safety"


class TestCrisisHandlerNode:
    """Tests for crisis handler node."""

    def test_crisis_response_content(self) -> None:
        """Test crisis handler generates appropriate response."""
        state = create_initial_state("user-1", "session-1", "I want to hurt myself")
        state["safety_flags"] = {"risk_level": "high", "crisis_detected": True}
        updates = crisis_handler_node(state)
        assert updates["processing_phase"] == ProcessingPhase.CRISIS_HANDLING.value
        assert "988" in updates["final_response"]
        assert "crisis" in updates["final_response"].lower() or "support" in updates["final_response"].lower()

    def test_crisis_shows_resources(self) -> None:
        """Test crisis handler shows safety resources."""
        state = create_initial_state("user-1", "session-1", "I don't want to live anymore")
        state["safety_flags"] = {"risk_level": "critical"}
        updates = crisis_handler_node(state)
        assert updates["safety_flags"]["safety_resources_shown"] is True

    def test_crisis_adds_message(self) -> None:
        """Test crisis handler adds response message."""
        state = create_initial_state("user-1", "session-1", "suicide")
        state["safety_flags"] = {"risk_level": "high"}
        updates = crisis_handler_node(state)
        assert len(updates["messages"]) == 1
        assert updates["messages"][0]["role"] == "assistant"


class TestAgentNodes:
    """Tests for individual agent nodes."""

    def test_chat_agent_node(self) -> None:
        """Test chat agent node generates response."""
        state = create_initial_state("user-1", "session-1", "Hello!")
        updates = chat_agent_node(state)
        assert "agent_results" in updates
        assert updates["agent_results"][0]["agent_type"] == "chat"
        assert updates["agent_results"][0]["response_content"] is not None

    def test_chat_agent_with_personality(self) -> None:
        """Test chat agent adapts to personality style."""
        state = create_initial_state("user-1", "session-1", "Hello!")
        state["personality_style"] = {"warmth": 0.9}
        updates = chat_agent_node(state)
        response = updates["agent_results"][0]["response_content"]
        assert "thank you" in response.lower() or "sharing" in response.lower()

    def test_diagnosis_agent_node(self) -> None:
        """Test diagnosis agent node generates response."""
        state = create_initial_state("user-1", "session-1", "I haven't been sleeping well")
        updates = diagnosis_agent_node(state)
        assert updates["agent_results"][0]["agent_type"] == "diagnosis"
        assert updates["agent_results"][0]["success"] is True

    def test_therapy_agent_node(self) -> None:
        """Test therapy agent node generates response."""
        state = create_initial_state("user-1", "session-1", "I need help coping")
        updates = therapy_agent_node(state)
        assert updates["agent_results"][0]["agent_type"] == "therapy"
        assert updates["agent_results"][0]["metadata"]["modality"] == "ACT"

    def test_personality_agent_node(self) -> None:
        """Test personality agent node generates style."""
        state = create_initial_state("user-1", "session-1", "Hello")
        updates = personality_agent_node(state)
        assert "personality_style" in updates
        assert "warmth" in updates["personality_style"]
        assert updates["agent_results"][0]["agent_type"] == "personality"


class TestAggregatorNode:
    """Tests for aggregator node."""

    def test_aggregates_single_response(self) -> None:
        """Test aggregator with single agent response."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["agent_results"] = [
            {"agent_type": "chat", "response_content": "Hi there!", "success": True}
        ]
        updates = aggregator_node(state)
        assert updates["final_response"] == "Hi there!"
        assert updates["processing_phase"] == ProcessingPhase.AGGREGATION.value

    def test_aggregates_multiple_responses(self) -> None:
        """Test aggregator with multiple agent responses."""
        state = create_initial_state("user-1", "session-1", "I feel anxious")
        state["agent_results"] = [
            {"agent_type": "diagnosis", "response_content": "Assessment response", "success": True},
            {"agent_type": "therapy", "response_content": "Therapy response", "success": True},
        ]
        updates = aggregator_node(state)
        assert updates["final_response"] == "Therapy response"

    def test_aggregates_with_personality_warmth(self) -> None:
        """Test aggregator applies personality warmth."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["agent_results"] = [{"agent_type": "chat", "response_content": "Response", "success": True}]
        state["personality_style"] = {"warmth": 0.9}
        updates = aggregator_node(state)
        assert "appreciate" in updates["final_response"].lower()

    def test_aggregates_handles_empty_results(self) -> None:
        """Test aggregator handles empty results."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["agent_results"] = []
        updates = aggregator_node(state)
        assert "here to support" in updates["final_response"].lower()


class TestSafetyPostcheckNode:
    """Tests for safety postcheck node."""

    def test_safe_response(self) -> None:
        """Test postcheck with safe response."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["final_response"] = "I'm here to help you today."
        updates = safety_postcheck_node(state)
        assert updates["processing_phase"] == ProcessingPhase.COMPLETED.value

    def test_response_with_harmful_pattern(self) -> None:
        """Test postcheck detects potentially harmful patterns."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["final_response"] = "You should just give up on that."
        updates = safety_postcheck_node(state)
        assert updates["agent_results"][0]["metadata"]["filtered"] is True


class TestRoutingFunctions:
    """Tests for routing functions."""

    def test_route_after_safety_crisis(self) -> None:
        """Test routing to crisis handler after safety check."""
        state = create_initial_state("user-1", "session-1", "I want to die")
        state["safety_flags"] = {"crisis_detected": True, "risk_level": "high"}
        route = route_after_safety(state)
        assert route == "crisis_handler"

    def test_route_after_safety_safe(self) -> None:
        """Test routing to supervisor after safe check."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["safety_flags"] = {"crisis_detected": False, "risk_level": "none"}
        route = route_after_safety(state)
        assert route == "supervisor"

    def test_route_to_agents_default(self) -> None:
        """Test routing to agents with no selection defaults to chat."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["selected_agents"] = []
        routes = route_to_agents(state)
        assert "chat_agent" in routes

    def test_route_to_agents_therapy(self) -> None:
        """Test routing to therapy agent."""
        state = create_initial_state("user-1", "session-1", "I need help")
        state["selected_agents"] = ["therapy", "personality"]
        routes = route_to_agents(state)
        assert "therapy_agent" in routes
        assert "personality_agent" in routes

    def test_route_to_agents_all(self) -> None:
        """Test routing to multiple agents."""
        state = create_initial_state("user-1", "session-1", "Help with symptoms")
        state["selected_agents"] = ["diagnosis", "therapy", "chat"]
        routes = route_to_agents(state)
        assert "diagnosis_agent" in routes
        assert "therapy_agent" in routes
        assert "chat_agent" in routes


class TestOrchestratorGraphBuilder:
    """Tests for OrchestratorGraphBuilder class."""

    def test_build_graph(self) -> None:
        """Test building the orchestrator graph."""
        builder = OrchestratorGraphBuilder()
        graph = builder.build()
        assert graph is not None

    def test_compile_graph(self) -> None:
        """Test compiling the orchestrator graph."""
        builder = OrchestratorGraphBuilder()
        builder.build()
        compiled = builder.compile()
        assert compiled is not None

    def test_compile_with_checkpointer(self) -> None:
        """Test compiling with checkpointer enabled."""
        settings = GraphBuilderSettings(enable_checkpointing=True)
        builder = OrchestratorGraphBuilder(settings)
        builder.build()
        compiled = builder.compile()
        checkpointer = builder.get_checkpointer()
        assert checkpointer is not None

    def test_compile_without_checkpointer(self) -> None:
        """Test compiling without checkpointer."""
        settings = GraphBuilderSettings(enable_checkpointing=False)
        builder = OrchestratorGraphBuilder(settings)
        builder.build()
        compiled = builder.compile()
        checkpointer = builder.get_checkpointer()
        assert checkpointer is None

    def test_get_compiled_graph_lazy(self) -> None:
        """Test getting compiled graph builds if needed."""
        builder = OrchestratorGraphBuilder()
        compiled = builder.get_compiled_graph()
        assert compiled is not None

    def test_invoke_sync(self) -> None:
        """Test synchronous graph invocation."""
        builder = OrchestratorGraphBuilder()
        builder.compile()
        state = create_initial_state("user-1", "session-1", "Hello, how are you?")
        result = builder.invoke_sync(state)
        assert result["final_response"] != ""
        assert result["processing_phase"] in ("completed", "crisis_handling")

    def test_invoke_sync_crisis(self) -> None:
        """Test synchronous invocation with crisis message."""
        builder = OrchestratorGraphBuilder()
        builder.compile()
        state = create_initial_state("user-1", "session-1", "I want to kill myself")
        result = builder.invoke_sync(state)
        assert result["processing_phase"] == "crisis_handling"
        assert "988" in result["final_response"]

    def test_invoke_with_thread_id(self) -> None:
        """Test invocation with explicit thread ID."""
        builder = OrchestratorGraphBuilder()
        builder.compile()
        thread_id = "test-thread-123"
        state = create_initial_state("user-1", "session-1", "Hello")
        result = builder.invoke_sync(state, thread_id=thread_id)
        assert result["final_response"] != ""


class TestGraphBuilderIntegration:
    """Integration tests for full graph execution."""

    @pytest.fixture
    def builder(self) -> OrchestratorGraphBuilder:
        """Create graph builder for tests."""
        builder = OrchestratorGraphBuilder()
        builder.compile()
        return builder

    def test_full_flow_general_chat(self, builder: OrchestratorGraphBuilder) -> None:
        """Test full flow for general chat message."""
        state = create_initial_state("user-1", "session-1", "Hello, nice to meet you!")
        result = builder.invoke_sync(state)
        assert result["processing_phase"] == "completed"
        assert result["intent"] == "general_chat"
        assert result["final_response"] != ""

    def test_full_flow_emotional_support(self, builder: OrchestratorGraphBuilder) -> None:
        """Test full flow for emotional support message."""
        state = create_initial_state("user-1", "session-1", "I've been feeling really depressed and anxious")
        result = builder.invoke_sync(state)
        assert result["processing_phase"] == "completed"
        assert result["intent"] == "emotional_support"

    def test_full_flow_treatment_inquiry(self, builder: OrchestratorGraphBuilder) -> None:
        """Test full flow for treatment inquiry."""
        state = create_initial_state("user-1", "session-1", "What therapy techniques help with anxiety?")
        result = builder.invoke_sync(state)
        assert result["processing_phase"] == "completed"
        assert result["intent"] == "treatment_inquiry"

    def test_full_flow_maintains_safety_flags(self, builder: OrchestratorGraphBuilder) -> None:
        """Test that safety flags are maintained through flow."""
        state = create_initial_state("user-1", "session-1", "I feel depressed")
        result = builder.invoke_sync(state)
        assert "safety_flags" in result
        assert "risk_level" in result["safety_flags"]

    def test_full_flow_records_agent_results(self, builder: OrchestratorGraphBuilder) -> None:
        """Test that agent results are recorded through flow."""
        state = create_initial_state("user-1", "session-1", "Hello there")
        result = builder.invoke_sync(state)
        assert len(result["agent_results"]) > 0
        agent_types = [r["agent_type"] for r in result["agent_results"]]
        assert "safety" in agent_types
        assert "supervisor" in agent_types

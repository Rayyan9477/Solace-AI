"""
Solace-AI Orchestrator Service - Response Pipeline Tests.
Tests for aggregator module.
"""
import pytest

from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    IntentType,
    ProcessingPhase,
    RiskLevel,
    AgentResult,
    SafetyFlags,
    create_initial_state,
)
from services.orchestrator_service.src.langgraph.aggregator import (
    Aggregator,
    AggregatorSettings,
    AggregationStrategy,
    AgentContribution,
    AggregationResult,
    ResponseRanker,
    ResponseMerger,
    aggregator_node,
)


class TestAggregatorSettings:
    """Tests for AggregatorSettings."""

    def test_default_settings(self):
        settings = AggregatorSettings()
        assert settings.strategy == "priority_based"
        assert settings.confidence_threshold == 0.5
        assert settings.max_response_length == 1500


class TestResponseRanker:
    """Tests for ResponseRanker."""

    def test_rank_results(self):
        settings = AggregatorSettings()
        ranker = ResponseRanker(settings)
        results = [
            {"agent_type": "chat", "response_content": "Chat response", "confidence": 0.7},
            {"agent_type": "therapy", "response_content": "Therapy response", "confidence": 0.8},
        ]
        ranked = ranker.rank(results)
        assert len(ranked) == 2
        assert ranked[0].agent_type == AgentType.THERAPY

    def test_rank_empty_content_filtered(self):
        settings = AggregatorSettings()
        ranker = ResponseRanker(settings)
        results = [
            {"agent_type": "chat", "response_content": None, "confidence": 0.7},
            {"agent_type": "therapy", "response_content": "Content", "confidence": 0.8},
        ]
        ranked = ranker.rank(results)
        assert len(ranked) == 1


class TestResponseMerger:
    """Tests for ResponseMerger."""

    def test_merge_single_contribution(self):
        settings = AggregatorSettings()
        merger = ResponseMerger(settings)
        contributions = [
            AgentContribution(
                agent_type=AgentType.THERAPY,
                content="Therapy response",
                confidence=0.8,
                priority_score=0.9,
            ),
        ]
        result = merger.merge(contributions, AggregationStrategy.PRIORITY_BASED)
        assert result == "Therapy response"

    def test_merge_empty_contributions(self):
        settings = AggregatorSettings()
        merger = ResponseMerger(settings)
        result = merger.merge([], AggregationStrategy.FIRST_SUCCESS)
        assert result == settings.fallback_response


class TestAgentContribution:
    """Tests for AgentContribution."""

    def test_contribution_to_dict(self):
        contribution = AgentContribution(
            agent_type=AgentType.THERAPY,
            content="Test content",
            confidence=0.8,
            priority_score=0.9,
        )
        result = contribution.to_dict()
        assert result["agent_type"] == "therapy"
        assert result["confidence"] == 0.8


class TestAggregator:
    """Tests for Aggregator."""

    def test_aggregator_process(self):
        aggregator = Aggregator()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["agent_results"] = [
            {"agent_type": "chat", "response_content": "Hello!", "confidence": 0.7},
        ]
        result = aggregator.aggregate(state)
        assert "final_response" in result
        assert result["processing_phase"] == ProcessingPhase.AGGREGATION.value

    def test_aggregator_fallback(self):
        aggregator = Aggregator()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["agent_results"] = []
        result = aggregator.aggregate(state)
        assert result["final_response"] == AggregatorSettings().fallback_response


class TestAggregatorNode:
    """Tests for aggregator_node function."""

    def test_aggregator_node_function(self):
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["agent_results"] = [
            {"agent_type": "chat", "response_content": "Response", "confidence": 0.7},
        ]
        result = aggregator_node(state)
        assert "final_response" in result


class TestResponsePipelineIntegration:
    """Integration tests for the response pipeline."""

    def _classify_intent(self, state):
        from services.orchestrator_service.src.langgraph.supervisor import IntentClassifier, SupervisorSettings
        classifier = IntentClassifier(SupervisorSettings())
        message = state.get("current_message", "")
        intent, confidence, keywords = classifier.classify(message)
        return {
            "intent": intent.value,
            "intent_confidence": confidence,
        }

    def test_full_pipeline_general_chat(self):
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello, how are you?",
        )
        route_result = self._classify_intent(state)
        state.update(route_result)
        state["agent_results"] = [
            {"agent_type": "chat", "response_content": "Hello! I'm doing well.", "confidence": 0.8},
        ]
        aggregator = Aggregator()
        agg_result = aggregator.aggregate(state)
        state.update(agg_result)
        assert "final_response" in agg_result
        assert len(agg_result["final_response"]) > 0

    def test_full_pipeline_crisis(self):
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="I want to end my life",
        )
        route_result = self._classify_intent(state)
        assert route_result["intent"] == IntentType.CRISIS_DISCLOSURE.value
        state.update(route_result)
        state["safety_flags"] = {
            "risk_level": "CRITICAL",
            "crisis_detected": True,
            "safety_resources_shown": False,
        }
        state["agent_results"] = [
            {
                "agent_type": "safety",
                "response_content": "I'm very concerned about what you're sharing.",
                "confidence": 1.0,
            },
        ]
        aggregator = Aggregator()
        agg_result = aggregator.aggregate(state)
        assert "final_response" in agg_result

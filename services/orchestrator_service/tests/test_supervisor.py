"""
Unit tests for Orchestrator Service Supervisor Agent.
Tests intent classification, agent routing, and supervisor decisions.
"""
from __future__ import annotations
from uuid import uuid4
import pytest

from services.orchestrator_service.src.langgraph.supervisor import (
    SupervisorSettings,
    SupervisorDecision,
    IntentClassifier,
    AgentRouter,
    SupervisorAgent,
    supervisor_node,
)
from services.orchestrator_service.src.langgraph.state_schema import (
    IntentType,
    AgentType,
    RiskLevel,
    create_initial_state,
)


class TestSupervisorSettings:
    """Tests for SupervisorSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default supervisor settings."""
        settings = SupervisorSettings()
        assert settings.intent_confidence_threshold == 0.6
        assert settings.max_parallel_agents == 4
        assert settings.enable_personality_adaptation is True
        assert settings.enable_diagnosis_agent is True
        assert settings.enable_therapy_agent is True

    def test_crisis_keywords_exist(self) -> None:
        """Test crisis keywords are defined."""
        settings = SupervisorSettings()
        assert len(settings.crisis_keywords) > 0
        assert "suicide" in settings.crisis_keywords

    def test_emotional_keywords_exist(self) -> None:
        """Test emotional keywords are defined."""
        settings = SupervisorSettings()
        assert len(settings.emotional_keywords) > 0
        assert "depressed" in settings.emotional_keywords


class TestSupervisorDecision:
    """Tests for SupervisorDecision dataclass."""

    def test_create_decision(self) -> None:
        """Test creating a supervisor decision."""
        decision = SupervisorDecision(
            intent=IntentType.EMOTIONAL_SUPPORT,
            confidence=0.85,
            selected_agents=[AgentType.THERAPY, AgentType.PERSONALITY],
            routing_reason="Emotional support request detected",
        )
        assert decision.intent == IntentType.EMOTIONAL_SUPPORT
        assert decision.confidence == 0.85
        assert len(decision.selected_agents) == 2
        assert decision.requires_safety_override is False
        assert decision.processing_priority == "normal"

    def test_decision_with_safety_override(self) -> None:
        """Test decision with safety override."""
        decision = SupervisorDecision(
            intent=IntentType.CRISIS_DISCLOSURE,
            confidence=0.95,
            selected_agents=[AgentType.SAFETY],
            routing_reason="Crisis detected",
            requires_safety_override=True,
            processing_priority="critical",
        )
        assert decision.requires_safety_override is True
        assert decision.processing_priority == "critical"

    def test_decision_to_dict(self) -> None:
        """Test decision serialization."""
        decision = SupervisorDecision(
            intent=IntentType.TREATMENT_INQUIRY,
            confidence=0.75,
            selected_agents=[AgentType.THERAPY],
            routing_reason="Treatment inquiry",
            metadata={"matched_keywords": ["therapy"]},
        )
        data = decision.to_dict()
        assert data["intent"] == "treatment_inquiry"
        assert data["confidence"] == 0.75
        assert "therapy" in data["selected_agents"]
        assert data["metadata"]["matched_keywords"] == ["therapy"]


class TestIntentClassifier:
    """Tests for IntentClassifier class."""

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create intent classifier for tests."""
        return IntentClassifier(SupervisorSettings())

    def test_classify_crisis_message(self, classifier: IntentClassifier) -> None:
        """Test classifying crisis message."""
        intent, confidence, keywords = classifier.classify("I want to kill myself")
        assert intent == IntentType.CRISIS_DISCLOSURE
        assert confidence >= 0.9
        assert "kill myself" in keywords

    def test_classify_suicide_mention(self, classifier: IntentClassifier) -> None:
        """Test classifying suicide mention."""
        intent, confidence, keywords = classifier.classify("I've been thinking about suicide")
        assert intent == IntentType.CRISIS_DISCLOSURE
        assert "suicide" in keywords

    def test_classify_emotional_support(self, classifier: IntentClassifier) -> None:
        """Test classifying emotional support request."""
        intent, confidence, keywords = classifier.classify("I've been feeling really depressed and anxious lately")
        assert intent == IntentType.EMOTIONAL_SUPPORT
        assert confidence >= 0.5
        assert "depressed" in keywords or "anxious" in keywords

    def test_classify_symptom_discussion(self, classifier: IntentClassifier) -> None:
        """Test classifying symptom discussion."""
        intent, confidence, keywords = classifier.classify("I haven't been sleeping well and my appetite is gone")
        assert intent == IntentType.SYMPTOM_DISCUSSION
        assert "sleep" in keywords or "appetite" in keywords

    def test_classify_treatment_inquiry(self, classifier: IntentClassifier) -> None:
        """Test classifying treatment inquiry."""
        intent, confidence, keywords = classifier.classify("What therapy techniques can help with anxiety?")
        assert intent == IntentType.TREATMENT_INQUIRY
        assert "therapy" in keywords or "technique" in keywords

    def test_classify_coping_request(self, classifier: IntentClassifier) -> None:
        """Test classifying coping strategy request."""
        intent, confidence, keywords = classifier.classify("How can I cope when I feel overwhelmed?")
        assert intent == IntentType.COPING_STRATEGY
        assert "coping" in keywords

    def test_classify_psychoeducation(self, classifier: IntentClassifier) -> None:
        """Test classifying psychoeducation request."""
        intent, confidence, keywords = classifier.classify("What is depression and why do I feel this way?")
        assert intent == IntentType.PSYCHOEDUCATION

    def test_classify_general_chat(self, classifier: IntentClassifier) -> None:
        """Test classifying general chat message."""
        intent, confidence, keywords = classifier.classify("Hello, how are you today?")
        assert intent == IntentType.GENERAL_CHAT
        assert confidence <= 0.6

    def test_classify_with_context(self, classifier: IntentClassifier) -> None:
        """Test classifying with conversation context."""
        context = "User has been discussing anxiety symptoms"
        intent, confidence, keywords = classifier.classify(
            "It's getting worse",
            conversation_context=context,
        )
        assert intent in (IntentType.EMOTIONAL_SUPPORT, IntentType.GENERAL_CHAT, IntentType.PROGRESS_UPDATE)


class TestAgentRouter:
    """Tests for AgentRouter class."""

    @pytest.fixture
    def router(self) -> AgentRouter:
        """Create agent router for tests."""
        return AgentRouter(SupervisorSettings())

    def test_route_crisis_disclosure(self, router: AgentRouter) -> None:
        """Test routing for crisis disclosure."""
        agents, reason = router.select_agents(
            IntentType.CRISIS_DISCLOSURE,
            {"crisis_detected": False},
        )
        assert AgentType.SAFETY in agents
        assert "safety" in reason.lower() or "crisis" in reason.lower()

    def test_route_with_active_crisis(self, router: AgentRouter) -> None:
        """Test routing when crisis is already detected."""
        agents, reason = router.select_agents(
            IntentType.GENERAL_CHAT,
            {"crisis_detected": True, "risk_level": "HIGH"},
        )
        assert agents == [AgentType.SAFETY]
        assert "crisis" in reason.lower()

    def test_route_emotional_support(self, router: AgentRouter) -> None:
        """Test routing for emotional support."""
        agents, reason = router.select_agents(
            IntentType.EMOTIONAL_SUPPORT,
            {},
        )
        assert AgentType.THERAPY in agents
        assert AgentType.PERSONALITY in agents

    def test_route_symptom_discussion(self, router: AgentRouter) -> None:
        """Test routing for symptom discussion."""
        agents, reason = router.select_agents(
            IntentType.SYMPTOM_DISCUSSION,
            {},
        )
        assert AgentType.DIAGNOSIS in agents

    def test_route_treatment_inquiry(self, router: AgentRouter) -> None:
        """Test routing for treatment inquiry."""
        agents, reason = router.select_agents(
            IntentType.TREATMENT_INQUIRY,
            {},
        )
        assert AgentType.THERAPY in agents

    def test_route_assessment_request(self, router: AgentRouter) -> None:
        """Test routing for assessment request."""
        agents, reason = router.select_agents(
            IntentType.ASSESSMENT_REQUEST,
            {},
        )
        assert AgentType.DIAGNOSIS in agents

    def test_route_general_chat(self, router: AgentRouter) -> None:
        """Test routing for general chat."""
        agents, reason = router.select_agents(
            IntentType.GENERAL_CHAT,
            {},
        )
        assert AgentType.CHAT in agents

    def test_route_with_active_treatment(self, router: AgentRouter) -> None:
        """Test routing when user has active treatment."""
        agents, reason = router.select_agents(
            IntentType.PROGRESS_UPDATE,
            {},
            has_active_treatment=True,
        )
        assert AgentType.THERAPY in agents

    def test_route_respects_max_agents(self, router: AgentRouter) -> None:
        """Test routing respects max parallel agents limit."""
        agents, _ = router.select_agents(IntentType.PSYCHOEDUCATION, {})
        assert len(agents) <= router._settings.max_parallel_agents


class TestSupervisorAgent:
    """Tests for SupervisorAgent class."""

    @pytest.fixture
    def supervisor(self) -> SupervisorAgent:
        """Create supervisor agent for tests."""
        return SupervisorAgent()

    @pytest.mark.asyncio
    async def test_make_decision_crisis(self, supervisor: SupervisorAgent) -> None:
        """Test making decision for crisis message."""
        decision = await supervisor.make_decision("I want to end my life")
        assert decision.intent == IntentType.CRISIS_DISCLOSURE
        assert decision.requires_safety_override is True
        assert decision.processing_priority == "critical"
        assert AgentType.SAFETY in decision.selected_agents

    @pytest.mark.asyncio
    async def test_make_decision_emotional(self, supervisor: SupervisorAgent) -> None:
        """Test making decision for emotional message."""
        decision = await supervisor.make_decision("I've been feeling so depressed and hopeless")
        assert decision.intent == IntentType.EMOTIONAL_SUPPORT
        assert decision.processing_priority in ("high", "normal")

    @pytest.mark.asyncio
    async def test_make_decision_general(self, supervisor: SupervisorAgent) -> None:
        """Test making decision for general message."""
        decision = await supervisor.make_decision("Hello, nice to meet you!")
        assert decision.intent == IntentType.GENERAL_CHAT
        assert decision.processing_priority == "normal"

    @pytest.mark.asyncio
    async def test_make_decision_with_safety_flags(self, supervisor: SupervisorAgent) -> None:
        """Test making decision with existing safety flags."""
        decision = await supervisor.make_decision(
            "Can we continue our conversation?",
            safety_flags={"crisis_detected": True, "risk_level": "HIGH"},
        )
        assert decision.requires_safety_override is True

    @pytest.mark.asyncio
    async def test_make_decision_tracks_keywords(self, supervisor: SupervisorAgent) -> None:
        """Test that matched keywords are tracked."""
        decision = await supervisor.make_decision("I've been very anxious about my symptoms")
        assert "matched_keywords" in decision.metadata

    @pytest.mark.asyncio
    async def test_process_state(self, supervisor: SupervisorAgent) -> None:
        """Test processing state updates."""
        state = create_initial_state("user-1", "session-1", "I feel anxious")
        updates = await supervisor.process(state)
        assert "intent" in updates
        assert "intent_confidence" in updates
        assert "selected_agents" in updates
        assert "processing_phase" in updates
        assert "agent_results" in updates

    @pytest.mark.asyncio
    async def test_process_state_crisis_detection(self, supervisor: SupervisorAgent) -> None:
        """Test state processing detects crisis."""
        state = create_initial_state("user-1", "session-1", "I want to hurt myself")
        updates = await supervisor.process(state)
        assert updates["intent"] == IntentType.CRISIS_DISCLOSURE.value
        assert AgentType.SAFETY.value in updates["selected_agents"]

    @pytest.mark.asyncio
    async def test_get_statistics(self, supervisor: SupervisorAgent) -> None:
        """Test getting supervisor statistics."""
        state1 = create_initial_state("user-1", "session-1", "Test message")
        state2 = create_initial_state("user-1", "session-1", "Another message")
        await supervisor.process(state1)
        await supervisor.process(state2)
        stats = supervisor.get_statistics()
        assert stats["total_decisions"] == 2
        assert "settings" in stats


class TestSupervisorNode:
    """Tests for supervisor_node function."""

    @pytest.mark.asyncio
    async def test_supervisor_node_function(self) -> None:
        """Test supervisor node function creates agent and processes."""
        state = create_initial_state("user-1", "session-1", "Hello there")
        updates = await supervisor_node(state)
        assert "intent" in updates
        assert "selected_agents" in updates
        assert updates["processing_phase"] == "agent_routing"

    @pytest.mark.asyncio
    async def test_supervisor_node_handles_empty_message(self) -> None:
        """Test supervisor node handles edge case of minimal message."""
        state = create_initial_state("user-1", "session-1", "hi")
        updates = await supervisor_node(state)
        assert updates["intent"] == IntentType.GENERAL_CHAT.value


class TestIntentClassifierEdgeCases:
    """Tests for edge cases in intent classification."""

    @pytest.fixture
    def classifier(self) -> IntentClassifier:
        """Create intent classifier for tests."""
        return IntentClassifier(SupervisorSettings())

    def test_classify_empty_message(self, classifier: IntentClassifier) -> None:
        """Test classifying empty message."""
        intent, confidence, keywords = classifier.classify("")
        assert intent == IntentType.GENERAL_CHAT
        assert confidence <= 0.6

    def test_classify_mixed_content(self, classifier: IntentClassifier) -> None:
        """Test classifying message with mixed intents."""
        intent, confidence, keywords = classifier.classify(
            "I've been depressed but want to learn coping strategies"
        )
        assert intent in (IntentType.EMOTIONAL_SUPPORT, IntentType.COPING_STRATEGY, IntentType.TREATMENT_INQUIRY)

    def test_classify_case_insensitive(self, classifier: IntentClassifier) -> None:
        """Test classification is case insensitive."""
        intent1, _, _ = classifier.classify("I feel DEPRESSED")
        intent2, _, _ = classifier.classify("I feel depressed")
        assert intent1 == intent2

    def test_classify_partial_keyword_match(self, classifier: IntentClassifier) -> None:
        """Test classification with partial keyword in sentence."""
        intent, confidence, keywords = classifier.classify("I can't sleep well lately")
        assert intent == IntentType.SYMPTOM_DISCUSSION or "sleep" in keywords or intent == IntentType.GENERAL_CHAT

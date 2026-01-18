"""
Unit tests for Solace-AI Orchestrator Service - Agents Module.
Tests for safety, diagnosis, therapy, personality, and chat agents.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from services.orchestrator_service.src.agents import (
    # Safety Agent
    SafetyAgent,
    SafetyAgentSettings,
    SafetyCheckRequest,
    SafetyCheckResult,
    SafetyCheckType,
    CrisisLevel,
    RiskFactorDTO,
    ProtectiveFactorDTO,
    safety_agent_node,
    # Diagnosis Agent
    DiagnosisAgent,
    DiagnosisAgentSettings,
    DiagnosisPhase,
    SymptomType,
    SeverityLevel,
    SymptomDTO,
    HypothesisDTO,
    DifferentialDTO,
    AssessmentResult,
    diagnosis_agent_node,
    # Therapy Agent
    TherapyAgent,
    TherapyAgentSettings,
    SessionPhase,
    TherapyModality,
    TechniqueCategory,
    TechniqueDTO,
    HomeworkDTO,
    TherapyResponse,
    therapy_agent_node,
    # Personality Agent
    PersonalityAgent,
    PersonalityAgentSettings,
    PersonalityTrait,
    CommunicationStyleType,
    AssessmentSource,
    TraitScoreDTO,
    OceanScoresDTO,
    StyleParametersDTO,
    PersonalityDetectionResult,
    personality_agent_node,
    # Chat Agent
    ChatAgent,
    ChatAgentSettings,
    ConversationTone,
    TopicCategory,
    TopicClassification,
    ChatResponse,
    TopicClassifier,
    chat_agent_node,
)
from services.orchestrator_service.src.langgraph.state_schema import (
    OrchestratorState,
    create_initial_state,
    AgentType,
    RiskLevel,
)


# ============================================================================
# Safety Agent Tests
# ============================================================================

class TestSafetyAgentSettings:
    """Tests for SafetyAgentSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = SafetyAgentSettings()
        assert settings.service_url == "http://localhost:8001"
        assert settings.timeout_seconds == 10.0
        assert settings.enable_escalation is True
        assert settings.fallback_on_service_error is True

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = SafetyAgentSettings(
            service_url="http://safety:9000",
            timeout_seconds=30.0,
            enable_escalation=False,
        )
        assert settings.service_url == "http://safety:9000"
        assert settings.timeout_seconds == 30.0
        assert settings.enable_escalation is False


class TestSafetyCheckRequest:
    """Tests for SafetyCheckRequest."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        request = SafetyCheckRequest(
            user_id="user-1",
            session_id="session-1",
            message_id="msg-1",
            content="test message",
            check_type=SafetyCheckType.PRE_CHECK,
            include_resources=True,
        )
        data = request.to_dict()
        assert data["user_id"] == "user-1"
        assert data["check_type"] == "PRE_CHECK"
        assert data["include_resources"] is True


class TestSafetyCheckResult:
    """Tests for SafetyCheckResult."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "is_safe": False,
            "crisis_level": "HIGH",
            "risk_score": 0.8,
            "risk_factors": [{"factor_type": "suicidal_ideation", "severity": 0.9, "evidence": "explicit", "confidence": 0.95}],
            "protective_factors": [],
            "requires_escalation": True,
            "requires_human_review": True,
            "crisis_resources": [{"name": "988", "contact": "988"}],
            "triggered_keywords": ["suicide"],
        }
        result = SafetyCheckResult.from_dict(data)
        assert result.is_safe is False
        assert result.crisis_level == CrisisLevel.HIGH
        assert result.risk_score == Decimal("0.8")
        assert len(result.risk_factors) == 1
        assert result.requires_escalation is True

    def test_to_risk_level(self):
        """Test conversion to orchestrator risk level."""
        result = SafetyCheckResult(
            is_safe=False,
            crisis_level=CrisisLevel.CRITICAL,
            risk_score=Decimal("0.95"),
            risk_factors=[],
            protective_factors=[],
            requires_escalation=True,
            requires_human_review=True,
            crisis_resources=[],
        )
        assert result.to_risk_level() == RiskLevel.CRITICAL


class TestRiskFactorDTO:
    """Tests for RiskFactorDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "factor_type": "suicidal_ideation",
            "severity": 0.8,
            "evidence": "explicit statement",
            "confidence": 0.9,
            "detection_layer": 2,
        }
        factor = RiskFactorDTO.from_dict(data)
        assert factor.factor_type == "suicidal_ideation"
        assert factor.severity == 0.8
        assert factor.detection_layer == 2


class TestSafetyAgent:
    """Tests for SafetyAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = SafetyAgent()
        assert agent._check_count == 0
        assert agent._settings.service_url == "http://localhost:8001"

    def test_should_include_resources(self):
        """Test crisis resource inclusion logic."""
        agent = SafetyAgent()
        assert agent._should_include_resources("I want to kill myself") is True
        assert agent._should_include_resources("I feel a bit sad today") is False
        assert agent._should_include_resources("I'm thinking about self-harm") is True

    def test_build_fallback_response_crisis(self):
        """Test fallback response for crisis message."""
        agent = SafetyAgent()
        result = agent._build_fallback_response("I want to kill myself", {})
        safety_flags = result["safety_flags"]
        assert safety_flags["crisis_detected"] is True
        assert safety_flags["risk_level"] == RiskLevel.HIGH.value
        assert "kill myself" in safety_flags["triggered_keywords"]

    def test_build_fallback_response_safe(self):
        """Test fallback response for safe message."""
        agent = SafetyAgent()
        result = agent._build_fallback_response("Hello, how are you?", {})
        safety_flags = result["safety_flags"]
        assert safety_flags["crisis_detected"] is False
        assert safety_flags["risk_level"] == RiskLevel.NONE.value

    def test_determine_monitoring_level(self):
        """Test monitoring level determination."""
        agent = SafetyAgent()
        critical_result = SafetyCheckResult(
            is_safe=False,
            crisis_level=CrisisLevel.CRITICAL,
            risk_score=Decimal("0.95"),
            risk_factors=[],
            protective_factors=[],
            requires_escalation=True,
            requires_human_review=True,
            crisis_resources=[],
        )
        assert agent._determine_monitoring_level(critical_result) == "intensive"
        elevated_result = SafetyCheckResult(
            is_safe=True,
            crisis_level=CrisisLevel.ELEVATED,
            risk_score=Decimal("0.5"),
            risk_factors=[],
            protective_factors=[],
            requires_escalation=False,
            requires_human_review=False,
            crisis_resources=[],
        )
        assert agent._determine_monitoring_level(elevated_result) == "enhanced"

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = SafetyAgent()
        stats = agent.get_statistics()
        assert stats["total_checks"] == 0
        assert "service_url" in stats


# ============================================================================
# Diagnosis Agent Tests
# ============================================================================

class TestDiagnosisAgentSettings:
    """Tests for DiagnosisAgentSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = DiagnosisAgentSettings()
        assert settings.service_url == "http://localhost:8002"
        assert settings.enable_differential is True
        assert settings.min_confidence_threshold == 0.5


class TestSymptomDTO:
    """Tests for SymptomDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "symptom_id": "sym-1",
            "name": "Depressed mood",
            "description": "Persistent sadness",
            "symptom_type": "EMOTIONAL",
            "severity": "MODERATE",
            "confidence": 0.8,
            "onset": "2 weeks ago",
            "triggers": ["stress", "loss"],
        }
        symptom = SymptomDTO.from_dict(data)
        assert symptom.name == "Depressed mood"
        assert symptom.symptom_type == SymptomType.EMOTIONAL
        assert symptom.severity == SeverityLevel.MODERATE
        assert len(symptom.triggers) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        symptom = SymptomDTO(
            symptom_id="sym-1",
            name="Anxiety",
            description="Excessive worry",
            symptom_type=SymptomType.COGNITIVE,
            severity=SeverityLevel.MILD,
            confidence=0.7,
        )
        data = symptom.to_dict()
        assert data["symptom_type"] == "COGNITIVE"
        assert data["severity"] == "MILD"


class TestHypothesisDTO:
    """Tests for HypothesisDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "hypothesis_id": "hyp-1",
            "name": "Major Depressive Disorder",
            "confidence": 0.75,
            "dsm5_code": "296.2x",
            "criteria_met": ["depressed mood", "anhedonia"],
            "criteria_missing": ["weight changes"],
            "severity": "MODERATE",
        }
        hypothesis = HypothesisDTO.from_dict(data)
        assert hypothesis.name == "Major Depressive Disorder"
        assert hypothesis.confidence == 0.75
        assert len(hypothesis.criteria_met) == 2


class TestDifferentialDTO:
    """Tests for DifferentialDTO."""

    def test_from_dict_with_primary(self):
        """Test creation from dictionary with primary hypothesis."""
        data = {
            "primary": {
                "hypothesis_id": "hyp-1",
                "name": "MDD",
                "confidence": 0.8,
            },
            "alternatives": [
                {"hypothesis_id": "hyp-2", "name": "GAD", "confidence": 0.5},
            ],
            "ruled_out": ["Bipolar"],
            "missing_info": ["sleep patterns"],
        }
        diff = DifferentialDTO.from_dict(data)
        assert diff.primary is not None
        assert diff.primary.name == "MDD"
        assert len(diff.alternatives) == 1
        assert "Bipolar" in diff.ruled_out


class TestDiagnosisAgent:
    """Tests for DiagnosisAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DiagnosisAgent()
        assert agent._assessment_count == 0

    def test_build_fallback_response_with_symptoms(self):
        """Test fallback response when symptoms detected."""
        agent = DiagnosisAgent()
        result = agent._build_fallback_response("I've been feeling very sad and tired")
        agent_result = result["agent_results"][0]
        assert agent_result["agent_type"] == AgentType.DIAGNOSIS.value
        assert agent_result["success"] is True
        assert "fallback_mode" in agent_result["metadata"]

    def test_build_fallback_response_general(self):
        """Test fallback response for general message."""
        agent = DiagnosisAgent()
        result = agent._build_fallback_response("Hello there")
        agent_result = result["agent_results"][0]
        assert "how you've been feeling" in agent_result["response_content"]

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = DiagnosisAgent()
        stats = agent.get_statistics()
        assert stats["total_assessments"] == 0
        assert stats["differential_enabled"] is True


# ============================================================================
# Therapy Agent Tests
# ============================================================================

class TestTherapyAgentSettings:
    """Tests for TherapyAgentSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = TherapyAgentSettings()
        assert settings.service_url == "http://localhost:8003"
        assert settings.default_modality == "CBT"
        assert settings.enable_homework is True


class TestTechniqueDTO:
    """Tests for TechniqueDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "technique_id": "tech-1",
            "name": "Cognitive Restructuring",
            "modality": "CBT",
            "category": "COGNITIVE_RESTRUCTURING",
            "description": "Challenge negative thoughts",
            "duration_minutes": 20,
            "requires_homework": True,
        }
        technique = TechniqueDTO.from_dict(data)
        assert technique.name == "Cognitive Restructuring"
        assert technique.modality == TherapyModality.CBT
        assert technique.requires_homework is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        technique = TechniqueDTO(
            technique_id="tech-1",
            name="Grounding",
            modality=TherapyModality.DBT,
            category=TechniqueCategory.DISTRESS_TOLERANCE,
            description="5-4-3-2-1 technique",
        )
        data = technique.to_dict()
        assert data["modality"] == "DBT"
        assert data["category"] == "DISTRESS_TOLERANCE"


class TestHomeworkDTO:
    """Tests for HomeworkDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "homework_id": "hw-1",
            "title": "Thought Record",
            "description": "Track negative thoughts",
            "technique_id": "tech-1",
            "completed": False,
        }
        homework = HomeworkDTO.from_dict(data)
        assert homework.title == "Thought Record"
        assert homework.completed is False


class TestTherapyAgent:
    """Tests for TherapyAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = TherapyAgent()
        assert agent._session_count == 0

    def test_emotional_support_response_anxiety(self):
        """Test emotional support for anxiety."""
        agent = TherapyAgent()
        response = agent._emotional_support_response("i feel so anxious")
        assert "anxiety" in response.lower()
        assert "ground" in response.lower()

    def test_emotional_support_response_depression(self):
        """Test emotional support for depression."""
        agent = TherapyAgent()
        response = agent._emotional_support_response("i feel so sad and depressed")
        assert "down" in response.lower() or "sad" in response.lower()

    def test_coping_strategy_response(self):
        """Test coping strategy response."""
        agent = TherapyAgent()
        response = agent._coping_strategy_response("I feel overwhelmed")
        assert "overwhelm" in response.lower() or "step" in response.lower()

    def test_treatment_inquiry_response(self):
        """Test treatment inquiry response."""
        agent = TherapyAgent()
        response = agent._treatment_inquiry_response("What treatments are available?")
        assert "CBT" in response or "evidence-based" in response.lower()

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = TherapyAgent()
        stats = agent.get_statistics()
        assert stats["total_sessions"] == 0
        assert stats["default_modality"] == "CBT"


# ============================================================================
# Personality Agent Tests
# ============================================================================

class TestPersonalityAgentSettings:
    """Tests for PersonalityAgentSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = PersonalityAgentSettings()
        assert settings.service_url == "http://localhost:8004"
        assert settings.enable_style_adaptation is True
        assert settings.min_text_length_for_detection == 50


class TestTraitScoreDTO:
    """Tests for TraitScoreDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "trait": "OPENNESS",
            "value": 0.75,
            "confidence_lower": 0.65,
            "confidence_upper": 0.85,
            "evidence_markers": ["creative", "curious"],
        }
        score = TraitScoreDTO.from_dict(data)
        assert score.trait == PersonalityTrait.OPENNESS
        assert score.value == 0.75
        assert len(score.evidence_markers) == 2


class TestOceanScoresDTO:
    """Tests for OceanScoresDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.4,
            "agreeableness": 0.8,
            "neuroticism": 0.3,
            "overall_confidence": 0.75,
            "assessed_at": "2024-01-15T10:00:00+00:00",
        }
        scores = OceanScoresDTO.from_dict(data)
        assert scores.openness == 0.7
        assert scores.agreeableness == 0.8
        assert scores.overall_confidence == 0.75

    def test_dominant_traits(self):
        """Test dominant traits detection."""
        scores = OceanScoresDTO(
            openness=0.8,
            conscientiousness=0.4,
            extraversion=0.7,
            agreeableness=0.5,
            neuroticism=0.3,
            overall_confidence=0.8,
            assessed_at=datetime.now(timezone.utc),
        )
        dominant = scores.dominant_traits(threshold=0.6)
        assert PersonalityTrait.OPENNESS in dominant
        assert PersonalityTrait.EXTRAVERSION in dominant
        assert PersonalityTrait.CONSCIENTIOUSNESS not in dominant


class TestStyleParametersDTO:
    """Tests for StyleParametersDTO."""

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "warmth": 0.8,
            "structure": 0.6,
            "complexity": 0.5,
            "directness": 0.4,
            "energy": 0.7,
            "validation_level": 0.75,
            "style_type": "EXPRESSIVE",
        }
        style = StyleParametersDTO.from_dict(data)
        assert style.warmth == 0.8
        assert style.style_type == CommunicationStyleType.EXPRESSIVE

    def test_default(self):
        """Test default style creation."""
        style = StyleParametersDTO.default()
        assert style.warmth == 0.6
        assert style.style_type == CommunicationStyleType.BALANCED


class TestPersonalityAgent:
    """Tests for PersonalityAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = PersonalityAgent()
        assert agent._detection_count == 0

    def test_scores_to_style(self):
        """Test conversion of OCEAN scores to style parameters."""
        agent = PersonalityAgent()
        scores = OceanScoresDTO(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.75,
            neuroticism=0.3,
            overall_confidence=0.8,
            assessed_at=datetime.now(timezone.utc),
        )
        style = agent._scores_to_style(scores)
        assert style.warmth > 0.5
        assert style.structure == 0.7
        assert style.complexity == 0.8

    def test_determine_style_type_expressive(self):
        """Test style type determination for expressive."""
        agent = PersonalityAgent()
        scores = OceanScoresDTO(
            openness=0.8,
            conscientiousness=0.5,
            extraversion=0.8,
            agreeableness=0.5,
            neuroticism=0.3,
            overall_confidence=0.8,
            assessed_at=datetime.now(timezone.utc),
        )
        style_type = agent._determine_style_type(scores)
        assert style_type == CommunicationStyleType.EXPRESSIVE

    def test_determine_style_type_amiable(self):
        """Test style type determination for amiable."""
        agent = PersonalityAgent()
        scores = OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.4,
            agreeableness=0.85,
            neuroticism=0.3,
            overall_confidence=0.8,
            assessed_at=datetime.now(timezone.utc),
        )
        style_type = agent._determine_style_type(scores)
        assert style_type == CommunicationStyleType.AMIABLE

    def test_build_fallback_response_with_existing(self):
        """Test fallback with existing style."""
        agent = PersonalityAgent()
        existing = {"warmth": 0.9, "style_type": "AMIABLE"}
        result = agent._build_fallback_response(existing)
        assert result["personality_style"]["warmth"] == 0.9

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = PersonalityAgent()
        stats = agent.get_statistics()
        assert stats["total_detections"] == 0


# ============================================================================
# Chat Agent Tests
# ============================================================================

class TestChatAgentSettings:
    """Tests for ChatAgentSettings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = ChatAgentSettings()
        assert settings.default_warmth == 0.7
        assert settings.include_follow_up_questions is True
        assert settings.empathy_phrases_enabled is True


class TestTopicClassifier:
    """Tests for TopicClassifier."""

    def test_classify_greeting(self):
        """Test greeting classification."""
        classifier = TopicClassifier()
        result = classifier.classify("Hello!")
        assert result.category == TopicCategory.GREETING
        assert result.confidence >= 0.8

    def test_classify_farewell(self):
        """Test farewell classification."""
        classifier = TopicClassifier()
        result = classifier.classify("Goodbye, take care!")
        assert result.category == TopicCategory.FAREWELL

    def test_classify_gratitude(self):
        """Test gratitude classification."""
        classifier = TopicClassifier()
        result = classifier.classify("Thank you so much!")
        assert result.category == TopicCategory.GRATITUDE

    def test_classify_check_in(self):
        """Test check-in classification."""
        classifier = TopicClassifier()
        result = classifier.classify("How are you doing?")
        assert result.category == TopicCategory.CHECK_IN

    def test_classify_clarification(self):
        """Test clarification classification."""
        classifier = TopicClassifier()
        result = classifier.classify("Can you explain that?")
        assert result.category == TopicCategory.CLARIFICATION

    def test_classify_general(self):
        """Test general message classification."""
        classifier = TopicClassifier()
        result = classifier.classify("I've been thinking about changing my career path")
        assert result.category == TopicCategory.GENERAL


class TestChatResponse:
    """Tests for ChatResponse."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = ChatResponse(
            content="Hello!",
            tone=ConversationTone.WARM,
            topic=TopicCategory.GREETING,
            includes_follow_up=True,
            empathy_applied=True,
            warmth_level=0.8,
        )
        data = response.to_dict()
        assert data["tone"] == "warm"
        assert data["topic"] == "greeting"
        assert data["warmth_level"] == 0.8


class TestChatAgent:
    """Tests for ChatAgent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = ChatAgent()
        assert agent._message_count == 0

    def test_process_greeting(self):
        """Test processing greeting message."""
        agent = ChatAgent()
        state = create_initial_state("user-1", "session-1", "Hello!")
        result = agent.process(state)
        assert "agent_results" in result
        agent_result = result["agent_results"][0]
        assert agent_result["agent_type"] == AgentType.CHAT.value
        assert agent_result["success"] is True
        assert "response_content" in agent_result

    def test_process_farewell(self):
        """Test processing farewell message."""
        agent = ChatAgent()
        state = create_initial_state("user-1", "session-1", "Goodbye!")
        result = agent.process(state)
        agent_result = result["agent_results"][0]
        assert "take care" in agent_result["response_content"].lower()

    def test_process_with_personality_style(self):
        """Test processing with personality style."""
        agent = ChatAgent()
        state = create_initial_state("user-1", "session-1", "Hello!")
        state["personality_style"] = {"warmth": 0.9, "validation_level": 0.8}
        result = agent.process(state)
        agent_result = result["agent_results"][0]
        assert agent_result["success"] is True

    def test_get_statistics(self):
        """Test statistics retrieval."""
        agent = ChatAgent()
        agent.process(create_initial_state("user-1", "session-1", "Hi"))
        stats = agent.get_statistics()
        assert stats["total_messages"] == 1


# ============================================================================
# Node Function Tests
# ============================================================================

class TestNodeFunctions:
    """Tests for LangGraph node functions."""

    def test_chat_agent_node(self):
        """Test chat_agent_node function."""
        state = create_initial_state("user-1", "session-1", "Hello!")
        result = chat_agent_node(state)
        assert "agent_results" in result
        assert result["agent_results"][0]["agent_type"] == AgentType.CHAT.value

    @pytest.mark.asyncio
    async def test_safety_agent_node_fallback(self):
        """Test safety_agent_node with fallback."""
        state = create_initial_state("user-1", "session-1", "I want to kill myself")
        result = await safety_agent_node(state)
        assert "safety_flags" in result
        assert result["safety_flags"]["crisis_detected"] is True

    @pytest.mark.asyncio
    async def test_diagnosis_agent_node_fallback(self):
        """Test diagnosis_agent_node with fallback."""
        settings = DiagnosisAgentSettings(fallback_on_service_error=True)
        agent = DiagnosisAgent(settings)
        state = create_initial_state("user-1", "session-1", "I've been feeling tired")
        result = await agent.process(state)
        assert "agent_results" in result

    @pytest.mark.asyncio
    async def test_therapy_agent_node_fallback(self):
        """Test therapy_agent_node with fallback."""
        settings = TherapyAgentSettings(fallback_on_service_error=True)
        agent = TherapyAgent(settings)
        state = create_initial_state("user-1", "session-1", "I feel anxious")
        state["intent"] = "emotional_support"
        result = await agent.process(state)
        assert "agent_results" in result

    @pytest.mark.asyncio
    async def test_personality_agent_node_fallback(self):
        """Test personality_agent_node with fallback."""
        settings = PersonalityAgentSettings(fallback_on_service_error=True)
        agent = PersonalityAgent(settings)
        state = create_initial_state("user-1", "session-1", "Hello there")
        result = await agent.process(state)
        assert "personality_style" in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestAgentIntegration:
    """Integration tests for agent coordination."""

    def test_chat_agent_with_initial_state(self):
        """Test chat agent with properly initialized state."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="How are you today?",
        )
        agent = ChatAgent()
        result = agent.process(state)
        assert result["agent_results"][0]["success"] is True
        assert "metadata" in result["agent_results"][0]

    def test_multiple_agents_state_updates(self):
        """Test that agent state updates are compatible."""
        state = create_initial_state("user-1", "session-1", "Hello!")
        chat_agent = ChatAgent()
        chat_result = chat_agent.process(state)
        assert "agent_results" in chat_result
        assert isinstance(chat_result["agent_results"], list)

    @pytest.mark.asyncio
    async def test_safety_agent_crisis_detection(self):
        """Test safety agent crisis detection end-to-end."""
        state = create_initial_state(
            user_id="user-1",
            session_id="session-1",
            message="I've been having thoughts of ending my life",
        )
        agent = SafetyAgent()
        result = await agent.process(state)
        assert result["safety_flags"]["crisis_detected"] is True
        assert result["safety_flags"]["risk_level"] in ("high", "critical")

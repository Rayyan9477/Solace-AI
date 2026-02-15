"""
Solace-AI Orchestrator Service - Response Module Tests.
Tests for router, aggregator, generator, style applicator, and safety wrapper.
"""
import pytest
from datetime import datetime, timezone
from decimal import Decimal

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
from services.orchestrator_service.src.response.generator import (
    ResponseGenerator,
    GeneratorSettings,
    ResponseType,
    ResponseFormat,
    ResponseContext,
    GeneratedResponse,
    EmpathyEnhancer,
    FollowUpGenerator,
    ContentFormatter,
    generator_node,
)
from services.orchestrator_service.src.response.style_applicator import (
    StyleApplicator,
    StyleApplicatorSettings,
    CommunicationStyle,
    StyleParameters,
    StyledResponse,
    WarmthAdjuster,
    ComplexityAdjuster,
    StructureAdjuster,
    DirectnessAdjuster,
    style_applicator_node,
)
from services.orchestrator_service.src.response.safety_wrapper import (
    SafetyWrapper,
    SafetyWrapperSettings,
    ResourceType,
    CrisisResource,
    SafetyWrapResult,
    ResourceProvider,
    ContentFilter,
    DisclaimerInjector,
    safety_wrapper_node,
)


# =============================================================================
# Aggregator Tests
# =============================================================================

class TestAggregatorSettings:
    """Tests for AggregatorSettings."""

    def test_default_settings(self):
        """Test default aggregator settings."""
        settings = AggregatorSettings()
        assert settings.strategy == "priority_based"
        assert settings.confidence_threshold == 0.5
        assert settings.max_response_length == 1500


class TestResponseRanker:
    """Tests for ResponseRanker."""

    def test_rank_results(self):
        """Test ranking agent results."""
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
        """Test that empty content is filtered."""
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
        """Test merging single contribution."""
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
        """Test merging empty contributions."""
        settings = AggregatorSettings()
        merger = ResponseMerger(settings)
        result = merger.merge([], AggregationStrategy.FIRST_SUCCESS)
        assert result == settings.fallback_response


class TestAgentContribution:
    """Tests for AgentContribution."""

    def test_contribution_to_dict(self):
        """Test contribution serialization."""
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
        """Test aggregator processing."""
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
        """Test aggregator fallback response."""
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
        """Test aggregator node function."""
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


# =============================================================================
# Generator Tests
# =============================================================================

class TestGeneratorSettings:
    """Tests for GeneratorSettings."""

    def test_default_settings(self):
        """Test default generator settings."""
        settings = GeneratorSettings()
        assert settings.max_response_length == 2000
        assert settings.enable_empathy_phrases is True
        assert settings.enable_follow_up_questions is True


class TestEmpathyEnhancer:
    """Tests for EmpathyEnhancer."""

    def test_enhance_warm_content(self):
        """Test enhancing with high warmth."""
        settings = GeneratorSettings()
        enhancer = EmpathyEnhancer(settings)
        result, empathy, validation = enhancer.enhance(
            "This is a response.",
            warmth=0.8,
            validation_level=0.7,
            response_type=ResponseType.THERAPEUTIC,
        )
        assert empathy is True
        assert "appreciate" in result.lower() or "hear" in result.lower()

    def test_enhance_crisis_content_unchanged(self):
        """Test that crisis content is not enhanced."""
        settings = GeneratorSettings()
        enhancer = EmpathyEnhancer(settings)
        original = "Crisis response content."
        result, empathy, validation = enhancer.enhance(
            original,
            warmth=0.8,
            validation_level=0.7,
            response_type=ResponseType.CRISIS,
        )
        assert result == original
        assert empathy is False


class TestFollowUpGenerator:
    """Tests for FollowUpGenerator."""

    def test_generate_emotional_follow_up(self):
        """Test generating emotional support follow-up."""
        settings = GeneratorSettings()
        generator = FollowUpGenerator(settings)
        result = generator.generate(
            IntentType.EMOTIONAL_SUPPORT,
            is_first_message=False,
            content="Response without question",
        )
        assert result is not None
        assert "?" in result

    def test_no_follow_up_when_question_exists(self):
        """Test no follow-up when content ends with question."""
        settings = GeneratorSettings()
        generator = FollowUpGenerator(settings)
        result = generator.generate(
            IntentType.EMOTIONAL_SUPPORT,
            is_first_message=False,
            content="How are you feeling?",
        )
        assert result is None


class TestContentFormatter:
    """Tests for ContentFormatter."""

    def test_format_normalizes_whitespace(self):
        """Test whitespace normalization."""
        settings = GeneratorSettings()
        formatter = ContentFormatter(settings)
        result = formatter.format(
            "Content   with   extra   spaces",
            ResponseFormat.PLAIN,
            ResponseType.CONVERSATIONAL,
        )
        assert "   " not in result

    def test_format_ensures_punctuation(self):
        """Test punctuation is ensured."""
        settings = GeneratorSettings()
        formatter = ContentFormatter(settings)
        result = formatter.format(
            "Content without period",
            ResponseFormat.PLAIN,
            ResponseType.CONVERSATIONAL,
        )
        assert result.endswith(".")


class TestResponseGenerator:
    """Tests for ResponseGenerator."""

    def test_generator_process(self):
        """Test generator processing."""
        generator = ResponseGenerator()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response content"
        state["intent"] = IntentType.GENERAL_CHAT.value
        result = generator.generate(state)
        assert "final_response" in result
        assert result["processing_phase"] == ProcessingPhase.RESPONSE_GENERATION.value

    def test_generator_statistics(self):
        """Test generator statistics."""
        generator = ResponseGenerator()
        stats = generator.get_statistics()
        assert "total_generations" in stats


class TestGeneratorNode:
    """Tests for generator_node function."""

    def test_generator_node_function(self):
        """Test generator node function."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response"
        state["intent"] = IntentType.GENERAL_CHAT.value
        result = generator_node(state)
        assert "final_response" in result


# =============================================================================
# Style Applicator Tests
# =============================================================================

class TestStyleApplicatorSettings:
    """Tests for StyleApplicatorSettings."""

    def test_default_settings(self):
        """Test default style applicator settings."""
        settings = StyleApplicatorSettings()
        assert settings.enable_warmth_adjustment is True
        assert settings.enable_complexity_adjustment is True
        assert settings.preserve_crisis_content is True


class TestStyleParameters:
    """Tests for StyleParameters."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"warmth": 0.8, "style_type": "amiable"}
        params = StyleParameters.from_dict(data)
        assert params.warmth == 0.8
        assert params.style_type == CommunicationStyle.AMIABLE

    def test_to_dict(self):
        """Test converting to dictionary."""
        params = StyleParameters(warmth=0.9)
        result = params.to_dict()
        assert result["warmth"] == 0.9


class TestWarmthAdjuster:
    """Tests for WarmthAdjuster."""

    def test_adjust_high_warmth(self):
        """Test adjusting with high warmth."""
        adjuster = WarmthAdjuster()
        result, adjusted = adjuster.adjust("Test content.", warmth=0.8, is_crisis=False)
        assert adjusted is True
        assert "appreciate" in result.lower() or "trust" in result.lower()

    def test_adjust_crisis_unchanged(self):
        """Test crisis content unchanged."""
        adjuster = WarmthAdjuster()
        original = "Crisis content."
        result, adjusted = adjuster.adjust(original, warmth=0.8, is_crisis=True)
        assert result == original
        assert adjusted is False


class TestComplexityAdjuster:
    """Tests for ComplexityAdjuster."""

    def test_simplify_complex_words(self):
        """Test simplifying complex words."""
        adjuster = ComplexityAdjuster()
        result, adjusted = adjuster.adjust(
            "We need to utilize this and facilitate the process.",
            complexity=0.3,
        )
        assert "use" in result.lower()
        assert "help" in result.lower()


class TestStructureAdjuster:
    """Tests for StructureAdjuster."""

    def test_reduce_structure(self):
        """Test reducing structure."""
        adjuster = StructureAdjuster()
        result, adjusted = adjuster.adjust(
            "1. First point. 2. Second point.",
            structure=0.2,
        )
        assert "1." not in result or "2." not in result


class TestDirectnessAdjuster:
    """Tests for DirectnessAdjuster."""

    def test_make_indirect(self):
        """Test making content more indirect."""
        adjuster = DirectnessAdjuster()
        result, adjusted = adjuster.adjust(
            "You should do this exercise.",
            directness=0.3,
        )
        assert "might consider" in result.lower()


class TestStyleApplicator:
    """Tests for StyleApplicator."""

    def test_applicator_process(self):
        """Test style applicator processing."""
        applicator = StyleApplicator()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response content"
        state["personality_style"] = {"warmth": 0.8, "style_type": "amiable"}
        result = applicator.apply(state)
        assert "final_response" in result

    def test_applicator_crisis_preserved(self):
        """Test crisis content is preserved."""
        applicator = StyleApplicator()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Crisis response"
        state["safety_flags"] = {"crisis_detected": True}
        result = applicator.apply(state)
        assert result["final_response"] == "Crisis response"


class TestStyleApplicatorNode:
    """Tests for style_applicator_node function."""

    def test_style_applicator_node_function(self):
        """Test style applicator node function."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response"
        result = style_applicator_node(state)
        assert "final_response" in result


# =============================================================================
# Safety Wrapper Tests
# =============================================================================

class TestSafetyWrapperSettings:
    """Tests for SafetyWrapperSettings."""

    def test_default_settings(self):
        """Test default safety wrapper settings."""
        settings = SafetyWrapperSettings()
        assert settings.enable_resource_injection is True
        assert settings.always_include_988 is True
        assert settings.max_resources_shown == 3


class TestCrisisResource:
    """Tests for CrisisResource."""

    def test_crisis_resource_to_dict(self):
        """Test crisis resource serialization."""
        resource = CrisisResource(
            name="Test Line",
            resource_type=ResourceType.HOTLINE,
            contact="123-456-7890",
            description="Test description",
        )
        result = resource.to_dict()
        assert result["name"] == "Test Line"
        assert result["resource_type"] == "hotline"

    def test_format_display(self):
        """Test display formatting."""
        resource = CrisisResource(
            name="Test Line",
            resource_type=ResourceType.HOTLINE,
            contact="123-456-7890",
            description="Test",
            available_24_7=True,
        )
        display = resource.format_display()
        assert "Test Line" in display
        assert "24/7" in display


class TestResourceProvider:
    """Tests for ResourceProvider."""

    def test_get_resources_high_risk(self):
        """Test getting resources for high risk."""
        settings = SafetyWrapperSettings()
        provider = ResourceProvider(settings)
        resources = provider.get_resources(RiskLevel.HIGH)
        assert len(resources) > 0
        assert any("988" in r.contact for r in resources)

    def test_get_resources_low_risk(self):
        """Test getting resources for low risk."""
        settings = SafetyWrapperSettings()
        provider = ResourceProvider(settings)
        resources = provider.get_resources(RiskLevel.NONE)
        assert len(resources) <= settings.max_resources_shown


class TestContentFilter:
    """Tests for ContentFilter."""

    def test_filter_harmful_phrases(self):
        """Test filtering harmful phrases."""
        settings = SafetyWrapperSettings()
        filter_obj = ContentFilter(settings)
        result, filtered = filter_obj.filter("You should just give up on this.")
        assert "give up" not in result.lower() or filtered is True

    def test_safe_content_unchanged(self):
        """Test safe content unchanged."""
        settings = SafetyWrapperSettings()
        filter_obj = ContentFilter(settings)
        original = "This is helpful content."
        result, filtered = filter_obj.filter(original)
        assert result == original
        assert filtered is False


class TestDisclaimerInjector:
    """Tests for DisclaimerInjector."""

    def test_inject_high_risk_disclaimer(self):
        """Test injecting disclaimer for high risk."""
        settings = SafetyWrapperSettings()
        injector = DisclaimerInjector(settings)
        result, injected = injector.inject(
            "Response content.",
            RiskLevel.HIGH,
            include_resources=True,
        )
        assert injected is True
        assert "not a substitute" in result.lower()

    def test_no_disclaimer_for_low_risk(self):
        """Test no disclaimer for low risk."""
        settings = SafetyWrapperSettings()
        injector = DisclaimerInjector(settings)
        original = "Response content."
        result, injected = injector.inject(
            original,
            RiskLevel.LOW,
            include_resources=False,
        )
        assert injected is False


class TestSafetyWrapper:
    """Tests for SafetyWrapper."""

    def test_wrapper_process(self):
        """Test safety wrapper processing."""
        wrapper = SafetyWrapper()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response"
        state["safety_flags"] = {"risk_level": "NONE", "crisis_detected": False}
        result = wrapper.wrap(state)
        assert "final_response" in result
        assert result["processing_phase"] == ProcessingPhase.SAFETY_POSTCHECK.value

    def test_wrapper_adds_resources_for_high_risk(self):
        """Test resources added for high risk."""
        wrapper = SafetyWrapper()
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="I want to hurt myself",
        )
        state["final_response"] = "I hear you and I'm concerned."
        state["safety_flags"] = {
            "risk_level": "HIGH",
            "crisis_detected": True,
            "safety_resources_shown": False,
        }
        result = wrapper.wrap(state)
        assert "988" in result["final_response"]

    def test_wrapper_statistics(self):
        """Test wrapper statistics."""
        wrapper = SafetyWrapper()
        stats = wrapper.get_statistics()
        assert "total_wraps" in stats


class TestSafetyWrapperNode:
    """Tests for safety_wrapper_node function."""

    def test_safety_wrapper_node_function(self):
        """Test safety wrapper node function."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello",
        )
        state["final_response"] = "Test response"
        state["safety_flags"] = {"risk_level": "NONE"}
        result = safety_wrapper_node(state)
        assert "final_response" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestResponsePipelineIntegration:
    """Integration tests for the response pipeline."""

    def test_full_pipeline_general_chat(self):
        """Test full pipeline for general chat."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Hello, how are you?",
        )
        router = Router()
        route_result = router.route(state)
        state.update(route_result)
        state["agent_results"] = [
            {"agent_type": "chat", "response_content": "Hello! I'm doing well.", "confidence": 0.8},
        ]
        aggregator = Aggregator()
        agg_result = aggregator.aggregate(state)
        state.update(agg_result)
        generator = ResponseGenerator()
        gen_result = generator.generate(state)
        state.update(gen_result)
        applicator = StyleApplicator()
        style_result = applicator.apply(state)
        state.update(style_result)
        wrapper = SafetyWrapper()
        final_result = wrapper.wrap(state)
        assert "final_response" in final_result
        assert len(final_result["final_response"]) > 0

    def test_full_pipeline_crisis(self):
        """Test full pipeline for crisis message."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="I want to end my life",
        )
        router = Router()
        route_result = router.route(state)
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
        state.update(agg_result)
        wrapper = SafetyWrapper()
        final_result = wrapper.wrap(state)
        assert "988" in final_result["final_response"]

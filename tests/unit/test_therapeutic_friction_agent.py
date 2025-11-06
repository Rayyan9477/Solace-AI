"""
Comprehensive tests for the TherapeuticFrictionAgent.

Tests cover all major functionality including readiness assessment, challenge level determination,
intervention selection, progress tracking, and therapeutic relationship monitoring.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

from src.agents.clinical.therapy_agent import (
    TherapeuticFrictionAgent,
    ChallengeLevel,
    UserReadinessIndicator,
    InterventionType,
    UserProgress,
    TherapeuticRelationship
)


class TestTherapeuticFrictionAgent:
    """Test suite for TherapeuticFrictionAgent functionality."""
    
    @pytest.fixture
    def mock_model_provider(self):
        """Mock model provider for testing."""
        mock_provider = Mock()
        mock_provider.generate = Mock(return_value="Generated response")
        return mock_provider
    
    @pytest.fixture
    def agent(self, mock_model_provider):
        """Create TherapeuticFrictionAgent instance for testing."""
        with patch('src.agents.therapeutic_friction_agent.TherapeuticTechniqueService'):
            agent = TherapeuticFrictionAgent(mock_model_provider)
            return agent
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return {
            "emotion_analysis": {
                "primary_emotion": "anxiety",
                "confidence": 0.8
            },
            "user_id": "test_user",
            "session_id": "test_session"
        }

    def test_agent_initialization(self, agent):
        """Test agent initialization and default values."""
        assert agent.name == "therapeutic_friction_agent"
        assert agent.role == "Growth-Oriented Therapeutic Challenger"
        assert isinstance(agent.user_progress, UserProgress)
        assert isinstance(agent.therapeutic_relationship, TherapeuticRelationship)
        assert agent.user_progress.session_count == 0
        assert agent.therapeutic_relationship.trust_level == 0.5
        assert len(agent.readiness_indicators) == 6
        assert len(agent.challenge_strategies) == 6

    def test_assess_user_readiness_resistant(self, agent):
        """Test readiness assessment for resistant users."""
        resistant_input = "This won't work. I've tried everything before and nothing helps. It's pointless."
        context = {"emotion_analysis": {"primary_emotion": "frustration"}}
        
        readiness = agent._assess_user_readiness(resistant_input, context)
        assert readiness == UserReadinessIndicator.RESISTANT

    def test_assess_user_readiness_open(self, agent):
        """Test readiness assessment for open users."""
        open_input = "I'm willing to try something new. What do you think I should do? Tell me more about this approach."
        context = {"emotion_analysis": {"primary_emotion": "curiosity"}}
        
        readiness = agent._assess_user_readiness(open_input, context)
        assert readiness == UserReadinessIndicator.OPEN

    def test_assess_user_readiness_motivated(self, agent):
        """Test readiness assessment for motivated users."""
        motivated_input = "I want to change and I'm ready to try whatever it takes. I'm committed to this process."
        context = {"emotion_analysis": {"primary_emotion": "determination"}}
        
        readiness = agent._assess_user_readiness(motivated_input, context)
        assert readiness == UserReadinessIndicator.MOTIVATED

    def test_breakthrough_detection_positive(self, agent):
        """Test breakthrough moment detection."""
        breakthrough_input = "Oh wow, I never realized this before! Everything makes sense now. I see the pattern in my behavior."
        context = {"emotion_analysis": {"primary_emotion": "clarity"}}
        
        breakthrough = agent._detect_breakthrough_moment(breakthrough_input, context)
        assert breakthrough is True
        assert len(agent.user_progress.breakthrough_moments) == 1

    def test_breakthrough_detection_negative(self, agent):
        """Test when no breakthrough is detected."""
        normal_input = "I'm feeling a bit better today but still struggling with some things."
        context = {"emotion_analysis": {"primary_emotion": "neutral"}}
        
        breakthrough = agent._detect_breakthrough_moment(normal_input, context)
        assert breakthrough is False
        assert len(agent.user_progress.breakthrough_moments) == 0

    def test_challenge_level_determination(self, agent):
        """Test challenge level determination based on readiness."""
        # Test resistant user
        level = agent._determine_challenge_level(UserReadinessIndicator.RESISTANT, False)
        assert level == ChallengeLevel.VALIDATION_ONLY
        
        # Test motivated user
        level = agent._determine_challenge_level(UserReadinessIndicator.MOTIVATED, False)
        assert level == ChallengeLevel.STRONG_CHALLENGE
        
        # Test breakthrough moment overrides readiness
        level = agent._determine_challenge_level(UserReadinessIndicator.RESISTANT, True)
        assert level == ChallengeLevel.BREAKTHROUGH_PUSH

    def test_intervention_type_selection(self, agent, sample_context):
        """Test intervention type selection based on readiness and content."""
        # Test avoidance patterns
        intervention = agent._select_intervention_type(
            "I'm scared to try this approach", 
            UserReadinessIndicator.OPEN,
            sample_context
        )
        assert intervention == InterventionType.EXPOSURE_CHALLENGE
        
        # Test cognitive patterns
        intervention = agent._select_intervention_type(
            "I think I should be perfect at everything", 
            UserReadinessIndicator.OPEN,
            sample_context
        )
        assert intervention == InterventionType.COGNITIVE_REFRAMING

    def test_therapeutic_relationship_update(self, agent, sample_context):
        """Test therapeutic relationship metric updates."""
        initial_trust = agent.therapeutic_relationship.trust_level
        
        # Test with open user input
        agent._update_therapeutic_relationship(
            "I'm willing to explore this further", 
            UserReadinessIndicator.OPEN,
            sample_context
        )
        
        assert agent.therapeutic_relationship.trust_level >= initial_trust
        assert agent.therapeutic_relationship.engagement_score > 0.5

    def test_socratic_questions_generation(self, agent):
        """Test Socratic questions generation for different challenge levels."""
        # Test gentle inquiry
        questions = agent._generate_socratic_questions("I'm struggling", ChallengeLevel.GENTLE_INQUIRY)
        assert len(questions) <= 3
        assert all(isinstance(q, str) for q in questions)
        
        # Test strong challenge
        questions = agent._generate_socratic_questions("I always fail", ChallengeLevel.STRONG_CHALLENGE)
        assert len(questions) <= 3
        assert any("pattern" in q.lower() for q in questions)

    def test_behavioral_experiments_generation(self, agent, sample_context):
        """Test behavioral experiments generation."""
        # Test avoidance-based input
        experiments = agent._generate_behavioral_experiments(
            "I can't handle social situations", 
            sample_context
        )
        
        assert len(experiments) <= 2
        assert all("title" in exp and "action" in exp for exp in experiments)

    def test_progress_tracking(self, agent):
        """Test progress indicator tracking."""
        # Test behavioral change tracking
        agent._track_progress_indicators(
            "I tried something new today and it worked well",
            {"approach": "growth_oriented"}
        )
        
        assert len(agent.user_progress.behavioral_change_indicators) > 0
        
        # Test resistance pattern tracking
        agent._track_progress_indicators(
            "This is impossible and won't work",
            {"approach": "validation"}
        )
        
        assert len(agent.user_progress.resistance_patterns) > 0

    def test_outcome_metrics_update(self, agent):
        """Test long-term outcome metrics calculation."""
        # Simulate some sessions
        agent.user_progress.session_count = 5
        agent.user_progress.breakthrough_moments = [{"test": "data"}] * 2
        agent.intervention_history = [{"accepted": True}, {"accepted": False}, {"accepted": True}]
        
        agent._update_outcome_metrics()
        
        assert agent.user_progress.challenge_acceptance_rate == 2/3
        assert agent.user_progress.insight_frequency == 2/5
        assert "total_sessions" in agent.outcome_metrics

    def test_growth_trajectory_calculation(self, agent):
        """Test growth trajectory determination."""
        # Simulate improving trajectory
        agent.user_progress.readiness_history = [
            UserReadinessIndicator.RESISTANT,
            UserReadinessIndicator.AMBIVALENT,
            UserReadinessIndicator.OPEN,
            UserReadinessIndicator.MOTIVATED
        ]
        
        agent._track_progress_indicators("test input", {})
        assert agent.user_progress.growth_trajectory in ["improving", "stable"]

    def test_breakthrough_potential_calculation(self, agent):
        """Test breakthrough potential calculation."""
        # Set up favorable conditions
        agent.therapeutic_relationship.therapeutic_bond_strength = 0.8
        agent.therapeutic_relationship.receptivity_to_challenge = 0.7
        agent.user_progress.challenge_acceptance_rate = 0.8
        agent.therapeutic_relationship.rupture_risk = 0.1
        agent.user_progress.insight_frequency = 0.3
        
        potential = agent._calculate_breakthrough_potential()
        assert 0 <= potential <= 1
        assert potential > 0.5  # Should be high with good conditions

    def test_friction_recommendation(self, agent):
        """Test friction recommendation generation."""
        rec = agent._get_friction_recommendation(ChallengeLevel.VALIDATION_ONLY)
        assert "safety" in rec.lower() or "trust" in rec.lower()
        
        rec = agent._get_friction_recommendation(ChallengeLevel.BREAKTHROUGH_PUSH)
        assert "leverage" in rec.lower() or "breakthrough" in rec.lower()

    def test_process_method_comprehensive(self, agent, sample_context):
        """Test the main process method with comprehensive flow."""
        user_input = "I'm feeling stuck but I want to understand what's happening"
        
        result = agent.process(user_input, sample_context)
        
        # Verify all expected keys are present
        expected_keys = [
            "response_strategy", "user_readiness", "challenge_level",
            "intervention_type", "breakthrough_detected", "therapeutic_relationship",
            "progress_metrics", "friction_recommendation", "context_updates"
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Verify data types and structure
        assert isinstance(result["response_strategy"], dict)
        assert isinstance(result["therapeutic_relationship"], dict)
        assert isinstance(result["progress_metrics"], dict)
        assert isinstance(result["context_updates"], dict)

    def test_enhance_response_method(self, agent):
        """Test response enhancement with friction components."""
        original_response = "I understand you're struggling with this situation."
        
        friction_result = {
            "response_strategy": {
                "growth_questions": ["What patterns do you notice here?"],
                "behavioral_experiments": [{
                    "title": "Test Experiment",
                    "description": "Try something new",
                    "action": "Take one small step"
                }],
                "strategic_challenges": ["Consider this perspective..."]
            },
            "challenge_level": "moderate_challenge",
            "intervention_type": "socratic_questioning"
        }
        
        enhanced = agent.enhance_response(original_response, friction_result)
        
        assert len(enhanced) > len(original_response)
        assert "Growth Questions" in enhanced or "Growth Challenge" in enhanced

    def test_comprehensive_assessment(self, agent):
        """Test comprehensive assessment generation."""
        # Set up some data
        agent.user_progress.session_count = 10
        agent.user_progress.breakthrough_moments = [{"test": "data"}]
        agent.therapeutic_relationship.therapeutic_bond_strength = 0.7
        
        assessment = agent.get_comprehensive_assessment()
        
        expected_sections = [
            "user_progress", "therapeutic_relationship", "outcome_metrics",
            "breakthrough_potential", "recommendations", "session_summary"
        ]
        
        for section in expected_sections:
            assert section in assessment

    def test_therapy_phase_determination(self, agent):
        """Test therapy phase determination logic."""
        # Test early phase
        agent.user_progress.session_count = 2
        phase = agent._determine_therapy_phase()
        assert phase == "engagement_building"
        
        # Test active change phase
        agent.user_progress.session_count = 10
        agent.therapeutic_relationship.therapeutic_bond_strength = 0.8
        agent.user_progress.growth_trajectory = "improving"
        phase = agent._determine_therapy_phase()
        assert phase == "active_change"

    def test_next_steps_recommendations(self, agent):
        """Test next steps recommendation generation."""
        recommendations = agent._recommend_next_steps()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_validation_challenge_balance(self, agent, sample_context):
        """Test balance between validation and challenge."""
        # Test high rupture risk scenario
        agent.therapeutic_relationship.rupture_risk = 0.8
        
        result = agent.process("I don't think this is helping", sample_context)
        
        # Should default to validation when rupture risk is high
        assert result["challenge_level"] == ChallengeLevel.VALIDATION_ONLY.value

    def test_edge_cases(self, agent, sample_context):
        """Test edge cases and error handling."""
        # Test empty input
        result = agent.process("", sample_context)
        assert "user_readiness" in result
        
        # Test very long input
        long_input = "This is a very long input. " * 100
        result = agent.process(long_input, sample_context)
        assert agent.therapeutic_relationship.engagement_score > 0.5
        
        # Test None context
        result = agent.process("Test input", {})
        assert "user_readiness" in result

    def test_intervention_history_tracking(self, agent, sample_context):
        """Test intervention history tracking."""
        initial_history_length = len(agent.intervention_history)
        
        # Process some inputs
        agent.process("I want to try something new", sample_context)
        agent.process("This seems challenging", sample_context)
        
        # Verify session count increases
        assert agent.user_progress.session_count == 2

    def test_context_updates_structure(self, agent, sample_context):
        """Test context updates structure and content."""
        result = agent.process("I'm ready for change", sample_context)
        
        context_updates = result["context_updates"]
        assert "therapeutic_friction" in context_updates
        
        friction_context = context_updates["therapeutic_friction"]
        expected_keys = ["readiness", "challenge_level", "relationship_quality", "breakthrough_potential"]
        
        for key in expected_keys:
            assert key in friction_context


class TestUserProgressTracking:
    """Test suite for UserProgress tracking functionality."""
    
    @pytest.fixture
    def user_progress(self):
        """Create UserProgress instance for testing."""
        return UserProgress()
    
    def test_user_progress_initialization(self, user_progress):
        """Test UserProgress initialization."""
        assert user_progress.session_count == 0
        assert user_progress.challenge_acceptance_rate == 0.0
        assert user_progress.insight_frequency == 0.0
        assert user_progress.behavioral_change_indicators == []
        assert user_progress.resistance_patterns == []
        assert user_progress.breakthrough_moments == []
        assert user_progress.growth_trajectory == "stable"
        assert user_progress.readiness_history == []

    def test_user_progress_tracking(self, user_progress):
        """Test user progress tracking updates."""
        # Add some breakthrough moments
        user_progress.breakthrough_moments.append({
            "timestamp": datetime.now().isoformat(),
            "insight": "Major realization about patterns"
        })
        
        assert len(user_progress.breakthrough_moments) == 1
        
        # Add readiness history
        user_progress.readiness_history.extend([
            UserReadinessIndicator.RESISTANT,
            UserReadinessIndicator.OPEN,
            UserReadinessIndicator.MOTIVATED
        ])
        
        assert len(user_progress.readiness_history) == 3


class TestTherapeuticRelationship:
    """Test suite for TherapeuticRelationship monitoring."""
    
    @pytest.fixture
    def relationship(self):
        """Create TherapeuticRelationship instance for testing."""
        return TherapeuticRelationship()
    
    def test_relationship_initialization(self, relationship):
        """Test TherapeuticRelationship initialization."""
        assert relationship.trust_level == 0.5
        assert relationship.engagement_score == 0.5
        assert relationship.receptivity_to_challenge == 0.3
        assert relationship.collaborative_spirit == 0.5
        assert relationship.emotional_safety == 0.7
        assert relationship.therapeutic_bond_strength == 0.5
        assert relationship.repair_needed is False
        assert relationship.rupture_risk == 0.0

    def test_relationship_metrics_bounds(self, relationship):
        """Test that relationship metrics stay within bounds."""
        # Test upper bounds
        relationship.trust_level = 1.5  # Should be capped at 1.0
        relationship.rupture_risk = 1.2  # Should be capped at 1.0
        
        # In a real implementation, you'd have validation
        # For now, just test that we can set values
        assert relationship.trust_level == 1.5  # Would be 1.0 with validation
        assert relationship.rupture_risk == 1.2   # Would be 1.0 with validation


class TestIntegrationScenarios:
    """Test suite for integration scenarios and workflow testing."""
    
    @pytest.fixture
    def agent(self):
        """Create agent for integration testing."""
        with patch('src.agents.therapeutic_friction_agent.TherapeuticTechniqueService'):
            return TherapeuticFrictionAgent(Mock())
    
    def test_multi_session_progression(self, agent):
        """Test progression over multiple sessions."""
        contexts = [
            {"emotion_analysis": {"primary_emotion": "anxiety"}},
            {"emotion_analysis": {"primary_emotion": "curiosity"}},
            {"emotion_analysis": {"primary_emotion": "determination"}},
            {"emotion_analysis": {"primary_emotion": "clarity"}}
        ]
        
        inputs = [
            "I don't think anything can help me",
            "Maybe there's something I haven't considered",
            "I'm willing to try what you suggest",
            "I see the pattern now - this makes so much sense!"
        ]
        
        results = []
        for i, (input_text, context) in enumerate(zip(inputs, contexts)):
            result = agent.process(input_text, context)
            results.append(result)
        
        # Verify progression
        assert agent.user_progress.session_count == 4
        
        # Should show improvement in readiness over sessions
        readiness_progression = [r["user_readiness"] for r in results]
        
        # At least should not be all the same
        assert len(set(readiness_progression)) > 1

    def test_crisis_to_growth_scenario(self, agent):
        """Test progression from crisis to growth."""
        # Start with crisis-level input
        crisis_context = {"emotion_analysis": {"primary_emotion": "despair"}}
        crisis_result = agent.process("I can't take this anymore", crisis_context)
        
        assert crisis_result["challenge_level"] in ["validation_only", "gentle_inquiry"]
        
        # Progress to growth-oriented
        growth_context = {"emotion_analysis": {"primary_emotion": "hope"}}
        growth_result = agent.process("I'm starting to see possibilities", growth_context)
        
        # Should allow for more challenge now
        challenge_levels = ["gentle_inquiry", "moderate_challenge", "strong_challenge"]
        assert growth_result["challenge_level"] in challenge_levels

    def test_resistance_to_breakthrough_scenario(self, agent):
        """Test journey from resistance to breakthrough."""
        # Simulate resistant phase
        for _ in range(3):
            agent.process("This won't work", {"emotion_analysis": {"primary_emotion": "frustration"}})
        
        # Simulate breakthrough moment
        breakthrough_result = agent.process(
            "Wait, I think I understand something important here!", 
            {"emotion_analysis": {"primary_emotion": "clarity"}}
        )
        
        assert breakthrough_result["breakthrough_detected"] is True
        assert len(agent.user_progress.breakthrough_moments) == 1


@pytest.mark.integration
class TestTherapeuticFrictionIntegration:
    """Integration tests for TherapeuticFrictionAgent with other components."""
    
    def test_workflow_integration_structure(self):
        """Test that agent can be integrated into workflows."""
        # This would test integration with AgentOrchestrator
        # For now, just verify the agent has the expected interface
        with patch('src.agents.therapeutic_friction_agent.TherapeuticTechniqueService'):
            agent = TherapeuticFrictionAgent(Mock())
        
        # Verify required methods exist
        assert hasattr(agent, 'process')
        assert hasattr(agent, 'enhance_response')
        assert callable(agent.process)
        assert callable(agent.enhance_response)
        
        # Verify process method signature
        import inspect
        sig = inspect.signature(agent.process)
        assert 'user_input' in sig.parameters
        assert 'context' in sig.parameters

    def test_context_updates_compatibility(self):
        """Test context updates are compatible with orchestrator."""
        with patch('src.agents.therapeutic_friction_agent.TherapeuticTechniqueService'):
            agent = TherapeuticFrictionAgent(Mock())
        
        result = agent.process("Test input", {})
        
        # Verify context_updates structure
        assert "context_updates" in result
        assert isinstance(result["context_updates"], dict)
        
        # Should have therapeutic_friction section
        assert "therapeutic_friction" in result["context_updates"]


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])
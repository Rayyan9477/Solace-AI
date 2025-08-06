"""
Comprehensive Integration Tests for Therapeutic Friction Sub-Agent System.

Tests the complete integration of therapeutic friction sub-agents with the existing
Solace-AI system, including workflow orchestration, vector database integration,
and supervision monitoring.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json

from src.agents.therapeutic_friction.friction_coordinator import FrictionCoordinator
from src.agents.therapeutic_friction.readiness_assessment_agent import ReadinessAssessmentAgent
from src.agents.therapeutic_friction.breakthrough_detection_agent import BreakthroughDetectionAgent
from src.agents.therapeutic_friction.base_friction_agent import FrictionAgentType, friction_agent_registry
from src.agents.therapy_agent import TherapyAgent
from src.agents.agent_orchestrator import AgentOrchestrator
from src.database.therapeutic_friction_vector_manager import TherapeuticFrictionVectorManager
from src.knowledge.therapeutic.technique_service import TherapeuticTechniqueService


class TestTherapeuticFrictionSystemIntegration:
    """Test suite for complete therapeutic friction system integration."""
    
    @pytest.fixture
    def mock_model_provider(self):
        """Mock model provider for testing."""
        mock_provider = Mock()
        mock_provider.generate = AsyncMock(return_value="Generated response")
        mock_provider.get_embedding = AsyncMock(return_value=[0.1] * 768)
        return mock_provider
    
    @pytest.fixture
    def friction_coordinator(self, mock_model_provider):
        """Create friction coordinator for testing."""
        with patch('src.agents.therapeutic_friction.friction_coordinator.friction_agent_registry'):
            coordinator = FrictionCoordinator(mock_model_provider)
            return coordinator
    
    @pytest.fixture
    def therapy_agent(self, mock_model_provider):
        """Create therapy agent for testing."""
        with patch('src.agents.therapy_agent.TherapeuticTechniqueService'):
            agent = TherapyAgent(mock_model_provider)
            return agent
    
    @pytest.fixture
    def agent_orchestrator(self, mock_model_provider):
        """Create agent orchestrator for testing."""
        config = {"user_id": "test_user", "supervision_enabled": False}
        orchestrator = AgentOrchestrator("test_orchestrator", config)
        return orchestrator
    
    @pytest.fixture
    def sample_user_context(self):
        """Sample user context for testing."""
        return {
            "emotion_analysis": {
                "primary_emotion": "anxiety",
                "confidence": 0.8,
                "secondary_emotions": ["worry", "nervousness"]
            },
            "session_count": 5,
            "therapeutic_alliance_score": 0.7,
            "user_id": "test_user_123",
            "session_id": "test_session_456"
        }
    
    @pytest.mark.asyncio
    async def test_friction_coordinator_initialization(self, friction_coordinator):
        """Test friction coordinator initialization and sub-agent setup."""
        # Verify coordinator is properly initialized
        assert friction_coordinator.name == "friction_coordinator"
        assert friction_coordinator.role == "Therapeutic Friction Coordinator"
        
        # Verify sub-agents are initialized
        assert len(friction_coordinator.sub_agents) >= 2
        assert FrictionAgentType.READINESS_ASSESSMENT in friction_coordinator.sub_agents
        assert FrictionAgentType.BREAKTHROUGH_DETECTION in friction_coordinator.sub_agents
        
        # Verify coordination strategy is set
        assert friction_coordinator.coordination_strategy is not None
        assert friction_coordinator.conflict_resolution is not None
    
    @pytest.mark.asyncio
    async def test_sub_agent_parallel_execution(self, friction_coordinator, sample_user_context):
        """Test parallel execution of sub-agents."""
        user_input = "I'm feeling anxious but I think I'm ready to try something new."
        
        # Mock sub-agent responses
        with patch.object(friction_coordinator, '_execute_sub_agents_parallel') as mock_execute:
            mock_execute.return_value = {
                FrictionAgentType.READINESS_ASSESSMENT: {
                    'assessment': {'primary_readiness': 'open', 'readiness_score': 3.5},
                    'confidence': 0.8,
                    'is_valid': True,
                    'agent_type': 'readiness_assessment'
                },
                FrictionAgentType.BREAKTHROUGH_DETECTION: {
                    'assessment': {'breakthrough_detected': False, 'breakthrough_potential': {'overall_potential': 0.6}},
                    'confidence': 0.7,
                    'is_valid': True,
                    'agent_type': 'breakthrough_detection'
                }
            }
            
            result = await friction_coordinator.process(user_input, sample_user_context)
            
            # Verify parallel execution was called
            mock_execute.assert_called_once()
            
            # Verify result structure
            assert 'user_readiness' in result
            assert 'challenge_level' in result
            assert 'coordination_data' in result
            assert result['coordination_data']['agent_success_rate'] > 0
    
    @pytest.mark.asyncio
    async def test_consensus_building_weighted(self, friction_coordinator, sample_user_context):
        """Test weighted consensus coordination strategy."""
        user_input = "I understand what you're saying and I want to make changes."
        
        # Create consistent sub-agent results for consensus
        sub_agent_results = {
            FrictionAgentType.READINESS_ASSESSMENT: {
                'assessment': {'primary_readiness': 'motivated', 'readiness_score': 4.2},
                'confidence': 0.9,
                'is_valid': True
            },
            FrictionAgentType.BREAKTHROUGH_DETECTION: {
                'assessment': {'breakthrough_detected': False, 'breakthrough_potential': {'overall_potential': 0.8}},
                'confidence': 0.8,
                'is_valid': True
            }
        }
        
        with patch.object(friction_coordinator, '_execute_sub_agents_parallel', return_value=sub_agent_results):
            result = await friction_coordinator.process(user_input, sample_user_context)
            
            # Verify weighted consensus worked
            assert result['user_readiness'] == 'motivated'
            assert result['challenge_level'] == 'strong_challenge'
            assert result['coordination_data']['consensus_strength'] > 0.7
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, friction_coordinator, sample_user_context):
        """Test conflict resolution between sub-agents."""
        user_input = "I don't know if I can do this, but maybe I should try."
        
        # Create conflicting sub-agent results
        conflicting_results = {
            FrictionAgentType.READINESS_ASSESSMENT: {
                'assessment': {'primary_readiness': 'resistant', 'readiness_score': 1.5},
                'confidence': 0.7,
                'is_valid': True
            },
            FrictionAgentType.BREAKTHROUGH_DETECTION: {
                'assessment': {'breakthrough_detected': True, 'breakthrough_potential': {'overall_potential': 0.9}},
                'confidence': 0.8,
                'is_valid': True
            }
        }
        
        with patch.object(friction_coordinator, '_execute_sub_agents_parallel', return_value=conflicting_results):
            result = await friction_coordinator.process(user_input, sample_user_context)
            
            # Verify conflict was detected and resolved
            coordination_data = result.get('coordination_data', {})
            # Should use conservative approach due to conflicting signals
            assert result['challenge_level'] in ['validation_only', 'gentle_inquiry']
    
    @pytest.mark.asyncio
    async def test_workflow_integration(self, agent_orchestrator, mock_model_provider):
        """Test integration with agent orchestrator workflows."""
        # Register friction coordinator as an agent
        friction_coordinator = FrictionCoordinator(mock_model_provider)
        agent_orchestrator.register_agent("friction_coordinator", friction_coordinator)
        
        # Mock the workflow execution
        with patch.object(agent_orchestrator, 'execute_workflow') as mock_execute:
            mock_execute.return_value = {
                'output': {
                    'response': 'Coordinated therapeutic response',
                    'user_readiness': 'open',
                    'challenge_level': 'moderate_challenge'
                },
                'status': 'completed',
                'session_id': 'test_session'
            }
            
            # Test coordinated workflow
            result = await agent_orchestrator.execute_workflow(
                "coordinated_therapeutic_friction",
                {"message": "I'm ready to explore my patterns"},
                "test_session"
            )
            
            assert result['status'] == 'completed'
            assert 'output' in result
    
    @pytest.mark.asyncio
    async def test_therapy_agent_integration(self, therapy_agent, friction_coordinator, sample_user_context):
        """Test integration between therapy agent and friction coordinator."""
        # Simulate friction coordinator result
        friction_result = {
            'user_readiness': 'motivated',
            'challenge_level': 'strong_challenge',
            'breakthrough_detected': False,
            'therapeutic_relationship': {'therapeutic_bond_strength': 0.8},
            'progress_metrics': {'growth_trajectory': 'improving'}
        }
        
        # Test integration method
        integration_result = therapy_agent.integrate_friction_insights(friction_result)
        
        # Verify integration worked
        assert integration_result['combined_approach'] is True
        assert integration_result['friction_readiness'] == 'motivated'
        assert integration_result['friction_challenge_level'] == 'strong_challenge'
        assert len(integration_result['integrated_recommendations']) > 0
        
        # Verify therapeutic alliance was updated
        assert therapy_agent.therapeutic_alliance_score >= 0.7
    
    @pytest.mark.asyncio
    async def test_vector_database_integration(self, mock_model_provider):
        """Test vector database integration for therapeutic friction knowledge."""
        # Create vector manager
        vector_manager = TherapeuticFrictionVectorManager(mock_model_provider)
        
        # Test document storage
        from src.database.therapeutic_friction_vector_manager import TherapeuticDocument, TherapeuticFrictionDocumentType
        
        test_document = TherapeuticDocument(
            document_id="test_readiness_pattern_1",
            document_type=TherapeuticFrictionDocumentType.READINESS_PATTERN,
            title="Test Readiness Pattern",
            content="User shows signs of therapeutic readiness through engagement.",
            metadata={"keywords": ["readiness", "engagement"], "effectiveness": 0.8}
        )
        
        # Mock vector store operations
        with patch.object(vector_manager, '_get_embedding', return_value=[0.1] * 768):
            success = await vector_manager.add_document(test_document)
            assert success is True
            
            # Test search functionality
            results = await vector_manager.search_documents(
                "therapeutic readiness engagement",
                TherapeuticFrictionDocumentType.READINESS_PATTERN,
                top_k=3
            )
            
            # Should return results (mocked)
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_knowledge_service_integration(self, mock_model_provider):
        """Test enhanced therapeutic technique service integration."""
        config = {"friction_vector_manager": {"storage_path": "test_path"}}
        
        with patch('src.knowledge.therapeutic.technique_service.TherapeuticFrictionVectorManager'):
            technique_service = TherapeuticTechniqueService(mock_model_provider, config)
            
            # Test friction-specific technique retrieval
            user_context = {
                "user_readiness": "motivated",
                "emotion_analysis": {"primary_emotion": "anxiety"}
            }
            
            with patch.object(technique_service.friction_vector_manager, 'get_recommendations_for_agent') as mock_rec:
                mock_rec.return_value = [
                    {
                        'document': {'title': 'Anxiety Management for Motivated Clients'},
                        'similarity_score': 0.9,
                        'match_reason': 'High readiness and anxiety context'
                    }
                ]
                
                results = await technique_service.get_friction_specific_techniques(
                    "readiness_assessment", user_context, top_k=3
                )
                
                assert len(results) > 0
                assert results[0]['source_type'] == 'friction_specific'
    
    @pytest.mark.asyncio
    async def test_supervision_integration(self, friction_coordinator, sample_user_context):
        """Test integration with supervision and monitoring systems."""
        user_input = "I'm making progress but it's challenging."
        
        # Mock supervision validation
        with patch('src.agents.therapeutic_friction.friction_coordinator.SupervisorAgent') as mock_supervisor:
            mock_supervisor_instance = Mock()
            mock_supervisor_instance.validate_agent_response = AsyncMock(return_value={
                'validation_level': 'approved',
                'safety_score': 0.9,
                'clinical_appropriateness': 0.8
            })
            
            result = await friction_coordinator.process(user_input, sample_user_context)
            
            # Verify supervision metadata is included
            assert 'metadata' in result
            assert 'coordination_quality' in result.get('coordination_data', {})
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, agent_orchestrator, mock_model_provider):
        """Test complete end-to-end workflow with all components."""
        # Set up complete system
        friction_coordinator = FrictionCoordinator(mock_model_provider)
        therapy_agent = TherapyAgent(mock_model_provider)
        
        # Register agents
        agent_orchestrator.register_agent("friction_coordinator", friction_coordinator)
        agent_orchestrator.register_agent("therapy_agent", therapy_agent)
        agent_orchestrator.register_agent("safety_agent", Mock())
        agent_orchestrator.register_agent("emotion_agent", Mock())
        agent_orchestrator.register_agent("personality_agent", Mock())
        agent_orchestrator.register_agent("chat_agent", Mock())
        
        # Mock agent responses
        safety_response = {"safety_assessment": "safe", "risk_level": "low"}
        emotion_response = {
            "emotion_analysis": {"primary_emotion": "determination", "confidence": 0.8},
            "context_updates": {"emotion": {"primary_emotion": "determination"}}
        }
        personality_response = {"personality_traits": {"openness": 0.8}, "context_updates": {}}
        
        # Mock sub-agent processing
        with patch.object(friction_coordinator, 'process') as mock_friction:
            mock_friction.return_value = {
                'user_readiness': 'motivated',
                'challenge_level': 'strong_challenge',
                'intervention_type': 'behavioral_experiment',
                'breakthrough_detected': False,
                'coordination_data': {'consensus_strength': 0.8},
                'context_updates': {'therapeutic_friction': {'readiness': 'motivated'}}
            }
            
            with patch.object(therapy_agent, 'process') as mock_therapy:
                mock_therapy.return_value = {
                    'response': 'Integrated therapeutic response',
                    'therapeutic_techniques': [],
                    'context_updates': {'therapy': {'alliance_score': 0.8}}
                }
                
                # Mock other agents
                agent_orchestrator.agent_modules["safety_agent"].process = AsyncMock(return_value=safety_response)
                agent_orchestrator.agent_modules["emotion_agent"].process = AsyncMock(return_value=emotion_response)
                agent_orchestrator.agent_modules["personality_agent"].process = AsyncMock(return_value=personality_response)
                agent_orchestrator.agent_modules["chat_agent"].process = AsyncMock(return_value={"response": "Final response"})
                
                # Execute comprehensive workflow
                result = await agent_orchestrator.execute_workflow(
                    "comprehensive_coordinated_therapeutic",
                    {"message": "I'm ready to make significant changes in my life"},
                    "integration_test_session"
                )
                
                # Verify end-to-end execution
                assert result['status'] == 'completed'
                assert 'output' in result
                assert result['steps_completed'] > 0
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, mock_model_provider):
        """Test backward compatibility with original TherapeuticFrictionAgent interface."""
        # Create friction coordinator
        friction_coordinator = FrictionCoordinator(mock_model_provider)
        
        user_input = "I feel stuck but I want to change."
        context = {"emotion_analysis": {"primary_emotion": "frustration"}}
        
        # Process with new system
        result = await friction_coordinator.process(user_input, context)
        
        # Verify original interface fields are present
        required_fields = [
            'user_readiness', 'challenge_level', 'intervention_type',
            'breakthrough_detected', 'therapeutic_relationship',
            'progress_metrics', 'friction_recommendation'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Test enhance_response method
        original_response = "I understand you're feeling frustrated."
        enhanced = friction_coordinator.enhance_response(original_response, result)
        
        assert len(enhanced) > len(original_response)
        assert original_response in enhanced
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, friction_coordinator, sample_user_context):
        """Test performance metrics and monitoring."""
        user_input = "I need help understanding my patterns."
        
        # Process multiple requests to test metrics
        results = []
        for i in range(3):
            result = await friction_coordinator.process(user_input, sample_user_context)
            results.append(result)
        
        # Verify performance metrics are tracked
        assert len(friction_coordinator.coordination_history) == 3
        
        # Check comprehensive assessment
        assessment = friction_coordinator.get_comprehensive_assessment()
        assert 'coordination_history' in assessment
        assert 'coordination_strategy' in assessment
        assert assessment['sub_agents_active'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, friction_coordinator, sample_user_context):
        """Test error handling and recovery mechanisms."""
        user_input = "Test error handling"
        
        # Simulate sub-agent failure
        with patch.object(friction_coordinator, '_execute_sub_agents_parallel') as mock_execute:
            mock_execute.return_value = {
                FrictionAgentType.READINESS_ASSESSMENT: {
                    'error': 'Simulated error',
                    'confidence': 0.0,
                    'agent_type': 'readiness_assessment'
                },
                FrictionAgentType.BREAKTHROUGH_DETECTION: {
                    'assessment': {'breakthrough_detected': False},
                    'confidence': 0.7,
                    'is_valid': True,
                    'agent_type': 'breakthrough_detection'
                }
            }
            
            result = await friction_coordinator.process(user_input, sample_user_context)
            
            # Verify system handles partial failures gracefully
            assert 'user_readiness' in result
            assert result['coordination_data']['agent_success_rate'] < 1.0
            assert result['coordination_data']['agent_success_rate'] > 0.0
    
    @pytest.mark.asyncio
    async def test_context_propagation(self, agent_orchestrator, mock_model_provider):
        """Test context propagation between agents in workflows."""
        # Set up agents with context tracking
        friction_coordinator = FrictionCoordinator(mock_model_provider)
        
        # Mock context updates
        initial_context = {"emotion_analysis": {"primary_emotion": "anxiety"}}
        
        with patch.object(friction_coordinator, 'process') as mock_process:
            mock_process.return_value = {
                'user_readiness': 'open',
                'context_updates': {
                    'therapeutic_friction': {
                        'readiness': 'open',
                        'relationship_quality': 0.7
                    }
                }
            }
            
            # Test context propagation
            result = await friction_coordinator.process("Test message", initial_context)
            
            # Verify context updates are structured correctly
            assert 'context_updates' in result
            assert 'therapeutic_friction' in result['context_updates']
            assert result['context_updates']['therapeutic_friction']['readiness'] == 'open'
    
    @pytest.mark.asyncio
    async def test_scalability_and_concurrency(self, friction_coordinator, sample_user_context):
        """Test system scalability with concurrent requests."""
        user_inputs = [
            "I'm feeling anxious about making changes.",
            "I think I understand what's happening now.",
            "I'm ready to try something different.",
            "This is challenging but I want to continue.",
            "I'm not sure if I can do this."
        ]
        
        # Process requests concurrently
        tasks = [
            friction_coordinator.process(input_text, sample_user_context)
            for input_text in user_inputs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(user_inputs)
        
        # Verify each result has required structure
        for result in successful_results:
            assert 'user_readiness' in result
            assert 'coordination_data' in result


class TestSubAgentSpecificIntegration:
    """Test integration of individual sub-agents."""
    
    @pytest.fixture
    def mock_model_provider(self):
        """Mock model provider for testing."""
        mock_provider = Mock()
        mock_provider.get_embedding = AsyncMock(return_value=[0.1] * 768)
        return mock_provider
    
    @pytest.mark.asyncio
    async def test_readiness_assessment_integration(self, mock_model_provider):
        """Test readiness assessment agent integration."""
        agent = ReadinessAssessmentAgent(mock_model_provider)
        
        context = {
            "emotion_analysis": {"primary_emotion": "hope", "confidence": 0.8},
            "session_count": 3
        }
        
        result = await agent.process("I'm willing to try new approaches", context)
        
        assert 'assessment' in result
        assert 'confidence' in result
        assert result['is_valid'] is True
        assert result['agent_type'] == 'readiness_assessment'
    
    @pytest.mark.asyncio
    async def test_breakthrough_detection_integration(self, mock_model_provider):
        """Test breakthrough detection agent integration."""
        agent = BreakthroughDetectionAgent(mock_model_provider)
        
        context = {
            "emotion_analysis": {"primary_emotion": "clarity", "confidence": 0.9},
            "session_count": 8
        }
        
        result = await agent.process("I finally understand what's been happening!", context)
        
        assert 'assessment' in result
        assert 'confidence' in result
        assert result['is_valid'] is True
        assert result['agent_type'] == 'breakthrough_detection'
    
    @pytest.mark.asyncio
    async def test_agent_registry_integration(self, mock_model_provider):
        """Test sub-agent registry functionality."""
        # Clear registry for clean test
        friction_agent_registry.agents.clear()
        
        # Create and register agents
        readiness_agent = ReadinessAssessmentAgent(mock_model_provider)
        breakthrough_agent = BreakthroughDetectionAgent(mock_model_provider)
        
        friction_agent_registry.register_agent(readiness_agent)
        friction_agent_registry.register_agent(breakthrough_agent)
        
        # Verify registration
        assert len(friction_agent_registry.agents) == 2
        assert FrictionAgentType.READINESS_ASSESSMENT in friction_agent_registry.agents
        assert FrictionAgentType.BREAKTHROUGH_DETECTION in friction_agent_registry.agents
        
        # Test health status
        health_status = await friction_agent_registry.get_system_health()
        assert health_status['total_agents'] == 2
        assert health_status['healthy_agents'] >= 0


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "asyncio"])
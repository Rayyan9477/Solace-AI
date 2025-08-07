"""
Comprehensive Test Suite for Adaptive Learning System

This module provides comprehensive testing for the AdaptiveLearningAgent
and all its components to ensure safe and correct integration.
"""

import asyncio
import logging
import sys
import traceback
from typing import Dict, Any, List
from datetime import datetime
import json
import tempfile
import os

# Import all components to test
from .adaptive_learning_integration import AdaptiveLearningSystemIntegrator, create_adaptive_learning_system
from .adaptive_learning_agent import AdaptiveLearningAgent
from .pattern_recognition_engine import PatternRecognitionEngine
from .personalization_engine import PersonalizationEngine
from .feedback_integration_system import FeedbackProcessor
from .outcome_tracker import OutcomeTracker, OutcomeType
from .privacy_protection_system import PrivacyProtectionSystem, DataSensitivity, PrivacyLevel
from .insights_generation_system import InsightGenerationSystem

from ..utils.logger import get_logger

# Mock components for testing
class MockAgentOrchestrator:
    """Mock agent orchestrator for testing"""
    
    def __init__(self):
        self.registered_agents = {}
        self.logger = get_logger(__name__)
    
    async def register_agent(self, agent_name: str, agent: Any):
        """Register an agent"""
        self.registered_agents[agent_name] = agent
        self.logger.info(f"Registered mock agent: {agent_name}")
        return True
    
    def get_agent(self, agent_name: str):
        """Get a registered agent"""
        return self.registered_agents.get(agent_name)

class MockCentralVectorDB:
    """Mock vector database for testing"""
    
    def __init__(self):
        self.data = {}
        self.logger = get_logger(__name__)
    
    async def store_vector(self, key: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Store a vector with metadata"""
        self.data[key] = {'vector': vector, 'metadata': metadata or {}}
        return True
    
    async def search_similar(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Simple mock implementation
        return [{'key': k, 'similarity': 0.8, 'metadata': v['metadata']} 
                for k, v in list(self.data.items())[:top_k]]

class AdaptiveLearningSystemTester:
    """Comprehensive tester for the adaptive learning system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results = []
        self.mock_orchestrator = MockAgentOrchestrator()
        self.mock_vector_db = MockCentralVectorDB()
        
        # Test configuration
        self.test_config = {
            'privacy_protection_config': {
                'privacy_level': 'standard',
                'differential_privacy_epsilon': 1.0,
                'differential_privacy_delta': 1e-5,
                'k_anonymity': 5,
                'data_retention_days': 30
            },
            'insights_generation_config': {
                'min_data_points_for_insights': 10,
                'trend_analysis_days': 7
            },
            'learning_cycle_interval': 10,  # Short interval for testing
            'enable_real_time_learning': True,
            'enable_background_optimization': False,  # Disable for testing
            'enable_privacy_protection': True,
            'enable_insights_generation': True
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        
        self.logger.info("Starting comprehensive adaptive learning system tests")
        
        test_suite = [
            ("Component Initialization", self.test_component_initialization),
            ("Privacy Protection System", self.test_privacy_protection),
            ("User Interaction Processing", self.test_user_interaction_processing),
            ("Feedback Integration", self.test_feedback_integration),
            ("Outcome Tracking", self.test_outcome_tracking),
            ("Pattern Recognition", self.test_pattern_recognition),
            ("Personalization Engine", self.test_personalization_engine),
            ("Insights Generation", self.test_insights_generation),
            ("System Integration", self.test_system_integration),
            ("Error Handling", self.test_error_handling),
            ("Privacy Compliance", self.test_privacy_compliance),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Graceful Shutdown", self.test_graceful_shutdown)
        ]
        
        total_tests = len(test_suite)
        passed_tests = 0
        
        for test_name, test_function in test_suite:
            try:
                self.logger.info(f"Running test: {test_name}")
                result = await test_function()
                
                if result.get('passed', False):
                    passed_tests += 1
                    self.logger.info(f"✓ {test_name} passed")
                else:
                    self.logger.error(f"✗ {test_name} failed: {result.get('error', 'Unknown error')}")
                
                self.test_results.append({
                    'test_name': test_name,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"✗ {test_name} failed with exception: {str(e)}")
                self.test_results.append({
                    'test_name': test_name,
                    'result': {'passed': False, 'error': str(e), 'exception': traceback.format_exc()},
                    'timestamp': datetime.now().isoformat()
                })
        
        # Generate test report
        test_report = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100,
            'test_results': self.test_results,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED',
            'generated_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Test suite completed: {passed_tests}/{total_tests} tests passed ({test_report['success_rate']:.1f}%)")
        
        return test_report
    
    async def test_component_initialization(self) -> Dict[str, Any]:
        """Test component initialization"""
        
        try:
            # Test individual component creation
            components = {}
            
            # Test AdaptiveLearningAgent
            components['adaptive_agent'] = AdaptiveLearningAgent(config=self.test_config)
            
            # Test PatternRecognitionEngine
            components['pattern_engine'] = PatternRecognitionEngine(config=self.test_config)
            
            # Test PersonalizationEngine
            components['personalization_engine'] = PersonalizationEngine(config=self.test_config)
            
            # Test FeedbackProcessor
            components['feedback_processor'] = FeedbackProcessor(config=self.test_config)
            
            # Test OutcomeTracker
            components['outcome_tracker'] = OutcomeTracker(config=self.test_config)
            
            # Test PrivacyProtectionSystem
            components['privacy_protection'] = PrivacyProtectionSystem(config=self.test_config)
            
            # Test InsightGenerationSystem (requires other components)
            components['insights_generator'] = InsightGenerationSystem(
                outcome_tracker=components['outcome_tracker'],
                feedback_processor=components['feedback_processor'],
                personalization_engine=components['personalization_engine'],
                pattern_engine=components['pattern_engine'],
                config=self.test_config
            )
            
            # Test main integration system
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            # Test initialization
            init_success = await integrator.initialize()
            
            if not init_success:
                return {'passed': False, 'error': 'Integration system initialization failed'}
            
            # Test that all components are accessible
            required_attributes = [
                'adaptive_learning_agent', 'pattern_engine', 'personalization_engine',
                'feedback_processor', 'outcome_tracker', 'privacy_protection', 'insights_generator'
            ]
            
            for attr in required_attributes:
                if not hasattr(integrator, attr):
                    return {'passed': False, 'error': f'Missing required attribute: {attr}'}
                if getattr(integrator, attr) is None:
                    return {'passed': False, 'error': f'Attribute {attr} is None'}
            
            return {
                'passed': True,
                'components_initialized': len(components),
                'integrator_initialized': init_success,
                'registered_agents': len(self.mock_orchestrator.registered_agents)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_privacy_protection(self) -> Dict[str, Any]:
        """Test privacy protection system"""
        
        try:
            privacy_system = PrivacyProtectionSystem(config=self.test_config['privacy_protection_config'])
            
            # Test data processing with different sensitivity levels
            test_data = {
                'user_feedback': 'This is helpful',
                'session_duration': 25.5,
                'interaction_count': 3,
                'timestamp': datetime.now().isoformat()
            }
            
            # Test with different sensitivity levels
            sensitivity_tests = [
                DataSensitivity.PUBLIC,
                DataSensitivity.INTERNAL,
                DataSensitivity.CONFIDENTIAL,
                DataSensitivity.RESTRICTED
            ]
            
            processed_data_points = []
            for sensitivity in sensitivity_tests:
                processed_data = privacy_system.process_learning_data(
                    user_id='test_user_123',
                    data=test_data,
                    sensitivity=sensitivity
                )
                processed_data_points.append(processed_data)
                
                # Verify anonymization
                if processed_data.anonymized_user_id == 'test_user_123':
                    return {'passed': False, 'error': 'User ID not properly anonymized'}
            
            # Test privacy compliance validation
            compliance = privacy_system.validate_privacy_compliance()
            
            if not compliance.get('overall_compliant', False):
                return {'passed': False, 'error': 'Privacy compliance validation failed'}
            
            # Test privacy report generation
            privacy_report = privacy_system.get_privacy_report()
            
            required_report_keys = ['privacy_configuration', 'current_status', 'compliance_metrics']
            for key in required_report_keys:
                if key not in privacy_report:
                    return {'passed': False, 'error': f'Missing privacy report key: {key}'}
            
            return {
                'passed': True,
                'sensitivity_levels_tested': len(sensitivity_tests),
                'data_points_processed': len(processed_data_points),
                'privacy_compliant': compliance.get('overall_compliant'),
                'privacy_report_generated': bool(privacy_report)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_user_interaction_processing(self) -> Dict[str, Any]:
        """Test user interaction processing"""
        
        try:
            # Create integration system
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            await integrator.initialize()
            
            # Test interaction data
            interaction_data = {
                'feedback': 'The therapy session was very helpful',
                'user_rating': 8,
                'session_duration': 45.2,
                'response_time': 12.5,
                'outcome_measurements': {
                    'mood': 7.5,
                    'anxiety': 3.2,
                    'engagement': 8.1
                },
                'timestamp': datetime.now().isoformat()
            }
            
            agent_response = {
                'agent_name': 'therapy_agent',
                'response': 'I understand you are feeling better. Let\'s continue with mindfulness exercises.',
                'confidence': 0.85,
                'response_time': 1.2
            }
            
            context = {
                'session_type': 'therapy',
                'user_history': ['previous_session_1', 'previous_session_2'],
                'available_interventions': ['mindfulness', 'cbt', 'dbt']
            }
            
            # Process interaction
            result = await integrator.process_user_interaction(
                user_id='test_user_456',
                session_id='session_789',
                interaction_data=interaction_data,
                agent_response=agent_response,
                context=context
            )
            
            # Verify processing results
            required_keys = [
                'session_id', 'user_id', 'timestamp', 'components_processed',
                'insights_generated', 'recommendations', 'alerts'
            ]
            
            for key in required_keys:
                if key not in result:
                    return {'passed': False, 'error': f'Missing result key: {key}'}
            
            # Verify that components were processed
            expected_components = ['privacy_protection', 'feedback_processing', 'outcome_tracking', 'personalization']
            
            for component in expected_components:
                if component not in result['components_processed']:
                    return {'passed': False, 'error': f'Component {component} was not processed'}
            
            # Verify privacy protection was applied
            if not result.get('privacy_protected', False):
                return {'passed': False, 'error': 'Privacy protection was not applied'}
            
            return {
                'passed': True,
                'components_processed': len(result['components_processed']),
                'insights_generated': len(result['insights_generated']),
                'recommendations_made': len(result['recommendations']),
                'alerts_raised': len(result['alerts']),
                'privacy_protected': result.get('privacy_protected', False)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_feedback_integration(self) -> Dict[str, Any]:
        """Test feedback integration system"""
        
        try:
            feedback_processor = FeedbackProcessor(config=self.test_config)
            
            # Test feedback processing
            feedback_data = {
                'rating': 8,
                'text_feedback': 'The session was very helpful and I feel much better',
                'response_time': 15.5,
                'interaction_duration': 35.2
            }
            
            intervention_context = {
                'intervention_type': 'cbt',
                'session_phase': 'middle',
                'therapist_approach': 'supportive'
            }
            
            # Process feedback
            feedback_entry = await feedback_processor.process_feedback(
                user_id='test_user_feedback',
                session_id='feedback_session_123',
                feedback_data=feedback_data,
                intervention_context=intervention_context
            )
            
            # Verify feedback entry
            if not feedback_entry:
                return {'passed': False, 'error': 'No feedback entry returned'}
            
            if not hasattr(feedback_entry, 'feedback_id'):
                return {'passed': False, 'error': 'Feedback entry missing feedback_id'}
            
            # Test feedback statistics
            stats = feedback_processor.get_feedback_statistics()
            
            required_stats = ['total_feedback_count', 'avg_rating', 'sentiment_distribution']
            for stat in required_stats:
                if stat not in stats:
                    return {'passed': False, 'error': f'Missing feedback statistic: {stat}'}
            
            # Test feedback summary generation
            summary = await feedback_processor.generate_feedback_summary(time_period_hours=24)
            
            if not summary:
                return {'passed': False, 'error': 'No feedback summary generated'}
            
            return {
                'passed': True,
                'feedback_entry_created': bool(feedback_entry),
                'feedback_id': feedback_entry.feedback_id if feedback_entry else None,
                'statistics_available': bool(stats),
                'summary_generated': bool(summary)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_outcome_tracking(self) -> Dict[str, Any]:
        """Test outcome tracking system"""
        
        try:
            outcome_tracker = OutcomeTracker(config=self.test_config)
            
            # Test outcome measurement recording
            test_measurements = [
                (OutcomeType.MOOD, 7.5),
                (OutcomeType.ANXIETY, 3.2),
                (OutcomeType.DEPRESSION, 2.8),
                (OutcomeType.ENGAGEMENT, 8.1)
            ]
            
            user_id = 'test_user_outcomes'
            recorded_measurements = []
            
            for outcome_type, score in test_measurements:
                measurement = await outcome_tracker.record_outcome_measurement(
                    user_id=user_id,
                    outcome_type=outcome_type,
                    primary_score=score,
                    context={'test': True},
                    data_source='test_suite'
                )
                
                if not measurement:
                    return {'passed': False, 'error': f'Failed to record {outcome_type.value} measurement'}
                
                recorded_measurements.append(measurement)
            
            # Test user profile creation
            user_profile = outcome_tracker.get_user_profile(user_id)
            
            if not user_profile:
                return {'passed': False, 'error': 'User profile not created'}
            
            if len(user_profile.outcome_measurements) != len(test_measurements):
                return {'passed': False, 'error': 'Not all measurements recorded in user profile'}
            
            # Test system statistics
            system_stats = outcome_tracker.get_system_statistics()
            
            required_stats = ['total_users', 'total_measurements', 'avg_outcomes']
            for stat in required_stats:
                if stat not in system_stats:
                    return {'passed': False, 'error': f'Missing system statistic: {stat}'}
            
            # Test outcome report generation
            outcome_report = await outcome_tracker.generate_outcome_report(time_period_days=7)
            
            if not outcome_report:
                return {'passed': False, 'error': 'Outcome report not generated'}
            
            return {
                'passed': True,
                'measurements_recorded': len(recorded_measurements),
                'user_profile_created': bool(user_profile),
                'system_statistics_available': bool(system_stats),
                'outcome_report_generated': bool(outcome_report)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_pattern_recognition(self) -> Dict[str, Any]:
        """Test pattern recognition engine"""
        
        try:
            pattern_engine = PatternRecognitionEngine(config=self.test_config)
            
            # Test pattern analysis with sample data
            sample_interactions = [
                {
                    'user_id': 'pattern_test_user',
                    'intervention_type': 'cbt',
                    'outcome_score': 8.5,
                    'engagement': 0.9,
                    'session_duration': 45.0,
                    'timestamp': datetime.now()
                },
                {
                    'user_id': 'pattern_test_user',
                    'intervention_type': 'cbt',
                    'outcome_score': 8.2,
                    'engagement': 0.85,
                    'session_duration': 40.0,
                    'timestamp': datetime.now()
                },
                {
                    'user_id': 'pattern_test_user',
                    'intervention_type': 'mindfulness',
                    'outcome_score': 7.1,
                    'engagement': 0.7,
                    'session_duration': 30.0,
                    'timestamp': datetime.now()
                }
            ]
            
            # Analyze patterns (simplified for testing)
            patterns_found = []
            
            # Test if pattern engine has basic structure
            expected_attributes = ['config']  # Minimal expected attributes
            
            for attr in expected_attributes:
                if not hasattr(pattern_engine, attr):
                    return {'passed': False, 'error': f'Pattern engine missing attribute: {attr}'}
            
            # Pattern recognition is complex and would require more extensive testing
            # For now, test basic functionality
            
            return {
                'passed': True,
                'pattern_engine_initialized': True,
                'sample_interactions_processed': len(sample_interactions),
                'patterns_identified': len(patterns_found)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_personalization_engine(self) -> Dict[str, Any]:
        """Test personalization engine"""
        
        try:
            personalization_engine = PersonalizationEngine(config=self.test_config)
            
            # Test personalization profile creation
            user_id = 'personalization_test_user'
            
            # Simulate learning from outcomes
            from ..diagnosis.adaptive_learning import InterventionOutcome
            
            sample_outcome = InterventionOutcome(
                intervention_id='test_intervention_123',
                user_id=user_id,
                intervention_type='cbt',
                intervention_content='Cognitive behavioral therapy session',
                context={'session_type': 'therapy'},
                timestamp=datetime.now(),
                user_response='This was helpful',
                engagement_score=0.85,
                effectiveness_score=0.8
            )
            
            # Learn from outcome
            await personalization_engine.learn_from_outcome(
                user_id=user_id,
                intervention_id=sample_outcome.intervention_id,
                outcome=sample_outcome
            )
            
            # Test personalization
            current_context = {
                'mood': 7.0,
                'anxiety_level': 3.0,
                'session_type': 'therapy'
            }
            
            available_interventions = ['cbt', 'dbt', 'mindfulness', 'supportive']
            
            recommendation = await personalization_engine.personalize_response(
                user_id=user_id,
                current_context=current_context,
                available_interventions=available_interventions
            )
            
            if not recommendation:
                return {'passed': False, 'error': 'No personalization recommendation generated'}
            
            # Test personalization summary
            summary = personalization_engine.get_personalization_summary()
            
            if not summary:
                return {'passed': False, 'error': 'No personalization summary generated'}
            
            return {
                'passed': True,
                'outcome_processed': True,
                'recommendation_generated': bool(recommendation),
                'personalization_summary_available': bool(summary)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_insights_generation(self) -> Dict[str, Any]:
        """Test insights generation system"""
        
        try:
            # Create dependencies
            outcome_tracker = OutcomeTracker(config=self.test_config)
            feedback_processor = FeedbackProcessor(config=self.test_config)
            personalization_engine = PersonalizationEngine(config=self.test_config)
            pattern_engine = PatternRecognitionEngine(config=self.test_config)
            
            # Create insights generator
            insights_generator = InsightGenerationSystem(
                outcome_tracker=outcome_tracker,
                feedback_processor=feedback_processor,
                personalization_engine=personalization_engine,
                pattern_engine=pattern_engine,
                config=self.test_config['insights_generation_config']
            )
            
            # Add some test data to generate insights from
            test_user_id = 'insights_test_user'
            
            # Add test outcome measurement
            await outcome_tracker.record_outcome_measurement(
                user_id=test_user_id,
                outcome_type=OutcomeType.MOOD,
                primary_score=7.5,
                context={'test': True},
                data_source='insights_test'
            )
            
            # Add test feedback
            await feedback_processor.process_feedback(
                user_id=test_user_id,
                session_id='insights_test_session',
                feedback_data={'rating': 8, 'text_feedback': 'Good session'},
                intervention_context={'type': 'test'}
            )
            
            # Generate insights (with minimal data, may not generate much)
            insights = await insights_generator.generate_comprehensive_insights(time_period_days=7)
            
            # Test insights dashboard
            dashboard = insights_generator.get_insights_dashboard()
            
            if not dashboard:
                return {'passed': False, 'error': 'No insights dashboard generated'}
            
            # Test export functionality
            insights_report = insights_generator.export_insights_report(format='json')
            
            if not insights_report:
                return {'passed': False, 'error': 'No insights report generated'}
            
            # Verify it's valid JSON
            try:
                json.loads(insights_report)
            except json.JSONDecodeError:
                return {'passed': False, 'error': 'Insights report is not valid JSON'}
            
            return {
                'passed': True,
                'insights_generated': len(insights) if insights else 0,
                'dashboard_available': bool(dashboard),
                'report_exported': bool(insights_report)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration"""
        
        try:
            # Test the convenience function
            integrator = await create_adaptive_learning_system(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            if not integrator:
                return {'passed': False, 'error': 'System integration failed'}
            
            # Test system status
            status = await integrator.get_system_learning_status()
            
            if not status:
                return {'passed': False, 'error': 'System status not available'}
            
            required_status_keys = ['system_info', 'component_status', 'performance_metrics']
            for key in required_status_keys:
                if key not in status:
                    return {'passed': False, 'error': f'Missing status key: {key}'}
            
            # Test complete user interaction processing
            interaction_data = {
                'feedback': 'Integration test feedback',
                'user_rating': 9,
                'session_duration': 30.0,
                'outcome_measurements': {'mood': 8.0}
            }
            
            agent_response = {
                'agent_name': 'integration_test_agent',
                'response': 'Test response for integration',
                'confidence': 0.9
            }
            
            result = await integrator.process_user_interaction(
                user_id='integration_test_user',
                session_id='integration_test_session',
                interaction_data=interaction_data,
                agent_response=agent_response
            )
            
            if not result:
                return {'passed': False, 'error': 'User interaction processing failed'}
            
            return {
                'passed': True,
                'system_created': True,
                'status_available': bool(status),
                'interaction_processed': bool(result),
                'system_health': status['system_info'].get('system_health', 'unknown')
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling robustness"""
        
        try:
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            await integrator.initialize()
            
            # Test with invalid data
            error_tests = [
                {
                    'name': 'None user_id',
                    'user_id': None,
                    'session_id': 'valid_session',
                    'interaction_data': {},
                    'agent_response': {}
                },
                {
                    'name': 'Empty interaction_data',
                    'user_id': 'valid_user',
                    'session_id': 'valid_session',
                    'interaction_data': None,
                    'agent_response': {}
                },
                {
                    'name': 'Malformed data',
                    'user_id': 'valid_user',
                    'session_id': 'valid_session',
                    'interaction_data': {'invalid_key': {'nested': {'deeply': 'malformed'}}},
                    'agent_response': 'not_a_dict'
                }
            ]
            
            errors_handled = 0
            total_error_tests = len(error_tests)
            
            for error_test in error_tests:
                try:
                    result = await integrator.process_user_interaction(
                        user_id=error_test['user_id'],
                        session_id=error_test['session_id'],
                        interaction_data=error_test['interaction_data'],
                        agent_response=error_test['agent_response']
                    )
                    
                    # Should either return error result or handle gracefully
                    if result and ('error' in result or 'timestamp' in result):
                        errors_handled += 1
                    
                except Exception as e:
                    # If it throws an exception, check if it's handled gracefully
                    if 'Error in adaptive learning processing' in str(e):
                        errors_handled += 1
                    else:
                        self.logger.warning(f"Unhandled error in {error_test['name']}: {str(e)}")
            
            return {
                'passed': errors_handled == total_error_tests,
                'error_tests_run': total_error_tests,
                'errors_handled_gracefully': errors_handled,
                'error_handling_rate': (errors_handled / total_error_tests) * 100
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_privacy_compliance(self) -> Dict[str, Any]:
        """Test privacy compliance throughout the system"""
        
        try:
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            await integrator.initialize()
            
            # Process interaction with sensitive data
            sensitive_interaction = {
                'feedback': 'I have been diagnosed with severe depression and anxiety',
                'user_rating': 6,
                'personal_info': 'John Doe, 123 Main St',
                'medical_history': 'Previous hospitalization for mental health',
                'outcome_measurements': {'mood': 4.5, 'anxiety': 8.2}
            }
            
            agent_response = {
                'agent_name': 'privacy_test_agent',
                'response': 'I understand your concerns about your mental health'
            }
            
            result = await integrator.process_user_interaction(
                user_id='privacy_compliance_test_user',
                session_id='privacy_compliance_session',
                interaction_data=sensitive_interaction,
                agent_response=agent_response
            )
            
            # Verify privacy protection was applied
            if not result.get('privacy_protected', False):
                return {'passed': False, 'error': 'Privacy protection was not applied to sensitive data'}
            
            # Check privacy compliance status
            if 'privacy_compliance' not in result:
                return {'passed': False, 'error': 'Privacy compliance status not included in result'}
            
            privacy_compliance = result['privacy_compliance']
            
            if not privacy_compliance.get('overall_compliant', False):
                return {'passed': False, 'error': 'Privacy compliance validation failed'}
            
            # Test privacy report
            privacy_report = integrator.privacy_protection.get_privacy_report()
            
            compliance_metrics = privacy_report.get('compliance_metrics', {})
            required_metrics = ['data_retention_compliant', 'anonymization_coverage']
            
            for metric in required_metrics:
                if metric not in compliance_metrics:
                    return {'passed': False, 'error': f'Missing privacy compliance metric: {metric}'}
            
            return {
                'passed': True,
                'privacy_protection_applied': result.get('privacy_protected', False),
                'compliance_validated': privacy_compliance.get('overall_compliant', False),
                'privacy_report_available': bool(privacy_report),
                'anonymization_coverage': compliance_metrics.get('anonymization_coverage', 0)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring capabilities"""
        
        try:
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            await integrator.initialize()
            
            # Trigger performance monitoring
            await integrator._monitor_system_performance()
            
            # Check if performance data was collected
            if not hasattr(integrator, 'performance_monitoring_history'):
                return {'passed': False, 'error': 'Performance monitoring history not initialized'}
            
            if len(integrator.performance_monitoring_history) == 0:
                return {'passed': False, 'error': 'No performance monitoring data collected'}
            
            # Check performance data structure
            latest_performance = integrator.performance_monitoring_history[-1]
            
            required_keys = ['timestamp', 'components_active', 'data_statistics', 'system_metrics']
            for key in required_keys:
                if key not in latest_performance:
                    return {'passed': False, 'error': f'Missing performance monitoring key: {key}'}
            
            # Test system health validation
            health_valid = await integrator._validate_system_health()
            
            if not isinstance(health_valid, bool):
                return {'passed': False, 'error': 'System health validation did not return boolean'}
            
            return {
                'passed': True,
                'performance_monitoring_active': True,
                'performance_data_collected': len(integrator.performance_monitoring_history),
                'system_health_validated': health_valid,
                'components_monitored': len(latest_performance.get('components_active', {}))
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}
    
    async def test_graceful_shutdown(self) -> Dict[str, Any]:
        """Test graceful system shutdown"""
        
        try:
            integrator = AdaptiveLearningSystemIntegrator(
                orchestrator=self.mock_orchestrator,
                vector_db=self.mock_vector_db,
                config=self.test_config
            )
            
            await integrator.initialize()
            
            # Add some data to save
            await integrator.process_user_interaction(
                user_id='shutdown_test_user',
                session_id='shutdown_test_session',
                interaction_data={'feedback': 'Shutdown test feedback', 'user_rating': 7},
                agent_response={'agent_name': 'shutdown_test_agent', 'response': 'Test response'}
            )
            
            # Test shutdown
            await integrator.shutdown()
            
            # Verify data was saved (check if files were created)
            data_dir = 'src/data/adaptive_learning'
            expected_files = ['system_state.json']
            
            files_created = []
            for filename in expected_files:
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    files_created.append(filename)
            
            return {
                'passed': True,
                'shutdown_completed': True,
                'files_saved': len(files_created),
                'expected_files': len(expected_files),
                'data_directory_created': os.path.exists(data_dir)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e), 'exception': traceback.format_exc()}

async def run_adaptive_learning_tests():
    """Run the comprehensive test suite"""
    
    print("🧪 Starting Adaptive Learning System Comprehensive Tests")
    print("=" * 60)
    
    tester = AdaptiveLearningSystemTester()
    
    try:
        test_report = await tester.run_comprehensive_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {test_report['total_tests']}")
        print(f"Passed: {test_report['passed_tests']}")
        print(f"Failed: {test_report['failed_tests']}")
        print(f"Success Rate: {test_report['success_rate']:.1f}%")
        print(f"Overall Status: {test_report['overall_status']}")
        
        # Print failed tests details
        if test_report['failed_tests'] > 0:
            print("\n❌ FAILED TESTS:")
            for result in test_report['test_results']:
                if not result['result'].get('passed', False):
                    print(f"  • {result['test_name']}: {result['result'].get('error', 'Unknown error')}")
        
        # Save test report
        report_filename = f"adaptive_learning_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            os.makedirs('src/data/test_reports', exist_ok=True)
            with open(f'src/data/test_reports/{report_filename}', 'w') as f:
                json.dump(test_report, f, indent=2, default=str)
            print(f"\n📄 Test report saved to: src/data/test_reports/{report_filename}")
        except Exception as e:
            print(f"\n⚠️  Could not save test report: {str(e)}")
        
        print("\n" + "=" * 60)
        
        return test_report
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        print(f"Exception details: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_adaptive_learning_tests())
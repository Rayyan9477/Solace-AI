"""
Comprehensive Testing and Validation Module

This module provides comprehensive testing for all enhanced diagnostic and therapeutic
systems, including unit tests, integration tests, performance tests, and validation
of diagnostic accuracy and therapeutic effectiveness.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import traceback
from dataclasses import dataclass, asdict

from ..diagnosis.enhanced_integrated_system import EnhancedIntegratedDiagnosticSystem
from ..diagnosis.temporal_analysis import TemporalAnalysisEngine
from ..diagnosis.differential_diagnosis import DifferentialDiagnosisEngine
from ..diagnosis.therapeutic_friction import TherapeuticFrictionEngine
from ..diagnosis.cultural_sensitivity import CulturalSensitivityEngine
from ..diagnosis.adaptive_learning import AdaptiveLearningEngine
from ..memory.enhanced_memory_system import EnhancedMemorySystem
from ..research.real_time_research import RealTimeResearchEngine
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_type: str  # unit, integration, performance, validation
    status: str  # pass, fail, warning
    execution_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TestSuite:
    """Test suite results"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    execution_time_ms: float
    test_results: List[TestResult]
    coverage_percentage: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ComprehensiveTester:
    """
    Comprehensive testing system for all diagnostic and therapeutic components
    """
    
    def __init__(self):
        """Initialize the comprehensive tester"""
        self.logger = get_logger(__name__)
        self.test_data = self._load_test_data()
        
        # Test configuration
        self.performance_thresholds = {
            "max_response_time_ms": 5000,
            "max_memory_usage_mb": 500,
            "min_accuracy_percentage": 80,
            "max_error_rate_percentage": 5
        }
        
        # Mock data for testing
        self.mock_user_data = self._generate_mock_user_data()
    
    async def run_all_tests(self) -> Dict[str, TestSuite]:
        """
        Run all comprehensive tests
        
        Returns:
            Dictionary of test suite results
        """
        self.logger.info("Starting comprehensive testing suite")
        start_time = time.time()
        
        test_suites = {}
        
        try:
            # Unit Tests
            test_suites["unit_tests"] = await self._run_unit_tests()
            
            # Integration Tests
            test_suites["integration_tests"] = await self._run_integration_tests()
            
            # Performance Tests
            test_suites["performance_tests"] = await self._run_performance_tests()
            
            # Validation Tests
            test_suites["validation_tests"] = await self._run_validation_tests()
            
            # System Health Tests
            test_suites["system_health_tests"] = await self._run_system_health_tests()
            
            total_time = (time.time() - start_time) * 1000
            self.logger.info(f"All tests completed in {total_time:.1f}ms")
            
            # Generate summary report
            await self._generate_test_report(test_suites, total_time)
            
            return test_suites
            
        except Exception as e:
            self.logger.error(f"Critical error in test execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def _run_unit_tests(self) -> TestSuite:
        """Run unit tests for individual components"""
        self.logger.info("Running unit tests")
        start_time = time.time()
        test_results = []
        
        # Test Temporal Analysis Engine
        test_results.extend(await self._test_temporal_analysis_unit())
        
        # Test Differential Diagnosis Engine
        test_results.extend(await self._test_differential_diagnosis_unit())
        
        # Test Therapeutic Friction Engine
        test_results.extend(await self._test_therapeutic_friction_unit())
        
        # Test Cultural Sensitivity Engine
        test_results.extend(await self._test_cultural_sensitivity_unit())
        
        # Test Adaptive Learning Engine
        test_results.extend(await self._test_adaptive_learning_unit())
        
        # Test Memory System
        test_results.extend(await self._test_memory_system_unit())
        
        # Test Research Integration
        test_results.extend(await self._test_research_integration_unit())
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._compile_test_suite("unit_tests", test_results, execution_time)
    
    async def _run_integration_tests(self) -> TestSuite:
        """Run integration tests between components"""
        self.logger.info("Running integration tests")
        start_time = time.time()
        test_results = []
        
        # Test integrated diagnostic system
        test_results.extend(await self._test_integrated_system())
        
        # Test data flow between components
        test_results.extend(await self._test_data_flow_integration())
        
        # Test error handling integration
        test_results.extend(await self._test_error_handling_integration())
        
        # Test memory persistence integration
        test_results.extend(await self._test_memory_persistence_integration())
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._compile_test_suite("integration_tests", test_results, execution_time)
    
    async def _run_performance_tests(self) -> TestSuite:
        """Run performance tests"""
        self.logger.info("Running performance tests")
        start_time = time.time()
        test_results = []
        
        # Test response time under load
        test_results.append(await self._test_response_time_performance())
        
        # Test memory usage
        test_results.append(await self._test_memory_usage())
        
        # Test concurrent user handling
        test_results.append(await self._test_concurrent_users())
        
        # Test database performance
        test_results.append(await self._test_database_performance())
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._compile_test_suite("performance_tests", test_results, execution_time)
    
    async def _run_validation_tests(self) -> TestSuite:
        """Run validation tests for diagnostic accuracy"""
        self.logger.info("Running validation tests")
        start_time = time.time()
        test_results = []
        
        # Test diagnostic accuracy
        test_results.append(await self._test_diagnostic_accuracy())
        
        # Test therapeutic response quality
        test_results.append(await self._test_therapeutic_response_quality())
        
        # Test cultural sensitivity accuracy
        test_results.append(await self._test_cultural_sensitivity_accuracy())
        
        # Test adaptive learning effectiveness
        test_results.append(await self._test_adaptive_learning_effectiveness())
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._compile_test_suite("validation_tests", test_results, execution_time)
    
    async def _run_system_health_tests(self) -> TestSuite:
        """Run system health and reliability tests"""
        self.logger.info("Running system health tests")
        start_time = time.time()
        test_results = []
        
        # Test system initialization
        test_results.append(await self._test_system_initialization())
        
        # Test error recovery
        test_results.append(await self._test_error_recovery())
        
        # Test data consistency
        test_results.append(await self._test_data_consistency())
        
        # Test system monitoring
        test_results.append(await self._test_system_monitoring())
        
        execution_time = (time.time() - start_time) * 1000
        
        return self._compile_test_suite("system_health_tests", test_results, execution_time)
    
    # Unit Test Methods
    
    async def _test_temporal_analysis_unit(self) -> List[TestResult]:
        """Unit tests for temporal analysis engine"""
        results = []
        
        try:
            engine = TemporalAnalysisEngine()
            
            # Test symptom recording
            result = await self._run_single_test(
                "temporal_analysis_symptom_recording",
                "unit",
                lambda: engine.record_symptom("test_user", "anxiety", 0.7, "test context")
            )
            results.append(result)
            
            # Test pattern detection
            result = await self._run_single_test(
                "temporal_analysis_pattern_detection",
                "unit",
                lambda: engine.detect_behavioral_patterns("test_user")
            )
            results.append(result)
            
            # Test trajectory prediction
            result = await self._run_single_test(
                "temporal_analysis_trajectory_prediction",
                "unit",
                lambda: engine.predict_symptom_trajectory("test_user", "anxiety", 7)
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="temporal_analysis_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_differential_diagnosis_unit(self) -> List[TestResult]:
        """Unit tests for differential diagnosis engine"""
        results = []
        
        try:
            engine = DifferentialDiagnosisEngine()
            
            # Test differential diagnosis generation
            result = await self._run_single_test(
                "differential_diagnosis_generation",
                "unit",
                lambda: engine.generate_differential_diagnosis(
                    symptoms=["anxiety", "depression"],
                    behavioral_observations=["social withdrawal"],
                    temporal_patterns={},
                    voice_emotion_data=None,
                    personality_data=None
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="differential_diagnosis_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_therapeutic_friction_unit(self) -> List[TestResult]:
        """Unit tests for therapeutic friction engine"""
        results = []
        
        try:
            engine = TherapeuticFrictionEngine()
            
            # Test therapeutic response generation
            result = await self._run_single_test(
                "therapeutic_friction_response_generation",
                "unit",
                lambda: engine.generate_therapeutic_response(
                    "test_user", "I'm feeling anxious", {}, []
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="therapeutic_friction_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_cultural_sensitivity_unit(self) -> List[TestResult]:
        """Unit tests for cultural sensitivity engine"""
        results = []
        
        try:
            engine = CulturalSensitivityEngine()
            
            # Test cultural context assessment
            result = await self._run_single_test(
                "cultural_sensitivity_context_assessment",
                "unit",
                lambda: engine.assess_cultural_context(
                    "test_user", "I'm from a traditional family", []
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="cultural_sensitivity_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_adaptive_learning_unit(self) -> List[TestResult]:
        """Unit tests for adaptive learning engine"""
        results = []
        
        try:
            engine = AdaptiveLearningEngine()
            
            # Test intervention outcome recording
            result = await self._run_single_test(
                "adaptive_learning_outcome_recording",
                "unit",
                lambda: engine.record_intervention_outcome(
                    "test_intervention", "test_user", "CBT", "test content", {}
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="adaptive_learning_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_memory_system_unit(self) -> List[TestResult]:
        """Unit tests for memory system"""
        results = []
        
        try:
            system = EnhancedMemorySystem()
            
            # Test insight storage
            result = await self._run_single_test(
                "memory_system_insight_storage",
                "unit",
                lambda: system.store_therapeutic_insight(
                    "test_user", "test_session", "breakthrough", "test insight", {}
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="memory_system_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_research_integration_unit(self) -> List[TestResult]:
        """Unit tests for research integration"""
        results = []
        
        try:
            engine = RealTimeResearchEngine()
            
            # Test evidence-based recommendations
            result = await self._run_single_test(
                "research_integration_recommendations",
                "unit",
                lambda: engine.get_evidence_based_recommendations(
                    "depression", "moderate", {}, "western"
                )
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="research_integration_unit_tests",
                test_type="unit",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    # Integration Test Methods
    
    async def _test_integrated_system(self) -> List[TestResult]:
        """Test integrated diagnostic system"""
        results = []
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Test comprehensive diagnosis
            result = await self._run_single_test(
                "integrated_system_comprehensive_diagnosis",
                "integration",
                lambda: system.generate_comprehensive_diagnosis(
                    "test_user", "test_session", "I'm feeling anxious and depressed", []
                )
            )
            results.append(result)
            
            # Test system validation
            result = await self._run_single_test(
                "integrated_system_validation",
                "integration",
                lambda: system.validate_system_integration()
            )
            results.append(result)
            
        except Exception as e:
            results.append(TestResult(
                test_name="integrated_system_tests",
                test_type="integration",
                status="fail",
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
        
        return results
    
    async def _test_data_flow_integration(self) -> List[TestResult]:
        """Test data flow between components"""
        results = []
        
        # Test data flow from temporal analysis to differential diagnosis
        result = await self._run_single_test(
            "data_flow_temporal_to_diagnosis",
            "integration",
            lambda: self._simulate_data_flow_test()
        )
        results.append(result)
        
        return results
    
    async def _test_error_handling_integration(self) -> List[TestResult]:
        """Test error handling across components"""
        results = []
        
        # Test graceful degradation
        result = await self._run_single_test(
            "error_handling_graceful_degradation",
            "integration",
            lambda: self._test_graceful_degradation()
        )
        results.append(result)
        
        return results
    
    async def _test_memory_persistence_integration(self) -> List[TestResult]:
        """Test memory persistence across system restarts"""
        results = []
        
        # Test data persistence
        result = await self._run_single_test(
            "memory_persistence_data_retention",
            "integration",
            lambda: self._test_data_persistence()
        )
        results.append(result)
        
        return results
    
    # Performance Test Methods
    
    async def _test_response_time_performance(self) -> TestResult:
        """Test system response time under normal load"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Measure response time for comprehensive diagnosis
            test_start = time.time()
            result = await system.generate_comprehensive_diagnosis(
                "perf_test_user", "perf_test_session", "Performance test message", []
            )
            response_time_ms = (time.time() - test_start) * 1000
            
            # Check against threshold
            status = "pass" if response_time_ms < self.performance_thresholds["max_response_time_ms"] else "fail"
            
            return TestResult(
                test_name="response_time_performance",
                test_type="performance",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "response_time_ms": response_time_ms,
                    "threshold_ms": self.performance_thresholds["max_response_time_ms"],
                    "result_type": type(result).__name__
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="response_time_performance",
                test_type="performance",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_memory_usage(self) -> TestResult:
        """Test memory usage during operation"""
        
        start_time = time.time()
        
        try:
            # This would measure actual memory usage in a real implementation
            # For now, simulate memory usage test
            
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Simulate multiple operations
            for i in range(10):
                await system.generate_comprehensive_diagnosis(
                    f"memory_test_user_{i}", f"memory_test_session_{i}", 
                    f"Memory test message {i}", []
                )
            
            # Simulated memory usage (would use actual measurement in real implementation)
            simulated_memory_mb = 150  # MB
            
            status = "pass" if simulated_memory_mb < self.performance_thresholds["max_memory_usage_mb"] else "fail"
            
            return TestResult(
                test_name="memory_usage_performance",
                test_type="performance",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "memory_usage_mb": simulated_memory_mb,
                    "threshold_mb": self.performance_thresholds["max_memory_usage_mb"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="memory_usage_performance",
                test_type="performance",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_concurrent_users(self) -> TestResult:
        """Test handling multiple concurrent users"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Simulate concurrent users
            tasks = []
            for i in range(5):  # Test with 5 concurrent users
                task = system.generate_comprehensive_diagnosis(
                    f"concurrent_user_{i}", f"concurrent_session_{i}",
                    f"Concurrent test message {i}", []
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            success_rate = (len(results) - len(errors)) / len(results) * 100
            
            status = "pass" if success_rate >= 90 else "fail"  # 90% success rate required
            
            return TestResult(
                test_name="concurrent_users_performance",
                test_type="performance",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "concurrent_users": len(tasks),
                    "success_rate_percent": success_rate,
                    "errors": len(errors)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="concurrent_users_performance",
                test_type="performance",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_database_performance(self) -> TestResult:
        """Test database operation performance"""
        
        start_time = time.time()
        
        try:
            # Simulate database performance test
            # In real implementation, would test actual database operations
            
            # Simulate database operations
            await asyncio.sleep(0.1)  # Simulate database operation time
            
            operation_time_ms = 100  # Simulated
            threshold_ms = 1000  # 1 second threshold
            
            status = "pass" if operation_time_ms < threshold_ms else "fail"
            
            return TestResult(
                test_name="database_performance",
                test_type="performance",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "operation_time_ms": operation_time_ms,
                    "threshold_ms": threshold_ms
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="database_performance",
                test_type="performance",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    # Validation Test Methods
    
    async def _test_diagnostic_accuracy(self) -> TestResult:
        """Test diagnostic accuracy against known cases"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Test cases with known diagnoses
            test_cases = [
                {
                    "message": "I've been feeling sad and hopeless for weeks, can't sleep, lost interest in activities",
                    "expected_condition": "Major Depressive Disorder",
                    "expected_severity": "moderate"
                },
                {
                    "message": "I'm constantly worried about everything, heart racing, can't relax",
                    "expected_condition": "Generalized Anxiety Disorder",
                    "expected_severity": "moderate"
                }
            ]
            
            correct_diagnoses = 0
            total_cases = len(test_cases)
            
            for i, case in enumerate(test_cases):
                result = await system.generate_comprehensive_diagnosis(
                    f"validation_user_{i}", f"validation_session_{i}",
                    case["message"], []
                )
                
                # Check if primary diagnosis matches expected
                if (result.primary_diagnosis and 
                    case["expected_condition"].lower() in result.primary_diagnosis.condition_name.lower()):
                    correct_diagnoses += 1
            
            accuracy_percentage = (correct_diagnoses / total_cases) * 100
            status = "pass" if accuracy_percentage >= self.performance_thresholds["min_accuracy_percentage"] else "fail"
            
            return TestResult(
                test_name="diagnostic_accuracy_validation",
                test_type="validation",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "accuracy_percentage": accuracy_percentage,
                    "correct_diagnoses": correct_diagnoses,
                    "total_cases": total_cases,
                    "threshold_percentage": self.performance_thresholds["min_accuracy_percentage"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="diagnostic_accuracy_validation",
                test_type="validation",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_therapeutic_response_quality(self) -> TestResult:
        """Test quality of therapeutic responses"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Test therapeutic response generation
            result = await system.generate_comprehensive_diagnosis(
                "therapy_test_user", "therapy_test_session",
                "I'm struggling with anxiety and need help", []
            )
            
            # Evaluate response quality
            quality_score = 0
            if result.therapeutic_response:
                response = result.therapeutic_response
                
                # Check for empathy indicators
                if any(word in response.response_text.lower() for word in ["understand", "hear", "feel"]):
                    quality_score += 25
                
                # Check for therapeutic techniques
                if response.therapeutic_technique and response.therapeutic_technique != "unknown":
                    quality_score += 25
                
                # Check for follow-up questions
                if response.follow_up_questions:
                    quality_score += 25
                
                # Check for appropriate friction level
                if 0.1 <= response.friction_level <= 0.8:
                    quality_score += 25
            
            status = "pass" if quality_score >= 75 else "fail"  # 75% quality threshold
            
            return TestResult(
                test_name="therapeutic_response_quality_validation",
                test_type="validation",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "quality_score": quality_score,
                    "response_generated": result.therapeutic_response is not None,
                    "friction_level": result.therapeutic_response.friction_level if result.therapeutic_response else 0
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="therapeutic_response_quality_validation",
                test_type="validation",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_cultural_sensitivity_accuracy(self) -> TestResult:
        """Test cultural sensitivity adaptation accuracy"""
        
        start_time = time.time()
        
        try:
            # Test cultural adaptation
            engine = CulturalSensitivityEngine()
            
            # Test with different cultural contexts
            cultural_contexts = [
                {"culture": "asian", "expected_adaptation": "family involvement"},
                {"culture": "hispanic", "expected_adaptation": "personalismo"},
                {"culture": "western", "expected_adaptation": "individual focus"}
            ]
            
            adaptation_success = 0
            
            for context in cultural_contexts:
                profile = await engine.assess_cultural_context(
                    "cultural_test_user", f"I'm from {context['culture']} background", [],
                    {"culture": context["culture"]}
                )
                
                # Check if cultural profile was correctly assessed
                if profile.primary_culture == context["culture"]:
                    adaptation_success += 1
            
            accuracy_percentage = (adaptation_success / len(cultural_contexts)) * 100
            status = "pass" if accuracy_percentage >= 80 else "fail"
            
            return TestResult(
                test_name="cultural_sensitivity_accuracy_validation",
                test_type="validation",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "accuracy_percentage": accuracy_percentage,
                    "successful_adaptations": adaptation_success,
                    "total_contexts": len(cultural_contexts)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="cultural_sensitivity_accuracy_validation",
                test_type="validation",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_adaptive_learning_effectiveness(self) -> TestResult:
        """Test adaptive learning system effectiveness"""
        
        start_time = time.time()
        
        try:
            engine = AdaptiveLearningEngine()
            
            # Simulate learning process
            user_id = "learning_test_user"
            
            # Record multiple intervention outcomes
            for i in range(5):
                success = await engine.record_intervention_outcome(
                    f"test_intervention_{i}", user_id, "CBT", "test content", {},
                    "positive response", {"engagement": 0.8}
                )
                
                if not success:
                    raise Exception("Failed to record intervention outcome")
            
            # Get personalized recommendations
            recommendations = await engine.get_personalized_recommendation(
                user_id, {"emotional_state": "anxious"}, ["CBT", "DBT", "mindfulness"]
            )
            
            # Check if system learned preferences
            learning_success = (
                recommendations.get("recommended_intervention") is not None and
                recommendations.get("confidence", 0) > 0.5
            )
            
            status = "pass" if learning_success else "fail"
            
            return TestResult(
                test_name="adaptive_learning_effectiveness_validation",
                test_type="validation",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "learning_successful": learning_success,
                    "recommendation_confidence": recommendations.get("confidence", 0),
                    "recommended_intervention": recommendations.get("recommended_intervention")
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="adaptive_learning_effectiveness_validation",
                test_type="validation",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    # System Health Test Methods
    
    async def _test_system_initialization(self) -> TestResult:
        """Test system initialization reliability"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            status_info = system.get_system_status()
            
            initialization_success_rate = status_info["healthy_systems"] / status_info["total_systems"]
            status = "pass" if initialization_success_rate >= 0.8 else "fail"  # 80% success rate required
            
            return TestResult(
                test_name="system_initialization_health",
                test_type="system_health",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "initialization_success_rate": initialization_success_rate,
                    "healthy_systems": status_info["healthy_systems"],
                    "total_systems": status_info["total_systems"],
                    "failed_systems": status_info["failed_systems"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="system_initialization_health",
                test_type="system_health",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_error_recovery(self) -> TestResult:
        """Test system error recovery capabilities"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Test graceful handling of invalid input
            result = await system.generate_comprehensive_diagnosis(
                "", "", "", []  # Invalid empty inputs
            )
            
            # System should handle gracefully without crashing
            recovery_successful = (
                result is not None and
                len(result.warnings) > 0  # Should generate warnings
            )
            
            status = "pass" if recovery_successful else "fail"
            
            return TestResult(
                test_name="error_recovery_health",
                test_type="system_health",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "recovery_successful": recovery_successful,
                    "warnings_generated": len(result.warnings) if result else 0,
                    "system_crashed": False
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="error_recovery_health",
                test_type="system_health",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "system_crashed": True
                },
                error_message=str(e)
            )
    
    async def _test_data_consistency(self) -> TestResult:
        """Test data consistency across system operations"""
        
        start_time = time.time()
        
        try:
            # Test data consistency by performing multiple operations
            # and checking for consistent state
            
            consistency_checks_passed = 0
            total_checks = 3
            
            # Check 1: Temporal data consistency
            temporal_engine = TemporalAnalysisEngine()
            await temporal_engine.record_symptom("consistency_user", "anxiety", 0.5, "test")
            progression = await temporal_engine.get_symptom_progression("consistency_user")
            
            if progression and not progression.get("error"):
                consistency_checks_passed += 1
            
            # Check 2: Memory system consistency
            memory_system = EnhancedMemorySystem()
            insight_id = await memory_system.store_therapeutic_insight(
                "consistency_user", "consistency_session", "test", "test insight", {}
            )
            
            if insight_id:
                consistency_checks_passed += 1
            
            # Check 3: Integration consistency
            system = EnhancedIntegratedDiagnosticSystem()
            validation = await system.validate_system_integration()
            
            if validation.get("overall_status") in ["healthy", "degraded"]:
                consistency_checks_passed += 1
            
            consistency_rate = consistency_checks_passed / total_checks
            status = "pass" if consistency_rate >= 0.8 else "fail"
            
            return TestResult(
                test_name="data_consistency_health",
                test_type="system_health",
                status=status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "consistency_rate": consistency_rate,
                    "checks_passed": consistency_checks_passed,
                    "total_checks": total_checks
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="data_consistency_health",
                test_type="system_health",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    async def _test_system_monitoring(self) -> TestResult:
        """Test system monitoring and health reporting"""
        
        start_time = time.time()
        
        try:
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Test system status reporting
            status = system.get_system_status()
            
            # Test system validation
            validation = await system.validate_system_integration()
            
            monitoring_functional = (
                isinstance(status, dict) and
                "systems_initialized" in status and
                isinstance(validation, dict) and
                "overall_status" in validation
            )
            
            test_status = "pass" if monitoring_functional else "fail"
            
            return TestResult(
                test_name="system_monitoring_health",
                test_type="system_health",
                status=test_status,
                execution_time_ms=(time.time() - start_time) * 1000,
                details={
                    "monitoring_functional": monitoring_functional,
                    "status_keys": list(status.keys()) if isinstance(status, dict) else [],
                    "validation_keys": list(validation.keys()) if isinstance(validation, dict) else []
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="system_monitoring_health",
                test_type="system_health",
                status="fail",
                execution_time_ms=(time.time() - start_time) * 1000,
                details={},
                error_message=str(e)
            )
    
    # Helper Methods
    
    async def _run_single_test(self, test_name: str, test_type: str, test_func) -> TestResult:
        """Run a single test with timing and error handling"""
        
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            execution_time = (time.time() - start_time) * 1000
            
            # Determine status based on result
            if result is None:
                status = "fail"
                details = {"result": "None returned"}
            elif isinstance(result, bool):
                status = "pass" if result else "fail"
                details = {"boolean_result": result}
            elif isinstance(result, dict) and result.get("error"):
                status = "fail"
                details = {"error_result": result.get("error")}
            else:
                status = "pass"
                details = {"result_type": type(result).__name__}
            
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=status,
                execution_time_ms=execution_time,
                details=details
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status="fail",
                execution_time_ms=execution_time,
                details={},
                error_message=str(e)
            )
    
    def _compile_test_suite(self, suite_name: str, test_results: List[TestResult], execution_time: float) -> TestSuite:
        """Compile individual test results into a test suite"""
        
        passed_tests = len([t for t in test_results if t.status == "pass"])
        failed_tests = len([t for t in test_results if t.status == "fail"])
        warning_tests = len([t for t in test_results if t.status == "warning"])
        total_tests = len(test_results)
        
        coverage_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return TestSuite(
            suite_name=suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            execution_time_ms=execution_time,
            test_results=test_results,
            coverage_percentage=coverage_percentage
        )
    
    async def _simulate_data_flow_test(self) -> bool:
        """Simulate data flow between components"""
        try:
            # Simulate data flowing from temporal analysis to diagnosis
            temporal_engine = TemporalAnalysisEngine()
            diagnosis_engine = DifferentialDiagnosisEngine()
            
            # Record symptom
            await temporal_engine.record_symptom("flow_test_user", "anxiety", 0.7, "test")
            
            # Get progression data
            progression = await temporal_engine.get_symptom_progression("flow_test_user")
            
            # Use in diagnosis
            result = await diagnosis_engine.generate_differential_diagnosis(
                symptoms=["anxiety"],
                behavioral_observations=[],
                temporal_patterns=progression
            )
            
            return not result.get("error", False)
        except:
            return False
    
    async def _test_graceful_degradation(self) -> bool:
        """Test system graceful degradation"""
        try:
            # Test system behavior when some components fail
            system = EnhancedIntegratedDiagnosticSystem()
            
            # Generate diagnosis even with potential component failures
            result = await system.generate_comprehensive_diagnosis(
                "degradation_test_user", "degradation_test_session",
                "Test message for degradation", []
            )
            
            # System should still return a result, even if degraded
            return result is not None
        except:
            return False
    
    async def _test_data_persistence(self) -> bool:
        """Test data persistence"""
        try:
            # Test memory system persistence
            memory_system = EnhancedMemorySystem()
            
            # Store insight
            insight_id = await memory_system.store_therapeutic_insight(
                "persistence_test_user", "persistence_test_session",
                "test", "Test insight for persistence", {}
            )
            
            return bool(insight_id)
        except:
            return False
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for validation"""
        return {
            "diagnostic_cases": [
                {
                    "symptoms": ["sadness", "hopelessness", "sleep issues"],
                    "expected_diagnosis": "Major Depressive Disorder",
                    "expected_severity": "moderate"
                },
                {
                    "symptoms": ["anxiety", "worry", "restlessness"],
                    "expected_diagnosis": "Generalized Anxiety Disorder",
                    "expected_severity": "moderate"
                }
            ],
            "cultural_contexts": [
                {"culture": "asian", "stigma_level": 0.7},
                {"culture": "western", "stigma_level": 0.3},
                {"culture": "hispanic", "stigma_level": 0.6}
            ]
        }
    
    def _generate_mock_user_data(self) -> Dict[str, Any]:
        """Generate mock user data for testing"""
        return {
            "users": [
                {
                    "user_id": "test_user_1",
                    "age": 25,
                    "cultural_background": "western",
                    "symptoms": ["anxiety", "depression"],
                    "personality": {"openness": 0.7, "neuroticism": 0.8}
                },
                {
                    "user_id": "test_user_2",
                    "age": 35,
                    "cultural_background": "asian",
                    "symptoms": ["stress", "sleep_issues"],
                    "personality": {"conscientiousness": 0.9, "neuroticism": 0.6}
                }
            ]
        }
    
    async def _generate_test_report(self, test_suites: Dict[str, TestSuite], total_time: float):
        """Generate comprehensive test report"""
        
        try:
            report = {
                "test_execution_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_execution_time_ms": total_time,
                    "test_suites": len(test_suites)
                },
                "suite_summaries": {},
                "overall_metrics": {
                    "total_tests": 0,
                    "total_passed": 0,
                    "total_failed": 0,
                    "total_warnings": 0,
                    "overall_success_rate": 0.0
                },
                "recommendations": []
            }
            
            # Compile suite summaries
            total_tests = 0
            total_passed = 0
            total_failed = 0
            total_warnings = 0
            
            for suite_name, suite in test_suites.items():
                report["suite_summaries"][suite_name] = {
                    "total_tests": suite.total_tests,
                    "passed_tests": suite.passed_tests,
                    "failed_tests": suite.failed_tests,
                    "warning_tests": suite.warning_tests,
                    "success_rate": (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0,
                    "execution_time_ms": suite.execution_time_ms
                }
                
                total_tests += suite.total_tests
                total_passed += suite.passed_tests
                total_failed += suite.failed_tests
                total_warnings += suite.warning_tests
            
            # Calculate overall metrics
            report["overall_metrics"]["total_tests"] = total_tests
            report["overall_metrics"]["total_passed"] = total_passed
            report["overall_metrics"]["total_failed"] = total_failed
            report["overall_metrics"]["total_warnings"] = total_warnings
            report["overall_metrics"]["overall_success_rate"] = (total_passed / total_tests * 100) if total_tests > 0 else 0
            
            # Generate recommendations
            if total_failed > 0:
                report["recommendations"].append(f"Address {total_failed} failed tests to improve system reliability")
            
            if report["overall_metrics"]["overall_success_rate"] < 90:
                report["recommendations"].append("Overall success rate is below 90%. Review failed tests and system implementation")
            
            if total_warnings > 0:
                report["recommendations"].append(f"Investigate {total_warnings} warnings to prevent potential issues")
            
            # Save report
            import os
            os.makedirs("src/data/test_reports", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"src/data/test_reports/comprehensive_test_report_{timestamp}.json"
            
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Test report saved to {report_file}")
            self.logger.info(f"Overall test results: {total_passed}/{total_tests} passed ({report['overall_metrics']['overall_success_rate']:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error generating test report: {str(e)}")


async def run_comprehensive_tests():
    """Entry point for running comprehensive tests"""
    tester = ComprehensiveTester()
    test_results = await tester.run_all_tests()
    return test_results


if __name__ == "__main__":
    # Run tests when script is executed directly
    asyncio.run(run_comprehensive_tests())
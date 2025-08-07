"""
Automated Quality Assurance and Validation Framework for Solace-AI

This module provides comprehensive quality assurance capabilities including:
- Automated testing and validation pipelines
- Content quality assessment
- Clinical safety validation
- Performance quality gates
- Continuous quality monitoring
- Quality metrics tracking
- Automated remediation
- Compliance validation
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import threading
from abc import ABC, abstractmethod
import statistics

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority
from src.integration.supervision_mesh import (
    SupervisionMesh, QualityGateType, ValidationRequest, ValidationGateResult, ConsensusResult
)

logger = get_logger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class TestType(Enum):
    """Types of automated tests."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    SAFETY_TEST = "safety_test"
    COMPLIANCE_TEST = "compliance_test"
    USER_ACCEPTANCE_TEST = "user_acceptance_test"
    REGRESSION_TEST = "regression_test"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKER = "blocker"


@dataclass
class QualityMetric:
    """Quality metric definition and measurement."""
    
    metric_name: str
    value: float
    target_value: float
    unit: str
    quality_level: QualityLevel
    measurement_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_meeting_target(self) -> bool:
        """Check if metric meets target value."""
        return self.value >= self.target_value
    
    def calculate_deviation(self) -> float:
        """Calculate percentage deviation from target."""
        if self.target_value == 0:
            return 0.0
        return ((self.value - self.target_value) / self.target_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'target_value': self.target_value,
            'unit': self.unit,
            'quality_level': self.quality_level.value,
            'meets_target': self.is_meeting_target(),
            'deviation_percentage': self.calculate_deviation(),
            'measurement_time': self.measurement_time.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class QualityGate:
    """Quality gate definition for automated validation."""
    
    gate_id: str
    name: str
    description: str
    quality_criteria: List[Dict[str, Any]]
    required_score: float
    weight: float = 1.0
    is_blocking: bool = True
    timeout_seconds: int = 30
    
    def evaluate(self, metrics: List[QualityMetric]) -> Tuple[bool, float, List[str]]:
        """Evaluate quality gate against metrics."""
        issues = []
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in self.quality_criteria:
            metric_name = criterion['metric']
            min_value = criterion.get('min_value')
            max_value = criterion.get('max_value')
            weight = criterion.get('weight', 1.0)
            
            # Find matching metric
            matching_metrics = [m for m in metrics if m.metric_name == metric_name]
            if not matching_metrics:
                issues.append(f"Missing metric: {metric_name}")
                continue
            
            metric = matching_metrics[0]
            
            # Evaluate criteria
            score = 100.0
            if min_value is not None and metric.value < min_value:
                score = (metric.value / min_value) * 100
                issues.append(f"{metric_name} below minimum: {metric.value} < {min_value}")
            
            if max_value is not None and metric.value > max_value:
                score = (max_value / metric.value) * 100
                issues.append(f"{metric_name} above maximum: {metric.value} > {max_value}")
            
            total_score += score * weight
            total_weight += weight
        
        average_score = total_score / max(total_weight, 1)
        passes = average_score >= self.required_score and len(issues) == 0
        
        return passes, average_score, issues


@dataclass
class TestCase:
    """Test case definition for automated testing."""
    
    test_id: str
    name: str
    test_type: TestType
    description: str
    test_function: Optional[Callable] = None
    expected_outcome: Any = None
    timeout_seconds: int = 60
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    preconditions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding functions)."""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'test_type': self.test_type.value,
            'description': self.description,
            'expected_outcome': str(self.expected_outcome),
            'timeout_seconds': self.timeout_seconds,
            'preconditions': self.preconditions,
            'tags': self.tags
        }


@dataclass
class TestResult:
    """Test execution result."""
    
    test_id: str
    passed: bool
    execution_time: float
    actual_outcome: Any = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'actual_outcome': str(self.actual_outcome),
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    
    report_id: str
    assessment_type: str
    overall_quality_score: float
    quality_level: QualityLevel
    metrics: List[QualityMetric]
    quality_gates_passed: int
    quality_gates_failed: int
    test_results: List[TestResult]
    recommendations: List[str]
    critical_issues: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'assessment_type': self.assessment_type,
            'overall_quality_score': self.overall_quality_score,
            'quality_level': self.quality_level.value,
            'metrics': [m.to_dict() for m in self.metrics],
            'quality_gates_passed': self.quality_gates_passed,
            'quality_gates_failed': self.quality_gates_failed,
            'test_results': [tr.to_dict() for tr in self.test_results],
            'recommendations': self.recommendations,
            'critical_issues': self.critical_issues,
            'timestamp': self.timestamp.isoformat()
        }


class QualityAssessment(ABC):
    """Abstract base class for quality assessments."""
    
    def __init__(self, assessment_name: str):
        self.assessment_name = assessment_name
    
    @abstractmethod
    async def assess(self, target: Any, context: Dict[str, Any] = None) -> List[QualityMetric]:
        """Perform quality assessment and return metrics."""
        pass


class ContentQualityAssessment(QualityAssessment):
    """Assessment for content quality (responses, messages, etc.)."""
    
    def __init__(self):
        super().__init__("content_quality")
    
    async def assess(self, content: str, context: Dict[str, Any] = None) -> List[QualityMetric]:
        """Assess content quality."""
        metrics = []
        
        # Length appropriateness
        length_score = self._assess_length(content)
        metrics.append(QualityMetric(
            metric_name="content_length_appropriateness",
            value=length_score,
            target_value=70.0,
            unit="score",
            quality_level=self._score_to_quality_level(length_score)
        ))
        
        # Clarity and coherence
        clarity_score = await self._assess_clarity(content)
        metrics.append(QualityMetric(
            metric_name="content_clarity",
            value=clarity_score,
            target_value=80.0,
            unit="score",
            quality_level=self._score_to_quality_level(clarity_score)
        ))
        
        # Professional tone
        tone_score = await self._assess_tone(content)
        metrics.append(QualityMetric(
            metric_name="professional_tone",
            value=tone_score,
            target_value=85.0,
            unit="score",
            quality_level=self._score_to_quality_level(tone_score)
        ))
        
        # Empathy and sensitivity
        empathy_score = await self._assess_empathy(content, context)
        metrics.append(QualityMetric(
            metric_name="empathy_sensitivity",
            value=empathy_score,
            target_value=80.0,
            unit="score",
            quality_level=self._score_to_quality_level(empathy_score)
        ))
        
        return metrics
    
    def _assess_length(self, content: str) -> float:
        """Assess content length appropriateness."""
        word_count = len(content.split())
        
        # Optimal range: 50-500 words for most responses
        if 50 <= word_count <= 500:
            return 100.0
        elif word_count < 50:
            return max(20.0, (word_count / 50) * 100)
        else:
            return max(20.0, 100 - ((word_count - 500) / 10))
    
    async def _assess_clarity(self, content: str) -> float:
        """Assess content clarity and coherence."""
        # Simplified assessment - would use NLP models in practice
        
        # Check for clear structure
        sentences = content.split('.')
        avg_sentence_length = statistics.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length: 15-25 words
        if 15 <= avg_sentence_length <= 25:
            structure_score = 100.0
        else:
            structure_score = max(50.0, 100 - abs(avg_sentence_length - 20) * 2)
        
        # Check for transitional words/phrases
        transitions = ['however', 'therefore', 'additionally', 'furthermore', 'meanwhile', 'consequently']
        transition_count = sum(1 for word in transitions if word in content.lower())
        transition_score = min(100.0, (transition_count / len(sentences)) * 200)
        
        return (structure_score + transition_score) / 2
    
    async def _assess_tone(self, content: str) -> float:
        """Assess professional tone."""
        # Simplified tone assessment
        
        # Check for professional language
        professional_indicators = [
            'understand', 'support', 'assistance', 'recommend', 'consider',
            'appropriate', 'effective', 'beneficial', 'important'
        ]
        
        unprofessional_indicators = [
            'obviously', 'duh', 'stupid', 'ridiculous', 'whatever',
            'yeah right', 'come on', 'seriously'
        ]
        
        content_lower = content.lower()
        professional_count = sum(1 for word in professional_indicators if word in content_lower)
        unprofessional_count = sum(1 for word in unprofessional_indicators if word in content_lower)
        
        total_words = len(content.split())
        
        professional_ratio = (professional_count / max(total_words, 1)) * 1000
        unprofessional_penalty = (unprofessional_count / max(total_words, 1)) * 2000
        
        tone_score = max(0.0, min(100.0, 80 + professional_ratio - unprofessional_penalty))
        return tone_score
    
    async def _assess_empathy(self, content: str, context: Dict[str, Any] = None) -> float:
        """Assess empathy and sensitivity."""
        
        empathy_indicators = [
            'understand how you feel', 'that must be difficult', 'i can imagine',
            'it sounds like', 'you\'re going through', 'i hear you',
            'that\'s understandable', 'it\'s natural to feel', 'many people experience'
        ]
        
        sensitivity_indicators = [
            'please', 'gently', 'carefully', 'thoughtfully', 'respectfully',
            'at your own pace', 'when you\'re ready', 'if you feel comfortable'
        ]
        
        content_lower = content.lower()
        
        empathy_count = sum(1 for phrase in empathy_indicators if phrase in content_lower)
        sensitivity_count = sum(1 for word in sensitivity_indicators if word in content_lower)
        
        # Bonus for emotional context awareness
        context_bonus = 0
        if context and context.get('emotional_state'):
            emotional_words = ['sad', 'angry', 'frustrated', 'anxious', 'depressed']
            if any(emotion in content_lower for emotion in emotional_words):
                context_bonus = 10
        
        empathy_score = min(100.0, (empathy_count * 15) + (sensitivity_count * 10) + context_bonus + 50)
        return empathy_score
    
    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class ClinicalSafetyAssessment(QualityAssessment):
    """Assessment for clinical safety and appropriateness."""
    
    def __init__(self):
        super().__init__("clinical_safety")
    
    async def assess(self, content: str, context: Dict[str, Any] = None) -> List[QualityMetric]:
        """Assess clinical safety."""
        metrics = []
        
        # Risk assessment appropriateness
        risk_score = await self._assess_risk_handling(content, context)
        metrics.append(QualityMetric(
            metric_name="risk_assessment_appropriateness",
            value=risk_score,
            target_value=90.0,
            unit="score",
            quality_level=self._score_to_quality_level(risk_score)
        ))
        
        # Professional boundary maintenance
        boundary_score = await self._assess_boundaries(content)
        metrics.append(QualityMetric(
            metric_name="professional_boundaries",
            value=boundary_score,
            target_value=95.0,
            unit="score",
            quality_level=self._score_to_quality_level(boundary_score)
        ))
        
        # Appropriate referral recommendations
        referral_score = await self._assess_referrals(content, context)
        metrics.append(QualityMetric(
            metric_name="appropriate_referrals",
            value=referral_score,
            target_value=85.0,
            unit="score",
            quality_level=self._score_to_quality_level(referral_score)
        ))
        
        # Crisis intervention appropriateness
        crisis_score = await self._assess_crisis_handling(content, context)
        metrics.append(QualityMetric(
            metric_name="crisis_intervention",
            value=crisis_score,
            target_value=98.0,
            unit="score",
            quality_level=self._score_to_quality_level(crisis_score)
        ))
        
        return metrics
    
    async def _assess_risk_handling(self, content: str, context: Dict[str, Any] = None) -> float:
        """Assess risk handling appropriateness."""
        content_lower = content.lower()
        
        # Check for appropriate risk indicators
        risk_awareness = [
            'if you\'re having thoughts', 'safety planning', 'professional help',
            'emergency services', 'crisis line', 'immediate assistance'
        ]
        
        inappropriate_minimizing = [
            'it\'s not that bad', 'just think positive', 'everyone feels this way',
            'you\'ll be fine', 'don\'t worry about it'
        ]
        
        risk_count = sum(1 for phrase in risk_awareness if phrase in content_lower)
        minimizing_count = sum(1 for phrase in inappropriate_minimizing if phrase in content_lower)
        
        # Base score
        base_score = 80.0
        
        # Context-based adjustments
        if context and context.get('risk_indicators'):
            risk_level = context.get('risk_level', 'low')
            if risk_level in ['high', 'severe'] and risk_count > 0:
                base_score += 15.0
            elif risk_level in ['high', 'severe'] and risk_count == 0:
                base_score -= 30.0
        
        # Penalties for inappropriate responses
        base_score -= minimizing_count * 20
        
        return max(0.0, min(100.0, base_score))
    
    async def _assess_boundaries(self, content: str) -> float:
        """Assess professional boundary maintenance."""
        content_lower = content.lower()
        
        boundary_violations = [
            'i love you', 'you\'re my friend', 'let\'s meet', 'my personal',
            'i\'ll solve this for you', 'don\'t tell anyone', 'this is between us'
        ]
        
        appropriate_boundaries = [
            'professional relationship', 'within my role', 'appropriate support',
            'professional guidance', 'therapeutic relationship'
        ]
        
        violation_count = sum(1 for phrase in boundary_violations if phrase in content_lower)
        appropriate_count = sum(1 for phrase in appropriate_boundaries if phrase in content_lower)
        
        boundary_score = max(0.0, 95.0 - (violation_count * 25) + (appropriate_count * 5))
        return min(100.0, boundary_score)
    
    async def _assess_referrals(self, content: str, context: Dict[str, Any] = None) -> float:
        """Assess appropriateness of referral recommendations."""
        content_lower = content.lower()
        
        referral_indicators = [
            'professional help', 'therapist', 'counselor', 'psychiatrist',
            'mental health professional', 'specialized treatment', 'clinical assessment'
        ]
        
        appropriate_language = [
            'might benefit from', 'consider speaking with', 'could be helpful to',
            'recommend consulting', 'suggest connecting with'
        ]
        
        referral_count = sum(1 for phrase in referral_indicators if phrase in content_lower)
        appropriate_language_count = sum(1 for phrase in appropriate_language if phrase in content_lower)
        
        base_score = 70.0
        
        # Context-based scoring
        if context:
            severity = context.get('severity', 'mild')
            if severity in ['severe', 'critical'] and referral_count > 0:
                base_score += 20.0
            elif severity in ['moderate', 'severe'] and referral_count == 0:
                base_score -= 15.0
        
        # Bonus for appropriate language
        base_score += appropriate_language_count * 5
        
        return max(0.0, min(100.0, base_score))
    
    async def _assess_crisis_handling(self, content: str, context: Dict[str, Any] = None) -> float:
        """Assess crisis intervention appropriateness."""
        content_lower = content.lower()
        
        crisis_indicators = [
            'crisis hotline', 'emergency services', '911', 'immediate help',
            'safety plan', 'crisis intervention', 'emergency room'
        ]
        
        inappropriate_crisis_responses = [
            'calm down', 'it\'ll pass', 'just relax', 'don\'t be dramatic'
        ]
        
        crisis_response_count = sum(1 for phrase in crisis_indicators if phrase in content_lower)
        inappropriate_count = sum(1 for phrase in inappropriate_crisis_responses if phrase in content_lower)
        
        base_score = 95.0  # High baseline for crisis handling
        
        # Context-based adjustments
        if context:
            crisis_level = context.get('crisis_level', 'none')
            if crisis_level in ['high', 'immediate'] and crisis_response_count > 0:
                base_score = 100.0
            elif crisis_level in ['high', 'immediate'] and crisis_response_count == 0:
                base_score = 0.0  # Critical failure
        
        # Severe penalties for inappropriate responses
        base_score -= inappropriate_count * 40
        
        return max(0.0, min(100.0, base_score))
    
    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert score to quality level with stricter thresholds for safety."""
        if score >= 95:
            return QualityLevel.EXCELLENT
        elif score >= 90:
            return QualityLevel.GOOD
        elif score >= 80:
            return QualityLevel.ACCEPTABLE
        elif score >= 70:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class PerformanceQualityAssessment(QualityAssessment):
    """Assessment for performance quality metrics."""
    
    def __init__(self):
        super().__init__("performance_quality")
    
    async def assess(self, performance_data: Dict[str, Any], context: Dict[str, Any] = None) -> List[QualityMetric]:
        """Assess performance quality."""
        metrics = []
        
        # Response time quality
        response_time = performance_data.get('response_time', 0)
        response_time_score = self._assess_response_time(response_time)
        metrics.append(QualityMetric(
            metric_name="response_time_quality",
            value=response_time_score,
            target_value=80.0,
            unit="score",
            quality_level=self._score_to_quality_level(response_time_score)
        ))
        
        # Throughput quality
        throughput = performance_data.get('throughput', 0)
        throughput_score = self._assess_throughput(throughput)
        metrics.append(QualityMetric(
            metric_name="throughput_quality",
            value=throughput_score,
            target_value=75.0,
            unit="score",
            quality_level=self._score_to_quality_level(throughput_score)
        ))
        
        # Error rate quality
        error_rate = performance_data.get('error_rate', 0)
        error_rate_score = self._assess_error_rate(error_rate)
        metrics.append(QualityMetric(
            metric_name="error_rate_quality",
            value=error_rate_score,
            target_value=90.0,
            unit="score",
            quality_level=self._score_to_quality_level(error_rate_score)
        ))
        
        # Availability quality
        availability = performance_data.get('availability', 0)
        availability_score = self._assess_availability(availability)
        metrics.append(QualityMetric(
            metric_name="availability_quality",
            value=availability_score,
            target_value=95.0,
            unit="score",
            quality_level=self._score_to_quality_level(availability_score)
        ))
        
        return metrics
    
    def _assess_response_time(self, response_time: float) -> float:
        """Assess response time quality."""
        # Excellent: < 2s, Good: < 5s, Acceptable: < 10s
        if response_time < 2:
            return 100.0
        elif response_time < 5:
            return max(80.0, 100 - ((response_time - 2) * 10))
        elif response_time < 10:
            return max(60.0, 80 - ((response_time - 5) * 4))
        else:
            return max(0.0, 60 - ((response_time - 10) * 2))
    
    def _assess_throughput(self, throughput: float) -> float:
        """Assess throughput quality."""
        # Target: >50 requests/min is excellent
        if throughput >= 50:
            return 100.0
        elif throughput >= 20:
            return 75 + ((throughput - 20) / 30) * 25
        elif throughput >= 10:
            return 50 + ((throughput - 10) / 10) * 25
        else:
            return (throughput / 10) * 50
    
    def _assess_error_rate(self, error_rate: float) -> float:
        """Assess error rate quality (lower is better)."""
        # Excellent: <1%, Good: <5%, Acceptable: <10%
        if error_rate < 0.01:  # <1%
            return 100.0
        elif error_rate < 0.05:  # <5%
            return max(80.0, 100 - ((error_rate - 0.01) / 0.04) * 20)
        elif error_rate < 0.1:   # <10%
            return max(60.0, 80 - ((error_rate - 0.05) / 0.05) * 20)
        else:
            return max(0.0, 60 - ((error_rate - 0.1) * 600))
    
    def _assess_availability(self, availability: float) -> float:
        """Assess availability quality."""
        # Availability should be a percentage (0-100)
        if availability >= 99.9:
            return 100.0
        elif availability >= 99.0:
            return 90 + ((availability - 99.0) / 0.9) * 10
        elif availability >= 95.0:
            return 70 + ((availability - 95.0) / 4.0) * 20
        else:
            return (availability / 95.0) * 70
    
    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE


class AutomatedTestRunner:
    """Runs automated tests and collects results."""
    
    def __init__(self):
        self.test_cases: Dict[str, TestCase] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        self.test_suites: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
    
    def register_test(self, test_case: TestCase) -> None:
        """Register a test case."""
        with self.lock:
            self.test_cases[test_case.test_id] = test_case
        logger.info(f"Registered test case: {test_case.test_id}")
    
    def register_test_suite(self, suite_name: str, test_ids: List[str]) -> None:
        """Register a test suite."""
        with self.lock:
            self.test_suites[suite_name] = test_ids
        logger.info(f"Registered test suite '{suite_name}' with {len(test_ids)} tests")
    
    async def run_test(self, test_id: str, context: Dict[str, Any] = None) -> TestResult:
        """Run a single test case."""
        if test_id not in self.test_cases:
            return TestResult(
                test_id=test_id,
                passed=False,
                execution_time=0.0,
                error_message=f"Test {test_id} not found"
            )
        
        test_case = self.test_cases[test_id]
        start_time = time.time()
        
        try:
            # Run setup if available
            if test_case.setup_function:
                if asyncio.iscoroutinefunction(test_case.setup_function):
                    await test_case.setup_function(context)
                else:
                    test_case.setup_function(context)
            
            # Run the actual test
            if test_case.test_function:
                if asyncio.iscoroutinefunction(test_case.test_function):
                    actual_outcome = await asyncio.wait_for(
                        test_case.test_function(context),
                        timeout=test_case.timeout_seconds
                    )
                else:
                    actual_outcome = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: asyncio.wait_for(
                            asyncio.coroutine(lambda: test_case.test_function(context))(),
                            timeout=test_case.timeout_seconds
                        )
                    )
                
                # Evaluate result
                passed = self._evaluate_test_result(actual_outcome, test_case.expected_outcome)
                
                result = TestResult(
                    test_id=test_id,
                    passed=passed,
                    execution_time=time.time() - start_time,
                    actual_outcome=actual_outcome
                )
            else:
                result = TestResult(
                    test_id=test_id,
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message="No test function defined"
                )
        
        except asyncio.TimeoutError:
            result = TestResult(
                test_id=test_id,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=f"Test timed out after {test_case.timeout_seconds} seconds"
            )
        
        except Exception as e:
            result = TestResult(
                test_id=test_id,
                passed=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                stack_trace=str(e.__traceback__) if hasattr(e, '__traceback__') else None
            )
        
        finally:
            # Run teardown if available
            try:
                if test_case.teardown_function:
                    if asyncio.iscoroutinefunction(test_case.teardown_function):
                        await test_case.teardown_function(context)
                    else:
                        test_case.teardown_function(context)
            except Exception as teardown_error:
                logger.warning(f"Teardown failed for test {test_id}: {teardown_error}")
        
        # Store result
        with self.lock:
            self.test_results[test_id].append(result)
        
        return result
    
    async def run_test_suite(self, suite_name: str, context: Dict[str, Any] = None) -> List[TestResult]:
        """Run all tests in a test suite."""
        if suite_name not in self.test_suites:
            return []
        
        test_ids = self.test_suites[suite_name]
        results = []
        
        # Run tests concurrently
        tasks = [self.run_test(test_id, context) for test_id in test_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to TestResults
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, TestResult):
                valid_results.append(result)
            else:
                # Create error result for failed test
                valid_results.append(TestResult(
                    test_id=test_ids[i],
                    passed=False,
                    execution_time=0.0,
                    error_message=f"Test execution failed: {str(result)}"
                ))
        
        return valid_results
    
    def _evaluate_test_result(self, actual: Any, expected: Any) -> bool:
        """Evaluate if test result matches expected outcome."""
        if expected is None:
            # If no expected outcome specified, just check that it didn't raise an exception
            return True
        
        if callable(expected):
            # If expected is a function, use it to evaluate the result
            try:
                return expected(actual)
            except Exception:
                return False
        
        # Simple equality check
        return actual == expected
    
    def get_test_statistics(self, test_type: TestType = None) -> Dict[str, Any]:
        """Get test execution statistics."""
        with self.lock:
            all_results = []
            for results_list in self.test_results.values():
                all_results.extend(results_list)
            
            if test_type:
                # Filter by test type
                filtered_results = []
                for result in all_results:
                    if result.test_id in self.test_cases:
                        test_case = self.test_cases[result.test_id]
                        if test_case.test_type == test_type:
                            filtered_results.append(result)
                all_results = filtered_results
            
            if not all_results:
                return {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'pass_rate': 0.0,
                    'average_execution_time': 0.0
                }
            
            passed_tests = sum(1 for r in all_results if r.passed)
            failed_tests = len(all_results) - passed_tests
            pass_rate = (passed_tests / len(all_results)) * 100
            avg_execution_time = statistics.mean([r.execution_time for r in all_results])
            
            return {
                'total_tests': len(all_results),
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': pass_rate,
                'average_execution_time': avg_execution_time,
                'test_type': test_type.value if test_type else 'all'
            }


class QualityAssuranceFramework:
    """
    Comprehensive quality assurance framework providing automated testing,
    quality assessment, and continuous monitoring capabilities.
    """
    
    def __init__(self, event_bus: EventBus, supervision_mesh: SupervisionMesh):
        self.event_bus = event_bus
        self.supervision_mesh = supervision_mesh
        
        # Assessment components
        self.content_assessment = ContentQualityAssessment()
        self.safety_assessment = ClinicalSafetyAssessment()
        self.performance_assessment = PerformanceQualityAssessment()
        
        # Testing components
        self.test_runner = AutomatedTestRunner()
        
        # Quality gates and metrics
        self.quality_gates: Dict[str, QualityGate] = {}
        self.quality_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.continuous_monitoring_enabled = True
        self.quality_threshold = 70.0
        self.auto_remediation_enabled = True
        
        # Background tasks
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default quality gates
        self._setup_default_quality_gates()
        
        # Setup default test cases
        self._setup_default_tests()
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        logger.info("QualityAssuranceFramework initialized")
    
    async def start(self) -> None:
        """Start the quality assurance framework."""
        if self._running:
            return
        
        self._running = True
        
        # Start continuous monitoring
        if self.continuous_monitoring_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Emit startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="quality_assurance",
            data={'component': 'qa_framework', 'status': 'started'}
        ))
        
        logger.info("QualityAssuranceFramework started")
    
    async def stop(self) -> None:
        """Stop the quality assurance framework."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("QualityAssuranceFramework stopped")
    
    async def assess_quality(self, 
                           target: Any, 
                           assessment_type: str = "comprehensive",
                           context: Dict[str, Any] = None) -> QualityReport:
        """Perform comprehensive quality assessment."""
        
        report_id = f"qa_report_{int(time.time() * 1000000)}"
        all_metrics = []
        test_results = []
        
        context = context or {}
        
        try:
            # Content quality assessment
            if assessment_type in ["comprehensive", "content"]:
                if isinstance(target, str):
                    content_metrics = await self.content_assessment.assess(target, context)
                    all_metrics.extend(content_metrics)
            
            # Clinical safety assessment
            if assessment_type in ["comprehensive", "clinical", "safety"]:
                if isinstance(target, str):
                    safety_metrics = await self.safety_assessment.assess(target, context)
                    all_metrics.extend(safety_metrics)
            
            # Performance assessment
            if assessment_type in ["comprehensive", "performance"]:
                if isinstance(target, dict) and 'performance_data' in target:
                    perf_metrics = await self.performance_assessment.assess(
                        target['performance_data'], context
                    )
                    all_metrics.extend(perf_metrics)
            
            # Run relevant tests
            if assessment_type in ["comprehensive", "testing"]:
                test_suite_name = context.get('test_suite', 'default')
                if test_suite_name in self.test_runner.test_suites:
                    test_results = await self.test_runner.run_test_suite(test_suite_name, context)
            
            # Evaluate quality gates
            gates_passed = 0
            gates_failed = 0
            
            for gate_id, quality_gate in self.quality_gates.items():
                try:
                    passes, score, issues = quality_gate.evaluate(all_metrics)
                    if passes:
                        gates_passed += 1
                    else:
                        gates_failed += 1
                except Exception as e:
                    logger.error(f"Error evaluating quality gate {gate_id}: {e}")
                    gates_failed += 1
            
            # Calculate overall quality score
            if all_metrics:
                overall_score = statistics.mean([m.value for m in all_metrics])
            else:
                overall_score = 0.0
            
            # Adjust score based on test results
            if test_results:
                test_pass_rate = sum(1 for tr in test_results if tr.passed) / len(test_results)
                overall_score = (overall_score + test_pass_rate * 100) / 2
            
            # Determine quality level
            if overall_score >= 90:
                quality_level = QualityLevel.EXCELLENT
            elif overall_score >= 80:
                quality_level = QualityLevel.GOOD
            elif overall_score >= 70:
                quality_level = QualityLevel.ACCEPTABLE
            elif overall_score >= 50:
                quality_level = QualityLevel.POOR
            else:
                quality_level = QualityLevel.UNACCEPTABLE
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(all_metrics, test_results)
            
            # Identify critical issues
            critical_issues = [
                f"{metric.metric_name}: {metric.value:.1f} (target: {metric.target_value:.1f})"
                for metric in all_metrics
                if metric.quality_level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]
            ]
            
            # Add test failures as critical issues
            critical_issues.extend([
                f"Test failure: {tr.test_id} - {tr.error_message}"
                for tr in test_results
                if not tr.passed and tr.error_message
            ])
            
            # Create quality report
            report = QualityReport(
                report_id=report_id,
                assessment_type=assessment_type,
                overall_quality_score=overall_score,
                quality_level=quality_level,
                metrics=all_metrics,
                quality_gates_passed=gates_passed,
                quality_gates_failed=gates_failed,
                test_results=test_results,
                recommendations=recommendations,
                critical_issues=critical_issues
            )
            
            # Store metrics history
            for metric in all_metrics:
                self.quality_metrics_history[metric.metric_name].append(metric)
            
            # Publish quality report event
            await self.event_bus.publish(Event(
                event_type="quality_assessment_completed",
                source_agent="quality_assurance",
                priority=EventPriority.HIGH if quality_level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE] else EventPriority.NORMAL,
                data={
                    'report_id': report_id,
                    'quality_score': overall_score,
                    'quality_level': quality_level.value,
                    'critical_issues_count': len(critical_issues)
                }
            ))
            
            return report
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            
            # Return error report
            return QualityReport(
                report_id=report_id,
                assessment_type=assessment_type,
                overall_quality_score=0.0,
                quality_level=QualityLevel.UNACCEPTABLE,
                metrics=[],
                quality_gates_passed=0,
                quality_gates_failed=len(self.quality_gates),
                test_results=[],
                recommendations=[f"Quality assessment failed: {str(e)}"],
                critical_issues=[f"Assessment error: {str(e)}"]
            )
    
    def add_quality_gate(self, quality_gate: QualityGate) -> None:
        """Add a quality gate."""
        self.quality_gates[quality_gate.gate_id] = quality_gate
        logger.info(f"Added quality gate: {quality_gate.gate_id}")
    
    def remove_quality_gate(self, gate_id: str) -> bool:
        """Remove a quality gate."""
        if gate_id in self.quality_gates:
            del self.quality_gates[gate_id]
            logger.info(f"Removed quality gate: {gate_id}")
            return True
        return False
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case."""
        self.test_runner.register_test(test_case)
    
    def add_test_suite(self, suite_name: str, test_ids: List[str]) -> None:
        """Add a test suite."""
        self.test_runner.register_test_suite(suite_name, test_ids)
    
    async def run_quality_tests(self, test_suite: str = "default") -> List[TestResult]:
        """Run quality tests."""
        return await self.test_runner.run_test_suite(test_suite)
    
    def get_quality_trends(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Get quality trends for a specific metric."""
        
        if metric_name not in self.quality_metrics_history:
            return {'error': f'No data for metric: {metric_name}'}
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.quality_metrics_history[metric_name]
            if m.measurement_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': f'No recent data for metric: {metric_name}'}
        
        values = [m.value for m in recent_metrics]
        targets = [m.target_value for m in recent_metrics]
        
        return {
            'metric_name': metric_name,
            'data_points': len(values),
            'current_value': values[-1],
            'target_value': targets[-1],
            'average_value': statistics.mean(values),
            'min_value': min(values),
            'max_value': max(values),
            'trend': 'improving' if values[-1] > values[0] else 'declining',
            'meets_target': values[-1] >= targets[-1],
            'days_analyzed': days
        }
    
    def get_overall_quality_status(self) -> Dict[str, Any]:
        """Get overall quality status summary."""
        
        # Get recent quality scores
        recent_scores = []
        for metric_history in self.quality_metrics_history.values():
            if metric_history:
                recent_scores.append(metric_history[-1].value)
        
        if recent_scores:
            overall_score = statistics.mean(recent_scores)
            quality_level = self._score_to_quality_level(overall_score)
        else:
            overall_score = 0.0
            quality_level = QualityLevel.UNACCEPTABLE
        
        # Get test statistics
        test_stats = self.test_runner.get_test_statistics()
        
        return {
            'overall_quality_score': overall_score,
            'quality_level': quality_level.value,
            'metrics_tracked': len(self.quality_metrics_history),
            'quality_gates_configured': len(self.quality_gates),
            'test_statistics': test_stats,
            'monitoring_enabled': self.continuous_monitoring_enabled,
            'auto_remediation_enabled': self.auto_remediation_enabled,
            'timestamp': datetime.now().isoformat()
        }
    
    def _setup_default_quality_gates(self) -> None:
        """Setup default quality gates."""
        
        # Content quality gate
        content_gate = QualityGate(
            gate_id="content_quality_gate",
            name="Content Quality Gate",
            description="Ensures content meets quality standards",
            quality_criteria=[
                {'metric': 'content_clarity', 'min_value': 75.0, 'weight': 1.0},
                {'metric': 'professional_tone', 'min_value': 80.0, 'weight': 1.0},
                {'metric': 'empathy_sensitivity', 'min_value': 70.0, 'weight': 1.0}
            ],
            required_score=75.0
        )
        self.add_quality_gate(content_gate)
        
        # Clinical safety gate
        safety_gate = QualityGate(
            gate_id="clinical_safety_gate",
            name="Clinical Safety Gate",
            description="Ensures clinical safety standards",
            quality_criteria=[
                {'metric': 'risk_assessment_appropriateness', 'min_value': 90.0, 'weight': 2.0},
                {'metric': 'professional_boundaries', 'min_value': 95.0, 'weight': 2.0},
                {'metric': 'crisis_intervention', 'min_value': 98.0, 'weight': 3.0}
            ],
            required_score=95.0,
            is_blocking=True
        )
        self.add_quality_gate(safety_gate)
        
        # Performance quality gate
        performance_gate = QualityGate(
            gate_id="performance_quality_gate",
            name="Performance Quality Gate",
            description="Ensures performance standards",
            quality_criteria=[
                {'metric': 'response_time_quality', 'min_value': 70.0, 'weight': 1.0},
                {'metric': 'error_rate_quality', 'min_value': 85.0, 'weight': 2.0},
                {'metric': 'availability_quality', 'min_value': 95.0, 'weight': 1.0}
            ],
            required_score=80.0
        )
        self.add_quality_gate(performance_gate)
        
        logger.info("Default quality gates configured")
    
    def _setup_default_tests(self) -> None:
        """Setup default test cases."""
        
        # Response quality test
        def test_response_quality(context):
            response = context.get('response', '')
            return len(response) > 10 and 'error' not in response.lower()
        
        response_test = TestCase(
            test_id="response_quality_test",
            name="Response Quality Test",
            test_type=TestType.UNIT_TEST,
            description="Tests basic response quality",
            test_function=test_response_quality,
            expected_outcome=True
        )
        self.add_test_case(response_test)
        
        # Safety compliance test
        def test_safety_compliance(context):
            content = context.get('content', '')
            dangerous_phrases = ['ignore safety', 'bypass validation', 'skip checks']
            return not any(phrase in content.lower() for phrase in dangerous_phrases)
        
        safety_test = TestCase(
            test_id="safety_compliance_test",
            name="Safety Compliance Test",
            test_type=TestType.SAFETY_TEST,
            description="Tests safety compliance",
            test_function=test_safety_compliance,
            expected_outcome=True
        )
        self.add_test_case(safety_test)
        
        # Performance benchmark test
        async def test_performance_benchmark(context):
            start_time = time.time()
            # Simulate some processing
            await asyncio.sleep(0.1)
            processing_time = time.time() - start_time
            return processing_time < 1.0  # Should complete within 1 second
        
        performance_test = TestCase(
            test_id="performance_benchmark_test",
            name="Performance Benchmark Test",
            test_type=TestType.PERFORMANCE_TEST,
            description="Tests performance benchmarks",
            test_function=test_performance_benchmark,
            expected_outcome=True,
            timeout_seconds=5
        )
        self.add_test_case(performance_test)
        
        # Create default test suite
        self.add_test_suite("default", [
            "response_quality_test",
            "safety_compliance_test",
            "performance_benchmark_test"
        ])
        
        logger.info("Default test cases configured")
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for quality monitoring."""
        
        # Monitor agent responses for quality assessment
        self.event_bus.subscribe(
            EventType.AGENT_RESPONSE,
            self._handle_agent_response_quality,
            agent_id="quality_assurance"
        )
        
        # Monitor clinical assessments
        self.event_bus.subscribe(
            EventType.CLINICAL_ASSESSMENT,
            self._handle_clinical_assessment_quality,
            agent_id="quality_assurance"
        )
        
        logger.info("Quality assurance event subscriptions configured")
    
    async def _handle_agent_response_quality(self, event: Event) -> None:
        """Handle agent response quality assessment."""
        try:
            if not self.continuous_monitoring_enabled:
                return
            
            response_data = event.data
            content = response_data.get('response', '')
            
            if content and isinstance(content, str):
                # Perform quality assessment
                report = await self.assess_quality(
                    target=content,
                    assessment_type="content",
                    context={
                        'agent_id': event.source_agent,
                        'session_id': event.session_id,
                        'user_id': event.user_id
                    }
                )
                
                # Take action if quality is below threshold
                if report.overall_quality_score < self.quality_threshold:
                    await self._handle_quality_issue(report, event)
        
        except Exception as e:
            logger.error(f"Error in agent response quality assessment: {e}")
    
    async def _handle_clinical_assessment_quality(self, event: Event) -> None:
        """Handle clinical assessment quality monitoring."""
        try:
            if not self.continuous_monitoring_enabled:
                return
            
            assessment_data = event.data
            diagnosis_result = assessment_data.get('diagnosis_result', {})
            
            # Extract content for assessment
            content = str(diagnosis_result.get('reasoning', ''))
            if diagnosis_result.get('recommendations'):
                content += ' ' + ' '.join(diagnosis_result['recommendations'])
            
            if content:
                # Perform clinical safety assessment
                report = await self.assess_quality(
                    target=content,
                    assessment_type="clinical",
                    context={
                        'agent_id': event.source_agent,
                        'session_id': event.session_id,
                        'user_id': event.user_id,
                        'severity': diagnosis_result.get('severity', 'mild'),
                        'risk_level': diagnosis_result.get('risk_level', 'low')
                    }
                )
                
                # Clinical assessments have stricter quality requirements
                clinical_threshold = max(self.quality_threshold, 80.0)
                if report.overall_quality_score < clinical_threshold:
                    await self._handle_quality_issue(report, event, is_clinical=True)
        
        except Exception as e:
            logger.error(f"Error in clinical assessment quality monitoring: {e}")
    
    async def _handle_quality_issue(self, report: QualityReport, event: Event, is_clinical: bool = False) -> None:
        """Handle quality issues with automated remediation."""
        
        severity = EventPriority.CRITICAL if is_clinical else EventPriority.HIGH
        
        # Publish quality issue event
        await self.event_bus.publish(Event(
            event_type="quality_issue_detected",
            source_agent="quality_assurance",
            priority=severity,
            data={
                'report_id': report.report_id,
                'quality_score': report.overall_quality_score,
                'critical_issues': report.critical_issues,
                'original_event_id': event.event_id,
                'is_clinical': is_clinical
            }
        ))
        
        # Automated remediation if enabled
        if self.auto_remediation_enabled:
            await self._attempt_auto_remediation(report, event)
    
    async def _attempt_auto_remediation(self, report: QualityReport, event: Event) -> None:
        """Attempt automated remediation of quality issues."""
        try:
            remediation_actions = []
            
            # Generate remediation actions based on critical issues
            for issue in report.critical_issues:
                if 'content_clarity' in issue:
                    remediation_actions.append("Request content revision for clarity")
                elif 'professional_tone' in issue:
                    remediation_actions.append("Request tone adjustment")
                elif 'empathy_sensitivity' in issue:
                    remediation_actions.append("Request empathy enhancement")
                elif 'risk_assessment' in issue:
                    remediation_actions.append("Escalate to clinical supervisor")
                elif 'crisis_intervention' in issue:
                    remediation_actions.append("Immediate clinical review required")
            
            if remediation_actions:
                # Publish remediation event
                await self.event_bus.publish(Event(
                    event_type="auto_remediation_triggered",
                    source_agent="quality_assurance",
                    priority=EventPriority.HIGH,
                    data={
                        'report_id': report.report_id,
                        'remediation_actions': remediation_actions,
                        'original_event_id': event.event_id
                    }
                ))
                
                logger.info(f"Auto-remediation triggered for report {report.report_id}")
        
        except Exception as e:
            logger.error(f"Error in auto-remediation: {e}")
    
    async def _generate_recommendations(self, 
                                      metrics: List[QualityMetric], 
                                      test_results: List[TestResult]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze metrics for recommendations
        for metric in metrics:
            if metric.quality_level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]:
                if 'clarity' in metric.metric_name:
                    recommendations.append("Improve content structure and use clearer language")
                elif 'tone' in metric.metric_name:
                    recommendations.append("Adopt more professional and appropriate tone")
                elif 'empathy' in metric.metric_name:
                    recommendations.append("Include more empathetic and supportive language")
                elif 'risk' in metric.metric_name:
                    recommendations.append("Enhance risk assessment and safety protocols")
                elif 'response_time' in metric.metric_name:
                    recommendations.append("Optimize system performance to reduce response times")
                elif 'error_rate' in metric.metric_name:
                    recommendations.append("Investigate and fix underlying causes of errors")
        
        # Analyze test failures for recommendations
        failed_tests = [tr for tr in test_results if not tr.passed]
        for test_result in failed_tests:
            if 'safety' in test_result.test_id:
                recommendations.append("Review and strengthen safety compliance measures")
            elif 'performance' in test_result.test_id:
                recommendations.append("Investigate performance bottlenecks and optimize")
            elif 'quality' in test_result.test_id:
                recommendations.append("Review quality standards and implementation")
        
        # Remove duplicates
        return list(set(recommendations))
    
    def _score_to_quality_level(self, score: float) -> QualityLevel:
        """Convert numeric score to quality level."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for continuous quality assessment."""
        
        while self._running:
            try:
                # Run periodic quality tests
                test_results = await self.test_runner.run_test_suite("default")
                
                # Check test results for issues
                failed_tests = [tr for tr in test_results if not tr.passed]
                if failed_tests:
                    await self.event_bus.publish(Event(
                        event_type="quality_tests_failed",
                        source_agent="quality_assurance",
                        priority=EventPriority.HIGH,
                        data={
                            'failed_tests': len(failed_tests),
                            'total_tests': len(test_results),
                            'failure_details': [tr.to_dict() for tr in failed_tests[:5]]  # Limit details
                        }
                    ))
                
                # Wait before next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error


# Factory function
def create_quality_assurance_framework(event_bus: EventBus, supervision_mesh: SupervisionMesh) -> QualityAssuranceFramework:
    """Create a quality assurance framework instance."""
    return QualityAssuranceFramework(event_bus, supervision_mesh)
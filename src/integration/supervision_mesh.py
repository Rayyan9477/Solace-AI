"""
Distributed Supervision System for Solace-AI

This module provides a comprehensive supervision mesh that integrates with the existing
supervisor agent to provide distributed validation, quality gates, circuit breakers,
and multi-validator consensus mechanisms for ensuring clinical safety and quality.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

from src.agents.supervisor_agent import SupervisorAgent, ValidationLevel, ValidationResult
from src.integration.event_bus import EventBus, Event, EventType, get_event_bus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QualityGateType(Enum):
    """Types of quality gates in the supervision mesh."""
    CLINICAL_SAFETY = "clinical_safety"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    THERAPEUTIC_BOUNDARY = "therapeutic_boundary"
    RISK_ASSESSMENT = "risk_assessment"
    CONTENT_APPROPRIATENESS = "content_appropriateness"
    INTERVENTION_VALIDATION = "intervention_validation"
    CONSENSUS_REQUIRED = "consensus_required"


class ConsensusStrategy(Enum):
    """Strategies for multi-validator consensus."""
    UNANIMOUS = "unanimous"          # All validators must agree
    MAJORITY = "majority"            # More than 50% must agree
    QUORUM = "quorum"               # Minimum number must agree
    WEIGHTED_MAJORITY = "weighted"   # Weighted votes based on validator expertise
    EXPERT_OVERRIDE = "expert"       # Single expert can override


@dataclass
class ValidationRequest:
    """Request for validation through the supervision mesh."""
    
    request_id: str
    content: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    required_gates: Set[QualityGateType] = field(default_factory=set)
    priority: int = 5  # 1-10, higher is more urgent
    requesting_agent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    requires_consensus: bool = False
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY
    consensus_threshold: float = 0.51  # For majority/quorum strategies
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Check if validation request has expired."""
        return (datetime.now() - self.created_at).total_seconds() > self.timeout_seconds


@dataclass
class ValidationGateResult:
    """Result from a single validation gate."""
    
    gate_type: QualityGateType
    validator_id: str
    result: ValidationLevel
    confidence: float = 1.0
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsensusResult:
    """Result from multi-validator consensus."""
    
    request_id: str
    final_result: ValidationLevel
    confidence: float
    participating_validators: List[str]
    individual_results: List[ValidationGateResult]
    consensus_achieved: bool
    consensus_strategy: ConsensusStrategy
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'request_id': self.request_id,
            'final_result': self.final_result.value,
            'confidence': self.confidence,
            'participating_validators': self.participating_validators,
            'individual_results': [
                {
                    'gate_type': result.gate_type.value,
                    'validator_id': result.validator_id,
                    'result': result.result.value,
                    'confidence': result.confidence,
                    'message': result.message,
                    'details': result.details
                }
                for result in self.individual_results
            ],
            'consensus_achieved': self.consensus_achieved,
            'consensus_strategy': self.consensus_strategy.value,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat()
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states for validator resilience."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ValidatorCircuitBreaker:
    """Circuit breaker for individual validators."""
    
    validator_id: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def should_allow_request(self) -> bool:
        """Check if validator should handle requests."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        
        # HALF_OPEN state allows one request to test recovery
        return True
    
    def record_success(self) -> None:
        """Record successful validation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self) -> None:
        """Record failed validation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ValidationGate(ABC):
    """Abstract base class for validation gates."""
    
    def __init__(self, gate_type: QualityGateType, validator_id: str, weight: float = 1.0):
        self.gate_type = gate_type
        self.validator_id = validator_id
        self.weight = weight
        self.circuit_breaker = ValidatorCircuitBreaker(validator_id)
        self._metrics = {
            'validations_performed': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'average_processing_time': 0.0,
            'processing_times': []
        }
    
    @abstractmethod
    async def validate(self, request: ValidationRequest) -> ValidationGateResult:
        """
        Perform validation on the request.
        
        Args:
            request: Validation request
            
        Returns:
            Validation gate result
        """
        pass
    
    async def safe_validate(self, request: ValidationRequest) -> Optional[ValidationGateResult]:
        """
        Safely perform validation with circuit breaker protection.
        
        Args:
            request: Validation request
            
        Returns:
            Validation result or None if circuit breaker is open
        """
        if not self.circuit_breaker.should_allow_request():
            logger.warning(f"Circuit breaker open for validator {self.validator_id}")
            return None
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.validate(request),
                timeout=request.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.circuit_breaker.record_success()
            self._update_metrics(result, processing_time)
            
            return result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self._metrics['validations_failed'] += 1
            
            logger.error(f"Validation failed for {self.validator_id}: {e}")
            return ValidationGateResult(
                gate_type=self.gate_type,
                validator_id=self.validator_id,
                result=ValidationLevel.CRITICAL,
                confidence=0.0,
                message=f"Validation error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _update_metrics(self, result: ValidationGateResult, processing_time: float) -> None:
        """Update validation metrics."""
        self._metrics['validations_performed'] += 1
        
        if result.result in [ValidationLevel.PASS, ValidationLevel.WARNING]:
            self._metrics['validations_passed'] += 1
        else:
            self._metrics['validations_failed'] += 1
        
        self._metrics['processing_times'].append(processing_time)
        if len(self._metrics['processing_times']) > 100:
            self._metrics['processing_times'] = self._metrics['processing_times'][-100:]
        
        self._metrics['average_processing_time'] = sum(self._metrics['processing_times']) / len(self._metrics['processing_times'])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validator metrics."""
        return self._metrics.copy()


class SupervisorValidationGate(ValidationGate):
    """Validation gate that integrates with the existing SupervisorAgent."""
    
    def __init__(self, supervisor_agent: SupervisorAgent, gate_type: QualityGateType = QualityGateType.CLINICAL_SAFETY):
        super().__init__(gate_type, f"supervisor_{gate_type.value}")
        self.supervisor_agent = supervisor_agent
    
    async def validate(self, request: ValidationRequest) -> ValidationGateResult:
        """
        Validate using the SupervisorAgent.
        
        Args:
            request: Validation request
            
        Returns:
            Validation gate result
        """
        # Map request to supervisor validation format
        content = request.content.get('message', '')
        if isinstance(content, dict):
            content = json.dumps(content)
        
        # Perform validation based on gate type
        if self.gate_type == QualityGateType.CLINICAL_SAFETY:
            result = await self.supervisor_agent.validate_clinical_safety(
                content,
                request.context.get('user_profile', {}),
                request.context.get('session_context', {})
            )
        elif self.gate_type == QualityGateType.ETHICAL_COMPLIANCE:
            result = await self.supervisor_agent.validate_ethical_compliance(
                content,
                request.context.get('user_profile', {}),
                request.context.get('session_context', {})
            )
        elif self.gate_type == QualityGateType.THERAPEUTIC_BOUNDARY:
            result = await self.supervisor_agent.validate_therapeutic_boundaries(
                content,
                request.context.get('user_profile', {}),
                request.context.get('session_context', {})
            )
        else:
            # Default comprehensive validation
            result = await self.supervisor_agent.comprehensive_validation(
                content,
                request.context.get('agent_response', {}),
                request.context.get('user_profile', {}),
                request.context.get('session_context', {})
            )
        
        return ValidationGateResult(
            gate_type=self.gate_type,
            validator_id=self.validator_id,
            result=result.level,
            confidence=result.confidence,
            message=result.message,
            details={
                'recommendations': result.recommendations,
                'risk_factors': result.risk_factors,
                'metadata': result.metadata
            }
        )


class ConsensusValidator:
    """Handles multi-validator consensus for critical decisions."""
    
    def __init__(self, validators: List[ValidationGate]):
        self.validators = {v.validator_id: v for v in validators}
        self.pending_requests: Dict[str, ValidationRequest] = {}
        self.consensus_cache: Dict[str, ConsensusResult] = {}
    
    async def validate_with_consensus(
        self,
        request: ValidationRequest
    ) -> ConsensusResult:
        """
        Validate request using multiple validators and consensus.
        
        Args:
            request: Validation request
            
        Returns:
            Consensus validation result
        """
        start_time = time.time()
        self.pending_requests[request.request_id] = request
        
        try:
            # Determine which validators to use
            validators_to_use = self._select_validators(request)
            
            # Run validations in parallel
            validation_tasks = []
            for validator in validators_to_use:
                if request.required_gates and validator.gate_type not in request.required_gates:
                    continue
                task = asyncio.create_task(validator.safe_validate(request))
                validation_tasks.append(task)
            
            # Wait for all validations to complete
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            valid_results = []
            for result in results:
                if isinstance(result, ValidationGateResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Validation exception: {result}")
            
            # Apply consensus strategy
            consensus_result = self._apply_consensus(request, valid_results, time.time() - start_time)
            
            # Cache result
            self.consensus_cache[request.request_id] = consensus_result
            
            return consensus_result
            
        finally:
            # Clean up
            self.pending_requests.pop(request.request_id, None)
    
    def _select_validators(self, request: ValidationRequest) -> List[ValidationGate]:
        """Select appropriate validators for the request."""
        validators = []
        
        # Always include relevant gate types
        for validator in self.validators.values():
            if request.required_gates and validator.gate_type in request.required_gates:
                validators.append(validator)
            elif not request.required_gates:
                # Use all available validators if no specific gates required
                validators.append(validator)
        
        # Sort by weight (higher weight first)
        validators.sort(key=lambda v: v.weight, reverse=True)
        
        return validators
    
    def _apply_consensus(
        self,
        request: ValidationRequest,
        results: List[ValidationGateResult],
        processing_time: float
    ) -> ConsensusResult:
        """Apply consensus strategy to validation results."""
        if not results:
            return ConsensusResult(
                request_id=request.request_id,
                final_result=ValidationLevel.CRITICAL,
                confidence=0.0,
                participating_validators=[],
                individual_results=[],
                consensus_achieved=False,
                consensus_strategy=request.consensus_strategy,
                processing_time=processing_time
            )
        
        strategy = request.consensus_strategy
        
        if strategy == ConsensusStrategy.UNANIMOUS:
            consensus_result, confidence = self._unanimous_consensus(results)
        elif strategy == ConsensusStrategy.MAJORITY:
            consensus_result, confidence = self._majority_consensus(results, request.consensus_threshold)
        elif strategy == ConsensusStrategy.QUORUM:
            consensus_result, confidence = self._quorum_consensus(results, request.consensus_threshold)
        elif strategy == ConsensusStrategy.WEIGHTED_MAJORITY:
            consensus_result, confidence = self._weighted_consensus(results)
        else:
            # Default to majority
            consensus_result, confidence = self._majority_consensus(results, 0.51)
        
        consensus_achieved = confidence >= request.consensus_threshold
        
        return ConsensusResult(
            request_id=request.request_id,
            final_result=consensus_result,
            confidence=confidence,
            participating_validators=[r.validator_id for r in results],
            individual_results=results,
            consensus_achieved=consensus_achieved,
            consensus_strategy=strategy,
            processing_time=processing_time
        )
    
    def _unanimous_consensus(self, results: List[ValidationGateResult]) -> Tuple[ValidationLevel, float]:
        """Apply unanimous consensus strategy."""
        if not results:
            return ValidationLevel.CRITICAL, 0.0
        
        # All results must be the same
        first_result = results[0].result
        if all(r.result == first_result for r in results):
            avg_confidence = sum(r.confidence for r in results) / len(results)
            return first_result, avg_confidence
        else:
            # No consensus - use most restrictive
            most_restrictive = max(results, key=lambda r: self._get_restriction_level(r.result))
            return most_restrictive.result, 0.0
    
    def _majority_consensus(self, results: List[ValidationGateResult], threshold: float) -> Tuple[ValidationLevel, float]:
        """Apply majority consensus strategy."""
        if not results:
            return ValidationLevel.CRITICAL, 0.0
        
        # Count votes for each validation level
        vote_counts = defaultdict(int)
        confidence_sums = defaultdict(float)
        
        for result in results:
            vote_counts[result.result] += 1
            confidence_sums[result.result] += result.confidence
        
        # Find majority
        total_votes = len(results)
        required_votes = int(total_votes * threshold)
        
        for level, count in vote_counts.items():
            if count >= required_votes:
                avg_confidence = confidence_sums[level] / count
                return level, avg_confidence
        
        # No majority - use most restrictive
        most_restrictive = max(results, key=lambda r: self._get_restriction_level(r.result))
        return most_restrictive.result, 0.0
    
    def _quorum_consensus(self, results: List[ValidationGateResult], min_validators: float) -> Tuple[ValidationLevel, float]:
        """Apply quorum consensus strategy."""
        required_count = int(min_validators) if min_validators >= 1 else int(len(results) * min_validators)
        
        if len(results) < required_count:
            return ValidationLevel.CRITICAL, 0.0
        
        # Use majority of available results
        return self._majority_consensus(results, 0.51)
    
    def _weighted_consensus(self, results: List[ValidationGateResult]) -> Tuple[ValidationLevel, float]:
        """Apply weighted consensus strategy."""
        if not results:
            return ValidationLevel.CRITICAL, 0.0
        
        # Get validator weights
        weighted_votes = defaultdict(float)
        total_weight = 0
        
        for result in results:
            validator = self.validators.get(result.validator_id)
            weight = validator.weight if validator else 1.0
            weighted_votes[result.result] += weight * result.confidence
            total_weight += weight
        
        # Find highest weighted vote
        if total_weight == 0:
            return ValidationLevel.CRITICAL, 0.0
        
        best_level = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        confidence = weighted_votes[best_level] / total_weight
        
        return best_level, confidence
    
    def _get_restriction_level(self, level: ValidationLevel) -> int:
        """Get numeric restriction level for comparison."""
        return {
            ValidationLevel.PASS: 1,
            ValidationLevel.WARNING: 2,
            ValidationLevel.CRITICAL: 3,
            ValidationLevel.BLOCKED: 4
        }.get(level, 4)


class SupervisionMesh:
    """
    Distributed supervision system that coordinates multiple validators
    and provides quality gates, consensus mechanisms, and circuit breakers.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or get_event_bus()
        self.validators: Dict[str, ValidationGate] = {}
        self.consensus_validator = ConsensusValidator([])
        self.quality_gates: Dict[QualityGateType, List[ValidationGate]] = defaultdict(list)
        self.metrics = {
            'validations_requested': 0,
            'validations_completed': 0,
            'consensus_requests': 0,
            'circuit_breaker_activations': 0
        }
        
        # Subscribe to validation events
        self._setup_event_subscriptions()
        
        logger.info("SupervisionMesh initialized")
    
    def add_validator(self, validator: ValidationGate) -> None:
        """Add a validator to the mesh."""
        self.validators[validator.validator_id] = validator
        self.quality_gates[validator.gate_type].append(validator)
        
        # Update consensus validator
        self.consensus_validator = ConsensusValidator(list(self.validators.values()))
        
        logger.info(f"Added validator {validator.validator_id} for gate {validator.gate_type}")
    
    def remove_validator(self, validator_id: str) -> None:
        """Remove a validator from the mesh."""
        if validator_id in self.validators:
            validator = self.validators[validator_id]
            del self.validators[validator_id]
            self.quality_gates[validator.gate_type].remove(validator)
            
            # Update consensus validator
            self.consensus_validator = ConsensusValidator(list(self.validators.values()))
            
            logger.info(f"Removed validator {validator_id}")
    
    async def validate(
        self,
        content: Dict[str, Any],
        context: Dict[str, Any] = None,
        required_gates: Set[QualityGateType] = None,
        requires_consensus: bool = False,
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.MAJORITY,
        priority: int = 5,
        requesting_agent: str = None
    ) -> Union[ValidationGateResult, ConsensusResult]:
        """
        Validate content through the supervision mesh.
        
        Args:
            content: Content to validate
            context: Additional context for validation
            required_gates: Specific quality gates to use
            requires_consensus: Whether consensus is required
            consensus_strategy: Strategy for consensus
            priority: Validation priority (1-10)
            requesting_agent: Agent requesting validation
            
        Returns:
            Validation or consensus result
        """
        request = ValidationRequest(
            request_id=f"val_{int(time.time() * 1000000)}",
            content=content,
            context=context or {},
            required_gates=required_gates or set(),
            priority=priority,
            requesting_agent=requesting_agent,
            requires_consensus=requires_consensus,
            consensus_strategy=consensus_strategy
        )
        
        self.metrics['validations_requested'] += 1
        
        try:
            if requires_consensus or len(self._get_applicable_validators(request)) > 1:
                # Use consensus validation
                self.metrics['consensus_requests'] += 1
                result = await self.consensus_validator.validate_with_consensus(request)
                
                # Publish consensus result event
                await self.event_bus.publish(Event(
                    event_type=EventType.VALIDATION_RESULT,
                    source_agent="supervision_mesh",
                    data={
                        'request_id': request.request_id,
                        'result': result.to_dict(),
                        'consensus': True
                    }
                ))
                
                return result
            else:
                # Use single validator
                validators = self._get_applicable_validators(request)
                if validators:
                    result = await validators[0].safe_validate(request)
                    
                    if result:
                        # Publish validation result event
                        await self.event_bus.publish(Event(
                            event_type=EventType.VALIDATION_RESULT,
                            source_agent="supervision_mesh",
                            data={
                                'request_id': request.request_id,
                                'result': asdict(result),
                                'consensus': False
                            }
                        ))
                        
                        self.metrics['validations_completed'] += 1
                        return result
                
                # No validators available or all failed
                return ValidationGateResult(
                    gate_type=QualityGateType.CLINICAL_SAFETY,
                    validator_id="mesh_fallback",
                    result=ValidationLevel.CRITICAL,
                    confidence=0.0,
                    message="No validators available"
                )
        
        except Exception as e:
            logger.error(f"Validation error for request {request.request_id}: {e}")
            return ValidationGateResult(
                gate_type=QualityGateType.CLINICAL_SAFETY,
                validator_id="mesh_error",
                result=ValidationLevel.CRITICAL,
                confidence=0.0,
                message=f"Validation error: {str(e)}"
            )
    
    def _get_applicable_validators(self, request: ValidationRequest) -> List[ValidationGate]:
        """Get validators applicable to the request."""
        validators = []
        
        if request.required_gates:
            for gate_type in request.required_gates:
                validators.extend(self.quality_gates[gate_type])
        else:
            # Use all validators if no specific gates required
            validators = list(self.validators.values())
        
        # Filter by circuit breaker status
        available_validators = [
            v for v in validators
            if v.circuit_breaker.should_allow_request()
        ]
        
        return available_validators
    
    def _setup_event_subscriptions(self) -> None:
        """Set up event subscriptions for validation requests."""
        self.event_bus.subscribe(
            EventType.VALIDATION_REQUEST,
            self._handle_validation_request,
            agent_id="supervision_mesh"
        )
    
    async def _handle_validation_request(self, event: Event) -> None:
        """Handle incoming validation request events."""
        try:
            request_data = event.data
            result = await self.validate(
                content=request_data.get('content', {}),
                context=request_data.get('context', {}),
                required_gates=set(request_data.get('required_gates', [])),
                requires_consensus=request_data.get('requires_consensus', False),
                requesting_agent=event.source_agent
            )
            
            # Send result back if reply_to is specified
            if event.reply_to:
                await self.event_bus.publish(Event(
                    event_type=EventType.VALIDATION_RESULT,
                    source_agent="supervision_mesh",
                    target_agent=event.reply_to,
                    correlation_id=event.correlation_id,
                    data={
                        'request_id': event.event_id,
                        'result': result.to_dict() if hasattr(result, 'to_dict') else asdict(result)
                    }
                ))
        
        except Exception as e:
            logger.error(f"Error handling validation request: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get supervision mesh metrics."""
        validator_metrics = {
            validator_id: validator.get_metrics()
            for validator_id, validator in self.validators.items()
        }
        
        circuit_breaker_states = {
            validator_id: validator.circuit_breaker.state.value
            for validator_id, validator in self.validators.items()
        }
        
        return {
            'mesh_metrics': self.metrics,
            'validator_metrics': validator_metrics,
            'circuit_breaker_states': circuit_breaker_states,
            'active_validators': len(self.validators),
            'quality_gates': {
                gate_type.value: len(validators)
                for gate_type, validators in self.quality_gates.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the supervision mesh."""
        total_validators = len(self.validators)
        healthy_validators = sum(
            1 for v in self.validators.values()
            if v.circuit_breaker.state == CircuitBreakerState.CLOSED
        )
        
        health_percentage = (healthy_validators / total_validators * 100) if total_validators > 0 else 0
        
        return {
            'status': 'healthy' if health_percentage >= 80 else 'degraded' if health_percentage >= 50 else 'unhealthy',
            'healthy_validators': healthy_validators,
            'total_validators': total_validators,
            'health_percentage': health_percentage,
            'circuit_breakers_open': sum(
                1 for v in self.validators.values()
                if v.circuit_breaker.state == CircuitBreakerState.OPEN
            )
        }


# Factory functions for common validator configurations

def create_clinical_supervision_mesh(supervisor_agent: SupervisorAgent, event_bus: Optional[EventBus] = None) -> SupervisionMesh:
    """Create a supervision mesh with clinical validators."""
    mesh = SupervisionMesh(event_bus)
    
    # Add clinical safety gates
    mesh.add_validator(SupervisorValidationGate(
        supervisor_agent,
        QualityGateType.CLINICAL_SAFETY
    ))
    
    mesh.add_validator(SupervisorValidationGate(
        supervisor_agent,
        QualityGateType.ETHICAL_COMPLIANCE
    ))
    
    mesh.add_validator(SupervisorValidationGate(
        supervisor_agent,
        QualityGateType.THERAPEUTIC_BOUNDARY
    ))
    
    mesh.add_validator(SupervisorValidationGate(
        supervisor_agent,
        QualityGateType.RISK_ASSESSMENT
    ))
    
    return mesh
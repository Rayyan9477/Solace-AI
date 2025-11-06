"""
Agent Orchestrator Module for coordinating multiple specialized agents.

This module manages agent interactions, message passing, and workflow coordination
using the module system with integrated SupervisorAgent oversight.
"""

import asyncio
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Set
import time
import json
import uuid
import threading
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import weakref

from src.components.base_module import Module, get_module_manager
from src.utils.logger import get_logger
from src.utils.vector_db_integration import get_conversation_tracker, search_relevant_data
from src.agents.orchestration.supervisor_agent import SupervisorAgent, ValidationLevel
from src.monitoring.supervisor_metrics import MetricsCollector, QualityMetrics
from src.auditing.audit_system import AuditTrail, AuditEventType, AuditSeverity

# Import security exceptions
try:
    from src.core.exceptions import CircuitBreakerOpen
    SECURITY_EXCEPTIONS_AVAILABLE = True
except ImportError:
    SECURITY_EXCEPTIONS_AVAILABLE = False

# Import diagnosis services
try:
    from src.services.diagnosis import (
        IDiagnosisOrchestrator, IDiagnosisAgentAdapter,
        DiagnosisRequest, DiagnosisType
    )
    from src.infrastructure.di.container import get_container
    DIAGNOSIS_SERVICES_AVAILABLE = True
except ImportError:
    DIAGNOSIS_SERVICES_AVAILABLE = False

# Enhanced orchestration components
class MessageType(Enum):
    """Types of messages in the event-driven system."""
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    VALIDATION_REQUEST = "validation_request"
    VALIDATION_RESPONSE = "validation_response"
    ERROR_EVENT = "error_event"
    METRICS_EVENT = "metrics_event"
    CONTEXT_UPDATE = "context_update"
    HEALTH_CHECK = "health_check"
    CIRCUIT_BREAKER_EVENT = "circuit_breaker_event"

class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class Message:
    """Event message structure for agent communication."""
    type: MessageType
    sender: str
    recipient: Optional[str]
    payload: Dict[str, Any]
    # Optional fields follow to satisfy dataclass init rules
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()

@dataclass
class ValidationRequest:
    """Request for agent response validation."""
    agent_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    session_id: str
    validation_types: List[str] = None
    priority: str = "normal"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    reset_timeout: int = 60  # seconds
    success_threshold: int = 2  # for half-open state
    timeout: int = 30  # request timeout

class MessageBus:
    """Event-driven message bus for agent communication."""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.lock = threading.RLock()
        self.running = False
        self.logger = get_logger(__name__ + ".MessageBus")
        
    async def start(self):
        """Start the message bus processing."""
        self.running = True
        asyncio.create_task(self._process_messages())
        self.logger.info("Message bus started")
    
    async def stop(self):
        """Stop the message bus."""
        self.running = False
        self.logger.info("Message bus stopped")
    
    def subscribe(self, message_type: MessageType, handler: Callable):
        """Subscribe to specific message types."""
        with self.lock:
            self.subscribers[message_type].append(handler)
        self.logger.debug(f"Handler subscribed to {message_type.value}")
    
    def unsubscribe(self, message_type: MessageType, handler: Callable):
        """Unsubscribe from message types."""
        with self.lock:
            if handler in self.subscribers[message_type]:
                self.subscribers[message_type].remove(handler)
        self.logger.debug(f"Handler unsubscribed from {message_type.value}")
    
    async def publish(self, message: Message):
        """Publish a message to the bus."""
        if not self.running:
            self.logger.warning("Message bus not running, message dropped")
            return
        
        try:
            await self.message_queue.put(message)
        except asyncio.QueueFull:
            self.logger.error("Message queue full, message dropped")
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._route_message(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
    
    async def _route_message(self, message: Message):
        """Route message to appropriate handlers."""
        handlers = []
        with self.lock:
            handlers = self.subscribers[message.type].copy()
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.logger.error(f"Handler error for {message.type.value}: {str(e)}")

class ValidatorRegistry:
    """Registry for managing multiple validator agents in supervision mesh."""
    
    def __init__(self):
        self.validators = {}
        self.validator_configs = {}
        self.lock = threading.RLock()
        self.logger = get_logger(__name__ + ".ValidatorRegistry")
        
    def register_validator(self, validator_id: str, validator: Any, 
                          validator_types: List[str] = None, weight: float = 1.0):
        """Register a validator agent."""
        with self.lock:
            self.validators[validator_id] = validator
            self.validator_configs[validator_id] = {
                "types": validator_types or ["general"],
                "weight": weight,
                "enabled": True,
                "success_count": 0,
                "failure_count": 0
            }
        self.logger.info(f"Registered validator: {validator_id}")
    
    def get_validators_for_type(self, validation_type: str = "general") -> List[tuple]:
        """Get appropriate validators for validation type."""
        with self.lock:
            valid_validators = []
            for validator_id, config in self.validator_configs.items():
                if (config["enabled"] and 
                    (validation_type in config["types"] or "general" in config["types"])):
                    validator = self.validators[validator_id]
                    valid_validators.append((validator_id, validator, config["weight"]))
            return valid_validators
    
    async def validate_with_consensus(self, request: ValidationRequest) -> Dict[str, Any]:
        """Perform validation using multiple validators with consensus."""
        validators = self.get_validators_for_type("general")
        
        if not validators:
            self.logger.warning("No validators available")
            return {"error": "No validators available"}
        
        validation_results = []
        
        for validator_id, validator, weight in validators:
            try:
                if hasattr(validator, 'validate_agent_response'):
                    result = await validator.validate_agent_response(
                        request.agent_name, request.input_data, 
                        request.output_data, request.session_id
                    )
                    validation_results.append((validator_id, result, weight))
                    
                    # Update success count
                    with self.lock:
                        self.validator_configs[validator_id]["success_count"] += 1
                        
            except Exception as e:
                self.logger.error(f"Validator {validator_id} failed: {str(e)}")
                with self.lock:
                    self.validator_configs[validator_id]["failure_count"] += 1
        
        return self._calculate_consensus(validation_results)
    
    def _calculate_consensus(self, results: List[tuple]) -> Dict[str, Any]:
        """Calculate consensus from multiple validation results."""
        if not results:
            return {"error": "No validation results"}
        
        # Simple weighted voting for now
        total_weight = sum(weight for _, _, weight in results)
        weighted_scores = []
        all_issues = set()
        blocking_votes = 0
        
        for validator_id, result, weight in results:
            if hasattr(result, 'accuracy_score'):
                weighted_scores.append(result.accuracy_score * weight)
            
            if hasattr(result, 'critical_issues'):
                all_issues.update(result.critical_issues)
            
            if hasattr(result, 'validation_level') and result.validation_level == ValidationLevel.BLOCKED:
                blocking_votes += weight
        
        consensus_score = sum(weighted_scores) / total_weight if weighted_scores else 0.5
        is_blocked = blocking_votes > total_weight * 0.5  # More than 50% vote to block
        
        return {
            "consensus_score": consensus_score,
            "is_blocked": is_blocked,
            "validator_count": len(results),
            "critical_issues": list(all_issues),
            "individual_results": [(vid, asdict(result) if hasattr(result, '__dict__') else str(result)) 
                                  for vid, result, _ in results]
        }

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, agent_id: str, config: CircuitBreakerConfig):
        self.agent_id = agent_id
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
        self.logger = get_logger(__name__ + f".CircuitBreaker.{agent_id}")
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info(f"Circuit breaker {self.agent_id} moving to HALF_OPEN")
                else:
                    # Use CircuitBreakerOpen exception if available
                    if SECURITY_EXCEPTIONS_AVAILABLE:
                        raise CircuitBreakerOpen(
                            f"Circuit breaker is open for {self.agent_id}",
                            service_name=self.agent_id,
                            failure_count=self.failure_count
                        )
                    else:
                        raise Exception(f"Circuit breaker {self.agent_id} is OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), 
                                          timeout=self.config.timeout)
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time > self.config.reset_timeout)
    
    async def _on_success(self):
        """Handle successful execution."""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.agent_id} CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self):
        """Handle failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                self.logger.error(f"Circuit breaker {self.agent_id} OPENED")
                
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                self.logger.error(f"Circuit breaker {self.agent_id} back to OPEN")

class ContextManager:
    """Enhanced context management with propagation and versioning."""
    
    def __init__(self):
        self.contexts = {}
        self.context_versions = defaultdict(int)
        self.context_locks = defaultdict(threading.RLock)
        self.context_cache = {}
        self.inheritance_map = defaultdict(set)
        self.logger = get_logger(__name__ + ".ContextManager")
        
    async def get_context(self, session_id: str, include_inherited: bool = True) -> Dict[str, Any]:
        """Get context with optional inheritance."""
        with self.context_locks[session_id]:
            context = self.contexts.get(session_id, {}).copy()
            
            if include_inherited:
                # Add inherited contexts
                for parent_session in self.inheritance_map[session_id]:
                    parent_context = self.contexts.get(parent_session, {})
                    # Merge parent context (child values take precedence)
                    for key, value in parent_context.items():
                        if key not in context:
                            context[key] = value
            
            return context
    
    async def update_context(self, session_id: str, updates: Dict[str, Any], 
                           merge: bool = True, increment_version: bool = True):
        """Update context with versioning."""
        with self.context_locks[session_id]:
            if session_id not in self.contexts:
                self.contexts[session_id] = {}
            
            if merge:
                self._deep_merge(self.contexts[session_id], updates)
            else:
                self.contexts[session_id] = updates
            
            if increment_version:
                self.context_versions[session_id] += 1
            
            # Clear cache for this session
            self.context_cache.pop(session_id, None)
            
            self.logger.debug(f"Context updated for {session_id}, version {self.context_versions[session_id]}")
    
    def add_inheritance(self, child_session: str, parent_session: str):
        """Add context inheritance relationship."""
        self.inheritance_map[child_session].add(parent_session)
        self.logger.debug(f"Added inheritance: {child_session} inherits from {parent_session}")
    
    def get_context_version(self, session_id: str) -> int:
        """Get current context version."""
        return self.context_versions[session_id]
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

class PerformanceMonitor:
    """Real-time performance monitoring for agents."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.agent_stats = defaultdict(lambda: {
            "requests": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "last_request_time": None,
            "health_status": "unknown"
        })
        self.performance_alerts = deque(maxlen=100)
        self.lock = threading.RLock()
        self.logger = get_logger(__name__ + ".PerformanceMonitor")
    
    async def record_agent_performance(self, agent_name: str, response_time: float, 
                                     success: bool, session_id: str = None):
        """Record agent performance metrics."""
        with self.lock:
            stats = self.agent_stats[agent_name]
            stats["requests"] += 1
            stats["last_request_time"] = datetime.now()
            
            if not success:
                stats["errors"] += 1
            
            # Update average response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            if stats["avg_response_time"] == 0:
                stats["avg_response_time"] = response_time
            else:
                stats["avg_response_time"] = (alpha * response_time + 
                                            (1 - alpha) * stats["avg_response_time"])
            
            # Update health status
            error_rate = stats["errors"] / stats["requests"]
            if error_rate > 0.1:  # More than 10% errors
                stats["health_status"] = "unhealthy"
            elif error_rate > 0.05:  # More than 5% errors
                stats["health_status"] = "degraded"
            else:
                stats["health_status"] = "healthy"
        
        # Record in metrics collector
        self.metrics_collector.record_metric(
            f"agent_response_time_{agent_name}", response_time,
            metadata={"agent": agent_name, "session_id": session_id, "success": success}
        )
        
        # Check for performance alerts
        await self._check_performance_alerts(agent_name, stats)
    
    async def _check_performance_alerts(self, agent_name: str, stats: Dict[str, Any]):
        """Check and generate performance alerts."""
        alerts = []
        
        # High error rate alert
        error_rate = stats["errors"] / stats["requests"]
        if error_rate > 0.15:
            alerts.append(f"High error rate for {agent_name}: {error_rate:.2%}")
        
        # Slow response time alert
        if stats["avg_response_time"] > 5.0:
            alerts.append(f"Slow response time for {agent_name}: {stats['avg_response_time']:.2f}s")
        
        # Add alerts to queue
        for alert in alerts:
            self.performance_alerts.append({
                "timestamp": datetime.now(),
                "agent": agent_name,
                "message": alert,
                "stats": stats.copy()
            })
            self.logger.warning(alert)
    
    def get_agent_health(self, agent_name: str = None) -> Dict[str, Any]:
        """Get agent health status."""
        with self.lock:
            if agent_name:
                return self.agent_stats.get(agent_name, {"health_status": "unknown"})
            return {name: stats for name, stats in self.agent_stats.items()}
    
    def get_performance_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        return list(self.performance_alerts)[-limit:]

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = get_logger(__name__ + ".RetryManager")
    
    async def execute_with_retry(self, func: Callable, *args, 
                               retry_exceptions: tuple = (Exception,), **kwargs):
        """Execute function with exponential backoff retry."""
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except retry_exceptions as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Max retries exceeded for {func.__name__}: {str(e)}")
                    raise
                
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {func.__name__} after {delay}s: {str(e)}")
                await asyncio.sleep(delay)
        
        raise Exception("Should not reach here")

class AgentOrchestrator(Module):
    """
    Orchestrates interactions between multiple specialized agents.
    
    This class manages agent dependencies, message passing, and coordinated 
    workflows across different agent types.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the agent orchestrator with supervision capabilities"""
        super().__init__(module_id, config)
        self.agent_modules = {}
        self.workflows = {}
        self.current_workflows = {}
        self.workflow_history = {}
        
        # Configuration flags for enhanced features
        self.enhanced_features_enabled = config.get("enhanced_features_enabled", True) if config else True
        self.event_driven_enabled = config.get("event_driven_enabled", True) if config else True
        self.supervision_mesh_enabled = config.get("supervision_mesh_enabled", True) if config else True
        self.circuit_breaker_enabled = config.get("circuit_breaker_enabled", True) if config else True
        self.performance_monitoring_enabled = config.get("performance_monitoring_enabled", True) if config else True
        
        # Initialize context management (enhanced)
        if self.enhanced_features_enabled:
            self.context_manager = ContextManager()
        else:
            self.context_store = {}  # Backward compatibility
        
        # Store user_id for conversation tracking
        self.user_id = config.get("user_id", "default_user") if config else "default_user"
        self.conversation_tracker = None  # Will be initialized when needed
        
        # Initialize diagnosis system integration
        self.diagnosis_orchestrator = None
        self.diagnosis_adapter = None
        self.diagnosis_integration_enabled = (
            config.get("diagnosis_integration_enabled", True) if config else True
        ) and DIAGNOSIS_SERVICES_AVAILABLE
        
        # Initialize supervision system
        self.supervision_enabled = config.get("supervision_enabled", True) if config else True
        self.supervisor_agent = None
        self.metrics_collector = None
        self.audit_trail = None
        
        # Enhanced components
        self.message_bus = None
        self.validator_registry = None
        self.circuit_breakers = {}
        self.performance_monitor = None
        self.retry_manager = None
        
        if self.supervision_enabled:
            try:
                # Initialize SupervisorAgent
                model_provider = config.get("model_provider") if config else None
                self.supervisor_agent = SupervisorAgent(model_provider, config)
                
                # Initialize metrics collection
                self.metrics_collector = MetricsCollector()
                
                # Initialize audit trail
                self.audit_trail = AuditTrail()
                
                self.logger.info("Supervision system initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize supervision system: {str(e)}")
                self.supervision_enabled = False
        
        # Initialize enhanced features
        if self.enhanced_features_enabled:
            self._initialize_enhanced_features()
        
        self.logger.info(f"Agent orchestrator configured for user {self.user_id} with supervision {'enabled' if self.supervision_enabled else 'disabled'} and enhanced features {'enabled' if self.enhanced_features_enabled else 'disabled'}")
        
    def _initialize_enhanced_features(self):
        """Initialize enhanced orchestration features."""
        try:
            # Initialize message bus for event-driven communication
            if self.event_driven_enabled:
                self.message_bus = MessageBus()
                self.logger.info("Message bus initialized")
            
            # Initialize validator registry for supervision mesh
            if self.supervision_mesh_enabled:
                self.validator_registry = ValidatorRegistry()
                # Register the main supervisor as a validator
                if self.supervisor_agent:
                    self.validator_registry.register_validator(
                        "main_supervisor", self.supervisor_agent, 
                        ["safety", "ethics", "clinical"], weight=1.0
                    )
                self.logger.info("Validator registry initialized")
            
            # Initialize performance monitoring
            if self.performance_monitoring_enabled and self.metrics_collector:
                self.performance_monitor = PerformanceMonitor(self.metrics_collector)
                self.logger.info("Performance monitor initialized")
            
            # Initialize retry manager
            config = self.config or {}
            retry_config = config.get("retry_config", {})
            self.retry_manager = RetryManager(
                max_retries=retry_config.get("max_retries", 3),
                base_delay=retry_config.get("base_delay", 1.0),
                max_delay=retry_config.get("max_delay", 60.0)
            )
            self.logger.info("Retry manager initialized")
            
            self.logger.info("Enhanced features initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced features: {str(e)}")
            self.enhanced_features_enabled = False

    async def initialize(self) -> bool:
        """Initialize the orchestrator and register available agents"""
        await super().initialize()
        
        self.logger.info("Initializing Agent Orchestrator")
        
        # Start enhanced components
        if self.enhanced_features_enabled and self.message_bus:
            await self.message_bus.start()
            # Subscribe to events
            self.message_bus.subscribe(MessageType.VALIDATION_REQUEST, self._handle_validation_request)
            self.message_bus.subscribe(MessageType.ERROR_EVENT, self._handle_error_event)
            self.message_bus.subscribe(MessageType.METRICS_EVENT, self._handle_metrics_event)
        
        # Register workflow patterns
        self._register_workflows()
        
        # Initialize diagnosis system integration
        await self._initialize_diagnosis_integration()
        
        # Expose services (existing)
        self.expose_service("execute_workflow", self.execute_workflow)
        self.expose_service("register_agent", self.register_agent)
        self.expose_service("send_message", self.send_message)
        self.expose_service("get_context", self.get_context)
        self.expose_service("update_context", self.update_context)
        
        # Expose enhanced services
        if self.enhanced_features_enabled:
            self.expose_service("execute_workflow_enhanced", self.execute_workflow_enhanced)
            self.expose_service("register_validator", self.register_validator)
            self.expose_service("send_event", self.send_event)
            self.expose_service("get_agent_health", self.get_agent_health)
            self.expose_service("get_performance_metrics", self.get_performance_metrics)
        
        # Expose diagnosis service if available
        if self.diagnosis_integration_enabled:
            self.expose_service("diagnose", self.diagnose)
        
        return True
    
    async def _initialize_diagnosis_integration(self) -> None:
        """Initialize integration with the unified diagnosis system."""
        if not self.diagnosis_integration_enabled:
            self.logger.info("Diagnosis integration disabled or not available")
            return
        
        try:
            # Get DI container
            container = get_container()
            
            # Resolve diagnosis services
            self.diagnosis_orchestrator = await container.resolve(IDiagnosisOrchestrator)
            self.diagnosis_adapter = await container.resolve(IDiagnosisAgentAdapter)
            
            self.logger.info("Diagnosis system integration initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize diagnosis integration: {str(e)}")
            self.diagnosis_integration_enabled = False
    
    def _register_workflows(self) -> None:
        """Register standard workflow patterns"""
        # Register the basic chat workflow
        self.register_workflow(
            "basic_chat",
            ["safety_agent", "chat_agent"],
            {
                "description": "Basic chat workflow with safety checks",
                "default_timeout": 30
            }
        )
        
        # Register the diagnosis workflow
        self.register_workflow(
            "diagnosis",
            ["safety_agent", "emotion_agent", "diagnosis_agent"],
            {
                "description": "Psychological diagnosis workflow",
                "default_timeout": 60
            }
        )
        
        # Register the integrated diagnosis workflow
        self.register_workflow(
            "integrated_diagnosis",
            ["safety_agent", "emotion_agent", "personality_agent", "integrated_diagnosis_agent"],
            {
                "description": "Integrated psychological diagnosis with personality assessment",
                "default_timeout": 90
            }
        )
        
        # Register unified diagnosis workflow using new diagnosis system
        if DIAGNOSIS_SERVICES_AVAILABLE:
            self.register_workflow(
                "unified_diagnosis",
                ["safety_agent", "emotion_agent", "personality_agent", "unified_diagnosis_service"],
                {
                    "description": "Unified diagnosis using enhanced integrated diagnostic system",
                    "default_timeout": 120,
                    "use_diagnosis_service": True
                }
            )
            
            # Register comprehensive diagnosis workflow
            self.register_workflow(
                "comprehensive_diagnosis",
                ["safety_agent", "emotion_agent", "personality_agent", "unified_diagnosis_service"],
                {
                    "description": "Comprehensive diagnosis with all available systems",
                    "default_timeout": 150,
                    "use_diagnosis_service": True,
                    "diagnosis_type": "comprehensive"
                }
            )
        
        # Register the search workflow
        self.register_workflow(
            "search",
            ["safety_agent", "search_agent", "crawler_agent"],
            {
                "description": "Web search workflow with safety checks",
                "default_timeout": 45
            }
        )
        
        # Register enhanced chat workflow with Gemini 2.0
        self.register_workflow(
            "enhanced_empathetic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "chat_agent"],
            {
                "description": "Enhanced chat workflow with emotion analysis and personalized empathetic responses using Gemini 2.0",
                "default_timeout": 45
            }
        )
        
        # Register therapeutic chat workflow with actionable steps
        self.register_workflow(
            "therapeutic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "chat_agent"],
            {
                "description": "Therapeutic chat workflow with practical actionable steps based on evidence-based techniques",
                "default_timeout": 60
            }
        )
        
        # Register growth-oriented therapeutic friction workflow
        self.register_workflow(
            "therapeutic_friction_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapeutic_friction_agent", "chat_agent"],
            {
                "description": "Advanced therapeutic workflow with growth-oriented challenges and strategic friction",
                "default_timeout": 75,
                "use_legacy_friction_agent": True
            }
        )
        
        # Register coordinated sub-agent therapeutic friction workflow
        self.register_workflow(
            "coordinated_therapeutic_friction",
            ["safety_agent", "emotion_agent", "personality_agent", "friction_coordinator", "chat_agent"],
            {
                "description": "Advanced therapeutic workflow using coordinated sub-agent friction analysis",
                "default_timeout": 90,
                "use_sub_agent_coordination": True
            }
        )
        
        # Register integrated therapeutic workflow combining both approaches
        self.register_workflow(
            "integrated_therapeutic_chat",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "therapeutic_friction_agent", "chat_agent"],
            {
                "description": "Comprehensive therapeutic workflow combining evidence-based techniques with growth-oriented friction",
                "default_timeout": 90,
                "use_legacy_friction_agent": True
            }
        )
        
        # Register comprehensive coordinated therapeutic workflow
        self.register_workflow(
            "comprehensive_coordinated_therapeutic",
            ["safety_agent", "emotion_agent", "personality_agent", "therapy_agent", "friction_coordinator", "chat_agent"],
            {
                "description": "Comprehensive therapeutic workflow with evidence-based techniques and coordinated sub-agent friction analysis",
                "default_timeout": 120,
                "use_sub_agent_coordination": True,
                "enable_agent_integration": True
            }
        )
        
        self.logger.debug(f"Registered {len(self.workflows)} standard workflows")
    
    def register_agent(self, agent_id: str, agent_module: Module) -> bool:
        if agent_id in self.agent_modules:
            self.logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.agent_modules[agent_id] = agent_module
        self.logger.debug(f"Registered agent: {agent_id}")
        return True
    
    def register_workflow(self, workflow_id: str, agent_sequence: List[str], 
                         workflow_config: Dict[str, Any] = None) -> bool:
        """
        Register a workflow pattern with the orchestrator.
        
        Args:
            workflow_id: Identifier for the workflow
            agent_sequence: Ordered list of agent IDs for message passing
            workflow_config: Configuration for the workflow
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if workflow_id in self.workflows:
            self.logger.warning(f"Workflow {workflow_id} already registered")
            return False
        
        self.workflows[workflow_id] = {
            "agent_sequence": agent_sequence,
            "config": workflow_config or {}
        }
        
        self.logger.debug(f"Registered workflow: {workflow_id} with {len(agent_sequence)} agents")
        return True
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any],
                             session_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a registered workflow with the given input data.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            session_id: Optional session identifier for tracking
            context: Additional context for the workflow
            
        Returns:
            Results of the workflow execution
        """
        if workflow_id not in self.workflows:
            # Lazy-register standard workflows if none registered yet
            if not self.workflows:
                try:
                    self._register_workflows()
                    self.logger.info("Standard workflows lazily registered")
                except Exception as e:
                    self.logger.error(f"Failed lazy workflow registration: {str(e)}")
            # Re-check after lazy registration
            if workflow_id not in self.workflows:
                self.logger.error(f"Workflow {workflow_id} not found")
                # Ensure a consistent shape for callers expecting status
                return {"error": f"Workflow {workflow_id} not found", "status": "failed"}
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        # Get workflow definition
        workflow = self.workflows[workflow_id]
        agent_sequence = workflow["agent_sequence"]

        # Validate workflow agents before execution
        is_valid, missing_agents, error_message = self._validate_workflow_agents(workflow_id, agent_sequence)
        if not is_valid:
            return {
                "error": error_message,
                "status": "failed",
                "missing_agents": missing_agents,
                "workflow_id": workflow_id
            }

        # Update context store with initial context
        if context:
            await self.update_context(session_id, context)

        # Get full context for this session
        full_context = await self.get_context(session_id)

        # Prepare workflow state
        workflow_state = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "input": input_data,
            "context": full_context,
            "results": {},
            "start_time": time.time(),
            "current_step": 0,
            "steps_completed": 0,
            "agent_sequence": agent_sequence,
            "status": "in_progress"
        }
        
        # Store current workflow
        self.current_workflows[session_id] = workflow_state
        
        # Log workflow start
        self.logger.info(f"Starting workflow: {workflow_id}", 
                      {"session_id": session_id, "workflow": workflow_id})
        
        # Execute each agent in sequence
        current_data = input_data
        
        for idx, agent_id in enumerate(agent_sequence):
            # Update workflow state
            workflow_state["current_step"] = idx
            workflow_state["current_agent"] = agent_id
            
            # Check if this step should use the diagnosis service
            if (agent_id == "unified_diagnosis_service" and 
                workflow["config"].get("use_diagnosis_service", False) and 
                self.diagnosis_integration_enabled and 
                self.diagnosis_orchestrator):
                
                # Use unified diagnosis service instead of agent
                try:
                    self.logger.debug(f"Executing diagnosis service in workflow {workflow_id}", 
                                  {"session_id": session_id})
                    
                    # Record service interaction start
                    agent_start_time = time.time()
                    
                    result = await self._execute_diagnosis_service(
                        current_data, workflow_state, workflow["config"]
                    )
                    
                    # Calculate processing time
                    agent_processing_time = time.time() - agent_start_time
                    
                except Exception as e:
                    error_msg = f"Diagnosis service execution failed: {str(e)}"
                    self.logger.error(error_msg, {"session_id": session_id})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
            else:
                # Check if agent is available
                if agent_id not in self.agent_modules:
                    error_msg = f"Agent {agent_id} not found, workflow cannot continue"
                    self.logger.error(error_msg, {"session_id": session_id})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
                
                # Process with current agent
                try:
                    self.logger.debug(f"Executing agent {agent_id} in workflow {workflow_id}", 
                                  {"session_id": session_id})
                    
                    agent = self.agent_modules[agent_id]
                    agent_process = getattr(agent, "process", None)
                    
                    if not agent_process or not callable(agent_process):
                        error_msg = f"Agent {agent_id} does not have a valid process method"
                        self.logger.error(error_msg, {"session_id": session_id})
                        workflow_state["status"] = "failed"
                        workflow_state["error"] = error_msg
                        break
                    
                    # Refresh context before processing
                    workflow_state["context"] = await self.get_context(session_id)
                    
                    # Record agent interaction start
                    agent_start_time = time.time()
                    
                    # Execute agent's process method with updated context (supports sync/async)
                    result = await self._call_agent_process(agent_process, current_data, workflow_state["context"])
                    
                    # Calculate processing time
                    agent_processing_time = time.time() - agent_start_time
                    
                except Exception as e:
                    error_msg = f"Error processing agent {agent_id}: {str(e)}"
                    self.logger.error(error_msg, {"session_id": session_id, "exception": str(e)})
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    break
                
                # Perform supervision validation if enabled
                if self.supervision_enabled and self.supervisor_agent:
                    try:
                        validation_result = await self.supervisor_agent.validate_agent_response(
                            agent_name=agent_id,
                            input_data=current_data,
                            output_data=result,
                            session_id=session_id
                        )
                        
                        # Record metrics
                        if self.metrics_collector:
                            self.metrics_collector.record_validation_metrics(
                                agent_name=agent_id,
                                validation_result=validation_result,
                                processing_time=agent_processing_time,
                                session_id=session_id
                            )
                        
                        # Log audit event
                        if self.audit_trail:
                            self.audit_trail.log_agent_interaction(
                                session_id=session_id,
                                user_id=self.user_id,
                                agent_name=agent_id,
                                user_input=str(current_data.get("message", "")),
                                agent_response=str(result.get("response", "")),
                                validation_result=validation_result,
                                processing_time=agent_processing_time
                            )
                        
                        # Handle validation results
                        if validation_result.validation_level == ValidationLevel.BLOCKED:
                            # Block the response and use alternative
                            blocking_issues = getattr(validation_result, 'blocking_issues', 
                                                   getattr(validation_result, 'critical_issues', ['Safety concerns']))
                            self.logger.warning(f"Response blocked for {agent_id}", 
                                              {"session_id": session_id, "reason": blocking_issues})
                            
                            if validation_result.alternative_response:
                                result = {"response": validation_result.alternative_response}
                            else:
                                error_msg = f"Agent {agent_id} response blocked due to safety concerns"
                                workflow_state["status"] = "failed"
                                workflow_state["error"] = error_msg
                                break
                            
                            # Log response blocking
                            if self.audit_trail:
                                self.audit_trail.log_response_blocked(
                                    session_id=session_id,
                                    user_id=self.user_id,
                                    agent_name=agent_id,
                                    blocked_content=str(result.get("response", "")),
                                    reason="; ".join(blocking_issues) if isinstance(blocking_issues, list) else str(blocking_issues),
                                    alternative_provided=bool(validation_result.alternative_response)
                                )
                        
                        elif validation_result.validation_level == ValidationLevel.CRITICAL:
                            # Log critical issues but continue
                            self.logger.error(f"Critical validation issues for {agent_id}", 
                                            {"session_id": session_id, "issues": validation_result.critical_issues})
                        
                        # Store validation result in workflow state
                        workflow_state["validation_results"] = workflow_state.get("validation_results", {})
                        overall_score = getattr(validation_result, 'overall_score', 
                                              getattr(validation_result, 'accuracy_score', 0.5))
                        critical_issues = getattr(validation_result, 'critical_issues', [])
                        recommendations = getattr(validation_result, 'recommendations', [])
                        
                        workflow_state["validation_results"][agent_id] = {
                            "validation_level": validation_result.validation_level.value,
                            "overall_score": overall_score,
                            "critical_issues": critical_issues,
                            "recommendations": recommendations
                        }
                        
                    except Exception as validation_error:
                        self.logger.error(f"Validation error for {agent_id}: {str(validation_error)}", 
                                        {"session_id": session_id})
                        # Continue processing even if validation fails
                
            # Store result in workflow state
            workflow_state["results"][agent_id] = result
            
            # Extract and store context updates from the result if available
            if isinstance(result, dict) and "context_updates" in result:
                # Update context with agent's new information
                context_updates = result.pop("context_updates")
                if context_updates:
                    await self.update_context(session_id, context_updates)
                    self.logger.debug(f"Updated context from {agent_id}", 
                                  {"session_id": session_id, "context_keys": list(context_updates.keys())})
            
            # Update data for next agent
            current_data = result
            
            # Update workflow state
            workflow_state["steps_completed"] += 1
        
        # Complete workflow
        workflow_state["end_time"] = time.time()
        workflow_state["duration"] = workflow_state["end_time"] - workflow_state["start_time"]
        
        # Set final status
        if workflow_state["status"] != "failed":
            workflow_state["status"] = "completed"
        
        # Log workflow completion
        log_level = "info" if workflow_state["status"] == "completed" else "error"
        getattr(self.logger, log_level)(
            f"Workflow {workflow_id} {workflow_state['status']} in {workflow_state['duration']:.2f}s",
            {"session_id": session_id, "workflow": workflow_id, "status": workflow_state["status"]}
        )
        
        # Store in history and remove from current
        self.workflow_history[session_id] = workflow_state
        self.current_workflows.pop(session_id, None)
        
        # Add final context to the return data
        final_context = await self.get_context(session_id)
        
        # Return the final output
        return {
            "output": current_data,
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": workflow_state["status"],
            "duration": workflow_state["duration"],
            "steps_completed": workflow_state["steps_completed"],
            "context": final_context
        }
    
    async def send_message(self, sender_id: str, recipient_id: str, 
                         message: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: ID of the sending agent
            recipient_id: ID of the receiving agent
            message: Message content to send
            session_id: Optional session identifier for tracking
            
        Returns:
            Response from the recipient agent
        """
        if recipient_id not in self.agent_modules:
            error_msg = f"Recipient agent {recipient_id} not found"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        # Log message passing
        self.logger.debug(f"Message from {sender_id} to {recipient_id}", 
                      {"session_id": session_id, "sender": sender_id, "recipient": recipient_id})
        
        try:
            # Get the recipient agent
            recipient = self.agent_modules[recipient_id]
            
            # Get the receive_message method if available
            receive_method = getattr(recipient, "receive_message", None)
            
            if not receive_method or not callable(receive_method):
                # Fall back to process method
                receive_method = getattr(recipient, "process", None)
                
                if not receive_method or not callable(receive_method):
                    error_msg = f"Agent {recipient_id} has no valid message handling method"
                    self.logger.error(error_msg)
                    return {"error": error_msg}
            
            # Add metadata to message
            message_with_meta = {
                **message,
                "_meta": {
                    "sender": sender_id,
                    "timestamp": time.time(),
                    "session_id": session_id
                }
            }
            
            # Call the receive method
            response = await receive_method(message_with_meta)
            
            # Log successful message handling
            self.logger.debug(f"Message from {sender_id} to {recipient_id} processed successfully", 
                          {"session_id": session_id})
            
            return response
            
        except Exception as e:
            error_msg = f"Error sending message from {sender_id} to {recipient_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    async def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow by session ID.
        
        Args:
            session_id: Session identifier for the workflow
            
        Returns:
            Current workflow status or history if completed
        """
        # Check current workflows
        if session_id in self.current_workflows:
            return {
                "status": "in_progress",
                "workflow": self.current_workflows[session_id]
            }
        
        # Check workflow history
        if session_id in self.workflow_history:
            return {
                "status": "completed",
                "workflow": self.workflow_history[session_id]
            }
        
        return {
            "status": "not_found",
            "error": f"No workflow found for session {session_id}"
        }
    
    async def get_context(self, session_id: str, context_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get context data for a session
        
        Args:
            session_id: Session identifier
            context_type: Optional specific context type to retrieve (emotion, safety, personality, etc.)
            
        Returns:
            Context data for the session
        """
        if self.enhanced_features_enabled and self.context_manager:
            # Use enhanced context manager
            context = await self.context_manager.get_context(session_id, include_inherited=True)
            
            if context_type:
                return {context_type: context.get(context_type, {})}
            return context
        else:
            # Legacy context handling
            # Initialize context if not exists
            if session_id not in self.context_store:
                self.context_store[session_id] = {}
                
            # Return specific context type if requested
            if context_type:
                return {
                    context_type: self.context_store[session_id].get(context_type, {})
                }
                
            # Return all context
            return self.context_store[session_id]
        
    async def update_context(self, session_id: str, context_data: Dict[str, Any], merge: bool = True) -> bool:
        """
        Update context data for a session
        
        Args:
            session_id: Session identifier
            context_data: New context data to update
            merge: If True, merge with existing context; if False, replace it
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.enhanced_features_enabled and self.context_manager:
                # Use enhanced context manager
                await self.context_manager.update_context(session_id, context_data, merge)
                
                # Publish context update event
                if self.message_bus:
                    update_message = Message(
                        id=str(uuid.uuid4()),
                        type=MessageType.CONTEXT_UPDATE,
                        sender="orchestrator",
                        recipient=None,
                        payload={
                            "session_id": session_id,
                            "context_keys": list(context_data.keys()),
                            "merge": merge
                        },
                        session_id=session_id
                    )
                    await self.message_bus.publish(update_message)
                
            else:
                # Legacy context handling
                # Initialize context if not exists
                if session_id not in self.context_store:
                    self.context_store[session_id] = {}
                    
                # Update context based on merge strategy
                if merge:
                    # Recursively merge nested dictionaries
                    self._deep_merge(self.context_store[session_id], context_data)
                else:
                    # Replace context entirely
                    self.context_store[session_id] = context_data
                
            self.logger.debug(f"Updated context for session {session_id}", 
                          {"session_id": session_id, "context_keys": list(context_data.keys())})
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating context for session {session_id}: {str(e)}")
            return False
            
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dict into target dict

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            else:
                # Update or add non-dictionary items
                target[key] = value

    def _validate_workflow_agents(self, workflow_id: str, agent_sequence: List[str]) -> tuple:
        """
        Validate that all agents in the workflow sequence are registered.

        Args:
            workflow_id: ID of the workflow being validated
            agent_sequence: List of agent IDs in the workflow

        Returns:
            Tuple of (is_valid: bool, missing_agents: List[str], error_message: str)
        """
        missing_agents = []

        for agent_id in agent_sequence:
            # Special case: unified_diagnosis_service is not a regular agent
            if agent_id == "unified_diagnosis_service":
                continue

            # Check if agent is registered
            if agent_id not in self.agent_modules:
                missing_agents.append(agent_id)

        if missing_agents:
            error_msg = (
                f"Workflow '{workflow_id}' validation failed: "
                f"{len(missing_agents)} agent(s) not registered: {', '.join(missing_agents)}. "
                f"Registered agents: {', '.join(self.agent_modules.keys())}"
            )
            self.logger.error(error_msg, {
                "workflow_id": workflow_id,
                "missing_agents": missing_agents,
                "registered_agents": list(self.agent_modules.keys())
            })
            return False, missing_agents, error_msg

        self.logger.debug(f"Workflow '{workflow_id}' validation passed - all agents registered")
        return True, [], ""
    
    async def shutdown(self) -> bool:
        """Shutdown the orchestrator"""
        self.logger.info("Shutting down Agent Orchestrator")
        
        # Clean up any ongoing workflows
        for session_id, workflow in self.current_workflows.items():
            workflow["status"] = "aborted"
            workflow["end_time"] = time.time()
            workflow["duration"] = workflow["end_time"] - workflow["start_time"]
            self.workflow_history[session_id] = workflow
            
            self.logger.warning(f"Workflow {workflow['workflow_id']} aborted during shutdown", 
                            {"session_id": session_id})
        
        self.current_workflows.clear()
        
        # Shutdown supervision system
        if self.supervision_enabled:
            try:
                if self.metrics_collector:
                    # Export final metrics
                    from src.monitoring.supervisor_metrics import MetricsExporter
                    exporter = MetricsExporter(self.metrics_collector)
                    export_path = f"logs/final_metrics_{int(time.time())}.json"
                    exporter.export_to_json(export_path)
                    self.logger.info(f"Final metrics exported to {export_path}")
                
                if self.audit_trail:
                    # Cleanup expired audit records
                    cleaned = self.audit_trail.cleanup_expired_records()
                    self.logger.info(f"Cleaned up {cleaned} expired audit records")
                
                self.logger.info("Supervision system shutdown complete")
            except Exception as e:
                self.logger.error(f"Error during supervision system shutdown: {str(e)}")
        
        # Shutdown enhanced components
        if self.enhanced_features_enabled:
            if self.message_bus:
                await self.message_bus.stop()
                
        return await super().shutdown()

    async def process_message(self, message: str, user_id: str = None, workflow_id: str = "enhanced_empathetic_chat") -> Dict[str, Any]:
        """
        Process a user message through the appropriate workflow and track the conversation
        
        Args:
            message: The user message to process
            user_id: User identifier (optional, uses the default from initialization if not provided)
            workflow_id: ID of the workflow to use for processing
        
        Returns:
            Result of processing the message
        """
        # Generate session ID for this interaction
        session_id = f"session_{int(time.time())}"
        
        self.logger.info(f"Processing message with workflow {workflow_id}", 
                      {"session_id": session_id, "message_length": len(message)})
        
        # Set up initial context
        initial_context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id or self.user_id
        }
        
        # Enhance context with relevant data from vector database
        try:
            # Add relevant past conversations
            past_conversations = search_relevant_data(message, ["conversation"], limit=3)
            if past_conversations:
                initial_context["relevant_conversations"] = past_conversations
                self.logger.debug(f"Added {len(past_conversations)} relevant conversations to context")
                
            # Add most recent diagnosis if available
            from utils.vector_db_integration import get_user_data
            latest_diagnosis = get_user_data("diagnosis")
            if latest_diagnosis:
                initial_context["latest_diagnosis"] = latest_diagnosis
                self.logger.debug("Added latest diagnosis to context")
                
            # Add most recent personality assessment if available
            latest_personality = get_user_data("personality")
            if latest_personality:
                initial_context["personality"] = latest_personality
                self.logger.debug("Added personality profile to context")
                
        except Exception as e:
            self.logger.warning(f"Error enhancing context from vector DB: {str(e)}")
        
        # Execute the workflow
        result = await self.execute_workflow(
            workflow_id=workflow_id,
            input_data={"message": message},
            session_id=session_id,
            context=initial_context
        )
        
        # Extract response and emotion data
        response = ""
        emotion_data = None
        
        if isinstance(result, dict):
            # Extract the main response text
            if "output" in result and isinstance(result["output"], dict):
                response = result["output"].get("response", "")
            elif "response" in result:
                response = result["response"]
            
            # Extract emotion data if available
            if "emotion_agent" in result.get("steps_completed", []):
                emotion_result = result.get("results", {}).get("emotion_agent", {})
                if emotion_result and isinstance(emotion_result, dict):
                    emotion_data = emotion_result.get("emotion_analysis")
        
        # Track the conversation in our central vector DB
        metadata = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "duration": result.get("duration") if isinstance(result, dict) else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the conversation to the tracker
        if response:
            try:
                # Get conversation tracker from central vector DB
                if self.conversation_tracker is None:
                    self.conversation_tracker = get_conversation_tracker()
                
                if self.conversation_tracker:
                    conversation_id = self.conversation_tracker.add_conversation(
                        user_message=message,
                        assistant_response=response,
                        emotion_data=emotion_data,
                        metadata=metadata
                    )
                    if conversation_id:
                        self.logger.info(f"Tracked conversation: {conversation_id}")
                        if isinstance(result, dict):
                            result["conversation_id"] = conversation_id
                else:
                    self.logger.warning("Conversation tracker not available")
            except Exception as e:
                self.logger.error(f"Error tracking conversation: {str(e)}")
        
        return result
    
    # Supervision and Monitoring Methods
    
    async def get_supervision_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive supervision summary."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            summary = {
                "supervision_status": "active",
                "time_window_hours": time_window_hours,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get performance metrics
            if self.metrics_collector:
                from src.monitoring.supervisor_metrics import PerformanceDashboard
                dashboard = PerformanceDashboard(self.metrics_collector)
                summary["real_time_metrics"] = dashboard.get_real_time_metrics()
                summary["system_analytics"] = dashboard.get_system_analytics()
            
            # Get supervisor performance
            if self.supervisor_agent:
                summary["supervisor_metrics"] = self.supervisor_agent.get_performance_metrics()
            
            # Get audit statistics
            if self.audit_trail:
                from datetime import timedelta
                start_time = datetime.now() - timedelta(hours=time_window_hours)
                
                # Get critical events
                from src.auditing.audit_system import AuditEventType, AuditSeverity
                critical_events = self.audit_trail.get_events_by_type(
                    AuditEventType.CRISIS_DETECTED, start_time
                )
                blocked_responses = self.audit_trail.get_events_by_type(
                    AuditEventType.RESPONSE_BLOCKED, start_time
                )
                
                summary["audit_summary"] = {
                    "critical_events": len(critical_events),
                    "blocked_responses": len(blocked_responses),
                    "total_interactions": len(self.audit_trail.get_events_by_type(
                        AuditEventType.AGENT_INTERACTION, start_time
                    ))
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating supervision summary: {str(e)}")
            return {"error": f"Failed to generate summary: {str(e)}"}
    
    async def get_agent_quality_report(self, agent_name: str = None) -> Dict[str, Any]:
        """Get quality report for specific agent or all agents."""
        if not self.supervision_enabled or not self.metrics_collector:
            return {"error": "Supervision or metrics not available"}
        
        try:
            from src.monitoring.supervisor_metrics import PerformanceDashboard
            from datetime import timedelta
            
            dashboard = PerformanceDashboard(self.metrics_collector)
            report = dashboard.get_agent_performance_report(
                agent_name=agent_name,
                time_window=timedelta(days=1)
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating agent quality report: {str(e)}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    async def get_session_analysis(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a specific session."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            analysis = {
                "session_id": session_id,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Get audit trail for session
            if self.audit_trail:
                audit_events = self.audit_trail.get_session_audit_trail(session_id)
                analysis["audit_events_count"] = len(audit_events)
                
                # Categorize events
                event_summary = {}
                for event in audit_events:
                    event_type = event.event_type.value
                    event_summary[event_type] = event_summary.get(event_type, 0) + 1
                
                analysis["event_summary"] = event_summary
                
                # Check for critical issues
                critical_events = [e for e in audit_events if e.severity.value in ["critical", "emergency"]]
                analysis["critical_issues"] = len(critical_events)
                analysis["critical_details"] = [
                    {
                        "event_type": e.event_type.value,
                        "severity": e.severity.value,
                        "description": e.event_description,
                        "timestamp": e.timestamp.isoformat()
                    }
                    for e in critical_events
                ]
            
            # Get supervisor session summary
            if self.supervisor_agent:
                supervisor_summary = await self.supervisor_agent.get_session_summary(session_id)
                analysis["supervisor_summary"] = supervisor_summary
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating session analysis: {str(e)}")
            return {"error": f"Failed to generate analysis: {str(e)}"}
    
    async def export_compliance_report(self, compliance_standard: str, 
                                     start_date: str, end_date: str) -> Dict[str, Any]:
        """Export compliance report for regulatory purposes."""
        if not self.supervision_enabled or not self.audit_trail:
            return {"error": "Supervision or audit trail not available"}
        
        try:
            from src.auditing.audit_system import ComplianceStandard
            from datetime import datetime
            
            # Parse compliance standard
            try:
                standard = ComplianceStandard(compliance_standard.lower())
            except ValueError:
                return {"error": f"Invalid compliance standard: {compliance_standard}"}
            
            # Parse dates
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Generate compliance report
            compliance_report = self.audit_trail.generate_compliance_report(
                compliance_standard=standard,
                start_date=start_dt,
                end_date=end_dt
            )
            
            # Export audit data
            export_path = f"exports/compliance_{compliance_standard}_{start_date}_{end_date}.json"
            self.audit_trail.export_audit_data(
                output_path=export_path,
                start_date=start_dt,
                end_date=end_dt
            )
            
            return {
                "compliance_report": {
                    "report_id": compliance_report.report_id,
                    "compliance_standard": compliance_report.compliance_standard.value,
                    "reporting_period": {
                        "start": compliance_report.reporting_period["start"].isoformat(),
                        "end": compliance_report.reporting_period["end"].isoformat()
                    },
                    "total_events": compliance_report.total_events,
                    "violations_found": compliance_report.violations_found,
                    "compliance_score": compliance_report.compliance_score,
                    "critical_findings": compliance_report.critical_findings,
                    "recommendations": compliance_report.recommendations
                },
                "export_path": export_path,
                "generated_timestamp": compliance_report.generated_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting compliance report: {str(e)}")
            return {"error": f"Failed to export report: {str(e)}"}
    
    async def configure_supervision(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure supervision system parameters."""
        if not self.supervision_enabled:
            return {"error": "Supervision not enabled"}
        
        try:
            result = {"configured": [], "errors": []}
            
            # Configure supervisor agent
            if "supervisor_settings" in config and self.supervisor_agent:
                # This would update supervisor configuration
                result["configured"].append("supervisor_settings")
            
            # Configure metrics collection
            if "metrics_settings" in config and self.metrics_collector:
                metrics_config = config["metrics_settings"]
                
                # Update thresholds
                if "thresholds" in metrics_config:
                    self.metrics_collector.metric_thresholds.update(metrics_config["thresholds"])
                    result["configured"].append("metrics_thresholds")
            
            # Configure audit settings
            if "audit_settings" in config and self.audit_trail:
                # This would update audit configuration
                result["configured"].append("audit_settings")
            
            self.logger.info(f"Supervision configuration updated: {result['configured']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error configuring supervision: {str(e)}")
            return {"error": f"Configuration failed: {str(e)}"}
    
    def get_supervision_status(self) -> Dict[str, Any]:
        """Get current supervision system status."""
        return {
            "supervision_enabled": self.supervision_enabled,
            "supervisor_agent_active": self.supervisor_agent is not None,
            "metrics_collector_active": self.metrics_collector is not None,
            "audit_trail_active": self.audit_trail is not None,
            "active_workflows": len(self.current_workflows),
            "total_agents": len(self.agent_modules),
            "status_timestamp": datetime.now().isoformat()
        }
    
    # Diagnosis Service Integration Methods
    
    async def _execute_diagnosis_service(self, 
                                       current_data: Dict[str, Any],
                                       workflow_state: Dict[str, Any],
                                       workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute diagnosis using the unified diagnosis service."""
        try:
            # Convert current data to diagnosis request format
            diagnosis_request = await self.diagnosis_adapter.adapt_agent_request(
                agent_input=current_data,
                context={
                    "user_id": workflow_state.get("context", {}).get("user_id", self.user_id),
                    "session_id": workflow_state["session_id"],
                    "workflow_id": workflow_state["workflow_id"],
                    "agent_type": "orchestrator",
                    "diagnosis_type": workflow_config.get("diagnosis_type", "comprehensive")
                }
            )
            
            # Perform diagnosis
            diagnosis_result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Convert result back to agent format
            agent_format = workflow_config.get("agent_format", "comprehensive")
            adapted_result = await self.diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, agent_format
            )
            
            # Add context updates from diagnosis
            if "context_updates" in adapted_result:
                workflow_state["context"].update(adapted_result["context_updates"])
            
            self.logger.info(f"Diagnosis service executed successfully for session {workflow_state['session_id']}")
            return adapted_result
            
        except Exception as e:
            self.logger.error(f"Error executing diagnosis service: {str(e)}")
            raise
    
    async def diagnose(self, 
                      message: str, 
                      user_id: str = None,
                      diagnosis_type: str = "comprehensive",
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Direct diagnosis service endpoint."""
        if not self.diagnosis_integration_enabled or not self.diagnosis_orchestrator:
            return {"error": "Diagnosis service not available"}
        
        try:
            # Create diagnosis request
            session_id = f"direct_diagnosis_{int(time.time())}"
            
            # Convert diagnosis type string to enum
            try:
                diag_type = DiagnosisType(diagnosis_type.lower())
            except ValueError:
                diag_type = DiagnosisType.COMPREHENSIVE
            
            diagnosis_request = DiagnosisRequest(
                user_id=user_id or self.user_id,
                session_id=session_id,
                message=message,
                conversation_history=[],
                context=context or {},
                diagnosis_type=diag_type
            )
            
            # Perform diagnosis
            diagnosis_result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Adapt result for return
            adapted_result = await self.diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, "comprehensive"
            )
            
            return adapted_result
            
        except Exception as e:
            self.logger.error(f"Error in direct diagnosis: {str(e)}")
            return {"error": str(e)}
    
    def get_diagnosis_status(self) -> Dict[str, Any]:
        """Get diagnosis system integration status."""
        return {
            "diagnosis_integration_enabled": self.diagnosis_integration_enabled,
            "diagnosis_services_available": DIAGNOSIS_SERVICES_AVAILABLE,
            "diagnosis_orchestrator_active": self.diagnosis_orchestrator is not None,
            "diagnosis_adapter_active": self.diagnosis_adapter is not None,
            "status_timestamp": datetime.now().isoformat()
        }
    
    # Enhanced Orchestration Methods
    
    async def execute_workflow_enhanced(self, workflow_id: str, input_data: Dict[str, Any],
                                      session_id: str = None, context: Dict[str, Any] = None,
                                      use_circuit_breaker: bool = True, 
                                      use_supervision_mesh: bool = True) -> Dict[str, Any]:
        """
        Execute workflow with enhanced features: circuit breakers, mesh validation, event-driven communication.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            session_id: Optional session identifier for tracking
            context: Additional context for the workflow
            use_circuit_breaker: Whether to use circuit breaker protection
            use_supervision_mesh: Whether to use supervision mesh validation
            
        Returns:
            Results of the enhanced workflow execution
        """
        if not self.enhanced_features_enabled:
            return await self.execute_workflow(workflow_id, input_data, session_id, context)
        
        if workflow_id not in self.workflows:
            # Lazy-register standard workflows if none registered yet
            if not self.workflows:
                try:
                    self._register_workflows()
                    self.logger.info("Standard workflows lazily registered")
                except Exception as e:
                    self.logger.error(f"Failed lazy workflow registration: {str(e)}")
            # Re-check after lazy registration
            if workflow_id not in self.workflows:
                self.logger.error(f"Workflow {workflow_id} not found")
                return {"error": f"Workflow {workflow_id} not found", "status": "failed"}
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"enhanced_session_{int(time.time())}"

        # Get workflow definition
        workflow = self.workflows[workflow_id]
        agent_sequence = workflow["agent_sequence"]

        # Validate workflow agents before execution
        is_valid, missing_agents, error_message = self._validate_workflow_agents(workflow_id, agent_sequence)
        if not is_valid:
            return {
                "error": error_message,
                "status": "failed",
                "missing_agents": missing_agents,
                "workflow_id": workflow_id
            }

        # Update context store with initial context
        if context:
            await self.update_context(session_id, context)

        # Get full context for this session
        full_context = await self.get_context(session_id)

        # Prepare enhanced workflow state
        workflow_state = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "input": input_data,
            "context": full_context,
            "results": {},
            "start_time": time.time(),
            "current_step": 0,
            "steps_completed": 0,
            "agent_sequence": agent_sequence,
            "status": "in_progress",
            "enhanced_features": {
                "circuit_breaker": use_circuit_breaker,
                "supervision_mesh": use_supervision_mesh,
                "event_driven": True
            },
            "performance_metrics": {},
            "validation_results": {}
        }
        
        # Store current workflow
        self.current_workflows[session_id] = workflow_state
        
        # Log workflow start
        self.logger.info(f"Starting enhanced workflow: {workflow_id}", 
                      {"session_id": session_id, "workflow": workflow_id})
        
        # Execute each agent in sequence with enhanced features
        current_data = input_data
        
        for idx, agent_id in enumerate(agent_sequence):
            # Update workflow state
            workflow_state["current_step"] = idx
            workflow_state["current_agent"] = agent_id
            
            try:
                # Check if this step should use the diagnosis service
                if (agent_id == "unified_diagnosis_service" and 
                    workflow["config"].get("use_diagnosis_service", False) and 
                    self.diagnosis_integration_enabled and 
                    self.diagnosis_orchestrator):
                    
                    # Use unified diagnosis service instead of agent
                    agent_start_time = time.time()
                    
                    result = await self._execute_diagnosis_service_enhanced(
                        current_data, workflow_state, workflow["config"]
                    )
                    
                    agent_processing_time = time.time() - agent_start_time
                    
                else:
                    # Process with enhanced agent execution
                    result, agent_processing_time = await self._execute_agent_enhanced(
                        agent_id, current_data, workflow_state, 
                        use_circuit_breaker, use_supervision_mesh
                    )
                
                # Record performance metrics
                if self.performance_monitor:
                    await self.performance_monitor.record_agent_performance(
                        agent_id, agent_processing_time, True, session_id
                    )
                
                # Store result in workflow state
                workflow_state["results"][agent_id] = result
                workflow_state["performance_metrics"][agent_id] = {
                    "processing_time": agent_processing_time,
                    "success": True
                }
                
                # Extract and store context updates from the result if available
                if isinstance(result, dict) and "context_updates" in result:
                    context_updates = result.pop("context_updates")
                    if context_updates:
                        await self.update_context(session_id, context_updates)
                        self.logger.debug(f"Updated context from {agent_id}", 
                                      {"session_id": session_id, "context_keys": list(context_updates.keys())})
                
                # Update data for next agent
                current_data = result
                workflow_state["steps_completed"] += 1
                
            except Exception as e:
                # Enhanced error handling
                await self._handle_agent_error(agent_id, e, workflow_state, session_id)
                
                # Record failed performance
                if self.performance_monitor:
                    await self.performance_monitor.record_agent_performance(
                        agent_id, time.time() - workflow_state.get("agent_start_time", time.time()), 
                        False, session_id
                    )
                break
        
        # Complete workflow
        workflow_state["end_time"] = time.time()
        workflow_state["duration"] = workflow_state["end_time"] - workflow_state["start_time"]
        
        # Set final status
        if workflow_state["status"] != "failed":
            workflow_state["status"] = "completed"
        
        # Log workflow completion
        log_level = "info" if workflow_state["status"] == "completed" else "error"
        getattr(self.logger, log_level)(
            f"Enhanced workflow {workflow_id} {workflow_state['status']} in {workflow_state['duration']:.2f}s",
            {"session_id": session_id, "workflow": workflow_id, "status": workflow_state["status"]}
        )
        
        # Store in history and remove from current
        self.workflow_history[session_id] = workflow_state
        self.current_workflows.pop(session_id, None)
        
        # Add final context to the return data
        final_context = await self.get_context(session_id)
        
        # Return the enhanced output
        return {
            "output": current_data,
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": workflow_state["status"],
            "duration": workflow_state["duration"],
            "steps_completed": workflow_state["steps_completed"],
            "context": final_context,
            "enhanced_metrics": {
                "performance_metrics": workflow_state["performance_metrics"],
                "validation_results": workflow_state["validation_results"],
                "features_used": workflow_state["enhanced_features"]
            }
        }
    
    async def _execute_agent_enhanced(self, agent_id: str, current_data: Dict[str, Any], 
                                    workflow_state: Dict[str, Any], 
                                    use_circuit_breaker: bool, 
                                    use_supervision_mesh: bool) -> tuple:
        """Execute agent with enhanced features."""
        # Check if agent is available
        if agent_id not in self.agent_modules:
            error_msg = f"Agent {agent_id} not found, workflow cannot continue"
            self.logger.error(error_msg, {"session_id": workflow_state["session_id"]})
            workflow_state["status"] = "failed"
            workflow_state["error"] = error_msg
            raise Exception(error_msg)
        
        agent = self.agent_modules[agent_id]
        agent_process = getattr(agent, "process", None)
        
        if not agent_process or not callable(agent_process):
            error_msg = f"Agent {agent_id} does not have a valid process method"
            self.logger.error(error_msg, {"session_id": workflow_state["session_id"]})
            workflow_state["status"] = "failed"
            workflow_state["error"] = error_msg
            raise Exception(error_msg)
        
        # Refresh context before processing
        workflow_state["context"] = await self.get_context(workflow_state["session_id"])
        
        # Record agent interaction start
        agent_start_time = time.time()
        workflow_state["agent_start_time"] = agent_start_time
        
        # Execute with circuit breaker if enabled
        if use_circuit_breaker and self.circuit_breaker_enabled:
            circuit_breaker = self._get_or_create_circuit_breaker(agent_id)
            async def _runner():
                return await self._call_agent_process(agent_process, current_data, workflow_state["context"])
            result = await circuit_breaker.execute(_runner)
        else:
            # Execute agent's process method with retry if available
            if self.retry_manager:
                # Pass helper as the async callable for retries
                result = await self.retry_manager.execute_with_retry(
                    self._call_agent_process, agent_process, current_data, workflow_state["context"]
                )
            else:
                result = await self._call_agent_process(agent_process, current_data, workflow_state["context"])
        
        # Calculate processing time
        agent_processing_time = time.time() - agent_start_time
        
        # Enhanced validation with supervision mesh
        if use_supervision_mesh and self.supervision_mesh_enabled and self.validator_registry:
            validation_request = ValidationRequest(
                agent_name=agent_id,
                input_data=current_data,
                output_data=result,
                session_id=workflow_state["session_id"]
            )
            
            mesh_validation_result = await self.validator_registry.validate_with_consensus(validation_request)
            workflow_state["validation_results"][agent_id] = mesh_validation_result
            
            # Handle mesh validation results
            if mesh_validation_result.get("is_blocked", False):
                error_msg = f"Agent {agent_id} response blocked by supervision mesh"
                workflow_state["status"] = "failed"
                workflow_state["error"] = error_msg
                self.logger.warning(error_msg, {"session_id": workflow_state["session_id"]})
                raise Exception(error_msg)
        
        elif self.supervision_enabled and self.supervisor_agent:
            # Fallback to single supervisor validation
            validation_result = await self.supervisor_agent.validate_agent_response(
                agent_name=agent_id,
                input_data=current_data,
                output_data=result,
                session_id=workflow_state["session_id"]
            )
            
            workflow_state["validation_results"][agent_id] = {
                "validation_level": validation_result.validation_level.value,
                "overall_score": validation_result.accuracy_score,
                "critical_issues": validation_result.critical_issues,
                "recommendations": validation_result.recommendations
            }
            
            # Handle single validation results
            if validation_result.validation_level == ValidationLevel.BLOCKED:
                if validation_result.alternative_response:
                    result = {"response": validation_result.alternative_response}
                else:
                    error_msg = f"Agent {agent_id} response blocked due to safety concerns"
                    workflow_state["status"] = "failed"
                    workflow_state["error"] = error_msg
                    raise Exception(error_msg)
        
        return result, agent_processing_time

    async def _call_agent_process(self, agent_process: Callable, current_data: Dict[str, Any], context: Dict[str, Any]):
        """Call an agent's process method whether it's sync or async, and resolve awaitables."""
        if asyncio.iscoroutinefunction(agent_process):
            return await agent_process(current_data, context=context)
        # Call synchronously and await if needed
        result = agent_process(current_data, context=context)
        if inspect.isawaitable(result):
            return await result
        return result
    
    def _get_or_create_circuit_breaker(self, agent_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent."""
        if agent_id not in self.circuit_breakers:
            config = self.config or {}
            cb_config = config.get("circuit_breaker_config", {})
            
            self.circuit_breakers[agent_id] = CircuitBreaker(
                agent_id, 
                CircuitBreakerConfig(
                    failure_threshold=cb_config.get("failure_threshold", 5),
                    reset_timeout=cb_config.get("reset_timeout", 60),
                    success_threshold=cb_config.get("success_threshold", 2),
                    timeout=cb_config.get("timeout", 30)
                )
            )
        
        return self.circuit_breakers[agent_id]
    
    async def _execute_diagnosis_service_enhanced(self, current_data: Dict[str, Any], 
                                                workflow_state: Dict[str, Any], 
                                                workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute diagnosis service with enhanced features."""
        # Use retry manager for diagnosis service
        if self.retry_manager:
            return await self.retry_manager.execute_with_retry(
                self._execute_diagnosis_service, current_data, workflow_state, workflow_config
            )
        else:
            return await self._execute_diagnosis_service(current_data, workflow_state, workflow_config)
    
    async def _handle_agent_error(self, agent_id: str, error: Exception, 
                                workflow_state: Dict[str, Any], session_id: str):
        """Enhanced error handling for agent failures."""
        error_msg = f"Error processing agent {agent_id}: {str(error)}"
        self.logger.error(error_msg, {"session_id": session_id, "exception": str(error)})
        
        # Publish error event
        if self.message_bus:
            error_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.ERROR_EVENT,
                sender="orchestrator",
                recipient=None,
                payload={
                    "agent_id": agent_id,
                    "error": error_msg,
                    "session_id": session_id,
                    "workflow_id": workflow_state["workflow_id"]
                },
                session_id=session_id
            )
            await self.message_bus.publish(error_message)
        
        # Update workflow state
        workflow_state["status"] = "failed"
        workflow_state["error"] = error_msg
        workflow_state["failed_agent"] = agent_id
    
    # Event Handlers
    
    async def _handle_validation_request(self, message: Message):
        """Handle validation request messages."""
        try:
            payload = message.payload
            if self.validator_registry:
                request = ValidationRequest(**payload)
                result = await self.validator_registry.validate_with_consensus(request)
                
                # Send response if reply_to is specified
                if message.reply_to and self.message_bus:
                    response = Message(
                        id=str(uuid.uuid4()),
                        type=MessageType.VALIDATION_RESPONSE,
                        sender="orchestrator",
                        recipient=message.reply_to,
                        payload=result,
                        session_id=message.session_id,
                        correlation_id=message.id
                    )
                    await self.message_bus.publish(response)
        
        except Exception as e:
            self.logger.error(f"Error handling validation request: {str(e)}")
    
    async def _handle_error_event(self, message: Message):
        """Handle error event messages."""
        payload = message.payload
        self.logger.error(f"Error event received: {payload}")
        
        # Additional error handling logic can be added here
        # For example, triggering fallback workflows or alerting
    
    async def _handle_metrics_event(self, message: Message):
        """Handle metrics event messages."""
        try:
            payload = message.payload
            if self.metrics_collector:
                # Record the metric
                self.metrics_collector.record_metric(
                    payload.get("metric_name", "unknown"),
                    payload.get("value", 0),
                    metadata=payload.get("metadata", {})
                )
        except Exception as e:
            self.logger.error(f"Error handling metrics event: {str(e)}")
    
    # Enhanced Service Methods
    
    async def register_validator(self, validator_id: str, validator: Any, 
                               validator_types: List[str] = None, weight: float = 1.0) -> Dict[str, Any]:
        """Register a new validator in the supervision mesh."""
        if not self.supervision_mesh_enabled or not self.validator_registry:
            return {"error": "Supervision mesh not enabled"}
        
        try:
            self.validator_registry.register_validator(validator_id, validator, validator_types, weight)
            return {"success": True, "validator_id": validator_id}
        except Exception as e:
            return {"error": str(e)}
    
    async def send_event(self, event_type: str, payload: Dict[str, Any], 
                        recipient: str = None, session_id: str = None) -> Dict[str, Any]:
        """Send an event through the message bus."""
        if not self.event_driven_enabled or not self.message_bus:
            return {"error": "Event-driven communication not enabled"}
        
        try:
            message_type = MessageType(event_type)
            message = Message(
                id=str(uuid.uuid4()),
                type=message_type,
                sender="orchestrator",
                recipient=recipient,
                payload=payload,
                session_id=session_id
            )
            
            await self.message_bus.publish(message)
            return {"success": True, "message_id": message.id}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def get_agent_health(self, agent_name: str = None) -> Dict[str, Any]:
        """Get agent health status from performance monitor."""
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}
        
        try:
            health_data = self.performance_monitor.get_agent_health(agent_name)
            alerts = self.performance_monitor.get_performance_alerts()
            
            return {
                "health_status": health_data,
                "recent_alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def get_performance_metrics(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}
        
        try:
            metrics = {
                "agent_health": self.performance_monitor.get_agent_health(),
                "recent_alerts": self.performance_monitor.get_performance_alerts(),
                "circuit_breaker_status": {
                    cb_id: {
                        "state": cb.state.value,
                        "failure_count": cb.failure_count,
                        "success_count": cb.success_count
                    }
                    for cb_id, cb in self.circuit_breakers.items()
                },
                "message_bus_status": {
                    "running": self.message_bus.running if self.message_bus else False,
                    "queue_size": self.message_bus.message_queue.qsize() if self.message_bus else 0
                } if self.message_bus else None,
                "validator_registry_status": {
                    "validators_registered": len(self.validator_registry.validators) if self.validator_registry else 0
                } if self.validator_registry else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_enhanced_features_status(self) -> Dict[str, Any]:
        """Get status of all enhanced features."""
        return {
            "enhanced_features_enabled": self.enhanced_features_enabled,
            "event_driven_enabled": self.event_driven_enabled,
            "supervision_mesh_enabled": self.supervision_mesh_enabled,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "components_status": {
                "message_bus": {
                    "initialized": self.message_bus is not None,
                    "running": self.message_bus.running if self.message_bus else False
                },
                "validator_registry": {
                    "initialized": self.validator_registry is not None,
                    "validator_count": len(self.validator_registry.validators) if self.validator_registry else 0
                },
                "context_manager": {
                    "initialized": hasattr(self, 'context_manager') and self.context_manager is not None
                },
                "performance_monitor": {
                    "initialized": self.performance_monitor is not None
                },
                "retry_manager": {
                    "initialized": self.retry_manager is not None
                },
                "circuit_breakers": {
                    "count": len(self.circuit_breakers),
                    "agents": list(self.circuit_breakers.keys())
                }
            },
            "status_timestamp": datetime.now().isoformat()
        }
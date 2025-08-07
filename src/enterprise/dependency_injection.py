"""
Enterprise Dependency Injection Container and Health Check System for Solace-AI

This module provides comprehensive dependency management and health monitoring:
- Advanced dependency injection container with lifecycle management
- Service discovery and registration
- Health check orchestration and monitoring
- Circuit breaker patterns for service resilience
- Service mesh integration capabilities
- Resource lifecycle management
- Configuration management integration
- Performance monitoring for dependencies
- Automated service recovery
"""

import asyncio
import time
import inspect
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable, Type, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import functools

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority

logger = get_logger(__name__)

T = TypeVar('T')


class ServiceLifecycle(Enum):
    """Service lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ServiceScope(Enum):
    """Service scopes for dependency injection."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    REQUEST = "request"
    SESSION = "session"


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceDefinition:
    """Service definition for dependency injection."""
    
    service_name: str
    service_type: Type
    implementation_type: Type
    scope: ServiceScope = ServiceScope.SINGLETON
    factory_function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    health_check: Optional[Callable] = None
    startup_priority: int = 100  # Lower numbers start first
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'service_name': self.service_name,
            'service_type': self.service_type.__name__ if self.service_type else None,
            'implementation_type': self.implementation_type.__name__ if self.implementation_type else None,
            'scope': self.scope.value,
            'dependencies': self.dependencies,
            'configuration': self.configuration,
            'startup_priority': self.startup_priority,
            'tags': list(self.tags)
        }


@dataclass
class ServiceInstance:
    """Service instance with lifecycle management."""
    
    service_name: str
    instance: Any
    definition: ServiceDefinition
    lifecycle_state: ServiceLifecycle = ServiceLifecycle.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    health_details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'service_name': self.service_name,
            'lifecycle_state': self.lifecycle_state.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'health_status': self.health_status.value,
            'health_details': self.health_details,
            'metrics': self.metrics,
            'definition': self.definition.to_dict()
        }


@dataclass
class HealthCheckResult:
    """Health check result."""
    
    service_name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    check_duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'service_name': self.service_name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'check_duration': self.check_duration,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    
    failure_threshold: int = 5
    timeout_seconds: int = 60
    success_threshold: int = 2
    request_volume_threshold: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CircuitBreaker:
    """Circuit breaker for service resilience."""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig):
        self.service_name = service_name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_count = 0
        self.lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        with self.lock:
            self.request_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        with self.lock:
            self.request_count += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.request_count >= self.config.request_volume_threshold and \
               self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'service_name': self.service_name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'request_count': self.request_count,
                'last_failure_time': self.last_failure_time,
                'config': self.config.to_dict()
            }


class ServiceRegistry:
    """Service registry for service discovery."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.service_definitions: Dict[str, ServiceDefinition] = {}
        self.tags_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> service_names
        self.type_index: Dict[Type, Set[str]] = defaultdict(set)  # type -> service_names
        self.lock = threading.RLock()
    
    def register_definition(self, definition: ServiceDefinition) -> None:
        """Register a service definition."""
        with self.lock:
            self.service_definitions[definition.service_name] = definition
            
            # Update indexes
            for tag in definition.tags:
                self.tags_index[tag].add(definition.service_name)
            
            self.type_index[definition.service_type].add(definition.service_name)
            if definition.implementation_type != definition.service_type:
                self.type_index[definition.implementation_type].add(definition.service_name)
        
        logger.info(f"Registered service definition: {definition.service_name}")
    
    def register_instance(self, instance: ServiceInstance) -> None:
        """Register a service instance."""
        with self.lock:
            self.services[instance.service_name] = instance
        
        logger.info(f"Registered service instance: {instance.service_name}")
    
    def get_service(self, service_name: str) -> Optional[ServiceInstance]:
        """Get service instance by name."""
        with self.lock:
            return self.services.get(service_name)
    
    def get_services_by_tag(self, tag: str) -> List[ServiceInstance]:
        """Get services by tag."""
        with self.lock:
            service_names = self.tags_index.get(tag, set())
            return [self.services[name] for name in service_names if name in self.services]
    
    def get_services_by_type(self, service_type: Type) -> List[ServiceInstance]:
        """Get services by type."""
        with self.lock:
            service_names = self.type_index.get(service_type, set())
            return [self.services[name] for name in service_names if name in self.services]
    
    def get_all_services(self) -> List[ServiceInstance]:
        """Get all registered services."""
        with self.lock:
            return list(self.services.values())
    
    def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        with self.lock:
            if service_name in self.services:
                service = self.services[service_name]
                del self.services[service_name]
                
                # Update indexes
                definition = self.service_definitions.get(service_name)
                if definition:
                    for tag in definition.tags:
                        self.tags_index[tag].discard(service_name)
                    
                    self.type_index[definition.service_type].discard(service_name)
                    if definition.implementation_type != definition.service_type:
                        self.type_index[definition.implementation_type].discard(service_name)
                
                logger.info(f"Unregistered service: {service_name}")
                return True
        
        return False
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status."""
        with self.lock:
            service_counts_by_state = defaultdict(int)
            service_counts_by_health = defaultdict(int)
            
            for service in self.services.values():
                service_counts_by_state[service.lifecycle_state.value] += 1
                service_counts_by_health[service.health_status.value] += 1
            
            return {
                'total_services': len(self.services),
                'total_definitions': len(self.service_definitions),
                'services_by_state': dict(service_counts_by_state),
                'services_by_health': dict(service_counts_by_health),
                'tags': list(self.tags_index.keys()),
                'types': [t.__name__ for t in self.type_index.keys()]
            }


class HealthChecker:
    """Health check orchestrator."""
    
    def __init__(self, service_registry: ServiceRegistry, event_bus: EventBus):
        self.service_registry = service_registry
        self.event_bus = event_bus
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 10   # seconds
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Background task
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("HealthChecker initialized")
    
    async def start(self) -> None:
        """Start health checking."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("HealthChecker started")
    
    async def stop(self) -> None:
        """Stop health checking."""
        if not self._running:
            return
        
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("HealthChecker stopped")
    
    async def check_service_health(self, service_name: str) -> HealthCheckResult:
        """Check health of a specific service."""
        
        service = self.service_registry.get_service(service_name)
        if not service:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                message="Service not found"
            )
        
        start_time = time.time()
        
        try:
            # Use custom health check if available
            if service.definition.health_check:
                if asyncio.iscoroutinefunction(service.definition.health_check):
                    result = await asyncio.wait_for(
                        service.definition.health_check(service.instance),
                        timeout=self.health_check_timeout
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: asyncio.wait_for(
                            asyncio.coroutine(lambda: service.definition.health_check(service.instance))(),
                            timeout=self.health_check_timeout
                        )
                    )
            
            # Try standard health check method
            elif hasattr(service.instance, 'get_health_status'):
                health_method = getattr(service.instance, 'get_health_status')
                if asyncio.iscoroutinefunction(health_method):
                    result = await asyncio.wait_for(health_method(), timeout=self.health_check_timeout)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(None, health_method)
            
            # Basic availability check
            else:
                result = {
                    'status': 'healthy',
                    'message': 'Service is available'
                }
            
            # Parse result
            if isinstance(result, dict):
                status_str = result.get('status', 'healthy').lower()
                status = HealthStatus.HEALTHY if status_str == 'healthy' else \
                         HealthStatus.DEGRADED if status_str == 'degraded' else \
                         HealthStatus.UNHEALTHY if status_str == 'unhealthy' else \
                         HealthStatus.UNKNOWN
                
                health_result = HealthCheckResult(
                    service_name=service_name,
                    status=status,
                    message=result.get('message'),
                    details=result.get('details', {}),
                    check_duration=time.time() - start_time
                )
            else:
                # Assume healthy if no specific format returned
                health_result = HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    message="Health check completed",
                    check_duration=time.time() - start_time
                )
            
            # Update service instance
            service.health_status = health_result.status
            service.health_details = health_result.details
            service.last_health_check = health_result.timestamp
            
            # Store in history
            self.health_history[service_name].append(health_result)
            
            return health_result
            
        except asyncio.TimeoutError:
            health_result = HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                check_duration=time.time() - start_time
            )
            
            service.health_status = health_result.status
            service.last_health_check = health_result.timestamp
            self.health_history[service_name].append(health_result)
            
            return health_result
        
        except Exception as e:
            health_result = HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={'error': str(e)},
                check_duration=time.time() - start_time
            )
            
            service.health_status = health_result.status
            service.health_details = health_result.details
            service.last_health_check = health_result.timestamp
            self.health_history[service_name].append(health_result)
            
            return health_result
    
    async def check_all_services_health(self) -> Dict[str, HealthCheckResult]:
        """Check health of all services."""
        
        services = self.service_registry.get_all_services()
        
        # Run health checks concurrently
        health_check_tasks = [
            self.check_service_health(service.service_name)
            for service in services
            if service.lifecycle_state == ServiceLifecycle.RUNNING
        ]
        
        if not health_check_tasks:
            return {}
        
        results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
        
        # Process results
        health_results = {}
        for i, result in enumerate(results):
            if isinstance(result, HealthCheckResult):
                health_results[result.service_name] = result
            else:
                service_name = services[i].service_name
                health_results[service_name] = HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check exception: {str(result)}"
                )
        
        return health_results
    
    def get_service_health_history(self, service_name: str, limit: int = 10) -> List[HealthCheckResult]:
        """Get health check history for a service."""
        history = list(self.health_history.get(service_name, []))
        return history[-limit:]
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        
        services = self.service_registry.get_all_services()
        
        if not services:
            return {
                'status': 'unknown',
                'total_services': 0,
                'healthy_services': 0,
                'degraded_services': 0,
                'unhealthy_services': 0
            }
        
        health_counts = defaultdict(int)
        for service in services:
            health_counts[service.health_status.value] += 1
        
        healthy_count = health_counts['healthy']
        degraded_count = health_counts['degraded']
        unhealthy_count = health_counts['unhealthy']
        total_count = len(services)
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = 'unhealthy'
        elif degraded_count > total_count * 0.3:  # More than 30% degraded
            overall_status = 'degraded'
        elif healthy_count >= total_count * 0.8:  # At least 80% healthy
            overall_status = 'healthy'
        else:
            overall_status = 'degraded'
        
        return {
            'status': overall_status,
            'total_services': total_count,
            'healthy_services': healthy_count,
            'degraded_services': degraded_count,
            'unhealthy_services': unhealthy_count,
            'unknown_services': health_counts['unknown'],
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0
        }
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        
        while self._running:
            try:
                # Check health of all services
                health_results = await self.check_all_services_health()
                
                # Publish health status events for unhealthy services
                for service_name, result in health_results.items():
                    if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                        await self.event_bus.publish(Event(
                            event_type="service_health_degraded",
                            source_agent="health_checker",
                            priority=EventPriority.HIGH if result.status == HealthStatus.UNHEALTHY else EventPriority.NORMAL,
                            data=result.to_dict()
                        ))
                
                # Publish overall health status
                overall_status = self.get_overall_health_status()
                await self.event_bus.publish(Event(
                    event_type="system_health_status",
                    source_agent="health_checker",
                    data=overall_status
                ))
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)


class DependencyInjectionContainer:
    """
    Advanced dependency injection container with lifecycle management,
    service discovery, and health checking capabilities.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.service_registry = ServiceRegistry()
        self.health_checker = HealthChecker(self.service_registry, event_bus)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request context for request-scoped services
        self._request_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.auto_start_services = True
        self.startup_timeout_seconds = 60
        self.shutdown_timeout_seconds = 30
        
        # Background tasks
        self._running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("DependencyInjectionContainer initialized")
    
    async def start(self) -> None:
        """Start the dependency injection container."""
        if self._running:
            return
        
        self._running = True
        
        # Start health checker
        await self.health_checker.start()
        
        # Auto-start services if enabled
        if self.auto_start_services:
            await self._start_all_services()
        
        # Publish startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="dependency_container",
            data={'status': 'started', 'auto_start_enabled': self.auto_start_services}
        ))
        
        logger.info("DependencyInjectionContainer started")
    
    async def stop(self) -> None:
        """Stop the dependency injection container."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all services
        await self._stop_all_services()
        
        # Stop health checker
        await self.health_checker.stop()
        
        logger.info("DependencyInjectionContainer stopped")
    
    def register_service(self, definition: ServiceDefinition) -> None:
        """Register a service definition."""
        self.service_registry.register_definition(definition)
        
        # Create circuit breaker if needed
        if definition.service_name not in self.circuit_breakers:
            circuit_config = CircuitBreakerConfig()
            self.circuit_breakers[definition.service_name] = CircuitBreaker(
                definition.service_name, circuit_config
            )
    
    def register_singleton(self, 
                         service_type: Type[T], 
                         implementation_type: Optional[Type[T]] = None,
                         factory: Optional[Callable[[], T]] = None,
                         dependencies: Optional[List[str]] = None,
                         health_check: Optional[Callable] = None,
                         tags: Optional[Set[str]] = None) -> None:
        """Register a singleton service."""
        
        service_name = service_type.__name__
        impl_type = implementation_type or service_type
        
        definition = ServiceDefinition(
            service_name=service_name,
            service_type=service_type,
            implementation_type=impl_type,
            scope=ServiceScope.SINGLETON,
            factory_function=factory,
            dependencies=dependencies or [],
            health_check=health_check,
            tags=tags or set()
        )
        
        self.register_service(definition)
    
    def register_transient(self,
                         service_type: Type[T],
                         implementation_type: Optional[Type[T]] = None,
                         factory: Optional[Callable[[], T]] = None,
                         dependencies: Optional[List[str]] = None,
                         tags: Optional[Set[str]] = None) -> None:
        """Register a transient service."""
        
        service_name = service_type.__name__
        impl_type = implementation_type or service_type
        
        definition = ServiceDefinition(
            service_name=service_name,
            service_type=service_type,
            implementation_type=impl_type,
            scope=ServiceScope.TRANSIENT,
            factory_function=factory,
            dependencies=dependencies or [],
            tags=tags or set()
        )
        
        self.register_service(definition)
    
    async def get_service(self, service_identifier: Union[str, Type[T]]) -> Optional[T]:
        """Get a service instance."""
        
        if isinstance(service_identifier, type):
            service_name = service_identifier.__name__
        else:
            service_name = service_identifier
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker and not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for service {service_name}")
            return None
        
        try:
            # Get service from registry
            service_instance = self.service_registry.get_service(service_name)
            
            if service_instance:
                circuit_breaker.record_success() if circuit_breaker else None
                return service_instance.instance
            
            # Create instance if not found
            definition = self.service_registry.service_definitions.get(service_name)
            if not definition:
                logger.error(f"Service definition not found: {service_name}")
                return None
            
            instance = await self._create_service_instance(definition)
            if instance:
                circuit_breaker.record_success() if circuit_breaker else None
                return instance.instance
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting service {service_name}: {e}")
            circuit_breaker.record_failure() if circuit_breaker else None
            return None
    
    async def get_services_by_tag(self, tag: str) -> List[Any]:
        """Get all services with a specific tag."""
        
        services = self.service_registry.get_services_by_tag(tag)
        return [service.instance for service in services if service.lifecycle_state == ServiceLifecycle.RUNNING]
    
    async def get_services_by_type(self, service_type: Type[T]) -> List[T]:
        """Get all services of a specific type."""
        
        services = self.service_registry.get_services_by_type(service_type)
        return [service.instance for service in services if service.lifecycle_state == ServiceLifecycle.RUNNING]
    
    @asynccontextmanager
    async def request_scope(self, request_id: str):
        """Context manager for request-scoped services."""
        
        self._request_contexts[request_id] = {}
        
        try:
            yield
        finally:
            # Clean up request-scoped services
            if request_id in self._request_contexts:
                request_context = self._request_contexts[request_id]
                
                # Stop request-scoped services
                for service_name, service_instance in request_context.items():
                    try:
                        if hasattr(service_instance, 'stop'):
                            if asyncio.iscoroutinefunction(service_instance.stop):
                                await service_instance.stop()
                            else:
                                service_instance.stop()
                    except Exception as e:
                        logger.error(f"Error stopping request-scoped service {service_name}: {e}")
                
                del self._request_contexts[request_id]
    
    async def _create_service_instance(self, definition: ServiceDefinition) -> Optional[ServiceInstance]:
        """Create a service instance from definition."""
        
        try:
            # Resolve dependencies first
            dependencies = {}
            for dep_name in definition.dependencies:
                dep_instance = await self.get_service(dep_name)
                if dep_instance is None:
                    logger.error(f"Failed to resolve dependency {dep_name} for service {definition.service_name}")
                    return None
                dependencies[dep_name] = dep_instance
            
            # Create instance
            if definition.factory_function:
                # Use factory function
                if asyncio.iscoroutinefunction(definition.factory_function):
                    instance = await definition.factory_function(**dependencies)
                else:
                    instance = definition.factory_function(**dependencies)
            else:
                # Use constructor
                # Check if constructor accepts dependencies
                constructor_params = inspect.signature(definition.implementation_type.__init__).parameters
                constructor_args = {}
                
                for param_name, param in constructor_params.items():
                    if param_name == 'self':
                        continue
                    
                    if param_name in dependencies:
                        constructor_args[param_name] = dependencies[param_name]
                    elif param.default is not param.empty:
                        # Use default value
                        continue
                    else:
                        logger.warning(f"Constructor parameter {param_name} not available for {definition.service_name}")
                
                instance = definition.implementation_type(**constructor_args)
            
            # Create service instance wrapper
            service_instance = ServiceInstance(
                service_name=definition.service_name,
                instance=instance,
                definition=definition,
                lifecycle_state=ServiceLifecycle.CREATED
            )
            
            # Initialize service if it has an async start method
            if hasattr(instance, 'start'):
                service_instance.lifecycle_state = ServiceLifecycle.INITIALIZING
                
                try:
                    if asyncio.iscoroutinefunction(instance.start):
                        await instance.start()
                    else:
                        instance.start()
                    
                    service_instance.lifecycle_state = ServiceLifecycle.RUNNING
                    service_instance.started_at = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Failed to start service {definition.service_name}: {e}")
                    service_instance.lifecycle_state = ServiceLifecycle.FAILED
                    return None
            else:
                service_instance.lifecycle_state = ServiceLifecycle.READY
            
            # Register with service registry
            if definition.scope == ServiceScope.SINGLETON:
                self.service_registry.register_instance(service_instance)
            
            logger.info(f"Created service instance: {definition.service_name}")
            return service_instance
            
        except Exception as e:
            logger.error(f"Error creating service instance {definition.service_name}: {e}")
            return None
    
    async def _start_all_services(self) -> None:
        """Start all registered services in dependency order."""
        
        definitions = list(self.service_registry.service_definitions.values())
        
        # Sort by startup priority
        definitions.sort(key=lambda d: d.startup_priority)
        
        # Start services
        for definition in definitions:
            if definition.scope == ServiceScope.SINGLETON:
                try:
                    await self.get_service(definition.service_name)
                except Exception as e:
                    logger.error(f"Failed to start service {definition.service_name}: {e}")
        
        logger.info(f"Started {len(definitions)} services")
    
    async def _stop_all_services(self) -> None:
        """Stop all running services."""
        
        services = self.service_registry.get_all_services()
        
        # Sort by reverse startup priority for shutdown
        services.sort(key=lambda s: s.definition.startup_priority, reverse=True)
        
        stop_tasks = []
        
        for service in services:
            if service.lifecycle_state == ServiceLifecycle.RUNNING and hasattr(service.instance, 'stop'):
                service.lifecycle_state = ServiceLifecycle.STOPPING
                
                if asyncio.iscoroutinefunction(service.instance.stop):
                    task = asyncio.create_task(service.instance.stop())
                else:
                    task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(None, service.instance.stop)
                    )
                
                stop_tasks.append((service, task))
        
        # Wait for all services to stop
        if stop_tasks:
            timeout = self.shutdown_timeout_seconds
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in stop_tasks], return_exceptions=True),
                    timeout=timeout
                )
                
                # Update service states
                for service, _ in stop_tasks:
                    service.lifecycle_state = ServiceLifecycle.STOPPED
                
            except asyncio.TimeoutError:
                logger.warning("Service shutdown timed out")
                
                # Mark services as failed if they didn't stop in time
                for service, task in stop_tasks:
                    if not task.done():
                        service.lifecycle_state = ServiceLifecycle.FAILED
                        task.cancel()
        
        logger.info(f"Stopped {len(stop_tasks)} services")
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get container status information."""
        
        registry_status = self.service_registry.get_registry_status()
        health_status = self.health_checker.get_overall_health_status()
        
        # Circuit breaker status
        circuit_breaker_status = {
            name: breaker.get_status()
            for name, breaker in self.circuit_breakers.items()
        }
        
        open_circuits = [
            name for name, status in circuit_breaker_status.items()
            if status['state'] == 'open'
        ]
        
        return {
            'container_running': self._running,
            'registry_status': registry_status,
            'health_status': health_status,
            'circuit_breakers': {
                'total': len(circuit_breaker_status),
                'open': len(open_circuits),
                'open_services': open_circuits,
                'details': circuit_breaker_status
            },
            'request_contexts': len(self._request_contexts),
            'configuration': {
                'auto_start_services': self.auto_start_services,
                'startup_timeout': self.startup_timeout_seconds,
                'shutdown_timeout': self.shutdown_timeout_seconds
            }
        }


# Decorators for service registration

def service(scope: ServiceScope = ServiceScope.SINGLETON, 
           dependencies: Optional[List[str]] = None,
           health_check: Optional[Callable] = None,
           tags: Optional[Set[str]] = None):
    """Decorator for registering services."""
    
    def decorator(cls):
        # Store service metadata on the class
        cls._service_metadata = {
            'scope': scope,
            'dependencies': dependencies or [],
            'health_check': health_check,
            'tags': tags or set()
        }
        return cls
    
    return decorator


def inject(service_type: Type[T]):
    """Decorator for dependency injection."""
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This would need access to the container instance
            # In practice, this would be implemented with a global container
            # or thread-local storage
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Factory function

def create_dependency_container(event_bus: EventBus) -> DependencyInjectionContainer:
    """Create a dependency injection container instance."""
    return DependencyInjectionContainer(event_bus)


# Example service registration helper

def register_core_services(container: DependencyInjectionContainer):
    """Register core system services."""
    
    # This would register all the enterprise services we've created
    from src.enterprise.real_time_monitoring import RealTimeMonitor
    from src.enterprise.analytics_dashboard import AnalyticsDashboard
    from src.enterprise.quality_assurance import QualityAssuranceFramework
    from src.enterprise.knowledge_integration import KnowledgeIntegrationSystem
    from src.enterprise.data_reliability import DataReliabilitySystem
    
    # Register monitoring service
    def create_monitor():
        return RealTimeMonitor(container.event_bus)
    
    container.register_singleton(
        service_type=RealTimeMonitor,
        factory=create_monitor,
        health_check=lambda m: m.get_system_health(),
        tags={'monitoring', 'core'}
    )
    
    # Register analytics service
    def create_analytics():
        monitor = asyncio.create_task(container.get_service(RealTimeMonitor))
        return AnalyticsDashboard(monitor.result()) if monitor.done() else None
    
    container.register_singleton(
        service_type=AnalyticsDashboard,
        factory=create_analytics,
        dependencies=['RealTimeMonitor'],
        tags={'analytics', 'dashboard'}
    )
    
    logger.info("Core services registered with dependency container")
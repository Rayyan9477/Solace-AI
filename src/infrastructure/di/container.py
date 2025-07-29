"""
Dependency Injection Container implementation.

This module provides a powerful dependency injection container that
supports different lifecycle management strategies and automatic
dependency resolution.
"""

from typing import Dict, Any, Type, TypeVar, Callable, Optional, List, Union
from abc import ABC, abstractmethod
from enum import Enum
import inspect
import asyncio
from dataclasses import dataclass
import threading

T = TypeVar('T')


class LifecycleType(Enum):
    """Enum for different lifecycle management types."""
    TRANSIENT = "transient"  # New instance every time
    SINGLETON = "singleton"  # Single instance for container lifetime
    SCOPED = "scoped"       # Single instance per scope


@dataclass
class ServiceRegistration:
    """Represents a service registration in the container."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifecycle: LifecycleType = LifecycleType.TRANSIENT
    initialized: bool = False


class Injectable(ABC):
    """
    Base class for services that can be injected.
    
    Services implementing this interface can be automatically
    registered and resolved by the DI container.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources."""
        pass


class Singleton:
    """
    Marker class for singleton services.
    
    Services inheriting from this class will be automatically
    registered as singletons in the DI container.
    """
    pass


class DIContainer:
    """
    Dependency Injection Container.
    
    Provides service registration, resolution, and lifecycle management
    with support for different dependency injection patterns.
    """
    
    def __init__(self):
        """Initialize the DI container."""
        self._services: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._initialization_lock = threading.Lock()
        self._shutdown_handlers: List[Callable] = []
    
    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None
    ) -> 'DIContainer':
        """
        Register a transient service (new instance every time).
        
        Args:
            service_type: The service interface/type
            implementation_type: The concrete implementation
            factory: Optional factory function
            
        Returns:
            Self for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifecycle=LifecycleType.TRANSIENT
        )
        return self
    
    def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        instance: Optional[T] = None
    ) -> 'DIContainer':
        """
        Register a singleton service (single instance for container lifetime).
        
        Args:
            service_type: The service interface/type
            implementation_type: The concrete implementation
            factory: Optional factory function
            instance: Pre-created instance
            
        Returns:
            Self for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifecycle=LifecycleType.SINGLETON
        )
        return self
    
    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None
    ) -> 'DIContainer':
        """
        Register a scoped service (single instance per scope).
        
        Args:
            service_type: The service interface/type
            implementation_type: The concrete implementation
            factory: Optional factory function
            
        Returns:
            Self for method chaining
        """
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifecycle=LifecycleType.SCOPED
        )
        return self
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'DIContainer':
        """
        Register a pre-created instance as a singleton.
        
        Args:
            service_type: The service type
            instance: The instance to register
            
        Returns:
            Self for method chaining
        """
        return self.register_singleton(service_type, instance=instance)
    
    async def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance.
        
        Args:
            service_type: The service type to resolve
            
        Returns:
            Instance of the requested service
            
        Raises:
            ValueError: If service is not registered
            RuntimeError: If circular dependency is detected
        """
        if service_type not in self._services:
            raise ValueError(f"Service type {service_type} is not registered")
        
        registration = self._services[service_type]
        
        # Handle different lifecycle types
        if registration.lifecycle == LifecycleType.SINGLETON:
            return await self._resolve_singleton(registration)
        elif registration.lifecycle == LifecycleType.SCOPED:
            return await self._resolve_scoped(registration)
        else:  # TRANSIENT
            return await self._create_instance(registration)
    
    async def _resolve_singleton(self, registration: ServiceRegistration) -> Any:
        """Resolve a singleton service."""
        service_type = registration.service_type
        
        if service_type in self._instances:
            return self._instances[service_type]
        
        with self._initialization_lock:
            # Double-check locking pattern
            if service_type in self._instances:
                return self._instances[service_type]
            
            # Create and cache the instance
            if registration.instance is not None:
                instance = registration.instance
            else:
                instance = await self._create_instance(registration)
            
            self._instances[service_type] = instance
            return instance
    
    async def _resolve_scoped(self, registration: ServiceRegistration) -> Any:
        """Resolve a scoped service."""
        if self._current_scope is None:
            raise RuntimeError("No active scope. Use create_scope() context manager.")
        
        service_type = registration.service_type
        scope_instances = self._scoped_instances.get(self._current_scope, {})
        
        if service_type in scope_instances:
            return scope_instances[service_type]
        
        # Create new instance for this scope
        instance = await self._create_instance(registration)
        
        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}
        
        self._scoped_instances[self._current_scope][service_type] = instance
        return instance
    
    async def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create a new instance of a service."""
        if registration.factory:
            # Use factory function
            instance = registration.factory()
            if asyncio.iscoroutine(instance):
                instance = await instance
        elif registration.implementation_type:
            # Create instance using constructor injection
            instance = await self._create_with_dependencies(registration.implementation_type)
        else:
            # Use service type directly
            instance = await self._create_with_dependencies(registration.service_type)
        
        # Initialize if it's an Injectable
        if isinstance(instance, Injectable) and not registration.initialized:
            await instance.initialize()
            registration.initialized = True
        
        return instance
    
    async def _create_with_dependencies(self, service_type: Type) -> Any:
        """Create instance with automatic dependency injection."""
        # Get constructor signature
        signature = inspect.signature(service_type.__init__)
        dependencies = {}
        
        # Resolve dependencies
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                # Try to resolve the dependency
                try:
                    dependency = await self.resolve(param.annotation)
                    dependencies[param_name] = dependency
                except ValueError:
                    # If dependency not registered and has default, use default
                    if param.default != inspect.Parameter.empty:
                        dependencies[param_name] = param.default
                    else:
                        raise ValueError(
                            f"Cannot resolve dependency {param.annotation} for {service_type}"
                        )
        
        return service_type(**dependencies)
    
    def create_scope(self, scope_name: Optional[str] = None):
        """
        Create a new dependency scope.
        
        Args:
            scope_name: Optional name for the scope
            
        Returns:
            Scope context manager
        """
        return DIScope(self, scope_name)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services
    
    def get_registration_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """Get registration information for a service type."""
        if service_type not in self._services:
            return None
        
        registration = self._services[service_type]
        return {
            'service_type': registration.service_type.__name__,
            'implementation_type': registration.implementation_type.__name__ if registration.implementation_type else None,
            'lifecycle': registration.lifecycle.value,
            'has_factory': registration.factory is not None,
            'has_instance': registration.instance is not None,
            'initialized': registration.initialized
        }
    
    def get_all_registrations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered services."""
        return {
            service_type.__name__: self.get_registration_info(service_type)
            for service_type in self._services.keys()
        }
    
    async def initialize_all(self) -> bool:
        """Initialize all singleton services."""
        success = True
        
        for service_type, registration in self._services.items():
            if registration.lifecycle == LifecycleType.SINGLETON:
                try:
                    await self.resolve(service_type)
                except Exception as e:
                    print(f"Failed to initialize {service_type}: {e}")
                    success = False
        
        return success
    
    async def shutdown_all(self) -> None:
        """Shutdown all services and cleanup resources."""
        # Shutdown singletons
        for instance in self._instances.values():
            if isinstance(instance, Injectable):
                try:
                    await instance.shutdown()
                except Exception as e:
                    print(f"Error shutting down {type(instance)}: {e}")
        
        # Shutdown scoped instances
        for scope_instances in self._scoped_instances.values():
            for instance in scope_instances.values():
                if isinstance(instance, Injectable):
                    try:
                        await instance.shutdown()
                    except Exception as e:
                        print(f"Error shutting down {type(instance)}: {e}")
        
        # Call shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                print(f"Error in shutdown handler: {e}")
        
        # Clear all instances
        self._instances.clear()
        self._scoped_instances.clear()
    
    def add_shutdown_handler(self, handler: Callable) -> None:
        """Add a handler to be called during shutdown."""
        self._shutdown_handlers.append(handler)


class DIScope:
    """
    Context manager for dependency injection scopes.
    
    Provides a way to create scoped instances that are shared
    within the scope but cleaned up when the scope ends.
    """
    
    def __init__(self, container: DIContainer, scope_name: Optional[str] = None):
        """Initialize the scope."""
        self.container = container
        self.scope_name = scope_name or f"scope_{id(self)}"
        self.previous_scope = None
    
    async def __aenter__(self):
        """Enter the scope."""
        self.previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_name
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the scope and cleanup scoped instances."""
        # Shutdown scoped instances
        if self.scope_name in self.container._scoped_instances:
            scope_instances = self.container._scoped_instances[self.scope_name]
            
            for instance in scope_instances.values():
                if isinstance(instance, Injectable):
                    try:
                        await instance.shutdown()
                    except Exception as e:
                        print(f"Error shutting down scoped instance: {e}")
            
            # Remove the scope
            del self.container._scoped_instances[self.scope_name]
        
        # Restore previous scope
        self.container._current_scope = self.previous_scope


# Global container instance
_global_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    return _global_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    _global_container = DIContainer()
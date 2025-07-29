"""
Decorators for dependency injection.

This module provides decorators to simplify dependency injection
and service registration.
"""

from typing import Type, TypeVar, Callable, Any, Optional
from functools import wraps
import asyncio

from .container import get_container, DIContainer

T = TypeVar('T')


def inject(service_type: Type[T]) -> Callable[[Callable], Callable]:
    """
    Decorator to inject a service into a function parameter.
    
    Args:
        service_type: The type of service to inject
        
    Returns:
        Decorator function
        
    Example:
        @inject(MyService)
        async def my_function(service: MyService):
            return service.do_something()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            container = get_container()
            service = await container.resolve(service_type)
            
            # Add service to kwargs if not already provided
            if 'service' not in kwargs:
                kwargs['service'] = service
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def provides(
    service_type: Type[T],
    lifecycle: str = "transient",
    container: Optional[DIContainer] = None
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to automatically register a class as a service provider.
    
    Args:
        service_type: The service interface/type this class provides
        lifecycle: Lifecycle management ("transient", "singleton", "scoped")
        container: Optional specific container to register with
        
    Returns:
        The decorated class
        
    Example:
        @provides(IMyService, lifecycle="singleton")
        class MyService(IMyService):
            def do_something(self):
                return "Hello"
    """
    def decorator(cls: Type[T]) -> Type[T]:
        target_container = container or get_container()
        
        if lifecycle == "singleton":
            target_container.register_singleton(service_type, cls)
        elif lifecycle == "scoped":
            target_container.register_scoped(service_type, cls)
        else:  # transient
            target_container.register_transient(service_type, cls)
        
        return cls
    
    return decorator


def auto_wire(container: Optional[DIContainer] = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to automatically wire dependencies for a class.
    
    This decorator modifies the class __init__ method to automatically
    resolve dependencies from the DI container.
    
    Args:
        container: Optional specific container to use
        
    Returns:
        The decorated class with auto-wired dependencies
        
    Example:
        @auto_wire()
        class MyService:
            def __init__(self, logger: ILogger, db: IDatabase):
                self.logger = logger
                self.db = db
    """
    def decorator(cls: Type[T]) -> Type[T]:
        original_init = cls.__init__
        target_container = container or get_container()
        
        @wraps(original_init)
        async def new_init(self, *args, **kwargs):
            # Get constructor signature
            import inspect
            signature = inspect.signature(original_init)
            resolved_kwargs = kwargs.copy()
            
            # Resolve dependencies for parameters not provided
            for param_name, param in signature.parameters.items():
                if param_name == 'self' or param_name in kwargs:
                    continue
                
                if param.annotation != inspect.Parameter.empty:
                    try:
                        dependency = await target_container.resolve(param.annotation)
                        resolved_kwargs[param_name] = dependency
                    except ValueError:
                        # If dependency not registered and has default, use default
                        if param.default != inspect.Parameter.empty:
                            resolved_kwargs[param_name] = param.default
                        # Otherwise, let the original constructor handle the missing parameter
            
            # Call original constructor
            if asyncio.iscoroutinefunction(original_init):
                await original_init(self, *args, **resolved_kwargs)
            else:
                original_init(self, *args, **resolved_kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def service(
    interface: Optional[Type] = None,
    lifecycle: str = "transient",
    container: Optional[DIContainer] = None
) -> Callable[[Type[T]], Type[T]]:
    """
    Convenience decorator that combines @provides and @auto_wire.
    
    Args:
        interface: The service interface (if different from class)
        lifecycle: Lifecycle management ("transient", "singleton", "scoped")
        container: Optional specific container to register with
        
    Returns:
        The decorated class
        
    Example:
        @service(IMyService, lifecycle="singleton")
        class MyService(IMyService):
            def __init__(self, logger: ILogger):
                self.logger = logger
    """
    def decorator(cls: Type[T]) -> Type[T]:
        service_type = interface or cls
        
        # Apply auto_wire first
        cls = auto_wire(container)(cls)
        
        # Then apply provides
        cls = provides(service_type, lifecycle, container)(cls)
        
        return cls
    
    return decorator


def factory(
    service_type: Type[T],
    lifecycle: str = "transient",
    container: Optional[DIContainer] = None
) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """
    Decorator to register a factory function for a service.
    
    Args:
        service_type: The service type the factory creates
        lifecycle: Lifecycle management ("transient", "singleton", "scoped")
        container: Optional specific container to register with
        
    Returns:
        The decorated factory function
        
    Example:
        @factory(IMyService, lifecycle="singleton")
        def create_my_service() -> IMyService:
            return MyService(config=load_config())
    """
    def decorator(factory_func: Callable[[], T]) -> Callable[[], T]:
        target_container = container or get_container()
        
        if lifecycle == "singleton":
            target_container.register_singleton(service_type, factory=factory_func)
        elif lifecycle == "scoped":
            target_container.register_scoped(service_type, factory=factory_func)
        else:  # transient
            target_container.register_transient(service_type, factory=factory_func)
        
        return factory_func
    
    return decorator
"""
Base module system for the Contextual-Chatbot application.

This module provides the foundation for creating reusable, 
pluggable components with dependency management.
"""

import asyncio
import inspect
from typing import Dict, Any, List, Optional, Callable, Set, Type, Union
import time
from abc import ABC, abstractmethod
import logging
import importlib
import pkgutil
import sys
from pathlib import Path

from src.utils.logger import get_logger

# Singleton module manager instance
_module_manager = None


class Module(ABC):
    """
    Base module class for all application components.
    
    A module is a self-contained unit of functionality that can
    depend on other modules and expose services to them.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize a module with configuration"""
        self.module_id = module_id
        self.config = config or {}
        self.initialized = False
        self.logger = get_logger(f"module.{module_id}", {"module_id": module_id})
        self.dependencies = {}
        self.services = {}
        self.type_id = self.__class__.__name__
        self.health_status = "uninitialized"
        self.start_time = None
        self.logger.debug(f"Module {module_id} created")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the module and its resources.
        
        This method should be overridden by each module implementation
        to set up required resources and connections.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        self.start_time = time.time()
        self.logger.debug(f"Initializing module {self.module_id}")
        self.initialized = True
        self.health_status = "operational"
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the module and cleanup resources.
        
        This method should be overridden by each module implementation
        to properly release resources on shutdown.
        
        Returns:
            bool: True if shutdown succeeded, False otherwise
        """
        self.logger.debug(f"Shutting down module {self.module_id}")
        self.initialized = False
        self.health_status = "shutdown"
        return True
    
    def add_dependency(self, module_id: str, module: 'Module') -> None:
        """Add a dependency to this module"""
        self.dependencies[module_id] = module
        self.logger.debug(f"Added dependency on {module_id}")
    
    def get_dependency(self, module_id: str) -> Optional['Module']:
        """Get a dependency module by ID"""
        return self.dependencies.get(module_id)
    
    def expose_service(self, service_name: str, service_func: Callable) -> None:
        """
        Expose a service function to be used by other modules.
        
        Args:
            service_name: Name of the service
            service_func: Function implementing the service
        """
        self.services[service_name] = service_func
        self.logger.debug(f"Exposed service: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Callable]:
        """Get a service function by name"""
        return self.services.get(service_name)
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of this module.
        
        This method can be overridden to provide detailed health information
        specific to the module's functionality.
        
        Returns:
            A dictionary with health status information
        """
        uptime = time.time() - (self.start_time or time.time())
        
        health_data = {
            "status": self.health_status,
            "module_id": self.module_id,
            "type_id": self.type_id,
            "initialized": self.initialized,
            "uptime_seconds": int(uptime) if self.start_time else 0,
            "dependency_count": len(self.dependencies),
            "service_count": len(self.services)
        }
        
        return health_data


class ModuleManager:
    """
    Manages the lifecycle of all modules in the application.
    
    The module manager handles creation, initialization, 
    dependency resolution, and shutdown of modules.
    """
    
    def __init__(self):
        """Initialize the module manager"""
        self.modules: Dict[str, Module] = {}
        self.module_types: Dict[str, Type[Module]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.logger = get_logger("module_manager")
        self.logger.debug("Module manager created")
    
    def register_module_type(self, module_class: Type[Module]) -> None:
        """
        Register a module class for use in the application.
        
        Args:
            module_class: A class derived from Module
        """
        type_id = module_class.__name__
        self.module_types[type_id] = module_class
        self.logger.debug(f"Registered module type: {type_id}")
    
    def discover_module_types(self, package_name: str) -> int:
        """
        Discover and register module types from a package.
        
        Args:
            package_name: Name of the package to scan for modules
            
        Returns:
            Number of module types discovered
        """
        try:
            package = importlib.import_module(package_name)
            discovered_count = 0
            
            # Handle case where package doesn't have __path__
            if not hasattr(package, '__path__'):
                self.logger.error(f"Package {package_name} does not have a __path__ attribute")
                return 0
                
            for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if is_pkg:
                    # Recursively scan subpackages
                    try:
                        discovered_count += self.discover_module_types(name)
                    except Exception as e:
                        self.logger.error(f"Error scanning subpackage {name}: {str(e)}")
                else:
                    try:
                        # Import the module
                        module = importlib.import_module(name)
                        
                        # Find Module subclasses in the module
                        for attr_name in dir(module):
                            try:
                                attr = getattr(module, attr_name)
                                
                                # Check if it's a class and a Module subclass (but not Module itself)
                                if (inspect.isclass(attr) and 
                                    issubclass(attr, Module) and 
                                    attr is not Module):
                                    self.register_module_type(attr)
                                    discovered_count += 1
                                    self.logger.debug(f"Discovered module type: {attr.__name__} in {name}")
                            except Exception as e:
                                self.logger.error(f"Error processing attribute {attr_name} in {name}: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Error discovering modules in {name}: {str(e)}")
            
            return discovered_count
        except ModuleNotFoundError:
            self.logger.warning(f"Package {package_name} not found")
            return 0
        except ImportError as e:
            self.logger.error(f"Could not import package {package_name}: {str(e)}")
            return 0
        except Exception as e:
            self.logger.error(f"Unexpected error discovering modules in {package_name}: {str(e)}")
            return 0
    
    def create_module(self, type_id: str, module_id: str, 
                     config: Dict[str, Any] = None) -> Optional[Module]:
        """
        Create a module instance of the specified type.
        
        Args:
            type_id: The type ID of the module class
            module_id: Unique ID for the module instance
            config: Configuration dictionary for the module
            
        Returns:
            Module instance or None if creation failed
        """
        if module_id in self.modules:
            self.logger.warning(f"Module with ID {module_id} already exists")
            return self.modules[module_id]
        
        if type_id not in self.module_types:
            self.logger.error(f"Unknown module type: {type_id}")
            return None
        
        try:
            # Create the module instance
            module_class = self.module_types[type_id]
            module = module_class(module_id, config)
            
            # Store the module
            self.modules[module_id] = module
            self.dependency_graph[module_id] = set()
            
            self.logger.debug(f"Created module: {module_id} (type: {type_id})")
            return module
        except Exception as e:
            self.logger.error(f"Error creating module {type_id}/{module_id}: {str(e)}")
            return None
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a module by ID"""
        return self.modules.get(module_id)
    
    def add_dependency(self, module_id: str, dependency_id: str) -> bool:
        """
        Add a dependency between two modules.
        
        Args:
            module_id: ID of the dependent module
            dependency_id: ID of the module to depend on
            
        Returns:
            True if dependency was added, False otherwise
        """
        # Check that both modules exist
        if module_id not in self.modules:
            self.logger.error(f"Module {module_id} not found")
            return False
        
        if dependency_id not in self.modules:
            self.logger.error(f"Dependency module {dependency_id} not found")
            return False
        
        # Check for circular dependency
        if self._would_create_cycle(module_id, dependency_id):
            self.logger.error(f"Circular dependency detected: {module_id} -> {dependency_id}")
            return False
        
        # Add the dependency
        self.dependency_graph[module_id].add(dependency_id)
        
        # Update the module's dependency reference
        self.modules[module_id].add_dependency(
            dependency_id, self.modules[dependency_id]
        )
        
        self.logger.debug(f"Added dependency: {module_id} -> {dependency_id}")
        return True
    
    def _would_create_cycle(self, module_id: str, dependency_id: str) -> bool:
        """Check if adding a dependency would create a cycle"""
        # If the dependency already depends on the module, it would create a cycle
        visited = set()
        
        def dfs(current_id: str) -> bool:
            if current_id == module_id:
                return True
            
            if current_id in visited:
                return False
            
            visited.add(current_id)
            
            for next_id in self.dependency_graph.get(current_id, set()):
                if dfs(next_id):
                    return True
            
            return False
        
        return dfs(dependency_id)
    
    def _get_initialization_order(self) -> List[str]:
        """Determine the order in which modules should be initialized"""
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(module_id: str) -> None:
            if module_id in visited:
                return
            
            if module_id in temp_visited:
                # Circular dependency, should not happen if _would_create_cycle works
                self.logger.error(f"Circular dependency detected during initialization for {module_id}")
                return
            
            temp_visited.add(module_id)
            
            # Visit dependencies first
            for dependency_id in self.dependency_graph.get(module_id, set()):
                dfs(dependency_id)
            
            temp_visited.remove(module_id)
            visited.add(module_id)
            order.append(module_id)
        
        # Visit all modules
        for module_id in self.modules:
            if module_id not in visited:
                dfs(module_id)
        
        return order
    
    async def initialize_all(self) -> bool:
        """
        Initialize all modules in dependency order.
        
        Returns:
            True if all critical modules initialized successfully, False otherwise
        """
        initialization_order = self._get_initialization_order()
        self.logger.info(f"Initializing {len(initialization_order)} modules in dependency order")
        
        # Track critical and optional modules
        critical_modules = ["llm", "central_vector_db", "vector_store"]
        optional_modules = ["voice", "ui_manager"]
        
        all_critical_success = True
        initialized_count = 0
        failed_modules = []
        
        for module_id in initialization_order:
            module = self.modules[module_id]
            is_critical = module_id in critical_modules
            
            try:
                self.logger.debug(f"Initializing module: {module_id}")
                success = await module.initialize()
                
                if not success:
                    if is_critical:
                        self.logger.error(f"Critical module {module_id} initialization failed")
                        all_critical_success = False
                        failed_modules.append(module_id)
                    else:
                        self.logger.warning(f"Optional module {module_id} initialization failed")
                else:
                    initialized_count += 1
                    self.logger.debug(f"Module {module_id} initialized successfully")
                    
            except Exception as e:
                if is_critical:
                    self.logger.error(f"Error initializing critical module {module_id}: {str(e)}")
                    all_critical_success = False
                    failed_modules.append(module_id)
                else:
                    self.logger.warning(f"Error initializing optional module {module_id}: {str(e)}")
        
        # Log initialization summary
        if failed_modules:
            self.logger.warning(f"Failed modules: {', '.join(failed_modules)}")
        
        success_rate = f"{initialized_count}/{len(initialization_order)}"
        if all_critical_success:
            self.logger.info(f"All critical modules initialized successfully. Overall: {success_rate} modules")
        else:
            self.logger.error(f"Some critical modules failed. Overall: {success_rate} modules")
        
        return all_critical_success
    
    async def shutdown_all(self, reverse_order: bool = True) -> bool:
        """
        Shutdown all modules in reverse dependency order.
        
        Args:
            reverse_order: If True, shutdown in reverse dependency order
            
        Returns:
            True if all modules shutdown successfully, False otherwise
        """
        # Get modules in dependency order
        order = self._get_initialization_order()
        
        # Reverse for shutdown if requested
        if reverse_order:
            order.reverse()
        
        self.logger.info(f"Shutting down {len(order)} modules")
        
        all_success = True
        shutdown_count = 0
        
        for module_id in order:
            if module_id not in self.modules:
                continue
                
            module = self.modules[module_id]
            
            try:
                self.logger.debug(f"Shutting down module: {module_id}")
                success = await module.shutdown()
                
                if not success:
                    self.logger.error(f"Module {module_id} shutdown failed")
                    all_success = False
                else:
                    shutdown_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error shutting down module {module_id}: {str(e)}")
                all_success = False
        
        self.logger.info(f"Shutdown {shutdown_count}/{len(order)} modules")
        return all_success
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Perform health checks on all modules.
        
        Returns:
            Dictionary with overall health status and module statuses
        """
        health_info = {
            "modules": {},
            "overall_status": "operational",
            "total_modules": len(self.modules),
            "initialized_modules": 0
        }
        
        for module_id, module in self.modules.items():
            try:
                module_health = await module.check_health()
                health_info["modules"][module_id] = module_health
                
                # Count initialized modules
                if module.initialized:
                    health_info["initialized_modules"] += 1
                
                # If any module is not operational, overall status is degraded
                if module_health["status"] != "operational" and health_info["overall_status"] == "operational":
                    health_info["overall_status"] = "degraded"
            except Exception as e:
                self.logger.error(f"Error checking health for module {module_id}: {str(e)}")
                health_info["modules"][module_id] = {
                    "status": "error",
                    "error": str(e),
                    "module_id": module_id
                }
                health_info["overall_status"] = "degraded"
        
        # If no modules are initialized, system is non-operational
        if health_info["initialized_modules"] == 0 and health_info["total_modules"] > 0:
            health_info["overall_status"] = "non-operational"
        
        return health_info


def get_module_manager() -> ModuleManager:
    """Get the singleton module manager instance"""
    global _module_manager
    if _module_manager is None:
        _module_manager = ModuleManager()
    return _module_manager
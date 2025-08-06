"""
Diagnosis Services Registration

This module handles the registration of all diagnosis services
with the dependency injection container.
"""

import logging
from typing import Optional

from .container import DIContainer, LifecycleType
from src.services.diagnosis import (
    IDiagnosisService,
    IEnhancedDiagnosisService,
    IDiagnosisOrchestrator,
    IDiagnosisAgentAdapter,
    UnifiedDiagnosisService,
    DiagnosisOrchestrator,
    DiagnosisAgentAdapter
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def register_diagnosis_services(container: DIContainer) -> bool:
    """
    Register all diagnosis services with the DI container.
    
    Args:
        container: DI container instance
        
    Returns:
        True if registration successful, False otherwise
    """
    try:
        logger.info("Registering diagnosis services with DI container")
        
        # Register core diagnosis services
        _register_core_services(container)
        
        # Register orchestration services
        _register_orchestration_services(container)
        
        # Register adapter services  
        _register_adapter_services(container)
        
        logger.info("Successfully registered all diagnosis services")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register diagnosis services: {str(e)}")
        return False


def _register_core_services(container: DIContainer) -> None:
    """Register core diagnosis services."""
    
    # Register UnifiedDiagnosisService as singleton
    container.register_singleton(
        service_type=IDiagnosisService,
        implementation_type=UnifiedDiagnosisService
    )
    
    # Also register as enhanced service interface
    container.register_singleton(
        service_type=IEnhancedDiagnosisService,
        implementation_type=UnifiedDiagnosisService
    )
    
    # Register concrete type for direct access
    container.register_singleton(
        service_type=UnifiedDiagnosisService,
        implementation_type=UnifiedDiagnosisService
    )
    
    logger.debug("Registered core diagnosis services")


def _register_orchestration_services(container: DIContainer) -> None:
    """Register orchestration services."""
    
    # Register DiagnosisOrchestrator as singleton
    container.register_singleton(
        service_type=IDiagnosisOrchestrator,
        implementation_type=DiagnosisOrchestrator
    )
    
    # Register concrete type
    container.register_singleton(
        service_type=DiagnosisOrchestrator,
        implementation_type=DiagnosisOrchestrator
    )
    
    logger.debug("Registered orchestration services")


def _register_adapter_services(container: DIContainer) -> None:
    """Register adapter services."""
    
    # Register DiagnosisAgentAdapter as singleton
    container.register_singleton(
        service_type=IDiagnosisAgentAdapter,
        implementation_type=DiagnosisAgentAdapter
    )
    
    # Register concrete type
    container.register_singleton(
        service_type=DiagnosisAgentAdapter,
        implementation_type=DiagnosisAgentAdapter
    )
    
    logger.debug("Registered adapter services")


def create_diagnosis_orchestrator_factory(container: DIContainer):
    """
    Create a factory function for the diagnosis orchestrator with all services registered.
    
    Args:
        container: DI container instance
        
    Returns:
        Configured diagnosis orchestrator
    """
    async def factory() -> DiagnosisOrchestrator:
        try:
            # Resolve orchestrator
            orchestrator = await container.resolve(DiagnosisOrchestrator)
            
            # Register diagnosis services with orchestrator
            unified_service = await container.resolve(UnifiedDiagnosisService)
            await orchestrator.register_diagnosis_service("unified_diagnosis_service", unified_service)
            
            logger.info("Created and configured diagnosis orchestrator")
            return orchestrator
            
        except Exception as e:
            logger.error(f"Failed to create diagnosis orchestrator: {str(e)}")
            raise
    
    return factory


def setup_diagnosis_integration(container: DIContainer) -> bool:
    """
    Set up the complete diagnosis system integration.
    
    Args:
        container: DI container instance
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        logger.info("Setting up diagnosis system integration")
        
        # Register all services
        if not register_diagnosis_services(container):
            return False
        
        # Create orchestrator factory
        orchestrator_factory = create_diagnosis_orchestrator_factory(container)
        
        # Register the factory
        container.register_singleton(
            service_type=IDiagnosisOrchestrator,
            factory=orchestrator_factory
        )
        
        logger.info("Diagnosis system integration setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup diagnosis integration: {str(e)}")
        return False


async def initialize_diagnosis_services(container: DIContainer) -> bool:
    """
    Initialize all diagnosis services.
    
    Args:
        container: DI container instance
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing diagnosis services")
        
        # Initialize services in dependency order
        services_to_initialize = [
            UnifiedDiagnosisService,
            DiagnosisAgentAdapter,
            DiagnosisOrchestrator
        ]
        
        for service_type in services_to_initialize:
            try:
                service = await container.resolve(service_type)
                if hasattr(service, 'initialize'):
                    success = await service.initialize()
                    if not success:
                        logger.error(f"Failed to initialize {service_type.__name__}")
                        return False
                    logger.debug(f"Initialized {service_type.__name__}")
                else:
                    logger.debug(f"{service_type.__name__} does not require initialization")
            except Exception as e:
                logger.error(f"Error initializing {service_type.__name__}: {str(e)}")
                return False
        
        logger.info("All diagnosis services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize diagnosis services: {str(e)}")
        return False


async def validate_diagnosis_services(container: DIContainer) -> bool:
    """
    Validate that all diagnosis services are properly registered and healthy.
    
    Args:
        container: DI container instance
        
    Returns:
        True if validation successful, False otherwise
    """
    try:
        logger.info("Validating diagnosis services")
        
        # Check service registrations
        required_services = [
            IDiagnosisService,
            IEnhancedDiagnosisService,
            IDiagnosisOrchestrator,
            IDiagnosisAgentAdapter
        ]
        
        for service_type in required_services:
            if not container.is_registered(service_type):
                logger.error(f"Required service {service_type.__name__} is not registered")
                return False
        
        # Test service resolution
        try:
            unified_service = await container.resolve(IEnhancedDiagnosisService)
            orchestrator = await container.resolve(IDiagnosisOrchestrator)
            adapter = await container.resolve(IDiagnosisAgentAdapter)
            
            # Test service health
            if hasattr(unified_service, 'get_service_health'):
                health = await unified_service.get_service_health()
                if health.get("status") != "healthy":
                    logger.warning(f"Unified diagnosis service health: {health}")
            
            if hasattr(orchestrator, 'get_orchestrator_health'):
                health = await orchestrator.get_orchestrator_health()
                if health.get("orchestrator_status") not in ["healthy", "degraded"]:
                    logger.warning(f"Diagnosis orchestrator health: {health}")
            
            logger.info("Diagnosis services validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve diagnosis services: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to validate diagnosis services: {str(e)}")
        return False


def get_diagnosis_service_info(container: DIContainer) -> dict:
    """
    Get information about registered diagnosis services.
    
    Args:
        container: DI container instance
        
    Returns:
        Dictionary with service information
    """
    try:
        service_info = {
            "registered_services": [],
            "service_details": {},
            "status": "healthy"
        }
        
        # Check core services
        core_services = [
            IDiagnosisService,
            IEnhancedDiagnosisService,
            IDiagnosisOrchestrator,
            IDiagnosisAgentAdapter
        ]
        
        for service_type in core_services:
            service_name = service_type.__name__
            if container.is_registered(service_type):
                service_info["registered_services"].append(service_name)
                service_info["service_details"][service_name] = container.get_registration_info(service_type)
            else:
                service_info["status"] = "incomplete"
                logger.warning(f"Service {service_name} is not registered")
        
        return service_info
        
    except Exception as e:
        logger.error(f"Error getting diagnosis service info: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }
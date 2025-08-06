"""
Diagnosis System Integration Setup

This module provides easy setup and initialization for the complete
diagnosis system integration with the Solace-AI architecture.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from src.infrastructure.di.container import get_container
from src.infrastructure.di.diagnosis_registration import (
    setup_diagnosis_integration,
    initialize_diagnosis_services,
    validate_diagnosis_services
)
from src.utils.logger import get_logger

# Import with error handling
try:
    from src.database.central_vector_db import CentralVectorDB
    CENTRAL_VECTOR_DB_AVAILABLE = True
except ImportError:
    CentralVectorDB = None
    CENTRAL_VECTOR_DB_AVAILABLE = False

try:
    from src.memory.enhanced_memory_system import EnhancedMemorySystem
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    EnhancedMemorySystem = None
    ENHANCED_MEMORY_AVAILABLE = False

logger = get_logger(__name__)


class DiagnosisSystemIntegration:
    """
    Main class for setting up and managing diagnosis system integration.
    
    This class handles the complete setup process, dependency registration,
    service initialization, and validation of the integrated diagnosis system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the diagnosis system integration.
        
        Args:
            config: Configuration dictionary for the integration
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Integration status
        self.is_initialized = False
        self.services_registered = False
        self.services_initialized = False
        self.validation_passed = False
        
        # Service references
        self.container = None
        self.diagnosis_orchestrator = None
        self.diagnosis_adapter = None
        self.unified_service = None
        self.memory_integration = None
        
        # Integration configuration
        self.enable_memory_integration = self.config.get("enable_memory_integration", True)
        self.enable_vector_db_integration = self.config.get("enable_vector_db_integration", True)
        self.enable_compatibility_layer = self.config.get("enable_compatibility_layer", True)
        self.auto_validate = self.config.get("auto_validate", True)
    
    async def setup_complete_integration(self) -> Dict[str, Any]:
        """
        Set up the complete diagnosis system integration.
        
        Returns:
            Dictionary with setup results and status
        """
        setup_results = {
            "success": False,
            "steps_completed": [],
            "errors": [],
            "services": {},
            "integration_status": {}
        }
        
        try:
            self.logger.info("Starting complete diagnosis system integration setup")
            
            # Step 1: Get DI container
            setup_results["steps_completed"].append("container_access")
            self.container = get_container()
            
            # Step 2: Register all diagnosis services
            if await self._register_services():
                setup_results["steps_completed"].append("service_registration")
                self.services_registered = True
            else:
                setup_results["errors"].append("Failed to register services")
                return setup_results
            
            # Step 3: Initialize all services
            if await self._initialize_services():
                setup_results["steps_completed"].append("service_initialization")
                self.services_initialized = True
            else:
                setup_results["errors"].append("Failed to initialize services")
                return setup_results
            
            # Step 4: Set up memory and vector DB integration
            if await self._setup_data_integrations():
                setup_results["steps_completed"].append("data_integration_setup")
            else:
                setup_results["errors"].append("Failed to set up data integrations")
            
            # Step 5: Validate the complete system
            if self.auto_validate and await self._validate_integration():
                setup_results["steps_completed"].append("integration_validation")
                self.validation_passed = True
            else:
                setup_results["errors"].append("Integration validation failed")
            
            # Step 6: Get service references
            await self._get_service_references()
            setup_results["steps_completed"].append("service_reference_resolution")
            
            # Mark as initialized
            self.is_initialized = True
            setup_results["success"] = True
            
            # Get final status
            setup_results["integration_status"] = await self.get_integration_status()
            setup_results["services"] = await self._get_service_info()
            
            self.logger.info("Complete diagnosis system integration setup completed successfully")
            return setup_results
            
        except Exception as e:
            error_msg = f"Failed to setup diagnosis system integration: {str(e)}"
            self.logger.error(error_msg)
            setup_results["errors"].append(error_msg)
            return setup_results
    
    async def _register_services(self) -> bool:
        """Register all diagnosis services with the DI container."""
        try:
            self.logger.info("Registering diagnosis services")
            
            # Use the diagnosis registration module
            success = setup_diagnosis_integration(self.container)
            
            if success:
                self.logger.info("Successfully registered diagnosis services")
            else:
                self.logger.error("Failed to register diagnosis services")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error registering services: {str(e)}")
            return False
    
    async def _initialize_services(self) -> bool:
        """Initialize all diagnosis services."""
        try:
            self.logger.info("Initializing diagnosis services")
            
            # Use the diagnosis registration module
            success = await initialize_diagnosis_services(self.container)
            
            if success:
                self.logger.info("Successfully initialized diagnosis services")
            else:
                self.logger.error("Failed to initialize diagnosis services")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {str(e)}")
            return False
    
    async def _setup_data_integrations(self) -> bool:
        """Set up memory and vector database integrations."""
        try:
            success = True
            
            # Set up memory integration if enabled and available
            if self.enable_memory_integration and ENHANCED_MEMORY_AVAILABLE:
                memory_success = await self._setup_memory_integration()
                if not memory_success:
                    self.logger.warning("Memory integration setup failed")
                    success = False
            else:
                self.logger.info("Memory integration disabled or not available")
            
            # Set up vector DB integration if enabled and available
            if self.enable_vector_db_integration and CENTRAL_VECTOR_DB_AVAILABLE:
                vector_success = await self._setup_vector_integration()
                if not vector_success:
                    self.logger.warning("Vector DB integration setup failed")
                    success = False
            else:
                self.logger.info("Vector DB integration disabled or not available")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error setting up data integrations: {str(e)}")
            return False
    
    async def _setup_memory_integration(self) -> bool:
        """Set up memory system integration."""
        try:
            from .memory_integration import DiagnosisMemoryIntegrationService
            
            # Register memory integration service
            self.container.register_singleton(
                service_type=DiagnosisMemoryIntegrationService,
                implementation_type=DiagnosisMemoryIntegrationService
            )
            
            # Initialize the service
            memory_integration = await self.container.resolve(DiagnosisMemoryIntegrationService)
            await memory_integration.initialize()
            
            self.logger.info("Memory integration setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up memory integration: {str(e)}")
            return False
    
    async def _setup_vector_integration(self) -> bool:
        """Set up vector database integration."""
        try:
            # Vector DB integration is handled through the unified service
            # which already has vector DB support built-in
            self.logger.info("Vector DB integration setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up vector integration: {str(e)}")
            return False
    
    async def _validate_integration(self) -> bool:
        """Validate the complete integration."""
        try:
            self.logger.info("Validating diagnosis system integration")
            
            # Use the diagnosis registration module
            success = await validate_diagnosis_services(self.container)
            
            if success:
                self.logger.info("Integration validation passed")
            else:
                self.logger.error("Integration validation failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error validating integration: {str(e)}")
            return False
    
    async def _get_service_references(self) -> None:
        """Get references to the main services."""
        try:
            from .interfaces import IDiagnosisOrchestrator, IDiagnosisAgentAdapter
            from .unified_service import UnifiedDiagnosisService
            
            # Get service references
            self.diagnosis_orchestrator = await self.container.resolve(IDiagnosisOrchestrator)
            self.diagnosis_adapter = await self.container.resolve(IDiagnosisAgentAdapter)
            self.unified_service = await self.container.resolve(UnifiedDiagnosisService)
            
            # Try to get memory integration service
            try:
                from .memory_integration import DiagnosisMemoryIntegrationService
                self.memory_integration = await self.container.resolve(DiagnosisMemoryIntegrationService)
            except:
                self.memory_integration = None
            
            self.logger.debug("Service references resolved successfully")
            
        except Exception as e:
            self.logger.error(f"Error getting service references: {str(e)}")
    
    async def _get_service_info(self) -> Dict[str, Any]:
        """Get information about registered services."""
        try:
            from src.infrastructure.di.diagnosis_registration import get_diagnosis_service_info
            return get_diagnosis_service_info(self.container)
        except Exception as e:
            self.logger.error(f"Error getting service info: {str(e)}")
            return {}
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        try:
            status = {
                "integration_initialized": self.is_initialized,
                "services_registered": self.services_registered,
                "services_initialized": self.services_initialized,
                "validation_passed": self.validation_passed,
                "config": self.config,
                "service_availability": {
                    "orchestrator": self.diagnosis_orchestrator is not None,
                    "adapter": self.diagnosis_adapter is not None,
                    "unified_service": self.unified_service is not None,
                    "memory_integration": self.memory_integration is not None
                },
                "system_availability": {
                    "memory_system": ENHANCED_MEMORY_AVAILABLE,
                    "vector_db": CENTRAL_VECTOR_DB_AVAILABLE
                }
            }
            
            # Get service health if available
            if self.unified_service:
                try:
                    service_health = await self.unified_service.get_service_health()
                    status["unified_service_health"] = service_health
                except:
                    pass
            
            if self.diagnosis_orchestrator:
                try:
                    orchestrator_health = await self.diagnosis_orchestrator.get_orchestrator_health()
                    status["orchestrator_health"] = orchestrator_health
                except:
                    pass
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting integration status: {str(e)}")
            return {"error": str(e)}
    
    async def test_diagnosis(self, 
                           message: str, 
                           user_id: str = "test_user",
                           diagnosis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Test the diagnosis system with a sample request.
        
        Args:
            message: Test message
            user_id: Test user ID
            diagnosis_type: Type of diagnosis to test
            
        Returns:
            Test result
        """
        if not self.is_initialized:
            return {"error": "Integration not initialized"}
        
        try:
            from .interfaces import DiagnosisRequest, DiagnosisType
            
            # Create test request
            diagnosis_request = DiagnosisRequest(
                user_id=user_id,
                session_id=f"test_session_{int(asyncio.get_event_loop().time())}",
                message=message,
                conversation_history=[],
                diagnosis_type=DiagnosisType(diagnosis_type.lower())
            )
            
            # Perform diagnosis
            result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            return {
                "success": True,
                "test_message": message,
                "diagnosis_result": asdict(result),
                "processing_time_ms": result.processing_time_ms
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "test_message": message
            }
    
    async def shutdown_integration(self) -> bool:
        """Shutdown the integrated diagnosis system."""
        try:
            self.logger.info("Shutting down diagnosis system integration")
            
            # Shutdown services in reverse order
            if self.memory_integration:
                await self.memory_integration.shutdown()
            
            if self.unified_service:
                await self.unified_service.shutdown()
            
            if self.diagnosis_orchestrator:
                await self.diagnosis_orchestrator.shutdown()
            
            if self.diagnosis_adapter:
                await self.diagnosis_adapter.shutdown()
            
            # Shutdown container
            if self.container:
                await self.container.shutdown_all()
            
            self.is_initialized = False
            self.logger.info("Diagnosis system integration shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down integration: {str(e)}")
            return False


# Convenience functions for easy setup

async def quick_setup_diagnosis_integration(config: Dict[str, Any] = None) -> DiagnosisSystemIntegration:
    """
    Quick setup function for diagnosis system integration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized diagnosis system integration
    """
    integration = DiagnosisSystemIntegration(config)
    setup_result = await integration.setup_complete_integration()
    
    if not setup_result["success"]:
        raise RuntimeError(f"Failed to setup diagnosis integration: {setup_result['errors']}")
    
    return integration


async def test_diagnosis_integration(message: str = "I feel anxious and have trouble sleeping") -> Dict[str, Any]:
    """
    Test function for the diagnosis integration.
    
    Args:
        message: Test message for diagnosis
        
    Returns:
        Test results
    """
    try:
        # Quick setup
        integration = await quick_setup_diagnosis_integration()
        
        # Run test
        test_result = await integration.test_diagnosis(message)
        
        # Get status
        status = await integration.get_integration_status()
        
        return {
            "test_result": test_result,
            "integration_status": status,
            "success": test_result.get("success", False)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_integration_info() -> Dict[str, Any]:
    """Get information about available integration components."""
    return {
        "diagnosis_services_available": True,
        "memory_system_available": ENHANCED_MEMORY_AVAILABLE,
        "vector_db_available": CENTRAL_VECTOR_DB_AVAILABLE,
        "compatibility_layer_available": True,
        "integration_ready": True
    }
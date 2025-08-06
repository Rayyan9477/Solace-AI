"""
Diagnosis Orchestrator Service

This module provides orchestration for multiple diagnosis services,
managing service selection, routing, and integration with the agent system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
from dataclasses import asdict

from .interfaces import (
    IDiagnosisOrchestrator, IDiagnosisService, IEnhancedDiagnosisService,
    DiagnosisRequest, DiagnosisResult, DiagnosisType
)
from src.infrastructure.di.container import Injectable
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosisOrchestrator(Injectable, IDiagnosisOrchestrator):
    """
    Orchestrates diagnosis requests across multiple diagnosis services,
    providing intelligent routing and service management.
    """
    
    def __init__(self):
        """Initialize the diagnosis orchestrator."""
        self.services: Dict[str, IDiagnosisService] = {}
        self.service_preferences: Dict[DiagnosisType, List[str]] = {}
        self.logger = get_logger(__name__)
        
        # Orchestrator configuration
        self.max_concurrent_diagnoses = 3
        self.service_timeout_seconds = 30
        self.fallback_enabled = True
        
        # Health monitoring
        self.health_cache = {}
        self.health_cache_ttl_seconds = 60
        
        # Performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "service_usage": {}
        }
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator."""
        try:
            self.logger.info("Initializing DiagnosisOrchestrator")
            
            # Set up default service preferences
            self._setup_default_preferences()
            
            self.logger.info("DiagnosisOrchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DiagnosisOrchestrator: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        try:
            self.logger.info("Shutting down DiagnosisOrchestrator")
            
            # Shutdown all registered services
            for service_name, service in self.services.items():
                try:
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                    self.logger.info(f"Shutdown service: {service_name}")
                except Exception as e:
                    self.logger.error(f"Error shutting down service {service_name}: {str(e)}")
            
            self.services.clear()
            self.logger.info("DiagnosisOrchestrator shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during DiagnosisOrchestrator shutdown: {str(e)}")
    
    async def orchestrate_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Orchestrate diagnosis across multiple services.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Orchestrated diagnosis result
        """
        start_time = datetime.now()
        self.performance_metrics["total_requests"] += 1
        
        try:
            self.logger.info(f"Orchestrating diagnosis for user {request.user_id}, type: {request.diagnosis_type}")
            
            # Select appropriate service for the request
            selected_service = await self._select_service(request)
            
            if not selected_service:
                raise RuntimeError("No suitable diagnosis service available")
            
            # Perform diagnosis using selected service
            result = await self._perform_diagnosis_with_timeout(selected_service, request)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(selected_service[0], processing_time, True)
            
            self.logger.info(f"Diagnosis orchestration completed for user {request.user_id} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics("unknown", processing_time, False)
            
            error_msg = f"Diagnosis orchestration failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error result
            return DiagnosisResult(
                user_id=request.user_id,
                session_id=request.session_id,
                timestamp=start_time,
                diagnosis_type=request.diagnosis_type,
                primary_diagnosis=None,
                confidence_level="low",
                confidence_score=0.0,
                symptoms=[],
                potential_conditions=[],
                recommendations=["Unable to complete diagnosis orchestration."],
                processing_time_ms=processing_time,
                warnings=[error_msg],
                limitations=["Orchestration system error"],
                context_updates={},
                memory_insights=[],
                raw_response={"error": str(e)}
            )
    
    async def register_diagnosis_service(self, 
                                       service_name: str, 
                                       service: IDiagnosisService) -> bool:
        """
        Register a diagnosis service with the orchestrator.
        
        Args:
            service_name: Name of the service
            service: Service instance
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            if service_name in self.services:
                self.logger.warning(f"Service {service_name} already registered, replacing")
            
            # Validate service
            if not await self._validate_service(service):
                self.logger.error(f"Service {service_name} failed validation")
                return False
            
            # Register the service
            self.services[service_name] = service
            
            # Initialize performance tracking
            self.performance_metrics["service_usage"][service_name] = {
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time_ms": 0.0
            }
            
            # Update service preferences based on capabilities
            await self._update_service_preferences(service_name, service)
            
            self.logger.info(f"Registered diagnosis service: {service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering service {service_name}: {str(e)}")
            return False
    
    async def get_available_services(self) -> List[str]:
        """
        Get list of available diagnosis services.
        
        Returns:
            List of service names
        """
        try:
            available_services = []
            
            for service_name, service in self.services.items():
                # Check service health
                health = await self._get_service_health(service_name, service)
                if health.get("status") == "healthy":
                    available_services.append(service_name)
            
            return available_services
            
        except Exception as e:
            self.logger.error(f"Error getting available services: {str(e)}")
            return []
    
    async def get_orchestrator_health(self) -> Dict[str, Any]:
        """
        Get health status of the orchestrator and all registered services.
        
        Returns:
            Health status information
        """
        try:
            health_status = {
                "orchestrator_status": "healthy",
                "total_services": len(self.services),
                "healthy_services": 0,
                "unhealthy_services": 0,
                "service_health": {},
                "performance_metrics": self.performance_metrics.copy(),
                "last_check": datetime.now().isoformat()
            }
            
            # Check health of all services
            for service_name, service in self.services.items():
                service_health = await self._get_service_health(service_name, service)
                health_status["service_health"][service_name] = service_health
                
                if service_health.get("status") == "healthy":
                    health_status["healthy_services"] += 1
                else:
                    health_status["unhealthy_services"] += 1
            
            # Determine overall orchestrator health
            if health_status["healthy_services"] == 0:
                health_status["orchestrator_status"] = "unhealthy"
            elif health_status["unhealthy_services"] > 0:
                health_status["orchestrator_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error getting orchestrator health: {str(e)}")
            return {
                "orchestrator_status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    # Private helper methods
    
    def _setup_default_preferences(self) -> None:
        """Set up default service preferences for different diagnosis types."""
        self.service_preferences = {
            DiagnosisType.BASIC: ["basic_diagnosis_service", "unified_diagnosis_service"],
            DiagnosisType.COMPREHENSIVE: ["unified_diagnosis_service", "enhanced_diagnosis_service"],
            DiagnosisType.ENHANCED_INTEGRATED: ["unified_diagnosis_service", "enhanced_integrated_service"],
            DiagnosisType.DIFFERENTIAL: ["differential_diagnosis_service", "unified_diagnosis_service"],
            DiagnosisType.TEMPORAL: ["temporal_diagnosis_service", "unified_diagnosis_service"]
        }
    
    async def _select_service(self, request: DiagnosisRequest) -> Optional[tuple]:
        """
        Select the most appropriate service for a diagnosis request.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Tuple of (service_name, service_instance) or None
        """
        try:
            # Get preferred services for this diagnosis type
            preferred_services = self.service_preferences.get(request.diagnosis_type, [])
            
            # Add all services as fallback
            all_services = list(self.services.keys())
            candidate_services = preferred_services + [s for s in all_services if s not in preferred_services]
            
            # Find the first healthy service that supports the diagnosis type
            for service_name in candidate_services:
                if service_name not in self.services:
                    continue
                
                service = self.services[service_name]
                
                # Check if service supports the diagnosis type
                if not service.supports_diagnosis_type(request.diagnosis_type):
                    continue
                
                # Check service health
                health = await self._get_service_health(service_name, service)
                if health.get("status") != "healthy":
                    continue
                
                self.logger.info(f"Selected service {service_name} for diagnosis type {request.diagnosis_type}")
                return (service_name, service)
            
            self.logger.error(f"No suitable service found for diagnosis type {request.diagnosis_type}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error selecting service: {str(e)}")
            return None
    
    async def _perform_diagnosis_with_timeout(self, 
                                            selected_service: tuple, 
                                            request: DiagnosisRequest) -> DiagnosisResult:
        """
        Perform diagnosis with timeout handling.
        
        Args:
            selected_service: Tuple of (service_name, service_instance)
            request: Diagnosis request
            
        Returns:
            Diagnosis result
        """
        service_name, service = selected_service
        
        try:
            # Perform diagnosis with timeout
            result = await asyncio.wait_for(
                service.diagnose(request),
                timeout=self.service_timeout_seconds
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Diagnosis timeout for service {service_name}")
            raise RuntimeError(f"Diagnosis timeout after {self.service_timeout_seconds} seconds")
        except Exception as e:
            self.logger.error(f"Error during diagnosis with service {service_name}: {str(e)}")
            raise
    
    async def _validate_service(self, service: IDiagnosisService) -> bool:
        """
        Validate that a service implements the required interface correctly.
        
        Args:
            service: Service to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required methods exist
            required_methods = ['diagnose', 'validate_request', 'supports_diagnosis_type', 'get_service_health']
            
            for method_name in required_methods:
                if not hasattr(service, method_name):
                    self.logger.error(f"Service missing required method: {method_name}")
                    return False
                
                method = getattr(service, method_name)
                if not callable(method):
                    self.logger.error(f"Service method {method_name} is not callable")
                    return False
            
            # Test service health
            health = await service.get_service_health()
            if not isinstance(health, dict):
                self.logger.error("Service health check did not return dictionary")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating service: {str(e)}")
            return False
    
    async def _update_service_preferences(self, service_name: str, service: IDiagnosisService) -> None:
        """
        Update service preferences based on service capabilities.
        
        Args:
            service_name: Name of the service
            service: Service instance
        """
        try:
            # Check which diagnosis types the service supports
            for diagnosis_type in DiagnosisType:
                if service.supports_diagnosis_type(diagnosis_type):
                    if diagnosis_type not in self.service_preferences:
                        self.service_preferences[diagnosis_type] = []
                    
                    # Add service to preferences if not already there
                    if service_name not in self.service_preferences[diagnosis_type]:
                        self.service_preferences[diagnosis_type].insert(0, service_name)
        
        except Exception as e:
            self.logger.warning(f"Error updating service preferences for {service_name}: {str(e)}")
    
    async def _get_service_health(self, service_name: str, service: IDiagnosisService) -> Dict[str, Any]:
        """
        Get health status for a service with caching.
        
        Args:
            service_name: Name of the service
            service: Service instance
            
        Returns:
            Health status dictionary
        """
        try:
            # Check cache first
            cache_key = service_name
            if cache_key in self.health_cache:
                cached_health, cached_time = self.health_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.health_cache_ttl_seconds:
                    return cached_health
            
            # Get fresh health status
            health = await service.get_service_health()
            
            # Cache the result
            self.health_cache[cache_key] = (health, datetime.now())
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error getting health for service {service_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _update_performance_metrics(self, service_name: str, processing_time_ms: float, success: bool) -> None:
        """
        Update performance metrics for a service.
        
        Args:
            service_name: Name of the service
            processing_time_ms: Processing time in milliseconds
            success: Whether the request was successful
        """
        try:
            # Update overall metrics
            if success:
                self.performance_metrics["successful_requests"] += 1
            else:
                self.performance_metrics["failed_requests"] += 1
            
            # Update average response time
            total_requests = self.performance_metrics["total_requests"]
            current_avg = self.performance_metrics["average_response_time_ms"]
            new_avg = ((current_avg * (total_requests - 1)) + processing_time_ms) / total_requests
            self.performance_metrics["average_response_time_ms"] = new_avg
            
            # Update service-specific metrics
            if service_name in self.performance_metrics["service_usage"]:
                service_metrics = self.performance_metrics["service_usage"][service_name]
                service_metrics["requests"] += 1
                
                if success:
                    service_metrics["successful_requests"] += 1
                else:
                    service_metrics["failed_requests"] += 1
                
                # Update service average response time
                service_total = service_metrics["requests"]
                service_avg = service_metrics["average_response_time_ms"]
                service_new_avg = ((service_avg * (service_total - 1)) + processing_time_ms) / service_total
                service_metrics["average_response_time_ms"] = service_new_avg
        
        except Exception as e:
            self.logger.warning(f"Error updating performance metrics: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "service_usage": {name: {
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time_ms": 0.0
            } for name in self.services.keys()}
        }
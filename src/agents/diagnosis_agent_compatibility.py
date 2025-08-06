"""
Diagnosis Agent Compatibility Layer

This module provides backward compatibility adapters for existing diagnosis agents,
allowing them to work with the new unified diagnosis system while maintaining
their original interfaces.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.services.diagnosis import (
    DiagnosisRequest, DiagnosisType, IDiagnosisOrchestrator, IDiagnosisAgentAdapter
)
from src.infrastructure.di.container import get_container
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiagnosisAgentCompatibilityMixin:
    """
    Mixin class that adds unified diagnosis system integration to existing diagnosis agents.
    
    This mixin can be added to existing diagnosis agents to provide seamless integration
    with the new unified diagnosis system while maintaining backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the compatibility mixin."""
        super().__init__(*args, **kwargs)
        
        # Initialize diagnosis system integration
        self._diagnosis_orchestrator = None
        self._diagnosis_adapter = None
        self._diagnosis_integration_enabled = False
        self._compatibility_logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize diagnosis system integration asynchronously
        asyncio.create_task(self._initialize_diagnosis_integration())
    
    async def _initialize_diagnosis_integration(self) -> None:
        """Initialize integration with the unified diagnosis system."""
        try:
            # Get DI container
            container = get_container()
            
            # Resolve diagnosis services
            self._diagnosis_orchestrator = await container.resolve(IDiagnosisOrchestrator)
            self._diagnosis_adapter = await container.resolve(IDiagnosisAgentAdapter)
            
            self._diagnosis_integration_enabled = True
            self._compatibility_logger.info(f"Diagnosis integration enabled for {self.__class__.__name__}")
            
        except Exception as e:
            self._compatibility_logger.warning(f"Failed to initialize diagnosis integration for {self.__class__.__name__}: {str(e)}")
            self._diagnosis_integration_enabled = False
    
    async def enhanced_process(self, 
                             input_data: Dict[str, Any], 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced process method that can use either the unified diagnosis system or fall back to the original.
        
        Args:
            input_data: Input data for processing
            context: Processing context
            
        Returns:
            Processing result
        """
        # Try to use unified diagnosis system if available
        if self._diagnosis_integration_enabled and self._diagnosis_orchestrator:
            try:
                return await self._process_with_unified_system(input_data, context)
            except Exception as e:
                self._compatibility_logger.warning(f"Unified diagnosis system failed, falling back to original: {str(e)}")
        
        # Fallback to original process method
        if hasattr(super(), 'process'):
            return await super().process(input_data, context)
        else:
            return {"error": "No processing method available"}
    
    async def _process_with_unified_system(self, 
                                         input_data: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process using the unified diagnosis system."""
        try:
            # Convert input to diagnosis request
            diagnosis_request = await self._diagnosis_adapter.adapt_agent_request(
                agent_input=input_data,
                context={
                    **(context or {}),
                    "agent_type": self.__class__.__name__.lower(),
                    "compatibility_mode": True
                }
            )
            
            # Perform diagnosis
            diagnosis_result = await self._diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Adapt result back to agent format
            agent_format = self._get_agent_format()
            adapted_result = await self._diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, agent_format
            )
            
            # Add compatibility metadata
            adapted_result["_unified_diagnosis"] = True
            adapted_result["_processing_time_ms"] = diagnosis_result.processing_time_ms
            
            return adapted_result
            
        except Exception as e:
            self._compatibility_logger.error(f"Error in unified diagnosis processing: {str(e)}")
            raise
    
    def _get_agent_format(self) -> str:
        """Get the appropriate agent format based on the agent type."""
        class_name = self.__class__.__name__.lower()
        
        if "enhanced_integrated" in class_name:
            return "enhanced_integrated"
        elif "comprehensive" in class_name:
            return "comprehensive"
        elif "enhanced" in class_name:
            return "enhanced"
        elif "integrated" in class_name:
            return "integrated"
        else:
            return "basic"
    
    def get_diagnosis_integration_status(self) -> Dict[str, Any]:
        """Get status of diagnosis system integration."""
        return {
            "integration_enabled": self._diagnosis_integration_enabled,
            "orchestrator_available": self._diagnosis_orchestrator is not None,
            "adapter_available": self._diagnosis_adapter is not None,
            "agent_class": self.__class__.__name__,
            "supported_format": self._get_agent_format()
        }


class EnhancedDiagnosisAgentWrapper:
    """
    Wrapper class that enhances existing diagnosis agents with unified system capabilities.
    
    This wrapper can be used to wrap existing diagnosis agents and provide them with
    enhanced capabilities from the unified diagnosis system.
    """
    
    def __init__(self, original_agent, agent_name: str = None):
        """
        Initialize the wrapper.
        
        Args:
            original_agent: The original diagnosis agent to wrap
            agent_name: Optional name for the agent
        """
        self.original_agent = original_agent
        self.agent_name = agent_name or original_agent.__class__.__name__
        self.logger = get_logger(f"{__name__}.{self.agent_name}")
        
        # Initialize diagnosis system integration
        self.diagnosis_orchestrator = None
        self.diagnosis_adapter = None
        self.integration_enabled = False
        
        # Copy attributes from original agent
        self._copy_agent_attributes()
    
    def _copy_agent_attributes(self):
        """Copy important attributes from the original agent."""
        # Copy common attributes that agents might have
        attrs_to_copy = ['model', 'name', 'role', 'description', 'tools', 'memory', 'knowledge']
        
        for attr in attrs_to_copy:
            if hasattr(self.original_agent, attr):
                setattr(self, attr, getattr(self.original_agent, attr))
    
    async def initialize_diagnosis_integration(self) -> bool:
        """Initialize integration with the unified diagnosis system."""
        try:
            # Get DI container
            container = get_container()
            
            # Resolve diagnosis services
            self.diagnosis_orchestrator = await container.resolve(IDiagnosisOrchestrator)
            self.diagnosis_adapter = await container.resolve(IDiagnosisAgentAdapter)
            
            self.integration_enabled = True
            self.logger.info(f"Diagnosis integration enabled for wrapped agent {self.agent_name}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize diagnosis integration: {str(e)}")
            self.integration_enabled = False
            return False
    
    async def process(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process method that can use either unified system or original agent.
        
        Args:
            input_data: Input data for processing
            context: Processing context
            
        Returns:
            Processing result
        """
        # Determine processing mode based on input or context
        use_unified = self._should_use_unified_system(input_data, context)
        
        if use_unified and self.integration_enabled:
            try:
                return await self._process_with_unified_system(input_data, context)
            except Exception as e:
                self.logger.warning(f"Unified processing failed, using original agent: {str(e)}")
        
        # Use original agent
        return await self._process_with_original_agent(input_data, context)
    
    def _should_use_unified_system(self, input_data: Dict[str, Any], context: Dict[str, Any] = None) -> bool:
        """Determine whether to use the unified system or original agent."""
        # Check if explicitly requested
        if input_data.get("use_unified_diagnosis", False):
            return True
        
        if context and context.get("use_unified_diagnosis", False):
            return True
        
        # Check if comprehensive diagnosis is requested
        if input_data.get("diagnosis_type") in ["comprehensive", "enhanced_integrated"]:
            return True
        
        # Default to unified system if available
        return self.integration_enabled
    
    async def _process_with_unified_system(self, 
                                         input_data: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process using the unified diagnosis system."""
        try:
            # Convert input to diagnosis request
            diagnosis_request = await self.diagnosis_adapter.adapt_agent_request(
                agent_input=input_data,
                context={
                    **(context or {}),
                    "agent_type": self.agent_name.lower(),
                    "wrapped_agent": True,
                    "original_agent_class": self.original_agent.__class__.__name__
                }
            )
            
            # Perform diagnosis
            diagnosis_result = await self.diagnosis_orchestrator.orchestrate_diagnosis(diagnosis_request)
            
            # Adapt result back to agent format
            agent_format = self._determine_agent_format()
            adapted_result = await self.diagnosis_adapter.adapt_diagnosis_response(
                diagnosis_result, agent_format
            )
            
            # Add wrapper metadata
            adapted_result["_wrapped_agent"] = True
            adapted_result["_original_agent"] = self.original_agent.__class__.__name__
            adapted_result["_unified_diagnosis"] = True
            
            return adapted_result
            
        except Exception as e:
            self.logger.error(f"Error in unified diagnosis processing: {str(e)}")
            raise
    
    async def _process_with_original_agent(self, 
                                         input_data: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process using the original agent."""
        try:
            if hasattr(self.original_agent, 'process'):
                result = await self.original_agent.process(input_data, context)
            elif hasattr(self.original_agent, '_generate_response'):
                result = await self.original_agent._generate_response(input_data, context, {})
            else:
                raise AttributeError("Original agent has no compatible processing method")
            
            # Add wrapper metadata
            if isinstance(result, dict):
                result["_wrapped_agent"] = True
                result["_original_agent"] = self.original_agent.__class__.__name__
                result["_unified_diagnosis"] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in original agent processing: {str(e)}")
            raise
    
    def _determine_agent_format(self) -> str:
        """Determine the appropriate agent format for the original agent."""
        agent_class_name = self.original_agent.__class__.__name__.lower()
        
        if "enhanced_integrated" in agent_class_name:
            return "enhanced_integrated"
        elif "comprehensive" in agent_class_name:
            return "comprehensive"
        elif "enhanced" in agent_class_name:
            return "enhanced"
        elif "integrated" in agent_class_name:
            return "integrated"
        else:
            return "basic"
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the wrapper and integration."""
        return {
            "wrapper_enabled": True,
            "integration_enabled": self.integration_enabled,
            "original_agent": self.original_agent.__class__.__name__,
            "agent_name": self.agent_name,
            "supported_format": self._determine_agent_format(),
            "orchestrator_available": self.diagnosis_orchestrator is not None,
            "adapter_available": self.diagnosis_adapter is not None
        }
    
    def __getattr__(self, name):
        """Delegate attribute access to the original agent if not found in wrapper."""
        if hasattr(self.original_agent, name):
            return getattr(self.original_agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


async def wrap_diagnosis_agent(agent, agent_name: str = None) -> EnhancedDiagnosisAgentWrapper:
    """
    Convenience function to wrap a diagnosis agent and initialize integration.
    
    Args:
        agent: The diagnosis agent to wrap
        agent_name: Optional name for the agent
        
    Returns:
        Enhanced diagnosis agent wrapper
    """
    wrapper = EnhancedDiagnosisAgentWrapper(agent, agent_name)
    await wrapper.initialize_diagnosis_integration()
    return wrapper


def apply_compatibility_mixin(agent_class):
    """
    Decorator to apply the compatibility mixin to an existing diagnosis agent class.
    
    Args:
        agent_class: The agent class to enhance
        
    Returns:
        Enhanced agent class with compatibility mixin
    """
    class EnhancedAgent(DiagnosisAgentCompatibilityMixin, agent_class):
        """Enhanced agent class with unified diagnosis system compatibility."""
        pass
    
    # Preserve original class metadata
    EnhancedAgent.__name__ = f"Enhanced{agent_class.__name__}"
    EnhancedAgent.__qualname__ = f"Enhanced{agent_class.__qualname__}"
    EnhancedAgent.__module__ = agent_class.__module__
    
    return EnhancedAgent


# Compatibility registry for tracking enhanced agents
_compatibility_registry = {}


def register_enhanced_agent(agent_name: str, enhanced_agent):
    """Register an enhanced agent in the compatibility registry."""
    _compatibility_registry[agent_name] = enhanced_agent
    logger.info(f"Registered enhanced agent: {agent_name}")


def get_enhanced_agent(agent_name: str):
    """Get an enhanced agent from the compatibility registry."""
    return _compatibility_registry.get(agent_name)


def get_all_enhanced_agents() -> Dict[str, Any]:
    """Get all enhanced agents from the compatibility registry."""
    return _compatibility_registry.copy()


def is_diagnosis_integration_available() -> bool:
    """Check if diagnosis system integration is available."""
    try:
        from src.services.diagnosis import DIAGNOSIS_SERVICES_AVAILABLE
        return DIAGNOSIS_SERVICES_AVAILABLE
    except ImportError:
        return False
"""
Base class for Therapeutic Friction Sub-Agents.

This module provides a common foundation for all therapeutic friction sub-agents,
ensuring consistent interfaces, vector database integration, and supervision support.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from enum import Enum

from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger


class FrictionAgentType(Enum):
    """Types of therapeutic friction sub-agents."""
    READINESS_ASSESSMENT = "readiness_assessment"
    BREAKTHROUGH_DETECTION = "breakthrough_detection"
    RELATIONSHIP_MONITORING = "relationship_monitoring"
    INTERVENTION_STRATEGY = "intervention_strategy"
    PROGRESS_TRACKING = "progress_tracking"


class BaseFrictionAgent(BaseAgent, ABC):
    """
    Abstract base class for therapeutic friction sub-agents.
    
    Provides common functionality for vector database integration,
    supervision support, and coordination with other sub-agents.
    """
    
    def __init__(self, model_provider=None, agent_type: FrictionAgentType = None, config: Dict[str, Any] = None):
        """Initialize the base friction agent."""
        self.agent_type = agent_type
        self.config = config or {}
        
        super().__init__(
            model=model_provider,
            name=f"{agent_type.value}_agent" if agent_type else "base_friction_agent",
            role=f"Therapeutic {agent_type.value.replace('_', ' ').title()} Specialist" if agent_type else "Base Friction Agent",
            description=f"Specialized agent for {agent_type.value.replace('_', ' ')} in therapeutic contexts" if agent_type else "Base therapeutic friction agent"
        )
        
        self.logger = get_logger(self.name)
        self.vector_namespace = f"therapeutic_friction_{agent_type.value}" if agent_type else "therapeutic_friction_base"
        
        # Shared state for coordination
        self.last_assessment = None
        self.confidence_score = 0.0
        self.processing_time = 0.0
        
        # Integration points
        self.integration_callbacks = {}
        
    @abstractmethod
    async def assess(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform specialized assessment based on agent type.
        
        Args:
            user_input: User's message or query
            context: Contextual information from other agents
            
        Returns:
            Assessment results specific to this agent's domain
        """
        pass
    
    @abstractmethod
    def get_specialized_knowledge_query(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        Generate specialized query for vector database search.
        
        Args:
            user_input: User's message
            context: Contextual information
            
        Returns:
            Optimized query string for this agent's knowledge domain
        """
        pass
    
    @abstractmethod
    def validate_assessment(self, assessment: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate the quality and consistency of the assessment.
        
        Args:
            assessment: Assessment results to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    async def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input with specialized assessment and integration."""
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Perform specialized assessment
            assessment = await self.assess(user_input, context)
            
            # Validate assessment quality
            is_valid, issues = self.validate_assessment(assessment)
            
            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_time = processing_time
            
            # Calculate confidence based on validation and internal factors
            confidence = self._calculate_specialized_confidence(assessment, is_valid, issues)
            self.confidence_score = confidence
            
            # Prepare result with supervision metadata
            result = {
                'assessment': assessment,
                'agent_type': self.agent_type.value,
                'confidence': confidence,
                'is_valid': is_valid,
                'validation_issues': issues,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'processing_time': processing_time,
                    'vector_namespace': self.vector_namespace
                },
                'context_updates': self._generate_agent_specific_context(user_input, assessment, context)
            }
            
            # Store assessment for coordination
            self.last_assessment = result
            
            # Execute integration callbacks
            await self._execute_integration_callbacks(result, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} assessment: {str(e)}")
            return {
                'error': str(e),
                'agent_type': self.agent_type.value,
                'confidence': 0.0,
                'is_valid': False,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': self.name,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            }
    
    def _calculate_specialized_confidence(self, assessment: Dict[str, Any], 
                                        is_valid: bool, issues: List[str]) -> float:
        """Calculate confidence score based on assessment quality and validation."""
        base_confidence = 0.8 if is_valid else 0.4
        
        # Adjust based on validation issues
        if issues:
            base_confidence *= max(0.1, 1.0 - (len(issues) * 0.1))
        
        # Agent-specific confidence adjustments
        agent_confidence = self._get_agent_specific_confidence(assessment)
        
        return min(1.0, base_confidence * agent_confidence)
    
    def _get_agent_specific_confidence(self, assessment: Dict[str, Any]) -> float:
        """Get agent-specific confidence adjustments. Override in subclasses."""
        return 1.0
    
    def _generate_agent_specific_context(self, user_input: str, assessment: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context updates specific to this agent's domain."""
        return {
            f"{self.agent_type.value}_assessment": {
                "timestamp": datetime.now().isoformat(),
                "confidence": self.confidence_score,
                "key_findings": self._extract_key_findings(assessment),
                "processing_time": self.processing_time
            }
        }
    
    def _extract_key_findings(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from assessment. Override in subclasses."""
        return {"assessment_completed": True}
    
    async def _execute_integration_callbacks(self, result: Dict[str, Any], context: Dict[str, Any]):
        """Execute registered integration callbacks."""
        for callback_name, callback_func in self.integration_callbacks.items():
            try:
                if asyncio.iscoroutinefunction(callback_func):
                    await callback_func(result, context)
                else:
                    callback_func(result, context)
            except Exception as e:
                self.logger.warning(f"Integration callback {callback_name} failed: {str(e)}")
    
    def register_integration_callback(self, name: str, callback_func):
        """Register callback for integration with other system components."""
        self.integration_callbacks[name] = callback_func
    
    async def get_specialized_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve specialized knowledge from vector database."""
        try:
            # This would integrate with the enhanced vector database system
            # For now, return empty list as placeholder
            self.logger.debug(f"Knowledge query for {self.agent_type.value}: {query}")
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving specialized knowledge: {str(e)}")
            return []
    
    def get_coordination_data(self) -> Dict[str, Any]:
        """Get data needed for coordination with other sub-agents."""
        return {
            'agent_type': self.agent_type.value,
            'last_assessment': self.last_assessment,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'status': 'ready' if self.last_assessment else 'not_ready'
        }
    
    async def coordinate_with_agent(self, other_agent: 'BaseFrictionAgent', 
                                  shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate assessment with another sub-agent."""
        try:
            my_data = self.get_coordination_data()
            other_data = other_agent.get_coordination_data()
            
            coordination_result = {
                'agents': [my_data['agent_type'], other_data['agent_type']],
                'combined_confidence': (my_data['confidence_score'] + other_data['confidence_score']) / 2,
                'coordination_timestamp': datetime.now().isoformat()
            }
            
            # Agent-specific coordination logic can be implemented here
            coordination_result.update(
                await self._perform_specialized_coordination(other_agent, shared_context)
            )
            
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Coordination error with {other_agent.name}: {str(e)}")
            return {'error': str(e)}
    
    async def _perform_specialized_coordination(self, other_agent: 'BaseFrictionAgent', 
                                             shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform agent-specific coordination logic. Override in subclasses."""
        return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring and supervision."""
        return {
            'agent_name': self.name,
            'agent_type': self.agent_type.value,
            'status': 'healthy',
            'last_assessment_time': self.last_assessment.get('metadata', {}).get('timestamp') if self.last_assessment else None,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'integration_callbacks': list(self.integration_callbacks.keys())
        }
    
    async def reset_state(self):
        """Reset agent state for new session or clean start."""
        self.last_assessment = None
        self.confidence_score = 0.0
        self.processing_time = 0.0
        self.logger.info(f"{self.name} state reset completed")


class FrictionAgentRegistry:
    """Registry for managing therapeutic friction sub-agents."""
    
    def __init__(self):
        self.agents: Dict[FrictionAgentType, BaseFrictionAgent] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def register_agent(self, agent: BaseFrictionAgent):
        """Register a friction sub-agent."""
        if agent.agent_type in self.agents:
            self.logger.warning(f"Overriding existing agent for {agent.agent_type.value}")
        
        self.agents[agent.agent_type] = agent
        self.logger.info(f"Registered {agent.name} for {agent.agent_type.value}")
    
    def get_agent(self, agent_type: FrictionAgentType) -> Optional[BaseFrictionAgent]:
        """Get a specific sub-agent by type."""
        return self.agents.get(agent_type)
    
    def get_all_agents(self) -> Dict[FrictionAgentType, BaseFrictionAgent]:
        """Get all registered agents."""
        return self.agents.copy()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all registered agents."""
        health_status = {
            'total_agents': len(self.agents),
            'healthy_agents': 0,
            'agent_status': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for agent_type, agent in self.agents.items():
            try:
                status = agent.get_health_status()
                health_status['agent_status'][agent_type.value] = status
                if status['status'] == 'healthy':
                    health_status['healthy_agents'] += 1
            except Exception as e:
                health_status['agent_status'][agent_type.value] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_status
    
    async def reset_all_agents(self):
        """Reset state for all registered agents."""
        for agent in self.agents.values():
            try:
                await agent.reset_state()
            except Exception as e:
                self.logger.error(f"Error resetting {agent.name}: {str(e)}")


# Global registry instance
friction_agent_registry = FrictionAgentRegistry()
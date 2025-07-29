"""
Abstract interface for Agent components.

This interface defines the contract for all agents in the system,
enabling composition-based architecture and easy testing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Enum for different types of agents."""
    CHAT = "chat"
    EMOTION = "emotion"
    DIAGNOSIS = "diagnosis"
    THERAPY = "therapy"
    SAFETY = "safety"
    PERSONALITY = "personality"
    SEARCH = "search"
    ORCHESTRATOR = "orchestrator"


class AgentStatus(Enum):
    """Enum for agent status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentRequest:
    """Request object for agent processing."""
    content: str
    context: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


@dataclass
class AgentResponse:
    """Response object from agent processing."""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class AgentCapability:
    """Describes what an agent can do."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str] = None


class AgentInterface(ABC):
    """
    Abstract base class for all agents in the system.
    
    This interface ensures all agents implement consistent methods
    for processing, health checks, and lifecycle management.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the agent."""
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.INITIALIZED
        self._capabilities = []
        self._dependencies = {}
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """Return the capabilities of this agent."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent with its dependencies.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a request and return a response.
        
        Args:
            request: The request to process
            
        Returns:
            AgentResponse: The processed response
        """
        pass
    
    @abstractmethod
    async def validate_request(self, request: AgentRequest) -> bool:
        """
        Validate if the agent can handle this request.
        
        Args:
            request: The request to validate
            
        Returns:
            bool: True if the request can be handled
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the agent and cleanup resources.
        
        Returns:
            bool: True if shutdown was successful
        """
        pass
    
    def add_dependency(self, name: str, dependency: Any) -> None:
        """Add a dependency to this agent."""
        self._dependencies[name] = dependency
    
    def get_dependency(self, name: str) -> Any:
        """Get a dependency by name."""
        return self._dependencies.get(name)
    
    def has_dependency(self, name: str) -> bool:
        """Check if agent has a specific dependency."""
        return name in self._dependencies
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the agent.
        
        Returns:
            Dict containing health status
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": [cap.name for cap in self.capabilities],
            "dependencies": list(self._dependencies.keys()),
            "config": self.config
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics for monitoring.
        
        Returns:
            Dict containing agent metrics
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "requests_processed": getattr(self, '_requests_processed', 0),
            "average_response_time": getattr(self, '_avg_response_time', 0.0),
            "error_count": getattr(self, '_error_count', 0)
        }
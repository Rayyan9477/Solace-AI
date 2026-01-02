"""
Diagnosis Service Interfaces

This module defines the core interfaces for the unified diagnosis system,
providing clean abstractions for different diagnosis components and ensuring
consistent integration patterns across the application.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DiagnosisType(Enum):
    """Types of diagnosis available in the system"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    ENHANCED_INTEGRATED = "enhanced_integrated"
    DIFFERENTIAL = "differential"
    TEMPORAL = "temporal"


class ConfidenceLevel(Enum):
    """Confidence levels for diagnosis results"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DiagnosisRequest:
    """Request object for diagnosis operations"""
    user_id: str
    session_id: str
    message: str
    conversation_history: List[Dict[str, Any]]
    voice_emotion_data: Optional[Dict[str, Any]] = None
    personality_data: Optional[Dict[str, Any]] = None
    cultural_info: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    diagnosis_type: DiagnosisType = DiagnosisType.COMPREHENSIVE


@dataclass
class DiagnosisResult:
    """Result object for diagnosis operations"""
    user_id: str
    session_id: str
    timestamp: datetime
    diagnosis_type: DiagnosisType
    
    # Core diagnosis data
    primary_diagnosis: Optional[str]
    confidence_level: ConfidenceLevel
    confidence_score: float
    
    # Detailed results
    symptoms: List[str]
    potential_conditions: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Metadata
    processing_time_ms: float
    warnings: List[str]
    limitations: List[str]
    
    # Integration data
    context_updates: Dict[str, Any]
    memory_insights: List[Dict[str, Any]]
    
    # Raw response data
    raw_response: Dict[str, Any]


class IDiagnosisService(ABC):
    """
    Core interface for diagnosis services.
    
    This interface defines the contract that all diagnosis services must implement,
    ensuring consistent behavior across different diagnosis implementations.
    """
    
    @abstractmethod
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Perform diagnosis based on the provided request.
        
        Args:
            request: Diagnosis request containing all necessary data
            
        Returns:
            Diagnosis result with findings and recommendations
        """
        pass
    
    @abstractmethod
    async def validate_request(self, request: DiagnosisRequest) -> bool:
        """
        Validate a diagnosis request before processing.
        
        Args:
            request: Request to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_diagnosis_type(self, diagnosis_type: DiagnosisType) -> bool:
        """
        Check if this service supports a specific diagnosis type.
        
        Args:
            diagnosis_type: Type of diagnosis to check
            
        Returns:
            True if supported, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get the health status of the diagnosis service.
        
        Returns:
            Dictionary containing health status information
        """
        pass


class IEnhancedDiagnosisService(IDiagnosisService):
    """
    Enhanced diagnosis service interface with additional capabilities.
    
    This interface extends the basic diagnosis service with advanced features
    like temporal analysis, cultural sensitivity, and adaptive learning.
    """
    
    @abstractmethod
    async def get_comprehensive_diagnosis(self, request: DiagnosisRequest) -> Dict[str, Any]:
        """
        Get a comprehensive diagnosis using all available systems.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Comprehensive diagnosis result
        """
        pass
    
    @abstractmethod
    async def get_temporal_analysis(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get temporal analysis for a user's symptoms over time.
        
        Args:
            user_id: User identifier
            days_back: Number of days to analyze
            
        Returns:
            Temporal analysis results
        """
        pass
    
    @abstractmethod
    async def get_cultural_adaptations(self, 
                                     user_id: str, 
                                     diagnosis: str, 
                                     cultural_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get culturally adapted diagnosis and recommendations.
        
        Args:
            user_id: User identifier
            diagnosis: Primary diagnosis
            cultural_context: Cultural background information
            
        Returns:
            Culturally adapted results
        """
        pass
    
    @abstractmethod
    async def get_personalized_recommendations(self, 
                                             user_id: str, 
                                             diagnosis_result: DiagnosisResult) -> Dict[str, Any]:
        """
        Get personalized recommendations based on diagnosis and user history.
        
        Args:
            user_id: User identifier
            diagnosis_result: Previous diagnosis result
            
        Returns:
            Personalized recommendations
        """
        pass


class IMemoryIntegrationService(ABC):
    """
    Interface for memory system integration.
    
    This interface defines how diagnosis services interact with the
    memory system for storing insights and retrieving historical context.
    """
    
    @abstractmethod
    async def store_diagnosis_insights(self, 
                                     user_id: str, 
                                     session_id: str, 
                                     diagnosis_result: DiagnosisResult) -> bool:
        """
        Store diagnosis insights in the memory system.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            diagnosis_result: Diagnosis result to store
            
        Returns:
            True if stored successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_historical_context(self, 
                                   user_id: str, 
                                   lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get historical context for a user from the memory system.
        
        Args:
            user_id: User identifier
            lookback_days: Number of days to look back
            
        Returns:
            Historical context data
        """
        pass
    
    @abstractmethod
    async def get_session_continuity(self, 
                                   user_id: str, 
                                   session_id: str) -> Dict[str, Any]:
        """
        Get session continuity context.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Session continuity data
        """
        pass


class IVectorDatabaseIntegrationService(ABC):
    """
    Interface for vector database integration.
    
    This interface defines how diagnosis services interact with the
    vector database for storing and retrieving diagnosis data.
    """
    
    @abstractmethod
    async def store_diagnosis_vector(self, 
                                   user_id: str, 
                                   diagnosis_result: DiagnosisResult) -> str:
        """
        Store diagnosis result as vector in the database.
        
        Args:
            user_id: User identifier
            diagnosis_result: Diagnosis result to store
            
        Returns:
            Document ID of stored vector
        """
        pass
    
    @abstractmethod
    async def search_similar_cases(self, 
                                 symptoms: List[str], 
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar diagnosis cases in the vector database.
        
        Args:
            symptoms: List of symptoms to search for
            limit: Maximum number of results
            
        Returns:
            List of similar cases
        """
        pass
    
    @abstractmethod
    async def get_user_diagnosis_history(self, 
                                       user_id: str, 
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get diagnosis history for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            
        Returns:
            List of historical diagnoses
        """
        pass


class IDiagnosisOrchestrator(ABC):
    """
    Interface for diagnosis orchestration service.
    
    This interface defines the high-level orchestration of multiple
    diagnosis services and their integration with the agent system.
    """
    
    @abstractmethod
    async def orchestrate_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Orchestrate diagnosis across multiple services.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Orchestrated diagnosis result
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def get_available_services(self) -> List[str]:
        """
        Get list of available diagnosis services.
        
        Returns:
            List of service names
        """
        pass
    
    @abstractmethod
    async def get_orchestrator_health(self) -> Dict[str, Any]:
        """
        Get health status of the orchestrator and all registered services.
        
        Returns:
            Health status information
        """
        pass


class IDiagnosisAgentAdapter(ABC):
    """
    Interface for adapting diagnosis services to work with existing agents.
    
    This interface provides backward compatibility with existing diagnosis
    agents while enabling integration with the new unified system.
    """
    
    @abstractmethod
    async def adapt_agent_request(self, 
                                agent_input: Dict[str, Any], 
                                context: Dict[str, Any]) -> DiagnosisRequest:
        """
        Adapt agent input to diagnosis request format.
        
        Args:
            agent_input: Input from existing agent
            context: Agent context
            
        Returns:
            Converted diagnosis request
        """
        pass
    
    @abstractmethod
    async def adapt_diagnosis_response(self, 
                                     diagnosis_result: DiagnosisResult, 
                                     agent_format: str = "default") -> Dict[str, Any]:
        """
        Adapt diagnosis result to agent response format.
        
        Args:
            diagnosis_result: Diagnosis result to adapt
            agent_format: Target agent format
            
        Returns:
            Adapted response for agent consumption
        """
        pass
    
    @abstractmethod
    def get_supported_agents(self) -> List[str]:
        """
        Get list of supported agent types.
        
        Returns:
            List of supported agent type names
        """
        pass
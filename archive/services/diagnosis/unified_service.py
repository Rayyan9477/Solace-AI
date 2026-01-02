"""
Unified Diagnosis Service Implementation

This module provides the main implementation of the unified diagnosis service,
integrating all diagnosis components into a cohesive system that works
seamlessly with the existing Solace-AI architecture.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import asdict
import traceback

from .interfaces import (
    IDiagnosisService, IEnhancedDiagnosisService, IMemoryIntegrationService,
    IVectorDatabaseIntegrationService, DiagnosisRequest, DiagnosisResult,
    DiagnosisType, ConfidenceLevel
)
from src.infrastructure.di.container import Injectable
from src.utils.logger import get_logger

# Import existing diagnosis components
try:
    from src.diagnosis.enhanced_integrated_system import (
        EnhancedIntegratedDiagnosticSystem, ComprehensiveDiagnosticResult
    )
except ImportError:
    EnhancedIntegratedDiagnosticSystem = None
    ComprehensiveDiagnosticResult = None

try:
    from src.memory.enhanced_memory_system import EnhancedMemorySystem
except ImportError:
    EnhancedMemorySystem = None

try:
    from src.database.central_vector_db import CentralVectorDB
except ImportError:
    CentralVectorDB = None

logger = get_logger(__name__)


class UnifiedDiagnosisService(Injectable, IEnhancedDiagnosisService, 
                             IMemoryIntegrationService, IVectorDatabaseIntegrationService):
    """
    Unified diagnosis service that integrates all diagnosis components
    into a single, cohesive service with proper dependency injection support.
    """
    
    def __init__(self, 
                 vector_db: Optional[CentralVectorDB] = None,
                 memory_system: Optional[EnhancedMemorySystem] = None):
        """
        Initialize the unified diagnosis service.
        
        Args:
            vector_db: Central vector database instance
            memory_system: Enhanced memory system instance
        """
        self.vector_db = vector_db
        self.memory_system = memory_system
        self.logger = get_logger(__name__)
        
        # Initialize the enhanced integrated diagnostic system
        self.diagnostic_system = None
        self.initialization_error = None
        
        # Service configuration
        self.max_processing_time_seconds = 30
        self.min_confidence_threshold = 0.3
        self.service_health = {
            "status": "initializing",
            "last_check": datetime.now(),
            "components": {}
        }
        
        # Supported diagnosis types
        self.supported_types = {
            DiagnosisType.BASIC,
            DiagnosisType.COMPREHENSIVE,
            DiagnosisType.ENHANCED_INTEGRATED,
            DiagnosisType.DIFFERENTIAL,
            DiagnosisType.TEMPORAL
        }
    
    async def initialize(self) -> bool:
        """Initialize the service and all its components."""
        try:
            self.logger.info("Initializing UnifiedDiagnosisService")
            
            # Initialize the enhanced integrated diagnostic system
            if EnhancedIntegratedDiagnosticSystem:
                self.diagnostic_system = EnhancedIntegratedDiagnosticSystem(self.vector_db)
                self.logger.info("Enhanced integrated diagnostic system initialized")
            else:
                self.logger.warning("Enhanced integrated diagnostic system not available")
            
            # Validate system integration
            await self._validate_system_integration()
            
            # Update service health
            self.service_health["status"] = "healthy"
            self.service_health["last_check"] = datetime.now()
            
            self.logger.info("UnifiedDiagnosisService initialization completed")
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.service_health["status"] = "failed"
            self.service_health["error"] = str(e)
            self.logger.error(f"Failed to initialize UnifiedDiagnosisService: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources."""
        try:
            self.logger.info("Shutting down UnifiedDiagnosisService")
            
            # Cleanup diagnostic system if needed
            if hasattr(self.diagnostic_system, 'shutdown'):
                await self.diagnostic_system.shutdown()
            
            self.service_health["status"] = "shutdown"
            self.logger.info("UnifiedDiagnosisService shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during UnifiedDiagnosisService shutdown: {str(e)}")
    
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResult:
        """
        Perform diagnosis based on the provided request.
        
        Args:
            request: Diagnosis request containing all necessary data
            
        Returns:
            Diagnosis result with findings and recommendations
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            if not await self.validate_request(request):
                raise ValueError("Invalid diagnosis request")
            
            # Route to appropriate diagnosis method based on type
            if request.diagnosis_type == DiagnosisType.COMPREHENSIVE:
                result = await self._perform_comprehensive_diagnosis(request)
            elif request.diagnosis_type == DiagnosisType.ENHANCED_INTEGRATED:
                result = await self._perform_enhanced_integrated_diagnosis(request)
            elif request.diagnosis_type == DiagnosisType.BASIC:
                result = await self._perform_basic_diagnosis(request)
            elif request.diagnosis_type == DiagnosisType.DIFFERENTIAL:
                result = await self._perform_differential_diagnosis(request)
            elif request.diagnosis_type == DiagnosisType.TEMPORAL:
                result = await self._perform_temporal_diagnosis(request)
            else:
                # Default to comprehensive
                result = await self._perform_comprehensive_diagnosis(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            # Store insights in memory if available
            if self.memory_system:
                await self.store_diagnosis_insights(
                    request.user_id, request.session_id, result
                )
            
            # Store in vector database if available
            if self.vector_db:
                await self.store_diagnosis_vector(request.user_id, result)
            
            self.logger.info(f"Diagnosis completed for user {request.user_id} in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Diagnosis failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error result
            return DiagnosisResult(
                user_id=request.user_id,
                session_id=request.session_id,
                timestamp=start_time,
                diagnosis_type=request.diagnosis_type,
                primary_diagnosis=None,
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=0.0,
                symptoms=[],
                potential_conditions=[],
                recommendations=["Unable to complete diagnosis. Please try again."],
                processing_time_ms=processing_time,
                warnings=[error_msg],
                limitations=["System error during diagnosis"],
                context_updates={},
                memory_insights=[],
                raw_response={"error": str(e)}
            )
    
    async def validate_request(self, request: DiagnosisRequest) -> bool:
        """
        Validate a diagnosis request before processing.
        
        Args:
            request: Request to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        try:
            # Basic validation
            if not request.user_id or not request.session_id:
                self.logger.error("Missing user_id or session_id in diagnosis request")
                return False
            
            if not request.message and not request.conversation_history:
                self.logger.error("No message or conversation history provided")
                return False
            
            # Check if diagnosis type is supported
            if not self.supports_diagnosis_type(request.diagnosis_type):
                self.logger.error(f"Unsupported diagnosis type: {request.diagnosis_type}")
                return False
            
            # Check service health
            if self.service_health["status"] != "healthy":
                self.logger.error("Service is not healthy")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating diagnosis request: {str(e)}")
            return False
    
    def supports_diagnosis_type(self, diagnosis_type: DiagnosisType) -> bool:
        """
        Check if this service supports a specific diagnosis type.
        
        Args:
            diagnosis_type: Type of diagnosis to check
            
        Returns:
            True if supported, False otherwise
        """
        return diagnosis_type in self.supported_types
    
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get the health status of the diagnosis service.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            # Update component health
            components = {}
            
            if self.diagnostic_system:
                if hasattr(self.diagnostic_system, 'get_system_status'):
                    components["diagnostic_system"] = self.diagnostic_system.get_system_status()
                else:
                    components["diagnostic_system"] = {"status": "available"}
            else:
                components["diagnostic_system"] = {"status": "unavailable"}
            
            if self.memory_system:
                components["memory_system"] = {"status": "available"}
            else:
                components["memory_system"] = {"status": "unavailable"}
            
            if self.vector_db:
                components["vector_db"] = {"status": "available"}
            else:
                components["vector_db"] = {"status": "unavailable"}
            
            # Update overall health
            self.service_health["components"] = components
            self.service_health["last_check"] = datetime.now()
            self.service_health["initialization_error"] = self.initialization_error
            
            return self.service_health
            
        except Exception as e:
            self.logger.error(f"Error checking service health: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now()
            }
    
    # Enhanced Diagnosis Service Methods
    
    async def get_comprehensive_diagnosis(self, request: DiagnosisRequest) -> Dict[str, Any]:
        """
        Get a comprehensive diagnosis using all available systems.
        
        Args:
            request: Diagnosis request
            
        Returns:
            Comprehensive diagnosis result
        """
        if not self.diagnostic_system:
            raise RuntimeError("Diagnostic system not available")
        
        try:
            comprehensive_result = await self.diagnostic_system.generate_comprehensive_diagnosis(
                user_id=request.user_id,
                session_id=request.session_id,
                user_message=request.message,
                conversation_history=request.conversation_history,
                voice_emotion_data=request.voice_emotion_data,
                personality_data=request.personality_data,
                cultural_info=request.cultural_info
            )
            
            return asdict(comprehensive_result)
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive diagnosis: {str(e)}")
            raise
    
    async def get_temporal_analysis(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get temporal analysis for a user's symptoms over time.
        
        Args:
            user_id: User identifier
            days_back: Number of days to analyze
            
        Returns:
            Temporal analysis results
        """
        if not self.diagnostic_system or not hasattr(self.diagnostic_system, 'temporal_analysis'):
            raise RuntimeError("Temporal analysis not available")
        
        try:
            temporal_engine = self.diagnostic_system.temporal_analysis
            if temporal_engine:
                progression = await temporal_engine.get_symptom_progression(user_id, days_back)
                patterns = await temporal_engine.detect_behavioral_patterns(user_id)
                
                return {
                    "symptom_progression": progression,
                    "behavioral_patterns": [asdict(p) for p in patterns],
                    "analysis_period_days": days_back,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise RuntimeError("Temporal analysis engine not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {str(e)}")
            raise
    
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
        if not self.diagnostic_system or not hasattr(self.diagnostic_system, 'cultural_sensitivity'):
            raise RuntimeError("Cultural sensitivity system not available")
        
        try:
            cultural_engine = self.diagnostic_system.cultural_sensitivity
            if cultural_engine:
                # Assess cultural context
                cultural_profile = await cultural_engine.assess_cultural_context(
                    user_id, "", [], cultural_context
                )
                
                # Adapt therapeutic response (if available)
                adaptation = await cultural_engine.adapt_therapeutic_response(
                    user_id, diagnosis, "comprehensive_therapy", cultural_profile
                )
                
                return {
                    "cultural_profile": asdict(cultural_profile),
                    "adapted_approach": asdict(adaptation),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise RuntimeError("Cultural sensitivity engine not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in cultural adaptation: {str(e)}")
            raise
    
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
        if not self.diagnostic_system or not hasattr(self.diagnostic_system, 'adaptive_learning'):
            raise RuntimeError("Adaptive learning system not available")
        
        try:
            adaptive_engine = self.diagnostic_system.adaptive_learning
            if adaptive_engine:
                # Get available interventions
                available_interventions = [
                    "cognitive_behavioral_therapy", "dialectical_behavior_therapy",
                    "mindfulness_therapy", "acceptance_commitment_therapy"
                ]
                
                # Build current context
                current_context = {
                    "symptoms": diagnosis_result.symptoms,
                    "primary_diagnosis": diagnosis_result.primary_diagnosis,
                    "confidence_score": diagnosis_result.confidence_score
                }
                
                # Get personalized recommendations
                recommendations = await adaptive_engine.get_personalized_recommendation(
                    user_id, current_context, available_interventions
                )
                
                return recommendations
            else:
                raise RuntimeError("Adaptive learning engine not initialized")
                
        except Exception as e:
            self.logger.error(f"Error in personalized recommendations: {str(e)}")
            raise
    
    # Memory Integration Methods
    
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
        if not self.memory_system:
            return False
        
        try:
            # Store primary diagnosis insight
            if diagnosis_result.primary_diagnosis:
                insight_content = f"Diagnosis: {diagnosis_result.primary_diagnosis} (confidence: {diagnosis_result.confidence_score:.1%})"
                await self.memory_system.store_therapeutic_insight(
                    user_id, session_id, "diagnosis", insight_content,
                    {"diagnosis_result": asdict(diagnosis_result)}, 
                    diagnosis_result.confidence_score
                )
            
            # Store session memory
            session_data = {
                "diagnosis_type": diagnosis_result.diagnosis_type.value,
                "symptoms": diagnosis_result.symptoms,
                "recommendations": diagnosis_result.recommendations,
                "processing_time_ms": diagnosis_result.processing_time_ms
            }
            
            await self.memory_system.record_session_memory(user_id, session_id, session_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing diagnosis insights: {str(e)}")
            return False
    
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
        if not self.memory_system:
            return {}
        
        try:
            # Get contextual memory
            context = await self.memory_system.get_contextual_memory(
                user_id, "significant", lookback_days
            )
            
            # Get progress tracking
            progress = await self.memory_system.track_progress_milestones(user_id)
            
            return {
                "contextual_memory": context,
                "progress_tracking": progress,
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical context: {str(e)}")
            return {}
    
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
        if not self.memory_system:
            return {}
        
        try:
            continuity = await self.memory_system.get_session_continuity_context(
                user_id, session_id
            )
            return continuity
            
        except Exception as e:
            self.logger.error(f"Error getting session continuity: {str(e)}")
            return {}
    
    # Vector Database Integration Methods
    
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
        if not self.vector_db:
            return ""
        
        try:
            # Convert diagnosis result to storable format
            diagnosis_data = asdict(diagnosis_result)
            diagnosis_data["timestamp"] = diagnosis_result.timestamp.isoformat()
            
            # Store in vector database
            doc_id = await self.vector_db.add_document(
                collection_name="diagnosis",
                document=diagnosis_data,
                metadata={
                    "user_id": user_id,
                    "diagnosis_type": diagnosis_result.diagnosis_type.value,
                    "timestamp": diagnosis_result.timestamp.isoformat()
                }
            )
            
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error storing diagnosis vector: {str(e)}")
            return ""
    
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
        if not self.vector_db:
            return []
        
        try:
            # Create search query from symptoms
            query = " ".join(symptoms)
            
            # Search for similar cases
            results = await self.vector_db.search(
                collection_name="diagnosis",
                query=query,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar cases: {str(e)}")
            return []
    
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
        if not self.vector_db:
            return []
        
        try:
            # Search for user's diagnosis history
            results = await self.vector_db.search(
                collection_name="diagnosis",
                query="",
                filter={"user_id": user_id},
                limit=limit
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting user diagnosis history: {str(e)}")
            return []
    
    # Private helper methods
    
    async def _validate_system_integration(self) -> None:
        """Validate integration between all systems."""
        try:
            if self.diagnostic_system and hasattr(self.diagnostic_system, 'validate_system_integration'):
                validation_result = await self.diagnostic_system.validate_system_integration()
                if validation_result.get("overall_status") != "healthy":
                    self.logger.warning(f"System integration validation warning: {validation_result}")
        except Exception as e:
            self.logger.warning(f"Could not validate system integration: {str(e)}")
    
    async def _perform_comprehensive_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Perform comprehensive diagnosis using all available systems."""
        if not self.diagnostic_system:
            raise RuntimeError("Diagnostic system not available")
        
        # Get comprehensive diagnosis
        comprehensive_result = await self.diagnostic_system.generate_comprehensive_diagnosis(
            user_id=request.user_id,
            session_id=request.session_id,
            user_message=request.message,
            conversation_history=request.conversation_history,
            voice_emotion_data=request.voice_emotion_data,
            personality_data=request.personality_data,
            cultural_info=request.cultural_info
        )
        
        # Convert to DiagnosisResult format
        return self._convert_comprehensive_to_diagnosis_result(comprehensive_result, request)
    
    async def _perform_enhanced_integrated_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Perform enhanced integrated diagnosis."""
        # This would use the enhanced integrated system with all components
        return await self._perform_comprehensive_diagnosis(request)
    
    async def _perform_basic_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Perform basic diagnosis with limited components."""
        # Simplified diagnosis using basic components only
        return await self._perform_comprehensive_diagnosis(request)
    
    async def _perform_differential_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Perform differential diagnosis focusing on multiple conditions."""
        return await self._perform_comprehensive_diagnosis(request)
    
    async def _perform_temporal_diagnosis(self, request: DiagnosisRequest) -> DiagnosisResult:
        """Perform temporal diagnosis focusing on symptom progression."""
        return await self._perform_comprehensive_diagnosis(request)
    
    def _convert_comprehensive_to_diagnosis_result(self, 
                                                 comprehensive_result: Any, 
                                                 request: DiagnosisRequest) -> DiagnosisResult:
        """Convert comprehensive diagnosis result to DiagnosisResult format."""
        try:
            # Extract primary diagnosis
            primary_diagnosis = None
            confidence_score = 0.0
            
            if hasattr(comprehensive_result, 'primary_diagnosis') and comprehensive_result.primary_diagnosis:
                primary_diagnosis = comprehensive_result.primary_diagnosis.condition_name
                confidence_score = comprehensive_result.primary_diagnosis.confidence
            elif hasattr(comprehensive_result, 'differential_diagnoses') and comprehensive_result.differential_diagnoses:
                # Use the first diagnosis as primary
                primary_diagnosis = comprehensive_result.differential_diagnoses[0].condition_name
                confidence_score = comprehensive_result.differential_diagnoses[0].confidence
            
            # Determine confidence level
            if confidence_score >= 0.8:
                confidence_level = ConfidenceLevel.VERY_HIGH
            elif confidence_score >= 0.6:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence_score >= 0.4:
                confidence_level = ConfidenceLevel.MODERATE
            else:
                confidence_level = ConfidenceLevel.LOW
            
            # Extract symptoms and recommendations
            symptoms = []
            recommendations = []
            potential_conditions = []
            
            if hasattr(comprehensive_result, 'therapeutic_response') and comprehensive_result.therapeutic_response:
                recommendations.append(comprehensive_result.therapeutic_response.response_text)
            
            if hasattr(comprehensive_result, 'differential_diagnoses'):
                potential_conditions = [
                    {
                        "condition": d.condition_name,
                        "probability": d.probability,
                        "confidence": d.confidence,
                        "severity": d.severity
                    }
                    for d in comprehensive_result.differential_diagnoses
                ]
            
            # Build context updates
            context_updates = {}
            if hasattr(comprehensive_result, 'cultural_adaptations'):
                context_updates["cultural_adaptations"] = comprehensive_result.cultural_adaptations
            
            # Build memory insights
            memory_insights = []
            if hasattr(comprehensive_result, 'relevant_insights'):
                memory_insights = [
                    {"insight": insight.insight_content, "confidence": insight.confidence}
                    for insight in comprehensive_result.relevant_insights
                ]
            
            return DiagnosisResult(
                user_id=request.user_id,
                session_id=request.session_id,
                timestamp=comprehensive_result.timestamp if hasattr(comprehensive_result, 'timestamp') else datetime.now(),
                diagnosis_type=request.diagnosis_type,
                primary_diagnosis=primary_diagnosis,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                symptoms=symptoms,
                potential_conditions=potential_conditions,
                recommendations=recommendations,
                processing_time_ms=getattr(comprehensive_result, 'processing_time_ms', 0.0),
                warnings=getattr(comprehensive_result, 'warnings', []),
                limitations=getattr(comprehensive_result, 'limitations', []),
                context_updates=context_updates,
                memory_insights=memory_insights,
                raw_response=asdict(comprehensive_result) if hasattr(comprehensive_result, '__dict__') else {}
            )
            
        except Exception as e:
            self.logger.error(f"Error converting comprehensive result: {str(e)}")
            # Return a basic result
            return DiagnosisResult(
                user_id=request.user_id,
                session_id=request.session_id,
                timestamp=datetime.now(),
                diagnosis_type=request.diagnosis_type,
                primary_diagnosis=None,
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=0.0,
                symptoms=[],
                potential_conditions=[],
                recommendations=["Diagnosis processing encountered errors"],
                processing_time_ms=0.0,
                warnings=[f"Conversion error: {str(e)}"],
                limitations=["Unable to fully process diagnosis result"],
                context_updates={},
                memory_insights=[],
                raw_response={"error": str(e)}
            )
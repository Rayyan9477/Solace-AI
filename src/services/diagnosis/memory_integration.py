"""
Memory System Integration for Diagnosis Services

This module provides integration between the unified diagnosis system
and the existing memory and vector database systems.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import asdict

from .interfaces import DiagnosisResult, DiagnosisRequest
from src.infrastructure.di.container import Injectable
from src.utils.logger import get_logger

# Import memory and database systems with error handling
try:
    from src.memory.enhanced_memory_system import EnhancedMemorySystem, TherapeuticInsight
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    EnhancedMemorySystem = None
    TherapeuticInsight = None
    MEMORY_SYSTEM_AVAILABLE = False

try:
    from src.database.central_vector_db import CentralVectorDB
    VECTOR_DB_AVAILABLE = True
except ImportError:
    CentralVectorDB = None
    VECTOR_DB_AVAILABLE = False

try:
    from src.utils.vector_db_integration import add_user_data, get_user_data, search_relevant_data
    VECTOR_INTEGRATION_AVAILABLE = True
except ImportError:
    VECTOR_INTEGRATION_AVAILABLE = False

logger = get_logger(__name__)


class DiagnosisMemoryIntegrationService(Injectable):
    """
    Service that manages integration between diagnosis system and memory/vector database.
    
    This service handles storing diagnosis results, retrieving historical context,
    and managing continuity across diagnosis sessions.
    """
    
    def __init__(self, 
                 memory_system: Optional[EnhancedMemorySystem] = None,
                 vector_db: Optional[CentralVectorDB] = None):
        """
        Initialize the memory integration service.
        
        Args:
            memory_system: Enhanced memory system instance
            vector_db: Central vector database instance
        """
        self.memory_system = memory_system
        self.vector_db = vector_db
        self.logger = get_logger(__name__)
        
        # Service configuration
        self.max_historical_context_days = 90
        self.max_insights_per_query = 10
        self.memory_confidence_threshold = 0.5
        
        # Integration status
        self.memory_integration_enabled = MEMORY_SYSTEM_AVAILABLE and memory_system is not None
        self.vector_integration_enabled = VECTOR_DB_AVAILABLE and vector_db is not None
        self.utils_integration_enabled = VECTOR_INTEGRATION_AVAILABLE
    
    async def initialize(self) -> bool:
        """Initialize the integration service."""
        try:
            self.logger.info("Initializing DiagnosisMemoryIntegrationService")
            
            # Validate integrations
            if not any([self.memory_integration_enabled, self.vector_integration_enabled, self.utils_integration_enabled]):
                self.logger.warning("No memory or vector database integrations available")
            
            self.logger.info(f"Memory integration: {'enabled' if self.memory_integration_enabled else 'disabled'}")
            self.logger.info(f"Vector DB integration: {'enabled' if self.vector_integration_enabled else 'disabled'}")
            self.logger.info(f"Utils integration: {'enabled' if self.utils_integration_enabled else 'disabled'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory integration service: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the integration service."""
        try:
            self.logger.info("Shutting down DiagnosisMemoryIntegrationService")
            # No cleanup needed for this service
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    async def store_diagnosis_result(self, 
                                   diagnosis_result: DiagnosisResult,
                                   additional_context: Dict[str, Any] = None) -> Dict[str, bool]:
        """
        Store diagnosis result across all available storage systems.
        
        Args:
            diagnosis_result: Diagnosis result to store
            additional_context: Additional context to store
            
        Returns:
            Dictionary indicating success/failure for each storage system
        """
        storage_results = {
            "memory_system": False,
            "vector_db": False,
            "vector_utils": False
        }
        
        try:
            # Store in memory system
            if self.memory_integration_enabled:
                storage_results["memory_system"] = await self._store_in_memory_system(
                    diagnosis_result, additional_context
                )
            
            # Store in vector database
            if self.vector_integration_enabled:
                storage_results["vector_db"] = await self._store_in_vector_db(
                    diagnosis_result, additional_context
                )
            
            # Store using vector utils
            if self.utils_integration_enabled:
                storage_results["vector_utils"] = await self._store_with_vector_utils(
                    diagnosis_result, additional_context
                )
            
            self.logger.info(f"Stored diagnosis result for user {diagnosis_result.user_id}: {storage_results}")
            return storage_results
            
        except Exception as e:
            self.logger.error(f"Error storing diagnosis result: {str(e)}")
            return storage_results
    
    async def get_diagnosis_context(self, 
                                  diagnosis_request: DiagnosisRequest) -> Dict[str, Any]:
        """
        Get comprehensive context for a diagnosis request from all available sources.
        
        Args:
            diagnosis_request: Diagnosis request to get context for
            
        Returns:
            Comprehensive context dictionary
        """
        context = {
            "historical_diagnoses": [],
            "therapeutic_insights": [],
            "personality_data": {},
            "recent_sessions": [],
            "relevant_conversations": [],
            "progress_tracking": {},
            "context_sources": []
        }
        
        try:
            # Get context from memory system
            if self.memory_integration_enabled:
                memory_context = await self._get_memory_context(diagnosis_request)
                context.update(memory_context)
                context["context_sources"].append("memory_system")
            
            # Get context from vector database
            if self.vector_integration_enabled:
                vector_context = await self._get_vector_db_context(diagnosis_request)
                context.update(vector_context)
                context["context_sources"].append("vector_db")
            
            # Get context from vector utils
            if self.utils_integration_enabled:
                utils_context = await self._get_vector_utils_context(diagnosis_request)
                context.update(utils_context)
                context["context_sources"].append("vector_utils")
            
            self.logger.debug(f"Retrieved diagnosis context for user {diagnosis_request.user_id} from {len(context['context_sources'])} sources")
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting diagnosis context: {str(e)}")
            return context
    
    async def get_user_diagnosis_history(self, 
                                       user_id: str, 
                                       limit: int = 10,
                                       days_back: int = None) -> List[Dict[str, Any]]:
        """
        Get diagnosis history for a specific user from all available sources.
        
        Args:
            user_id: User identifier
            limit: Maximum number of results
            days_back: Number of days to look back (optional)
            
        Returns:
            List of historical diagnosis data
        """
        history = []
        
        try:
            # Get from memory system
            if self.memory_integration_enabled:
                memory_history = await self._get_memory_diagnosis_history(user_id, limit, days_back)
                history.extend(memory_history)
            
            # Get from vector database
            if self.vector_integration_enabled:
                vector_history = await self._get_vector_diagnosis_history(user_id, limit, days_back)
                history.extend(vector_history)
            
            # Get from vector utils
            if self.utils_integration_enabled:
                utils_history = await self._get_utils_diagnosis_history(user_id, limit, days_back)
                history.extend(utils_history)
            
            # Deduplicate and sort by timestamp
            history = self._deduplicate_and_sort_history(history, limit)
            
            self.logger.debug(f"Retrieved {len(history)} diagnosis history items for user {user_id}")
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting user diagnosis history: {str(e)}")
            return []
    
    async def search_similar_diagnoses(self, 
                                     diagnosis_result: DiagnosisResult,
                                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar diagnosis cases across all available sources.
        
        Args:
            diagnosis_result: Diagnosis result to find similar cases for
            limit: Maximum number of results
            
        Returns:
            List of similar diagnosis cases
        """
        similar_cases = []
        
        try:
            # Search query based on symptoms and diagnosis
            search_terms = []
            if diagnosis_result.symptoms:
                search_terms.extend(diagnosis_result.symptoms)
            if diagnosis_result.primary_diagnosis:
                search_terms.append(diagnosis_result.primary_diagnosis)
            
            search_query = " ".join(search_terms)
            
            # Search in vector database
            if self.vector_integration_enabled:
                vector_cases = await self._search_vector_db_diagnoses(search_query, limit)
                similar_cases.extend(vector_cases)
            
            # Search using vector utils
            if self.utils_integration_enabled:
                utils_cases = await self._search_utils_diagnoses(search_query, limit)
                similar_cases.extend(utils_cases)
            
            # Deduplicate and limit results
            similar_cases = self._deduplicate_similar_cases(similar_cases, limit)
            
            self.logger.debug(f"Found {len(similar_cases)} similar diagnosis cases")
            return similar_cases
            
        except Exception as e:
            self.logger.error(f"Error searching similar diagnoses: {str(e)}")
            return []
    
    # Private methods for memory system integration
    
    async def _store_in_memory_system(self, 
                                    diagnosis_result: DiagnosisResult,
                                    additional_context: Dict[str, Any] = None) -> bool:
        """Store diagnosis result in the memory system."""
        try:
            if not self.memory_system:
                return False
            
            # Store primary diagnosis as therapeutic insight
            if diagnosis_result.primary_diagnosis:
                insight_content = f"Diagnosis: {diagnosis_result.primary_diagnosis} (confidence: {diagnosis_result.confidence_score:.1%})"
                
                await self.memory_system.store_therapeutic_insight(
                    user_id=diagnosis_result.user_id,
                    session_id=diagnosis_result.session_id,
                    insight_type="diagnosis",
                    insight_content=insight_content,
                    metadata={
                        "diagnosis_result": asdict(diagnosis_result),
                        "additional_context": additional_context or {}
                    },
                    confidence=diagnosis_result.confidence_score
                )
            
            # Store session memory
            session_data = {
                "diagnosis_type": diagnosis_result.diagnosis_type.value,
                "primary_diagnosis": diagnosis_result.primary_diagnosis,
                "symptoms": diagnosis_result.symptoms,
                "recommendations": diagnosis_result.recommendations,
                "confidence_score": diagnosis_result.confidence_score,
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "timestamp": diagnosis_result.timestamp.isoformat()
            }
            
            await self.memory_system.record_session_memory(
                user_id=diagnosis_result.user_id,
                session_id=diagnosis_result.session_id,
                session_data=session_data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in memory system: {str(e)}")
            return False
    
    async def _store_in_vector_db(self, 
                                diagnosis_result: DiagnosisResult,
                                additional_context: Dict[str, Any] = None) -> bool:
        """Store diagnosis result in the vector database."""
        try:
            if not self.vector_db:
                return False
            
            # Convert diagnosis result to storable format
            diagnosis_data = asdict(diagnosis_result)
            diagnosis_data["timestamp"] = diagnosis_result.timestamp.isoformat()
            diagnosis_data["additional_context"] = additional_context or {}
            
            # Store in vector database
            doc_id = await self.vector_db.add_document(
                collection_name="diagnosis_results",
                document=diagnosis_data,
                metadata={
                    "user_id": diagnosis_result.user_id,
                    "session_id": diagnosis_result.session_id,
                    "diagnosis_type": diagnosis_result.diagnosis_type.value,
                    "timestamp": diagnosis_result.timestamp.isoformat(),
                    "primary_diagnosis": diagnosis_result.primary_diagnosis or "none"
                }
            )
            
            return bool(doc_id)
            
        except Exception as e:
            self.logger.error(f"Error storing in vector database: {str(e)}")
            return False
    
    async def _store_with_vector_utils(self, 
                                     diagnosis_result: DiagnosisResult,
                                     additional_context: Dict[str, Any] = None) -> bool:
        """Store diagnosis result using vector utils."""
        try:
            # Prepare diagnosis data
            diagnosis_data = {
                "user_id": diagnosis_result.user_id,
                "session_id": diagnosis_result.session_id,
                "timestamp": diagnosis_result.timestamp.isoformat(),
                "diagnosis_type": diagnosis_result.diagnosis_type.value,
                "primary_diagnosis": diagnosis_result.primary_diagnosis,
                "confidence_score": diagnosis_result.confidence_score,
                "symptoms": diagnosis_result.symptoms,
                "potential_conditions": diagnosis_result.potential_conditions,
                "recommendations": diagnosis_result.recommendations,
                "processing_time_ms": diagnosis_result.processing_time_ms,
                "additional_context": additional_context or {}
            }
            
            # Store using vector utils
            doc_id = add_user_data("diagnosis", diagnosis_data)
            
            return bool(doc_id)
            
        except Exception as e:
            self.logger.error(f"Error storing with vector utils: {str(e)}")
            return False
    
    # Private methods for retrieving context
    
    async def _get_memory_context(self, diagnosis_request: DiagnosisRequest) -> Dict[str, Any]:
        """Get context from memory system."""
        context = {}
        
        try:
            if not self.memory_system:
                return context
            
            # Get therapeutic insights
            contextual_memory = await self.memory_system.get_contextual_memory(
                user_id=diagnosis_request.user_id,
                context_type="significant",
                lookback_days=self.max_historical_context_days
            )
            
            if contextual_memory.get("significant_insights"):
                context["therapeutic_insights"] = contextual_memory["significant_insights"][:self.max_insights_per_query]
            
            # Get session continuity
            session_continuity = await self.memory_system.get_session_continuity_context(
                user_id=diagnosis_request.user_id,
                session_id=diagnosis_request.session_id
            )
            
            if session_continuity:
                context["session_continuity"] = session_continuity
            
            # Get progress tracking
            progress_tracking = await self.memory_system.track_progress_milestones(
                user_id=diagnosis_request.user_id
            )
            
            if progress_tracking:
                context["progress_tracking"] = progress_tracking
            
        except Exception as e:
            self.logger.error(f"Error getting memory context: {str(e)}")
        
        return context
    
    async def _get_vector_db_context(self, diagnosis_request: DiagnosisRequest) -> Dict[str, Any]:
        """Get context from vector database."""
        context = {}
        
        try:
            if not self.vector_db:
                return context
            
            # Search for relevant diagnoses
            search_query = diagnosis_request.message
            relevant_diagnoses = await self.vector_db.search(
                collection_name="diagnosis_results",
                query=search_query,
                filter={"user_id": diagnosis_request.user_id},
                limit=5
            )
            
            if relevant_diagnoses:
                context["historical_diagnoses"] = relevant_diagnoses
            
        except Exception as e:
            self.logger.error(f"Error getting vector DB context: {str(e)}")
        
        return context
    
    async def _get_vector_utils_context(self, diagnosis_request: DiagnosisRequest) -> Dict[str, Any]:
        """Get context from vector utils."""
        context = {}
        
        try:
            # Get relevant conversations
            relevant_conversations = search_relevant_data(
                query=diagnosis_request.message,
                data_types=["conversation"],
                limit=3
            )
            
            if relevant_conversations:
                context["relevant_conversations"] = relevant_conversations
            
            # Get latest diagnosis
            latest_diagnosis = get_user_data("diagnosis")
            if latest_diagnosis:
                context["latest_diagnosis"] = latest_diagnosis
            
            # Get personality data
            personality_data = get_user_data("personality")
            if personality_data:
                context["personality_data"] = personality_data
            
        except Exception as e:
            self.logger.error(f"Error getting vector utils context: {str(e)}")
        
        return context
    
    # Private methods for getting diagnosis history
    
    async def _get_memory_diagnosis_history(self, 
                                          user_id: str, 
                                          limit: int, 
                                          days_back: int = None) -> List[Dict[str, Any]]:
        """Get diagnosis history from memory system."""
        history = []
        
        try:
            if not self.memory_system:
                return history
            
            # Get contextual memory for diagnosis insights
            lookback_days = days_back or self.max_historical_context_days
            contextual_memory = await self.memory_system.get_contextual_memory(
                user_id=user_id,
                context_type="diagnosis",
                lookback_days=lookback_days
            )
            
            if contextual_memory.get("diagnosis_insights"):
                history.extend(contextual_memory["diagnosis_insights"][:limit])
            
        except Exception as e:
            self.logger.error(f"Error getting memory diagnosis history: {str(e)}")
        
        return history
    
    async def _get_vector_diagnosis_history(self, 
                                          user_id: str, 
                                          limit: int, 
                                          days_back: int = None) -> List[Dict[str, Any]]:
        """Get diagnosis history from vector database."""
        history = []
        
        try:
            if not self.vector_db:
                return history
            
            # Build filter
            filter_dict = {"user_id": user_id}
            
            if days_back:
                cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
                filter_dict["timestamp"] = {"$gte": cutoff_date}
            
            # Search for user's diagnosis history
            results = await self.vector_db.search(
                collection_name="diagnosis_results",
                query="",
                filter=filter_dict,
                limit=limit
            )
            
            history.extend(results)
            
        except Exception as e:
            self.logger.error(f"Error getting vector diagnosis history: {str(e)}")
        
        return history
    
    async def _get_utils_diagnosis_history(self, 
                                         user_id: str, 
                                         limit: int, 
                                         days_back: int = None) -> List[Dict[str, Any]]:
        """Get diagnosis history from vector utils."""
        history = []
        
        try:
            # Get diagnosis data (vector utils may not support complex filtering)
            diagnosis_data = get_user_data("diagnosis")
            if diagnosis_data:
                # Filter by date if specified
                if days_back:
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    if isinstance(diagnosis_data, list):
                        diagnosis_data = [
                            item for item in diagnosis_data
                            if datetime.fromisoformat(item.get("timestamp", "2000-01-01")) >= cutoff_date
                        ]
                
                # Add to history
                if isinstance(diagnosis_data, list):
                    history.extend(diagnosis_data[:limit])
                else:
                    history.append(diagnosis_data)
            
        except Exception as e:
            self.logger.error(f"Error getting utils diagnosis history: {str(e)}")
        
        return history
    
    # Private methods for searching similar diagnoses
    
    async def _search_vector_db_diagnoses(self, search_query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for similar diagnoses in vector database."""
        results = []
        
        try:
            if not self.vector_db:
                return results
            
            # Search for similar diagnosis cases
            search_results = await self.vector_db.search(
                collection_name="diagnosis_results",
                query=search_query,
                limit=limit
            )
            
            results.extend(search_results)
            
        except Exception as e:
            self.logger.error(f"Error searching vector DB diagnoses: {str(e)}")
        
        return results
    
    async def _search_utils_diagnoses(self, search_query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for similar diagnoses using vector utils."""
        results = []
        
        try:
            # Search for relevant diagnosis data
            relevant_diagnoses = search_relevant_data(
                query=search_query,
                data_types=["diagnosis"],
                limit=limit
            )
            
            if relevant_diagnoses:
                results.extend(relevant_diagnoses)
                
        except Exception as e:
            self.logger.error(f"Error searching utils diagnoses: {str(e)}")
        
        return results
    
    # Utility methods
    
    def _deduplicate_and_sort_history(self, history: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Deduplicate and sort history items by timestamp."""
        try:
            # Remove duplicates based on session_id or timestamp
            seen_sessions = set()
            unique_history = []
            
            for item in history:
                session_id = item.get("session_id")
                if session_id and session_id not in seen_sessions:
                    seen_sessions.add(session_id)
                    unique_history.append(item)
                elif not session_id:
                    unique_history.append(item)
            
            # Sort by timestamp (newest first)
            unique_history.sort(
                key=lambda x: datetime.fromisoformat(x.get("timestamp", "2000-01-01")),
                reverse=True
            )
            
            return unique_history[:limit]
            
        except Exception as e:
            self.logger.error(f"Error deduplicating history: {str(e)}")
            return history[:limit]
    
    def _deduplicate_similar_cases(self, cases: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Deduplicate similar cases and limit results."""
        try:
            # Simple deduplication based on primary diagnosis and symptoms
            seen_cases = set()
            unique_cases = []
            
            for case in cases:
                # Create a simple signature for the case
                diagnosis = case.get("primary_diagnosis", "")
                symptoms = str(sorted(case.get("symptoms", [])))
                signature = f"{diagnosis}_{symptoms}"
                
                if signature not in seen_cases:
                    seen_cases.add(signature)
                    unique_cases.append(case)
            
            return unique_cases[:limit]
            
        except Exception as e:
            self.logger.error(f"Error deduplicating similar cases: {str(e)}")
            return cases[:limit]
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        return {
            "memory_integration_enabled": self.memory_integration_enabled,
            "vector_integration_enabled": self.vector_integration_enabled,
            "utils_integration_enabled": self.utils_integration_enabled,
            "memory_system_available": MEMORY_SYSTEM_AVAILABLE,
            "vector_db_available": VECTOR_DB_AVAILABLE,
            "vector_utils_available": VECTOR_INTEGRATION_AVAILABLE,
            "memory_system_instance": self.memory_system is not None,
            "vector_db_instance": self.vector_db is not None
        }
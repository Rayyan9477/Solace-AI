"""
Vector database integration for therapeutic techniques.

This module provides functionality to store and retrieve therapeutic techniques
using vector embeddings for semantic similarity search.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime

import numpy as np
from src.database.vector_store import VectorStore
from src.knowledge.therapeutic.knowledge_base import TherapeuticKnowledgeBase
from src.database.therapeutic_friction_vector_manager import TherapeuticFrictionVectorManager, TherapeuticDocument, TherapeuticFrictionDocumentType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TherapeuticTechniqueService:
    """Service for managing and retrieving therapeutic techniques using vector embeddings."""
    
    def __init__(self, model_provider=None, config: Dict[str, Any] = None):
        """Initialize the therapeutic technique service.
        
        Args:
            model_provider: LLM provider for generating embeddings
            config: Configuration for enhanced features
        """
        self.knowledge_base = TherapeuticKnowledgeBase()
        self.model_provider = model_provider
        self.config = config or {}
        
        # Initialize vector store
        vector_store_path = os.path.join("src", "data", "vector_store", "therapeutic_techniques")
        os.makedirs(vector_store_path, exist_ok=True)
        
        self.vector_store = VectorStore(
            collection_name="therapeutic_techniques",
            vector_dimensions=768,  # Default embedding dimensions
            storage_path=vector_store_path
        )
        
        # Initialize enhanced therapeutic friction vector manager
        self.friction_vector_manager = TherapeuticFrictionVectorManager(
            model_provider=model_provider,
            config=self.config.get("friction_vector_manager", {})
        )
        
    def initialize_vector_store(self):
        """Initialize the vector store with therapeutic techniques."""
        techniques = self.knowledge_base.get_all_techniques()
        
        if not techniques:
            logger.warning("No therapeutic techniques found to index")
            return
            
        # Check if vector store is already populated
        if self.vector_store.count() >= len(techniques):
            logger.info(f"Vector store already contains {self.vector_store.count()} techniques")
            return
            
        # Clear existing data and repopulate
        self.vector_store.clear()
        
        for technique in techniques:
            # Create text representation for embedding
            text_to_embed = f"{technique['name']} - {technique['category']}: {technique['description']}"
            
            if technique.get('emotions'):
                text_to_embed += f" Emotions: {', '.join(technique['emotions'])}"
                
            # Generate embedding using the model provider or fallback to random embeddings for testing
            embedding = self._get_embedding(text_to_embed)
            
            # Store in vector database with technique ID as metadata
            self.vector_store.add_item(
                vector=embedding,
                metadata={"id": technique["id"]}
            )
            
        logger.info(f"Indexed {len(techniques)} therapeutic techniques in vector store")
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the model provider or fallback to random for testing."""
        if self.model_provider and hasattr(self.model_provider, "get_embedding"):
            try:
                return self.model_provider.get_embedding(text)
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                
        # Fallback to random embeddings for testing (replace in production)
        return list(np.random.rand(768).astype(float))
        
    def get_relevant_techniques(self, query: str, emotion: str = None, 
                               category: str = None, top_k: int = 3) -> List[Dict]:
        """Get therapeutic techniques relevant to the user's query/emotion.
        
        Args:
            query: User's query or context
            emotion: Detected emotion (optional)
            category: Specific therapeutic category (optional)
            top_k: Number of techniques to return
            
        Returns:
            List of relevant therapeutic techniques
        """
        # Combine query with emotion if available
        search_text = query
        if emotion:
            search_text = f"{query} {emotion}"
            
        # Get embedding for the search text
        embedding = self._get_embedding(search_text)
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=embedding,
            top_k=top_k * 2  # Get more results than needed for filtering
        )
        
        # Filter results if category specified
        techniques = []
        for result in results:
            technique_id = result.get("metadata", {}).get("id")
            if not technique_id:
                continue
                
            technique = self.knowledge_base.get_technique_by_id(technique_id)
            if not technique:
                continue
                
            # Apply category filter if specified
            if category and technique.get("category", "").lower() != category.lower():
                continue
                
            techniques.append(technique)
            
            # Break if we have enough techniques
            if len(techniques) >= top_k:
                break
                
        return techniques
        
    def format_techniques_for_response(self, techniques: List[Dict]) -> str:
        """Format therapeutic techniques into a response string.
        
        Args:
            techniques: List of therapeutic technique objects
            
        Returns:
            Formatted string with therapeutic techniques
        """
        if not techniques:
            return ""
            
        result = "# Practical Therapeutic Steps You Can Try\n\n"
        
        for technique in techniques:
            result += self.knowledge_base.format_technique_steps(technique)
            result += "\n---\n\n"
            
        return result.strip()
    
    # Enhanced methods for therapeutic friction integration
    
    async def get_friction_specific_techniques(self, agent_type: str, user_context: Dict[str, Any],
                                             top_k: int = 3) -> List[Dict[str, Any]]:
        """Get techniques specific to therapeutic friction analysis."""
        try:
            # Get recommendations from friction vector manager
            friction_recommendations = await self.friction_vector_manager.get_recommendations_for_agent(
                agent_type=agent_type,
                user_context=user_context,
                top_k=top_k
            )
            
            # Combine with traditional techniques
            traditional_query = self._build_traditional_query(user_context)
            traditional_techniques = self.get_relevant_techniques(
                query=traditional_query,
                emotion=user_context.get("emotion_analysis", {}).get("primary_emotion"),
                top_k=top_k
            )
            
            # Merge and rank results
            combined_results = self._merge_friction_and_traditional_techniques(
                friction_recommendations, traditional_techniques, user_context
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting friction-specific techniques: {str(e)}")
            return []
    
    async def store_friction_knowledge(self, knowledge_data: Dict[str, Any]) -> bool:
        """Store specialized therapeutic friction knowledge."""
        try:
            # Create therapeutic document
            document = TherapeuticDocument(
                document_id=f"friction_knowledge_{knowledge_data.get('id', 'unknown')}",
                document_type=TherapeuticFrictionDocumentType(knowledge_data.get('type', 'therapeutic_technique')),
                title=knowledge_data.get('title', 'Therapeutic Knowledge'),
                content=knowledge_data.get('content', ''),
                metadata={
                    'source': knowledge_data.get('source', 'therapeutic_service'),
                    'effectiveness_rating': knowledge_data.get('effectiveness_rating', 0.5),
                    'evidence_level': knowledge_data.get('evidence_level', 'clinical'),
                    'therapeutic_modalities': knowledge_data.get('therapeutic_modalities', []),
                    'keywords': knowledge_data.get('keywords', [])
                }
            )
            
            # Store in friction vector manager
            success = await self.friction_vector_manager.add_document(document)
            
            if success:
                logger.info(f"Stored friction knowledge: {document.document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing friction knowledge: {str(e)}")
            return False
    
    async def get_cross_domain_insights(self, query: str, primary_domain: str) -> Dict[str, Any]:
        """Get cross-domain therapeutic insights."""
        try:
            # Map domain string to enum
            domain_mapping = {
                'readiness': TherapeuticFrictionDocumentType.READINESS_PATTERN,
                'breakthrough': TherapeuticFrictionDocumentType.BREAKTHROUGH_INDICATOR,
                'relationship': TherapeuticFrictionDocumentType.RELATIONSHIP_DYNAMIC,
                'intervention': TherapeuticFrictionDocumentType.INTERVENTION_OUTCOME,
                'progress': TherapeuticFrictionDocumentType.PROGRESS_TRAJECTORY
            }
            
            primary_domain_type = domain_mapping.get(primary_domain, TherapeuticFrictionDocumentType.THERAPEUTIC_TECHNIQUE)
            
            # Perform cross-domain search
            insights = await self.friction_vector_manager.cross_domain_search(
                query=query,
                primary_domain=primary_domain_type,
                top_k=3
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting cross-domain insights: {str(e)}")
            return {}
    
    async def update_technique_effectiveness(self, technique_id: str, effectiveness_data: Dict[str, Any]) -> bool:
        """Update technique effectiveness based on outcomes."""
        try:
            # Store effectiveness data as new knowledge
            effectiveness_knowledge = {
                'id': f"{technique_id}_effectiveness_{int(datetime.now().timestamp())}",
                'type': 'intervention_outcome',
                'title': f"Effectiveness Data for {technique_id}",
                'content': f"Effectiveness data: {json.dumps(effectiveness_data)}",
                'source': 'outcome_tracking',
                'effectiveness_rating': effectiveness_data.get('rating', 0.5),
                'evidence_level': 'clinical_observation',
                'keywords': ['effectiveness', 'outcome', technique_id]
            }
            
            success = await self.store_friction_knowledge(effectiveness_knowledge)
            
            if success:
                logger.info(f"Updated effectiveness data for technique {technique_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating technique effectiveness: {str(e)}")
            return False
    
    def _build_traditional_query(self, user_context: Dict[str, Any]) -> str:
        """Build query for traditional technique retrieval."""
        query_parts = []
        
        # Add emotional context
        emotion = user_context.get("emotion_analysis", {}).get("primary_emotion")
        if emotion:
            query_parts.append(emotion)
        
        # Add readiness context
        readiness = user_context.get("user_readiness")
        if readiness:
            query_parts.append(f"{readiness} client")
        
        # Add therapeutic focus
        focus_areas = user_context.get("therapeutic_focus", [])
        if focus_areas:
            query_parts.extend(focus_areas[:2])  # Limit to prevent query bloat
        
        return " ".join(query_parts) if query_parts else "therapeutic technique"
    
    def _merge_friction_and_traditional_techniques(self, friction_recommendations: List[Dict[str, Any]],
                                                 traditional_techniques: List[Dict[str, Any]],
                                                 user_context: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge friction-specific and traditional techniques."""
        merged_results = []
        
        # Add friction recommendations with source tag
        for rec in friction_recommendations:
            rec['source_type'] = 'friction_specific'
            rec['relevance_score'] = rec.get('similarity_score', 0.0) * 1.1  # Slight boost for friction-specific
            merged_results.append(rec)
        
        # Add traditional techniques with source tag
        for tech in traditional_techniques:
            technique_result = {
                'document': {
                    'title': tech.get('name', 'Unknown Technique'),
                    'content': self.knowledge_base.format_technique_steps(tech),
                    'metadata': {
                        'category': tech.get('category', 'general'),
                        'difficulty': tech.get('difficulty', 'medium'),
                        'emotions': tech.get('emotions', []),
                        'steps': tech.get('steps', [])
                    }
                },
                'source_type': 'traditional',
                'relevance_score': 0.7,  # Default relevance for traditional techniques
                'similarity_score': 0.7,
                'match_reason': 'Traditional therapeutic technique match'
            }
            merged_results.append(technique_result)
        
        # Sort by relevance score
        merged_results.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        
        return merged_results
    
    async def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including friction vector manager data."""
        try:
            # Get traditional statistics
            traditional_stats = {
                'total_techniques': len(self.knowledge_base.get_all_techniques()),
                'vector_store_count': self.vector_store.count() if hasattr(self.vector_store, 'count') else 0
            }
            
            # Get friction vector manager statistics
            friction_stats = await self.friction_vector_manager.get_domain_statistics()
            
            # Get health status
            health_status = await self.friction_vector_manager.get_health_status()
            
            return {
                'traditional_techniques': traditional_stats,
                'friction_knowledge_domains': friction_stats,
                'system_health': health_status,
                'integration_status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced statistics: {str(e)}")
            return {'error': str(e)}
    
    async def initialize_friction_knowledge_base(self):
        """Initialize the friction knowledge base with default knowledge."""
        try:
            # Sample readiness patterns
            readiness_patterns = [
                {
                    'id': 'resistant_pattern_1',
                    'type': 'readiness_pattern',
                    'title': 'Verbal Resistance Indicators',
                    'content': 'Common verbal patterns indicating therapeutic resistance: "won\'t work", "tried everything", "pointless"',
                    'keywords': ['resistance', 'verbal_patterns', 'therapeutic_alliance'],
                    'therapeutic_modalities': ['motivational_interviewing', 'client_centered'],
                    'effectiveness_rating': 0.8
                },
                {
                    'id': 'motivated_pattern_1',
                    'type': 'readiness_pattern',
                    'title': 'Motivation Indicators',
                    'content': 'Language patterns indicating high motivation: "want to change", "ready to try", "committed to this"',
                    'keywords': ['motivation', 'readiness', 'commitment'],
                    'therapeutic_modalities': ['cbt', 'behavioral_activation'],
                    'effectiveness_rating': 0.9
                }
            ]
            
            # Sample breakthrough indicators
            breakthrough_indicators = [
                {
                    'id': 'cognitive_breakthrough_1',
                    'type': 'breakthrough_indicator',
                    'title': 'Cognitive Insight Markers',
                    'content': 'Indicators of cognitive breakthroughs: "I never realized", "it just clicked", "now I understand"',
                    'keywords': ['insight', 'cognitive', 'breakthrough', 'understanding'],
                    'therapeutic_modalities': ['cbt', 'psychodynamic'],
                    'effectiveness_rating': 0.85
                }
            ]
            
            # Store initial knowledge
            all_knowledge = readiness_patterns + breakthrough_indicators
            
            for knowledge_item in all_knowledge:
                await self.store_friction_knowledge(knowledge_item)
            
            logger.info(f"Initialized friction knowledge base with {len(all_knowledge)} items")
            
        except Exception as e:
            logger.error(f"Error initializing friction knowledge base: {str(e)}")
    
    async def cleanup_resources(self):
        """Clean up resources and connections."""
        try:
            # Any cleanup needed for vector stores or connections
            logger.info("Therapeutic technique service resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")
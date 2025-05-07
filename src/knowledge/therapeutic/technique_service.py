"""
Vector database integration for therapeutic techniques.

This module provides functionality to store and retrieve therapeutic techniques
using vector embeddings for semantic similarity search.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.database.vector_store import VectorStore
from src.knowledge.therapeutic.knowledge_base import TherapeuticKnowledgeBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TherapeuticTechniqueService:
    """Service for managing and retrieving therapeutic techniques using vector embeddings."""
    
    def __init__(self, model_provider=None):
        """Initialize the therapeutic technique service.
        
        Args:
            model_provider: LLM provider for generating embeddings
        """
        self.knowledge_base = TherapeuticKnowledgeBase()
        self.model_provider = model_provider
        
        # Initialize vector store
        vector_store_path = os.path.join("src", "data", "vector_store", "therapeutic_techniques")
        os.makedirs(vector_store_path, exist_ok=True)
        
        self.vector_store = VectorStore(
            collection_name="therapeutic_techniques",
            vector_dimensions=768,  # Default embedding dimensions
            storage_path=vector_store_path
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
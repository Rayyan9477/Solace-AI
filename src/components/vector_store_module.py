"""
Vector Store Module

Provides vector storage and retrieval capabilities for the application.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import os
from pathlib import Path

from src.components.base_module import Module
from src.config.settings import AppConfig

class VectorStoreModule(Module):
    """
    Vector Store Module for the Contextual-Chatbot.
    
    Provides vector storage and retrieval capabilities:
    - Document storage and retrieval
    - Semantic search
    - Vector storage management
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.vector_store = None
        self.storage_path = None
        self.embedding_dimension = 768  # Default embedding dimension
        
        # Initialize config
        self._load_config()
    
    def _load_config(self):
        """Load configuration values"""
        if not self.config:
            return
            
        self.storage_path = self.config.get("storage_path", os.path.join(
            Path(__file__).parents[2], "data", "vector_store"
        ))
        self.embedding_dimension = self.config.get("embedding_dimension", 768)
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the module"""
        await super().initialize()
        
        try:
            # Try to import the vector store
            try:
                from src.database.vector_store import VectorStore
                
                self.vector_store = VectorStore(
                    storage_path=self.storage_path,
                    embedding_dimension=self.embedding_dimension,
                    config=self.config
                )
                
                self.logger.info(f"Initialized vector store at {self.storage_path}")
                self._register_services()
                return True
                
            except ImportError as e:
                self.logger.error(f"Vector store not available: {str(e)}")
                self.health_status = "degraded"
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store module: {str(e)}")
            self.health_status = "failed"
            return False
    
    def _register_services(self):
        """Register services provided by this module"""
        self.expose_service("add_document", self.add_document)
        self.expose_service("add_documents", self.add_documents)
        self.expose_service("search", self.search)
        self.expose_service("delete_document", self.delete_document)
        self.expose_service("get_collection_info", self.get_collection_info)
    
    async def add_document(self, collection_name: str, document: Dict[str, Any], 
                         embedding: Optional[List[float]] = None) -> Optional[str]:
        """
        Add a document to a collection
        
        Args:
            collection_name: Name of the collection
            document: Document to add
            embedding: Optional pre-computed embedding
            
        Returns:
            Document ID or None if failed
        """
        if not self.initialized:
            self.logger.warning("Vector store module not initialized")
            return None
        
        try:
            if self.vector_store and hasattr(self.vector_store, "add_document"):
                doc_id = await self.vector_store.add_document(collection_name, document, embedding)
                return doc_id
            else:
                self.logger.error("Vector store does not support adding documents")
                return None
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return None
    
    async def add_documents(self, collection_name: str, documents: List[Dict[str, Any]], 
                          embeddings: Optional[List[List[float]]] = None) -> List[Optional[str]]:
        """
        Add multiple documents to a collection
        
        Args:
            collection_name: Name of the collection
            documents: List of documents to add
            embeddings: Optional list of pre-computed embeddings
            
        Returns:
            List of document IDs or None values for failures
        """
        if not self.initialized:
            self.logger.warning("Vector store module not initialized")
            return [None] * len(documents)
        
        try:
            if self.vector_store and hasattr(self.vector_store, "add_documents"):
                doc_ids = await self.vector_store.add_documents(collection_name, documents, embeddings)
                return doc_ids
            else:
                # Fallback to adding one by one
                result = []
                for i, doc in enumerate(documents):
                    embedding = embeddings[i] if embeddings and i < len(embeddings) else None
                    doc_id = await self.add_document(collection_name, doc, embedding)
                    result.append(doc_id)
                return result
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            return [None] * len(documents)
    
    async def search(self, collection_name: str, query: str, 
                   top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents in a collection
        
        Args:
            collection_name: Name of the collection
            query: Query string
            top_k: Number of results to return
            embedding: Optional pre-computed query embedding
            
        Returns:
            List of document dictionaries with similarity scores
        """
        if not self.initialized:
            self.logger.warning("Vector store module not initialized")
            return []
        
        try:
            if self.vector_store and hasattr(self.vector_store, "search"):
                results = await self.vector_store.search(collection_name, query, top_k, embedding)
                return results
            else:
                self.logger.error("Vector store does not support search")
                return []
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """
        Delete a document from a collection
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to delete
            
        Returns:
            Success status
        """
        if not self.initialized:
            self.logger.warning("Vector store module not initialized")
            return False
        
        try:
            if self.vector_store and hasattr(self.vector_store, "delete_document"):
                success = await self.vector_store.delete_document(collection_name, document_id)
                return success
            else:
                self.logger.error("Vector store does not support document deletion")
                return False
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        if not self.initialized:
            self.logger.warning("Vector store module not initialized")
            return {}
        
        try:
            if self.vector_store and hasattr(self.vector_store, "get_collection_info"):
                info = await self.vector_store.get_collection_info(collection_name)
                return info
            else:
                self.logger.error("Vector store does not support collection info")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    async def shutdown(self) -> bool:
        """Shutdown the module"""
        if self.vector_store and hasattr(self.vector_store, "shutdown"):
            await self.vector_store.shutdown()
        
        return await super().shutdown()

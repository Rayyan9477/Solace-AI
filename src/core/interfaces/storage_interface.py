"""
Abstract interface for Storage providers.

This interface ensures all storage providers implement the same methods,
allowing easy swapping between different storage backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class StorageType(Enum):
    """Enum for different types of storage."""
    VECTOR = "vector"
    DOCUMENT = "document"
    KEY_VALUE = "key_value"
    CONVERSATION = "conversation"


@dataclass
class StorageConfig:
    """Configuration for storage providers."""
    storage_type: StorageType
    connection_string: Optional[str] = None
    storage_path: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class DocumentMetadata:
    """Metadata for stored documents."""
    document_id: str
    timestamp: float
    content_type: str
    tags: List[str] = None
    user_id: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    document_id: str
    content: str
    similarity_score: float
    metadata: DocumentMetadata


class StorageInterface(ABC):
    """
    Abstract base class for all storage providers.
    
    This interface defines the contract that all storage providers must implement,
    ensuring consistent behavior regardless of the underlying storage system.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize the storage provider with configuration."""
        self.config = config
        self._initialized = False
    
    @property
    @abstractmethod
    def storage_type(self) -> StorageType:
        """Return the type of storage this provider handles."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the storage provider.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def store_document(
        self,
        document_id: str,
        content: Union[str, Dict[str, Any]],
        metadata: Optional[DocumentMetadata] = None
    ) -> bool:
        """
        Store a document in the storage system.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content (text or structured data)
            metadata: Optional metadata for the document
            
        Returns:
            bool: True if storage was successful
        """
        pass
    
    @abstractmethod
    async def retrieve_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Dict containing document content and metadata, or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from storage.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            bool: True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents based on query and filters.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of matching documents
        """
        pass
    
    @abstractmethod
    async def list_documents(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        List document IDs with optional filtering.
        
        Args:
            offset: Number of documents to skip
            limit: Maximum number of document IDs to return
            filters: Optional filters to apply
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage system.
        
        Returns:
            Dict containing storage statistics
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the storage provider is initialized."""
        return self._initialized
    
    @property
    def provider_name(self) -> str:
        """Get the name of the storage provider."""
        return self.__class__.__name__
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the storage provider.
        
        Returns:
            Dict containing health status information
        """
        try:
            stats = await self.get_storage_stats()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "storage_type": self.storage_type.value,
                "initialized": self.is_initialized,
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "storage_type": self.storage_type.value,
                "error": str(e),
                "initialized": self.is_initialized
            }


class VectorStorageInterface(StorageInterface):
    """
    Extended interface for vector storage providers.
    
    Adds vector-specific operations for similarity search and embeddings.
    """
    
    @property
    def storage_type(self) -> StorageType:
        """Vector storage type."""
        return StorageType.VECTOR
    
    @abstractmethod
    async def store_vector(
        self,
        document_id: str,
        vector: List[float],
        content: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> bool:
        """
        Store a vector with associated content and metadata.
        
        Args:
            document_id: Unique identifier for the document
            vector: The vector embedding
            content: Original text content
            metadata: Optional metadata
            
        Returns:
            bool: True if storage was successful
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform similarity search using a query vector.
        
        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            filters: Optional filters to apply
            
        Returns:
            List of search results with similarity scores
        """
        pass
    
    @abstractmethod
    async def get_vector_dimensions(self) -> int:
        """
        Get the dimensionality of vectors stored in this system.
        
        Returns:
            int: Vector dimensions
        """
        pass
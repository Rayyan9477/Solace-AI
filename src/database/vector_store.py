import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Vector store backends
import faiss
from qdrant_client import QdrantClient
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

from config.settings import AppConfig

logger = logging.getLogger(__name__)

class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to the vector store"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        pass
        
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass
        
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from the vector store"""
        pass
        
    @abstractmethod
    def clear(self) -> bool:
        """Clear all documents from the vector store"""
        pass

class FaissVectorStore(BaseVectorStore):
    """FAISS implementation of vector store with enhanced caching capabilities"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = None
        self.documents = {}  # Map IDs to documents
        self.embedder = None
        self.is_connected = False
        self.query_cache = {}  # Cache for query results
        self.cache_expiry = {}  # Track when cached results should expire
        self.cache_ttl = 3600  # Default cache TTL in seconds (1 hour)
        
    def connect(self) -> bool:
        """Initialize FAISS index and load existing data"""
        try:
            # Initialize embedder
            self.embedder = SentenceTransformer(AppConfig.EMBEDDING_CONFIG["model_name"])
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Load existing index if available
            self._load_index()
            
            self.is_connected = True
            logger.info("Successfully connected to FAISS vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to FAISS vector store: {str(e)}")
            self.is_connected = False
            return False
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return False
            
        try:
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            
            # Add to FAISS
            self.index.add(embeddings)
            
            # Store documents
            for i, doc in enumerate(documents):
                doc_id = str(len(self.documents))
                self.documents[doc_id] = {
                    **doc,
                    'embedding_id': len(self.documents) + i,
                    'timestamp': datetime.now().isoformat()
                }
                
            # Save index
            self._save_index()
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {str(e)}")
            return False
            
    def search(self, query: str, k: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar documents with caching support
        
        Args:
            query: The query text to search for
            k: Number of results to return
            use_cache: Whether to use cached results if available
            
        Returns:
            List of matching documents
        """
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return []
            
        # Check cache first if enabled
        cache_key = f"{query}_{k}"
        if use_cache and cache_key in self.query_cache:
            # Check if cache is still valid
            if datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                logger.info(f"Using cached results for query: {query}")
                return self.query_cache[cache_key]
            else:
                # Remove expired cache entry
                self._remove_from_cache(cache_key)
            
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], normalize_embeddings=True)
            
            # Search FAISS
            distances, indices = self.index.search(query_embedding, k)
            
            # Get documents
            results = []
            for i, idx in enumerate(indices[0]):
                doc_id = str(idx)
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    results.append({
                        **doc,
                        'score': float(distances[0][i])
                    })
            
            # Cache results if enabled
            if use_cache:
                self._add_to_cache(cache_key, results)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            return []
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set the time-to-live for cached results in seconds"""
        self.cache_ttl = ttl_seconds
        logger.info(f"Cache TTL set to {ttl_seconds} seconds")
    
    def clear_cache(self) -> None:
        """Clear all cached search results"""
        self.query_cache = {}
        self.cache_expiry = {}
        logger.info("Search result cache cleared")
    
    def _add_to_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Add search results to cache"""
        self.query_cache[cache_key] = results
        self.cache_expiry[cache_key] = datetime.now().timestamp() + self.cache_ttl
    
    def _remove_from_cache(self, cache_key: str) -> None:
        """Remove an item from cache"""
        if cache_key in self.query_cache:
            del self.query_cache[cache_key]
        if cache_key in self.cache_expiry:
            del self.cache_expiry[cache_key]
    
    def add_processed_result(self, query: str, result: Dict[str, Any], content_field: str = "content") -> str:
        """
        Store a processed result in the vector store for future reuse
        
        Args:
            query: The original query that produced this result
            result: The processed result to store
            content_field: The field name to use for the content
            
        Returns:
            ID of the stored result document
        """
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return ""
            
        try:
            # Prepare the document with the result
            document = {
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
                "original_query": query,
                "type": "processed_result",
                "created_at": datetime.now().isoformat()
            }
            
            # Add document to vector store
            self.add_documents([document])
            
            # Return the ID of the newly added document
            return str(len(self.documents) - 1)
            
        except Exception as e:
            logger.error(f"Error storing processed result: {str(e)}")
            return ""
    
    def find_similar_results(self, query: str, threshold: float = 0.8, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find previously processed results for similar queries
        
        Args:
            query: The query to find similar results for
            threshold: Similarity threshold (0-1)
            k: Number of candidates to consider
            
        Returns:
            List of similar processed results
        """
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return []
            
        try:
            # Search for similar queries
            results = self.search(query, k=k)
            
            # Filter by similarity threshold and type
            similar_results = []
            for result in results:
                if (result.get('score', float('inf')) <= threshold and 
                    result.get('type') == 'processed_result'):
                    # Parse the content if it's JSON
                    content = result.get('content', '{}')
                    try:
                        if isinstance(content, str):
                            parsed_content = json.loads(content)
                        else:
                            parsed_content = content
                        result['parsed_content'] = parsed_content
                    except:
                        result['parsed_content'] = content
                    similar_results.append(result)
                    
            return similar_results
            
        except Exception as e:
            logger.error(f"Error finding similar results: {str(e)}")
            return []
            
    def delete_documents(self, ids: List[str]) -> bool:
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return False
            
        try:
            # Remove from documents store
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    
            # Rebuild index
            self._rebuild_index()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from FAISS: {str(e)}")
            return False
            
    def clear(self) -> bool:
        if not self.is_connected:
            logger.error("Vector store not connected. Call connect() first.")
            return False
            
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self._save_index()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {str(e)}")
            return False
            
    def _save_index(self):
        """Save FAISS index, documents, and cache to disk"""
        if not self.is_connected:
            return
            
        index_path = Path(AppConfig.VECTOR_DB_CONFIG["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(
            self.index,
            str(index_path / "mental_health.index")
        )
        
        # Save documents
        with open(index_path / "documents.json", "w") as f:
            json.dump(self.documents, f)
            
        # Save cache (optional)
        try:
            cache_data = {
                "query_cache": self.query_cache,
                "cache_expiry": self.cache_expiry,
                "cache_ttl": self.cache_ttl
            }
            with open(index_path / "cache.json", "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {str(e)}")
            
    def _load_index(self):
        """Load FAISS index, documents, and cache from disk"""
        try:
            index_path = Path(AppConfig.VECTOR_DB_CONFIG["index_path"])
            
            # Load FAISS index
            if (index_path / "mental_health.index").exists():
                self.index = faiss.read_index(
                    str(index_path / "mental_health.index")
                )
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                
            # Load documents
            if (index_path / "documents.json").exists():
                with open(index_path / "documents.json", "r") as f:
                    self.documents = json.load(f)
            
            # Load cache (optional)
            if (index_path / "cache.json").exists():
                try:
                    with open(index_path / "cache.json", "r") as f:
                        cache_data = json.load(f)
                        self.query_cache = cache_data.get("query_cache", {})
                        self.cache_expiry = cache_data.get("cache_expiry", {})
                        self.cache_ttl = cache_data.get("cache_ttl", 3600)
                        
                        # Filter out expired cache entries
                        current_time = datetime.now().timestamp()
                        expired_keys = [k for k, v in self.cache_expiry.items() 
                                      if current_time > v]
                        for key in expired_keys:
                            self._remove_from_cache(key)
                except Exception as e:
                    logger.warning(f"Could not load cache, starting with empty cache: {str(e)}")
                    self.query_cache = {}
                    self.cache_expiry = {}
                    
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            
    def _rebuild_index(self):
        """Rebuild FAISS index from documents"""
        if not self.is_connected:
            return
            
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.documents:
            embeddings = []
            for doc in self.documents.values():
                text = doc['content']
                embedding = self.embedder.encode([text], normalize_embeddings=True)
                embeddings.append(embedding[0])
            self.index.add(np.array(embeddings))
            self._save_index()

class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation of vector store"""
    
    def __init__(self):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = AppConfig.VECTOR_DB_CONFIG["collection_name"]
        self.embedder = SentenceTransformer(AppConfig.EMBEDDING_CONFIG["model_name"])
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            # Generate embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True)
            
            # Prepare points
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append({
                    'id': str(len(points) + i),
                    'vector': embedding.tolist(),
                    'payload': {
                        **doc,
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to Qdrant: {str(e)}")
            return False
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
            
            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k
            )
            
            # Format results
            return [
                {
                    **hit.payload,
                    'score': float(hit.score)
                }
                for hit in results
            ]
            
        except Exception as e:
            logging.error(f"Error searching Qdrant: {str(e)}")
            return []
            
    def delete_documents(self, ids: List[str]) -> bool:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            return True
            
        except Exception as e:
            logging.error(f"Error deleting documents from Qdrant: {str(e)}")
            return False
            
    def clear(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            return True
            
        except Exception as e:
            logging.error(f"Error clearing Qdrant collection: {str(e)}")
            return False
            
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        if not self.client.get_collections().collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": AppConfig.VECTOR_DB_CONFIG["dimension"],
                    "distance": AppConfig.VECTOR_DB_CONFIG["metric_type"]
                }
            )

class VectorStore:
    """Factory class for vector store implementations"""
    
    @staticmethod
    def create(engine: str = None) -> BaseVectorStore:
        """Create a vector store instance"""
        engine = engine or AppConfig.VECTOR_DB_CONFIG["engine"]
        
        if engine == "faiss":
            return FaissVectorStore(
                dimension=AppConfig.VECTOR_DB_CONFIG["dimension"]
            )
        elif engine == "qdrant":
            return QdrantVectorStore()
        else:
            raise ValueError(f"Unsupported vector store engine: {engine}")
            
    @staticmethod
    def get_supported_engines() -> List[str]:
        """Get list of supported vector store engines"""
        return ["faiss", "qdrant"]
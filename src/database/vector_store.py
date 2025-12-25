import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import threading
import weakref

# Vector store backends - import faiss if available
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:  # Allow running without faiss installed
    faiss = None
    FAISS_AVAILABLE = False
# Optional backends - imported when needed to avoid Keras conflicts:
# from qdrant_client import QdrantClient
# from pymilvus import connections, Collection, utility
# from sentence_transformers import SentenceTransformer

from src.config.settings import AppConfig

# Import storage exceptions for better error handling
try:
    from src.core.exceptions import StorageError, StorageConnectionError
    STORAGE_EXCEPTIONS_AVAILABLE = True
except ImportError:
    STORAGE_EXCEPTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Reused constants
_ERR_NOT_CONNECTED = "Vector store not connected. Call connect() first."
_INDEX_FILENAME = "mental_health.index"
_DOCS_FILENAME = "documents.json"
_CACHE_FILENAME = "cache.json"

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

    @abstractmethod
    def count(self) -> int:
        """Return number of stored documents"""
        pass

class FaissVectorStore(BaseVectorStore):
    """
    FAISS implementation with graceful fallback and improved error handling.

    Features:
    - Thread-safe operations with locks
    - Memory management with limits
    - Better error handling with custom exceptions
    - Connection pooling ready
    - Graceful degradation when dependencies unavailable
    """

    # Class-level memory limit (in MB)
    MAX_MEMORY_MB = 2048
    # Maximum cache size (number of queries)
    MAX_CACHE_SIZE = 1000

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = None  # faiss index if available
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embedder = None  # sentence-transformers if available
        self.is_connected = False
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_expiry: Dict[str, float] = {}
        self.cache_ttl = 3600
        self._embeddings_matrix: Optional[np.ndarray] = None  # fallback storage

        # Thread safety
        self._lock = threading.RLock()
        self._connection_lock = threading.Lock()

        # Memory tracking
        self._memory_usage_mb = 0.0

        # Error tracking for circuit breaker pattern
        self._error_count = 0
        self._last_error_time = None
        self._max_errors = 5
        self._error_reset_time = 300  # 5 minutes
        
    def connect(self) -> bool:
        """
        Initialize FAISS index and load existing data with thread safety.

        Returns:
            bool: True if connection successful

        Raises:
            StorageConnectionError: If connection fails critically
        """
        with self._connection_lock:
            try:
                # Check if already connected
                if self.is_connected:
                    logger.info("Vector store already connected")
                    return True

                # Initialize embedder when available
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                    model_name = AppConfig.EMBEDDING_CONFIG.get("model_name")
                    if not model_name:
                        logger.warning("No embedding model specified, using simple embedder")
                        self.embedder = None
                    else:
                        self.embedder = SentenceTransformer(model_name)
                        logger.info(f"SentenceTransformer embedder initialized: {model_name}")
                except Exception as e:
                    logger.warning(f"SentenceTransformer unavailable ({e}). Falling back to simple embedder.")
                    self.embedder = None

                # Initialize FAISS or fallback backend
                if FAISS_AVAILABLE and faiss is not None:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self._load_index()
                    logger.info("FAISS backend initialized successfully")
                else:
                    self.index = None
                    self._embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)
                    logger.warning("FAISS not available; using in-memory numpy backend")

                self.is_connected = True
                self._error_count = 0  # Reset error count on successful connection
                return True

            except Exception as e:
                logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
                self.is_connected = False

                # Raise custom exception if available
                if STORAGE_EXCEPTIONS_AVAILABLE:
                    raise StorageConnectionError(
                        f"Failed to connect to vector store: {str(e)}",
                        storage_type="faiss",
                        cause=e
                    )
                return False
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents with memory management and error handling.

        Args:
            documents: List of documents to add

        Returns:
            bool: True if documents added successfully

        Raises:
            StorageError: If adding documents fails
        """
        if not self.is_connected:
            logger.error(_ERR_NOT_CONNECTED)
            if STORAGE_EXCEPTIONS_AVAILABLE:
                raise StorageConnectionError(_ERR_NOT_CONNECTED, storage_type="faiss")
            return False

        with self._lock:
            try:
                # Check memory limits before adding
                estimated_memory = len(documents) * self.dimension * 4 / (1024 * 1024)  # MB
                if self._memory_usage_mb + estimated_memory > self.MAX_MEMORY_MB:
                    logger.warning(
                        f"Memory limit approaching: {self._memory_usage_mb + estimated_memory:.2f}MB. "
                        f"Consider clearing cache or rebuilding index."
                    )

                # Generate embeddings
                precomputed = [doc.get('embedding') for doc in documents]
                if all(emb is not None for emb in precomputed):
                    embeddings = np.array(precomputed, dtype=np.float32)
                else:
                    texts = [doc.get('content', '') for doc in documents]
                    if self.embedder is not None:
                        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
                    else:
                        embeddings = self._simple_embed(texts)

                # Add to appropriate backend
                if self.index is not None:
                    # FAISS path
                    self.index.add(embeddings.astype(np.float32))
                else:
                    # Numpy fallback path
                    if self._embeddings_matrix is None:
                        self._embeddings_matrix = embeddings.astype(np.float32)
                    else:
                        self._embeddings_matrix = np.vstack([
                            self._embeddings_matrix,
                            embeddings.astype(np.float32)
                        ])

                # Store documents and track embedding row id
                start_row = len(self.documents)
                for i, doc in enumerate(documents):
                    doc_id = str(start_row + i)
                    self.documents[doc_id] = {
                        **doc,
                        'embedding_row': start_row + i,
                        'timestamp': datetime.now().isoformat()
                    }

                # Update memory usage
                self._memory_usage_mb += estimated_memory

                # Save if FAISS backend
                if self.index is not None:
                    self._save_index()

                logger.info(f"Added {len(documents)} documents. Memory usage: {self._memory_usage_mb:.2f}MB")
                return True

            except Exception as e:
                self._handle_error(e)
                logger.error(f"Error adding documents to FAISS: {str(e)}", exc_info=True)

                if STORAGE_EXCEPTIONS_AVAILABLE:
                    raise StorageError(
                        f"Failed to add documents: {str(e)}",
                        storage_type="faiss",
                        operation="add_documents",
                        cause=e
                    )
                return False
            
    def search(self, query: str = None, k: int = 5, use_cache: bool = True,
               query_vector: Optional[List[float]] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with caching support
        
        Args:
            query: The query text to search for (ignored if query_vector is provided)
            k: Number of results to return (alias: top_k)
            use_cache: Whether to use cached results if available
            query_vector: Optional precomputed embedding vector to search by
            top_k: Optional alias for k
            
        Returns:
            List of matching documents
        """
        if top_k is not None:
            k = top_k
        if not self.is_connected:
            logger.error(_ERR_NOT_CONNECTED)
            return []
            
        # Check cache first if enabled (only for text queries)
        cache_key = None
        if query_vector is None and query is not None:
            cache_key = f"{query}_{k}"
            if use_cache and cache_key in self.query_cache:
                if datetime.now().timestamp() < self.cache_expiry.get(cache_key, 0):
                    logger.info(f"Using cached results for query: {query}")
                    return self.query_cache[cache_key]
                else:
                    self._remove_from_cache(cache_key)
            
        try:
            # Generate or accept query embedding
            if query_vector is not None:
                query_embedding = np.array([query_vector], dtype=np.float32)
            else:
                if query is None:
                    logger.error("Either query text or query_vector must be provided")
                    return []
                if self.embedder is not None:
                    query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
                else:
                    query_embedding = self._simple_embed([query]).astype(np.float32)

            results: List[Dict[str, Any]] = []
            if self.index is not None:
                # FAISS search
                distances, indices = self.index.search(query_embedding, k)
                for i, idx in enumerate(indices[0]):
                    row_id = int(idx)
                    doc_id = str(row_id)
                    if doc_id in self.documents:
                        doc = self.documents[doc_id]
                        results.append({
                            **doc,
                            'score': float(distances[0][i])
                        })
            else:
                # Numpy fallback: cosine similarity
                if self._embeddings_matrix is None or self._embeddings_matrix.shape[0] == 0:
                    return []
                q = query_embedding[0]
                # Normalize rows
                A = self._embeddings_matrix
                denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(q) + 1e-8)
                sims = (A @ q) / denom
                # Highest similarity first
                top_idx = np.argsort(-sims)[:k]
                for idx in top_idx:
                    doc_id = str(int(idx))
                    if doc_id in self.documents:
                        doc = self.documents[doc_id]
                        results.append({
                            **doc,
                            'score': float(1.0 - sims[int(idx)])  # lower is better to match L2 semantics
                        })
            
            # Cache results if enabled and text query was used
            if use_cache and cache_key is not None:
                self._add_to_cache(cache_key, results)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            return []

    # Compatibility helpers for existing call sites
    def add_item(self, vector: List[float], metadata: Dict[str, Any]) -> bool:
        """Add a single item with a precomputed embedding vector and metadata"""
        doc = {"content": metadata.get("title") or metadata.get("document_id") or "", "embedding": vector, **{"metadata": metadata}}
        return self.add_documents([doc])

    def count(self) -> int:
        return len(self.documents)
    
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
    
    def add_processed_result(self, query: str, result: Dict[str, Any], _content_field: str = "content") -> str:
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
            logger.error(_ERR_NOT_CONNECTED)
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
            logger.error(_ERR_NOT_CONNECTED)
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
                    except Exception:
                        result['parsed_content'] = content
                    similar_results.append(result)
                    
            return similar_results
            
        except Exception as e:
            logger.error(f"Error finding similar results: {str(e)}")
            return []
            
    def delete_documents(self, ids: List[str]) -> bool:
        if not self.is_connected:
            logger.error(_ERR_NOT_CONNECTED)
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
            logger.error(_ERR_NOT_CONNECTED)
            return False
            
        try:
            if faiss is not None:
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self._embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)
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
        
        # Save FAISS index if backend available
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(index_path / _INDEX_FILENAME))
        
        # Save documents
        with open(index_path / _DOCS_FILENAME, "w") as f:
            json.dump(self.documents, f)
            
        # Save cache (optional)
        try:
            cache_data = {
                "query_cache": self.query_cache,
                "cache_expiry": self.cache_expiry,
                "cache_ttl": self.cache_ttl
            }
            with open(index_path / _CACHE_FILENAME, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {str(e)}")
            
    def _load_index(self):
        """Load FAISS index, documents, and cache from disk"""
        try:
            index_path = Path(AppConfig.VECTOR_DB_CONFIG["index_path"])
            
            # Load FAISS index
            if faiss is not None:
                if (index_path / _INDEX_FILENAME).exists():
                    self.index = faiss.read_index(str(index_path / _INDEX_FILENAME))
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
                
            # Load documents
            if (index_path / _DOCS_FILENAME).exists():
                with open(index_path / _DOCS_FILENAME, "r") as f:
                    self.documents = json.load(f)
            
            # Load cache (optional)
            if (index_path / _CACHE_FILENAME).exists():
                try:
                    with open(index_path / _CACHE_FILENAME, "r") as f:
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
            
        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.dimension)
            if self.documents:
                texts = [doc['content'] for doc in self.documents.values()]
                if self.embedder is not None:
                    embeddings = self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)
                else:
                    embeddings = self._simple_embed(texts).astype(np.float32)
                self.index.add(embeddings)
                self._save_index()
        else:
            # Fallback path already uses self._embeddings_matrix; rebuild not required
            pass

    def _simple_embed(self, texts: List[str]) -> np.ndarray:
        """
        Lightweight embedding fallback using character n-grams and multi-hash (BUG-013 fix).

        Produces deterministic fixed-size dense vectors without heavy dependencies.
        Uses character n-grams (3-grams) with multiple hash functions for:
        - Better semantic similarity (similar words share n-grams)
        - Denser vectors (more features activated per text)
        - More robust to typos and variations

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of shape (len(texts), self.dimension) with L2-normalized vectors
        """
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)

        # Use multiple hash functions for better distribution
        hash_seeds = [31, 37, 41, 43, 47]

        for i, text in enumerate(texts):
            if not text:
                # Empty text gets a random but consistent vector
                vectors[i] = self._deterministic_random_vector(i)
                continue

            text_lower = str(text).lower()

            # Generate character n-grams (trigrams work well for semantic similarity)
            for n in [2, 3, 4]:  # Use bi-grams, tri-grams, and quad-grams
                for j in range(max(1, len(text_lower) - n + 1)):
                    ngram = text_lower[j:j + n]

                    # Apply multiple hash functions for denser representation
                    for seed in hash_seeds:
                        # Combine ngram hash with seed for different projections
                        h = hash(ngram + str(seed))
                        idx = abs(h) % self.dimension

                        # Weight by n-gram length (longer n-grams are more specific)
                        weight = 1.0 / (n ** 0.5)
                        vectors[i, idx] += weight

            # Also add word-level features (important for semantics)
            words = text_lower.split()
            for word in words:
                if len(word) >= 2:  # Skip single chars
                    for seed in hash_seeds[:2]:  # Use fewer hash functions for words
                        h = hash(word + str(seed))
                        idx = abs(h) % self.dimension
                        vectors[i, idx] += 2.0  # Words get higher weight

            # L2 normalize for cosine similarity compatibility
            norm = np.linalg.norm(vectors[i])
            if norm > 1e-8:
                vectors[i] = vectors[i] / norm
            else:
                # Fallback for very short or empty text
                vectors[i] = self._deterministic_random_vector(i)

        return vectors

    def _deterministic_random_vector(self, seed: int) -> np.ndarray:
        """Generate a deterministic pseudo-random unit vector for empty/edge cases."""
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle errors with circuit breaker pattern.

        Args:
            error: The exception that occurred
        """
        current_time = datetime.now().timestamp()

        # Reset error count if enough time has passed
        if (self._last_error_time and
            current_time - self._last_error_time > self._error_reset_time):
            self._error_count = 0

        self._error_count += 1
        self._last_error_time = current_time

        # Log warning if approaching error threshold
        if self._error_count >= self._max_errors - 1:
            logger.warning(
                f"FAISS error count high ({self._error_count}/{self._max_errors}). "
                "Consider reconnecting or checking system health."
            )

    def _manage_cache_size(self) -> None:
        """
        Manage cache size to prevent memory issues.

        Removes least recently used entries if cache exceeds maximum size.
        """
        if len(self.query_cache) > self.MAX_CACHE_SIZE:
            # Sort by expiry time and remove oldest entries
            sorted_keys = sorted(
                self.cache_expiry.items(),
                key=lambda x: x[1]
            )

            # Remove oldest 10% of cache
            remove_count = len(self.query_cache) // 10
            for key, _ in sorted_keys[:remove_count]:
                self._remove_from_cache(key)

            logger.info(f"Cache cleanup: removed {remove_count} entries")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        return {
            'memory_usage_mb': round(self._memory_usage_mb, 2),
            'max_memory_mb': self.MAX_MEMORY_MB,
            'memory_usage_percent': round((self._memory_usage_mb / self.MAX_MEMORY_MB) * 100, 2),
            'document_count': len(self.documents),
            'cache_size': len(self.query_cache),
            'max_cache_size': self.MAX_CACHE_SIZE,
            'error_count': self._error_count,
            'is_connected': self.is_connected
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.

        Returns:
            Dictionary with health status
        """
        status = "healthy"
        issues = []

        if not self.is_connected:
            status = "disconnected"
            issues.append("Vector store not connected")

        if self._error_count >= self._max_errors:
            status = "unhealthy"
            issues.append(f"Error count threshold reached: {self._error_count}")

        if self._memory_usage_mb > self.MAX_MEMORY_MB * 0.9:
            status = "degraded" if status == "healthy" else status
            issues.append(f"Memory usage high: {self._memory_usage_mb:.2f}MB")

        if len(self.query_cache) > self.MAX_CACHE_SIZE * 0.9:
            if status == "healthy":
                status = "degraded"
            issues.append(f"Cache size high: {len(self.query_cache)}")

        return {
            'status': status,
            'issues': issues,
            'stats': self.get_memory_stats(),
            'backend': 'faiss' if self.index is not None else 'numpy',
            'embedder': 'sentence-transformers' if self.embedder is not None else 'simple'
        }

class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation of vector store"""
    
    def __init__(self):
        # Lazy imports to avoid dependency conflicts
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient("localhost", port=6333)
        except Exception as e:
            logger.warning(f"Failed to load QdrantClient: {e}")
            self.client = None
            
        self.collection_name = AppConfig.VECTOR_DB_CONFIG["collection_name"]
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(AppConfig.EMBEDDING_CONFIG["model_name"])
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            self.embedder = None
        
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
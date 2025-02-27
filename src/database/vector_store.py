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

class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
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
    """FAISS implementation of vector store"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = {}  # Map IDs to documents
        self.embedder = SentenceTransformer(AppConfig.EMBEDDING_CONFIG["model_name"])
        
        # Load existing index if available
        self._load_index()
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
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
            logging.error(f"Error adding documents to FAISS: {str(e)}")
            return False
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
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
                    
            return results
            
        except Exception as e:
            logging.error(f"Error searching FAISS: {str(e)}")
            return []
            
    def delete_documents(self, ids: List[str]) -> bool:
        try:
            # Remove from documents store
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    
            # Rebuild index
            self._rebuild_index()
            return True
            
        except Exception as e:
            logging.error(f"Error deleting documents from FAISS: {str(e)}")
            return False
            
    def clear(self) -> bool:
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self._save_index()
            return True
            
        except Exception as e:
            logging.error(f"Error clearing FAISS index: {str(e)}")
            return False
            
    def _save_index(self):
        """Save FAISS index and documents to disk"""
        index_path = Path(AppConfig.VECTOR_DB_CONFIG["index_path"])
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(
            self.index,
            str(index_path / "mental_health.index")
        )
        
        # Save documents
        import json
        with open(index_path / "documents.json", "w") as f:
            json.dump(self.documents, f)
            
    def _load_index(self):
        """Load FAISS index and documents from disk"""
        try:
            index_path = Path(AppConfig.VECTOR_DB_CONFIG["index_path"])
            
            # Load FAISS index
            if (index_path / "mental_health.index").exists():
                self.index = faiss.read_index(
                    str(index_path / "mental_health.index")
                )
                
            # Load documents
            if (index_path / "documents.json").exists():
                with open(index_path / "documents.json", "r") as f:
                    self.documents = json.load(f)
                    
        except Exception as e:
            logging.error(f"Error loading FAISS index: {str(e)}")
            
    def _rebuild_index(self):
        """Rebuild FAISS index from documents"""
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
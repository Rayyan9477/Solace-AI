import sqlite3  # Use standard sqlite3
from typing import List, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import numpy as np

class VectorStore:
    """
    A base interface for vector store operations.
    """

    def connect(self):
        raise NotImplementedError

    def upsert(self, embedding: List[float], data: Any):
        raise NotImplementedError

    def search(self, embedding: List[float], top_k: int = 5) -> List[Any]:
        raise NotImplementedError

    def as_retriever(self):
        """
        Optional: Return an object that can be used as a retriever in a pipeline.
        """
        raise NotImplementedError

class HybridVectorStore(VectorStore):
    """
    Attempts to use local Chroma ("duckdb+parquet"). If that fails, falls back to FAISS.
    """

    def __init__(self, collection_name: str = "mental_health_collection", dimension: int = 384):
        self.collection_name = collection_name
        self.dimension = dimension
        self.chroma_client: Optional[chromadb.Client] = None
        self.chroma_collection = None
        self.faiss_index = None
        # Shared embeddings for both Chroma and FAISS
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def connect(self):
        """
        Instantiates Chroma; if that fails, sets up a local FAISS vector store as fallback.
        """
        try:
            self.chroma_client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory="./chroma_db",
                    anonymized_telemetry=False
                )
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            print(f"Chroma initialization failed: {e}")
            print("Falling back to FAISS...")
            self._init_faiss()

    def _init_faiss(self):
        """
        Initializes an empty FAISS index or loads from local if you prefer.
        """
        # Simple approach: just create a new FAISS store with an initialization text
        # If you want to load from an existing local index, call FAISS.load_local(...) instead
        self.faiss_index = FAISS.from_texts(["initialization_text"], self.embeddings)

    def upsert(self, embedding: List[float], data: Any):
        """
        Adds a document to either Chroma or FAISS.
        """
        if self.chroma_collection:
            # Use Chroma
            doc_id = f"doc_{hash(str(data))}"
            try:
                self.chroma_collection.add(
                    embeddings=[embedding],
                    documents=[str(data)],
                    ids=[doc_id]
                )
                self.chroma_client.persist()
            except Exception as e:
                print(f"Chroma upsert failed: {e}. Falling back to FAISS...")
                self._init_faiss()
                self.faiss_index.add_texts([str(data)])
        elif self.faiss_index:
            # Use FAISS
            self.faiss_index.add_texts([str(data)])
        else:
            raise RuntimeError("No vector store ready. Try calling connect() first.")

    def search(self, embedding: List[float], top_k: int = 5) -> List[Any]:
        """
        Searches either Chroma or FAISS for nearest matches.
        """
        if self.chroma_collection:
            try:
                results = self.chroma_collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k
                )
                return results.get("documents", [[]])[0]
            except Exception as e:
                print(f"Chroma query failed: {e}. Falling back to FAISS...")
                self._init_faiss()
                return self._faiss_search(embedding, top_k)
        elif self.faiss_index:
            return self._faiss_search(embedding, top_k)
        else:
            raise RuntimeError("No vector store ready. Try calling connect() first.")

    def _faiss_search(self, embedding: List[float], top_k: int) -> List[str]:
        """
        Internal helper to convert embedding to NumPy array and query FAISS.
        """
        import torch

        # Convert to correct shape if needed
        vec = np.array(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = np.expand_dims(vec, axis=0)
        # The FAISS library in langchain_community can handle the search directly
        retrieved_docs = self.faiss_index.similarity_search_by_vector(vec[0], k=top_k)
        return [doc.page_content for doc in retrieved_docs]

    def as_retriever(self):
        """
        Returns a retriever-like object for usage in pipelines. 
        If using Chroma, return Chroma's retriever, else FAISS.
        """
        if self.chroma_collection:
            return Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            ).as_retriever()
        elif self.faiss_index:
            return self.faiss_index
        else:
            raise RuntimeError("No vector store is configured yet. Call connect().")
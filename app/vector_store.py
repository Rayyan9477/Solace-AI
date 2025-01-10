from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
import os
from pathlib import Path
import numpy as np

class VectorStore:
    def __init__(self, dimension=384):  # default dimension for all-MiniLM-L6-v2
        self.dimension = dimension
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.vector_store_path = Path("vector_store")
        self._initialize_store()

    def _initialize_store(self):
        """Initialize an empty vector store if it doesn't exist"""
        if self.vector_store_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings
                )
            except Exception as e:
                print(f"Error loading existing vector store: {e}")
                self._create_empty_store()
        else:
            self._create_empty_store()

    def _create_empty_store(self):
        """Create an empty FAISS vector store"""
        empty_vector = np.zeros((1, self.dimension))
        empty_texts = ["initialization_text"]
        
        self.vector_store = FAISS.from_texts(
            texts=empty_texts, 
            embedding=self.embeddings
        )
        self.vector_store.save_local(str(self.vector_store_path))

    def add_texts(self, texts):
        """Add new texts to the vector store"""
        if not texts:
            return
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
        else:
            self.vector_store.add_texts(texts)
        
        self.vector_store.save_local(str(self.vector_store_path))

    def similarity_search(self, query, k=3):
        """Perform similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)
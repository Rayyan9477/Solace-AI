import os
from typing import List, Any, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import AppConfig

class FAISSVectorStore:
    def __init__(self, path: str, collection: str, allow_dangerous_deserialization: bool = False):
        self.path = path
        self.collection = collection
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.index: Optional[FAISS] = None

    def __hash__(self):
        return hash((self.path, self.collection))

    def connect(self):
        index_path = os.path.join(self.path, self.collection)
        if os.path.exists(index_path):
            try:
                self.index = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=self.allow_dangerous_deserialization)
            except ValueError as e:
                print(f"Warning: Could not load existing index due to {str(e)}. Creating a new index.")
                self.index = FAISS.from_texts(["Initialization"], self.embeddings)
                self.save()
        else:
            self.index = FAISS.from_texts(["Initialization"], self.embeddings)
            self.save()

    def save(self):
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call connect() first.")
        index_path = os.path.join(self.path, self.collection)
        self.index.save_local(index_path, safe_serialize=True)

    def upsert(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call connect() first.")
        self.index.add_texts(texts, metadatas)
        self.save()

    def search(self, query: str, top_k: int = 5) -> List[Any]:
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call connect() first.")
        return self.index.similarity_search(query, k=top_k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call connect() first.")
        return self.index.as_retriever(search_kwargs=search_kwargs)


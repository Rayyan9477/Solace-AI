import os
from typing import List, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config.settings import AppConfig

class FAISSVectorStore:
    def __init__(self, path: str, collection: str, allow_dangerous_deserialization: bool = False):
        self.path = os.path.join(path, collection)
        self.allow_dangerous = allow_dangerous_deserialization
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
        self.index: Optional[FAISS] = None

    def connect(self):
        os.makedirs(self.path, exist_ok=True)
        try:
            if os.path.exists(os.path.join(self.path, "index.faiss")):
                self.index = FAISS.load_local(
                    self.path,
                    self.embeddings,
                    allow_dangerous_deserialization=self.allow_dangerous
                )
            else:
                self.index = FAISS.from_texts(
                    ["Initial document"],
                    self.embeddings,
                    metadatas=[{"source": "init"}]
                )
                self.save()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    def save(self):
        if self.index is None:
            raise ValueError("Index not initialized")
        self.index.save_local(self.path)

    def upsert(self, documents: List[Document]):
        if self.index is None:
            raise ValueError("Index not initialized")
        self.index.add_documents(documents)
        self.save()

    def search(self, query: str, k: int = AppConfig.RAG_TOP_K) -> List[Document]:
        if self.index is None:
            raise ValueError("Index not initialized")
        return self.index.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        if self.index is None:
            raise ValueError("Index not initialized")
        return self.index.as_retriever(
            search_kwargs=search_kwargs or {"k": AppConfig.RAG_TOP_K}
        )
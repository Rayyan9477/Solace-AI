from typing import List
from database.vector_store import VectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class SearchAgent:
    """
    SearchAgent integrates with a vector store for retrieval in RAG solutions.
    """
    def __init__(
        self,
        vector_store: VectorStore
    ):
        self.vector_store = vector_store

    def retrieve_context(self, query: str) -> str:
        """
        Retrieves relevant context from the vector store based on the query.
        """
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.encode(query).tolist()
        results = self.vector_store.search(query_embedding, top_k=3)
        context = " ".join(results)
        return context
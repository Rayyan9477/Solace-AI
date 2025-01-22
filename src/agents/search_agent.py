from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import AppConfig

class SearchAgent:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        self.embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)

    def retrieve_context(self, query: str) -> str:
        results = self.vector_store.similarity_search(query, k=AppConfig.RAG_TOP_K)
        return " ".join([doc.page_content for doc in results])


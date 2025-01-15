from typing import List, Any
import chromadb
from chromadb.config import Settings
from chromadb import Client

class VectorStore:
    """
    A base/abstract interface for vector store operations.
    """

    def connect(self):
        """
        Establish a connection to the vector database 
        (if any connecting steps are required).
        """
        raise NotImplementedError

    def upsert(self, embedding: List[float], data: Any):
        """
        Upsert a record (embedding + data) into the database.
        """
        raise NotImplementedError

    def search(self, embedding: List[float], top_k: int = 5) -> List[Any]:
        """
        Perform a similarity search and return the top K results.
        """
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store that stores embeddings in a local instance.
    No external server required.
    """

    def __init__(self, collection_name: str = "mental_health_collection"):
        self.collection_name = collection_name
        self.client: Client = None
        self.collection = None

    def connect(self):
        """
        Instantiates a local ChromaDB client and gets/creates a collection.
        Configured to persist data to disk.
        """
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"  # Directory to store embeddings
        ))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def upsert(self, embedding: List[float], data: Any):
        if not self.collection:
            raise RuntimeError("Chroma collection not connected.")
        doc_id = f"doc_{hash(str(data))}"
        self.collection.add(
            embeddings=[embedding],
            documents=[str(data)],
            ids=[doc_id]
        )
        self.client.persist()  # Ensure data is written to disk

    def search(self, embedding: List[float], top_k: int = 5) -> List[Any]:
        if not self.collection:
            raise RuntimeError("Chroma collection not connected.")
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results.get("documents", [[]])[0]
import os

class AppConfig:
    """
    AppConfig is a container for global settings.
    Customize and extend as needed 
    (e.g., reading environment variables or config files).
    """

    # LLM model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
    USE_CPU: bool = os.getenv("USE_CPU", "True").lower() == "true"

    # Vector store or database settings
    VECTOR_DB_HOST: str = os.getenv("VECTOR_DB_HOST", "localhost")
    VECTOR_DB_PORT: int = int(os.getenv("VECTOR_DB_PORT", 6333))
    VECTOR_DB_COLLECTION: str = os.getenv("VECTOR_DB_COLLECTION", "default_collection")

    # Additional RAG or retrieval settings 
    FETCH_LIMIT: int = 10
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.6

    # Logging or debugging
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
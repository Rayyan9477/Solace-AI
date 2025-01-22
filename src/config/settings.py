import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    # Application settings
    APP_NAME = "Mental Health Chatbot"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # LLM model settings
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
    USE_CPU = os.getenv("USE_CPU", "True").lower() == "true"

    # Vector store settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
    VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "mental_health_collection")

    # Embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # RAG settings
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))

    # Crawler settings
    CRAWLER_MAX_RESULTS = int(os.getenv("CRAWLER_MAX_RESULTS", 5))
    CRAWLER_MAX_DEPTH = int(os.getenv("CRAWLER_MAX_DEPTH", 2))

    # API keys and external services
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    # Deployment settings
    PORT = int(os.getenv("PORT", 8501))
    HOST = os.getenv("HOST", "0.0.0.0")

    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

    # Performance settings
    CACHE_TYPE = os.getenv("CACHE_TYPE", "simple")
    CACHE_REDIS_URL = os.getenv("CACHE_REDIS_URL", "redis://localhost:6379/0")

    # Monitoring and logging
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "False").lower() == "true"

    ALLOW_DANGEROUS_DESERIALIZATION = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "False").lower() == "true"

    @classmethod
    def get_vector_store_config(cls):
        return {
            "path": cls.VECTOR_DB_PATH,
            "collection": cls.VECTOR_DB_COLLECTION,
            "allow_dangerous_deserialization": cls.ALLOW_DANGEROUS_DESERIALIZATION,
        }

    @classmethod
    def get_rag_config(cls):
        return {
            "k": cls.RAG_TOP_K,
        }

    @classmethod
    def get_crawler_config(cls):
        return {
            "max_results": cls.CRAWLER_MAX_RESULTS,
            "max_depth": cls.CRAWLER_MAX_DEPTH,
        }


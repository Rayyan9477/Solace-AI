from typing import Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppConfig:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # API Keys and Authentication
    # ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Optional backup
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    
    # LLM Configuration
    LLM_CONFIG = {
        "model": "perplexity/r1-1776",
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Vector Database Configuration
    VECTOR_DB_CONFIG = {
        "engine": "faiss",  # Options: faiss, milvus, qdrant
        "dimension": 1536,  # Embedding dimension
        "index_type": "L2",
        "metric_type": "cosine",
        "index_path": str(DATA_DIR / "vector_indexes"),
        "collection_name": "mental_health_kb"
    }
    
    # Embedding Configuration
    EMBEDDING_CONFIG = {
        "model_name": "BAAI/bge-large-en-v1.5",
        "device": "cpu",
        "normalize_embeddings": True
    }
    
    # Memory Configuration
    MEMORY_CONFIG = {
        "type": "conversation",  # Options: conversation, summary, window
        "max_history": 10,
        "summary_interval": 5
    }
    
    # Knowledge Base Configuration
    KNOWLEDGE_CONFIG = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "max_chunks_per_doc": 100,
        "similarity_threshold": 0.75
    }
    
    # Agent Configuration
    AGENT_CONFIG = {
        "max_iterations": 5,
        "max_execution_time": 30,  # seconds
        "stream_output": True,
        "verbose": True
    }
    
    # Crawler Configuration
    CRAWLER_CONFIG = {
        "max_depth": 3,
        "max_results": 10,
        "request_delay": 1.0,
        "timeout": 10,
        "max_retries": 3,
        "trusted_domains": [
            "nimh.nih.gov",
            "who.int",
            "mayoclinic.org",
            "psychiatry.org",
            "healthline.com",
            "psychologytoday.com"
        ]
    }
    
    # Safety Configuration
    SAFETY_CONFIG = {
        "content_filtering": True,
        "max_toxicity": 0.7,
        "blocked_categories": [
            "self-harm",
            "violence",
            "hate",
            "sexual"
        ],
        "require_content_warnings": True
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO"
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(DATA_DIR / "app.log"),
                "formatter": "standard",
                "level": "DEBUG"
            }
        },
        "root": {
            "handlers": ["console", "file"],
            "level": "INFO"
        }
    }
    
    @classmethod
    def get_vector_store_path(cls) -> str:
        """Get the vector store path, creating if it doesn't exist"""
        path = cls.DATA_DIR / "vector_store"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get the path for a specific model"""
        path = cls.MODELS_DIR / model_name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate the configuration and return any issues"""
        issues = {}
        
        # Check required API keys
        if not cls.ANTHROPIC_API_KEY:
            issues["api_keys"] = "Missing ANTHROPIC_API_KEY"
        
        if not cls.PERPLEXITY_API_KEY:
            issues["api_keys"] = "Missing PERPLEXITY_API_KEY"
            
        # Check directories
        for dir_name, dir_path in [
            ("data", cls.DATA_DIR),
            ("models", cls.MODELS_DIR)
        ]:
            if not dir_path.exists():
                issues[f"{dir_name}_dir"] = f"Directory {dir_path} does not exist"
                
        # Check vector store configuration
        if cls.VECTOR_DB_CONFIG["engine"] not in ["faiss", "milvus", "qdrant"]:
            issues["vector_db"] = "Invalid vector database engine"
            
        return issues
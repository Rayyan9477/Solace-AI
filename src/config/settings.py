import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class AppConfig:
    # Core Application Settings
    APP_NAME = "Mental Health Companion"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecret-1234")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
    USE_CPU = os.getenv("USE_CPU", "True").lower() == "true"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", 512))
    
    # RAG Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
    VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "mental_health")
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))
    RAG_CHUNK_SIZE = 1024
    RAG_CHUNK_OVERLAP = 256
    
    # Safety & Compliance
    CRISIS_RESOURCES = """**Immediate Support Resources:**
- National Suicide Prevention Lifeline: 1-800-273-TALK (8255)
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911"""
    
    # Assessment Questions
    ASSESSMENT_QUESTIONS = [
        "Have you felt consistently sad or empty most days?",
        "Have you lost interest in activities you usually enjoy?",
        "Are you experiencing significant changes in appetite or sleep?",
        "Do you have difficulty concentrating or making decisions?",
        "Have you had thoughts of self-harm or suicide?"
    ]
    
    # Monitoring
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "False").lower() == "true"
    
    # Updated default to True for trusted data sources
    ALLOW_DANGEROUS_DESERIALIZATION = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "True").lower() == "true"
    
    @classmethod
    def get_vector_store_config(cls):
        return {
            "path": cls.VECTOR_DB_PATH,
            "collection": cls.VECTOR_DB_COLLECTION,
            "allow_dangerous_deserialization": cls.ALLOW_DANGEROUS_DESERIALIZATION
        }

    @classmethod
    def get_rag_config(cls):
        return {"k": cls.RAG_TOP_K}

    @classmethod
    def get_crawler_config(cls):
        return {
            "max_results": 3,
            "max_depth": 1,
            "timeout": 10
        }
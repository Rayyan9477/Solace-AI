import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "fine_tuned_model"
    VECTOR_STORE_DIR = BASE_DIR / "vector_store"

    # Model configurations
    MODEL_NAME = "RayyanAhmed9477/Health-Chatbot"
    MAX_LENGTH = 100
    TEMPERATURE = 0.7
    
    # Vector store configurations
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384  # dimension for all-MiniLM-L6-v2
    TOP_K_RESULTS = 3

    # Streamlit configurations
    STREAMLIT_TITLE = "Healthcare Assistant"
    STREAMLIT_ICON = "🏥"

    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist"""
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        Config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_model_path():
        """Get the appropriate model path"""
        if Config.MODEL_DIR.exists():
            return str(Config.MODEL_DIR)
        return Config.MODEL_NAME
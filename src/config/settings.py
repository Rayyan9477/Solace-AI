from pathlib import Path
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables from project root .env file
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

class AppConfig:
    """Application configuration settings"""

    # Application settings
    APP_NAME = "Mental Health Support Bot"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    USER_ID = os.getenv("USER_ID", "default_user")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    VECTOR_STORE_PATH = DATA_DIR / "vector_store"

    # Create directories if they don't exist
    for dir_path in [DATA_DIR, MODEL_DIR, VECTOR_STORE_PATH]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Monitoring settings
    SENTRY_DSN = os.getenv("SENTRY_DSN", "")
    PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Model settings
    # LLM providers & API keys
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    USE_CPU = True  # Always true for cloud-based models
    MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "2000"))

    # LLM Configuration (provider can be switched via env LLM_PROVIDER)
    LLM_CONFIG = {
        "provider": LLM_PROVIDER,
        "model": MODEL_NAME,
        # api_key will be selected in module based on provider
        "api_key": GEMINI_API_KEY if LLM_PROVIDER.lower() == "gemini" else OPENAI_API_KEY,
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "top_p": float(os.getenv("TOP_P", "0.9")),
        "top_k": int(os.getenv("TOP_K", "50")),
        "max_tokens": MAX_RESPONSE_TOKENS
    }

    # Embedding Configuration
    EMBEDDING_CONFIG = {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "normalize_embeddings": True
    }

    # Vector Database Configuration
    VECTOR_DB_CONFIG = {
        "engine": "faiss",
        "dimension": 768,  # Matches the embedding model dimension
        "index_type": "L2",
        "metric_type": "cosine",
        "index_path": str(VECTOR_STORE_PATH),
        "collection_name": "mental_health_kb",
        "central_data_enabled": True,  # Enable central vector DB
        "retention_days": 180,  # Store data for 180 days
        "embedder_model": "all-MiniLM-L6-v2",  # Sentence transformer model for embeddings
        "namespaces": [
            "user_profile",
            "conversation",
            "knowledge",
            "therapy_resource",
            "diagnostic_data",
            "personality_assessment",
            "emotion_record"
        ]
    }

    # Safety settings
    SAFETY_CONFIG = {
        "max_toxicity": float(os.getenv("MAX_TOXICITY", "0.7")),
        "blocked_categories": [
            "harmful",
            "unsafe",
            "toxic",
            "explicit",
            "biased",
            "discriminatory"
        ],
        "content_filters": {
            "profanity": True,
            "personal_info": True,
            "malicious_code": True,
            "sensitive_topics": True
        },
        "require_human_review": os.getenv("REQUIRE_HUMAN_REVIEW", "True").lower() == "true",
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "fallback_responses": {
            "error": "I apologize, but I'm having trouble processing your request safely.",
            "blocked": "I apologize, but I cannot provide that type of content or assistance.",
            "review": "This request requires human review for safety purposes."
        }
    }

    # Model paths and caching
    MODEL_CACHE_DIR = MODEL_DIR / "cache"
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Assessment questions
    ASSESSMENT_QUESTIONS = [
        "Have you been feeling down or depressed?",
        "Do you have trouble sleeping or sleeping too much?",
        "Have you lost interest in activities you used to enjoy?",
        "Do you feel anxious or worried most of the time?",
        "Have you had thoughts of harming yourself?"
    ]

    # Personality assessment settings
    PERSONALITY_CONFIG = {
        "big_five": {
            "enabled": True,
            "num_questions": 20,  # Shortened version for better user experience
            "min_questions": 10,  # Minimum number of questions required for valid results
            "traits": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        },
        "mbti": {
            "enabled": True,
            "num_questions": 20,  # Standard MBTI short form
            "min_questions": 8,   # Minimum number of questions required for valid results
            "dimensions": ["E/I", "S/N", "T/F", "J/P"]
        }
    }

    # Standardized screening questionnaires
    PHQ9_QUESTIONS = [
        "Little interest or pleasure in doing things?",
        "Feeling down, depressed, or hopeless?",
        "Trouble falling or staying asleep, or sleeping too much?",
        "Feeling tired or having little energy?",
        "Poor appetite or overeating?",
        "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
        "Trouble concentrating on things, such as reading the newspaper or watching television?",
        "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?",
        "Thoughts that you would be better off dead or of hurting yourself in some way?"
    ]
    GAD7_QUESTIONS = [
        "Feeling nervous, anxious, or on edge?",
        "Not being able to stop or control worrying?",
        "Worrying too much about different things?",
        "Trouble relaxing?",
        "Being so restless that it's hard to sit still?",
        "Becoming easily annoyed or irritable?",
        "Feeling afraid as if something awful might happen?"
    ]

    # Crisis resources
    CRISIS_RESOURCES = """
    **Emergency Resources:**
    - National Crisis Hotline: 988
    - Emergency Services: 911
    - Crisis Text Line: Text HOME to 741741

    **Additional Support:**
    - National Alliance on Mental Health: 1-800-950-NAMI
    - Substance Abuse and Mental Health Services: 1-800-662-HELP
    """

    @classmethod
    def get_vector_store_config(cls) -> Dict[str, Any]:
        """Get vector store configuration"""
        return {
            "dimension": cls.VECTOR_DB_CONFIG["dimension"]
        }

    @classmethod
    def get_crawler_config(cls) -> Dict[str, Any]:
        """Get crawler configuration"""
        return {
            "max_depth": 2,
            "max_pages": 10,
            "allowed_domains": [
                "nimh.nih.gov",
                "who.int",
                "mayoclinic.org",
                "psychiatry.org"
            ],
            "user_agent": f"MentalHealthBot/{cls.APP_VERSION}"
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_dirs = [cls.DATA_DIR, cls.MODEL_DIR, cls.VECTOR_STORE_PATH]
        for dir_path in required_dirs:
            if not dir_path.exists():
                return False

        required_settings = [
            cls.APP_NAME,
            cls.MODEL_NAME,
            cls.EMBEDDING_CONFIG["model_name"]
        ]
        return all(required_settings)

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get path for a specific model"""
        return cls.MODEL_DIR / model_name

    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get path for a data file"""
        return cls.DATA_DIR / filename

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get complete model configuration"""
        return {
            **cls.LLM_CONFIG,
            "model_cache_dir": str(cls.MODEL_CACHE_DIR),
            "trust_remote_code": True,
            "device_map": "auto" if not cls.USE_CPU else None,
            "low_cpu_mem_usage": True
        }

    # Voice Configuration - using free models
    VOICE_CONFIG = {
        "stt_model": os.getenv("STT_MODEL", "openai/whisper-large-v3-turbo"),
        "tts_model": os.getenv("TTS_MODEL", "nari-labs/Dia-1.6B"),
        "cache_dir": str(Path(__file__).resolve().parent.parent / 'models' / 'cache'),
        "use_gpu": os.getenv("USE_GPU", "True").lower() == "true",
        "voice_styles": {
            "default": {},
            "male": {"voice_preset": "male"},
            "female": {"voice_preset": "female"},
            "warm": {
                "voice_preset": "female",
                "temperature": 0.7,
                "speaking_rate": 0.9
            }
        }
    }
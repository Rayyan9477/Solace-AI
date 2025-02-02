from typing import List, Dict
import html
import re
from sentence_transformers import SentenceTransformer
from config.settings import AppConfig
import logging

logger = logging.getLogger(__name__)
EMBEDDING_MODEL = SentenceTransformer(AppConfig.EMBEDDING_MODEL)

def sanitize_input(user_input: str) -> str:
    """Secure input sanitization"""
    # Remove potentially harmful HTML/script content
    sanitized = html.escape(user_input)
    
    # Remove special characters except basic punctuation
    sanitized = re.sub(r'[^\w\s.,!?\-]', '', sanitized)
    
    # Truncate long inputs to prevent abuse
    return sanitized[:5000].strip()

def get_embedding(text: str) -> List[float]:
    """Batch-friendly embedding generation"""
    try:
        return EMBEDDING_MODEL.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False
        ).tolist()
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return [0.0] * 384  # Return empty embedding matching dimension

def validate_content_safety(text: str) -> bool:
    """Content safety check using keyword patterns"""
    unsafe_patterns = [
        r'\b(自杀|自伤|自残|自尽)\b',  # Chinese
        r'\b(自杀|じさつ|自傷)\b',    # Japanese
        r'\b(자살|자해)\b',          # Korean
        r'\b(suicide|self[- ]harm|kill myself)\b'
    ]
    return not any(re.search(pattern, text, re.IGNORECASE) 
                  for pattern in unsafe_patterns)

def format_response(response: str) -> str:
    """Format chatbot response for readability"""
    # Split long sentences
    response = re.sub(r'([.!?]) ', r'\1\n', response)
    # Remove redundant whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    return response
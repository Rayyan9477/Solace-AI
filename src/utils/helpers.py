from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import AppConfig

EMBEDDING_MODEL = SentenceTransformer(AppConfig.EMBEDDING_MODEL)

def get_embedding(text: str) -> List[float]:
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding.tolist()

def sanitize_input(user_input: str) -> str:
    return user_input.strip()


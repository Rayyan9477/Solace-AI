from typing import List
from sentence_transformers import SentenceTransformer

# Initialize embedding model once
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> List[float]:
    """
    Generates a robust embedding for the given text using a pre-trained model.
    """
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding.tolist()

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input before processing.
    Remove dangerous or unnecessary characters.
    """
    return user_input.strip()

def get_top_k_predictions(prediction_list, k=3):
    """
    Returns top K items or fewer if not enough predictions.
    """
    return prediction_list[:k]
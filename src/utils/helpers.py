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

def get_basic_embedding(text: str) -> List[float]:
    """
    Computes a very naive embedding for demonstration purposes.
    Consider using a real embedding model in production.
    """
    # Example naive logic: Convert each character to its ordinal value
    # and normalize by some factor to get a float list.
    # This is obviously not suitable for real production usage.
    numeric_values = [ord(ch) for ch in text]
    max_value = max(numeric_values) if numeric_values else 1
    return [val / max_value for val in numeric_values]

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
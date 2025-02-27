from typing import List, Dict, Any, Optional, Union
import html
import re
from sentence_transformers import SentenceTransformer
from config.settings import AppConfig
import logging
import json
from datetime import datetime
from pathlib import Path
import aiofiles
from agno.utils import AsyncTokenizer, TextProcessor
from agno.types import Document

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

class TextHelper:
    """Helper class for text processing operations"""
    
    def __init__(self):
        self.tokenizer = AsyncTokenizer()
        self.processor = TextProcessor()
        
    async def preprocess_text(
        self,
        text: str,
        clean: bool = True,
        normalize: bool = True
    ) -> str:
        """Preprocess text with cleaning and normalization"""
        processed = text
        if clean:
            processed = await self.processor.clean_text(processed)
        if normalize:
            processed = await self.processor.normalize_text(processed)
        return processed
        
    async def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """Split text into overlapping chunks"""
        return await self.processor.chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """Extract key terms from text"""
        return await self.processor.extract_keywords(
            text=text,
            max_keywords=max_keywords
        )

class DocumentHelper:
    """Helper class for document operations"""
    
    def __init__(self):
        self.text_helper = TextHelper()
        
    async def create_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Document:
        """Create a document with processed text and metadata"""
        try:
            # Preprocess text
            processed_text = await self.text_helper.preprocess_text(text)
            
            # Extract keywords
            keywords = await self.text_helper.extract_keywords(processed_text)
            
            # Create document
            doc = Document(
                id=doc_id,
                text=processed_text,
                metadata={
                    'keywords': keywords,
                    'timestamp': datetime.now().isoformat(),
                    'char_count': len(processed_text),
                    **(metadata or {})
                }
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to create document: {str(e)}")
            raise

class FileHelper:
    """Helper class for file operations"""
    
    def __init__(self):
        self.config = AppConfig()
        
    async def save_json(
        self,
        data: Union[Dict, List],
        filepath: Union[str, Path],
        pretty: bool = True
    ) -> None:
        """Save data to JSON file asynchronously"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(filepath, 'w') as f:
                if pretty:
                    await f.write(json.dumps(data, indent=2))
                else:
                    await f.write(json.dumps(data))
                    
        except Exception as e:
            logger.error(f"Failed to save JSON file: {str(e)}")
            raise
            
    async def load_json(
        self,
        filepath: Union[str, Path]
    ) -> Union[Dict, List]:
        """Load data from JSON file asynchronously"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
                
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to load JSON file: {str(e)}")
            raise

class ValidationHelper:
    """Helper class for data validation"""
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> bool:
        """Validate metadata structure and content"""
        required_fields = ['timestamp', 'source']
        return all(field in metadata for field in required_fields)
        
    @staticmethod
    def validate_document(doc: Document) -> bool:
        """Validate document structure and content"""
        if not doc.text or not isinstance(doc.text, str):
            return False
        if not doc.metadata or not isinstance(doc.metadata, dict):
            return False
        return True
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
import asyncio
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

class TextHelper:
    """Helper class for text processing operations"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    async def preprocess_text(
        self,
        text: str,
        clean: bool = True,
        normalize: bool = True
    ) -> str:
        """Preprocess text with cleaning and normalization"""
        processed = text
        
        if clean:
            # Remove HTML tags
            processed = BeautifulSoup(processed, 'html.parser').get_text()
            # Remove special characters
            processed = re.sub(r'[^\w\s.,!?\-]', '', processed)
            # Remove extra whitespace
            processed = ' '.join(processed.split())
            
        if normalize:
            # Tokenize and normalize words
            words = word_tokenize(processed.lower())
            # Remove stop words and lemmatize
            words = [
                self.lemmatizer.lemmatize(word)
                for word in words
                if word not in self.stop_words
            ]
            processed = ' '.join(words)
            
        return processed
        
    async def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """Split text into overlapping chunks"""
        # Split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Add current chunk to results
                chunks.append(' '.join(current_chunk))
                # Keep overlap sentences
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_size += len(s)
                current_chunk = overlap_chunk
                current_length = overlap_size
                
            current_chunk.append(sentence)
            current_length += sentence_length
            
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """Extract key terms from text"""
        # Tokenize and normalize
        words = word_tokenize(text.lower())
        words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words and word.isalnum()
        ]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency and return top keywords
        keywords = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [word for word, _ in keywords[:max_keywords]]
        
    def get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()

class DocumentHelper:
    """Helper class for document operations"""
    
    def __init__(self):
        self.text_helper = TextHelper()
        self.embedding_model = SentenceTransformer(AppConfig.EMBEDDING_MODEL)
        
    async def create_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a document with processed text and metadata"""
        try:
            # Preprocess text
            processed_text = await self.text_helper.preprocess_text(text)
            
            # Extract keywords
            keywords = await self.text_helper.extract_keywords(processed_text)
            
            # Generate embedding
            embedding = self.embedding_model.encode(
                processed_text,
                convert_to_tensor=False,
                show_progress_bar=False
            ).tolist()
            
            # Create document
            doc = {
                'id': doc_id or str(hash(processed_text)),
                'text': processed_text,
                'embedding': embedding,
                'metadata': {
                    'keywords': keywords,
                    'timestamp': self.text_helper.get_timestamp(),
                    'char_count': len(processed_text),
                    **(metadata or {})
                }
            }
            
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
    def validate_document(doc: Dict[str, Any]) -> bool:
        """Validate document structure and content"""
        if not doc.get('text') or not isinstance(doc['text'], str):
            return False
        if not doc.get('metadata') or not isinstance(doc['metadata'], dict):
            return False
        if not doc.get('embedding') or not isinstance(doc['embedding'], list):
            return False
        return True
"""
Text Feature Extraction Components
"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from .base import BaseFeatureExtractor, FeatureExtractionResult, FeatureType, ExtractionStatus

logger = logging.getLogger(__name__)

# NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModel
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK or transformers not available. Some text features may not work.")

class TextFeatureExtractor(BaseFeatureExtractor):
    """Main text feature extractor using sentence transformers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        self.model = None
        
    def _get_feature_type(self) -> FeatureType:
        return FeatureType.TEXT
    
    def _initialize_components(self):
        """Initialize sentence transformer model"""
        if not NLTK_AVAILABLE:
            raise ImportError("sentence-transformers required for text feature extraction")
        
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Loaded sentence transformer model: {self.model_name}")
    
    def extract(self, data: Any, **kwargs) -> FeatureExtractionResult:
        """Extract text features using sentence transformers"""
        start_time = time.time()
        
        # Validate input
        valid, errors = self.validate_input(data)
        if not valid:
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features=np.array([]),
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=errors
            )
        
        try:
            # Convert input to string if necessary
            if isinstance(data, dict):
                text = data.get('text', str(data))
            else:
                text = str(data)
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract embeddings
            embeddings = self.model.encode([cleaned_text])[0]
            
            confidence = self._calculate_confidence(text, embeddings)
            
            result = FeatureExtractionResult(
                feature_type=self.feature_type,
                features=embeddings,
                confidence=confidence,
                status=ExtractionStatus.SUCCESS,
                extraction_time=time.time() - start_time,
                metadata={
                    'text_length': len(text),
                    'cleaned_length': len(cleaned_text),
                    'model_name': self.model_name,
                    'embedding_dim': len(embeddings)
                }
            )
            
            self._log_extraction(result)
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features=np.array([]),
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove URLs (simplified)
        text = re.sub(r'https?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        return text
    
    def _calculate_confidence(self, text: str, embeddings: np.ndarray) -> float:
        """Calculate confidence score for text extraction"""
        base_confidence = 0.8
        
        # Adjust based on text length
        if len(text) < 10:
            base_confidence *= 0.5
        elif len(text) > 1000:
            base_confidence *= 0.9
        
        # Adjust based on embedding quality (variance as proxy)
        embedding_variance = np.var(embeddings)
        if embedding_variance < 0.01:  # Low variance suggests poor representation
            base_confidence *= 0.7
        
        return min(1.0, base_confidence)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of text embeddings"""
        if self.model:
            return {"embeddings": self.model.get_sentence_embedding_dimension()}
        return {"embeddings": 384}  # Default for all-MiniLM-L6-v2
    
    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate text input"""
        issues = []
        
        if data is None:
            issues.append("Input data is None")
        elif isinstance(data, str) and len(data.strip()) == 0:
            issues.append("Input text is empty")
        elif isinstance(data, dict) and 'text' not in data:
            issues.append("Dictionary input must contain 'text' key")
        
        return len(issues) == 0, issues

class SemanticAnalyzer(BaseFeatureExtractor):
    """Semantic analysis of text content"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.symptom_keywords = self._load_symptom_keywords()
        self.emotion_keywords = self._load_emotion_keywords()
        
    def _get_feature_type(self) -> FeatureType:
        return FeatureType.TEXT
    
    def _load_symptom_keywords(self) -> Dict[str, List[str]]:
        """Load symptom-related keywords"""
        return {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'tension', 'stress'],
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'down', 'blue'],
            'sleep': ['insomnia', 'sleepless', 'tired', 'exhausted', 'fatigue', 'drowsy'],
            'appetite': ['hungry', 'appetite', 'eating', 'food', 'weight', 'nutrition'],
            'concentration': ['focus', 'attention', 'concentrate', 'memory', 'forgetful'],
            'energy': ['energy', 'motivation', 'drive', 'vigor', 'strength', 'power']
        }
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion-related keywords"""
        return {
            'positive': ['happy', 'joy', 'excited', 'pleased', 'content', 'satisfied'],
            'negative': ['angry', 'frustrated', 'annoyed', 'upset', 'irritated'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular']
        }
    
    def extract(self, data: Any, **kwargs) -> FeatureExtractionResult:
        """Extract semantic features from text"""
        start_time = time.time()
        
        valid, errors = self.validate_input(data)
        if not valid:
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features={},
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=errors
            )
        
        try:
            # Convert to text
            text = str(data).lower() if not isinstance(data, dict) else str(data.get('text', '')).lower()
            
            # Extract semantic features
            features = {}
            
            # Symptom presence scores
            for symptom, keywords in self.symptom_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text) / len(keywords)
                features[f'symptom_{symptom}'] = score
            
            # Emotion presence scores
            for emotion, keywords in self.emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text) / len(keywords)
                features[f'emotion_{emotion}'] = score
            
            # Text complexity metrics
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
            
            # Calculate overall confidence
            confidence = self._calculate_semantic_confidence(features, text)
            
            result = FeatureExtractionResult(
                feature_type=self.feature_type,
                features=features,
                confidence=confidence,
                status=ExtractionStatus.SUCCESS,
                extraction_time=time.time() - start_time,
                metadata={
                    'analyzed_symptoms': list(self.symptom_keywords.keys()),
                    'analyzed_emotions': list(self.emotion_keywords.keys())
                }
            )
            
            self._log_extraction(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features={},
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _calculate_semantic_confidence(self, features: Dict[str, float], text: str) -> float:
        """Calculate confidence for semantic analysis"""
        base_confidence = 0.7
        
        # Higher confidence for longer, more detailed text
        if len(text) > 100:
            base_confidence += 0.1
        if len(text) > 500:
            base_confidence += 0.1
        
        # Check if any symptoms/emotions were detected
        symptom_scores = [v for k, v in features.items() if k.startswith('symptom_') and v > 0]
        emotion_scores = [v for k, v in features.items() if k.startswith('emotion_') and v > 0]
        
        if symptom_scores or emotion_scores:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

class SentimentExtractor(BaseFeatureExtractor):
    """Sentiment analysis for text"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.analyzer = None
        
    def _get_feature_type(self) -> FeatureType:
        return FeatureType.TEXT
    
    def _initialize_components(self):
        """Initialize sentiment analyzer"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.analyzer = SentimentIntensityAnalyzer()
            except (ImportError, LookupError, OSError, RuntimeError):
                logger.warning("Could not initialize NLTK sentiment analyzer")
                
        # Fallback to transformers-based sentiment analysis
        if not self.analyzer:
            try:
                self.analyzer = pipeline("sentiment-analysis",
                                       model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except (ImportError, OSError, RuntimeError, ValueError):
                logger.warning("Could not initialize transformers sentiment analyzer")
    
    def extract(self, data: Any, **kwargs) -> FeatureExtractionResult:
        """Extract sentiment features"""
        start_time = time.time()
        
        valid, errors = self.validate_input(data)
        if not valid:
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features={},
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=errors
            )
        
        try:
            text = str(data) if not isinstance(data, dict) else str(data.get('text', ''))
            
            if isinstance(self.analyzer, SentimentIntensityAnalyzer):
                # NLTK VADER analysis
                scores = self.analyzer.polarity_scores(text)
                features = {
                    'sentiment_positive': scores['pos'],
                    'sentiment_negative': scores['neg'],
                    'sentiment_neutral': scores['neu'],
                    'sentiment_compound': scores['compound']
                }
                confidence = 0.8
                
            elif hasattr(self.analyzer, '__call__'):
                # Transformers pipeline analysis
                result = self.analyzer(text[:512])  # Truncate for transformer models
                if isinstance(result, list) and len(result) > 0:
                    sentiment_result = result[0]
                    features = {
                        'sentiment_label': sentiment_result['label'],
                        'sentiment_score': sentiment_result['score']
                    }
                    confidence = sentiment_result['score']
                else:
                    features = {'sentiment_score': 0.0}
                    confidence = 0.0
            else:
                # No analyzer available
                features = {'sentiment_score': 0.0}
                confidence = 0.0
            
            result = FeatureExtractionResult(
                feature_type=self.feature_type,
                features=features,
                confidence=confidence,
                status=ExtractionStatus.SUCCESS,
                extraction_time=time.time() - start_time,
                metadata={'analyzer_type': type(self.analyzer).__name__ if self.analyzer else 'none'}
            )
            
            self._log_extraction(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment extraction: {e}")
            return FeatureExtractionResult(
                feature_type=self.feature_type,
                features={},
                confidence=0.0,
                status=ExtractionStatus.FAILED,
                extraction_time=time.time() - start_time,
                errors=[str(e)]
            )
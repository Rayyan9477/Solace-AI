"""
Base classes for feature extraction components
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features that can be extracted"""
    TEXT = "text"
    VOICE = "voice"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    PHYSIOLOGICAL = "physiological"
    CONTEXTUAL = "contextual"
    MULTIMODAL = "multimodal"

class ExtractionStatus(Enum):
    """Status of feature extraction process"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class FeatureExtractionResult:
    """Result of feature extraction process"""
    feature_type: FeatureType
    features: Union[np.ndarray, Dict[str, Any]]
    confidence: float
    status: ExtractionStatus
    extraction_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'feature_type': self.feature_type.value,
            'features': self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            'confidence': self.confidence,
            'status': self.status.value,
            'extraction_time': self.extraction_time,
            'metadata': self.metadata,
            'errors': self.errors,
            'timestamp': self.timestamp.isoformat()
        }

class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.feature_type = self._get_feature_type()
        self.is_initialized = False
        self._extraction_history = []
        
    @abstractmethod
    def _get_feature_type(self) -> FeatureType:
        """Get the feature type this extractor handles"""
        pass
    
    @abstractmethod
    def extract(self, data: Any, **kwargs) -> FeatureExtractionResult:
        """Extract features from input data"""
        pass
    
    def initialize(self) -> bool:
        """Initialize the feature extractor"""
        try:
            self._initialize_components()
            self.is_initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize extractor-specific components - override in subclasses"""
        pass
    
    def validate_input(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data - override in subclasses"""
        if data is None:
            return False, ["Input data is None"]
        return True, []
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get the dimensions of extracted features"""
        return {"default": 0}
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about feature extraction performance"""
        if not self._extraction_history:
            return {"total_extractions": 0}
        
        successful = [h for h in self._extraction_history if h['status'] == ExtractionStatus.SUCCESS]
        failed = [h for h in self._extraction_history if h['status'] == ExtractionStatus.FAILED]
        
        avg_time = np.mean([h['extraction_time'] for h in self._extraction_history])
        avg_confidence = np.mean([h['confidence'] for h in successful]) if successful else 0
        
        return {
            "total_extractions": len(self._extraction_history),
            "successful_extractions": len(successful),
            "failed_extractions": len(failed),
            "success_rate": len(successful) / len(self._extraction_history) if self._extraction_history else 0,
            "average_extraction_time": avg_time,
            "average_confidence": avg_confidence
        }
    
    def _log_extraction(self, result: FeatureExtractionResult):
        """Log extraction result for statistics"""
        self._extraction_history.append({
            'status': result.status,
            'extraction_time': result.extraction_time,
            'confidence': result.confidence,
            'timestamp': result.timestamp
        })
        
        # Keep only last 1000 entries
        if len(self._extraction_history) > 1000:
            self._extraction_history = self._extraction_history[-1000:]

class FeatureValidator:
    """Utility class for validating extracted features"""
    
    @staticmethod
    def validate_vector_features(features: np.ndarray, 
                               expected_shape: Optional[Tuple[int, ...]] = None,
                               value_range: Optional[Tuple[float, float]] = None) -> Tuple[bool, List[str]]:
        """Validate vector-based features"""
        issues = []
        
        if not isinstance(features, np.ndarray):
            issues.append("Features must be numpy array")
            return False, issues
        
        if expected_shape and features.shape != expected_shape:
            issues.append(f"Expected shape {expected_shape}, got {features.shape}")
        
        if np.any(np.isnan(features)):
            issues.append("Features contain NaN values")
        
        if np.any(np.isinf(features)):
            issues.append("Features contain infinite values")
        
        if value_range:
            min_val, max_val = value_range
            if np.any(features < min_val) or np.any(features > max_val):
                issues.append(f"Features outside expected range [{min_val}, {max_val}]")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_dict_features(features: Dict[str, Any], 
                             required_keys: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Validate dictionary-based features"""
        issues = []
        
        if not isinstance(features, dict):
            issues.append("Features must be dictionary")
            return False, issues
        
        if required_keys:
            missing_keys = set(required_keys) - set(features.keys())
            if missing_keys:
                issues.append(f"Missing required keys: {list(missing_keys)}")
        
        # Check for None values
        none_keys = [k for k, v in features.items() if v is None]
        if none_keys:
            issues.append(f"Keys with None values: {none_keys}")
        
        return len(issues) == 0, issues

class FeatureNormalizer:
    """Utility class for normalizing extracted features"""
    
    @staticmethod
    def normalize_vector(features: np.ndarray, method: str = "l2") -> np.ndarray:
        """Normalize vector features"""
        if method == "l2":
            norm = np.linalg.norm(features)
            return features / norm if norm > 0 else features
        elif method == "minmax":
            min_val, max_val = features.min(), features.max()
            if max_val > min_val:
                return (features - min_val) / (max_val - min_val)
            return features
        elif method == "zscore":
            mean_val, std_val = features.mean(), features.std()
            if std_val > 0:
                return (features - mean_val) / std_val
            return features
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def normalize_dict_values(features: Dict[str, float], 
                            method: str = "minmax") -> Dict[str, float]:
        """Normalize dictionary values"""
        values = np.array(list(features.values()))
        
        if method == "minmax":
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                normalized_values = (values - min_val) / (max_val - min_val)
            else:
                normalized_values = values
        elif method == "zscore":
            mean_val, std_val = values.mean(), values.std()
            if std_val > 0:
                normalized_values = (values - mean_val) / std_val
            else:
                normalized_values = values
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return dict(zip(features.keys(), normalized_values))

class FeatureAggregator:
    """Utility class for aggregating features from multiple sources"""
    
    @staticmethod
    def concatenate_vectors(*vectors: np.ndarray) -> np.ndarray:
        """Concatenate multiple feature vectors"""
        return np.concatenate(vectors, axis=-1)
    
    @staticmethod
    def weighted_average_vectors(*vectors_and_weights: Tuple[np.ndarray, float]) -> np.ndarray:
        """Compute weighted average of feature vectors"""
        weighted_sum = np.zeros_like(vectors_and_weights[0][0])
        total_weight = 0
        
        for vector, weight in vectors_and_weights:
            weighted_sum += vector * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any], strategy: str = "update") -> Dict[str, Any]:
        """Merge multiple feature dictionaries"""
        if strategy == "update":
            result = {}
            for d in dicts:
                result.update(d)
            return result
        elif strategy == "average":
            # Average numeric values with same keys
            all_keys = set()
            for d in dicts:
                all_keys.update(d.keys())
            
            result = {}
            for key in all_keys:
                values = [d.get(key, 0) for d in dicts if key in d]
                if values and all(isinstance(v, (int, float)) for v in values):
                    result[key] = sum(values) / len(values)
                else:
                    # Take the last non-None value
                    for d in reversed(dicts):
                        if key in d and d[key] is not None:
                            result[key] = d[key]
                            break
            
            return result
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

class FeatureCache:
    """Simple in-memory cache for extracted features"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_times = {}
    
    def get(self, key: str) -> Optional[FeatureExtractionResult]:
        """Get cached feature extraction result"""
        if key in self._cache:
            self._access_times[key] = datetime.utcnow()
            return self._cache[key]
        return None
    
    def put(self, key: str, result: FeatureExtractionResult):
        """Cache feature extraction result"""
        if len(self._cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[lru_key]
            del self._access_times[lru_key]
        
        self._cache[key] = result
        self._access_times[key] = datetime.utcnow()
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
        self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': 0.0  # Would need hit/miss tracking for accurate rate
        }
"""
Feature Extractors Module

This module contains all feature extraction components organized by data type:
- Text feature extraction (NLP, embeddings, semantic analysis)
- Voice/Audio feature extraction (acoustic, prosodic, emotional)
- Behavioral pattern extraction (activity, social, temporal patterns)
- Physiological feature extraction (biometric, health indicators)
- Contextual feature extraction (environmental, situational context)
"""

from .base import BaseFeatureExtractor, FeatureExtractionResult
from .text_extractors import TextFeatureExtractor, SemanticAnalyzer, SentimentExtractor
from .voice_extractors import VoiceFeatureExtractor, AudioFeatureExtractor, EmotionalVoiceAnalyzer
from .behavioral_extractors import BehavioralAnalyzer, ActivityPatternExtractor, SocialPatternExtractor
from .temporal_extractors import TemporalAnalyzer, PatternDetector, TrendAnalyzer
from .contextual_extractors import ContextualAnalyzer, EnvironmentalExtractor
from .multimodal_fusion import MultiModalFeatureFusion, FeatureFusionEngine

__all__ = [
    'BaseFeatureExtractor',
    'FeatureExtractionResult',
    'TextFeatureExtractor',
    'SemanticAnalyzer',
    'SentimentExtractor',
    'VoiceFeatureExtractor',
    'AudioFeatureExtractor', 
    'EmotionalVoiceAnalyzer',
    'BehavioralAnalyzer',
    'ActivityPatternExtractor',
    'SocialPatternExtractor',
    'TemporalAnalyzer',
    'PatternDetector',
    'TrendAnalyzer',
    'ContextualAnalyzer',
    'EnvironmentalExtractor',
    'MultiModalFeatureFusion',
    'FeatureFusionEngine'
]
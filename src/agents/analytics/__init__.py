"""Analytics agents module - Learning and pattern recognition"""

from .adaptive_learning_agent import AdaptiveLearningAgent
from .pattern_recognition_engine import PatternRecognitionEngine
from .insights_generation_system import InsightGenerationSystem
from .outcome_tracker import OutcomeTracker

# Alias for backward compatibility
InsightsGenerationSystem = InsightGenerationSystem

__all__ = [
    'AdaptiveLearningAgent',
    'PatternRecognitionEngine',
    'InsightGenerationSystem',
    'InsightsGenerationSystem',  # Backward compat
    'OutcomeTracker'
]

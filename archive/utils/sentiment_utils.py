"""
Sentiment Analysis Utilities - Centralized sentiment analysis functions.

This module provides utility functions for sentiment analysis to eliminate
code duplication across agents (particularly EmotionAgent).
"""

from typing import Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

# Initialize VADER sentiment analyzer as module-level singleton
_sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using VADER sentiment analysis.

    This function provides a centralized implementation of sentiment analysis
    to avoid code duplication across agents.

    Args:
        text: The text to analyze for sentiment

    Returns:
        Dictionary containing:
        - sentiment_scores: Raw VADER scores (neg, neu, pos, compound)
        - compound_score: Overall sentiment score (-1 to 1)
        - normalized_intensity: Intensity scaled to 0-10
        - success: Whether analysis succeeded
        - error: Error message if analysis failed

    Example:
        >>> result = analyze_text_sentiment("I feel great today!")
        >>> print(result['compound_score'])
        0.6369
        >>> print(result['normalized_intensity'])
        6.369
    """
    try:
        # Perform VADER sentiment analysis
        sentiment = _sentiment_analyzer.polarity_scores(text)

        return {
            'sentiment_scores': sentiment,
            'compound_score': sentiment['compound'],
            'normalized_intensity': abs(sentiment['compound']) * 10,
            'success': True
        }

    except Exception as e:
        logger.warning(f"VADER sentiment analysis failed: {str(e)}")
        return {
            'sentiment_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
            'compound_score': 0.0,
            'normalized_intensity': 0.0,
            'success': False,
            'error': str(e)
        }


def get_emotion_from_sentiment(compound_score: float) -> Dict[str, Any]:
    """
    Convert VADER compound score to emotion labels.

    Args:
        compound_score: VADER compound score (-1 to 1)

    Returns:
        Dictionary containing:
        - primary_emotion: Main emotion label
        - secondary_emotions: List of related emotions
        - intensity: Emotion intensity (1-10)

    Example:
        >>> emotion = get_emotion_from_sentiment(0.6)
        >>> print(emotion['primary_emotion'])
        'happy'
    """
    emotion_data = {
        'intensity': min(10, max(1, int(abs(compound_score) * 10)))
    }

    if compound_score > 0.05:
        emotion_data['primary_emotion'] = 'happy'
        emotion_data['secondary_emotions'] = ['content', 'satisfied']
    elif compound_score < -0.05:
        emotion_data['primary_emotion'] = 'sad'
        emotion_data['secondary_emotions'] = ['disappointed', 'frustrated']
    else:
        emotion_data['primary_emotion'] = 'neutral'
        emotion_data['secondary_emotions'] = ['calm', 'balanced']

    return emotion_data


def detect_emotional_triggers(text: str) -> list:
    """
    Detect potential emotional triggers in text based on keywords.

    Args:
        text: The text to analyze for triggers

    Returns:
        List of detected triggers

    Example:
        >>> triggers = detect_emotional_triggers("I'm stressed about work and family")
        >>> print(triggers)
        ['work-related stress', 'family concerns']
    """
    triggers = []
    text_lower = text.lower()

    # Work-related triggers
    if 'work' in text_lower or 'job' in text_lower:
        triggers.append('work-related stress')

    # Family-related triggers
    if 'family' in text_lower or 'parent' in text_lower:
        triggers.append('family concerns')

    # Relationship triggers
    if 'relationship' in text_lower or 'partner' in text_lower:
        triggers.append('relationship issues')

    # Health triggers
    if 'health' in text_lower or 'sick' in text_lower:
        triggers.append('health concerns')

    # Financial triggers
    if 'money' in text_lower or 'financial' in text_lower or 'debt' in text_lower:
        triggers.append('financial stress')

    # Loss/grief triggers
    if 'loss' in text_lower or 'death' in text_lower or 'died' in text_lower:
        triggers.append('grief/loss')

    return triggers


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """
    Get the module-level sentiment analyzer instance.

    Returns:
        SentimentIntensityAnalyzer: The shared VADER analyzer
    """
    return _sentiment_analyzer

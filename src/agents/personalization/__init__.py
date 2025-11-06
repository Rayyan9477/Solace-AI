"""Personalization agents module - User personalization and feedback"""

from .personalization_engine import PersonalizationEngine
from .feedback_integration_system import FeedbackProcessor

# Alias for backward compatibility
FeedbackIntegrationSystem = FeedbackProcessor

__all__ = ['PersonalizationEngine', 'FeedbackProcessor', 'FeedbackIntegrationSystem']

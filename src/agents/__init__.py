"""
Agents module initialization - Hierarchically organized agent implementations

This package contains various agent implementations for the Contextual-Chatbot,
organized into logical categories:

- base: Core agent infrastructure
- orchestration: Agent coordination and supervision
- core: Primary user-facing agents (chat, emotion, personality, safety)
- clinical: Diagnosis and therapy agents
- support: Search and crawler agents
- analytics: Learning, pattern recognition, insights
- personalization: User personalization and feedback systems
- security: Privacy and security agents
- therapeutic_friction: Growth-oriented challenge systems
- validation: Response validation
"""

# Base agents
from .base import Agent

# Orchestration
from .orchestration import AgentOrchestrator, SupervisorAgent

# Core agents
from .core import ChatAgent, EmotionAgent, PersonalityAgent, SafetyAgent

# Clinical agents
from .clinical import DiagnosisAgent, EnhancedDiagnosisAgent, TherapyAgent

# Support agents
from .support import SearchAgent, CrawlerAgent

# Analytics agents
from .analytics import (
    AdaptiveLearningAgent,
    PatternRecognitionEngine,
    InsightsGenerationSystem,
    OutcomeTracker
)

# Personalization agents
from .personalization import PersonalizationEngine, FeedbackIntegrationSystem

# Security agents
from .security import PrivacyProtectionSystem

# Therapeutic friction
from .therapeutic_friction import (
    BaseFrictionAgent,
    FrictionAgentType,
    ReadinessAssessmentAgent,
    BreakthroughDetectionAgent,
    FrictionCoordinator
)

# Validation
from .validation import ResponseValidator

__all__ = [
    # Base
    'Agent',
    # Orchestration
    'AgentOrchestrator',
    'SupervisorAgent',
    # Core
    'ChatAgent',
    'EmotionAgent',
    'PersonalityAgent',
    'SafetyAgent',
    # Clinical
    'DiagnosisAgent',
    'EnhancedDiagnosisAgent',
    'TherapyAgent',
    # Support
    'SearchAgent',
    'CrawlerAgent',
    # Analytics
    'AdaptiveLearningAgent',
    'PatternRecognitionEngine',
    'InsightsGenerationSystem',
    'OutcomeTracker',
    # Personalization
    'PersonalizationEngine',
    'FeedbackIntegrationSystem',
    # Security
    'PrivacyProtectionSystem',
    # Therapeutic Friction
    'BaseFrictionAgent',
    'FrictionAgentType',
    'ReadinessAssessmentAgent',
    'BreakthroughDetectionAgent',
    'FrictionCoordinator',
    # Validation
    'ResponseValidator',
]
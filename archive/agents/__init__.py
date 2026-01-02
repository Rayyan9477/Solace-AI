"""
Agents module initialization - Hierarchically organized agent implementations

This package contains agent implementations for the Contextual-Chatbot,
organized into functional categories:

- base: Core agent infrastructure
- orchestration: Agent coordination and supervision (AgentOrchestrator)
- core: Primary user-facing agents (chat, emotion, personality, safety)
- clinical: Clinical agents (diagnosis, therapy)
- support: Utility agents (search, crawler)
- therapeutic_friction: Growth-oriented challenge system with sub-agents
- validation: Response validation components
"""

# Base agents
from .base import BaseAgent

# Orchestration - Core coordination system
from .orchestration import AgentOrchestrator, SupervisorAgent

# Core agents - Primary user interaction
from .core import ChatAgent, EmotionAgent, PersonalityAgent, SafetyAgent

# Clinical agents - Therapeutic and diagnostic
from .clinical import DiagnosisAgent, EnhancedDiagnosisAgent, TherapyAgent

# Support agents - Utility functions
from .support import SearchAgent, CrawlerAgent

# Therapeutic friction - Growth challenges
from .therapeutic_friction import (
    BaseFrictionAgent,
    FrictionAgentType,
    ReadinessAssessmentAgent,
    BreakthroughDetectionAgent,
    FrictionCoordinator
)

__all__ = [
    # Base
    'BaseAgent',
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
    # Therapeutic Friction
    'BaseFrictionAgent',
    'FrictionAgentType',
    'ReadinessAssessmentAgent',
    'BreakthroughDetectionAgent',
    'FrictionCoordinator',
]
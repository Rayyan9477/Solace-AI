"""
Enhanced Memory Module

This module provides advanced memory management for therapeutic contexts,
including insight storage, session continuity, and progress tracking.
"""

try:
    from .enhanced_memory_system import (
        EnhancedMemorySystem,
        TherapeuticInsight,
        ProgressMilestone,
        SessionMemory,
        UserProfileMemory
    )
except ImportError:
    EnhancedMemorySystem = None
    TherapeuticInsight = None
    ProgressMilestone = None
    SessionMemory = None
    UserProfileMemory = None

# Legacy compatibility
try:
    from .semantic_memory.semantic_memory_manager import SemanticMemoryManager
except ImportError:
    SemanticMemoryManager = None

__all__ = [
    'EnhancedMemorySystem',
    'TherapeuticInsight',
    'ProgressMilestone',
    'SessionMemory',
    'UserProfileMemory',
    'SemanticMemoryManager'
]
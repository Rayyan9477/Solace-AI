"""
Enhanced Memory Module

This module provides advanced memory management for therapeutic contexts,
including insight storage, session continuity, progress tracking, and
centralized memory factory for agent initialization.
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

# Memory factory functions (canonical location)
try:
    from .memory_factory import (
        create_agent_memory,
        create_stateless_memory,
        get_or_create_memory
    )
except ImportError:
    create_agent_memory = None
    create_stateless_memory = None
    get_or_create_memory = None

# Legacy compatibility
try:
    from .semantic_memory.semantic_memory_manager import SemanticMemoryManager
except ImportError:
    SemanticMemoryManager = None

__all__ = [
    # Enhanced memory system
    'EnhancedMemorySystem',
    'TherapeuticInsight',
    'ProgressMilestone',
    'SessionMemory',
    'UserProfileMemory',
    'SemanticMemoryManager',
    # Memory factory functions
    'create_agent_memory',
    'create_stateless_memory',
    'get_or_create_memory',
]
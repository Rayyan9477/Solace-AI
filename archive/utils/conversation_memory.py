"""
Conversation Memory - Backwards Compatibility Re-export

DEPRECATED: This module has been relocated to src/memory/conversation_memory.py
This file exists for backwards compatibility only. Update imports to:

    from src.memory.conversation_memory import ConversationMemory

This re-export will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "Importing from src.utils.conversation_memory is deprecated. "
    "Please update imports to: from src.memory.conversation_memory import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location
from src.memory.conversation_memory import ConversationMemory

__all__ = ['ConversationMemory']

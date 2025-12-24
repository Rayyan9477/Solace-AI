"""
Context Aware Memory Adapter - Backwards Compatibility Re-export

DEPRECATED: This module has been relocated to src/memory/context_aware_memory.py
This file exists for backwards compatibility only. Update imports to:

    from src.memory.context_aware_memory import ContextAwareMemoryAdapter

This re-export will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "Importing from src.utils.context_aware_memory is deprecated. "
    "Please update imports to: from src.memory.context_aware_memory import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location
from src.memory.context_aware_memory import ContextAwareMemoryAdapter

__all__ = ['ContextAwareMemoryAdapter']

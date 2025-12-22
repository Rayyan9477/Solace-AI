"""
Memory Factory - Backwards Compatibility Re-export

DEPRECATED: This module has been relocated to src/memory/memory_factory.py
This file exists for backwards compatibility only. Update imports to:

    from src.memory.memory_factory import create_agent_memory, get_or_create_memory

This re-export will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "Importing from src.utils.memory_factory is deprecated. "
    "Please update imports to: from src.memory.memory_factory import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location
from src.memory.memory_factory import (
    create_agent_memory,
    create_stateless_memory,
    get_or_create_memory,
)

__all__ = [
    'create_agent_memory',
    'create_stateless_memory',
    'get_or_create_memory',
]

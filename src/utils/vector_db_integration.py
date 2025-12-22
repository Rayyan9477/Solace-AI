"""
Vector Database Integration - Backwards Compatibility Re-export

DEPRECATED: This module has been relocated to src/database/vector_db_integration.py
This file exists for backwards compatibility only. Update imports to:

    from src.database.vector_db_integration import get_central_vector_db, ...

This re-export will be removed in a future version.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "Importing from src.utils.vector_db_integration is deprecated. "
    "Please update imports to: from src.database.vector_db_integration import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from canonical location
from src.database.vector_db_integration import (
    get_central_vector_db,
    get_conversation_tracker,
    add_user_data,
    get_user_data,
    search_relevant_data,
    update_user_data,
    store_interaction,
    get_recent_interactions,
)

__all__ = [
    'get_central_vector_db',
    'get_conversation_tracker',
    'add_user_data',
    'get_user_data',
    'search_relevant_data',
    'update_user_data',
    'store_interaction',
    'get_recent_interactions',
]

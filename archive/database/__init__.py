"""
Database Module

This module provides database and vector storage functionality including:
- Central vector database for semantic storage
- Vector store implementations (FAISS, ChromaDB)
- Conversation tracking
- Vector database integration utilities
"""

try:
    from .central_vector_db import CentralVectorDB, DocumentType
except ImportError:
    CentralVectorDB = None
    DocumentType = None

try:
    from .vector_store import VectorStore, FaissVectorStore
except ImportError:
    VectorStore = None
    FaissVectorStore = None

try:
    from .conversation_tracker import ConversationTracker
except ImportError:
    ConversationTracker = None

# Vector DB integration utilities (canonical location)
try:
    from .vector_db_integration import (
        get_central_vector_db,
        get_conversation_tracker,
        add_user_data,
        get_user_data,
        search_relevant_data,
        update_user_data,
        store_interaction,
        get_recent_interactions,
    )
except ImportError:
    get_central_vector_db = None
    get_conversation_tracker = None
    add_user_data = None
    get_user_data = None
    search_relevant_data = None
    update_user_data = None
    store_interaction = None
    get_recent_interactions = None

__all__ = [
    # Core database classes
    'CentralVectorDB',
    'DocumentType',
    'VectorStore',
    'FaissVectorStore',
    'ConversationTracker',
    # Integration utilities
    'get_central_vector_db',
    'get_conversation_tracker',
    'add_user_data',
    'get_user_data',
    'search_relevant_data',
    'update_user_data',
    'store_interaction',
    'get_recent_interactions',
]

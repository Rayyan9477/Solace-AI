"""
Vector Database Integration Utilities

This module provides utility functions to access the central vector database
from anywhere in the application.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from database.central_vector_db import CentralVectorDB, DocumentType

# Import lazily to avoid circular imports
def get_module_manager():
    from components.base_module import get_module_manager as _get_module_manager
    return _get_module_manager()

logger = logging.getLogger(__name__)

# Query validation constants
MAX_QUERY_LENGTH = 10000  # Maximum query string length
MIN_QUERY_LENGTH = 1  # Minimum query string length
MAX_LIMIT = 100  # Maximum results limit
DEFAULT_LIMIT = 5  # Default results limit
VALID_DATA_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')  # Alphanumeric, underscore, hyphen only

def get_central_vector_db() -> Optional['CentralVectorDB']:
    """
    Get the central vector database instance
    
    Returns:
        CentralVectorDB instance or None if not available
    """
    try:
        module_manager = get_module_manager()
        vector_db_module = module_manager.get_module("central_vector_db")
        
        if not vector_db_module:
            logger.warning("Central vector database module not found")
            return None
        
        return vector_db_module.vector_db
    except Exception as e:
        logger.error(f"Error accessing central vector database: {str(e)}")
        return None

def add_user_data(data_type: str, data: Dict[str, Any], item_id: Optional[str] = None) -> str:
    """
    Add user data to the central vector database
    
    Args:
        data_type: Type of data (profile, diagnosis, personality, etc.)
        data: Data to store
        item_id: Optional ID for the data
        
    Returns:
        ID of the stored data or empty string if failed
    """
    try:
        vector_db = get_central_vector_db()
        if not vector_db:
            return ""
        
        # Import here to avoid circular imports
        from database.central_vector_db import DocumentType
        
        if data_type == "profile":
            return vector_db.add_user_profile(data)
        elif data_type == "diagnosis":
            return vector_db.add_diagnostic_data(data, item_id)
        elif data_type == "personality":
            return vector_db.add_personality_assessment(data, item_id)
        elif data_type == "knowledge":
            return vector_db.add_knowledge_item(data, item_id)
        elif data_type == "therapy":
            return vector_db.add_therapy_resource(data, item_id)
        else:
            # Use generic document storage
            return vector_db.add_document(
                document=data,
                namespace=data_type,
                doc_id=item_id
            )
    except Exception as e:
        logger.error(f"Error adding user data: {str(e)}")
        return ""

def get_user_data(data_type: str, item_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get user data from the central vector database
    
    Args:
        data_type: Type of data (profile, diagnosis, personality, etc.)
        item_id: Optional ID of the data (gets latest if not provided)
        
    Returns:
        Requested data or None if failed
    """
    try:
        vector_db = get_central_vector_db()
        if not vector_db:
            return None
        
        if data_type == "profile":
            return vector_db.get_user_profile()
        elif data_type == "diagnosis" and not item_id:
            return vector_db.get_latest_diagnosis()
        elif data_type == "personality" and not item_id:
            return vector_db.get_latest_personality()
        elif item_id:
            # Import here to avoid circular imports
            from database.central_vector_db import DocumentType
            
            # Get document by ID
            doc = vector_db.get_document(item_id)
            
            # Extract the data field if available
            if doc:
                if f"{data_type}_data" in doc:
                    return doc[f"{data_type}_data"]
                return doc
            
        return None
    except Exception as e:
        logger.error(f"Error getting user data: {str(e)}")
        return None

def search_relevant_data(query: str, data_types: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant data across all or specific data types

    Args:
        query: Search query
        data_types: Optional list of data types to search in (searches all if not provided)
        limit: Maximum number of results to return

    Returns:
        List of relevant data items
    """
    # Validate query parameter
    if not isinstance(query, str):
        logger.warning(f"Invalid query type: {type(query).__name__}, expected str")
        return []

    query = query.strip()
    if len(query) < MIN_QUERY_LENGTH:
        logger.warning("Empty or too short query provided")
        return []

    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(f"Query too long ({len(query)} chars), truncating to {MAX_QUERY_LENGTH}")
        query = query[:MAX_QUERY_LENGTH]

    # Validate limit parameter
    if not isinstance(limit, int) or limit < 1:
        logger.warning(f"Invalid limit {limit}, using default {DEFAULT_LIMIT}")
        limit = DEFAULT_LIMIT
    elif limit > MAX_LIMIT:
        logger.warning(f"Limit {limit} exceeds maximum {MAX_LIMIT}, capping")
        limit = MAX_LIMIT

    # Validate data_types parameter
    if data_types is not None:
        if not isinstance(data_types, list):
            logger.warning(f"Invalid data_types type: {type(data_types).__name__}, expected list")
            data_types = None
        else:
            # Filter and validate each data type
            valid_types = []
            for dt in data_types:
                if isinstance(dt, str) and VALID_DATA_TYPE_PATTERN.match(dt):
                    valid_types.append(dt)
                else:
                    logger.warning(f"Invalid data type '{dt}' skipped")
            data_types = valid_types if valid_types else None

    try:
        vector_db = get_central_vector_db()
        if not vector_db:
            return []

        # If specific data types are requested
        if data_types:
            results = []
            per_type_limit = max(1, limit // len(data_types))
            
            for data_type in data_types:
                # Import here to avoid circular imports
                from database.central_vector_db import DocumentType
                
                # Search by data type
                type_results = vector_db.search_documents(
                    query=query,
                    namespace=data_type,
                    limit=per_type_limit
                )
                
                results.extend(type_results)
            
            # Sort results by relevance and limit
            results = sorted(results, key=lambda x: x.get("score", 1.0))[:limit]
            return results
        else:
            # Search all document types
            return vector_db.search_documents(
                query=query,
                limit=limit
            )
    except Exception as e:
        logger.error(f"Error searching data: {str(e)}")
        return []

def get_conversation_tracker():
    """
    Get the conversation tracker instance
    
    Returns:
        ConversationTracker instance or None if not available
    """
    try:
        vector_db = get_central_vector_db()
        if not vector_db:
            return None
        
        return vector_db.get_conversation_tracker()
    except Exception as e:
        logger.error(f"Error getting conversation tracker: {str(e)}")
        return None

def search_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search knowledge items in the central vector DB"""
    return search_relevant_data(query, data_types=["knowledge"], limit=limit)

def find_therapy_resources(topic: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find therapy resources in the central vector DB"""
    return search_relevant_data(topic, data_types=["therapy"], limit=limit)

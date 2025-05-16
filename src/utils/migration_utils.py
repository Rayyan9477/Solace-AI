"""
Migration Utilities for Central Vector Database

This module provides utility functions to migrate existing data from various
sources into the new central vector database.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.utils.vector_db_integration import add_user_data, get_user_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

def migrate_conversations(source_path: Optional[str] = None, user_id: str = "default_user") -> int:
    """
    Migrate conversation data to the central vector database
    
    Args:
        source_path: Path to the conversation data directory (defaults to standard location)
        user_id: User ID for the data
        
    Returns:
        Number of successfully migrated conversations
    """
    try:
        # Use default path if not provided
        if not source_path:
            source_path = Path(__file__).parent.parent / 'data' / 'conversations'
        else:
            source_path = Path(source_path)
        
        # Ensure the path exists
        if not source_path.exists():
            logger.warning(f"Conversation source path does not exist: {source_path}")
            return 0
            
        # Find all conversation files
        conversation_files = list(source_path.glob(f"{user_id}_*.json"))
        
        if not conversation_files:
            logger.info(f"No conversation files found for user {user_id}")
            return 0
            
        # Track successful migrations
        success_count = 0
        
        # Process each conversation file
        for file_path in conversation_files:
            try:
                # Load conversation data
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                
                # Migrate each conversation
                for conv in conversations:
                    try:
                        # Prepare conversation data
                        conv_data = {
                            "user_message": conv.get("user_message", ""),
                            "assistant_response": conv.get("assistant_response", ""),
                            "timestamp": conv.get("timestamp", datetime.now().isoformat()),
                            "user_id": user_id,
                            "source": "migration",
                            "original_file": str(file_path)
                        }
                        
                        # Add emotion data if available
                        if "emotion_data" in conv:
                            conv_data["emotion_data"] = conv["emotion_data"]
                            
                        # Store in vector database
                        doc_id = add_user_data("conversation", conv_data)
                        
                        if doc_id:
                            success_count += 1
                            logger.debug(f"Migrated conversation to vector DB: {doc_id}")
                        else:
                            logger.warning(f"Failed to migrate conversation from {file_path}")
                            
                    except Exception as e:
                        logger.error(f"Error migrating individual conversation: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing conversation file {file_path}: {str(e)}")
        
        logger.info(f"Successfully migrated {success_count} conversations for user {user_id}")
        return success_count
        
    except Exception as e:
        logger.error(f"Error migrating conversations: {str(e)}")
        return 0
        
def migrate_personality_data(source_path: Optional[str] = None, user_id: str = "default_user") -> int:
    """
    Migrate personality assessment data to the central vector database
    
    Args:
        source_path: Path to the personality data directory (defaults to standard location)
        user_id: User ID for the data
        
    Returns:
        Number of successfully migrated personality assessments
    """
    try:
        # Use default path if not provided
        if not source_path:
            source_path = Path(__file__).parent.parent / 'data' / 'personality'
        else:
            source_path = Path(source_path)
        
        # Ensure the path exists
        if not source_path.exists():
            logger.warning(f"Personality data source path does not exist: {source_path}")
            return 0
            
        # Find all personality files
        personality_files = list(source_path.glob(f"{user_id}_*.json"))
        
        if not personality_files:
            logger.info(f"No personality files found for user {user_id}")
            return 0
            
        # Track successful migrations
        success_count = 0
        
        # Process each personality file
        for file_path in personality_files:
            try:
                # Load personality data
                with open(file_path, 'r', encoding='utf-8') as f:
                    personality_data = json.load(f)
                
                # Add metadata
                if not isinstance(personality_data, dict):
                    logger.warning(f"Invalid personality data format in {file_path}")
                    continue
                    
                personality_data["timestamp"] = personality_data.get("timestamp", datetime.now().isoformat())
                personality_data["user_id"] = user_id
                personality_data["source"] = "migration"
                personality_data["original_file"] = str(file_path)
                
                # Store in vector database
                doc_id = add_user_data("personality", personality_data)
                
                if doc_id:
                    success_count += 1
                    logger.debug(f"Migrated personality data to vector DB: {doc_id}")
                else:
                    logger.warning(f"Failed to migrate personality data from {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing personality file {file_path}: {str(e)}")
        
        logger.info(f"Successfully migrated {success_count} personality assessments for user {user_id}")
        return success_count
        
    except Exception as e:
        logger.error(f"Error migrating personality data: {str(e)}")
        return 0
        
def migrate_diagnostic_data(source_path: Optional[str] = None, user_id: str = "default_user") -> int:
    """
    Migrate diagnostic assessment data to the central vector database
    
    Args:
        source_path: Path to the diagnostic data directory (defaults to standard location)
        user_id: User ID for the data
        
    Returns:
        Number of successfully migrated diagnostic assessments
    """
    try:
        # Use default path if not provided
        if not source_path:
            source_path = Path(__file__).parent.parent / 'data' / 'diagnosis'
        else:
            source_path = Path(source_path)
        
        # Ensure the path exists
        if not source_path.exists():
            logger.warning(f"Diagnostic data source path does not exist: {source_path}")
            return 0
            
        # Find all diagnostic files
        diagnostic_files = list(source_path.glob(f"{user_id}_*.json"))
        
        if not diagnostic_files:
            logger.info(f"No diagnostic files found for user {user_id}")
            return 0
            
        # Track successful migrations
        success_count = 0
        
        # Process each diagnostic file
        for file_path in diagnostic_files:
            try:
                # Load diagnostic data
                with open(file_path, 'r', encoding='utf-8') as f:
                    diagnostic_data = json.load(f)
                
                # Add metadata
                if not isinstance(diagnostic_data, dict):
                    logger.warning(f"Invalid diagnostic data format in {file_path}")
                    continue
                    
                diagnostic_data["timestamp"] = diagnostic_data.get("timestamp", datetime.now().isoformat())
                diagnostic_data["user_id"] = user_id
                diagnostic_data["source"] = "migration"
                diagnostic_data["original_file"] = str(file_path)
                
                # Store in vector database
                doc_id = add_user_data("diagnosis", diagnostic_data)
                
                if doc_id:
                    success_count += 1
                    logger.debug(f"Migrated diagnostic data to vector DB: {doc_id}")
                else:
                    logger.warning(f"Failed to migrate diagnostic data from {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing diagnostic file {file_path}: {str(e)}")
        
        logger.info(f"Successfully migrated {success_count} diagnostic assessments for user {user_id}")
        return success_count
        
    except Exception as e:
        logger.error(f"Error migrating diagnostic data: {str(e)}")
        return 0

def migrate_knowledge_base(source_path: Optional[str] = None) -> int:
    """
    Migrate knowledge base data to the central vector database
    
    Args:
        source_path: Path to the knowledge base directory (defaults to standard location)
        
    Returns:
        Number of successfully migrated knowledge items
    """
    try:
        # Use default path if not provided
        if not source_path:
            source_path = Path(__file__).parent.parent / 'knowledge'
        else:
            source_path = Path(source_path)
        
        # Ensure the path exists
        if not source_path.exists():
            logger.warning(f"Knowledge base path does not exist: {source_path}")
            return 0
            
        # Find all knowledge files recursively
        knowledge_files = []
        for ext in ['*.json', '*.txt', '*.md']:
            knowledge_files.extend(list(source_path.glob(f"**/{ext}")))
        
        if not knowledge_files:
            logger.info("No knowledge files found")
            return 0
            
        # Track successful migrations
        success_count = 0
        
        for file_path in knowledge_files:
            try:
                # Determine file type and load appropriately
                if file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        item_data = json.load(f)
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Create structured data for non-JSON files
                    item_data = {
                        "content": content,
                        "title": file_path.stem,
                        "file_type": file_path.suffix.lstrip('.'),
                        "path": str(file_path.relative_to(source_path)),
                        "category": file_path.parent.name
                    }
                
                # Add metadata
                item_data["timestamp"] = datetime.now().isoformat()
                
                # Store in central vector DB
                doc_id = add_user_data("knowledge", item_data)
                
                if doc_id:
                    success_count += 1
                    logger.debug(f"Migrated knowledge item: {file_path.name}")
                else:
                    logger.warning(f"Failed to migrate knowledge item: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error migrating knowledge file {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Migrated {success_count} knowledge items")
        return success_count
        
    except Exception as e:
        logger.error(f"Error migrating knowledge base: {str(e)}")
        return 0

def migrate_therapy_resources(source_path: Optional[str] = None) -> int:
    """
    Migrate therapy resources to the central vector database
    
    Args:
        source_path: Path to the therapy resources directory (defaults to standard location)
        
    Returns:
        Number of successfully migrated therapy resources
    """
    try:
        # Use default path if not provided
        if not source_path:
            source_path = Path(__file__).parent.parent / 'knowledge' / 'therapeutic'
        else:
            source_path = Path(source_path)
        
        # Ensure the path exists
        if not source_path.exists():
            logger.warning(f"Therapy resources path does not exist: {source_path}")
            return 0
            
        # Find all resource files recursively
        resource_files = []
        for ext in ['*.json', '*.txt', '*.md']:
            resource_files.extend(list(source_path.glob(f"**/{ext}")))
        
        if not resource_files:
            logger.info("No therapy resource files found")
            return 0
            
        # Track successful migrations
        success_count = 0
        
        for file_path in resource_files:
            try:
                # Determine file type and load appropriately
                if file_path.suffix == '.json':
                    with open(file_path, 'r') as f:
                        resource_data = json.load(f)
                else:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Create structured data for non-JSON files
                    resource_data = {
                        "content": content,
                        "title": file_path.stem,
                        "file_type": file_path.suffix.lstrip('.'),
                        "resource_type": file_path.parent.name,
                        "condition": _extract_condition_from_path(file_path)
                    }
                
                # Add metadata
                resource_data["timestamp"] = datetime.now().isoformat()
                
                # Store in central vector DB
                doc_id = add_user_data("therapy", resource_data)
                
                if doc_id:
                    success_count += 1
                    logger.debug(f"Migrated therapy resource: {file_path.name}")
                else:
                    logger.warning(f"Failed to migrate therapy resource: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error migrating therapy resource {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Migrated {success_count} therapy resources")
        return success_count
        
    except Exception as e:
        logger.error(f"Error migrating therapy resources: {str(e)}")
        return 0

def _extract_condition_from_path(file_path: Path) -> str:
    """
    Extract mental health condition from file path
    
    Args:
        file_path: Path to the therapy resource file
        
    Returns:
        Extracted condition or "general"
    """
    try:
        # Try to determine condition from parent directory name
        parent_name = file_path.parent.name.lower()
        
        # Common conditions to check for
        conditions = [
            "anxiety", "depression", "bipolar", "ptsd", "trauma", 
            "ocd", "addiction", "adhd", "autism", "grief", 
            "stress", "insomnia", "eating", "social", "schizophrenia"
        ]
        
        for condition in conditions:
            if condition in parent_name:
                return condition
                
        # Try to extract from filename
        filename = file_path.stem.lower()
        for condition in conditions:
            if condition in filename:
                return condition
        
        # Default to general if no match
        return "general"
    
    except Exception:
        return "general"

def migrate_all_user_data(user_id: str = "default_user") -> Dict[str, int]:
    """
    Migrate all user data to the central vector database
    
    Args:
        user_id: User ID for the data
        
    Returns:
        Dictionary with counts of migrated data by type
    """
    results = {
        "conversations": migrate_conversations(user_id=user_id),
        "personality": migrate_personality_data(user_id=user_id),
        "diagnosis": migrate_diagnostic_data(user_id=user_id),
        "knowledge": migrate_knowledge_base(),
        "therapy": migrate_therapy_resources()
    }
    
    total = sum(results.values())
    logger.info(f"Total of {total} items migrated for user {user_id}")
    
    return results

if __name__ == "__main__":
    # When run as a script, migrate all data for the default user
    results = migrate_all_user_data()
    print(f"Migration complete: {results}")

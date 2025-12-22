"""
Central Vector Database Manager

This module provides a unified interface for all vector storage and retrieval operations
across the application. It serves as a central hub for:
- User profiles
- Conversation history
- Knowledge base documents
- Therapy resources
- Diagnostic data
- Personality assessments

All data is stored using vector embeddings for semantic search and retrieval.
"""

import os
import json
import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# from sentence_transformers import SentenceTransformer  # Lazy import when needed
# Make faiss optional to allow running without it installed
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from src.database.vector_store import FaissVectorStore
from src.database.conversation_tracker import ConversationTracker
from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types of documents stored in the vector database"""
    USER_PROFILE = "user_profile"
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    THERAPY_RESOURCE = "therapy_resource"
    DIAGNOSTIC_DATA = "diagnostic_data"
    PERSONALITY_ASSESSMENT = "personality_assessment"
    EMOTION_RECORD = "emotion_record"
    CUSTOM = "custom"

class CentralVectorDB:
    """
    Central Vector Database Manager
    
    Provides a unified interface for all vector storage operations across the application.
    Implements namespacing to separate different types of data while maintaining
    a single efficient vector store backend.
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None, 
                user_id: str = "default_user",
                dimension: int = 1536):
        """
        Initialize the central vector database
        
        Args:
            config: Configuration options
            user_id: User identifier
            dimension: Dimension of embedding vectors
        """
        self.config = config or {}
        self.user_id = user_id
        self.dimension = dimension
        
        # Initialize main vector store
        self.vector_store = FaissVectorStore(dimension=dimension)
        
        # Initialize specialized trackers
        self.conversation_tracker = ConversationTracker(user_id=user_id, dimension=dimension)
        
        # Storage locations
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Namespaced document collections
        self.namespaces = {
            DocumentType.USER_PROFILE.value: {},
            DocumentType.KNOWLEDGE.value: {},
            DocumentType.THERAPY_RESOURCE.value: {},
            DocumentType.DIAGNOSTIC_DATA.value: {},
            DocumentType.PERSONALITY_ASSESSMENT.value: {},
            DocumentType.EMOTION_RECORD.value: {},
            DocumentType.CUSTOM.value: {}
        }
        
        # Initialize connection to vector store
        self._connect()
        
        # Load namespace metadata
        self._load_namespace_metadata()
    
    def _connect(self) -> bool:
        """Connect to the main vector store"""
        try:
            result = self.vector_store.connect()
            if result:
                logger.info(f"Connected to central vector database for user {self.user_id}")
            else:
                logger.warning(f"Failed to connect to central vector database for user {self.user_id}")
            return result
        except Exception as e:
            logger.error(f"Error connecting to central vector database: {str(e)}")
            return False
    
    def _load_namespace_metadata(self) -> None:
        """Load namespace metadata from storage"""
        for namespace in self.namespaces.keys():
            try:
                namespace_dir = self.data_dir / namespace
                namespace_dir.mkdir(parents=True, exist_ok=True)
                metadata_path = namespace_dir / f"{self.user_id}_metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.namespaces[namespace] = json.load(f)
                    logger.debug(f"Loaded metadata for namespace: {namespace}")
            except Exception as e:
                logger.error(f"Error loading metadata for namespace {namespace}: {str(e)}")
    
    def _save_namespace_metadata(self, namespace: str) -> bool:
        """Save namespace metadata to storage"""
        try:
            namespace_dir = self.data_dir / namespace
            namespace_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = namespace_dir / f"{self.user_id}_metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(self.namespaces[namespace], f, indent=2)
            
            logger.debug(f"Saved metadata for namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata for namespace {namespace}: {str(e)}")
            return False
    
    def add_document(self, 
                    document: Dict[str, Any],
                    namespace: Union[str, DocumentType] = DocumentType.CUSTOM,
                    doc_id: Optional[str] = None) -> str:
        """
        Add a document to the vector database
        
        Args:
            document: Document to add
            namespace: Document namespace/type
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        # Extract namespace string if Enum provided
        if isinstance(namespace, DocumentType):
            namespace = namespace.value
        
        # Validate namespace
        if namespace not in self.namespaces:
            logger.warning(f"Unknown namespace: {namespace}, defaulting to custom")
            namespace = DocumentType.CUSTOM.value
        
        # Generate document ID if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Prepare the document
        timestamp = datetime.now().isoformat()
        enriched_doc = {
            **document,
            "doc_id": doc_id,
            "namespace": namespace,
            "user_id": self.user_id,
            "timestamp": timestamp
        }
        
        # Add to vector store
        try:
            self.vector_store.add_documents([enriched_doc])
            
            # Update namespace metadata
            if doc_id not in self.namespaces[namespace]:
                self.namespaces[namespace][doc_id] = {
                    "timestamp": timestamp,
                    "title": document.get("title", "Untitled"),
                    "summary": document.get("summary", "")[:100] if document.get("summary") else "",
                }
                
                # Save updated metadata
                self._save_namespace_metadata(namespace)
            
            logger.info(f"Added document to namespace {namespace}: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document to namespace {namespace}: {str(e)}")
            return ""
    
    def get_document(self, doc_id: str, namespace: Union[str, DocumentType] = None) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID
        
        Args:
            doc_id: Document ID
            namespace: Optional namespace to search in (searches all if not provided)
            
        Returns:
            Document or None if not found
        """
        # Extract namespace string if Enum provided
        if isinstance(namespace, DocumentType):
            namespace = namespace.value
        
        try:
            # If we know the namespace, check metadata first
            if namespace and namespace in self.namespaces:
                if doc_id not in self.namespaces[namespace]:
                    logger.warning(f"Document {doc_id} not found in namespace {namespace}")
                    return None
                
                # Use metadata to construct a search query
                metadata = self.namespaces[namespace][doc_id]
                query = metadata.get("title", "") or metadata.get("summary", "")
                
                # Search for the document
                results = self.vector_store.search(query=query, k=10)
                
                # Find the exact document by ID
                for result in results:
                    if result.get("doc_id") == doc_id:
                        return result
            
            # If namespace not provided or document not found in specific namespace,
            # search all documents
            results = self.vector_store.search(query=doc_id, k=20)
            
            # Find the exact document by ID
            for result in results:
                if result.get("doc_id") == doc_id:
                    return result
            
            logger.warning(f"Document {doc_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def search_documents(self, 
                        query: str,
                        namespace: Union[str, DocumentType] = None,
                        limit: int = 5,
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents by semantic similarity
        
        Args:
            query: Search query
            namespace: Optional namespace to search in (searches all if not provided)
            limit: Maximum number of results to return
            filters: Additional filters to apply
            
        Returns:
            List of matching documents
        """
        # Extract namespace string if Enum provided
        if isinstance(namespace, DocumentType):
            namespace = namespace.value
            
        try:
            # Search vector store
            results = self.vector_store.search(query=query, k=limit * 3)  # Get more for filtering
            
            # Filter results
            filtered_results = []
            
            for result in results:
                # Filter by namespace if provided
                if namespace and result.get("namespace") != namespace:
                    continue
                
                # Filter by user_id
                if result.get("user_id") != self.user_id:
                    continue
                
                # Apply custom filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if key in result:
                            # Handle list values
                            if isinstance(value, list):
                                if result[key] not in value:
                                    skip = True
                                    break
                            # Handle date range filters
                            elif key.endswith("_from") and key[:-5] in result:
                                actual_key = key[:-5]
                                if result[actual_key] < value:
                                    skip = True
                                    break
                            elif key.endswith("_to") and key[:-3] in result:
                                actual_key = key[:-3]
                                if result[actual_key] > value:
                                    skip = True
                                    break
                            # Simple equality check
                            elif result[key] != value:
                                skip = True
                                break
                    
                    if skip:
                        continue
                
                filtered_results.append(result)
                
                # Break if we have enough results
                if len(filtered_results) >= limit:
                    break
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def update_document(self, doc_id: str, updates: Dict[str, Any], namespace: Union[str, DocumentType] = None) -> bool:
        """
        Update an existing document
        
        Note: This is a "soft" update that creates a new document version rather than modifying
        the existing vector. For full update, delete and re-add the document.
        
        Args:
            doc_id: Document ID
            updates: Updates to apply
            namespace: Document namespace (searches all if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        # Extract namespace string if Enum provided
        if isinstance(namespace, DocumentType):
            namespace = namespace.value
            
        try:
            # Get current document
            current_doc = self.get_document(doc_id, namespace)
            if not current_doc:
                logger.warning(f"Document {doc_id} not found for update")
                return False
            
            # Determine actual namespace from document if not provided
            actual_namespace = namespace or current_doc.get("namespace", DocumentType.CUSTOM.value)
            
            # Create updated document
            updated_doc = {**current_doc, **updates, "updated_at": datetime.now().isoformat()}
            
            # Add updated document
            new_id = self.add_document(updated_doc, actual_namespace, doc_id)
            
            # Update was successful if we got back the same ID
            success = new_id == doc_id
            
            if success:
                logger.info(f"Updated document {doc_id} in namespace {actual_namespace}")
            else:
                logger.warning(f"Failed to update document {doc_id} in namespace {actual_namespace}")
            
            return success
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str, namespace: Union[str, DocumentType] = None) -> bool:
        """
        Delete a document from the vector database
        
        Args:
            doc_id: Document ID
            namespace: Document namespace (searches all if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        # Extract namespace string if Enum provided
        if isinstance(namespace, DocumentType):
            namespace = namespace.value
            
        try:
            # Determine the namespace if not provided
            if not namespace:
                doc = self.get_document(doc_id)
                if not doc:
                    logger.warning(f"Document {doc_id} not found for deletion")
                    return False
                namespace = doc.get("namespace", DocumentType.CUSTOM.value)
            
            # Remove from namespace metadata
            if namespace in self.namespaces and doc_id in self.namespaces[namespace]:
                del self.namespaces[namespace][doc_id]
                self._save_namespace_metadata(namespace)
            
            # Note: FAISS doesn't support true deletion without rebuilding the index
            # For a production system, consider implementing proper deletion
            # by rebuilding the index periodically or using a database with deletion support
            logger.info(f"Removed document {doc_id} from namespace {namespace} metadata")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def add_user_profile(self, profile_data: Dict[str, Any]) -> str:
        """
        Add or update user profile information
        
        Args:
            profile_data: User profile data
            
        Returns:
            Profile document ID
        """
        # Prepare profile document
        doc = {
            "content": json.dumps(profile_data),
            "title": f"User Profile - {self.user_id}",
            "summary": f"Profile data for user {self.user_id}",
            "profile_data": profile_data
        }
        
        # Use a consistent ID for user profiles
        profile_id = f"profile_{self.user_id}"
        
        return self.add_document(doc, DocumentType.USER_PROFILE, profile_id)
    
    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get user profile information
        
        Returns:
            User profile data
        """
        profile_id = f"profile_{self.user_id}"
        doc = self.get_document(profile_id, DocumentType.USER_PROFILE)
        
        if doc and "profile_data" in doc:
            return doc["profile_data"]
        
        # Return empty profile if not found
        return {}
    
    def add_diagnostic_data(self, diagnosis: Dict[str, Any], assessment_id: Optional[str] = None) -> str:
        """
        Add diagnostic assessment data
        
        Args:
            diagnosis: Diagnostic data
            assessment_id: Optional assessment ID
            
        Returns:
            Diagnostic document ID
        """
        # Prepare diagnostic document
        timestamp = datetime.now().isoformat()
        
        # Extract summary if available
        summary = ""
        if "overall_status" in diagnosis:
            summary = f"Overall: {diagnosis['overall_status']}"
        elif "summary" in diagnosis:
            summary = diagnosis["summary"]
        
        doc = {
            "content": json.dumps(diagnosis),
            "title": f"Diagnostic Assessment - {timestamp}",
            "summary": summary,
            "diagnosis_data": diagnosis,
            "assessment_date": timestamp
        }
        
        # Use provided ID or generate one
        doc_id = assessment_id or f"diagnosis_{int(time.time())}"
        
        return self.add_document(doc, DocumentType.DIAGNOSTIC_DATA, doc_id)
    
    def get_latest_diagnosis(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest diagnostic assessment
        
        Returns:
            Latest diagnostic data or None if not found
        """
        results = self.search_documents(
            query="diagnostic assessment",
            namespace=DocumentType.DIAGNOSTIC_DATA,
            limit=5
        )
        
        if not results:
            return None
        
        # Sort by assessment date
        sorted_results = sorted(
            results,
            key=lambda x: x.get("assessment_date", ""),
            reverse=True
        )
        
        # Return the latest
        if sorted_results and "diagnosis_data" in sorted_results[0]:
            return sorted_results[0]["diagnosis_data"]
        
        return None
    
    def add_personality_assessment(self, personality_data: Dict[str, Any], assessment_id: Optional[str] = None) -> str:
        """
        Add personality assessment data
        
        Args:
            personality_data: Personality assessment data
            assessment_id: Optional assessment ID
            
        Returns:
            Personality assessment document ID
        """
        # Prepare personality document
        timestamp = datetime.now().isoformat()
        
        # Extract summary
        summary = ""
        if "type" in personality_data:
            summary = f"Type: {personality_data['type']}"
        elif "traits" in personality_data:
            traits = personality_data["traits"]
            summary = ", ".join([f"{k}: {v['category'] if isinstance(v, dict) else v}" 
                                for k, v in traits.items()][:3])
        
        doc = {
            "content": json.dumps(personality_data),
            "title": f"Personality Assessment - {timestamp}",
            "summary": summary,
            "personality_data": personality_data,
            "assessment_date": timestamp
        }
        
        # Use provided ID or generate one
        doc_id = assessment_id or f"personality_{int(time.time())}"
        
        return self.add_document(doc, DocumentType.PERSONALITY_ASSESSMENT, doc_id)
    
    def get_latest_personality(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest personality assessment
        
        Returns:
            Latest personality data or None if not found
        """
        results = self.search_documents(
            query="personality assessment",
            namespace=DocumentType.PERSONALITY_ASSESSMENT,
            limit=5
        )
        
        if not results:
            return None
        
        # Sort by assessment date
        sorted_results = sorted(
            results,
            key=lambda x: x.get("assessment_date", ""),
            reverse=True
        )
        
        # Return the latest
        if sorted_results and "personality_data" in sorted_results[0]:
            return sorted_results[0]["personality_data"]
        
        return None
    
    def add_knowledge_item(self, knowledge_item: Dict[str, Any], item_id: Optional[str] = None) -> str:
        """
        Add a knowledge base item
        
        Args:
            knowledge_item: Knowledge item data
            item_id: Optional item ID
            
        Returns:
            Knowledge item document ID
        """
        # Extract content and metadata
        content = knowledge_item.get("content", "")
        title = knowledge_item.get("title", "Untitled Knowledge Item")
        category = knowledge_item.get("category", "general")
        tags = knowledge_item.get("tags", [])
        
        doc = {
            "content": content,
            "title": title,
            "summary": content[:100] + "..." if len(content) > 100 else content,
            "category": category,
            "tags": tags,
            "knowledge_item": knowledge_item
        }
        
        # Use provided ID or generate one
        doc_id = item_id or f"knowledge_{category}_{int(time.time())}"
        
        return self.add_document(doc, DocumentType.KNOWLEDGE, doc_id)
    
    def search_knowledge(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            category: Optional category to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching knowledge items
        """
        # Prepare filters
        filters = {}
        if category:
            filters["category"] = category
        
        # Search with filters
        results = self.search_documents(
            query=query,
            namespace=DocumentType.KNOWLEDGE,
            limit=limit,
            filters=filters
        )
        
        # Extract knowledge items
        knowledge_items = []
        for result in results:
            if "knowledge_item" in result:
                knowledge_items.append(result["knowledge_item"])
            else:
                # If no specific knowledge_item field, return the whole document
                knowledge_items.append(result)
        
        return knowledge_items
    
    def add_therapy_resource(self, resource: Dict[str, Any], resource_id: Optional[str] = None) -> str:
        """
        Add a therapy resource
        
        Args:
            resource: Therapy resource data
            resource_id: Optional resource ID
            
        Returns:
            Therapy resource document ID
        """
        # Extract content and metadata
        content = resource.get("content", "")
        title = resource.get("title", "Untitled Therapy Resource")
        resource_type = resource.get("type", "general")
        target_conditions = resource.get("target_conditions", [])
        
        doc = {
            "content": content,
            "title": title,
            "summary": content[:100] + "..." if len(content) > 100 else content,
            "resource_type": resource_type,
            "target_conditions": target_conditions,
            "therapy_resource": resource
        }
        
        # Use provided ID or generate one
        doc_id = resource_id or f"therapy_{resource_type}_{int(time.time())}"
        
        return self.add_document(doc, DocumentType.THERAPY_RESOURCE, doc_id)
    
    def find_therapy_resources(self, 
                            condition: str, 
                            resource_type: Optional[str] = None, 
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find therapy resources for a specific condition
        
        Args:
            condition: Mental health condition
            resource_type: Optional resource type
            limit: Maximum number of results to return
            
        Returns:
            List of matching therapy resources
        """
        # Prepare query that combines the condition name
        query = f"therapy resources for {condition}"
        
        # Prepare filters
        filters = {}
        if resource_type:
            filters["resource_type"] = resource_type
        
        # Search with filters
        results = self.search_documents(
            query=query,
            namespace=DocumentType.THERAPY_RESOURCE,
            limit=limit,
            filters=filters
        )
        
        # Extract therapy resources
        resources = []
        for result in results:
            if "therapy_resource" in result:
                resources.append(result["therapy_resource"])
            else:
                # If no specific therapy_resource field, return the whole document
                resources.append(result)
        
        return resources
    
    def get_conversation_tracker(self) -> ConversationTracker:
        """
        Get the conversation tracker instance

        Returns:
            ConversationTracker instance
        """
        return self.conversation_tracker

    def close(self) -> bool:
        """
        Close database connections and release resources.

        This method ensures proper cleanup of all database resources including:
        - Vector store connections
        - Conversation tracker resources
        - Cached namespace metadata

        Returns:
            True if closed successfully, False otherwise
        """
        try:
            # Close vector store if it has a close method
            if hasattr(self.vector_store, 'close'):
                self.vector_store.close()
            elif hasattr(self.vector_store, 'disconnect'):
                self.vector_store.disconnect()

            # Close conversation tracker if it has a close method
            if hasattr(self.conversation_tracker, 'close'):
                self.conversation_tracker.close()
            elif hasattr(self.conversation_tracker, 'disconnect'):
                self.conversation_tracker.disconnect()

            # Clear namespace cache
            self.namespaces.clear()

            logger.info(f"CentralVectorDB closed for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"Error closing CentralVectorDB: {str(e)}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are closed."""
        self.close()
        return False  # Don't suppress exceptions

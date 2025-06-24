"""
Central Vector Database Module

This module integrates the CentralVectorDB into the module system.
Provides a unified interface for all vector storage operations.
"""

from typing import Dict, Any, List, Optional, Union
import json
import logging

from src.components.base_module import Module
from src.database.central_vector_db import CentralVectorDB, DocumentType

class CentralVectorDBModule(Module):
    """
    Central Vector Database Module for the Contextual-Chatbot.
    
    Provides all vector storage and retrieval capabilities as services:
    - Store and retrieve user profiles
    - Store and retrieve diagnostic assessments
    - Store and retrieve personality assessments
    - Store and search knowledge base items
    - Store and search therapy resources
    - Track and analyze conversation history
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None):
        """Initialize the module"""
        super().__init__(module_id, config)
        self.vector_db = None
        self.user_id = config.get("user_id", "default_user") if config else "default_user"
    
    async def initialize(self) -> bool:
        """Initialize the module"""
        await super().initialize()
        
        try:
            # Initialize central vector database
            self.vector_db = CentralVectorDB(
                config=self.config,
                user_id=self.user_id
            )
            
            self.logger.info(f"Initialized central vector database for user {self.user_id}")
            
            # Register services
            self._register_services()
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing central vector database: {str(e)}")
            return False
    
    def _register_services(self):
        """Register services provided by this module"""
        # User profile services
        self.expose_service("add_user_profile", self.add_user_profile)
        self.expose_service("get_user_profile", self.get_user_profile)
        
        # Diagnostic services
        self.expose_service("add_diagnostic_data", self.add_diagnostic_data)
        self.expose_service("get_latest_diagnosis", self.get_latest_diagnosis)
        
        # Personality services
        self.expose_service("add_personality_assessment", self.add_personality_assessment)
        self.expose_service("get_latest_personality", self.get_latest_personality)
        
        # Knowledge services
        self.expose_service("add_knowledge_item", self.add_knowledge_item)
        self.expose_service("search_knowledge", self.search_knowledge)
        
        # Therapy resource services
        self.expose_service("add_therapy_resource", self.add_therapy_resource)
        self.expose_service("find_therapy_resources", self.find_therapy_resources)
        
        # Generic document services
        self.expose_service("add_document", self.add_document)
        self.expose_service("get_document", self.get_document)
        self.expose_service("search_documents", self.search_documents)
        self.expose_service("update_document", self.update_document)
        self.expose_service("delete_document", self.delete_document)
        
        # Conversation services
        self.expose_service("get_conversation_tracker", self.get_conversation_tracker)
    
    # User profile services
    
    async def add_user_profile(self, profile_data: Dict[str, Any]) -> str:
        """Add or update user profile"""
        try:
            profile_id = self.vector_db.add_user_profile(profile_data)
            self.logger.info(f"Added/updated user profile: {profile_id}")
            return profile_id
        except Exception as e:
            self.logger.error(f"Error adding user profile: {str(e)}")
            return ""
    
    async def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        try:
            profile = self.vector_db.get_user_profile()
            return profile
        except Exception as e:
            self.logger.error(f"Error getting user profile: {str(e)}")
            return {}
    
    # Diagnostic services
    
    async def add_diagnostic_data(self, diagnosis: Dict[str, Any], assessment_id: Optional[str] = None) -> str:
        """Add diagnostic assessment data"""
        try:
            doc_id = self.vector_db.add_diagnostic_data(diagnosis, assessment_id)
            self.logger.info(f"Added diagnostic data: {doc_id}")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error adding diagnostic data: {str(e)}")
            return ""
    
    async def get_latest_diagnosis(self) -> Optional[Dict[str, Any]]:
        """Get latest diagnostic assessment"""
        try:
            diagnosis = self.vector_db.get_latest_diagnosis()
            return diagnosis
        except Exception as e:
            self.logger.error(f"Error getting latest diagnosis: {str(e)}")
            return None
    
    # Personality services
    
    async def add_personality_assessment(self, personality_data: Dict[str, Any], assessment_id: Optional[str] = None) -> str:
        """Add personality assessment data"""
        try:
            doc_id = self.vector_db.add_personality_assessment(personality_data, assessment_id)
            self.logger.info(f"Added personality assessment: {doc_id}")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error adding personality assessment: {str(e)}")
            return ""
    
    async def get_latest_personality(self) -> Optional[Dict[str, Any]]:
        """Get latest personality assessment"""
        try:
            personality = self.vector_db.get_latest_personality()
            return personality
        except Exception as e:
            self.logger.error(f"Error getting latest personality: {str(e)}")
            return None
    
    # Knowledge services
    
    async def add_knowledge_item(self, knowledge_item: Dict[str, Any], item_id: Optional[str] = None) -> str:
        """Add knowledge base item"""
        try:
            doc_id = self.vector_db.add_knowledge_item(knowledge_item, item_id)
            self.logger.info(f"Added knowledge item: {doc_id}")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error adding knowledge item: {str(e)}")
            return ""
    
    async def search_knowledge(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        try:
            results = self.vector_db.search_knowledge(query, category, limit)
            self.logger.debug(f"Found {len(results)} knowledge items for query: {query}")
            return results
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {str(e)}")
            return []
    
    # Therapy resource services
    
    async def add_therapy_resource(self, resource: Dict[str, Any], resource_id: Optional[str] = None) -> str:
        """Add therapy resource"""
        try:
            doc_id = self.vector_db.add_therapy_resource(resource, resource_id)
            self.logger.info(f"Added therapy resource: {doc_id}")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error adding therapy resource: {str(e)}")
            return ""
    
    async def find_therapy_resources(self, condition: str, resource_type: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Find therapy resources for condition"""
        try:
            results = self.vector_db.find_therapy_resources(condition, resource_type, limit)
            self.logger.debug(f"Found {len(results)} therapy resources for condition: {condition}")
            return results
        except Exception as e:
            self.logger.error(f"Error finding therapy resources: {str(e)}")
            return []
    
    # Generic document services
    
    async def add_document(self, document: Dict[str, Any], namespace: Union[str, DocumentType] = DocumentType.CUSTOM, doc_id: Optional[str] = None) -> str:
        """Add document to vector database"""
        try:
            return self.vector_db.add_document(document, namespace, doc_id)
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            return ""
    
    async def get_document(self, doc_id: str, namespace: Union[str, DocumentType] = None) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            return self.vector_db.get_document(doc_id, namespace)
        except Exception as e:
            self.logger.error(f"Error getting document: {str(e)}")
            return None
    
    async def search_documents(self, query: str, namespace: Union[str, DocumentType] = None, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents"""
        try:
            return self.vector_db.search_documents(query, namespace, limit, filters)
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []
    
    async def update_document(self, doc_id: str, updates: Dict[str, Any], namespace: Union[str, DocumentType] = None) -> bool:
        """Update document"""
        try:
            return self.vector_db.update_document(doc_id, updates, namespace)
        except Exception as e:
            self.logger.error(f"Error updating document: {str(e)}")
            return False
    
    async def delete_document(self, doc_id: str, namespace: Union[str, DocumentType] = None) -> bool:
        """Delete document"""
        try:
            return self.vector_db.delete_document(doc_id, namespace)
        except Exception as e:
            self.logger.error(f"Error deleting document: {str(e)}")
            return False
    
    # Conversation services
    
    async def get_conversation_tracker(self) -> Any:
        """Get conversation tracker instance"""
        try:
            return self.vector_db.get_conversation_tracker()
        except Exception as e:
            self.logger.error(f"Error getting conversation tracker: {str(e)}")
            return None

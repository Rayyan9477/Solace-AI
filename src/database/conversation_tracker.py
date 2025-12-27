"""
Conversation Tracking Database
Stores conversation history with timestamps, user inputs, responses, and emotion data in Faiss.
Provides methods for retrieval and analysis of past conversations.
"""

import os
import json
import logging
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import uuid
# Sentence embedding model can be lazily imported in future if needed
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

# tqdm not required here; remove to avoid optional dependency

from src.database.vector_store import FaissVectorStore
from src.config.settings import AppConfig

logger = logging.getLogger(__name__)

# Reused error message
_ERR_CONNECT_FAIL = "Failed to connect to vector store"

class ConversationTracker:
    """
    Specialized database for tracking and analyzing conversation history.
    Uses Faiss for efficient vector storage and retrieval of conversations.
    
    Features:
    - Store conversation history with timestamps, user inputs, responses, and emotion data
    - Search past conversations by semantic similarity
    - Analyze emotion patterns over time
    - Filter conversations by date, emotion, topic, and more
    - Export conversation history for external analysis
    """
    
    def __init__(self, 
                user_id: str = "default_user", 
                dimension: int = 1536, 
                retention_days: int = 180,
                embedder_model: str = None):
        """
        Initialize the conversation tracker
        
        Args:
            user_id: Unique identifier for the user
            dimension: Dimension of embedding vectors
            retention_days: Number of days to retain conversation history
            embedder_model: Name of the sentence transformer model to use for embeddings
        """
        self.user_id = user_id
        self.dimension = dimension
        self.retention_days = retention_days
        
        # Get the embedder model from config or use default
        if embedder_model is None:
            if hasattr(AppConfig, 'EMBEDDING_CONFIG') and 'model_name' in AppConfig.EMBEDDING_CONFIG:
                self.embedder_model = AppConfig.EMBEDDING_CONFIG['model_name']
            else:
                self.embedder_model = 'all-MiniLM-L6-v2'
        else:
            self.embedder_model = embedder_model
        
        # Initialize FAISS store
        self.vector_store = FaissVectorStore(dimension=dimension)

        # Thread safety lock for metadata operations
        self._metadata_lock = threading.RLock()

        # Internal state
        self.is_connected = False
        self.conversation_metadata = {}  # Stores metadata for quick access without vector search
        self.session_start_time = datetime.now()
        
        # Data storage paths
        self.data_dir = Path(__file__).parent.parent / 'data' / 'conversations'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_data_dir = self.data_dir / user_id
        self.user_data_dir.mkdir(exist_ok=True)
        self.metadata_path = self.user_data_dir / "metadata.json"
        
        # Connect to the vector store
        self._connect()
        
        # Load metadata if it exists
        self._load_metadata()
    
    def _connect(self) -> bool:
        """Connect to the vector store"""
        try:
            result = self.vector_store.connect()
            self.is_connected = result
            if result:
                logger.info(f"Connected to conversation tracker for user {self.user_id}")
            else:
                logger.warning(f"Failed to connect to conversation tracker for user {self.user_id}")
            return result
        except Exception as e:
            logger.error(f"Error connecting to conversation tracker: {str(e)}")
            return False
    
    def _load_metadata(self) -> None:
        """Load conversation metadata from disk (thread-safe)"""
        with self._metadata_lock:
            try:
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r') as f:
                        self.conversation_metadata = json.load(f)
                    logger.info(f"Loaded conversation metadata for user {self.user_id}")
                else:
                    logger.info(f"No existing metadata found for user {self.user_id}")
                    self.conversation_metadata = {
                        "conversations": {},
                        "statistics": {
                            "total_conversations": 0,
                            "total_messages": 0,
                            "emotion_counts": {},
                            "first_conversation": None,
                            "last_conversation": None
                        }
                    }
            except Exception as e:
                logger.error(f"Error loading conversation metadata: {str(e)}")
                self.conversation_metadata = {
                    "conversations": {},
                    "statistics": {
                        "total_conversations": 0,
                        "total_messages": 0,
                        "emotion_counts": {},
                        "first_conversation": None,
                        "last_conversation": None
                    }
                }
    
    def _save_metadata(self) -> bool:
        """Save conversation metadata to disk (thread-safe)"""
        with self._metadata_lock:
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.conversation_metadata, f, indent=2)
                logger.info(f"Saved conversation metadata for user {self.user_id}")
                return True
            except Exception as e:
                logger.error(f"Error saving conversation metadata: {str(e)}")
                return False
    
    def add_conversation(self, 
                        user_message: str, 
                        assistant_response: str, 
                        emotion_data: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a conversation turn to the tracker
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            emotion_data: Emotion analysis data if available
            metadata: Additional metadata about the conversation
            
        Returns:
            Conversation ID if successful, empty string otherwise
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return ""
        
        try:
            # Create conversation document
            timestamp = datetime.now().isoformat()
            conversation_id = str(uuid.uuid4())
            
            # Extract primary emotion if available
            primary_emotion = None
            if emotion_data and "primary_emotion" in emotion_data:
                primary_emotion = emotion_data["primary_emotion"]
            
            # Prepare metadata
            meta = metadata or {}
            if emotion_data:
                meta["emotion"] = emotion_data
            
            # Add session info
            meta["session_id"] = str(self.session_start_time.timestamp())
            
            # Prepare conversation document
            document = {
                "content": f"User: {user_message}\nAssistant: {assistant_response}",
                "user_id": self.user_id,
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": timestamp,
                "primary_emotion": primary_emotion,
                "type": "conversation",
                "metadata": meta
            }
            
            # Add to vector store
            self.vector_store.add_documents([document])
            
            # Update metadata
            self._update_metadata(conversation_id, document)
            
            return conversation_id
        except Exception as e:
            logger.error(f"Error adding conversation to tracker: {str(e)}") 
            return "" 
    
    def _update_metadata(self, conversation_id: str, document: Dict[str, Any]) -> None:
        """Update metadata with new conversation information (thread-safe)"""
        with self._metadata_lock:
            try:
                # Store conversation metadata for quick access
                self.conversation_metadata["conversations"][conversation_id] = {
                    "timestamp": document["timestamp"],
                    "primary_emotion": document.get("primary_emotion"),
                    "user_message": document["user_message"][:100] + "..." if len(document["user_message"]) > 100 else document["user_message"],
                    "assistant_response": document["assistant_response"][:100] + "..." if len(document["assistant_response"]) > 100 else document["assistant_response"]
                }

                # Update statistics
                stats = self.conversation_metadata["statistics"]
                stats["total_conversations"] += 1
                stats["total_messages"] += 1

                # Update emotion counts
                if document.get("primary_emotion"):
                    emotion = document["primary_emotion"]
                    stats["emotion_counts"][emotion] = stats["emotion_counts"].get(emotion, 0) + 1

                # Update first/last conversation timestamps
                if stats["first_conversation"] is None:
                    stats["first_conversation"] = document["timestamp"]
                stats["last_conversation"] = document["timestamp"]

                # Save updated metadata (lock is reentrant, so this is safe)
                self._save_metadata()
            except Exception as e:
                logger.error(f"Error updating metadata: {str(e)}")
    
    def search_conversations(self, 
                            query: str, 
                            limit: int = 5, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            emotion: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant past conversations with filtering
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            start_date: Filter conversations on or after this date (ISO format)
            end_date: Filter conversations on or before this date (ISO format)
            emotion: Filter by primary emotion
            
        Returns:
            List of relevant conversations
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return []
        
        try:
            # Search vector store
            results = self.vector_store.search(
                query=query, 
                k=limit * 3,  # Get more results for filtering
                use_cache=True
            )
            
            # Filter results
            filtered_results = []
            for result in results:
                # Apply structural, date, and emotion filters
                if not self._is_valid_conversation_result(result):
                    continue
                if not self._within_date_range(result.get("timestamp"), start_date, end_date):
                    continue
                if emotion and not self._matches_emotion(result, emotion):
                    continue
                
                filtered_results.append(result)
                
                # Break if we have enough results
                if len(filtered_results) >= limit:
                    break
            
            return filtered_results
        except Exception as e:
            logger.error(f"Error searching conversations: {str(e)}") 
            return [] 

    def _is_valid_conversation_result(self, result: Dict[str, Any]) -> bool:
        """Check basic constraints for a conversation search result"""
        try:
            if result.get('type') != 'conversation':
                return False
            if result.get('user_id') != self.user_id:
                return False
            return True
        except (KeyError, TypeError, AttributeError):
            return False

    def _within_date_range(self, timestamp_iso: Optional[str], start_date: Optional[str], end_date: Optional[str]) -> bool:
        """Check if an ISO timestamp is within the provided date range (inclusive). If no range, accept."""
        try:
            if not (start_date or end_date):
                return True
            if not timestamp_iso:
                return False
            ts = datetime.fromisoformat(timestamp_iso)
            if start_date and ts < datetime.fromisoformat(start_date):
                return False
            if end_date and ts > datetime.fromisoformat(end_date):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def _matches_emotion(self, result: Dict[str, Any], emotion: str) -> bool:
        """Check if a result matches the requested primary emotion."""
        try:
            if not emotion:
                return True
            return result.get("primary_emotion") == emotion
        except (KeyError, TypeError, AttributeError):
            return False
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific conversation by ID
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            Conversation dictionary or None if not found
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return None
        
        try:
            # Check if we have this conversation in metadata
            if conversation_id not in self.conversation_metadata["conversations"]:
                logger.warning(f"Conversation ID {conversation_id} not found in metadata")
                return None
            
            # Use the first few words of the stored message as a search query
            meta = self.conversation_metadata["conversations"][conversation_id]
            query = meta["user_message"][:50]  # Use the first 50 chars of the message
            
            # Search for this conversation
            results = self.vector_store.search(query=query, k=10)
            
            # Find the exact conversation by ID
            for result in results:
                if result.get("conversation_id") == conversation_id:
                    return result
            
            logger.warning(f"Conversation ID {conversation_id} not found in vector store")
            return None
        except Exception as e:
            logger.error(f"Error retrieving conversation: {str(e)}") 
            return None 
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent conversations
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of recent conversations
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return []
        
        try:
            # Get conversation IDs sorted by timestamp
            conversations = self.conversation_metadata["conversations"]
            sorted_ids = sorted(
                conversations.keys(),
                key=lambda x: conversations[x]["timestamp"],
                reverse=True
            )[:limit]
            
            # Retrieve full conversations
            results = []
            for conv_id in sorted_ids:
                conversation = self.get_conversation_by_id(conv_id)
                if conversation:
                    results.append(conversation)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving recent conversations: {str(e)}") 
            return [] 
    
    def get_conversations_by_date(self, 
                                date: str, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversations from a specific date
        
        Args:
            date: Date in ISO format (YYYY-MM-DD)
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversations from the specified date
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return []
        
        try:
            # Parse date
            target_date = datetime.fromisoformat(date).date()
            
            # Check all conversations
            matching = []
            for conv_id, meta in self.conversation_metadata["conversations"].items():
                try:
                    # Convert timestamp to date
                    conv_date = datetime.fromisoformat(meta["timestamp"]).date()
                    
                    # Check if date matches
                    if conv_date == target_date:
                        # Get full conversation
                        conversation = self.get_conversation_by_id(conv_id)
                        if conversation:
                            matching.append(conversation)
                            
                        # Break if we have enough
                        if len(matching) >= limit:
                            break
                except (KeyError, TypeError, ValueError):
                    continue

            return matching
        except Exception as e:
            logger.error(f"Error retrieving conversations by date: {str(e)}") 
            return [] 
    
    def get_conversations_by_emotion(self, 
                                    emotion: str, 
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversations with a specific primary emotion
        
        Args:
            emotion: The primary emotion to filter by
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversations with the specified emotion
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return []
        
        try:
            # Check all conversations
            matching = []
            for conv_id, meta in self.conversation_metadata["conversations"].items():
                try:
                    # Check if emotion matches
                    if meta.get("primary_emotion") == emotion:
                        # Get full conversation
                        conversation = self.get_conversation_by_id(conv_id)
                        if conversation:
                            matching.append(conversation)

                        # Break if we have enough
                        if len(matching) >= limit:
                            break
                except (KeyError, TypeError, AttributeError):
                    continue
            
            return matching
        except Exception as e:
            logger.error(f"Error retrieving conversations by emotion: {str(e)}") 
            return [] 
    
    def get_emotion_distribution(self, 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> Dict[str, int]:
        """
        Get distribution of emotions in conversations
        
        Args:
            start_date: Start date for the analysis (ISO format)
            end_date: End date for the analysis (ISO format)
            
        Returns:
            Dictionary mapping emotions to counts
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return {}
        
        try:
            # If no date range specified, return overall statistics
            if not start_date and not end_date:
                return self.conversation_metadata["statistics"]["emotion_counts"]

            # Count emotions in the specified date range
            emotion_counts: Dict[str, int] = {}
            for meta in self.conversation_metadata["conversations"].values():
                try:
                    if not self._within_date_range(meta.get("timestamp"), start_date, end_date):
                        continue
                    emotion = meta.get("primary_emotion")
                    if emotion:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                except (KeyError, TypeError, AttributeError):
                    continue

            return emotion_counts
        except Exception as e:
            logger.error(f"Error getting emotion distribution: {str(e)}")
            return {}
    
    def export_conversations(self, 
                            output_path: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> str:
        """
        Export conversations to a JSON file
        
        Args:
            output_path: Path to save the exported file (optional)
            start_date: Start date for export (ISO format)
            end_date: End date for export (ISO format)
            
        Returns:
            Path to the exported file
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return ""
        
        try:
            # Generate default output path if not provided
            if not output_path:
                output_path = self._default_export_path()

            # Prepare export data
            export_data = {
                "user_id": self.user_id,
                "export_timestamp": datetime.now().isoformat(),
                "date_range": {"start": start_date, "end": end_date},
                "conversations": [],
            }

            # Get all conversations within date range
            for conv_id, meta in self.conversation_metadata["conversations"].items():
                try:
                    if not self._within_date_range(meta.get("timestamp"), start_date, end_date):
                        continue
                    conversation = self.get_conversation_by_id(conv_id)
                    if conversation:
                        export_data["conversations"].append(conversation)
                except (KeyError, TypeError, ValueError):
                    continue

            # Save to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(export_data['conversations'])} conversations to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error exporting conversations: {str(e)}")
            return ""

    def _default_export_path(self) -> str:
        """Create a default export file path under the user's conversation folder."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self.user_data_dir / f"export_{ts}.json")
    
    def cleanup_old_conversations(self) -> int:
        """
        Remove conversations older than retention period
        
        Returns:
            Number of conversations removed
        """
        if not self.is_connected and not self._connect():
            logger.error(_ERR_CONNECT_FAIL)
            return 0
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            # cutoff string omitted (unused)
            
            # Find conversations to remove
            to_remove = []
            for conv_id, meta in self.conversation_metadata["conversations"].items():
                try:
                    timestamp = datetime.fromisoformat(meta["timestamp"])
                    if timestamp < cutoff_date:
                        to_remove.append(conv_id)
                except (KeyError, ValueError, TypeError):
                    continue
            
            # Remove conversations
            removed_count = 0
            for conv_id in to_remove:
                # Remove from metadata
                del self.conversation_metadata["conversations"][conv_id]
                removed_count += 1
            
            # Update statistics
            self.conversation_metadata["statistics"]["total_conversations"] -= removed_count
            
            # Recalculate emotion counts
            emotion_counts = {}
            for meta in self.conversation_metadata["conversations"].values():
                emotion = meta.get("primary_emotion")
                if emotion:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            self.conversation_metadata["statistics"]["emotion_counts"] = emotion_counts
            
            # Save metadata
            self._save_metadata()
            
            # For proper cleanup in vector store, we would need to rebuild the index
            # which is not directly supported by FAISS without reindexing all data
            # In a production environment, consider implementing periodic reindexing
            
            logger.info(f"Removed {removed_count} old conversations")
            return removed_count
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {str(e)}") 
            return 0 
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about conversation history
        
        Returns:
            Dictionary with conversation statistics
        """
        try:
            stats = self.conversation_metadata["statistics"].copy()
            
            # Add date information
            if stats["first_conversation"]:
                stats["first_conversation_date"] = datetime.fromisoformat(
                    stats["first_conversation"]
                ).strftime("%Y-%m-%d")
            if stats["last_conversation"]:
                stats["last_conversation_date"] = datetime.fromisoformat(
                    stats["last_conversation"]
                ).strftime("%Y-%m-%d")
            
            # Add top emotions
            if "emotion_counts" in stats and stats["emotion_counts"]:
                sorted_emotions = sorted(
                    stats["emotion_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                stats["top_emotions"] = sorted_emotions[:3]
            
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}") 
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "emotion_counts": {},
                "error": str(e)
            }

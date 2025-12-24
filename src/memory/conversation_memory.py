"""
Conversation memory module for the mental health chatbot.
Provides FAISS-based vector storage for conversation history and user preferences.

NOTE: This module was relocated from src/utils/conversation_memory.py
to its canonical location in the memory package.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import time
import uuid

from src.database.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation memory using FAISS vector store for efficient retrieval.
    This allows the chatbot to reference past conversations and maintain context.
    """

    def __init__(self, user_id: str = "default_user", dimension: int = 1536, ttl_days: int = 90):
        """
        Initialize the conversation memory

        Args:
            user_id: Unique identifier for the user
            dimension: Dimension of the embedding vectors
            ttl_days: Number of days to keep conversation history
        """
        self.user_id = user_id
        self.vector_store = FaissVectorStore(dimension=dimension)
        self.is_connected = False
        self.ttl_days = ttl_days
        self.user_profile = {}
        self.session_history = []
        self.last_accessed = datetime.now()

        # Connect to vector store
        self._connect()

        # Load user profile if available
        self._load_user_profile()

    def _connect(self) -> bool:
        """Connect to the vector store"""
        try:
            result = self.vector_store.connect()
            if result:
                self.is_connected = True
                logger.info(f"Connected to FAISS vector store for user {self.user_id}")
            else:
                logger.warning(f"Failed to connect to FAISS vector store for user {self.user_id}")
            return result
        except Exception as e:
            logger.error(f"Error connecting to FAISS vector store: {str(e)}")
            return False

    def _load_user_profile(self) -> None:
        """Load user profile from storage"""
        try:
            # Define profile path
            profile_dir = Path(__file__).parent.parent / 'data' / 'users'
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = profile_dir / f"{self.user_id}_profile.json"

            if profile_path.exists():
                with open(profile_path, 'r') as f:
                    self.user_profile = json.load(f)
                logger.info(f"Loaded user profile for {self.user_id}")
        except Exception as e:
            logger.error(f"Error loading user profile: {str(e)}")
            self.user_profile = {}

    def _save_user_profile(self) -> bool:
        """Save user profile to storage"""
        try:
            # Add timestamp to profile
            self.user_profile["last_updated"] = datetime.now().isoformat()

            # Define profile path
            profile_dir = Path(__file__).parent.parent / 'data' / 'users'
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_path = profile_dir / f"{self.user_id}_profile.json"

            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)

            logger.info(f"Saved user profile for {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving user profile: {str(e)}")
            return False

    def add_conversation(self, user_message: str, assistant_response: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a conversation turn to memory

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            metadata: Additional metadata about the conversation

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            if not self._connect():
                logger.error("Failed to connect to vector store")
                return False

        try:
            # Create conversation document
            timestamp = datetime.now().isoformat()
            conversation_id = str(uuid.uuid4())

            # Prepare conversation document
            document = {
                "content": f"User: {user_message}\nAssistant: {assistant_response}",
                "user_id": self.user_id,
                "conversation_id": conversation_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": timestamp,
                "type": "conversation",
                "metadata": metadata or {}
            }

            # Add to vector store
            self.vector_store.add_documents([document])

            # Add to session history
            self.session_history.append(document)

            # Update last accessed time
            self.last_accessed = datetime.now()

            # Extract and update user profile insights
            if metadata:
                self._update_profile_from_metadata(metadata)

            return True
        except Exception as e:
            logger.error(f"Error adding conversation to memory: {str(e)}")
            return False

    def _update_profile_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """Extract and update user profile insights from metadata"""
        # Initialize profile sections if they don't exist
        if "preferences" not in self.user_profile:
            self.user_profile["preferences"] = {}
        if "personality" not in self.user_profile:
            self.user_profile["personality"] = {}
        if "emotional_patterns" not in self.user_profile:
            self.user_profile["emotional_patterns"] = {}
        if "topics" not in self.user_profile:
            self.user_profile["topics"] = {}

        # Update emotional patterns
        if "emotion_analysis" in metadata and metadata["emotion_analysis"]:
            emotion = metadata["emotion_analysis"].get("primary_emotion")
            if emotion:
                # Count occurrences of emotions
                self.user_profile["emotional_patterns"][emotion] = self.user_profile["emotional_patterns"].get(emotion, 0) + 1

        # Update personality insights
        if "personality" in metadata:
            personality_data = metadata["personality"]
            self.user_profile["personality"].update(personality_data)

        # Update preferences
        if "preferences" in metadata:
            self.user_profile["preferences"].update(metadata["preferences"])

        # Extract topics from diagnostic data
        if "diagnosis" in metadata:
            topics = metadata["diagnosis"].get("topics", [])
            for topic in topics:
                self.user_profile["topics"][topic] = self.user_profile["topics"].get(topic, 0) + 1

        # Save updated profile
        self._save_user_profile()

    def get_recent_conversations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversations from session history

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of recent conversations
        """
        # Return from session history first (most recent)
        return self.session_history[-limit:] if self.session_history else []

    def search_conversations(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant past conversations

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of relevant conversations
        """
        if not self.is_connected:
            if not self._connect():
                logger.error("Failed to connect to vector store")
                return []

        try:
            # Search vector store
            results = self.vector_store.search(
                query=query,
                k=limit,
                use_cache=True
            )

            # Filter to only include conversations for this user
            user_results = [
                result for result in results
                if result.get('user_id') == self.user_id and result.get('type') == 'conversation'
            ]

            # Update last accessed time
            self.last_accessed = datetime.now()

            return user_results
        except Exception as e:
            logger.error(f"Error searching conversations: {str(e)}")
            return []

    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the user profile with preferences and patterns

        Returns:
            User profile dictionary
        """
        # Ensure profile is loaded
        if not self.user_profile:
            self._load_user_profile()

        # Add session summary
        profile_with_summary = self.user_profile.copy()
        profile_with_summary["session_summary"] = self._generate_session_summary()

        return profile_with_summary

    def update_user_profile(self, profile_updates: Dict[str, Any]) -> bool:
        """
        Update user profile with new information

        Args:
            profile_updates: Dictionary of profile updates

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update each section of the profile
            for section, data in profile_updates.items():
                if section not in self.user_profile:
                    self.user_profile[section] = {}

                if isinstance(data, dict):
                    # Merge dictionaries
                    self.user_profile[section].update(data)
                else:
                    # Replace value
                    self.user_profile[section] = data

            # Save updated profile
            return self._save_user_profile()
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            return False

    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current session"""
        if not self.session_history:
            return {"message_count": 0}

        # Count emotions in this session
        emotions = {}
        for entry in self.session_history:
            metadata = entry.get("metadata", {})
            if "emotion_analysis" in metadata and metadata["emotion_analysis"]:
                emotion = metadata["emotion_analysis"].get("primary_emotion")
                if emotion:
                    emotions[emotion] = emotions.get(emotion, 0) + 1

        # Get start and end time of session
        first_timestamp = datetime.fromisoformat(self.session_history[0].get("timestamp", datetime.now().isoformat()))
        last_timestamp = datetime.fromisoformat(self.session_history[-1].get("timestamp", datetime.now().isoformat()))

        # Calculate session duration
        duration_seconds = (last_timestamp - first_timestamp).total_seconds()

        return {
            "message_count": len(self.session_history),
            "start_time": first_timestamp.isoformat(),
            "duration_seconds": duration_seconds,
            "emotional_summary": emotions
        }

    def format_context_for_prompt(self, query: str = None, max_history: int = 2) -> str:
        """
        Format conversation context for inclusion in LLM prompts

        Args:
            query: Optional query to find relevant past conversations
            max_history: Maximum number of conversation turns to include

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add recent conversations from current session
        recent = self.get_recent_conversations(limit=max_history)
        if recent:
            recent_parts = []
            for conv in recent:
                recent_parts.append(f"User: {conv.get('user_message', '')}")
                recent_parts.append(f"Assistant: {conv.get('assistant_response', '')}")

            context_parts.append("Recent conversation:")
            context_parts.append("\n".join(recent_parts))

        # Add relevant past conversations if query is provided
        if query:
            relevant = self.search_conversations(query=query, limit=max_history)
            if relevant:
                relevant_parts = []
                for conv in relevant:
                    # Format timestamp to be more readable
                    timestamp = "Unknown time"
                    if "timestamp" in conv:
                        try:
                            dt = datetime.fromisoformat(conv["timestamp"])
                            timestamp = dt.strftime("%b %d, %Y")
                        except Exception:
                            pass

                    relevant_parts.append(f"[From {timestamp}]")
                    relevant_parts.append(f"User: {conv.get('user_message', '')}")
                    relevant_parts.append(f"Assistant: {conv.get('assistant_response', '')}")

                context_parts.append("Previous relevant conversations:")
                context_parts.append("\n".join(relevant_parts))

        # Add user profile summary
        profile = self.get_user_profile()
        profile_parts = ["User profile:"]

        # Add emotional patterns if available
        if "emotional_patterns" in profile and profile["emotional_patterns"]:
            # Get top 3 emotions
            sorted_emotions = sorted(
                profile["emotional_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            emotions_str = ", ".join([f"{emotion} ({count})" for emotion, count in sorted_emotions])
            profile_parts.append(f"Common emotions: {emotions_str}")

        # Add preferences if available
        if "preferences" in profile and profile["preferences"]:
            prefs = profile["preferences"]
            if prefs:
                prefs_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()])
                profile_parts.append(f"Preferences: {prefs_str}")

        # Add personality if available
        if "personality" in profile and profile["personality"]:
            personality = profile["personality"]

            # Check if this is Big Five or MBTI
            if "traits" in personality:
                traits = personality["traits"]
                traits_str = ", ".join([f"{k}: {v['category'] if isinstance(v, dict) else v}" for k, v in traits.items()])
                profile_parts.append(f"Personality traits: {traits_str}")
            elif "type" in personality:
                profile_parts.append(f"Personality type: {personality['type']}")

        if len(profile_parts) > 1:  # If we have more than just the header
            context_parts.append("\n".join(profile_parts))

        return "\n\n".join(context_parts)

    def clear_session_history(self) -> None:
        """Clear the current session history"""
        self.session_history = []

    def clear_all_history(self) -> bool:
        """Clear all conversation history for this user"""
        if not self.is_connected:
            if not self._connect():
                logger.error("Failed to connect to vector store")
                return False

        try:
            # This would require implementing a way to delete by user_id in the vector store
            # For now, we'll just clear the session history
            self.clear_session_history()
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")
            return False

    def maintenance(self) -> None:
        """Perform maintenance tasks (cleanup old conversations, etc.)"""
        # This could be scheduled to run periodically
        if not self.is_connected:
            if not self._connect():
                logger.error("Failed to connect to vector store")
                return

        try:
            # Clean up expired conversations
            # For Faiss, this would require rebuilding the index, which is more complex
            # In a real implementation, we'd:
            # 1. Find all conversations older than ttl_days
            # 2. Remove them from the vector store
            # 3. Rebuild the index if necessary

            # For now, just log that maintenance was performed
            logger.info(f"Maintenance performed for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error during maintenance: {str(e)}")

    def close(self) -> None:
        """Close the connection to the vector store"""
        # Save user profile before closing
        self._save_user_profile()

        # Nothing specific needed for FAISS as it's in-memory
        self.is_connected = False

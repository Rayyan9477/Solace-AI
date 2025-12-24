"""
Context Aware Memory Adapter for the Contextual Chatbot.
Provides advanced memory integration with personality adaptation capabilities.

NOTE: This module was relocated from src/utils/context_aware_memory.py
to its canonical location in the memory package.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import json
import os
from datetime import datetime
from pathlib import Path
import time
import re

from .semantic_memory import SemanticMemoryManager

logger = logging.getLogger(__name__)


class ContextAwareMemoryAdapter:
    """
    Integrates semantic memory with personality adaptation for context-aware responses.
    """

    def __init__(
        self,
        user_id: str = "default_user",
        conversation_threshold: int = 15,
        storage_dir: Optional[str] = None,
        personality_agent = None,
        emotion_agent = None
    ):
        """
        Initialize the context-aware memory adapter

        Args:
            user_id: Unique identifier for the user
            conversation_threshold: Number of exchanges before summarizing conversation
            storage_dir: Directory to store memory data
            personality_agent: Optional personality agent for enhanced adaptations
            emotion_agent: Optional emotion agent for emotional context
        """
        self.user_id = user_id
        self.conversation_threshold = conversation_threshold
        self.conversation_count = 0
        self.last_summary_time = time.time()
        self.summaries = []

        # Initialize semantic memory manager
        self.memory_manager = SemanticMemoryManager(
            user_id=user_id,
            storage_dir=storage_dir
        )

        # Store agent references for context enhancement
        self.personality_agent = personality_agent
        self.emotion_agent = emotion_agent

        # Context windows for different aspects
        self.emotion_context = {}
        self.personality_context = {}
        self.conversation_context = []

        # Personality adaptation state
        self.personality_adaptations = {
            "current_tone": "supportive",
            "warmth": 0.7,
            "formal": 0.4,
            "complex": 0.5,
            "emotion_mirroring": 0.6
        }

    async def add_message(self, message: str, role: str = "user", metadata: Dict[str, Any] = None) -> int:
        """
        Add a message to memory

        Args:
            message: Message text
            role: Role of the message sender (user or assistant)
            metadata: Additional metadata about the message

        Returns:
            ID of the memory entry
        """
        try:
            # Create memory content
            content = {
                "role": role,
                "text": message,
                "metadata": metadata or {}
            }

            # Add to memory
            memory_id = self.memory_manager.add_memory(content, memory_type="conversation")

            # Update conversation count
            self.conversation_count += 1

            # Add to conversation context
            self.conversation_context.append(content)

            # Trim conversation context if too long
            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]

            # Check if we should generate a summary
            if self._should_summarize():
                await self.generate_summary()

            return memory_id
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
            return -1

    def _should_summarize(self) -> bool:
        """Determine if conversation should be summarized"""
        # Summarize if conversation count exceeds threshold
        if self.conversation_count >= self.conversation_threshold:
            return True

        # Summarize if it's been more than 30 minutes since last summary
        current_time = time.time()
        if current_time - self.last_summary_time > 1800:  # 30 minutes
            return True

        return False

    async def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the recent conversation

        Returns:
            Summary dictionary
        """
        try:
            # Generate basic summary using memory manager
            summary = self.memory_manager.generate_conversation_summary()

            # Add summary to list
            self.summaries.append(summary)

            # Reset conversation count
            self.conversation_count = 0
            self.last_summary_time = time.time()

            # In a real implementation, we would use an LLM to generate a better summary
            # and extract key themes, emotional patterns, etc.

            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary_text": f"Error generating summary: {str(e)}",
                "message_count": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def search_conversations(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversation history with semantic search

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching memory entries
        """
        try:
            return self.memory_manager.search_memory(
                query=query,
                memory_type="conversation",
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error searching conversations: {str(e)}")
            return []

    async def get_relevant_context(
        self,
        current_message: str,
        top_k: int = 3,
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Get relevant context for the current message

        Args:
            current_message: Current user message
            top_k: Number of relevant memories to include
            include_summaries: Whether to include conversation summaries

        Returns:
            Dictionary with relevant context
        """
        try:
            # Get relevant memories
            relevant_memories = await self.search_conversations(current_message, top_k)

            # Get relevant summaries if requested
            relevant_summaries = []
            if include_summaries and self.summaries:
                # Simple approach: use the most recent summary
                relevant_summaries = [self.summaries[-1]]

            # Combine into context
            context = {
                "relevant_memories": relevant_memories,
                "summaries": relevant_summaries,
                "emotion_context": self.emotion_context,
                "personality_context": self.personality_context,
                "conversation_context": self.conversation_context[-5:] if self.conversation_context else []
            }

            return context
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return {
                "relevant_memories": [],
                "summaries": [],
                "emotion_context": {},
                "personality_context": {},
                "conversation_context": []
            }

    async def update_emotion_context(self, emotion_data: Dict[str, Any]) -> None:
        """
        Update emotional context

        Args:
            emotion_data: Emotion analysis data
        """
        try:
            self.emotion_context = emotion_data

            # Store in memory for future reference
            self.memory_manager.add_memory(emotion_data, memory_type="emotion")

            # Adjust personality based on emotion if needed
            if emotion_data.get("primary_emotion"):
                await self._adjust_personality_for_emotion(emotion_data)
        except Exception as e:
            logger.error(f"Error updating emotion context: {str(e)}")

    async def update_personality_context(self, personality_data: Dict[str, Any]) -> None:
        """
        Update personality context

        Args:
            personality_data: Personality assessment data
        """
        try:
            self.personality_context = personality_data

            # Store in memory for future reference
            self.memory_manager.add_memory(personality_data, memory_type="assessment")
        except Exception as e:
            logger.error(f"Error updating personality context: {str(e)}")

    async def _adjust_personality_for_emotion(self, emotion_data: Dict[str, Any]) -> None:
        """
        Adjust personality based on detected emotion

        Args:
            emotion_data: Emotion analysis data
        """
        try:
            primary_emotion = emotion_data.get("primary_emotion", "neutral")
            intensity = emotion_data.get("intensity", 5) / 10.0  # Normalize to 0-1

            # Base adjustments
            adaptations = {
                "warmth": self.personality_adaptations["warmth"],
                "formal": self.personality_adaptations["formal"],
                "complex": self.personality_adaptations["complex"],
                "emotion_mirroring": self.personality_adaptations["emotion_mirroring"]
            }

            # Adjust based on emotion
            if primary_emotion in ["sad", "depressed", "grief"]:
                adaptations["warmth"] = min(1.0, adaptations["warmth"] + 0.2 * intensity)
                adaptations["formal"] = max(0.2, adaptations["formal"] - 0.1 * intensity)
                adaptations["emotion_mirroring"] = min(0.7, adaptations["emotion_mirroring"] + 0.1 * intensity)
                adaptations["current_tone"] = "empathetic"

            elif primary_emotion in ["anxious", "worried", "stressed"]:
                adaptations["warmth"] = min(0.9, adaptations["warmth"] + 0.1 * intensity)
                adaptations["complex"] = max(0.3, adaptations["complex"] - 0.2 * intensity)
                adaptations["current_tone"] = "reassuring"

            elif primary_emotion in ["angry", "frustrated"]:
                adaptations["warmth"] = min(0.8, adaptations["warmth"] + 0.1)
                adaptations["formal"] = min(0.6, adaptations["formal"] + 0.1 * intensity)
                adaptations["emotion_mirroring"] = max(0.3, adaptations["emotion_mirroring"] - 0.1 * intensity)
                adaptations["current_tone"] = "calm"

            elif primary_emotion in ["happy", "excited", "grateful"]:
                adaptations["warmth"] = min(0.9, adaptations["warmth"] + 0.1)
                adaptations["formal"] = max(0.3, adaptations["formal"] - 0.1)
                adaptations["emotion_mirroring"] = min(0.8, adaptations["emotion_mirroring"] + 0.1)
                adaptations["current_tone"] = "cheerful"

            elif primary_emotion in ["neutral", "calm"]:
                # Gradually return to baseline
                for key in ["warmth", "formal", "complex", "emotion_mirroring"]:
                    baseline = 0.5 if key != "warmth" else 0.7
                    current = adaptations[key]
                    adaptations[key] = current + (baseline - current) * 0.2
                adaptations["current_tone"] = "balanced"

            # Update personality adaptations
            self.personality_adaptations.update(adaptations)

        except Exception as e:
            logger.error(f"Error adjusting personality for emotion: {str(e)}")

    def get_current_personality_adjustments(self) -> Dict[str, Any]:
        """
        Get current personality adjustments

        Returns:
            Dictionary of personality adjustments
        """
        return self.personality_adaptations.copy()

    def format_context_for_prompt(self,
                                  include_memories: bool = True,
                                  include_summaries: bool = True,
                                  include_emotions: bool = True) -> str:
        """
        Format context for inclusion in prompts

        Args:
            include_memories: Whether to include relevant memories
            include_summaries: Whether to include conversation summaries
            include_emotions: Whether to include emotional context

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add relevant memories if requested
        if include_memories and hasattr(self, "current_context") and "relevant_memories" in self.current_context:
            memories = self.current_context["relevant_memories"]
            if memories:
                memory_texts = []
                for memory in memories[:3]:  # Limit to 3 memories
                    content = memory.get("content", {})
                    role = content.get("role", "unknown")
                    text = content.get("text", "")
                    memory_texts.append(f"{role}: {text}")

                context_parts.append("Relevant past conversations:")
                context_parts.append("\n".join(memory_texts))

        # Add summaries if requested
        if include_summaries and self.summaries:
            recent_summary = self.summaries[-1]
            summary_text = recent_summary.get("summary_text", "")
            if summary_text and not summary_text.startswith("No recent conversation"):
                context_parts.append("Conversation summary:")
                context_parts.append(summary_text[:200] + "..." if len(summary_text) > 200 else summary_text)

        # Add emotional context if requested
        if include_emotions and self.emotion_context:
            primary_emotion = self.emotion_context.get("primary_emotion", "")
            if primary_emotion:
                context_parts.append(f"User emotional state: {primary_emotion}")

                # Add secondary emotions if available
                secondary_emotions = self.emotion_context.get("secondary_emotions", [])
                if secondary_emotions:
                    context_parts.append(f"Secondary emotions: {', '.join(secondary_emotions[:3])}")

        # Combine all parts
        return "\n\n".join(context_parts)

    def clear_memory(self) -> bool:
        """
        Clear all memories

        Returns:
            Success status
        """
        try:
            result = self.memory_manager.clear_all_memories()
            self.conversation_count = 0
            self.last_summary_time = time.time()
            self.summaries = []
            self.conversation_context = []
            return result
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False

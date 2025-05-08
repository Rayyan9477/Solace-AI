"""
Semantic Memory Manager for the Contextual Chatbot.
Provides advanced memory storage and retrieval capabilities with semantic search.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re

logger = logging.getLogger(__name__)

class SemanticMemoryManager:
    """
    Manages memory storage and retrieval with semantic search capabilities.
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        storage_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "flat",
        max_memory_entries: int = 1000
    ):
        """
        Initialize the semantic memory manager
        
        Args:
            user_id: Unique identifier for the user
            storage_dir: Directory to store memory data
            embedding_model: Model name for sentence transformers
            dimension: Embedding dimension
            index_type: FAISS index type (flat, ivf, or hnsw)
            max_memory_entries: Maximum number of memory entries to maintain
        """
        self.user_id = user_id
        self.embedding_dimension = dimension
        self.max_memory_entries = max_memory_entries
        
        # Set up storage directory
        if storage_dir is None:
            self.storage_dir = Path(__file__).parent.parent.parent / "data" / "memory" / user_id
        else:
            self.storage_dir = Path(storage_dir) / user_id
            
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to simpler mechanisms if model fails to load
            self.embedding_model = None
        
        # Initialize FAISS index based on type
        self.index_type = index_type
        self.index = self._create_index()
        
        # Memory data structures
        self.memory_entries = []
        self.id_to_entry = {}
        self.next_id = 0
        
        # Load existing memories
        self._load_memories()
    
    def _create_index(self) -> Any:
        """Create a FAISS index of the specified type"""
        try:
            if self.index_type == "flat":
                return faiss.IndexFlatIP(self.embedding_dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                return faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
            elif self.index_type == "hnsw":
                return faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            else:
                logger.warning(f"Unknown index type: {self.index_type}, using flat index")
                return faiss.IndexFlatIP(self.embedding_dimension)
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return faiss.IndexFlatIP(self.embedding_dimension)
    
    def _load_memories(self) -> None:
        """Load memories from disk"""
        try:
            index_path = self.storage_dir / "memory_index.faiss"
            entries_path = self.storage_dir / "memory_entries.pkl"
            
            if index_path.exists() and entries_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load memory entries
                with open(entries_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.memory_entries = loaded_data.get('entries', [])
                    self.id_to_entry = loaded_data.get('id_to_entry', {})
                    self.next_id = loaded_data.get('next_id', len(self.memory_entries))
                
                logger.info(f"Loaded {len(self.memory_entries)} memories from {entries_path}")
        except Exception as e:
            logger.error(f"Error loading memories: {str(e)}")
            # If loading fails, start with empty memories
            self.memory_entries = []
            self.id_to_entry = {}
            self.next_id = 0
    
    def _save_memories(self) -> None:
        """Save memories to disk"""
        try:
            # Save FAISS index
            index_path = self.storage_dir / "memory_index.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save memory entries
            entries_path = self.storage_dir / "memory_entries.pkl"
            with open(entries_path, 'wb') as f:
                pickle.dump({
                    'entries': self.memory_entries,
                    'id_to_entry': self.id_to_entry,
                    'next_id': self.next_id
                }, f)
            
            logger.info(f"Saved {len(self.memory_entries)} memories to {entries_path}")
        except Exception as e:
            logger.error(f"Error saving memories: {str(e)}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self.embedding_model is None:
            # Fallback to random embedding if model isn't available
            return np.random.randn(self.embedding_dimension).astype(np.float32)
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.random.randn(self.embedding_dimension).astype(np.float32)
    
    def add_memory(self, content: Dict[str, Any], memory_type: str = "conversation") -> int:
        """
        Add a new memory entry
        
        Args:
            content: Memory content (dict with at least 'text' field)
            memory_type: Type of memory (conversation, emotion, assessment, etc.)
            
        Returns:
            ID of the new memory entry
        """
        try:
            # Ensure content has required fields
            if not content or not isinstance(content, dict):
                raise ValueError("Memory content must be a non-empty dictionary")
            
            if 'text' not in content and memory_type == "conversation":
                raise ValueError("Conversation memory must include 'text' field")
            
            # Create memory entry
            memory_id = self.next_id
            timestamp = datetime.now().isoformat()
            
            # Create searchable text
            searchable_text = self._create_searchable_text(content, memory_type)
            
            # Generate embedding
            embedding = self._get_embedding(searchable_text)
            
            # Create memory entry
            memory_entry = {
                'id': memory_id,
                'type': memory_type,
                'content': content,
                'timestamp': timestamp,
                'searchable_text': searchable_text,
                'embedding': embedding
            }
            
            # Add to memory structures
            self.memory_entries.append(memory_entry)
            self.id_to_entry[memory_id] = memory_entry
            
            # Add to FAISS index
            self.index.add(np.array([embedding]))
            
            # Update next ID
            self.next_id += 1
            
            # Check if we need to prune old memories
            if len(self.memory_entries) > self.max_memory_entries:
                self._prune_old_memories()
            
            # Save memories periodically
            if self.next_id % 10 == 0:
                self._save_memories()
            
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            return -1
    
    def _prune_old_memories(self) -> None:
        """Remove oldest memories to stay within memory limit"""
        try:
            # Sort memories by timestamp (oldest first)
            sorted_memories = sorted(
                self.memory_entries,
                key=lambda x: x['timestamp']
            )
            
            # Determine how many to remove
            excess = len(sorted_memories) - self.max_memory_entries
            to_remove = sorted_memories[:excess]
            
            # Create fresh FAISS index
            new_index = self._create_index()
            
            # Build new memory structures without old memories
            new_entries = []
            new_id_to_entry = {}
            
            for entry in sorted_memories[excess:]:
                new_entries.append(entry)
                new_id_to_entry[entry['id']] = entry
                new_index.add(np.array([entry['embedding']]))
            
            # Replace old structures
            self.memory_entries = new_entries
            self.id_to_entry = new_id_to_entry
            self.index = new_index
            
            logger.info(f"Pruned {excess} old memories")
        except Exception as e:
            logger.error(f"Error pruning memories: {str(e)}")
    
    def _create_searchable_text(self, content: Dict[str, Any], memory_type: str) -> str:
        """Create searchable text from memory content"""
        if memory_type == "conversation":
            return content.get('text', '')
        elif memory_type == "emotion":
            emotions = content.get('primary_emotion', 'neutral')
            triggers = ', '.join(content.get('triggers', []))
            return f"Emotion: {emotions}. Triggers: {triggers}"
        elif memory_type == "assessment":
            assessment_type = content.get('assessment_type', 'unknown')
            results = content.get('results', {})
            return f"Assessment: {assessment_type}. Results: {json.dumps(results)}"
        elif memory_type == "summary":
            return content.get('summary_text', '')
        else:
            # Fallback to JSON string
            return json.dumps(content)
    
    def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity
        
        Args:
            query: Search query
            memory_type: Optional filter by memory type
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching memory entries
        """
        try:
            if len(self.memory_entries) == 0:
                return []
            
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Search FAISS index
            similarity_scores, indices = self.index.search(
                np.array([query_embedding]),
                min(top_k * 2, len(self.memory_entries))  # Get more than needed for filtering
            )
            
            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.memory_entries):
                    continue
                
                # Get similarity score
                similarity = similarity_scores[0][i]
                
                # Skip if below threshold
                if similarity < min_similarity:
                    continue
                
                # Get memory entry
                memory_entry = self.memory_entries[idx]
                
                # Filter by memory type if specified
                if memory_type and memory_entry['type'] != memory_type:
                    continue
                
                # Add to results without embedding
                result = {
                    'id': memory_entry['id'],
                    'type': memory_entry['type'],
                    'content': memory_entry['content'],
                    'timestamp': memory_entry['timestamp'],
                    'similarity': float(similarity)
                }
                
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get memory by ID"""
        try:
            memory_entry = self.id_to_entry.get(memory_id)
            if memory_entry:
                # Return without embedding
                return {
                    'id': memory_entry['id'],
                    'type': memory_entry['type'],
                    'content': memory_entry['content'],
                    'timestamp': memory_entry['timestamp']
                }
            return None
        except Exception as e:
            logger.error(f"Error getting memory by ID: {str(e)}")
            return None
    
    def get_recent_memories(
        self,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most recent memories
        
        Args:
            memory_type: Optional filter by memory type
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memory entries
        """
        try:
            # Filter by type if specified
            if memory_type:
                filtered_memories = [m for m in self.memory_entries if m['type'] == memory_type]
            else:
                filtered_memories = self.memory_entries
            
            # Sort by timestamp (newest first)
            sorted_memories = sorted(
                filtered_memories,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            # Take only requested limit
            limited_memories = sorted_memories[:limit]
            
            # Format for return (without embeddings)
            results = []
            for memory in limited_memories:
                results.append({
                    'id': memory['id'],
                    'type': memory['type'],
                    'content': memory['content'],
                    'timestamp': memory['timestamp']
                })
            
            return results
        except Exception as e:
            logger.error(f"Error getting recent memories: {str(e)}")
            return []
    
    def generate_conversation_summary(self, max_messages: int = 20) -> Dict[str, Any]:
        """
        Generate a summary of recent conversation
        
        Args:
            max_messages: Maximum number of messages to include in summary
            
        Returns:
            Summary dictionary
        """
        try:
            # Get recent conversation memories
            recent_messages = self.get_recent_memories(memory_type="conversation", limit=max_messages)
            
            if not recent_messages:
                return {
                    "summary_text": "No recent conversation to summarize.",
                    "message_count": 0,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Extract conversation text
            conversation = []
            for message in recent_messages:
                content = message['content']
                role = content.get('role', 'unknown')
                text = content.get('text', '')
                conversation.append(f"{role}: {text}")
            
            # Create summary text
            conversation_text = "\n".join(conversation)
            
            # For now, simple truncation-based summary
            # In real implementation, use LLM to generate proper summary
            if len(conversation_text) > 1000:
                summary_text = conversation_text[:997] + "..."
            else:
                summary_text = conversation_text
            
            # Create summary object
            summary = {
                "summary_text": summary_text,
                "message_count": len(recent_messages),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the summary
            self.add_memory(summary, memory_type="summary")
            
            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return {
                "summary_text": f"Error generating summary: {str(e)}",
                "message_count": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_all_memories(self) -> bool:
        """Clear all memories"""
        try:
            # Reset memory structures
            self.memory_entries = []
            self.id_to_entry = {}
            self.next_id = 0
            
            # Reset FAISS index
            self.index = self._create_index()
            
            # Save empty state
            self._save_memories()
            
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {str(e)}")
            return False
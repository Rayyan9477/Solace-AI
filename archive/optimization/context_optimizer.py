"""
Context Window Optimization for Multi-Agent System

Implements intelligent context compression and management to optimize
token usage and improve response quality.
"""

from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContextItem:
    """Represents a single context item with metadata."""
    content: str
    importance_score: float
    timestamp: datetime
    agent_source: str
    token_count: int
    item_type: str  # 'memory', 'knowledge', 'emotion', 'safety', etc.


class SemanticContextCompressor:
    """Compress context using semantic importance and relevance."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.compression_cache = {}

    def compress(self, context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Compress context to fit within token limits while preserving important information.

        Args:
            context: Full context dictionary
            query: Current user query for relevance scoring

        Returns:
            Compressed context within token limits
        """
        # Extract and score all context items
        context_items = self._extract_context_items(context)

        # Score items by relevance to query and recency
        scored_items = self._score_items(context_items, query)

        # Select items within token budget
        selected_items = self._select_items_within_budget(scored_items)

        # Reconstruct compressed context
        compressed_context = self._reconstruct_context(selected_items)

        return compressed_context

    def _extract_context_items(self, context: Dict[str, Any]) -> List[ContextItem]:
        """Extract all context items with metadata."""
        items = []

        # Extract memory items
        if 'memory' in context and context['memory']:
            for idx, mem in enumerate(context['memory']):
                items.append(ContextItem(
                    content=mem.get('content', ''),
                    importance_score=0.8 - (idx * 0.1),  # Recent memories more important
                    timestamp=datetime.now() - timedelta(minutes=idx * 5),
                    agent_source='memory',
                    token_count=len(mem.get('content', '')) // 4,
                    item_type='memory'
                ))

        # Extract knowledge items
        if 'knowledge' in context and context['knowledge']:
            for item in context['knowledge']:
                items.append(ContextItem(
                    content=item.get('content', ''),
                    importance_score=item.get('relevance_score', 0.5),
                    timestamp=datetime.now(),
                    agent_source='knowledge_base',
                    token_count=len(item.get('content', '')) // 4,
                    item_type='knowledge'
                ))

        # Extract emotion context
        if 'emotion' in context:
            emotion_str = json.dumps(context['emotion'])
            items.append(ContextItem(
                content=emotion_str,
                importance_score=0.9,  # Emotion context is very important
                timestamp=datetime.now(),
                agent_source='emotion_agent',
                token_count=len(emotion_str) // 4,
                item_type='emotion'
            ))

        # Extract safety context
        if 'safety' in context:
            safety_str = json.dumps(context['safety'])
            items.append(ContextItem(
                content=safety_str,
                importance_score=0.95,  # Safety is critical
                timestamp=datetime.now(),
                agent_source='safety_agent',
                token_count=len(safety_str) // 4,
                item_type='safety'
            ))

        return items

    def _score_items(self, items: List[ContextItem], query: str) -> List[Tuple[float, ContextItem]]:
        """Score items by relevance and importance."""
        scored = []

        for item in items:
            # Calculate relevance score (simple keyword matching for now)
            relevance = self._calculate_relevance(item.content, query)

            # Calculate recency score
            age_minutes = (datetime.now() - item.timestamp).total_seconds() / 60
            recency_score = max(0, 1 - (age_minutes / 60))  # Decay over 1 hour

            # Combined score
            final_score = (
                item.importance_score * 0.4 +
                relevance * 0.4 +
                recency_score * 0.2
            )

            scored.append((final_score, item))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance between content and query."""
        # Simple keyword overlap for now
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        overlap = len(content_words.intersection(query_words))
        relevance = overlap / len(query_words)

        return min(1.0, relevance)

    def _select_items_within_budget(self, scored_items: List[Tuple[float, ContextItem]]) -> List[ContextItem]:
        """Select items that fit within token budget."""
        selected = []
        total_tokens = 0

        # Always include critical items (safety, emotion) if present
        critical_types = ['safety', 'emotion']
        for score, item in scored_items:
            if item.item_type in critical_types:
                if total_tokens + item.token_count <= self.max_tokens:
                    selected.append(item)
                    total_tokens += item.token_count

        # Add remaining items by score
        for score, item in scored_items:
            if item not in selected:
                if total_tokens + item.token_count <= self.max_tokens * 0.9:  # Leave 10% buffer
                    selected.append(item)
                    total_tokens += item.token_count
                else:
                    # Try to compress item content if it's text
                    if item.item_type in ['memory', 'knowledge']:
                        compressed_content = self._truncate_text(
                            item.content,
                            max_tokens=min(200, self.max_tokens - total_tokens)
                        )
                        item.content = compressed_content
                        item.token_count = len(compressed_content) // 4
                        if total_tokens + item.token_count <= self.max_tokens:
                            selected.append(item)
                            total_tokens += item.token_count

        return selected

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Intelligently truncate text to fit token limit."""
        max_chars = max_tokens * 4  # Rough estimate

        if len(text) <= max_chars:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        cut_point = max(last_period, last_newline)
        if cut_point > max_chars * 0.7:  # Only if we don't lose too much
            truncated = truncated[:cut_point + 1]

        return truncated + "..."

    def _reconstruct_context(self, items: List[ContextItem]) -> Dict[str, Any]:
        """Reconstruct context dictionary from selected items."""
        compressed_context = {
            'memory': [],
            'knowledge': [],
            'emotion': None,
            'safety': None,
            'compression_metadata': {
                'original_items': len(items),
                'total_tokens': sum(item.token_count for item in items),
                'compression_ratio': 0.0
            }
        }

        for item in items:
            if item.item_type == 'memory':
                compressed_context['memory'].append({
                    'role': 'user',
                    'content': item.content,
                    'timestamp': item.timestamp.isoformat()
                })
            elif item.item_type == 'knowledge':
                compressed_context['knowledge'].append({
                    'content': item.content,
                    'relevance_score': item.importance_score
                })
            elif item.item_type == 'emotion':
                try:
                    compressed_context['emotion'] = json.loads(item.content)
                except (json.JSONDecodeError, TypeError, ValueError):
                    compressed_context['emotion'] = {'primary_emotion': 'unknown'}
            elif item.item_type == 'safety':
                try:
                    compressed_context['safety'] = json.loads(item.content)
                except (json.JSONDecodeError, TypeError, ValueError):
                    compressed_context['safety'] = {'safe': True}

        return compressed_context


class AgentResultCache:
    """Cache agent results to avoid redundant processing."""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
        self.hit_count = 0
        self.miss_count = 0

    def get(self, agent_name: str, input_hash: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        cache_key = f"{agent_name}:{input_hash}"

        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.ttl:
                self.hit_count += 1
                logger.debug(f"Cache hit for {agent_name}")
                return result
            else:
                # Expired, remove it
                del self.cache[cache_key]

        self.miss_count += 1
        return None

    def set(self, agent_name: str, input_hash: str, result: Any):
        """Cache an agent result."""
        cache_key = f"{agent_name}:{input_hash}"
        self.cache[cache_key] = (result, datetime.now())

        # Cleanup old entries
        self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cached_items': len(self.cache),
            'memory_usage_kb': sum(
                len(str(result)) for result, _ in self.cache.values()
            ) / 1024
        }

    @staticmethod
    def compute_input_hash(input_data: Any, context: Dict[str, Any]) -> str:
        """Compute hash for cache key."""
        # Create deterministic string representation
        cache_data = {
            'input': str(input_data),
            'context_keys': sorted(context.keys()) if context else []
        }

        # Add critical context values (not full context to avoid cache misses)
        if context:
            if 'emotion' in context and context['emotion']:
                cache_data['emotion'] = context['emotion'].get('primary_emotion')
            if 'safety' in context and context['safety']:
                cache_data['risk_level'] = context['safety'].get('risk_level')

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()


class ContextWindowManager:
    """Manage context windows across multiple agents."""

    def __init__(self, default_window_size: int = 4000):
        self.default_window_size = default_window_size

        # Load window sizes from config if available
        try:
            from src.config import OptimizationConfig
            self.agent_windows = OptimizationConfig.CONTEXT_CONFIG.get("agent_windows", {})
            if not self.agent_windows:
                raise ImportError("No agent windows configured")
        except (ImportError, AttributeError):
            # Fallback to default window sizes
            self.agent_windows = {
                # Critical agents get larger windows
                'chat_agent': 8000,
                'therapy_agent': 6000,
                'diagnosis_agent': 6000,

                # Standard agents
                'emotion_agent': 4000,
                'personality_agent': 4000,
                'safety_agent': 3000,

                # Support agents get smaller windows
                'search_agent': 2000,
                'crawler_agent': 2000
            }

        self.compressor = SemanticContextCompressor()

        # Configure cache with TTL from config if available
        try:
            from src.config import OptimizationConfig
            cache_ttl = OptimizationConfig.CACHE_CONFIG.get("ttl_seconds", 300)
        except (ImportError, AttributeError):
            cache_ttl = 300

        self.cache = AgentResultCache(ttl_seconds=cache_ttl)

    def get_optimized_context(self, agent_name: str, full_context: Dict[str, Any],
                              query: str) -> Dict[str, Any]:
        """Get optimized context for a specific agent."""
        # Get window size for this agent
        window_size = self.agent_windows.get(agent_name, self.default_window_size)

        # Configure compressor for this window
        self.compressor.max_tokens = window_size

        # Compress context
        optimized_context = self.compressor.compress(full_context, query)

        # Add agent-specific optimizations
        if agent_name == 'chat_agent':
            # Chat agent needs more conversation history
            if 'memory' in full_context:
                optimized_context['memory'] = full_context['memory'][-5:]  # Last 5 messages

        elif agent_name in ['search_agent', 'crawler_agent']:
            # These agents don't need conversation history
            optimized_context.pop('memory', None)

        elif agent_name == 'safety_agent':
            # Safety agent needs all safety-related context
            if 'safety_history' in full_context:
                optimized_context['safety_history'] = full_context['safety_history']

        return optimized_context

    def should_use_cache(self, agent_name: str, input_data: Any,
                        context: Dict[str, Any]) -> Tuple[bool, Optional[Any]]:
        """Check if cached result can be used."""
        # Generate cache key
        input_hash = self.cache.compute_input_hash(input_data, context)

        # Check cache
        cached_result = self.cache.get(agent_name, input_hash)

        if cached_result:
            # Validate cached result is still appropriate
            if self._is_cache_valid(agent_name, cached_result, context):
                return True, cached_result

        return False, None

    def _is_cache_valid(self, agent_name: str, cached_result: Any,
                        context: Dict[str, Any]) -> bool:
        """Validate if cached result is still appropriate."""
        # Don't use cache for safety-critical agents if risk level changed
        if agent_name == 'safety_agent':
            if context.get('safety', {}).get('risk_level') in ['HIGH', 'SEVERE']:
                return False

        # Don't use cache for therapy agent if emotion changed significantly
        if agent_name == 'therapy_agent':
            current_emotion = context.get('emotion', {}).get('primary_emotion')
            if current_emotion in ['crisis', 'severe_depression', 'suicidal']:
                return False

        return True

    def cache_result(self, agent_name: str, input_data: Any,
                    context: Dict[str, Any], result: Any):
        """Cache an agent's result."""
        input_hash = self.cache.compute_input_hash(input_data, context)
        self.cache.set(agent_name, input_hash, result)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'cache_stats': self.cache.get_cache_stats(),
            'compression_stats': {
                'average_compression_ratio': 0.7,  # Placeholder
                'tokens_saved': self.default_window_size * 0.3
            },
            'agent_window_usage': self.agent_windows
        }
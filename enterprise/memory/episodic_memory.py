"""
Episodic Memory System for Session-Specific Context and Experiences
Manages temporary memories that are relevant for current sessions
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import uuid
from collections import deque, defaultdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class EpisodeType(Enum):
    SESSION_START = "session_start"
    CONVERSATION_TURN = "conversation_turn"
    EMOTION_DETECTED = "emotion_detected"
    THERAPY_INTERVENTION = "therapy_intervention"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"
    SESSION_END = "session_end"
    CRISIS_DETECTED = "crisis_detected"


class EmotionalState(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class Episode:
    """Individual episodic memory"""
    episode_id: str
    session_id: str
    user_id: str
    episode_type: EpisodeType
    timestamp: datetime
    content: str
    emotional_state: EmotionalState
    importance: float  # 0.0 to 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    duration: Optional[float] = None  # in seconds
    related_episodes: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Session-wide context and state"""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    emotional_trajectory: List[Tuple[datetime, EmotionalState]]
    key_topics: List[str]
    therapy_techniques_used: List[str]
    breakthroughs: List[str]
    current_emotional_state: EmotionalState
    session_goals: List[str]
    progress_indicators: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemoryManager:
    """
    Manages episodic memories for sessions with advanced context tracking
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.active_sessions: Dict[str, SessionContext] = {}
        self.episode_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.consolidation_threshold = 50  # episodes before consolidation
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        logger.info("Episodic memory manager initialized")
        
    async def start_session(self, user_id: str, session_goals: List[str] = None) -> str:
        """Start a new session and create session context"""
        session_id = str(uuid.uuid4())
        
        session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            emotional_trajectory=[],
            key_topics=[],
            therapy_techniques_used=[],
            breakthroughs=[],
            current_emotional_state=EmotionalState.NEUTRAL,
            session_goals=session_goals or [],
            progress_indicators={}
        )
        
        self.active_sessions[session_id] = session_context
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            f"session:{session_id}",
            mapping={
                "user_id": user_id,
                "started_at": session_context.started_at.isoformat(),
                "session_goals": json.dumps(session_goals or []),
                "status": "active"
            }
        )
        
        # Create session start episode
        await self.record_episode(
            session_id=session_id,
            episode_type=EpisodeType.SESSION_START,
            content=f"Session started with goals: {session_goals}",
            emotional_state=EmotionalState.NEUTRAL,
            importance=0.8,
            context={"goals": session_goals or []}
        )
        
        logger.info(f"Started session {session_id} for user {user_id}")
        return session_id
        
    async def end_session(self, session_id: str, session_summary: str = "") -> Dict[str, Any]:
        """End a session and consolidate memories"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session_context = self.active_sessions[session_id]
        
        # Create session end episode
        await self.record_episode(
            session_id=session_id,
            episode_type=EpisodeType.SESSION_END,
            content=f"Session ended. Summary: {session_summary}",
            emotional_state=session_context.current_emotional_state,
            importance=0.9,
            context={"summary": session_summary}
        )
        
        # Consolidate session memories
        session_insights = await self.consolidate_session_memories(session_id)
        
        # Update Redis
        await self.redis_client.hset(
            f"session:{session_id}",
            mapping={
                "ended_at": datetime.utcnow().isoformat(),
                "status": "completed",
                "summary": session_summary,
                "insights": json.dumps(session_insights)
            }
        )
        
        # Move to inactive sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Ended session {session_id}")
        return session_insights
        
    async def record_episode(self, session_id: str, episode_type: EpisodeType,
                           content: str, emotional_state: EmotionalState,
                           importance: float, context: Dict[str, Any] = None,
                           tags: List[str] = None, duration: float = None) -> str:
        """Record a new episode in the session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session_context = self.active_sessions[session_id]
        episode_id = str(uuid.uuid4())
        
        episode = Episode(
            episode_id=episode_id,
            session_id=session_id,
            user_id=session_context.user_id,
            episode_type=episode_type,
            timestamp=datetime.utcnow(),
            content=content,
            emotional_state=emotional_state,
            importance=importance,
            context=context or {},
            tags=tags or [],
            duration=duration
        )
        
        # Add to buffer
        self.episode_buffer[session_id].append(episode)
        
        # Update session context
        session_context.last_activity = episode.timestamp
        session_context.current_emotional_state = emotional_state
        session_context.emotional_trajectory.append((episode.timestamp, emotional_state))
        
        # Extract and update key topics
        await self._update_key_topics(session_context, content)
        
        # Store in Redis
        await self.redis_client.hset(
            f"episode:{episode_id}",
            mapping={
                "session_id": session_id,
                "user_id": session_context.user_id,
                "episode_type": episode_type.value,
                "timestamp": episode.timestamp.isoformat(),
                "content": content,
                "emotional_state": emotional_state.value,
                "importance": importance,
                "context": json.dumps(context or {}),
                "tags": json.dumps(tags or []),
                "duration": duration or 0
            }
        )
        
        # Add to session episodes list
        await self.redis_client.lpush(f"session_episodes:{session_id}", episode_id)
        
        # Check for consolidation
        if len(self.episode_buffer[session_id]) >= self.consolidation_threshold:
            await self._consolidate_episodes(session_id)
            
        return episode_id
        
    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get current session context"""
        return self.active_sessions.get(session_id)
        
    async def get_recent_episodes(self, session_id: str, count: int = 10) -> List[Episode]:
        """Get recent episodes from a session"""
        if session_id in self.episode_buffer:
            episodes = list(self.episode_buffer[session_id])
            return episodes[-count:]
        
        # Fallback to Redis
        episode_ids = await self.redis_client.lrange(f"session_episodes:{session_id}", 0, count-1)
        episodes = []
        
        for episode_id in episode_ids:
            episode_data = await self.redis_client.hgetall(f"episode:{episode_id}")
            if episode_data:
                episode = Episode(
                    episode_id=episode_id,
                    session_id=episode_data["session_id"],
                    user_id=episode_data["user_id"],
                    episode_type=EpisodeType(episode_data["episode_type"]),
                    timestamp=datetime.fromisoformat(episode_data["timestamp"]),
                    content=episode_data["content"],
                    emotional_state=EmotionalState(episode_data["emotional_state"]),
                    importance=float(episode_data["importance"]),
                    context=json.loads(episode_data.get("context", "{}")),
                    tags=json.loads(episode_data.get("tags", "[]")),
                    duration=float(episode_data["duration"]) if episode_data.get("duration") else None
                )
                episodes.append(episode)
                
        return episodes[::-1]  # Reverse to get chronological order
        
    async def get_emotional_trajectory(self, session_id: str) -> List[Tuple[datetime, EmotionalState]]:
        """Get the emotional trajectory for a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].emotional_trajectory
            
        # Reconstruct from episodes
        episodes = await self.get_recent_episodes(session_id, 100)
        trajectory = [(ep.timestamp, ep.emotional_state) for ep in episodes]
        return sorted(trajectory, key=lambda x: x[0])
        
    async def detect_emotional_patterns(self, session_id: str) -> Dict[str, Any]:
        """Detect patterns in emotional states during the session"""
        trajectory = await self.get_emotional_trajectory(session_id)
        
        if len(trajectory) < 3:
            return {"pattern": "insufficient_data"}
            
        # Calculate emotional trend
        emotional_values = {
            EmotionalState.VERY_NEGATIVE: -2,
            EmotionalState.NEGATIVE: -1,
            EmotionalState.NEUTRAL: 0,
            EmotionalState.POSITIVE: 1,
            EmotionalState.VERY_POSITIVE: 2
        }
        
        values = [emotional_values[state] for _, state in trajectory]
        
        # Simple trend analysis
        if len(values) >= 5:
            recent_trend = sum(values[-3:]) - sum(values[:3])
            overall_trend = values[-1] - values[0]
            
            # Detect patterns
            pattern = "stable"
            if recent_trend > 1:
                pattern = "improving"
            elif recent_trend < -1:
                pattern = "declining"
            elif abs(overall_trend) > 2:
                pattern = "volatile"
                
            # Calculate emotional variability
            variability = sum(abs(values[i] - values[i-1]) for i in range(1, len(values))) / len(values)
            
            return {
                "pattern": pattern,
                "recent_trend": recent_trend,
                "overall_trend": overall_trend,
                "variability": variability,
                "current_state": trajectory[-1][1].value,
                "trajectory_length": len(trajectory)
            }
            
        return {"pattern": "insufficient_data"}
        
    async def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about the session"""
        if session_id not in self.active_sessions:
            # Try to load from Redis
            session_data = await self.redis_client.hgetall(f"session:{session_id}")
            if not session_data:
                return {"error": "Session not found"}
                
        context = self.active_sessions.get(session_id)
        episodes = await self.get_recent_episodes(session_id, 100)
        emotional_patterns = await self.detect_emotional_patterns(session_id)
        
        # Calculate session statistics
        session_duration = None
        if context:
            session_duration = (datetime.utcnow() - context.started_at).total_seconds() / 60  # minutes
            
        # Analyze episode types
        episode_type_counts = defaultdict(int)
        important_episodes = []
        
        for episode in episodes:
            episode_type_counts[episode.episode_type.value] += 1
            if episode.importance > 0.7:
                important_episodes.append({
                    "episode_id": episode.episode_id,
                    "type": episode.episode_type.value,
                    "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                    "importance": episode.importance,
                    "timestamp": episode.timestamp.isoformat()
                })
                
        # Calculate progress indicators
        progress_score = 0.0
        if episodes:
            # Simple progress calculation based on emotional improvement
            if emotional_patterns.get("recent_trend", 0) > 0:
                progress_score += 0.3
            if emotional_patterns.get("overall_trend", 0) > 0:
                progress_score += 0.4
            if len([ep for ep in episodes if ep.episode_type == EpisodeType.BREAKTHROUGH_MOMENT]) > 0:
                progress_score += 0.3
                
        return {
            "session_id": session_id,
            "session_duration_minutes": session_duration,
            "total_episodes": len(episodes),
            "episode_type_distribution": dict(episode_type_counts),
            "emotional_patterns": emotional_patterns,
            "important_episodes": important_episodes,
            "progress_score": progress_score,
            "key_topics": context.key_topics if context else [],
            "therapy_techniques_used": context.therapy_techniques_used if context else [],
            "breakthrough_count": len([ep for ep in episodes if ep.episode_type == EpisodeType.BREAKTHROUGH_MOMENT])
        }
        
    async def consolidate_session_memories(self, session_id: str) -> Dict[str, Any]:
        """Consolidate session memories into long-term insights"""
        insights = await self.get_session_insights(session_id)
        episodes = await self.get_recent_episodes(session_id, 100)
        
        # Extract key learnings
        key_learnings = []
        breakthrough_moments = [ep for ep in episodes if ep.episode_type == EpisodeType.BREAKTHROUGH_MOMENT]
        
        for breakthrough in breakthrough_moments:
            key_learnings.append({
                "type": "breakthrough",
                "content": breakthrough.content,
                "timestamp": breakthrough.timestamp.isoformat(),
                "emotional_state": breakthrough.emotional_state.value
            })
            
        # Identify most effective therapy techniques
        technique_episodes = [ep for ep in episodes if ep.episode_type == EpisodeType.THERAPY_INTERVENTION]
        technique_effectiveness = {}
        
        for i, technique_ep in enumerate(technique_episodes):
            # Look at emotional state changes after the technique
            if i < len(technique_episodes) - 1:
                next_episodes = episodes[episodes.index(technique_ep)+1:episodes.index(technique_ep)+3]
                emotional_improvement = 0
                for next_ep in next_episodes:
                    if next_ep.emotional_state.value in ["positive", "very_positive"]:
                        emotional_improvement += 1
                        
                technique_name = technique_ep.context.get("technique", "unknown")
                if technique_name not in technique_effectiveness:
                    technique_effectiveness[technique_name] = []
                technique_effectiveness[technique_name].append(emotional_improvement)
                
        # Calculate average effectiveness
        avg_technique_effectiveness = {}
        for technique, improvements in technique_effectiveness.items():
            avg_technique_effectiveness[technique] = sum(improvements) / len(improvements)
            
        consolidation_result = {
            "session_summary": insights,
            "key_learnings": key_learnings,
            "technique_effectiveness": avg_technique_effectiveness,
            "emotional_journey": {
                "start_state": episodes[0].emotional_state.value if episodes else "unknown",
                "end_state": episodes[-1].emotional_state.value if episodes else "unknown",
                "peak_positive": max([ep.emotional_state.value for ep in episodes if ep.emotional_state.value in ["positive", "very_positive"]], default="none"),
                "lowest_point": min([ep.emotional_state.value for ep in episodes if ep.emotional_state.value in ["negative", "very_negative"]], default="none")
            },
            "recommendations": await self._generate_recommendations(session_id, insights)
        }
        
        # Store consolidation in Redis
        await self.redis_client.set(
            f"session_consolidation:{session_id}",
            json.dumps(consolidation_result, default=str),
            ex=86400 * 30  # Keep for 30 days
        )
        
        return consolidation_result
        
    async def _update_key_topics(self, session_context: SessionContext, content: str):
        """Extract and update key topics from episode content"""
        # Simple keyword extraction (in production, use NLP)
        import re
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filter for meaningful topics (this is simplified)
        therapy_keywords = {
            "anxiety", "depression", "stress", "trauma", "relationships", 
            "family", "work", "sleep", "anger", "fear", "guilt", "shame",
            "mindfulness", "meditation", "breathing", "relaxation", "coping"
        }
        
        found_topics = [word for word in words if word in therapy_keywords]
        
        for topic in found_topics:
            if topic not in session_context.key_topics:
                session_context.key_topics.append(topic)
                
    async def _consolidate_episodes(self, session_id: str):
        """Consolidate episodes in buffer to Redis storage"""
        episodes = list(self.episode_buffer[session_id])
        
        # Clear buffer
        self.episode_buffer[session_id].clear()
        
        # Process episodes for patterns and store in Redis
        for episode in episodes:
            await self.redis_client.hset(
                f"episode:{episode.episode_id}",
                mapping={
                    "session_id": episode.session_id,
                    "user_id": episode.user_id,
                    "episode_type": episode.episode_type.value,
                    "timestamp": episode.timestamp.isoformat(),
                    "content": episode.content,
                    "emotional_state": episode.emotional_state.value,
                    "importance": episode.importance,
                    "context": json.dumps(episode.context),
                    "tags": json.dumps(episode.tags)
                }
            )
            
        logger.info(f"Consolidated {len(episodes)} episodes for session {session_id}")
        
    async def _generate_recommendations(self, session_id: str, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on session insights"""
        recommendations = []
        
        emotional_patterns = insights.get("emotional_patterns", {})
        pattern = emotional_patterns.get("pattern", "stable")
        
        if pattern == "declining":
            recommendations.append("Consider implementing more frequent emotional check-ins")
            recommendations.append("Explore additional coping strategies for difficult emotions")
            
        elif pattern == "improving":
            recommendations.append("Continue with current therapeutic approach")
            recommendations.append("Consider introducing more advanced techniques")
            
        elif pattern == "volatile":
            recommendations.append("Focus on emotional regulation techniques")
            recommendations.append("Implement grounding exercises during sessions")
            
        # Add technique-specific recommendations
        technique_effectiveness = insights.get("technique_effectiveness", {})
        if technique_effectiveness:
            best_technique = max(technique_effectiveness.items(), key=lambda x: x[1])
            recommendations.append(f"Continue using {best_technique[0]} - showing good results")
            
        return recommendations
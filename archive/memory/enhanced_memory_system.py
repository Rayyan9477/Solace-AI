"""
Enhanced Memory System for Better Contextual Recall and Therapeutic Insights

This module implements an advanced memory system that stores therapeutic insights,
tracks progress over time, builds comprehensive user profiles, and ensures
seamless session continuity with milestone recognition.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import json
import numpy as np
from collections import defaultdict, deque
import hashlib
import os
import warnings

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger
from ..models.llm import get_llm

logger = get_logger(__name__)


class MemoryJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for memory system dataclasses.

    Handles serialization of:
    - datetime objects (ISO format)
    - dataclasses (via asdict)
    - numpy arrays (via tolist)
    - defaultdict (via dict conversion)

    Security: This replaces pickle serialization to prevent CWE-502
    (Deserialization of Untrusted Data) vulnerabilities.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        elif isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "value": obj.tolist()}
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, '__dataclass_fields__'):
            # Handle dataclasses
            data = asdict(obj)
            data["__dataclass__"] = obj.__class__.__name__
            return data
        elif isinstance(obj, defaultdict):
            return {"__defaultdict__": True, "value": dict(obj)}
        elif isinstance(obj, deque):
            return {"__deque__": True, "value": list(obj)}
        return super().default(obj)


def _decode_memory_object(obj: Dict[str, Any]) -> Any:
    """
    Decode JSON objects back to their original types.

    This function is used as object_hook in json.load() to reconstruct
    datetime objects, numpy arrays, and dataclasses from their JSON
    representation.

    Security: Only reconstructs known safe types - no arbitrary code execution.
    """
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["value"])
    elif "__ndarray__" in obj:
        return np.array(obj["value"])
    elif "__defaultdict__" in obj:
        return defaultdict(list, obj["value"])
    elif "__deque__" in obj:
        return deque(obj["value"])
    elif "__dataclass__" in obj:
        dataclass_type = obj.pop("__dataclass__")
        # Reconstruct known dataclasses only (whitelist approach for security)
        if dataclass_type == "TherapeuticInsight":
            # Convert datetime fields
            if "timestamp" in obj and isinstance(obj["timestamp"], str):
                obj["timestamp"] = datetime.fromisoformat(obj["timestamp"])
            return TherapeuticInsight(**obj)
        elif dataclass_type == "ProgressMilestone":
            if "achieved_at" in obj and isinstance(obj["achieved_at"], str):
                obj["achieved_at"] = datetime.fromisoformat(obj["achieved_at"])
            return ProgressMilestone(**obj)
        elif dataclass_type == "SessionMemory":
            if "start_time" in obj and isinstance(obj["start_time"], str):
                obj["start_time"] = datetime.fromisoformat(obj["start_time"])
            if "end_time" in obj and isinstance(obj["end_time"], str):
                obj["end_time"] = datetime.fromisoformat(obj["end_time"])
            return SessionMemory(**obj)
        elif dataclass_type == "UserProfileMemory":
            if "created_at" in obj and isinstance(obj["created_at"], str):
                obj["created_at"] = datetime.fromisoformat(obj["created_at"])
            if "last_updated" in obj and isinstance(obj["last_updated"], str):
                obj["last_updated"] = datetime.fromisoformat(obj["last_updated"])
            return UserProfileMemory(**obj)
        else:
            # Unknown dataclass type - return as dict (safe fallback)
            logger.warning(f"Unknown dataclass type during deserialization: {dataclass_type}")
            return obj
    return obj


@dataclass
class TherapeuticInsight:
    """Record of a therapeutic insight or breakthrough"""
    insight_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    insight_type: str  # breakthrough, pattern_recognition, coping_strategy, etc.
    content: str
    context: Dict[str, Any]
    triggers: List[str]
    emotional_state: str
    significance_score: float  # 0.0 to 1.0
    therapist_notes: str
    user_acknowledgment: bool
    follow_up_actions: List[str]
    outcome_tracking: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProgressMilestone:
    """Significant progress milestone"""
    milestone_id: str
    user_id: str
    milestone_type: str  # symptom_improvement, skill_mastery, goal_achievement
    title: str
    description: str
    achieved_at: datetime
    baseline_data: Dict[str, Any]
    current_data: Dict[str, Any]
    improvement_metrics: Dict[str, float]
    celebration_status: str  # pending, acknowledged, celebrated
    related_insights: List[str]
    next_goals: List[str]

@dataclass
class SessionMemory:
    """Memory of a therapeutic session"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    session_type: str
    topics_discussed: List[str]
    insights_gained: List[str]
    interventions_used: List[str]
    user_mood_start: str
    user_mood_end: Optional[str]
    progress_indicators: Dict[str, float]
    homework_assigned: List[str]
    follow_up_needed: bool
    session_quality_score: float
    key_quotes: List[str]
    session_summary: str

@dataclass
class UserProfileMemory:
    """Comprehensive user profile with memory"""
    user_id: str
    created_at: datetime
    last_updated: datetime
    
    # Basic information
    demographics: Dict[str, Any]
    preferences: Dict[str, Any]
    goals: List[str]
    
    # Clinical memory
    diagnosis_history: List[Dict[str, Any]]
    symptom_timeline: List[Dict[str, Any]]
    medication_history: List[Dict[str, Any]]
    therapy_history: List[Dict[str, Any]]
    
    # Progress tracking
    baseline_assessments: Dict[str, Any]
    progress_metrics: Dict[str, List[float]]
    milestones_achieved: List[str]
    setbacks_experienced: List[Dict[str, Any]]
    
    # Relationship and social context
    support_system: Dict[str, Any]
    relationship_patterns: List[str]
    social_triggers: List[str]
    
    # Coping and resilience
    coping_strategies: Dict[str, Any]  # strategy -> effectiveness_score
    stress_triggers: List[str]
    resilience_factors: List[str]
    
    # Therapeutic alliance
    therapeutic_preferences: Dict[str, Any]
    communication_style: str
    engagement_patterns: Dict[str, Any]
    
    # Memory consolidation
    significant_memories: List[str]  # IDs of important insights/sessions
    recurring_themes: Dict[str, int]
    unresolved_issues: List[str]

class EnhancedMemorySystem:
    """
    Advanced memory system for comprehensive therapeutic context retention,
    insight storage, and progress tracking with seamless session continuity.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the enhanced memory system"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # Memory storage
        self.therapeutic_insights = defaultdict(list)  # user_id -> insights
        self.progress_milestones = defaultdict(list)   # user_id -> milestones
        self.session_memories = defaultdict(list)      # user_id -> sessions
        self.user_profiles = {}                        # user_id -> profile
        
        # Memory management
        self.memory_cache = {}
        self.insight_index = {}  # For fast insight lookup
        self.session_index = {}  # For fast session lookup
        
        # Configuration
        self.max_cache_size = 1000
        self.insight_retention_days = 365
        self.session_retention_days = 180
        self.milestone_threshold = 0.7
        
        # Load persisted memory
        self._load_memory_data()
    
    async def store_therapeutic_insight(self,
                                      user_id: str,
                                      session_id: str,
                                      insight_type: str,
                                      content: str,
                                      context: Dict[str, Any],
                                      significance_score: float = 0.5) -> str:
        """
        Store a therapeutic insight with full context
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            insight_type: Type of insight
            content: Insight content
            context: Contextual information
            significance_score: Importance score (0.0 to 1.0)
            
        Returns:
            Insight ID
        """
        try:
            insight_id = self._generate_insight_id(user_id, content)
            
            # Extract triggers and emotional state from context
            triggers = self._extract_triggers(context)
            emotional_state = context.get("emotional_state", "unknown")
            
            # Generate therapist notes using LLM
            therapist_notes = await self._generate_therapist_notes(
                insight_type, content, context
            )
            
            # Create insight record
            insight = TherapeuticInsight(
                insight_id=insight_id,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.now(),
                insight_type=insight_type,
                content=content,
                context=context,
                triggers=triggers,
                emotional_state=emotional_state,
                significance_score=significance_score,
                therapist_notes=therapist_notes,
                user_acknowledgment=False,
                follow_up_actions=[]
            )
            
            # Store insight
            self.therapeutic_insights[user_id].append(insight)
            self.insight_index[insight_id] = insight
            
            # Update user profile
            await self._update_profile_with_insight(user_id, insight)
            
            # Check for milestone achievement
            await self._check_milestone_achievement(user_id, insight)
            
            # Store in vector database
            if self.vector_db:
                await self._store_insight_in_db(insight)
            
            # Consolidate memory if needed
            await self._consolidate_memory(user_id)
            
            self.logger.info(f"Stored therapeutic insight {insight_id} for user {user_id}")
            return insight_id
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error storing therapeutic insight: {str(e)}")
            return ""
    
    async def record_session_memory(self,
                                  user_id: str,
                                  session_id: str,
                                  session_data: Dict[str, Any]) -> bool:
        """
        Record comprehensive session memory
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            session_data: Complete session data
            
        Returns:
            Success status
        """
        try:
            # Extract key information
            topics_discussed = self._extract_topics(session_data)
            insights_gained = session_data.get("insights", [])
            interventions_used = session_data.get("interventions", [])
            key_quotes = self._extract_key_quotes(session_data)
            
            # Generate session summary
            session_summary = await self._generate_session_summary(session_data)
            
            # Calculate session quality score
            quality_score = self._calculate_session_quality(session_data)
            
            # Create session memory
            session_memory = SessionMemory(
                session_id=session_id,
                user_id=user_id,
                start_time=session_data.get("start_time", datetime.now()),
                end_time=session_data.get("end_time"),
                session_type=session_data.get("session_type", "therapy"),
                topics_discussed=topics_discussed,
                insights_gained=insights_gained,
                interventions_used=interventions_used,
                user_mood_start=session_data.get("mood_start", "unknown"),
                user_mood_end=session_data.get("mood_end"),
                progress_indicators=session_data.get("progress_indicators", {}),
                homework_assigned=session_data.get("homework", []),
                follow_up_needed=session_data.get("follow_up_needed", False),
                session_quality_score=quality_score,
                key_quotes=key_quotes,
                session_summary=session_summary
            )
            
            # Store session memory
            self.session_memories[user_id].append(session_memory)
            self.session_index[session_id] = session_memory
            
            # Update user profile
            await self._update_profile_with_session(user_id, session_memory)
            
            # Store in vector database
            if self.vector_db:
                await self._store_session_in_db(session_memory)
            
            self.logger.info(f"Recorded session memory {session_id} for user {user_id}")
            return True
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error recording session memory: {str(e)}")
            return False
    
    async def get_contextual_memory(self,
                                  user_id: str,
                                  context_type: str = "recent",
                                  lookback_days: int = 30) -> Dict[str, Any]:
        """
        Retrieve contextual memory for therapeutic continuity
        
        Args:
            user_id: User identifier
            context_type: Type of context (recent, significant, patterns, progress)
            lookback_days: Days to look back
            
        Returns:
            Contextual memory data
        """
        try:
            self.logger.info(f"Retrieving contextual memory for user {user_id}, type: {context_type}")
            
            # Get user profile
            profile = await self._get_or_create_profile(user_id)
            
            # Get relevant time window
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Retrieve based on context type
            if context_type == "recent":
                return await self._get_recent_context(user_id, cutoff_date)
            elif context_type == "significant":
                return await self._get_significant_context(user_id, cutoff_date)
            elif context_type == "patterns":
                return await self._get_pattern_context(user_id, cutoff_date)
            elif context_type == "progress":
                return await self._get_progress_context(user_id, cutoff_date)
            elif context_type == "comprehensive":
                return await self._get_comprehensive_context(user_id, cutoff_date)
            else:
                return await self._get_recent_context(user_id, cutoff_date)
                
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error retrieving contextual memory: {str(e)}")
            return {"error": str(e)}
    
    async def identify_recurring_themes(self,
                                      user_id: str,
                                      analysis_period: int = 90) -> Dict[str, Any]:
        """
        Identify recurring themes in user's therapeutic journey
        
        Args:
            user_id: User identifier
            analysis_period: Days to analyze
            
        Returns:
            Recurring themes analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=analysis_period)
            
            # Get all insights and sessions in period
            insights = [i for i in self.therapeutic_insights.get(user_id, []) 
                       if i.timestamp >= cutoff_date]
            sessions = [s for s in self.session_memories.get(user_id, [])
                       if s.start_time >= cutoff_date]
            
            # Extract themes from insights
            insight_themes = defaultdict(int)
            for insight in insights:
                themes = await self._extract_themes_from_text(insight.content)
                for theme in themes:
                    insight_themes[theme] += 1
            
            # Extract themes from session topics
            session_themes = defaultdict(int)
            for session in sessions:
                for topic in session.topics_discussed:
                    themes = await self._extract_themes_from_text(topic)
                    for theme in themes:
                        session_themes[theme] += 1
            
            # Combine and rank themes
            all_themes = defaultdict(int)
            for theme, count in insight_themes.items():
                all_themes[theme] += count * 2  # Weight insights higher
            for theme, count in session_themes.items():
                all_themes[theme] += count
            
            # Get top themes
            top_themes = sorted(all_themes.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Analyze theme evolution
            theme_evolution = await self._analyze_theme_evolution(user_id, top_themes, analysis_period)
            
            # Generate insights about themes
            theme_insights = await self._generate_theme_insights(top_themes, theme_evolution)
            
            return {
                "user_id": user_id,
                "analysis_period_days": analysis_period,
                "top_themes": dict(top_themes),
                "theme_evolution": theme_evolution,
                "insights": theme_insights,
                "total_insights_analyzed": len(insights),
                "total_sessions_analyzed": len(sessions)
            }
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error identifying recurring themes: {str(e)}")
            return {"error": str(e)}
    
    async def track_progress_milestones(self,
                                      user_id: str,
                                      metric_type: str = "all") -> Dict[str, Any]:
        """
        Track and celebrate progress milestones
        
        Args:
            user_id: User identifier
            metric_type: Type of metrics to track
            
        Returns:
            Progress milestone data
        """
        try:
            profile = await self._get_or_create_profile(user_id)
            milestones = self.progress_milestones.get(user_id, [])
            
            # Analyze current progress
            current_metrics = await self._calculate_current_metrics(user_id)
            baseline_metrics = profile.baseline_assessments
            
            # Check for new milestones
            new_milestones = await self._detect_new_milestones(
                user_id, baseline_metrics, current_metrics
            )
            
            # Store new milestones
            for milestone in new_milestones:
                milestones.append(milestone)
                self.progress_milestones[user_id] = milestones
                
                # Store in vector database
                if self.vector_db:
                    await self._store_milestone_in_db(milestone)
            
            # Generate progress summary
            progress_summary = await self._generate_progress_summary(
                user_id, milestones, current_metrics, baseline_metrics
            )
            
            return {
                "user_id": user_id,
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
                "milestones_achieved": len(milestones),
                "new_milestones": len(new_milestones),
                "recent_milestones": [asdict(m) for m in milestones[-5:]],
                "progress_summary": progress_summary,
                "celebration_pending": len([m for m in milestones if m.celebration_status == "pending"])
            }
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error tracking progress milestones: {str(e)}")
            return {"error": str(e)}
    
    async def get_session_continuity_context(self,
                                           user_id: str,
                                           current_session_id: str) -> Dict[str, Any]:
        """
        Get context for seamless session continuity
        
        Args:
            user_id: User identifier
            current_session_id: Current session identifier
            
        Returns:
            Session continuity context
        """
        try:
            # Get last session
            user_sessions = self.session_memories.get(user_id, [])
            if not user_sessions:
                return {"first_session": True, "context": {}}
            
            last_session = user_sessions[-1]
            
            # Get unresolved items from last session
            unresolved_items = []
            if last_session.follow_up_needed:
                unresolved_items.append("Follow-up from previous session needed")
            
            if last_session.homework_assigned:
                unresolved_items.extend([f"Homework: {hw}" for hw in last_session.homework_assigned])
            
            # Get recent insights that need follow-up
            recent_insights = [
                i for i in self.therapeutic_insights.get(user_id, [])
                if i.timestamp >= datetime.now() - timedelta(days=7) and not i.user_acknowledgment
            ]
            
            # Get progress since last session
            progress_updates = await self._get_progress_since_session(user_id, last_session.session_id)
            
            # Get mood trajectory
            mood_trajectory = self._get_mood_trajectory(user_id, days_back=14)
            
            # Generate continuity summary
            continuity_summary = await self._generate_continuity_summary(
                last_session, unresolved_items, recent_insights, progress_updates
            )
            
            return {
                "user_id": user_id,
                "last_session_date": last_session.start_time.isoformat(),
                "days_since_last_session": (datetime.now() - last_session.start_time).days,
                "last_session_summary": last_session.session_summary,
                "unresolved_items": unresolved_items,
                "recent_insights_count": len(recent_insights),
                "progress_updates": progress_updates,
                "mood_trajectory": mood_trajectory,
                "continuity_summary": continuity_summary,
                "suggested_opening": await self._generate_session_opening(
                    last_session, unresolved_items, progress_updates
                )
            }
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error getting session continuity context: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _generate_insight_id(self, user_id: str, content: str) -> str:
        """Generate unique insight ID"""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"insight_{user_id}_{timestamp}_{content_hash}"
    
    def _extract_triggers(self, context: Dict[str, Any]) -> List[str]:
        """Extract triggers from context"""
        triggers = []
        
        # Look for explicit triggers
        if "triggers" in context:
            triggers.extend(context["triggers"])
        
        # Extract from conversation
        if "conversation" in context:
            conversation = context["conversation"]
            trigger_keywords = [
                "stressed by", "triggered by", "upset by", "bothered by",
                "caused by", "due to", "because of"
            ]
            
            for keyword in trigger_keywords:
                if keyword in conversation.lower():
                    # Extract trigger phrase
                    parts = conversation.lower().split(keyword)
                    if len(parts) > 1:
                        trigger_phrase = parts[1].split('.')[0].strip()
                        triggers.append(trigger_phrase)
        
        return triggers[:5]  # Limit to top 5 triggers
    
    async def _generate_therapist_notes(self,
                                      insight_type: str,
                                      content: str,
                                      context: Dict[str, Any]) -> str:
        """Generate therapist notes using LLM"""
        try:
            prompt = f"""
            Generate brief therapist notes for this therapeutic insight:
            
            Type: {insight_type}
            Content: {content}
            Context: {json.dumps(context, default=str)}
            
            Notes should include:
            - Clinical significance
            - Therapeutic implications
            - Suggested follow-up actions
            
            Keep notes concise and professional.
            """
            
            notes = await self.llm.generate_response(prompt)
            return notes.strip()
            
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error generating therapist notes: {str(e)}")
            return f"Standard {insight_type} insight documented"
    
    async def _update_profile_with_insight(self, user_id: str, insight: TherapeuticInsight):
        """Update user profile with new insight"""
        profile = await self._get_or_create_profile(user_id)
        
        # Add to significant memories if high significance
        if insight.significance_score > 0.7:
            profile.significant_memories.append(insight.insight_id)
        
        # Update recurring themes
        themes = await self._extract_themes_from_text(insight.content)
        for theme in themes:
            profile.recurring_themes[theme] = profile.recurring_themes.get(theme, 0) + 1
        
        # Update unresolved issues
        if insight.insight_type == "problem_identification":
            profile.unresolved_issues.append(insight.content[:100])
        elif insight.insight_type == "resolution":
            # Remove resolved issues
            resolved_content = insight.content.lower()
            profile.unresolved_issues = [
                issue for issue in profile.unresolved_issues
                if not any(word in issue.lower() for word in resolved_content.split()[:5])
            ]
        
        profile.last_updated = datetime.now()
        self.user_profiles[user_id] = profile
    
    async def _check_milestone_achievement(self, user_id: str, insight: TherapeuticInsight):
        """Check if insight represents a milestone achievement"""
        if insight.significance_score < self.milestone_threshold:
            return
        
        # Define milestone types
        milestone_indicators = {
            "breakthrough": ["breakthrough", "realization", "suddenly understand", "aha moment"],
            "skill_mastery": ["confident", "mastered", "learned", "can now"],
            "symptom_improvement": ["feeling better", "less anxious", "improved", "progress"],
            "goal_achievement": ["achieved", "accomplished", "completed", "succeeded"]
        }
        
        content_lower = insight.content.lower()
        
        for milestone_type, indicators in milestone_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                await self._create_milestone(user_id, milestone_type, insight)
                break
    
    async def _create_milestone(self, user_id: str, milestone_type: str, insight: TherapeuticInsight):
        """Create a progress milestone"""
        milestone_id = f"milestone_{user_id}_{int(datetime.now().timestamp())}"
        
        # Get baseline data
        profile = await self._get_or_create_profile(user_id)
        baseline_data = profile.baseline_assessments
        
        # Calculate current data (simplified)
        current_data = await self._calculate_current_metrics(user_id)
        
        # Calculate improvement metrics
        improvement_metrics = {}
        for metric, baseline_value in baseline_data.items():
            if metric in current_data:
                improvement = current_data[metric] - baseline_value
                improvement_metrics[metric] = improvement
        
        milestone = ProgressMilestone(
            milestone_id=milestone_id,
            user_id=user_id,
            milestone_type=milestone_type,
            title=f"{milestone_type.replace('_', ' ').title()} Achievement",
            description=insight.content[:200],
            achieved_at=insight.timestamp,
            baseline_data=baseline_data,
            current_data=current_data,
            improvement_metrics=improvement_metrics,
            celebration_status="pending",
            related_insights=[insight.insight_id],
            next_goals=[]
        )
        
        self.progress_milestones[user_id].append(milestone)
    
    async def _get_or_create_profile(self, user_id: str) -> UserProfileMemory:
        """Get existing profile or create new one"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Create new profile
        profile = UserProfileMemory(
            user_id=user_id,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            demographics={},
            preferences={},
            goals=[],
            diagnosis_history=[],
            symptom_timeline=[],
            medication_history=[],
            therapy_history=[],
            baseline_assessments={},
            progress_metrics={},
            milestones_achieved=[],
            setbacks_experienced=[],
            support_system={},
            relationship_patterns=[],
            social_triggers=[],
            coping_strategies={},
            stress_triggers=[],
            resilience_factors=[],
            therapeutic_preferences={},
            communication_style="unknown",
            engagement_patterns={},
            significant_memories=[],
            recurring_themes={},
            unresolved_issues=[]
        )
        
        self.user_profiles[user_id] = profile
        return profile
    
    async def _get_recent_context(self, user_id: str, cutoff_date: datetime) -> Dict[str, Any]:
        """Get recent context for continuity"""
        # Get recent insights
        recent_insights = [
            i for i in self.therapeutic_insights.get(user_id, [])
            if i.timestamp >= cutoff_date
        ]
        
        # Get recent sessions
        recent_sessions = [
            s for s in self.session_memories.get(user_id, [])
            if s.start_time >= cutoff_date
        ]
        
        # Get recent milestones
        recent_milestones = [
            m for m in self.progress_milestones.get(user_id, [])
            if m.achieved_at >= cutoff_date
        ]
        
        return {
            "recent_insights": [asdict(i) for i in recent_insights[-5:]],
            "recent_sessions": [asdict(s) for s in recent_sessions[-3:]],
            "recent_milestones": [asdict(m) for m in recent_milestones],
            "context_type": "recent",
            "timeframe": f"Last {(datetime.now() - cutoff_date).days} days"
        }
    
    async def _get_significant_context(self, user_id: str, cutoff_date: datetime) -> Dict[str, Any]:
        """Get significant/breakthrough context"""
        # Get high-significance insights
        significant_insights = [
            i for i in self.therapeutic_insights.get(user_id, [])
            if i.significance_score > 0.7 and i.timestamp >= cutoff_date
        ]
        
        # Get breakthrough moments
        breakthrough_insights = [
            i for i in significant_insights
            if i.insight_type == "breakthrough"
        ]
        
        return {
            "significant_insights": [asdict(i) for i in significant_insights],
            "breakthrough_moments": [asdict(i) for i in breakthrough_insights],
            "context_type": "significant"
        }
    
    async def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from text using LLM"""
        try:
            prompt = f"""
            Extract 3-5 key psychological/therapeutic themes from this text:
            
            "{text}"
            
            Return only the theme names, one per line.
            Focus on therapeutic concepts, emotions, coping strategies, relationships, etc.
            """
            
            response = await self.llm.generate_response(prompt)
            themes = [line.strip() for line in response.split('\n') if line.strip()]
            return themes[:5]
            
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error extracting themes: {str(e)}")
            return []
    
    def _extract_topics(self, session_data: Dict[str, Any]) -> List[str]:
        """Extract discussion topics from session data"""
        topics = []
        
        # Get explicit topics
        if "topics" in session_data:
            topics.extend(session_data["topics"])
        
        # Extract from conversation
        if "conversation" in session_data:
            # Simple keyword extraction
            conversation = session_data["conversation"]
            topic_keywords = [
                "anxiety", "depression", "stress", "relationships", "work",
                "family", "health", "goals", "fears", "anger", "sadness"
            ]
            
            for keyword in topic_keywords:
                if keyword in conversation.lower():
                    topics.append(keyword)
        
        return list(set(topics))  # Remove duplicates
    
    def _extract_key_quotes(self, session_data: Dict[str, Any]) -> List[str]:
        """Extract key quotes from session"""
        quotes = []
        
        if "conversation" in session_data:
            conversation = session_data["conversation"]
            # Look for impactful statements
            sentences = conversation.split('.')
            
            for sentence in sentences:
                # Look for emotional or significant statements
                if any(word in sentence.lower() for word in [
                    "realize", "understand", "feel", "think", "believe",
                    "breakthrough", "insight", "important", "significant"
                ]):
                    if len(sentence.strip()) > 20:  # Meaningful length
                        quotes.append(sentence.strip())
        
        return quotes[:5]  # Limit to top 5
    
    async def _generate_session_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate session summary using LLM"""
        try:
            prompt = f"""
            Generate a concise therapeutic session summary based on this data:
            
            {json.dumps(session_data, default=str)}
            
            Include:
            - Main topics discussed
            - Key insights or breakthroughs
            - Therapeutic interventions used
            - Progress indicators
            - Next steps
            
            Keep it professional and concise (3-4 sentences).
            """
            
            summary = await self.llm.generate_response(prompt)
            return summary.strip()
            
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error generating session summary: {str(e)}")
            return "Session completed with therapeutic discussion and intervention."
    
    def _calculate_session_quality(self, session_data: Dict[str, Any]) -> float:
        """Calculate session quality score"""
        score = 0.5  # Base score
        
        # Factors that increase quality
        if session_data.get("insights", []):
            score += 0.2
        
        if session_data.get("user_engagement", 0) > 0.7:
            score += 0.1
        
        if session_data.get("interventions", []):
            score += 0.1
        
        if session_data.get("mood_improvement", 0) > 0:
            score += 0.1
        
        # Duration factor
        duration = session_data.get("duration_minutes", 30)
        if 20 <= duration <= 60:  # Optimal range
            score += 0.1
        
        return min(1.0, score)
    
    async def _calculate_current_metrics(self, user_id: str) -> Dict[str, float]:
        """Calculate current metrics for progress tracking"""
        # This would integrate with assessment tools
        # For now, return sample metrics
        return {
            "anxiety_level": 0.4,
            "depression_level": 0.3,
            "coping_effectiveness": 0.7,
            "social_engagement": 0.6,
            "goal_progress": 0.5
        }
    
    async def _detect_new_milestones(self,
                                   user_id: str,
                                   baseline_metrics: Dict[str, float],
                                   current_metrics: Dict[str, float]) -> List[ProgressMilestone]:
        """Detect new milestone achievements"""
        milestones = []
        
        # Check for significant improvements
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric, 0.5)
            improvement = current_value - baseline_value
            
            # Check if improvement is significant enough for milestone
            if improvement > 0.3:  # 30% improvement threshold
                milestone = ProgressMilestone(
                    milestone_id=f"milestone_{user_id}_{metric}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    milestone_type="symptom_improvement",
                    title=f"{metric.replace('_', ' ').title()} Improvement",
                    description=f"Significant improvement in {metric}: {improvement:.1%}",
                    achieved_at=datetime.now(),
                    baseline_data={metric: baseline_value},
                    current_data={metric: current_value},
                    improvement_metrics={metric: improvement},
                    celebration_status="pending",
                    related_insights=[],
                    next_goals=[]
                )
                milestones.append(milestone)
        
        return milestones
    
    def _get_mood_trajectory(self, user_id: str, days_back: int) -> Dict[str, Any]:
        """Get mood trajectory over time"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Get sessions with mood data
        sessions_with_mood = [
            s for s in self.session_memories.get(user_id, [])
            if s.start_time >= cutoff_date and s.user_mood_start != "unknown"
        ]
        
        if not sessions_with_mood:
            return {"trend": "no_data"}
        
        # Analyze mood trend
        mood_scores = []
        for session in sessions_with_mood:
            # Convert mood to numeric score (simplified)
            mood_map = {
                "very_positive": 1.0, "positive": 0.8, "neutral": 0.5,
                "negative": 0.2, "very_negative": 0.0
            }
            score = mood_map.get(session.user_mood_start, 0.5)
            mood_scores.append(score)
        
        # Calculate trend
        if len(mood_scores) >= 2:
            trend_slope = np.polyfit(range(len(mood_scores)), mood_scores, 1)[0]
            if trend_slope > 0.1:
                trend = "improving"
            elif trend_slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "current_mood": mood_scores[-1] if mood_scores else 0.5,
            "average_mood": np.mean(mood_scores) if mood_scores else 0.5,
            "sessions_analyzed": len(sessions_with_mood)
        }
    
    async def _consolidate_memory(self, user_id: str):
        """Consolidate memory to prevent overflow"""
        # Clean up old, low-significance insights
        cutoff_date = datetime.now() - timedelta(days=self.insight_retention_days)
        
        user_insights = self.therapeutic_insights.get(user_id, [])
        retained_insights = [
            i for i in user_insights
            if i.timestamp >= cutoff_date or i.significance_score > 0.7
        ]
        
        self.therapeutic_insights[user_id] = retained_insights
        
        # Clean up old sessions
        session_cutoff = datetime.now() - timedelta(days=self.session_retention_days)
        user_sessions = self.session_memories.get(user_id, [])
        retained_sessions = [
            s for s in user_sessions
            if s.start_time >= session_cutoff or s.session_quality_score > 0.8
        ]
        
        self.session_memories[user_id] = retained_sessions
    
    # Storage methods
    
    async def _store_insight_in_db(self, insight: TherapeuticInsight):
        """Store insight in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "therapeutic_insight",
                "user_id": insight.user_id,
                "insight_type": insight.insight_type,
                "content": insight.content,
                "significance_score": insight.significance_score,
                "timestamp": insight.timestamp.isoformat(),
                "context": insight.context
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=insight.insight_id
            )
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error storing insight in database: {str(e)}")
    
    async def _store_session_in_db(self, session: SessionMemory):
        """Store session in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "session_memory",
                "user_id": session.user_id,
                "session_type": session.session_type,
                "topics_discussed": session.topics_discussed,
                "session_quality_score": session.session_quality_score,
                "timestamp": session.start_time.isoformat(),
                "summary": session.session_summary
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=session.session_id
            )
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error storing session in database: {str(e)}")
    
    async def _store_milestone_in_db(self, milestone: ProgressMilestone):
        """Store milestone in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "progress_milestone",
                "user_id": milestone.user_id,
                "milestone_type": milestone.milestone_type,
                "title": milestone.title,
                "description": milestone.description,
                "improvement_metrics": milestone.improvement_metrics,
                "timestamp": milestone.achieved_at.isoformat()
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=milestone.milestone_id
            )
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error storing milestone in database: {str(e)}")
    
    def _load_memory_data(self):
        """
        Load persisted memory data from JSON files.

        Security: Uses JSON instead of pickle to prevent CWE-502
        (Deserialization of Untrusted Data) vulnerabilities.

        Legacy .pkl files are NOT loaded for security reasons.
        If legacy files exist, a warning is logged and fresh data is used.
        """
        try:
            data_dir = "src/data/memory_system"
            loaded_count = 0

            # Check for legacy pickle files and warn (but don't load - security risk)
            legacy_files = [
                f"{data_dir}/insights.pkl",
                f"{data_dir}/sessions.pkl",
                f"{data_dir}/profiles.pkl",
                f"{data_dir}/milestones.pkl"
            ]
            for legacy_file in legacy_files:
                if os.path.exists(legacy_file):
                    self.logger.warning(
                        f"Legacy pickle file detected: {legacy_file}. "
                        f"Pickle files are not loaded due to security concerns (CWE-502). "
                        f"Please manually migrate data to JSON format or delete the file."
                    )

            # Load insights from JSON
            insights_file = f"{data_dir}/insights.json"
            if os.path.exists(insights_file):
                with open(insights_file, "r", encoding="utf-8") as f:
                    data = json.load(f, object_hook=_decode_memory_object)
                    if isinstance(data, dict):
                        self.therapeutic_insights = defaultdict(list, data)
                        loaded_count += 1

            # Load sessions from JSON
            sessions_file = f"{data_dir}/sessions.json"
            if os.path.exists(sessions_file):
                with open(sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f, object_hook=_decode_memory_object)
                    if isinstance(data, dict):
                        self.session_memories = defaultdict(list, data)
                        loaded_count += 1

            # Load profiles from JSON
            profiles_file = f"{data_dir}/profiles.json"
            if os.path.exists(profiles_file):
                with open(profiles_file, "r", encoding="utf-8") as f:
                    data = json.load(f, object_hook=_decode_memory_object)
                    if isinstance(data, dict):
                        self.user_profiles = data
                        loaded_count += 1

            # Load milestones from JSON
            milestones_file = f"{data_dir}/milestones.json"
            if os.path.exists(milestones_file):
                with open(milestones_file, "r", encoding="utf-8") as f:
                    data = json.load(f, object_hook=_decode_memory_object)
                    if isinstance(data, dict):
                        self.progress_milestones = defaultdict(list, data)
                        loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"Successfully loaded {loaded_count} memory data files")
            else:
                self.logger.info("No existing memory data found, starting fresh")

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in memory data: {str(e)}")
            self.logger.warning("Starting with fresh memory data due to corrupted files")
        except (OSError, ValueError, TypeError, KeyError) as e:
            self.logger.warning(f"Could not load memory data, starting fresh: {str(e)}")

    def _persist_memory_data(self):
        """
        Persist memory data to disk using JSON format.

        Security: Uses JSON instead of pickle to prevent CWE-502
        (Deserialization of Untrusted Data) vulnerabilities.

        The data is serialized with a custom encoder that handles:
        - datetime objects
        - dataclasses
        - numpy arrays
        - defaultdict/deque
        """
        try:
            data_dir = "src/data/memory_system"
            os.makedirs(data_dir, exist_ok=True)

            # Save insights as JSON
            insights_file = f"{data_dir}/insights.json"
            with open(insights_file, "w", encoding="utf-8") as f:
                json.dump(
                    dict(self.therapeutic_insights),
                    f,
                    cls=MemoryJSONEncoder,
                    indent=2,
                    ensure_ascii=False
                )

            # Save sessions as JSON
            sessions_file = f"{data_dir}/sessions.json"
            with open(sessions_file, "w", encoding="utf-8") as f:
                json.dump(
                    dict(self.session_memories),
                    f,
                    cls=MemoryJSONEncoder,
                    indent=2,
                    ensure_ascii=False
                )

            # Save profiles as JSON
            profiles_file = f"{data_dir}/profiles.json"
            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.user_profiles,
                    f,
                    cls=MemoryJSONEncoder,
                    indent=2,
                    ensure_ascii=False
                )

            # Save milestones as JSON
            milestones_file = f"{data_dir}/milestones.json"
            with open(milestones_file, "w", encoding="utf-8") as f:
                json.dump(
                    dict(self.progress_milestones),
                    f,
                    cls=MemoryJSONEncoder,
                    indent=2,
                    ensure_ascii=False
                )

            self.logger.debug("Memory data persisted successfully")

        except (TypeError, ValueError) as e:
            self.logger.error(f"Serialization error persisting memory data: {str(e)}")
        except OSError as e:
            self.logger.error(f"File system error persisting memory data: {str(e)}")
        except (RuntimeError, AttributeError, KeyError) as e:
            self.logger.error(f"Unexpected error persisting memory data: {str(e)}")

    def __del__(self):
        """Ensure data is persisted when object is destroyed"""
        try:
            self._persist_memory_data()
        except (OSError, IOError, ValueError, TypeError, RuntimeError):
            # Silently ignore errors during cleanup - logging may not be available
            pass
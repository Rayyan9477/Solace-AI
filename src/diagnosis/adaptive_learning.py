"""
Adaptive Learning System for Intervention Effectiveness Tracking

This module implements a reinforcement learning system that learns from user interactions
to improve diagnostic accuracy and therapeutic effectiveness. It tracks which approaches
work best for each user and continuously improves the system based on outcomes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np
from collections import defaultdict, deque
import os

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger
from ..models.llm import get_llm

logger = get_logger(__name__)


def _json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return {"__datetime__": True, "value": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "value": obj.tolist()}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if hasattr(obj, '__dict__'):
        return {"__class__": obj.__class__.__name__, **obj.__dict__}
    raise TypeError(f"Type {type(obj)} not serializable")


def _json_deserial(obj: Dict[str, Any]) -> Any:
    """JSON deserializer for custom types."""
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["value"])
    if "__ndarray__" in obj:
        return np.array(obj["value"])
    return obj

@dataclass
class InterventionOutcome:
    """Record of an intervention and its outcome"""
    intervention_id: str
    user_id: str
    intervention_type: str
    intervention_content: str
    context: Dict[str, Any]  # User state, symptoms, etc.
    timestamp: datetime
    user_response: Optional[str] = None
    engagement_score: float = 0.0  # 0.0 to 1.0
    effectiveness_score: float = 0.0  # 0.0 to 1.0
    symptom_change: float = 0.0  # -1.0 (worse) to 1.0 (better)
    breakthrough_indicator: bool = False
    follow_up_completed: bool = False
    long_term_outcome: Optional[float] = None

@dataclass
class UserProfile:
    """Adaptive user profile for personalized interventions"""
    user_id: str
    intervention_preferences: Dict[str, float]  # intervention_type -> preference score
    response_patterns: Dict[str, Any]  # patterns in user responses
    effective_approaches: List[str]  # historically effective approaches
    ineffective_approaches: List[str]  # historically ineffective approaches
    optimal_timing: Dict[str, Any]  # best times for different interventions
    engagement_patterns: Dict[str, Any]  # when user is most/least engaged
    learning_rate: float  # how quickly to adapt to new information
    confidence_level: float  # confidence in current profile
    last_updated: datetime
    total_interactions: int

@dataclass
class LearningInsight:
    """Insight gained from the adaptive learning process"""
    insight_id: str
    insight_type: str  # pattern, correlation, prediction, recommendation
    description: str
    evidence: List[str]
    confidence: float
    actionable_recommendation: str
    applies_to: str  # user_specific, demographic, general
    discovered_at: datetime

class AdaptiveLearningEngine:
    """
    Reinforcement learning engine that tracks intervention effectiveness
    and continuously improves therapeutic approaches based on outcomes.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the adaptive learning engine"""
        self.vector_db = vector_db
        self.llm = get_llm()
        self.logger = get_logger(__name__)
        
        # Data storage
        self.intervention_outcomes = defaultdict(list)  # user_id -> outcomes
        self.user_profiles = {}  # user_id -> UserProfile
        self.global_patterns = {}  # cross-user patterns
        self.learning_insights = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.effectiveness_decay = 0.95  # for temporal discounting
        self.min_data_points = 5  # minimum interactions before making recommendations
        self.confidence_threshold = 0.7
        
        # Model storage
        self.model_cache = {}
        self.pattern_cache = {}
        
        # Load existing data
        self._load_persisted_data()
    
    async def record_intervention_outcome(self,
                                        intervention_id: str,
                                        user_id: str,
                                        intervention_type: str,
                                        intervention_content: str,
                                        context: Dict[str, Any],
                                        user_response: str = None,
                                        engagement_metrics: Dict[str, Any] = None) -> bool:
        """
        Record the outcome of a therapeutic intervention
        
        Args:
            intervention_id: Unique identifier for the intervention
            user_id: User identifier
            intervention_type: Type of intervention (CBT, DBT, supportive, etc.)
            intervention_content: The actual intervention content
            context: Context when intervention was applied
            user_response: User's response to the intervention
            engagement_metrics: Metrics about user engagement
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Recording intervention outcome for user {user_id}")
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(
                user_response, engagement_metrics
            )
            
            # Calculate initial effectiveness score
            effectiveness_score = self._calculate_effectiveness_score(
                user_response, context, engagement_metrics
            )
            
            # Create outcome record
            outcome = InterventionOutcome(
                intervention_id=intervention_id,
                user_id=user_id,
                intervention_type=intervention_type,
                intervention_content=intervention_content,
                context=context,
                timestamp=datetime.now(),
                user_response=user_response,
                engagement_score=engagement_score,
                effectiveness_score=effectiveness_score,
                symptom_change=0.0,  # Will be updated with follow-up data
                breakthrough_indicator=self._detect_breakthrough(user_response, context),
                follow_up_completed=False
            )
            
            # Store outcome
            self.intervention_outcomes[user_id].append(outcome)
            
            # Update user profile
            await self._update_user_profile(user_id, outcome)
            
            # Update global patterns
            await self._update_global_patterns(outcome)
            
            # Store in vector database
            if self.vector_db:
                await self._store_outcome_in_db(outcome)
            
            # Trigger learning if enough data
            if len(self.intervention_outcomes[user_id]) >= self.min_data_points:
                await self._trigger_learning_update(user_id)
            
            # Persist data
            self._persist_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording intervention outcome: {str(e)}")
            return False
    
    async def get_personalized_recommendation(self,
                                            user_id: str,
                                            current_context: Dict[str, Any],
                                            available_interventions: List[str]) -> Dict[str, Any]:
        """
        Get personalized intervention recommendation based on learning
        
        Args:
            user_id: User identifier
            current_context: Current user context and state
            available_interventions: List of available intervention types
            
        Returns:
            Recommendation with confidence scores
        """
        try:
            self.logger.info(f"Generating personalized recommendation for user {user_id}")
            
            # Get or create user profile
            if user_id not in self.user_profiles:
                await self._initialize_user_profile(user_id)
            
            user_profile = self.user_profiles[user_id]
            
            # Score each available intervention
            intervention_scores = {}
            for intervention in available_interventions:
                score = await self._score_intervention(
                    intervention, user_id, current_context, user_profile
                )
                intervention_scores[intervention] = score
            
            # Rank interventions
            ranked_interventions = sorted(
                intervention_scores.items(), 
                key=lambda x: x[1]['total_score'], 
                reverse=True
            )
            
            # Get contextual factors
            contextual_factors = self._analyze_contextual_factors(
                current_context, user_profile
            )
            
            # Generate explanation
            explanation = await self._generate_recommendation_explanation(
                ranked_interventions[0] if ranked_interventions else None,
                user_profile, contextual_factors
            )
            
            return {
                "recommended_intervention": ranked_interventions[0][0] if ranked_interventions else None,
                "confidence": ranked_interventions[0][1]['confidence'] if ranked_interventions else 0.0,
                "all_scores": dict(ranked_interventions),
                "reasoning": explanation,
                "contextual_factors": contextual_factors,
                "user_profile_confidence": user_profile.confidence_level,
                "personalization_level": self._calculate_personalization_level(user_profile)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return {
                "recommended_intervention": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def analyze_intervention_patterns(self,
                                          user_id: str = None,
                                          time_period: int = 30) -> Dict[str, Any]:
        """
        Analyze patterns in intervention effectiveness
        
        Args:
            user_id: Specific user to analyze (None for global analysis)
            time_period: Days to look back for analysis
            
        Returns:
            Pattern analysis results
        """
        try:
            self.logger.info(f"Analyzing intervention patterns for {'user ' + user_id if user_id else 'global'}")
            
            # Get relevant outcomes
            if user_id:
                outcomes = self.intervention_outcomes.get(user_id, [])
            else:
                outcomes = []
                for user_outcomes in self.intervention_outcomes.values():
                    outcomes.extend(user_outcomes)
            
            # Filter by time period
            cutoff_date = datetime.now() - timedelta(days=time_period)
            recent_outcomes = [o for o in outcomes if o.timestamp >= cutoff_date]
            
            if not recent_outcomes:
                return {"error": "Insufficient data for analysis"}
            
            # Analyze patterns
            effectiveness_by_type = defaultdict(list)
            engagement_by_type = defaultdict(list)
            breakthrough_by_type = defaultdict(int)
            context_patterns = defaultdict(list)
            
            for outcome in recent_outcomes:
                effectiveness_by_type[outcome.intervention_type].append(outcome.effectiveness_score)
                engagement_by_type[outcome.intervention_type].append(outcome.engagement_score)
                if outcome.breakthrough_indicator:
                    breakthrough_by_type[outcome.intervention_type] += 1
                
                # Analyze context patterns
                for key, value in outcome.context.items():
                    context_patterns[f"{outcome.intervention_type}_{key}"].append(value)
            
            # Calculate statistics
            analysis_results = {
                "total_interventions": len(recent_outcomes),
                "time_period_days": time_period,
                "intervention_effectiveness": {},
                "intervention_engagement": {},
                "breakthrough_rates": {},
                "context_correlations": {},
                "trends": {},
                "insights": []
            }
            
            # Effectiveness analysis
            for intervention_type, scores in effectiveness_by_type.items():
                analysis_results["intervention_effectiveness"][intervention_type] = {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "count": len(scores),
                    "trend": self._calculate_trend(scores)
                }
            
            # Engagement analysis
            for intervention_type, scores in engagement_by_type.items():
                analysis_results["intervention_engagement"][intervention_type] = {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "count": len(scores)
                }
            
            # Breakthrough analysis
            for intervention_type, count in breakthrough_by_type.items():
                total_interventions = len(effectiveness_by_type[intervention_type])
                analysis_results["breakthrough_rates"][intervention_type] = {
                    "rate": count / total_interventions if total_interventions > 0 else 0,
                    "count": count,
                    "total": total_interventions
                }
            
            # Generate insights
            insights = await self._generate_insights_from_patterns(analysis_results)
            analysis_results["insights"] = insights
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
            return {"error": str(e)}
    
    async def update_long_term_outcome(self,
                                     intervention_id: str,
                                     user_id: str,
                                     outcome_score: float,
                                     follow_up_data: Dict[str, Any] = None) -> bool:
        """
        Update long-term outcome for an intervention
        
        Args:
            intervention_id: ID of the intervention to update
            user_id: User identifier
            outcome_score: Long-term outcome score (0.0 to 1.0)
            follow_up_data: Additional follow-up information
            
        Returns:
            Success status
        """
        try:
            # Find the intervention outcome
            user_outcomes = self.intervention_outcomes.get(user_id, [])
            outcome_to_update = None
            
            for outcome in user_outcomes:
                if outcome.intervention_id == intervention_id:
                    outcome_to_update = outcome
                    break
            
            if not outcome_to_update:
                self.logger.warning(f"Intervention {intervention_id} not found for user {user_id}")
                return False
            
            # Update outcome
            outcome_to_update.long_term_outcome = outcome_score
            outcome_to_update.follow_up_completed = True
            
            # Calculate symptom change if follow-up data available
            if follow_up_data and "initial_symptoms" in outcome_to_update.context:
                current_symptoms = follow_up_data.get("current_symptoms", {})
                initial_symptoms = outcome_to_update.context["initial_symptoms"]
                outcome_to_update.symptom_change = self._calculate_symptom_change(
                    initial_symptoms, current_symptoms
                )
            
            # Re-evaluate effectiveness with long-term data
            outcome_to_update.effectiveness_score = self._combine_effectiveness_scores(
                outcome_to_update.effectiveness_score, outcome_score
            )
            
            # Update user profile with new learning
            await self._update_user_profile_with_outcome(user_id, outcome_to_update)
            
            # Update global patterns
            await self._update_global_patterns(outcome_to_update)
            
            # Store updated outcome
            if self.vector_db:
                await self._store_outcome_in_db(outcome_to_update)
            
            self.logger.info(f"Updated long-term outcome for intervention {intervention_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating long-term outcome: {str(e)}")
            return False
    
    # Private helper methods
    
    def _calculate_engagement_score(self,
                                  user_response: str,
                                  engagement_metrics: Dict[str, Any]) -> float:
        """Calculate engagement score from user response and metrics"""
        
        if not user_response and not engagement_metrics:
            return 0.0
        
        score = 0.0
        
        # Response-based scoring
        if user_response:
            response_length = len(user_response.split())
            if response_length > 50:
                score += 0.4
            elif response_length > 20:
                score += 0.3
            elif response_length > 5:
                score += 0.2
            
            # Engagement indicators in response
            engagement_indicators = [
                "thank you", "helpful", "understand", "makes sense",
                "want to try", "will do", "good idea", "interesting"
            ]
            
            response_lower = user_response.lower()
            for indicator in engagement_indicators:
                if indicator in response_lower:
                    score += 0.1
        
        # Metrics-based scoring
        if engagement_metrics:
            if engagement_metrics.get("response_time_seconds", 0) < 300:  # Quick response
                score += 0.2
            
            if engagement_metrics.get("follow_up_questions", 0) > 0:
                score += 0.3
            
            if engagement_metrics.get("session_duration_minutes", 0) > 10:
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_effectiveness_score(self,
                                     user_response: str,
                                     context: Dict[str, Any],
                                     engagement_metrics: Dict[str, Any]) -> float:
        """Calculate effectiveness score from immediate response"""
        
        if not user_response:
            return 0.0
        
        score = 0.0
        response_lower = user_response.lower()
        
        # Positive effectiveness indicators
        positive_indicators = [
            "feel better", "helps", "clearer", "understand", "insight",
            "realize", "makes sense", "feel heard", "supported", "hopeful"
        ]
        
        for indicator in positive_indicators:
            if indicator in response_lower:
                score += 0.15
        
        # Negative effectiveness indicators
        negative_indicators = [
            "doesn't help", "confused", "worse", "don't understand",
            "not helpful", "disagree", "wrong"
        ]
        
        for indicator in negative_indicators:
            if indicator in response_lower:
                score -= 0.2
        
        # Breakthrough indicators
        breakthrough_indicators = [
            "never thought", "first time", "revelation", "breakthrough",
            "aha", "suddenly clear", "now i see"
        ]
        
        for indicator in breakthrough_indicators:
            if indicator in response_lower:
                score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def _detect_breakthrough(self, user_response: str, context: Dict[str, Any]) -> bool:
        """Detect if user had a breakthrough moment"""
        
        if not user_response:
            return False
        
        breakthrough_phrases = [
            "never realized", "first time i", "suddenly understand",
            "breakthrough", "revelation", "aha moment", "now i see",
            "makes total sense", "completely different", "eye opening"
        ]
        
        response_lower = user_response.lower()
        return any(phrase in response_lower for phrase in breakthrough_phrases)
    
    async def _update_user_profile(self, user_id: str, outcome: InterventionOutcome):
        """Update user profile based on intervention outcome"""
        
        if user_id not in self.user_profiles:
            await self._initialize_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # Update intervention preferences
        intervention_type = outcome.intervention_type
        current_pref = profile.intervention_preferences.get(intervention_type, 0.5)
        
        # Use effectiveness score to update preference
        new_pref = current_pref + self.learning_rate * (outcome.effectiveness_score - current_pref)
        profile.intervention_preferences[intervention_type] = new_pref
        
        # Update effective/ineffective approaches
        if outcome.effectiveness_score > 0.7:
            if intervention_type not in profile.effective_approaches:
                profile.effective_approaches.append(intervention_type)
        elif outcome.effectiveness_score < 0.3:
            if intervention_type not in profile.ineffective_approaches:
                profile.ineffective_approaches.append(intervention_type)
        
        # Update response patterns
        if outcome.user_response:
            self._update_response_patterns(profile, outcome)
        
        # Update confidence
        profile.total_interactions += 1
        profile.confidence_level = min(1.0, profile.total_interactions / 20)  # Max confidence after 20 interactions
        
        profile.last_updated = datetime.now()
    
    async def _update_global_patterns(self, outcome: InterventionOutcome):
        """Update global patterns across all users"""
        
        intervention_type = outcome.intervention_type
        
        if intervention_type not in self.global_patterns:
            self.global_patterns[intervention_type] = {
                "total_uses": 0,
                "total_effectiveness": 0.0,
                "total_engagement": 0.0,
                "breakthrough_count": 0,
                "context_patterns": defaultdict(list)
            }
        
        patterns = self.global_patterns[intervention_type]
        patterns["total_uses"] += 1
        patterns["total_effectiveness"] += outcome.effectiveness_score
        patterns["total_engagement"] += outcome.engagement_score
        
        if outcome.breakthrough_indicator:
            patterns["breakthrough_count"] += 1
        
        # Update context patterns
        for key, value in outcome.context.items():
            patterns["context_patterns"][key].append(value)
    
    async def _score_intervention(self,
                                intervention: str,
                                user_id: str,
                                current_context: Dict[str, Any],
                                user_profile: UserProfile) -> Dict[str, Any]:
        """Score an intervention for a specific user and context"""
        
        # Base preference score
        preference_score = user_profile.intervention_preferences.get(intervention, 0.5)
        
        # Context compatibility score
        context_score = await self._calculate_context_compatibility(
            intervention, current_context, user_profile
        )
        
        # Global effectiveness score
        global_score = self._get_global_effectiveness(intervention)
        
        # Recent performance score
        recent_score = self._get_recent_performance(intervention, user_id)
        
        # Combine scores with weights
        total_score = (
            0.4 * preference_score +
            0.3 * context_score +
            0.2 * global_score +
            0.1 * recent_score
        )
        
        # Calculate confidence
        confidence = min(1.0, user_profile.confidence_level * 0.7 + 0.3)
        
        return {
            "total_score": total_score,
            "preference_score": preference_score,
            "context_score": context_score,
            "global_score": global_score,
            "recent_score": recent_score,
            "confidence": confidence
        }
    
    async def _calculate_context_compatibility(self,
                                             intervention: str,
                                             current_context: Dict[str, Any],
                                             user_profile: UserProfile) -> float:
        """Calculate how well intervention fits current context"""
        
        # Get historical context patterns for this intervention
        if intervention not in self.global_patterns:
            return 0.5  # Default neutral score
        
        context_patterns = self.global_patterns[intervention]["context_patterns"]
        compatibility_score = 0.5
        
        # Compare current context to successful historical contexts
        for context_key, context_value in current_context.items():
            if context_key in context_patterns:
                historical_values = context_patterns[context_key]
                if historical_values:
                    # Calculate similarity to successful contexts
                    if isinstance(context_value, (int, float)):
                        mean_value = np.mean(historical_values)
                        std_value = np.std(historical_values) if len(historical_values) > 1 else 1.0
                        similarity = 1.0 - min(1.0, abs(context_value - mean_value) / (std_value + 0.1))
                        compatibility_score += 0.1 * similarity
                    elif isinstance(context_value, str):
                        match_rate = historical_values.count(context_value) / len(historical_values)
                        compatibility_score += 0.1 * match_rate
        
        return min(1.0, compatibility_score)
    
    def _get_global_effectiveness(self, intervention: str) -> float:
        """Get global effectiveness score for intervention"""
        
        if intervention not in self.global_patterns:
            return 0.5
        
        patterns = self.global_patterns[intervention]
        if patterns["total_uses"] == 0:
            return 0.5
        
        return patterns["total_effectiveness"] / patterns["total_uses"]
    
    def _get_recent_performance(self, intervention: str, user_id: str) -> float:
        """Get recent performance score for intervention with specific user"""
        
        user_outcomes = self.intervention_outcomes.get(user_id, [])
        
        # Get recent outcomes for this intervention type
        recent_cutoff = datetime.now() - timedelta(days=14)
        recent_outcomes = [
            o for o in user_outcomes 
            if o.intervention_type == intervention and o.timestamp >= recent_cutoff
        ]
        
        if not recent_outcomes:
            return 0.5
        
        return np.mean([o.effectiveness_score for o in recent_outcomes])
    
    async def _initialize_user_profile(self, user_id: str):
        """Initialize a new user profile"""
        
        profile = UserProfile(
            user_id=user_id,
            intervention_preferences={},
            response_patterns={},
            effective_approaches=[],
            ineffective_approaches=[],
            optimal_timing={},
            engagement_patterns={},
            learning_rate=self.learning_rate,
            confidence_level=0.0,
            last_updated=datetime.now(),
            total_interactions=0
        )
        
        self.user_profiles[user_id] = profile
    
    def _update_response_patterns(self, profile: UserProfile, outcome: InterventionOutcome):
        """Update response patterns in user profile"""
        
        if not outcome.user_response:
            return
        
        response_length = len(outcome.user_response.split())
        response_sentiment = self._analyze_response_sentiment(outcome.user_response)
        
        if "response_lengths" not in profile.response_patterns:
            profile.response_patterns["response_lengths"] = []
        
        profile.response_patterns["response_lengths"].append(response_length)
        
        if "sentiments" not in profile.response_patterns:
            profile.response_patterns["sentiments"] = []
        
        profile.response_patterns["sentiments"].append(response_sentiment)
        
        # Keep only recent patterns
        if len(profile.response_patterns["response_lengths"]) > 20:
            profile.response_patterns["response_lengths"] = profile.response_patterns["response_lengths"][-20:]
        if len(profile.response_patterns["sentiments"]) > 20:
            profile.response_patterns["sentiments"] = profile.response_patterns["sentiments"][-20:]
    
    def _analyze_response_sentiment(self, response: str) -> float:
        """Analyze sentiment of user response (-1.0 to 1.0)"""
        
        positive_words = [
            "good", "great", "helpful", "better", "thanks", "appreciate",
            "understand", "clear", "positive", "hopeful", "encouraged"
        ]
        
        negative_words = [
            "bad", "worse", "confused", "frustrated", "angry", "hopeless",
            "difficult", "hard", "struggling", "overwhelmed", "stuck"
        ]
        
        response_lower = response.lower()
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        total_words = len(response.split())
        
        if total_words == 0:
            return 0.0
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        return positive_ratio - negative_ratio
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from a series of scores"""
        
        if len(scores) < 3:
            return "insufficient_data"
        
        # Simple linear regression
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    async def _generate_insights_from_patterns(self, analysis_results: Dict[str, Any]) -> List[LearningInsight]:
        """Generate actionable insights from pattern analysis"""
        
        insights = []
        
        # Find most/least effective interventions
        effectiveness_data = analysis_results.get("intervention_effectiveness", {})
        if effectiveness_data:
            best_intervention = max(effectiveness_data.keys(), 
                                  key=lambda x: effectiveness_data[x]["mean_score"])
            worst_intervention = min(effectiveness_data.keys(), 
                                   key=lambda x: effectiveness_data[x]["mean_score"])
            
            insights.append(LearningInsight(
                insight_id=f"effectiveness_{int(datetime.now().timestamp())}",
                insight_type="recommendation",
                description=f"{best_intervention} shows highest effectiveness (avg: {effectiveness_data[best_intervention]['mean_score']:.2f})",
                evidence=[f"Based on {effectiveness_data[best_intervention]['count']} interventions"],
                confidence=0.8,
                actionable_recommendation=f"Prioritize {best_intervention} interventions",
                applies_to="general",
                discovered_at=datetime.now()
            ))
        
        # Find breakthrough patterns
        breakthrough_data = analysis_results.get("breakthrough_rates", {})
        if breakthrough_data:
            high_breakthrough = max(breakthrough_data.keys(),
                                  key=lambda x: breakthrough_data[x]["rate"])
            
            if breakthrough_data[high_breakthrough]["rate"] > 0.3:
                insights.append(LearningInsight(
                    insight_id=f"breakthrough_{int(datetime.now().timestamp())}",
                    insight_type="pattern",
                    description=f"{high_breakthrough} leads to breakthroughs {breakthrough_data[high_breakthrough]['rate']:.1%} of the time",
                    evidence=[f"{breakthrough_data[high_breakthrough]['count']} breakthroughs out of {breakthrough_data[high_breakthrough]['total']} uses"],
                    confidence=0.7,
                    actionable_recommendation=f"Use {high_breakthrough} when breakthrough moments are needed",
                    applies_to="general",
                    discovered_at=datetime.now()
                ))
        
        return insights
    
    def _calculate_symptom_change(self, 
                                initial_symptoms: Dict[str, float],
                                current_symptoms: Dict[str, float]) -> float:
        """Calculate overall symptom change score"""
        
        if not initial_symptoms or not current_symptoms:
            return 0.0
        
        changes = []
        for symptom, initial_score in initial_symptoms.items():
            if symptom in current_symptoms:
                change = initial_score - current_symptoms[symptom]  # Positive = improvement
                changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _combine_effectiveness_scores(self, immediate: float, long_term: float) -> float:
        """Combine immediate and long-term effectiveness scores"""
        return 0.3 * immediate + 0.7 * long_term  # Weight long-term more heavily
    
    async def _update_user_profile_with_outcome(self, user_id: str, outcome: InterventionOutcome):
        """Update user profile with long-term outcome data"""
        
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        intervention_type = outcome.intervention_type
        
        # Update preferences with long-term data
        if outcome.long_term_outcome is not None:
            current_pref = profile.intervention_preferences.get(intervention_type, 0.5)
            new_pref = current_pref + self.learning_rate * (outcome.long_term_outcome - current_pref)
            profile.intervention_preferences[intervention_type] = new_pref
        
        # Update effective/ineffective lists
        if outcome.long_term_outcome and outcome.long_term_outcome > 0.7:
            if intervention_type not in profile.effective_approaches:
                profile.effective_approaches.append(intervention_type)
            # Remove from ineffective if present
            if intervention_type in profile.ineffective_approaches:
                profile.ineffective_approaches.remove(intervention_type)
        elif outcome.long_term_outcome and outcome.long_term_outcome < 0.3:
            if intervention_type not in profile.ineffective_approaches:
                profile.ineffective_approaches.append(intervention_type)
            # Remove from effective if present
            if intervention_type in profile.effective_approaches:
                profile.effective_approaches.remove(intervention_type)
    
    async def _trigger_learning_update(self, user_id: str):
        """Trigger learning update when sufficient data is available"""
        
        # This could trigger more sophisticated ML model updates
        # For now, just log that learning is occurring
        self.logger.info(f"Triggering learning update for user {user_id}")
        
        # Could implement:
        # - Neural network updates
        # - Clustering analysis
        # - Pattern recognition algorithms
        # - Predictive modeling
    
    def _analyze_contextual_factors(self, 
                                   current_context: Dict[str, Any],
                                   user_profile: UserProfile) -> Dict[str, Any]:
        """Analyze contextual factors affecting intervention choice"""
        
        factors = {
            "emotional_state": current_context.get("emotional_state", "unknown"),
            "stress_level": current_context.get("stress_level", 0.5),
            "time_of_day": current_context.get("time_of_day", "unknown"),
            "session_length": current_context.get("session_length", 0),
            "previous_interventions": current_context.get("recent_interventions", []),
            "user_readiness": self._assess_user_readiness(current_context, user_profile)
        }
        
        return factors
    
    def _assess_user_readiness(self, context: Dict[str, Any], profile: UserProfile) -> float:
        """Assess user's readiness for interventions"""
        
        readiness = 0.5  # Base readiness
        
        # Adjust based on emotional state
        emotional_state = context.get("emotional_state", "stable")
        if emotional_state in ["crisis", "severe_distress"]:
            readiness = 0.2
        elif emotional_state in ["distressed", "anxious"]:
            readiness = 0.4
        elif emotional_state in ["stable", "motivated"]:
            readiness = 0.8
        
        # Adjust based on engagement patterns
        if profile.response_patterns.get("sentiments"):
            recent_sentiment = np.mean(profile.response_patterns["sentiments"][-5:])
            readiness += 0.2 * recent_sentiment
        
        return max(0.1, min(1.0, readiness))
    
    async def _generate_recommendation_explanation(self,
                                                 top_recommendation: Tuple[str, Dict],
                                                 user_profile: UserProfile,
                                                 contextual_factors: Dict[str, Any]) -> str:
        """Generate explanation for recommendation"""
        
        if not top_recommendation:
            return "Insufficient data for personalized recommendation"
        
        intervention, scores = top_recommendation
        
        explanation_parts = [
            f"Recommending {intervention} based on:",
            f"- Personal preference score: {scores['preference_score']:.2f}",
            f"- Context compatibility: {scores['context_score']:.2f}",
            f"- Global effectiveness: {scores['global_score']:.2f}"
        ]
        
        if intervention in user_profile.effective_approaches:
            explanation_parts.append(f"- Previously effective for you")
        
        if contextual_factors["user_readiness"] > 0.7:
            explanation_parts.append(f"- High readiness for intervention")
        
        return "\n".join(explanation_parts)
    
    def _calculate_personalization_level(self, user_profile: UserProfile) -> float:
        """Calculate how personalized recommendations can be"""
        
        # Based on amount of data available
        data_factor = min(1.0, user_profile.total_interactions / 20)
        confidence_factor = user_profile.confidence_level
        
        return (data_factor + confidence_factor) / 2
    
    async def _store_outcome_in_db(self, outcome: InterventionOutcome):
        """Store intervention outcome in vector database"""
        
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "intervention_outcome",
                "user_id": outcome.user_id,
                "intervention_type": outcome.intervention_type,
                "effectiveness_score": outcome.effectiveness_score,
                "engagement_score": outcome.engagement_score,
                "breakthrough_indicator": outcome.breakthrough_indicator,
                "timestamp": outcome.timestamp.isoformat(),
                "context": outcome.context
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=outcome.intervention_id
            )
            
        except Exception as e:
            self.logger.error(f"Error storing outcome in database: {str(e)}")
    
    def _persist_data(self):
        """Persist learning data to disk using JSON (CWE-502 fix: replaced pickle)"""

        try:
            data_dir = "src/data/adaptive_learning"
            os.makedirs(data_dir, exist_ok=True)

            # Save user profiles as JSON
            with open(f"{data_dir}/user_profiles.json", "w", encoding="utf-8") as f:
                json.dump(self.user_profiles, f, default=_json_serial, indent=2)

            # Save global patterns as JSON
            with open(f"{data_dir}/global_patterns.json", "w", encoding="utf-8") as f:
                json.dump(dict(self.global_patterns), f, default=_json_serial, indent=2)

            # Save intervention outcomes (recent only to manage size)
            recent_outcomes = {}
            cutoff = datetime.now() - timedelta(days=90)

            for user_id, outcomes in self.intervention_outcomes.items():
                recent_outcomes[user_id] = [
                    asdict(o) if hasattr(o, '__dataclass_fields__') else o
                    for o in outcomes if hasattr(o, 'timestamp') and o.timestamp >= cutoff
                ]

            with open(f"{data_dir}/recent_outcomes.json", "w", encoding="utf-8") as f:
                json.dump(recent_outcomes, f, default=_json_serial, indent=2)

        except Exception as e:
            self.logger.error(f"Error persisting data: {str(e)}")
    
    def _load_persisted_data(self):
        """Load persisted learning data from disk using JSON (CWE-502 fix: replaced pickle)"""

        try:
            data_dir = "src/data/adaptive_learning"

            # Load user profiles from JSON
            profiles_file = f"{data_dir}/user_profiles.json"
            if os.path.exists(profiles_file):
                with open(profiles_file, "r", encoding="utf-8") as f:
                    self.user_profiles = json.load(f, object_hook=_json_deserial)

            # Load global patterns from JSON
            patterns_file = f"{data_dir}/global_patterns.json"
            if os.path.exists(patterns_file):
                with open(patterns_file, "r", encoding="utf-8") as f:
                    loaded_patterns = json.load(f, object_hook=_json_deserial)
                    self.global_patterns = defaultdict(dict, loaded_patterns)

            # Load recent outcomes from JSON
            outcomes_file = f"{data_dir}/recent_outcomes.json"
            if os.path.exists(outcomes_file):
                with open(outcomes_file, "r", encoding="utf-8") as f:
                    loaded_outcomes = json.load(f, object_hook=_json_deserial)
                    self.intervention_outcomes = defaultdict(list, loaded_outcomes)

            self.logger.info("Successfully loaded persisted adaptive learning data")

        except Exception as e:
            self.logger.warning(f"Could not load persisted data, starting fresh: {str(e)}")
            # Start with fresh data structures
            self.user_profiles = {}
            self.global_patterns = {}
            self.intervention_outcomes = defaultdict(list)
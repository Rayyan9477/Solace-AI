"""
User Feedback Integration System for Continuous Improvement

This module implements comprehensive user feedback collection, analysis, and integration
systems for continuous improvement of therapeutic interventions and system performance.
It processes various types of feedback including explicit ratings, implicit behavioral
signals, and long-term outcome assessments.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import pickle
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from ..diagnosis.adaptive_learning import InterventionOutcome, UserProfile
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_BEHAVIORAL = "implicit_behavioral"
    TEXT_FEEDBACK = "text_feedback"
    SATISFACTION_SURVEY = "satisfaction_survey"
    OUTCOME_ASSESSMENT = "outcome_assessment"
    CRISIS_FEEDBACK = "crisis_feedback"
    ENGAGEMENT_METRIC = "engagement_metric"
    BREAKTHROUGH_INDICATOR = "breakthrough_indicator"

class FeedbackSentiment(Enum):
    """Sentiment classification for feedback"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class FeedbackEntry:
    """Individual feedback entry from a user"""
    feedback_id: str
    user_id: str
    session_id: str
    intervention_id: Optional[str]
    feedback_type: FeedbackType
    timestamp: datetime
    
    # Content
    rating_value: Optional[float] = None  # 1-10 scale
    text_content: Optional[str] = None
    structured_responses: Optional[Dict[str, Any]] = None
    behavioral_indicators: Optional[Dict[str, float]] = None
    
    # Analysis results
    sentiment: Optional[FeedbackSentiment] = None
    confidence: Optional[float] = None
    topics: Optional[List[str]] = None
    urgency_level: Optional[str] = None
    
    # Context
    context_at_feedback: Optional[Dict[str, Any]] = None
    agent_involved: Optional[str] = None
    intervention_type: Optional[str] = None
    
    # Processing metadata
    processed: bool = False
    processing_timestamp: Optional[datetime] = None
    processing_notes: Optional[str] = None

@dataclass
class FeedbackSummary:
    """Summary of feedback for a user or time period"""
    summary_id: str
    user_id: Optional[str]
    time_period_start: datetime
    time_period_end: datetime
    
    # Quantitative metrics
    total_feedback_count: int
    average_rating: float
    sentiment_distribution: Dict[str, int]
    feedback_type_distribution: Dict[str, int]
    
    # Qualitative insights
    common_themes: List[str]
    improvement_areas: List[str]
    positive_highlights: List[str]
    critical_issues: List[str]
    
    # Actionable recommendations
    recommended_actions: List[Dict[str, Any]]
    priority_level: str
    confidence_score: float
    
    # Metadata
    generated_at: datetime
    data_quality_score: float

@dataclass
class LearningUpdate:
    """Update to be applied based on feedback analysis"""
    update_id: str
    source_feedback_ids: List[str]
    target_component: str  # agent, intervention, system
    update_type: str  # parameter, weight, rule, strategy
    
    # Update details
    current_value: Any
    recommended_value: Any
    confidence: float
    expected_impact: float
    
    # Context
    reasoning: str
    supporting_evidence: List[str]
    risk_assessment: Dict[str, float]
    
    # Implementation
    priority: str
    estimated_effort: str
    requires_approval: bool
    expiry_date: datetime
    
    # Tracking
    created_at: datetime
    applied_at: Optional[datetime] = None
    application_result: Optional[Dict[str, Any]] = None

class SentimentAnalyzer:
    """Advanced sentiment analysis for user feedback"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Therapeutic domain-specific sentiment indicators
        self.positive_indicators = {
            'very_positive': [
                'breakthrough', 'amazing', 'life-changing', 'transformative',
                'incredibly helpful', 'profound', 'revolutionary', 'miraculous'
            ],
            'positive': [
                'helpful', 'better', 'improvement', 'progress', 'good', 'useful',
                'effective', 'beneficial', 'constructive', 'supportive', 'encouraging',
                'clear', 'understandable', 'insightful', 'valuable', 'appreciate'
            ]
        }
        
        self.negative_indicators = {
            'very_negative': [
                'horrible', 'devastating', 'traumatic', 'completely useless',
                'made everything worse', 'dangerous', 'harmful', 'terrible mistake'
            ],
            'negative': [
                'unhelpful', 'worse', 'decline', 'deterioration', 'bad', 'useless',
                'ineffective', 'harmful', 'destructive', 'unsupportive', 'discouraging',
                'confusing', 'unclear', 'pointless', 'waste', 'frustrated'
            ]
        }
        
        # Crisis/urgency indicators
        self.crisis_indicators = [
            'crisis', 'emergency', 'urgent', 'immediate help', 'can\'t cope',
            'breaking down', 'desperate', 'at my limit', 'overwhelmed',
            'self-harm', 'suicide', 'hurt myself', 'end it all'
        ]
        
        # Therapeutic progress indicators
        self.progress_indicators = [
            'understand better', 'making progress', 'feeling stronger',
            'coping better', 'tools are working', 'breakthrough moment',
            'perspective shift', 'emotional regulation', 'managing well'
        ]
    
    def analyze_sentiment(self, text: str, context: Dict[str, Any] = None) -> Tuple[FeedbackSentiment, float, Dict[str, Any]]:
        """Analyze sentiment with therapeutic context awareness"""
        
        if not text or not text.strip():
            return FeedbackSentiment.NEUTRAL, 0.0, {}
        
        text_lower = text.lower()
        analysis_details = {
            'crisis_detected': False,
            'progress_detected': False,
            'emotional_indicators': [],
            'therapeutic_themes': []
        }
        
        # Check for crisis indicators first
        crisis_matches = [indicator for indicator in self.crisis_indicators if indicator in text_lower]
        if crisis_matches:
            analysis_details['crisis_detected'] = True
            analysis_details['crisis_indicators'] = crisis_matches
            return FeedbackSentiment.VERY_NEGATIVE, 0.9, analysis_details
        
        # Check for progress indicators
        progress_matches = [indicator for indicator in self.progress_indicators if indicator in text_lower]
        if progress_matches:
            analysis_details['progress_detected'] = True
            analysis_details['progress_indicators'] = progress_matches
        
        # Calculate sentiment scores
        very_positive_score = sum(1 for indicator in self.positive_indicators['very_positive'] if indicator in text_lower)
        positive_score = sum(1 for indicator in self.positive_indicators['positive'] if indicator in text_lower)
        very_negative_score = sum(1 for indicator in self.negative_indicators['very_negative'] if indicator in text_lower)
        negative_score = sum(1 for indicator in self.negative_indicators['negative'] if indicator in text_lower)
        
        # Weight scores
        total_positive = very_positive_score * 3 + positive_score
        total_negative = very_negative_score * 3 + negative_score
        
        # Account for text length
        word_count = len(text.split())
        normalized_positive = total_positive / max(word_count, 10) * 100
        normalized_negative = total_negative / max(word_count, 10) * 100
        
        # Determine sentiment
        if very_positive_score > 0 or normalized_positive > 5:
            sentiment = FeedbackSentiment.VERY_POSITIVE if very_positive_score > 0 else FeedbackSentiment.POSITIVE
            confidence = min(0.9, 0.5 + normalized_positive / 10)
        elif very_negative_score > 0 or normalized_negative > 5:
            sentiment = FeedbackSentiment.VERY_NEGATIVE if very_negative_score > 0 else FeedbackSentiment.NEGATIVE
            confidence = min(0.9, 0.5 + normalized_negative / 10)
        elif abs(normalized_positive - normalized_negative) < 2:
            sentiment = FeedbackSentiment.NEUTRAL
            confidence = 0.6
        elif normalized_positive > normalized_negative:
            sentiment = FeedbackSentiment.POSITIVE
            confidence = min(0.8, 0.5 + (normalized_positive - normalized_negative) / 10)
        else:
            sentiment = FeedbackSentiment.NEGATIVE
            confidence = min(0.8, 0.5 + (normalized_negative - normalized_positive) / 10)
        
        # Add contextual adjustments
        if context:
            if context.get('intervention_effectiveness', 0) > 0.8 and sentiment in [FeedbackSentiment.NEUTRAL, FeedbackSentiment.POSITIVE]:
                # Boost positive sentiment for effective interventions
                if sentiment == FeedbackSentiment.NEUTRAL:
                    sentiment = FeedbackSentiment.POSITIVE
                confidence = min(0.9, confidence + 0.1)
            
            if context.get('user_crisis_history', False) and sentiment == FeedbackSentiment.NEGATIVE:
                # Be more sensitive to negative feedback for at-risk users
                confidence = min(0.9, confidence + 0.1)
                analysis_details['high_risk_user'] = True
        
        return sentiment, confidence, analysis_details

class TopicAnalyzer:
    """Analyze topics and themes in user feedback"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Therapeutic topic categories
        self.topic_patterns = {
            'anxiety_management': [
                r'\b(anxiety|anxious|panic|worry|worried|stress|overwhelm)\b',
                r'\b(breathing|relaxation|calm|grounding)\b'
            ],
            'depression_symptoms': [
                r'\b(depress|sad|hopeless|empty|numb|down)\b',
                r'\b(energy|motivation|sleep|appetite)\b'
            ],
            'coping_strategies': [
                r'\b(coping|strategy|technique|tool|method)\b',
                r'\b(manage|handle|deal with|work through)\b'
            ],
            'therapeutic_relationship': [
                r'\b(therapist|counselor|relationship|trust|safe)\b',
                r'\b(understand|listen|support|care)\b'
            ],
            'medication': [
                r'\b(medication|med|pill|dose|side effect)\b',
                r'\b(psychiatrist|doctor|prescription)\b'
            ],
            'family_relationships': [
                r'\b(family|parent|spouse|partner|child)\b',
                r'\b(relationship|communication|conflict)\b'
            ],
            'work_stress': [
                r'\b(work|job|career|boss|colleague)\b',
                r'\b(stress|pressure|deadline|performance)\b'
            ],
            'self_care': [
                r'\b(self.care|exercise|sleep|nutrition)\b',
                r'\b(routine|habit|wellness|health)\b'
            ]
        }
    
    def analyze_topics(self, text: str) -> List[Tuple[str, float]]:
        """Analyze topics in feedback text"""
        
        if not text:
            return []
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, patterns in self.topic_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, text_lower))
                matches += pattern_matches
                score += pattern_matches
            
            if matches > 0:
                # Normalize by text length
                word_count = len(text.split())
                normalized_score = min(1.0, score / max(word_count, 10) * 100)
                topic_scores[topic] = normalized_score
        
        # Sort by score and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics

class FeedbackProcessor:
    """Process and analyze user feedback for learning updates"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Analysis components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        
        # Storage
        self.feedback_entries = {}  # feedback_id -> FeedbackEntry
        self.user_feedback_history = defaultdict(list)  # user_id -> [feedback_ids]
        self.processed_summaries = {}  # summary_id -> FeedbackSummary
        self.learning_updates = {}  # update_id -> LearningUpdate
        
        # Processing parameters
        self.batch_size = self.config.get('batch_size', 50)
        self.summary_window_hours = self.config.get('summary_window_hours', 168)  # 1 week
        self.urgent_response_threshold = self.config.get('urgent_response_threshold', 0.8)
        
        # Learning thresholds
        self.min_feedback_for_update = self.config.get('min_feedback_for_update', 5)
        self.confidence_threshold_for_update = self.config.get('confidence_threshold', 0.7)
        
        # Performance tracking
        self.processing_metrics = defaultdict(list)
    
    async def process_feedback(self, 
                             user_id: str,
                             session_id: str,
                             feedback_data: Dict[str, Any],
                             intervention_context: Optional[Dict[str, Any]] = None) -> FeedbackEntry:
        """Process a single feedback entry"""
        
        try:
            self.logger.info(f"Processing feedback from user {user_id}")
            
            # Create feedback entry
            feedback_entry = await self._create_feedback_entry(
                user_id, session_id, feedback_data, intervention_context
            )
            
            # Analyze feedback content
            await self._analyze_feedback_content(feedback_entry)
            
            # Check for urgent issues
            await self._check_urgent_indicators(feedback_entry)
            
            # Store feedback
            self.feedback_entries[feedback_entry.feedback_id] = feedback_entry
            self.user_feedback_history[user_id].append(feedback_entry.feedback_id)
            
            # Mark as processed
            feedback_entry.processed = True
            feedback_entry.processing_timestamp = datetime.now()
            
            # Trigger learning updates if threshold reached
            if len(self.user_feedback_history[user_id]) % self.min_feedback_for_update == 0:
                await self._trigger_learning_updates(user_id)
            
            # Update processing metrics
            self.processing_metrics['feedback_processed'].append(datetime.now())
            
            return feedback_entry
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {str(e)}")
            raise
    
    async def _create_feedback_entry(self,
                                   user_id: str,
                                   session_id: str,
                                   feedback_data: Dict[str, Any],
                                   intervention_context: Optional[Dict[str, Any]]) -> FeedbackEntry:
        """Create a structured feedback entry from raw data"""
        
        feedback_id = f"feedback_{user_id}_{int(datetime.now().timestamp())}"
        
        # Determine feedback type
        feedback_type = self._determine_feedback_type(feedback_data)
        
        # Extract content based on type
        rating_value = feedback_data.get('rating')
        text_content = feedback_data.get('text_feedback')
        structured_responses = feedback_data.get('survey_responses')
        
        # Extract behavioral indicators
        behavioral_indicators = {}
        if 'response_time' in feedback_data:
            behavioral_indicators['response_time'] = feedback_data['response_time']
        if 'interaction_duration' in feedback_data:
            behavioral_indicators['interaction_duration'] = feedback_data['interaction_duration']
        if 'completion_rate' in feedback_data:
            behavioral_indicators['completion_rate'] = feedback_data['completion_rate']
        
        return FeedbackEntry(
            feedback_id=feedback_id,
            user_id=user_id,
            session_id=session_id,
            intervention_id=intervention_context.get('intervention_id') if intervention_context else None,
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            rating_value=rating_value,
            text_content=text_content,
            structured_responses=structured_responses,
            behavioral_indicators=behavioral_indicators,
            context_at_feedback=intervention_context,
            agent_involved=intervention_context.get('agent_name') if intervention_context else None,
            intervention_type=intervention_context.get('intervention_type') if intervention_context else None
        )
    
    def _determine_feedback_type(self, feedback_data: Dict[str, Any]) -> FeedbackType:
        """Determine the type of feedback from the data"""
        
        if 'rating' in feedback_data and feedback_data['rating'] is not None:
            return FeedbackType.EXPLICIT_RATING
        elif 'text_feedback' in feedback_data and feedback_data['text_feedback']:
            return FeedbackType.TEXT_FEEDBACK
        elif 'survey_responses' in feedback_data:
            return FeedbackType.SATISFACTION_SURVEY
        elif 'breakthrough_indicator' in feedback_data:
            return FeedbackType.BREAKTHROUGH_INDICATOR
        elif any(key in feedback_data for key in ['response_time', 'interaction_duration', 'completion_rate']):
            return FeedbackType.IMPLICIT_BEHAVIORAL
        else:
            return FeedbackType.ENGAGEMENT_METRIC
    
    async def _analyze_feedback_content(self, feedback_entry: FeedbackEntry) -> None:
        """Analyze feedback content for sentiment, topics, and insights"""
        
        # Sentiment analysis for text feedback
        if feedback_entry.text_content:
            context = {
                'intervention_effectiveness': feedback_entry.context_at_feedback.get('effectiveness_score', 0.5) if feedback_entry.context_at_feedback else 0.5,
                'user_crisis_history': False  # Would be determined from user history
            }
            
            sentiment, confidence, analysis_details = self.sentiment_analyzer.analyze_sentiment(
                feedback_entry.text_content, context
            )
            
            feedback_entry.sentiment = sentiment
            feedback_entry.confidence = confidence
            
            # Topic analysis
            topics_with_scores = self.topic_analyzer.analyze_topics(feedback_entry.text_content)
            feedback_entry.topics = [topic for topic, score in topics_with_scores if score > 0.1]
            
            # Determine urgency
            if analysis_details.get('crisis_detected', False):
                feedback_entry.urgency_level = 'critical'
            elif sentiment == FeedbackSentiment.VERY_NEGATIVE and confidence > 0.8:
                feedback_entry.urgency_level = 'high'
            elif sentiment == FeedbackSentiment.NEGATIVE:
                feedback_entry.urgency_level = 'medium'
            else:
                feedback_entry.urgency_level = 'low'
        
        # Analyze rating-based feedback
        elif feedback_entry.rating_value is not None:
            rating = feedback_entry.rating_value
            
            if rating >= 9:
                feedback_entry.sentiment = FeedbackSentiment.VERY_POSITIVE
            elif rating >= 7:
                feedback_entry.sentiment = FeedbackSentiment.POSITIVE
            elif rating >= 4:
                feedback_entry.sentiment = FeedbackSentiment.NEUTRAL
            elif rating >= 2:
                feedback_entry.sentiment = FeedbackSentiment.NEGATIVE
            else:
                feedback_entry.sentiment = FeedbackSentiment.VERY_NEGATIVE
            
            feedback_entry.confidence = 0.7  # Moderate confidence for rating-based
            feedback_entry.urgency_level = 'critical' if rating <= 2 else 'medium' if rating <= 4 else 'low'
        
        # Analyze behavioral feedback
        elif feedback_entry.behavioral_indicators:
            indicators = feedback_entry.behavioral_indicators
            
            # Analyze behavioral patterns
            completion_rate = indicators.get('completion_rate', 1.0)
            response_time = indicators.get('response_time', 0)
            duration = indicators.get('interaction_duration', 0)
            
            if completion_rate < 0.3:
                feedback_entry.sentiment = FeedbackSentiment.NEGATIVE
                feedback_entry.urgency_level = 'medium'
            elif completion_rate > 0.8 and duration > 300:  # 5+ minutes
                feedback_entry.sentiment = FeedbackSentiment.POSITIVE
                feedback_entry.urgency_level = 'low'
            else:
                feedback_entry.sentiment = FeedbackSentiment.NEUTRAL
                feedback_entry.urgency_level = 'low'
            
            feedback_entry.confidence = 0.6  # Lower confidence for behavioral indicators
    
    async def _check_urgent_indicators(self, feedback_entry: FeedbackEntry) -> None:
        """Check for urgent indicators requiring immediate attention"""
        
        urgent_flags = []
        
        # Critical sentiment with high confidence
        if (feedback_entry.sentiment == FeedbackSentiment.VERY_NEGATIVE and 
            feedback_entry.confidence and feedback_entry.confidence > self.urgent_response_threshold):
            urgent_flags.append('very_negative_sentiment')
        
        # Crisis indicators in text
        if feedback_entry.urgency_level == 'critical':
            urgent_flags.append('crisis_language_detected')
        
        # Very low ratings
        if feedback_entry.rating_value and feedback_entry.rating_value <= 2:
            urgent_flags.append('very_low_rating')
        
        # Behavioral red flags
        if feedback_entry.behavioral_indicators:
            completion_rate = feedback_entry.behavioral_indicators.get('completion_rate', 1.0)
            if completion_rate < 0.2:
                urgent_flags.append('very_low_engagement')
        
        # Log urgent issues
        if urgent_flags:
            self.logger.warning(f"Urgent feedback indicators detected for user {feedback_entry.user_id}: {urgent_flags}")
            feedback_entry.processing_notes = f"Urgent flags: {', '.join(urgent_flags)}"
    
    async def _trigger_learning_updates(self, user_id: str) -> None:
        """Trigger learning updates based on accumulated feedback"""
        
        try:
            # Get recent feedback for user
            recent_feedback_ids = self.user_feedback_history[user_id][-self.min_feedback_for_update:]
            recent_feedback = [self.feedback_entries[fid] for fid in recent_feedback_ids if fid in self.feedback_entries]
            
            if len(recent_feedback) < self.min_feedback_for_update:
                return
            
            # Analyze feedback patterns
            patterns = await self._analyze_feedback_patterns(recent_feedback)
            
            # Generate learning updates
            learning_updates = await self._generate_learning_updates(patterns, recent_feedback)
            
            # Store learning updates
            for update in learning_updates:
                self.learning_updates[update.update_id] = update
                
                self.logger.info(f"Generated learning update: {update.update_type} for {update.target_component}")
            
        except Exception as e:
            self.logger.error(f"Error triggering learning updates: {str(e)}")
    
    async def _analyze_feedback_patterns(self, feedback_entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Analyze patterns in feedback for learning insights"""
        
        patterns = {
            'sentiment_trend': [],
            'rating_trend': [],
            'topic_frequency': defaultdict(int),
            'intervention_effectiveness': defaultdict(list),
            'agent_performance': defaultdict(list),
            'urgency_patterns': [],
            'behavioral_trends': {}
        }
        
        for entry in feedback_entries:
            # Sentiment trend
            if entry.sentiment:
                sentiment_score = self._sentiment_to_score(entry.sentiment)
                patterns['sentiment_trend'].append((entry.timestamp, sentiment_score))
            
            # Rating trend
            if entry.rating_value:
                patterns['rating_trend'].append((entry.timestamp, entry.rating_value))
            
            # Topic frequency
            if entry.topics:
                for topic in entry.topics:
                    patterns['topic_frequency'][topic] += 1
            
            # Intervention effectiveness
            if entry.intervention_type and entry.rating_value:
                patterns['intervention_effectiveness'][entry.intervention_type].append(entry.rating_value)
            
            # Agent performance
            if entry.agent_involved and entry.rating_value:
                patterns['agent_performance'][entry.agent_involved].append(entry.rating_value)
            
            # Urgency patterns
            if entry.urgency_level:
                patterns['urgency_patterns'].append((entry.timestamp, entry.urgency_level))
            
            # Behavioral trends
            if entry.behavioral_indicators:
                for metric, value in entry.behavioral_indicators.items():
                    if metric not in patterns['behavioral_trends']:
                        patterns['behavioral_trends'][metric] = []
                    patterns['behavioral_trends'][metric].append((entry.timestamp, value))
        
        return patterns
    
    def _sentiment_to_score(self, sentiment: FeedbackSentiment) -> float:
        """Convert sentiment to numerical score"""
        sentiment_map = {
            FeedbackSentiment.VERY_NEGATIVE: 1.0,
            FeedbackSentiment.NEGATIVE: 3.0,
            FeedbackSentiment.NEUTRAL: 5.0,
            FeedbackSentiment.POSITIVE: 7.0,
            FeedbackSentiment.VERY_POSITIVE: 9.0
        }
        return sentiment_map.get(sentiment, 5.0)
    
    async def _generate_learning_updates(self,
                                       patterns: Dict[str, Any],
                                       feedback_entries: List[FeedbackEntry]) -> List[LearningUpdate]:
        """Generate specific learning updates from feedback patterns"""
        
        learning_updates = []
        
        # Analyze sentiment trends
        sentiment_trend = patterns['sentiment_trend']
        if len(sentiment_trend) >= 3:
            recent_scores = [score for _, score in sentiment_trend[-3:]]
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            if trend_slope < -1.0:  # Declining satisfaction
                update = LearningUpdate(
                    update_id=f"sentiment_decline_{int(datetime.now().timestamp())}",
                    source_feedback_ids=[entry.feedback_id for entry in feedback_entries],
                    target_component="system",
                    update_type="strategy",
                    current_value="standard_approach",
                    recommended_value="enhanced_support_approach",
                    confidence=0.8,
                    expected_impact=0.6,
                    reasoning="Declining satisfaction trend detected, recommend enhanced support",
                    supporting_evidence=[f"Sentiment declined by {abs(trend_slope):.2f} points"],
                    risk_assessment={"implementation_risk": 0.2, "user_impact_risk": 0.1},
                    priority="high",
                    estimated_effort="medium",
                    requires_approval=False,
                    expiry_date=datetime.now() + timedelta(days=30),
                    created_at=datetime.now()
                )
                learning_updates.append(update)
        
        # Analyze intervention effectiveness
        intervention_effectiveness = patterns['intervention_effectiveness']
        for intervention, ratings in intervention_effectiveness.items():
            if len(ratings) >= 3:
                avg_rating = np.mean(ratings)
                if avg_rating < 4.0:  # Poor performance
                    update = LearningUpdate(
                        update_id=f"intervention_poor_{intervention}_{int(datetime.now().timestamp())}",
                        source_feedback_ids=[entry.feedback_id for entry in feedback_entries if entry.intervention_type == intervention],
                        target_component="intervention",
                        update_type="weight",
                        current_value=1.0,
                        recommended_value=0.5,  # Reduce weight
                        confidence=0.7,
                        expected_impact=0.4,
                        reasoning=f"Intervention {intervention} showing poor performance (avg rating: {avg_rating:.2f})",
                        supporting_evidence=[f"{len(ratings)} ratings with average {avg_rating:.2f}"],
                        risk_assessment={"implementation_risk": 0.1, "user_impact_risk": 0.3},
                        priority="medium",
                        estimated_effort="low",
                        requires_approval=False,
                        expiry_date=datetime.now() + timedelta(days=14),
                        created_at=datetime.now()
                    )
                    learning_updates.append(update)
                
                elif avg_rating > 8.0:  # Excellent performance
                    update = LearningUpdate(
                        update_id=f"intervention_excellent_{intervention}_{int(datetime.now().timestamp())}",
                        source_feedback_ids=[entry.feedback_id for entry in feedback_entries if entry.intervention_type == intervention],
                        target_component="intervention",
                        update_type="weight",
                        current_value=1.0,
                        recommended_value=1.3,  # Increase weight
                        confidence=0.8,
                        expected_impact=0.5,
                        reasoning=f"Intervention {intervention} showing excellent performance (avg rating: {avg_rating:.2f})",
                        supporting_evidence=[f"{len(ratings)} ratings with average {avg_rating:.2f}"],
                        risk_assessment={"implementation_risk": 0.1, "user_impact_risk": 0.1},
                        priority="medium",
                        estimated_effort="low",
                        requires_approval=False,
                        expiry_date=datetime.now() + timedelta(days=14),
                        created_at=datetime.now()
                    )
                    learning_updates.append(update)
        
        # Analyze agent performance
        agent_performance = patterns['agent_performance']
        for agent, ratings in agent_performance.items():
            if len(ratings) >= 3:
                avg_rating = np.mean(ratings)
                if avg_rating < 4.0:  # Poor performance
                    update = LearningUpdate(
                        update_id=f"agent_performance_{agent}_{int(datetime.now().timestamp())}",
                        source_feedback_ids=[entry.feedback_id for entry in feedback_entries if entry.agent_involved == agent],
                        target_component="agent",
                        update_type="parameter",
                        current_value="standard_parameters",
                        recommended_value="enhanced_parameters",
                        confidence=0.7,
                        expected_impact=0.4,
                        reasoning=f"Agent {agent} receiving poor ratings (avg: {avg_rating:.2f})",
                        supporting_evidence=[f"{len(ratings)} user ratings averaging {avg_rating:.2f}"],
                        risk_assessment={"implementation_risk": 0.2, "user_impact_risk": 0.2},
                        priority="high" if avg_rating < 3.0 else "medium",
                        estimated_effort="medium",
                        requires_approval=True,
                        expiry_date=datetime.now() + timedelta(days=21),
                        created_at=datetime.now()
                    )
                    learning_updates.append(update)
        
        # Analyze behavioral trends
        behavioral_trends = patterns['behavioral_trends']
        if 'completion_rate' in behavioral_trends:
            completion_rates = [rate for _, rate in behavioral_trends['completion_rate']]
            avg_completion = np.mean(completion_rates)
            
            if avg_completion < 0.6:  # Low completion rate
                update = LearningUpdate(
                    update_id=f"low_completion_{int(datetime.now().timestamp())}",
                    source_feedback_ids=[entry.feedback_id for entry in feedback_entries],
                    target_component="system",
                    update_type="strategy",
                    current_value="current_engagement_strategy",
                    recommended_value="enhanced_engagement_strategy",
                    confidence=0.6,
                    expected_impact=0.5,
                    reasoning=f"Low average completion rate detected: {avg_completion:.2f}",
                    supporting_evidence=[f"Average completion rate: {avg_completion:.2f} across {len(completion_rates)} sessions"],
                    risk_assessment={"implementation_risk": 0.2, "user_impact_risk": 0.1},
                    priority="medium",
                    estimated_effort="medium",
                    requires_approval=False,
                    expiry_date=datetime.now() + timedelta(days=30),
                    created_at=datetime.now()
                )
                learning_updates.append(update)
        
        return learning_updates
    
    async def generate_feedback_summary(self,
                                      user_id: Optional[str] = None,
                                      time_period_hours: int = None) -> FeedbackSummary:
        """Generate comprehensive feedback summary"""
        
        if time_period_hours is None:
            time_period_hours = self.summary_window_hours
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_period_hours)
        
        # Filter feedback entries
        if user_id:
            feedback_ids = self.user_feedback_history.get(user_id, [])
            relevant_feedback = [
                self.feedback_entries[fid] for fid in feedback_ids
                if fid in self.feedback_entries and 
                start_time <= self.feedback_entries[fid].timestamp <= end_time
            ]
        else:
            relevant_feedback = [
                entry for entry in self.feedback_entries.values()
                if start_time <= entry.timestamp <= end_time
            ]
        
        if not relevant_feedback:
            return self._create_empty_summary(user_id, start_time, end_time)
        
        # Calculate quantitative metrics
        total_count = len(relevant_feedback)
        
        ratings = [entry.rating_value for entry in relevant_feedback if entry.rating_value is not None]
        average_rating = np.mean(ratings) if ratings else 0.0
        
        sentiment_dist = defaultdict(int)
        for entry in relevant_feedback:
            if entry.sentiment:
                sentiment_dist[entry.sentiment.value] += 1
        
        feedback_type_dist = defaultdict(int)
        for entry in relevant_feedback:
            feedback_type_dist[entry.feedback_type.value] += 1
        
        # Analyze themes and insights
        common_themes = self._extract_common_themes(relevant_feedback)
        improvement_areas = self._identify_improvement_areas(relevant_feedback)
        positive_highlights = self._extract_positive_highlights(relevant_feedback)
        critical_issues = self._identify_critical_issues(relevant_feedback)
        
        # Generate recommendations
        recommended_actions = await self._generate_actionable_recommendations(relevant_feedback)
        
        # Calculate priority and confidence
        priority_level = self._calculate_priority_level(relevant_feedback)
        confidence_score = self._calculate_confidence_score(relevant_feedback)
        data_quality_score = self._assess_data_quality(relevant_feedback)
        
        summary = FeedbackSummary(
            summary_id=f"summary_{user_id or 'global'}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            time_period_start=start_time,
            time_period_end=end_time,
            total_feedback_count=total_count,
            average_rating=average_rating,
            sentiment_distribution=dict(sentiment_dist),
            feedback_type_distribution=dict(feedback_type_dist),
            common_themes=common_themes,
            improvement_areas=improvement_areas,
            positive_highlights=positive_highlights,
            critical_issues=critical_issues,
            recommended_actions=recommended_actions,
            priority_level=priority_level,
            confidence_score=confidence_score,
            generated_at=datetime.now(),
            data_quality_score=data_quality_score
        )
        
        # Store summary
        self.processed_summaries[summary.summary_id] = summary
        
        return summary
    
    def _create_empty_summary(self, user_id: Optional[str], start_time: datetime, end_time: datetime) -> FeedbackSummary:
        """Create an empty summary for periods with no feedback"""
        return FeedbackSummary(
            summary_id=f"empty_summary_{user_id or 'global'}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            time_period_start=start_time,
            time_period_end=end_time,
            total_feedback_count=0,
            average_rating=0.0,
            sentiment_distribution={},
            feedback_type_distribution={},
            common_themes=[],
            improvement_areas=["No feedback data available"],
            positive_highlights=[],
            critical_issues=[],
            recommended_actions=[],
            priority_level="low",
            confidence_score=0.0,
            generated_at=datetime.now(),
            data_quality_score=0.0
        )
    
    def _extract_common_themes(self, feedback_entries: List[FeedbackEntry]) -> List[str]:
        """Extract common themes from feedback"""
        
        theme_counts = defaultdict(int)
        
        for entry in feedback_entries:
            if entry.topics:
                for topic in entry.topics:
                    theme_counts[topic] += 1
        
        # Return themes mentioned by at least 20% of feedback
        threshold = max(2, len(feedback_entries) * 0.2)
        common_themes = [theme for theme, count in theme_counts.items() if count >= threshold]
        
        return sorted(common_themes, key=lambda x: theme_counts[x], reverse=True)
    
    def _identify_improvement_areas(self, feedback_entries: List[FeedbackEntry]) -> List[str]:
        """Identify areas needing improvement based on negative feedback"""
        
        improvement_areas = []
        
        # Analyze negative sentiment feedback
        negative_feedback = [
            entry for entry in feedback_entries 
            if entry.sentiment in [FeedbackSentiment.NEGATIVE, FeedbackSentiment.VERY_NEGATIVE]
        ]
        
        if not negative_feedback:
            return improvement_areas
        
        # Common topics in negative feedback
        negative_themes = defaultdict(int)
        for entry in negative_feedback:
            if entry.topics:
                for topic in entry.topics:
                    negative_themes[topic] += 1
        
        # Areas mentioned frequently in negative context
        for theme, count in negative_themes.items():
            if count >= len(negative_feedback) * 0.3:  # 30% of negative feedback
                improvement_areas.append(f"Improve {theme.replace('_', ' ')}")
        
        # Analyze low ratings
        low_ratings = [entry for entry in feedback_entries if entry.rating_value and entry.rating_value <= 4]
        if len(low_ratings) > len(feedback_entries) * 0.3:
            improvement_areas.append("Address overall user satisfaction")
        
        return improvement_areas
    
    def _extract_positive_highlights(self, feedback_entries: List[FeedbackEntry]) -> List[str]:
        """Extract positive highlights from feedback"""
        
        positive_highlights = []
        
        # Analyze positive sentiment feedback
        positive_feedback = [
            entry for entry in feedback_entries 
            if entry.sentiment in [FeedbackSentiment.POSITIVE, FeedbackSentiment.VERY_POSITIVE]
        ]
        
        if not positive_feedback:
            return positive_highlights
        
        # Common topics in positive feedback
        positive_themes = defaultdict(int)
        for entry in positive_feedback:
            if entry.topics:
                for topic in entry.topics:
                    positive_themes[topic] += 1
        
        # Highlights mentioned frequently in positive context
        for theme, count in positive_themes.items():
            if count >= len(positive_feedback) * 0.3:  # 30% of positive feedback
                positive_highlights.append(f"Strong performance in {theme.replace('_', ' ')}")
        
        # High ratings
        high_ratings = [entry for entry in feedback_entries if entry.rating_value and entry.rating_value >= 8]
        if len(high_ratings) > len(feedback_entries) * 0.4:
            positive_highlights.append("High overall user satisfaction")
        
        return positive_highlights
    
    def _identify_critical_issues(self, feedback_entries: List[FeedbackEntry]) -> List[str]:
        """Identify critical issues requiring immediate attention"""
        
        critical_issues = []
        
        # Crisis indicators
        crisis_feedback = [entry for entry in feedback_entries if entry.urgency_level == 'critical']
        if crisis_feedback:
            critical_issues.append(f"Crisis indicators detected in {len(crisis_feedback)} feedback entries")
        
        # Very negative feedback
        very_negative = [entry for entry in feedback_entries if entry.sentiment == FeedbackSentiment.VERY_NEGATIVE]
        if len(very_negative) > len(feedback_entries) * 0.2:
            critical_issues.append("High proportion of very negative feedback")
        
        # Very low ratings
        very_low_ratings = [entry for entry in feedback_entries if entry.rating_value and entry.rating_value <= 2]
        if len(very_low_ratings) > len(feedback_entries) * 0.2:
            critical_issues.append("High proportion of very low ratings")
        
        # Low engagement
        low_engagement = [
            entry for entry in feedback_entries 
            if entry.behavioral_indicators and entry.behavioral_indicators.get('completion_rate', 1.0) < 0.3
        ]
        if len(low_engagement) > len(feedback_entries) * 0.4:
            critical_issues.append("Low user engagement detected")
        
        return critical_issues
    
    async def _generate_actionable_recommendations(self, feedback_entries: List[FeedbackEntry]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on feedback analysis"""
        
        recommendations = []
        
        # Analyze patterns for recommendations
        sentiment_scores = [self._sentiment_to_score(entry.sentiment) for entry in feedback_entries if entry.sentiment]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 5.0
        
        ratings = [entry.rating_value for entry in feedback_entries if entry.rating_value is not None]
        avg_rating = np.mean(ratings) if ratings else 5.0
        
        # Low satisfaction recommendations
        if avg_sentiment < 4.0 or avg_rating < 4.0:
            recommendations.append({
                'action': 'implement_enhanced_support',
                'priority': 'high',
                'description': 'Implement enhanced user support measures',
                'estimated_impact': 'high',
                'timeline': '1-2 weeks'
            })
        
        # Topic-based recommendations
        topic_counts = defaultdict(int)
        for entry in feedback_entries:
            if entry.topics:
                for topic in entry.topics:
                    topic_counts[topic] += 1
        
        for topic, count in topic_counts.items():
            if count >= len(feedback_entries) * 0.3:  # Frequently mentioned
                recommendations.append({
                    'action': f'focus_on_{topic}',
                    'priority': 'medium',
                    'description': f'Increase focus on {topic.replace("_", " ")} interventions',
                    'estimated_impact': 'medium',
                    'timeline': '2-4 weeks'
                })
        
        # Behavioral recommendations
        behavioral_metrics = defaultdict(list)
        for entry in feedback_entries:
            if entry.behavioral_indicators:
                for metric, value in entry.behavioral_indicators.items():
                    behavioral_metrics[metric].append(value)
        
        if 'completion_rate' in behavioral_metrics:
            avg_completion = np.mean(behavioral_metrics['completion_rate'])
            if avg_completion < 0.7:
                recommendations.append({
                    'action': 'improve_engagement',
                    'priority': 'medium',
                    'description': 'Improve user engagement and session completion',
                    'estimated_impact': 'medium',
                    'timeline': '3-4 weeks'
                })
        
        return recommendations
    
    def _calculate_priority_level(self, feedback_entries: List[FeedbackEntry]) -> str:
        """Calculate priority level for the feedback summary"""
        
        critical_count = len([entry for entry in feedback_entries if entry.urgency_level == 'critical'])
        high_count = len([entry for entry in feedback_entries if entry.urgency_level == 'high'])
        
        total_count = len(feedback_entries)
        
        if critical_count > 0 or (total_count > 0 and high_count / total_count > 0.3):
            return 'critical'
        elif high_count > 0 or (total_count > 0 and high_count / total_count > 0.2):
            return 'high'
        else:
            return 'medium'
    
    def _calculate_confidence_score(self, feedback_entries: List[FeedbackEntry]) -> float:
        """Calculate confidence score for the summary"""
        
        if not feedback_entries:
            return 0.0
        
        # Base confidence on amount of data
        data_confidence = min(1.0, len(feedback_entries) / 20)  # Max confidence at 20+ entries
        
        # Adjust based on feedback types
        text_feedback_count = len([entry for entry in feedback_entries if entry.text_content])
        rating_feedback_count = len([entry for entry in feedback_entries if entry.rating_value is not None])
        
        content_confidence = (text_feedback_count * 0.8 + rating_feedback_count * 0.6) / len(feedback_entries)
        
        # Confidence in analysis
        analyzed_count = len([entry for entry in feedback_entries if entry.confidence is not None])
        analysis_confidence = analyzed_count / len(feedback_entries)
        
        return (data_confidence + content_confidence + analysis_confidence) / 3
    
    def _assess_data_quality(self, feedback_entries: List[FeedbackEntry]) -> float:
        """Assess the quality of feedback data"""
        
        if not feedback_entries:
            return 0.0
        
        quality_factors = []
        
        # Completeness - how much of the feedback has content
        complete_entries = len([
            entry for entry in feedback_entries 
            if (entry.text_content and len(entry.text_content.strip()) > 10) or 
               entry.rating_value is not None or
               entry.behavioral_indicators
        ])
        quality_factors.append(complete_entries / len(feedback_entries))
        
        # Recency - how recent the feedback is
        now = datetime.now()
        recent_entries = len([
            entry for entry in feedback_entries 
            if (now - entry.timestamp).days <= 7
        ])
        quality_factors.append(min(1.0, recent_entries / len(feedback_entries) * 2))
        
        # Diversity - variety in feedback types
        unique_types = len(set(entry.feedback_type for entry in feedback_entries))
        quality_factors.append(min(1.0, unique_types / len(FeedbackType)))
        
        # Processing completeness
        processed_entries = len([entry for entry in feedback_entries if entry.processed])
        quality_factors.append(processed_entries / len(feedback_entries))
        
        return np.mean(quality_factors)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get overall feedback processing statistics"""
        
        now = datetime.now()
        
        return {
            'total_feedback_entries': len(self.feedback_entries),
            'total_users_with_feedback': len(self.user_feedback_history),
            'total_summaries_generated': len(self.processed_summaries),
            'total_learning_updates': len(self.learning_updates),
            'feedback_by_type': {
                ftype.value: len([
                    entry for entry in self.feedback_entries.values() 
                    if entry.feedback_type == ftype
                ])
                for ftype in FeedbackType
            },
            'processing_metrics': {
                'recent_processing_rate': len([
                    ts for ts in self.processing_metrics.get('feedback_processed', [])
                    if (now - ts).hours <= 1
                ]),
                'average_processing_time': 'N/A',  # Would need timing data
                'success_rate': 'N/A'  # Would need error tracking
            },
            'urgency_distribution': {
                'critical': len([
                    entry for entry in self.feedback_entries.values()
                    if entry.urgency_level == 'critical'
                ]),
                'high': len([
                    entry for entry in self.feedback_entries.values()
                    if entry.urgency_level == 'high'
                ]),
                'medium': len([
                    entry for entry in self.feedback_entries.values()
                    if entry.urgency_level == 'medium'
                ]),
                'low': len([
                    entry for entry in self.feedback_entries.values()
                    if entry.urgency_level == 'low'
                ])
            }
        }
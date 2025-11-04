"""
Insights Generation System for Adaptive Learning

This module provides comprehensive insights generation for system improvement
and agent training, including performance analytics, recommendation engines,
and predictive modeling for therapeutic optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import json
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger
from src.agents.analytics.outcome_tracker import OutcomeTracker, OutcomeMeasurement, UserOutcomeProfile
from src.agents.personalization.feedback_integration_system import FeedbackProcessor, FeedbackEntry, FeedbackSummary
from src.agents.personalization.personalization_engine import PersonalizationEngine
from src.agents.analytics.pattern_recognition_engine import PatternRecognitionEngine

logger = get_logger(__name__)

class InsightType(Enum):
    """Types of insights that can be generated"""
    PERFORMANCE_TREND = "performance_trend"
    USER_BEHAVIOR = "user_behavior"
    INTERVENTION_EFFECTIVENESS = "intervention_effectiveness"
    AGENT_OPTIMIZATION = "agent_optimization"
    PREDICTIVE_MODEL = "predictive_model"
    ANOMALY_DETECTION = "anomaly_detection"
    THERAPEUTIC_PATTERN = "therapeutic_pattern"
    SYSTEM_HEALTH = "system_health"

class InsightPriority(Enum):
    """Priority levels for insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SystemInsight:
    """Individual system insight"""
    insight_id: str
    insight_type: InsightType
    priority: InsightPriority
    title: str
    description: str
    
    # Data and evidence
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    statistical_significance: Optional[float] = None
    
    # Recommendations and actions
    recommendations: List[str] = field(default_factory=list)
    actionable_items: List[str] = field(default_factory=list)
    expected_impact: str = "medium"
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    data_source: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    # Tracking
    implemented: bool = False
    implementation_date: Optional[datetime] = None
    measured_impact: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceTrend:
    """System performance trend analysis"""
    metric_name: str
    time_period_days: int
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0-1, how strong the trend is
    current_value: float
    previous_period_value: float
    percentage_change: float
    is_significant: bool
    confidence_interval: Tuple[float, float]

@dataclass
class UserSegmentInsight:
    """Insights about user segments"""
    segment_id: str
    segment_characteristics: Dict[str, Any]
    user_count: int
    avg_outcomes: Dict[str, float]
    preferred_interventions: List[str]
    common_patterns: List[str]
    specific_recommendations: List[str]

@dataclass
class InterventionAnalysis:
    """Analysis of intervention effectiveness"""
    intervention_type: str
    success_rate: float
    avg_user_satisfaction: float
    optimal_conditions: Dict[str, Any]
    user_segments_most_effective: List[str]
    improvement_suggestions: List[str]
    statistical_confidence: float

class InsightGenerationSystem:
    """Main system for generating insights from adaptive learning data"""
    
    def __init__(self, 
                 outcome_tracker: OutcomeTracker,
                 feedback_processor: FeedbackProcessor,
                 personalization_engine: PersonalizationEngine,
                 pattern_engine: PatternRecognitionEngine,
                 config: Dict[str, Any] = None):
        
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Component references
        self.outcome_tracker = outcome_tracker
        self.feedback_processor = feedback_processor
        self.personalization_engine = personalization_engine
        self.pattern_engine = pattern_engine
        
        # Insight storage
        self.generated_insights = []
        self.insight_history = []
        self.performance_trends = {}
        
        # Analytics models
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.trend_analyzer = None
        self.user_segmentation_model = None
        
        # Configuration
        self.min_data_points = self.config.get('min_data_points_for_insights', 50)
        self.insight_retention_days = self.config.get('insight_retention_days', 365)
        self.trend_analysis_days = self.config.get('trend_analysis_days', 30)
        
        self.logger.info("Insights Generation System initialized")
    
    async def generate_comprehensive_insights(self, 
                                            time_period_days: int = 30) -> List[SystemInsight]:
        """Generate comprehensive system insights across all components"""
        
        try:
            self.logger.info(f"Generating comprehensive insights for {time_period_days}-day period")
            
            insights = []
            
            # 1. Performance trend analysis
            performance_insights = await self._analyze_performance_trends(time_period_days)
            insights.extend(performance_insights)
            
            # 2. User behavior analysis
            user_behavior_insights = await self._analyze_user_behavior(time_period_days)
            insights.extend(user_behavior_insights)
            
            # 3. Intervention effectiveness analysis
            intervention_insights = await self._analyze_intervention_effectiveness(time_period_days)
            insights.extend(intervention_insights)
            
            # 4. Agent optimization insights
            agent_insights = await self._analyze_agent_performance(time_period_days)
            insights.extend(agent_insights)
            
            # 5. Predictive model insights
            predictive_insights = await self._generate_predictive_insights(time_period_days)
            insights.extend(predictive_insights)
            
            # 6. Anomaly detection
            anomaly_insights = await self._detect_system_anomalies(time_period_days)
            insights.extend(anomaly_insights)
            
            # 7. Therapeutic pattern insights
            therapeutic_insights = await self._analyze_therapeutic_patterns(time_period_days)
            insights.extend(therapeutic_insights)
            
            # 8. System health insights
            health_insights = await self._analyze_system_health(time_period_days)
            insights.extend(health_insights)
            
            # Store generated insights
            self.generated_insights = insights
            self._update_insight_history(insights)
            
            # Generate summary report
            summary = self._create_insights_summary(insights)
            self.logger.info(f"Generated {len(insights)} insights: {summary}")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive insights: {str(e)}")
            return []
    
    async def _analyze_performance_trends(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze system performance trends"""
        
        insights = []
        
        try:
            # Get system performance data
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            # Analyze outcome trends
            outcome_data = self.outcome_tracker.get_system_statistics()
            
            if 'trend_data' in outcome_data:
                for metric_name, trend_data in outcome_data['trend_data'].items():
                    trend = self._calculate_performance_trend(metric_name, trend_data)
                    
                    if trend.is_significant:
                        insight = self._create_trend_insight(trend)
                        insights.append(insight)
            
            # Analyze feedback trends
            feedback_stats = self.feedback_processor.get_feedback_statistics()
            
            if feedback_stats.get('total_feedback_count', 0) > self.min_data_points:
                # Satisfaction trend analysis
                satisfaction_trend = self._analyze_satisfaction_trend(feedback_stats)
                if satisfaction_trend:
                    insights.append(satisfaction_trend)
                
                # Response time trend analysis
                response_trend = self._analyze_response_time_trend(feedback_stats)
                if response_trend:
                    insights.append(response_trend)
            
            self.logger.info(f"Generated {len(insights)} performance trend insights")
            
        except Exception as e:
            self.logger.error(f"Error in performance trend analysis: {str(e)}")
        
        return insights
    
    def _calculate_performance_trend(self, metric_name: str, trend_data: List[Tuple]) -> PerformanceTrend:
        """Calculate performance trend from time series data"""
        
        if len(trend_data) < 2:
            return PerformanceTrend(
                metric_name=metric_name,
                time_period_days=0,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                current_value=0.0,
                previous_period_value=0.0,
                percentage_change=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0)
            )
        
        # Extract values and timestamps
        timestamps = [point[0] for point in trend_data]
        values = [point[1] for point in trend_data]
        
        # Calculate trend
        current_value = values[-1]
        previous_value = values[0]
        percentage_change = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0.0
        
        # Simple linear regression for trend strength
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        correlation = np.corrcoef(x, values)[0, 1]
        
        # Determine trend direction
        if abs(percentage_change) < 5:
            trend_direction = "stable"
        elif percentage_change > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Check statistical significance (simplified)
        is_significant = abs(correlation) > 0.5 and abs(percentage_change) > 10
        
        # Calculate confidence interval (simplified)
        std_error = np.std(values) / np.sqrt(len(values))
        confidence_interval = (current_value - 1.96 * std_error, current_value + 1.96 * std_error)
        
        return PerformanceTrend(
            metric_name=metric_name,
            time_period_days=len(values),
            trend_direction=trend_direction,
            trend_strength=abs(correlation),
            current_value=current_value,
            previous_period_value=previous_value,
            percentage_change=percentage_change,
            is_significant=is_significant,
            confidence_interval=confidence_interval
        )
    
    def _create_trend_insight(self, trend: PerformanceTrend) -> SystemInsight:
        """Create insight from performance trend"""
        
        if trend.trend_direction == "increasing" and trend.current_value > 0.7:
            priority = InsightPriority.HIGH
            title = f"Positive trend in {trend.metric_name}"
            description = f"{trend.metric_name} has improved by {trend.percentage_change:.1f}% over the analysis period"
            recommendations = [
                f"Continue current strategies that are improving {trend.metric_name}",
                "Identify and replicate successful patterns",
                "Monitor for sustainability of improvements"
            ]
        elif trend.trend_direction == "decreasing" and trend.current_value < 0.5:
            priority = InsightPriority.CRITICAL
            title = f"Declining performance in {trend.metric_name}"
            description = f"{trend.metric_name} has decreased by {abs(trend.percentage_change):.1f}% over the analysis period"
            recommendations = [
                f"Immediate investigation required for {trend.metric_name} decline",
                "Review recent system changes that might affect performance",
                "Implement corrective measures",
                "Increase monitoring frequency"
            ]
        else:
            priority = InsightPriority.MEDIUM
            title = f"Performance pattern detected in {trend.metric_name}"
            description = f"{trend.metric_name} shows {trend.trend_direction} trend with {trend.percentage_change:.1f}% change"
            recommendations = [
                f"Monitor {trend.metric_name} for continued patterns",
                "Analyze contributing factors",
                "Consider optimization strategies"
            ]
        
        return SystemInsight(
            insight_id=f"trend_{trend.metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.PERFORMANCE_TREND,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'trend_data': asdict(trend),
                'confidence_interval': trend.confidence_interval,
                'statistical_significance': trend.is_significant
            },
            confidence_score=trend.trend_strength,
            recommendations=recommendations,
            data_source=['outcome_tracker'],
            affected_components=['system_performance']
        )
    
    def _analyze_satisfaction_trend(self, feedback_stats: Dict[str, Any]) -> Optional[SystemInsight]:
        """Analyze user satisfaction trends"""
        
        if 'satisfaction_history' not in feedback_stats:
            return None
        
        satisfaction_data = feedback_stats['satisfaction_history']
        if len(satisfaction_data) < self.min_data_points:
            return None
        
        # Calculate trend
        recent_satisfaction = np.mean(satisfaction_data[-10:])  # Last 10 data points
        historical_satisfaction = np.mean(satisfaction_data[:-10])  # Previous data points
        
        change = recent_satisfaction - historical_satisfaction
        percentage_change = (change / historical_satisfaction * 100) if historical_satisfaction > 0 else 0
        
        if abs(percentage_change) < 5:  # Not significant change
            return None
        
        if change > 0:
            priority = InsightPriority.HIGH
            title = "User satisfaction is improving"
            description = f"User satisfaction has increased by {percentage_change:.1f}% recently"
            recommendations = [
                "Continue current engagement strategies",
                "Identify and replicate successful interaction patterns",
                "Share best practices across all agents"
            ]
        else:
            priority = InsightPriority.CRITICAL
            title = "User satisfaction is declining"
            description = f"User satisfaction has decreased by {abs(percentage_change):.1f}% recently"
            recommendations = [
                "Immediate review of user feedback required",
                "Investigate recent changes that might affect satisfaction",
                "Implement user experience improvements",
                "Increase proactive user engagement"
            ]
        
        return SystemInsight(
            insight_id=f"satisfaction_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.USER_BEHAVIOR,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'recent_satisfaction': recent_satisfaction,
                'historical_satisfaction': historical_satisfaction,
                'percentage_change': percentage_change,
                'data_points': len(satisfaction_data)
            },
            confidence_score=min(1.0, abs(percentage_change) / 20.0),  # Higher confidence for larger changes
            recommendations=recommendations,
            data_source=['feedback_processor'],
            affected_components=['user_experience', 'agent_performance']
        )
    
    def _analyze_response_time_trend(self, feedback_stats: Dict[str, Any]) -> Optional[SystemInsight]:
        """Analyze system response time trends"""
        
        if 'response_times' not in feedback_stats:
            return None
        
        response_times = feedback_stats['response_times']
        if len(response_times) < self.min_data_points:
            return None
        
        # Calculate trend
        recent_avg = np.mean(response_times[-20:])
        historical_avg = np.mean(response_times[:-20])
        
        change = recent_avg - historical_avg
        percentage_change = (change / historical_avg * 100) if historical_avg > 0 else 0
        
        if abs(percentage_change) < 10:  # Not significant change
            return None
        
        if change < 0:  # Response time decreased (better)
            priority = InsightPriority.MEDIUM
            title = "System response times improving"
            description = f"Response times have improved by {abs(percentage_change):.1f}%"
            recommendations = [
                "Continue performance optimizations",
                "Monitor system resources to maintain improvements",
                "Document successful optimization strategies"
            ]
        else:  # Response time increased (worse)
            priority = InsightPriority.HIGH
            title = "System response times degrading"
            description = f"Response times have increased by {percentage_change:.1f}%"
            recommendations = [
                "Investigate system performance bottlenecks",
                "Review recent code changes for performance impact",
                "Consider scaling system resources",
                "Optimize slow database queries or API calls"
            ]
        
        return SystemInsight(
            insight_id=f"response_time_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.SYSTEM_HEALTH,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'recent_avg_response_time': recent_avg,
                'historical_avg_response_time': historical_avg,
                'percentage_change': percentage_change,
                'sample_size': len(response_times)
            },
            confidence_score=min(1.0, abs(percentage_change) / 30.0),
            recommendations=recommendations,
            data_source=['feedback_processor'],
            affected_components=['system_performance', 'user_experience']
        )
    
    async def _analyze_user_behavior(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze user behavior patterns"""
        
        insights = []
        
        try:
            # Get user outcome profiles
            all_profiles = self.outcome_tracker.user_profiles
            
            if len(all_profiles) < self.min_data_points:
                return insights
            
            # Perform user segmentation
            segments = await self._perform_user_segmentation(all_profiles)
            
            # Analyze each segment
            for segment in segments:
                segment_insight = self._create_user_segment_insight(segment)
                if segment_insight:
                    insights.append(segment_insight)
            
            # Identify behavioral anomalies
            anomaly_insights = await self._identify_behavioral_anomalies(all_profiles)
            insights.extend(anomaly_insights)
            
            self.logger.info(f"Generated {len(insights)} user behavior insights")
            
        except Exception as e:
            self.logger.error(f"Error in user behavior analysis: {str(e)}")
        
        return insights
    
    async def _perform_user_segmentation(self, user_profiles: Dict[str, UserOutcomeProfile]) -> List[UserSegmentInsight]:
        """Perform clustering-based user segmentation"""
        
        if len(user_profiles) < 10:  # Need minimum users for segmentation
            return []
        
        # Prepare data for clustering
        features = []
        user_ids = []
        
        for user_id, profile in user_profiles.items():
            user_features = [
                len(profile.outcome_measurements),  # Activity level
                np.mean([m.primary_score for m in profile.outcome_measurements]) if profile.outcome_measurements else 0,
                profile.total_sessions,
                profile.avg_session_duration,
                len(profile.preferred_interventions),
                profile.engagement_score
            ]
            features.append(user_features)
            user_ids.append(user_id)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Perform clustering
        optimal_clusters = min(5, len(user_profiles) // 10)  # Max 5 clusters, min 10 users per cluster
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Create segment insights
        segments = []
        for cluster_id in range(optimal_clusters):
            cluster_users = [user_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if len(cluster_users) < 5:  # Skip small clusters
                continue
            
            segment = self._analyze_user_cluster(cluster_id, cluster_users, user_profiles)
            segments.append(segment)
        
        return segments
    
    def _analyze_user_cluster(self, cluster_id: int, user_ids: List[str], 
                             user_profiles: Dict[str, UserOutcomeProfile]) -> UserSegmentInsight:
        """Analyze a specific user cluster"""
        
        cluster_profiles = [user_profiles[uid] for uid in user_ids if uid in user_profiles]
        
        # Calculate segment characteristics
        avg_outcomes = {}
        if cluster_profiles:
            # Calculate average outcomes across different metrics
            all_measurements = []
            for profile in cluster_profiles:
                all_measurements.extend(profile.outcome_measurements)
            
            if all_measurements:
                # Group by outcome type if available
                outcome_groups = defaultdict(list)
                for measurement in all_measurements:
                    outcome_type = getattr(measurement, 'outcome_type', 'general')
                    outcome_groups[outcome_type].append(measurement.primary_score)
                
                for outcome_type, scores in outcome_groups.items():
                    avg_outcomes[outcome_type] = np.mean(scores)
        
        # Identify preferred interventions
        all_interventions = []
        for profile in cluster_profiles:
            all_interventions.extend(profile.preferred_interventions)
        
        intervention_counts = Counter(all_interventions)
        preferred_interventions = [intervention for intervention, count in intervention_counts.most_common(3)]
        
        # Identify common patterns
        common_patterns = []
        avg_sessions = np.mean([p.total_sessions for p in cluster_profiles])
        avg_duration = np.mean([p.avg_session_duration for p in cluster_profiles])
        avg_engagement = np.mean([p.engagement_score for p in cluster_profiles])
        
        if avg_sessions > 10:
            common_patterns.append("high_activity")
        if avg_duration > 30:
            common_patterns.append("long_sessions")
        if avg_engagement > 0.7:
            common_patterns.append("highly_engaged")
        
        # Generate segment-specific recommendations
        recommendations = []
        if avg_engagement < 0.5:
            recommendations.append("Focus on engagement improvement strategies for this segment")
        if avg_outcomes.get('general', 0.5) < 0.6:
            recommendations.append("Implement targeted interventions to improve outcomes")
        if avg_sessions < 5:
            recommendations.append("Encourage more frequent interactions with this user group")
        
        segment_characteristics = {
            'avg_sessions': avg_sessions,
            'avg_session_duration': avg_duration,
            'avg_engagement': avg_engagement,
            'primary_patterns': common_patterns
        }
        
        return UserSegmentInsight(
            segment_id=f"segment_{cluster_id}",
            segment_characteristics=segment_characteristics,
            user_count=len(user_ids),
            avg_outcomes=avg_outcomes,
            preferred_interventions=preferred_interventions,
            common_patterns=common_patterns,
            specific_recommendations=recommendations
        )
    
    def _create_user_segment_insight(self, segment: UserSegmentInsight) -> Optional[SystemInsight]:
        """Create insight from user segment analysis"""
        
        if segment.user_count < 5:  # Skip insights for very small segments
            return None
        
        # Determine priority based on segment characteristics
        avg_outcome = segment.avg_outcomes.get('general', 0.5)
        engagement_level = segment.segment_characteristics.get('avg_engagement', 0.5)
        
        if avg_outcome < 0.4 or engagement_level < 0.3:
            priority = InsightPriority.HIGH
        elif avg_outcome > 0.8 and engagement_level > 0.8:
            priority = InsightPriority.MEDIUM  # Success story to replicate
        else:
            priority = InsightPriority.LOW
        
        title = f"User segment '{segment.segment_id}' analysis"
        description = (f"Identified user segment with {segment.user_count} users showing "
                      f"average engagement of {engagement_level:.2f} and outcomes of {avg_outcome:.2f}")
        
        actionable_items = [
            f"Customize interventions for {segment.user_count} users in this segment",
            f"Focus on preferred interventions: {', '.join(segment.preferred_interventions[:2])}",
            "Monitor segment performance metrics weekly"
        ]
        
        return SystemInsight(
            insight_id=f"user_segment_{segment.segment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.USER_BEHAVIOR,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'segment_data': asdict(segment),
                'key_metrics': {
                    'avg_outcome': avg_outcome,
                    'engagement_level': engagement_level,
                    'user_count': segment.user_count
                }
            },
            confidence_score=min(1.0, segment.user_count / 50.0),  # Higher confidence for larger segments
            recommendations=segment.specific_recommendations,
            actionable_items=actionable_items,
            data_source=['outcome_tracker', 'personalization_engine'],
            affected_components=['personalization', 'intervention_selection']
        )
    
    async def _identify_behavioral_anomalies(self, user_profiles: Dict[str, UserOutcomeProfile]) -> List[SystemInsight]:
        """Identify anomalous user behavior patterns"""
        
        insights = []
        
        try:
            if len(user_profiles) < self.min_data_points:
                return insights
            
            # Prepare anomaly detection features
            features = []
            user_ids = []
            
            for user_id, profile in user_profiles.items():
                user_features = [
                    profile.total_sessions,
                    profile.avg_session_duration,
                    profile.engagement_score,
                    len(profile.outcome_measurements),
                    np.mean([m.primary_score for m in profile.outcome_measurements]) if profile.outcome_measurements else 0,
                    len(profile.preferred_interventions)
                ]
                features.append(user_features)
                user_ids.append(user_id)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features)
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_users = [user_ids[i] for i in anomaly_indices]
                
                insight = SystemInsight(
                    insight_id=f"behavioral_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type=InsightType.ANOMALY_DETECTION,
                    priority=InsightPriority.HIGH,
                    title=f"Unusual behavior patterns detected in {len(anomaly_users)} users",
                    description=(f"Identified {len(anomaly_users)} users with anomalous behavior patterns "
                               f"that differ significantly from typical usage"),
                    supporting_data={
                        'anomaly_user_count': len(anomaly_users),
                        'total_users_analyzed': len(user_profiles),
                        'anomaly_rate': len(anomaly_users) / len(user_profiles)
                    },
                    confidence_score=0.8,
                    recommendations=[
                        "Investigate anomalous users for potential issues or unique needs",
                        "Consider personalized intervention strategies for these users",
                        "Monitor for system bugs that might cause unusual behavior patterns",
                        "Use anomalous patterns to identify edge cases for system improvement"
                    ],
                    data_source=['outcome_tracker'],
                    affected_components=['user_experience', 'personalization']
                )
                
                insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error in behavioral anomaly detection: {str(e)}")
        
        return insights
    
    async def _analyze_intervention_effectiveness(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze effectiveness of different interventions"""
        
        insights = []
        
        try:
            # Collect intervention data
            intervention_data = await self._collect_intervention_data(time_period_days)
            
            for intervention_type, analysis in intervention_data.items():
                if analysis['sample_size'] >= self.min_data_points:
                    insight = self._create_intervention_insight(intervention_type, analysis)
                    if insight:
                        insights.append(insight)
            
            self.logger.info(f"Generated {len(insights)} intervention effectiveness insights")
            
        except Exception as e:
            self.logger.error(f"Error in intervention effectiveness analysis: {str(e)}")
        
        return insights
    
    async def _collect_intervention_data(self, time_period_days: int) -> Dict[str, InterventionAnalysis]:
        """Collect and analyze intervention effectiveness data"""
        
        intervention_analyses = {}
        
        # Get data from outcome tracker
        all_profiles = self.outcome_tracker.user_profiles
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        
        # Group interventions by type
        intervention_outcomes = defaultdict(list)
        
        for profile in all_profiles.values():
            for intervention in profile.preferred_interventions:
                # Get recent outcome measurements for this user
                recent_outcomes = [
                    m for m in profile.outcome_measurements
                    if m.timestamp >= cutoff_date
                ]
                
                if recent_outcomes:
                    avg_score = np.mean([m.primary_score for m in recent_outcomes])
                    intervention_outcomes[intervention].append({
                        'user_id': profile.user_id,
                        'outcome_score': avg_score,
                        'engagement': profile.engagement_score,
                        'sessions': profile.total_sessions
                    })
        
        # Analyze each intervention type
        for intervention_type, outcomes in intervention_outcomes.items():
            if len(outcomes) >= 5:  # Minimum sample size
                analysis = self._calculate_intervention_effectiveness(intervention_type, outcomes)
                intervention_analyses[intervention_type] = analysis
        
        return intervention_analyses
    
    def _calculate_intervention_effectiveness(self, intervention_type: str, 
                                           outcomes: List[Dict[str, Any]]) -> InterventionAnalysis:
        """Calculate effectiveness metrics for an intervention"""
        
        outcome_scores = [o['outcome_score'] for o in outcomes]
        engagement_scores = [o['engagement'] for o in outcomes]
        
        success_rate = len([s for s in outcome_scores if s > 0.6]) / len(outcome_scores)
        avg_satisfaction = np.mean(outcome_scores)
        avg_engagement = np.mean(engagement_scores)
        
        # Identify optimal conditions (simplified)
        high_performing_users = [o for o in outcomes if o['outcome_score'] > 0.7]
        optimal_conditions = {}
        
        if high_performing_users:
            optimal_conditions = {
                'avg_sessions': np.mean([u['sessions'] for u in high_performing_users]),
                'avg_engagement': np.mean([u['engagement'] for u in high_performing_users])
            }
        
        # Generate improvement suggestions
        improvement_suggestions = []
        if success_rate < 0.6:
            improvement_suggestions.append("Consider modifying intervention approach to improve success rate")
        if avg_satisfaction < 0.5:
            improvement_suggestions.append("Focus on user satisfaction improvements")
        if avg_engagement < 0.6:
            improvement_suggestions.append("Enhance engagement strategies for this intervention")
        
        # Calculate statistical confidence (simplified)
        std_dev = np.std(outcome_scores)
        statistical_confidence = min(1.0, len(outcomes) / 50.0) * (1 - std_dev)
        
        return InterventionAnalysis(
            intervention_type=intervention_type,
            success_rate=success_rate,
            avg_user_satisfaction=avg_satisfaction,
            optimal_conditions=optimal_conditions,
            user_segments_most_effective=[],  # Would need more complex analysis
            improvement_suggestions=improvement_suggestions,
            statistical_confidence=statistical_confidence
        )
    
    def _create_intervention_insight(self, intervention_type: str, 
                                   analysis: InterventionAnalysis) -> Optional[SystemInsight]:
        """Create insight from intervention analysis"""
        
        if analysis.success_rate > 0.8:
            priority = InsightPriority.HIGH
            title = f"High-performing intervention: {intervention_type}"
            description = f"{intervention_type} shows excellent results with {analysis.success_rate:.1%} success rate"
            recommendations = [
                f"Scale up {intervention_type} usage across suitable user segments",
                "Document best practices for this intervention",
                "Train other agents on successful implementation techniques"
            ]
        elif analysis.success_rate < 0.4:
            priority = InsightPriority.CRITICAL
            title = f"Underperforming intervention: {intervention_type}"
            description = f"{intervention_type} shows low success rate of {analysis.success_rate:.1%}"
            recommendations = analysis.improvement_suggestions + [
                f"Consider reducing {intervention_type} usage until improvements are made",
                "Investigate root causes of poor performance",
                "Develop alternative intervention strategies"
            ]
        else:
            priority = InsightPriority.MEDIUM
            title = f"Intervention analysis: {intervention_type}"
            description = f"{intervention_type} shows moderate performance with {analysis.success_rate:.1%} success rate"
            recommendations = analysis.improvement_suggestions + [
                f"Monitor {intervention_type} performance for improvement opportunities",
                "A/B test variations of this intervention"
            ]
        
        return SystemInsight(
            insight_id=f"intervention_{intervention_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.INTERVENTION_EFFECTIVENESS,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'intervention_analysis': asdict(analysis),
                'key_metrics': {
                    'success_rate': analysis.success_rate,
                    'avg_satisfaction': analysis.avg_user_satisfaction,
                    'statistical_confidence': analysis.statistical_confidence
                }
            },
            confidence_score=analysis.statistical_confidence,
            recommendations=recommendations,
            data_source=['outcome_tracker', 'personalization_engine'],
            affected_components=['intervention_selection', 'agent_training']
        )
    
    async def _analyze_agent_performance(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze individual agent performance"""
        
        insights = []
        
        # This would typically interface with actual agent performance metrics
        # For now, create simulated agent performance insights
        
        try:
            simulated_agents = ['diagnosis_agent', 'therapy_agent', 'crisis_agent', 'wellness_agent']
            
            for agent_name in simulated_agents:
                # Simulate agent performance metrics
                performance_data = self._simulate_agent_performance_data(agent_name)
                
                if performance_data:
                    insight = self._create_agent_performance_insight(agent_name, performance_data)
                    if insight:
                        insights.append(insight)
            
            self.logger.info(f"Generated {len(insights)} agent performance insights")
            
        except Exception as e:
            self.logger.error(f"Error in agent performance analysis: {str(e)}")
        
        return insights
    
    def _simulate_agent_performance_data(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Simulate agent performance data for demonstration"""
        
        # In real implementation, this would pull actual agent metrics
        base_performance = {
            'diagnosis_agent': {'accuracy': 0.85, 'response_time': 1.2, 'user_satisfaction': 0.78},
            'therapy_agent': {'accuracy': 0.88, 'response_time': 0.9, 'user_satisfaction': 0.82},
            'crisis_agent': {'accuracy': 0.92, 'response_time': 0.7, 'user_satisfaction': 0.85},
            'wellness_agent': {'accuracy': 0.80, 'response_time': 1.1, 'user_satisfaction': 0.75}
        }
        
        return base_performance.get(agent_name)
    
    def _create_agent_performance_insight(self, agent_name: str, 
                                        performance_data: Dict[str, Any]) -> Optional[SystemInsight]:
        """Create insight from agent performance data"""
        
        accuracy = performance_data.get('accuracy', 0.5)
        response_time = performance_data.get('response_time', 1.0)
        satisfaction = performance_data.get('user_satisfaction', 0.5)
        
        # Overall performance score
        performance_score = (accuracy + satisfaction + (2.0 - response_time) / 2.0) / 2.5
        
        if performance_score > 0.8:
            priority = InsightPriority.MEDIUM
            title = f"High performance detected: {agent_name}"
            description = f"{agent_name} showing excellent performance across key metrics"
            recommendations = [
                f"Use {agent_name} as a model for other agent improvements",
                "Document successful strategies used by this agent",
                "Consider expanding this agent's responsibilities"
            ]
        elif performance_score < 0.6:
            priority = InsightPriority.HIGH
            title = f"Performance improvement needed: {agent_name}"
            description = f"{agent_name} showing below-average performance metrics"
            recommendations = [
                f"Investigate performance issues with {agent_name}",
                "Provide additional training or model updates",
                "Review agent configuration and parameters",
                "Consider temporary workload reduction while improvements are made"
            ]
        else:
            return None  # Skip moderate performance agents
        
        return SystemInsight(
            insight_id=f"agent_performance_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.AGENT_OPTIMIZATION,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'agent_name': agent_name,
                'performance_metrics': performance_data,
                'overall_score': performance_score
            },
            confidence_score=0.9,  # High confidence in performance metrics
            recommendations=recommendations,
            data_source=['agent_orchestrator'],
            affected_components=[agent_name, 'agent_training']
        )
    
    async def _generate_predictive_insights(self, time_period_days: int) -> List[SystemInsight]:
        """Generate predictive insights using machine learning models"""
        
        insights = []
        
        try:
            # Predict user outcome trends
            outcome_predictions = await self._predict_user_outcomes()
            if outcome_predictions:
                insights.extend(outcome_predictions)
            
            # Predict system resource needs
            resource_predictions = await self._predict_resource_needs()
            if resource_predictions:
                insights.extend(resource_predictions)
            
            self.logger.info(f"Generated {len(insights)} predictive insights")
            
        except Exception as e:
            self.logger.error(f"Error in predictive insight generation: {str(e)}")
        
        return insights
    
    async def _predict_user_outcomes(self) -> List[SystemInsight]:
        """Predict future user outcomes based on current patterns"""
        
        insights = []
        
        try:
            user_profiles = self.outcome_tracker.user_profiles
            
            if len(user_profiles) < self.min_data_points:
                return insights
            
            # Prepare prediction data
            features = []
            outcomes = []
            
            for profile in user_profiles.values():
                if len(profile.outcome_measurements) >= 3:
                    # Use recent measurements as features
                    recent_measurements = sorted(profile.outcome_measurements, key=lambda x: x.timestamp)[-3:]
                    feature_vector = [
                        profile.total_sessions,
                        profile.avg_session_duration,
                        profile.engagement_score,
                        len(profile.preferred_interventions),
                        np.mean([m.primary_score for m in recent_measurements[:-1]]),  # Exclude last as target
                    ]
                    features.append(feature_vector)
                    outcomes.append(recent_measurements[-1].primary_score)  # Predict this
            
            if len(features) >= 10:  # Minimum for prediction
                # Train simple prediction model
                predictor = RandomForestRegressor(n_estimators=10, random_state=42)
                predictor.fit(features, outcomes)
                
                # Make predictions for current users
                predictions = predictor.predict(features)
                
                # Identify users at risk of poor outcomes
                at_risk_users = []
                for i, prediction in enumerate(predictions):
                    if prediction < 0.4:  # Predicted poor outcome
                        at_risk_users.append(i)
                
                if len(at_risk_users) > 0:
                    insight = SystemInsight(
                        insight_id=f"outcome_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type=InsightType.PREDICTIVE_MODEL,
                        priority=InsightPriority.HIGH,
                        title=f"Predicted poor outcomes for {len(at_risk_users)} users",
                        description=f"ML model predicts {len(at_risk_users)} users at risk of poor outcomes",
                        supporting_data={
                            'at_risk_count': len(at_risk_users),
                            'total_analyzed': len(predictions),
                            'model_accuracy': 'estimated 0.75',  # Would need proper validation
                            'prediction_horizon': '1-2 weeks'
                        },
                        confidence_score=0.75,
                        recommendations=[
                            "Proactively reach out to at-risk users",
                            "Adjust intervention strategies for predicted poor outcomes",
                            "Increase monitoring frequency for these users",
                            "Consider escalating care for high-risk cases"
                        ],
                        actionable_items=[
                            "Generate priority list of at-risk users",
                            "Alert appropriate care coordinators",
                            "Schedule follow-up interventions"
                        ],
                        data_source=['outcome_tracker', 'ml_prediction_model'],
                        affected_components=['user_monitoring', 'intervention_planning']
                    )
                    insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error in user outcome prediction: {str(e)}")
        
        return insights
    
    async def _predict_resource_needs(self) -> List[SystemInsight]:
        """Predict future system resource needs"""
        
        insights = []
        
        # Simplified resource prediction based on usage trends
        try:
            feedback_stats = self.feedback_processor.get_feedback_statistics()
            current_load = feedback_stats.get('total_feedback_count', 0)
            
            if current_load > 1000:  # High usage threshold
                insight = SystemInsight(
                    insight_id=f"resource_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type=InsightType.PREDICTIVE_MODEL,
                    priority=InsightPriority.MEDIUM,
                    title="High system usage detected - resource scaling recommended",
                    description=f"Current usage at {current_load} interactions suggests need for resource scaling",
                    supporting_data={
                        'current_load': current_load,
                        'prediction': 'scaling_needed',
                        'confidence': 0.8
                    },
                    confidence_score=0.8,
                    recommendations=[
                        "Plan for system resource scaling within 2 weeks",
                        "Monitor system performance closely",
                        "Consider load balancing improvements",
                        "Prepare auto-scaling configurations"
                    ],
                    data_source=['feedback_processor', 'system_metrics'],
                    affected_components=['system_infrastructure']
                )
                insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error in resource need prediction: {str(e)}")
        
        return insights
    
    async def _detect_system_anomalies(self, time_period_days: int) -> List[SystemInsight]:
        """Detect system-level anomalies"""
        
        insights = []
        
        try:
            # Check for unusual patterns in feedback
            feedback_anomalies = await self._detect_feedback_anomalies(time_period_days)
            insights.extend(feedback_anomalies)
            
            # Check for unusual patterns in outcomes
            outcome_anomalies = await self._detect_outcome_anomalies(time_period_days)
            insights.extend(outcome_anomalies)
            
            self.logger.info(f"Generated {len(insights)} anomaly detection insights")
            
        except Exception as e:
            self.logger.error(f"Error in system anomaly detection: {str(e)}")
        
        return insights
    
    async def _detect_feedback_anomalies(self, time_period_days: int) -> List[SystemInsight]:
        """Detect anomalies in feedback patterns"""
        
        insights = []
        
        try:
            feedback_stats = self.feedback_processor.get_feedback_statistics()
            
            # Check for sudden drops in feedback volume
            recent_feedback = feedback_stats.get('recent_feedback_count', 0)
            historical_avg = feedback_stats.get('historical_avg_feedback', recent_feedback)
            
            if recent_feedback < historical_avg * 0.5 and historical_avg > 10:
                insight = SystemInsight(
                    insight_id=f"feedback_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type=InsightType.ANOMALY_DETECTION,
                    priority=InsightPriority.HIGH,
                    title="Significant drop in user feedback detected",
                    description=f"Recent feedback volume ({recent_feedback}) is {((historical_avg - recent_feedback) / historical_avg * 100):.1f}% below historical average",
                    supporting_data={
                        'recent_feedback_count': recent_feedback,
                        'historical_average': historical_avg,
                        'percentage_drop': (historical_avg - recent_feedback) / historical_avg * 100
                    },
                    confidence_score=0.9,
                    recommendations=[
                        "Investigate potential system issues preventing feedback submission",
                        "Check if user engagement has decreased",
                        "Review recent system changes that might affect feedback flow",
                        "Proactively request feedback from recent users"
                    ],
                    data_source=['feedback_processor'],
                    affected_components=['feedback_system', 'user_engagement']
                )
                insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error in feedback anomaly detection: {str(e)}")
        
        return insights
    
    async def _detect_outcome_anomalies(self, time_period_days: int) -> List[SystemInsight]:
        """Detect anomalies in user outcomes"""
        
        insights = []
        
        try:
            outcome_stats = self.outcome_tracker.get_system_statistics()
            
            # Check for unusual outcome distributions
            if 'outcome_distribution' in outcome_stats:
                distribution = outcome_stats['outcome_distribution']
                
                # Check for too many extremely low outcomes
                low_outcomes_rate = distribution.get('very_low', 0) / sum(distribution.values()) if distribution else 0
                
                if low_outcomes_rate > 0.3:  # More than 30% very low outcomes
                    insight = SystemInsight(
                        insight_id=f"outcome_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type=InsightType.ANOMALY_DETECTION,
                        priority=InsightPriority.CRITICAL,
                        title="Unusual pattern of low user outcomes detected",
                        description=f"High rate of very low outcomes ({low_outcomes_rate:.1%}) detected",
                        supporting_data={
                            'low_outcomes_rate': low_outcomes_rate,
                            'outcome_distribution': distribution,
                            'total_outcomes_analyzed': sum(distribution.values())
                        },
                        confidence_score=0.95,
                        recommendations=[
                            "Immediate investigation required for poor outcome trend",
                            "Review recent intervention strategies",
                            "Check for system bugs affecting outcome calculations",
                            "Consider emergency protocol activation if pattern continues",
                            "Increase supervision and quality assurance measures"
                        ],
                        data_source=['outcome_tracker'],
                        affected_components=['outcome_measurement', 'intervention_system', 'quality_assurance']
                    )
                    insights.append(insight)
        
        except Exception as e:
            self.logger.error(f"Error in outcome anomaly detection: {str(e)}")
        
        return insights
    
    async def _analyze_therapeutic_patterns(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze therapeutic patterns and effectiveness"""
        
        insights = []
        
        try:
            # This would interface with the pattern recognition engine
            if hasattr(self.pattern_engine, 'discovered_patterns'):
                patterns = self.pattern_engine.discovered_patterns
                
                # Analyze pattern effectiveness
                for pattern in patterns[-10:]:  # Analyze recent patterns
                    pattern_insight = self._create_therapeutic_pattern_insight(pattern)
                    if pattern_insight:
                        insights.append(pattern_insight)
            
            self.logger.info(f"Generated {len(insights)} therapeutic pattern insights")
            
        except Exception as e:
            self.logger.error(f"Error in therapeutic pattern analysis: {str(e)}")
        
        return insights
    
    def _create_therapeutic_pattern_insight(self, pattern: Any) -> Optional[SystemInsight]:
        """Create insight from therapeutic pattern"""
        
        # This is a simplified implementation
        # Real implementation would analyze actual pattern data
        
        pattern_effectiveness = getattr(pattern, 'effectiveness_score', 0.5)
        pattern_type = getattr(pattern, 'pattern_type', 'general')
        
        if pattern_effectiveness > 0.8:
            priority = InsightPriority.HIGH
            title = f"Highly effective therapeutic pattern identified: {pattern_type}"
            description = f"Pattern shows {pattern_effectiveness:.1%} effectiveness rate"
            recommendations = [
                f"Scale successful {pattern_type} pattern across suitable users",
                "Document pattern for training purposes",
                "Integrate pattern into standard intervention protocols"
            ]
        else:
            return None  # Skip less effective patterns
        
        return SystemInsight(
            insight_id=f"therapeutic_pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.THERAPEUTIC_PATTERN,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'pattern_type': pattern_type,
                'effectiveness_score': pattern_effectiveness
            },
            confidence_score=pattern_effectiveness,
            recommendations=recommendations,
            data_source=['pattern_recognition_engine'],
            affected_components=['intervention_protocols', 'agent_training']
        )
    
    async def _analyze_system_health(self, time_period_days: int) -> List[SystemInsight]:
        """Analyze overall system health"""
        
        insights = []
        
        try:
            # Collect system health metrics
            health_metrics = {
                'total_users': len(self.outcome_tracker.user_profiles),
                'active_users': len([p for p in self.outcome_tracker.user_profiles.values() 
                                   if p.total_sessions > 0]),
                'avg_satisfaction': self._calculate_avg_satisfaction(),
                'system_uptime': 0.998,  # Simulated
                'response_time': 1.2,    # Simulated
                'error_rate': 0.02       # Simulated
            }
            
            # Generate health insights
            health_insight = self._create_system_health_insight(health_metrics)
            if health_insight:
                insights.append(health_insight)
            
            self.logger.info(f"Generated {len(insights)} system health insights")
            
        except Exception as e:
            self.logger.error(f"Error in system health analysis: {str(e)}")
        
        return insights
    
    def _calculate_avg_satisfaction(self) -> float:
        """Calculate average user satisfaction across all users"""
        
        all_scores = []
        for profile in self.outcome_tracker.user_profiles.values():
            if profile.outcome_measurements:
                user_avg = np.mean([m.primary_score for m in profile.outcome_measurements])
                all_scores.append(user_avg)
        
        return np.mean(all_scores) if all_scores else 0.5
    
    def _create_system_health_insight(self, metrics: Dict[str, Any]) -> Optional[SystemInsight]:
        """Create system health insight"""
        
        # Calculate overall health score
        health_score = (
            min(1.0, metrics['avg_satisfaction']) * 0.3 +
            min(1.0, metrics['system_uptime']) * 0.2 +
            min(1.0, max(0.0, (2.0 - metrics['response_time']) / 2.0)) * 0.2 +
            min(1.0, max(0.0, 1.0 - metrics['error_rate'] * 10)) * 0.2 +
            min(1.0, metrics['active_users'] / max(1, metrics['total_users'])) * 0.1
        )
        
        if health_score > 0.8:
            priority = InsightPriority.MEDIUM
            title = "System health is excellent"
            description = f"Overall system health score: {health_score:.2f}"
            recommendations = [
                "Maintain current operational practices",
                "Continue monitoring key metrics",
                "Document successful strategies for future reference"
            ]
        elif health_score < 0.6:
            priority = InsightPriority.CRITICAL
            title = "System health needs attention"
            description = f"Overall system health score is low: {health_score:.2f}"
            recommendations = [
                "Immediate investigation of system performance issues",
                "Review all system components for problems",
                "Implement emergency procedures if necessary",
                "Increase monitoring frequency until issues are resolved"
            ]
        else:
            priority = InsightPriority.MEDIUM
            title = "System health is moderate"
            description = f"Overall system health score: {health_score:.2f}"
            recommendations = [
                "Monitor system health trends closely",
                "Identify specific areas for improvement",
                "Implement gradual optimization measures"
            ]
        
        return SystemInsight(
            insight_id=f"system_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=InsightType.SYSTEM_HEALTH,
            priority=priority,
            title=title,
            description=description,
            supporting_data={
                'health_metrics': metrics,
                'overall_score': health_score,
                'component_scores': {
                    'satisfaction': metrics['avg_satisfaction'],
                    'uptime': metrics['system_uptime'],
                    'response_time': metrics['response_time'],
                    'error_rate': metrics['error_rate']
                }
            },
            confidence_score=0.9,
            recommendations=recommendations,
            data_source=['system_monitoring', 'outcome_tracker', 'feedback_processor'],
            affected_components=['entire_system']
        )
    
    def _update_insight_history(self, insights: List[SystemInsight]):
        """Update insight history for tracking"""
        
        self.insight_history.extend(insights)
        
        # Keep only recent insights
        cutoff_date = datetime.now() - timedelta(days=self.insight_retention_days)
        self.insight_history = [
            insight for insight in self.insight_history
            if insight.generated_at >= cutoff_date
        ]
    
    def _create_insights_summary(self, insights: List[SystemInsight]) -> Dict[str, Any]:
        """Create summary of generated insights"""
        
        summary = {
            'total_insights': len(insights),
            'by_priority': {
                'critical': len([i for i in insights if i.priority == InsightPriority.CRITICAL]),
                'high': len([i for i in insights if i.priority == InsightPriority.HIGH]),
                'medium': len([i for i in insights if i.priority == InsightPriority.MEDIUM]),
                'low': len([i for i in insights if i.priority == InsightPriority.LOW])
            },
            'by_type': {},
            'avg_confidence': np.mean([i.confidence_score for i in insights]) if insights else 0,
            'actionable_items': sum(len(i.actionable_items) for i in insights),
            'recommendations': sum(len(i.recommendations) for i in insights)
        }
        
        # Count by type
        for insight_type in InsightType:
            count = len([i for i in insights if i.insight_type == insight_type])
            if count > 0:
                summary['by_type'][insight_type.value] = count
        
        return summary
    
    def get_insights_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive insights dashboard"""
        
        return {
            'summary': {
                'total_insights_generated': len(self.generated_insights),
                'insights_in_history': len(self.insight_history),
                'avg_confidence_score': np.mean([i.confidence_score for i in self.generated_insights]) if self.generated_insights else 0,
                'critical_insights_count': len([i for i in self.generated_insights if i.priority == InsightPriority.CRITICAL])
            },
            'recent_insights': [
                {
                    'title': insight.title,
                    'priority': insight.priority.value,
                    'type': insight.insight_type.value,
                    'confidence': insight.confidence_score,
                    'generated_at': insight.generated_at.isoformat(),
                    'implemented': insight.implemented
                }
                for insight in sorted(self.generated_insights, key=lambda x: x.generated_at, reverse=True)[:10]
            ],
            'insight_trends': self._calculate_insight_trends(),
            'implementation_status': self._calculate_implementation_status(),
            'top_recommendations': self._get_top_recommendations()
        }
    
    def _calculate_insight_trends(self) -> Dict[str, Any]:
        """Calculate trends in insight generation"""
        
        if len(self.insight_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Group insights by week
        weekly_counts = defaultdict(int)
        for insight in self.insight_history:
            week = insight.generated_at.strftime('%Y-W%U')
            weekly_counts[week] += 1
        
        recent_weeks = list(weekly_counts.values())[-4:]  # Last 4 weeks
        trend = "stable"
        
        if len(recent_weeks) >= 2:
            if recent_weeks[-1] > recent_weeks[0] * 1.2:
                trend = "increasing"
            elif recent_weeks[-1] < recent_weeks[0] * 0.8:
                trend = "decreasing"
        
        return {
            'trend_direction': trend,
            'weekly_average': np.mean(recent_weeks),
            'peak_week_count': max(weekly_counts.values()) if weekly_counts else 0
        }
    
    def _calculate_implementation_status(self) -> Dict[str, Any]:
        """Calculate implementation status of insights"""
        
        if not self.generated_insights:
            return {'status': 'no_insights'}
        
        implemented = len([i for i in self.generated_insights if i.implemented])
        total = len(self.generated_insights)
        implementation_rate = implemented / total
        
        return {
            'implementation_rate': implementation_rate,
            'implemented_count': implemented,
            'pending_count': total - implemented,
            'avg_time_to_implementation': 'not_calculated'  # Would need tracking
        }
    
    def _get_top_recommendations(self) -> List[str]:
        """Get most common recommendations across insights"""
        
        all_recommendations = []
        for insight in self.generated_insights:
            all_recommendations.extend(insight.recommendations)
        
        recommendation_counts = Counter(all_recommendations)
        return [rec for rec, count in recommendation_counts.most_common(5)]
    
    def export_insights_report(self, format: str = "json") -> str:
        """Export comprehensive insights report"""
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_insights',
                'format': format,
                'total_insights': len(self.generated_insights)
            },
            'executive_summary': self._create_insights_summary(self.generated_insights),
            'insights': [asdict(insight) for insight in self.generated_insights],
            'dashboard': self.get_insights_dashboard(),
            'recommendations_summary': {
                'critical_actions': [
                    insight.actionable_items for insight in self.generated_insights
                    if insight.priority == InsightPriority.CRITICAL
                ],
                'high_priority_recommendations': [
                    insight.recommendations for insight in self.generated_insights
                    if insight.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]
                ]
            }
        }
        
        if format.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        else:
            # Could implement other formats (CSV, HTML, etc.)
            return json.dumps(report_data, indent=2, default=str)
    
    def mark_insight_implemented(self, insight_id: str, measured_impact: Dict[str, Any] = None):
        """Mark an insight as implemented with optional impact measurement"""
        
        for insight in self.generated_insights:
            if insight.insight_id == insight_id:
                insight.implemented = True
                insight.implementation_date = datetime.now()
                if measured_impact:
                    insight.measured_impact = measured_impact
                self.logger.info(f"Marked insight {insight_id} as implemented")
                break
    
    async def cleanup_old_insights(self):
        """Clean up old insights to manage memory"""
        
        cutoff_date = datetime.now() - timedelta(days=self.insight_retention_days)
        
        original_count = len(self.insight_history)
        self.insight_history = [
            insight for insight in self.insight_history
            if insight.generated_at >= cutoff_date
        ]
        
        cleaned_count = original_count - len(self.insight_history)
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old insights")
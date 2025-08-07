"""
Long-term User Outcomes and Satisfaction Metrics Tracker

This module implements comprehensive tracking and analysis of long-term user outcomes,
satisfaction metrics, and therapeutic progress. It provides predictive analytics,
trend analysis, and automated alerts for significant changes in user wellbeing.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..diagnosis.adaptive_learning import InterventionOutcome, UserProfile
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OutcomeType(Enum):
    """Types of outcome measurements"""
    SYMPTOM_SEVERITY = "symptom_severity"
    FUNCTIONAL_IMPROVEMENT = "functional_improvement"
    QUALITY_OF_LIFE = "quality_of_life"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    MEDICATION_ADHERENCE = "medication_adherence"
    CRISIS_FREQUENCY = "crisis_frequency"
    SOCIAL_FUNCTIONING = "social_functioning"
    WORK_PERFORMANCE = "work_performance"
    SLEEP_QUALITY = "sleep_quality"
    MOOD_STABILITY = "mood_stability"
    ANXIETY_LEVELS = "anxiety_levels"
    COPING_SKILLS = "coping_skills"

class AlertSeverity(Enum):
    """Severity levels for outcome alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class TrendDirection(Enum):
    """Direction of outcome trends"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    FLUCTUATING = "fluctuating"
    UNKNOWN = "unknown"

@dataclass
class OutcomeMeasurement:
    """Individual outcome measurement"""
    measurement_id: str
    user_id: str
    outcome_type: OutcomeType
    measurement_date: datetime
    
    # Core measurements
    primary_score: float  # 0.0 to 10.0 scale
    secondary_scores: Dict[str, float] = field(default_factory=dict)
    
    # Context and metadata
    measurement_context: Dict[str, Any] = field(default_factory=dict)
    data_source: str = "user_report"  # user_report, clinical_assessment, behavioral_data
    confidence: float = 1.0
    notes: str = ""
    
    # Clinical context
    current_interventions: List[str] = field(default_factory=list)
    medication_changes: List[str] = field(default_factory=list)
    life_events: List[str] = field(default_factory=list)
    
    # Validity indicators
    is_baseline: bool = False
    is_follow_up: bool = False
    days_since_baseline: Optional[int] = None
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    alert_generated: bool = False

@dataclass
class UserOutcomeProfile:
    """Comprehensive outcome profile for a user"""
    user_id: str
    profile_created_at: datetime
    last_updated: datetime
    
    # Baseline measurements
    baseline_measurements: Dict[OutcomeType, OutcomeMeasurement] = field(default_factory=dict)
    baseline_date: Optional[datetime] = None
    
    # Current status
    current_measurements: Dict[OutcomeType, OutcomeMeasurement] = field(default_factory=dict)
    overall_trajectory: TrendDirection = TrendDirection.UNKNOWN
    
    # Historical data
    measurement_history: Dict[OutcomeType, List[OutcomeMeasurement]] = field(default_factory=lambda: defaultdict(list))
    
    # Analytics
    trend_analysis: Dict[OutcomeType, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    
    # Risk indicators
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    current_risk_level: str = "low"  # low, moderate, high, critical
    
    # Predictive analytics
    predicted_outcomes: Dict[str, Any] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Milestones and achievements
    milestones_achieved: List[Dict[str, Any]] = field(default_factory=list)
    goals_status: Dict[str, str] = field(default_factory=dict)  # goal_id -> status
    
    # Quality metrics
    data_completeness: float = 0.0
    measurement_reliability: float = 0.0
    profile_confidence: float = 0.0

@dataclass
class OutcomeAlert:
    """Alert for significant changes in outcomes"""
    alert_id: str
    user_id: str
    severity: AlertSeverity
    alert_type: str
    
    # Alert details
    title: str
    description: str
    affected_outcomes: List[OutcomeType]
    
    # Evidence
    supporting_data: Dict[str, Any]
    statistical_significance: Optional[float] = None
    clinical_significance: Optional[float] = None
    
    # Context
    potential_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    requires_immediate_attention: bool = False
    
    # Resolution
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis for an outcome"""
    outcome_type: OutcomeType
    user_id: str
    analysis_period_days: int
    
    # Trend metrics
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    
    # Statistical measures
    mean_change: float
    standard_deviation: float
    coefficient_of_variation: float
    
    # Change points
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    change_point_dates: List[datetime] = field(default_factory=list)
    
    # Predictions
    projected_value_30_days: float = 0.0
    projection_confidence: float = 0.0
    
    # Context
    correlating_factors: List[str] = field(default_factory=list)
    intervention_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    data_points_analyzed: int = 0
    analysis_quality: float = 0.0

class OutcomeTracker:
    """
    Comprehensive long-term outcome tracking and analysis system.
    
    Tracks user outcomes over time, identifies trends, generates alerts,
    and provides predictive analytics for therapeutic interventions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Data storage
        self.user_profiles = {}  # user_id -> UserOutcomeProfile
        self.outcome_measurements = {}  # measurement_id -> OutcomeMeasurement
        self.alerts = {}  # alert_id -> OutcomeAlert
        self.trend_analyses = {}  # analysis_id -> TrendAnalysis
        
        # Configuration
        self.measurement_validity_days = self.config.get('measurement_validity_days', 30)
        self.min_measurements_for_trend = self.config.get('min_measurements_for_trend', 3)
        self.significance_threshold = self.config.get('significance_threshold', 0.05)
        self.clinical_significance_threshold = self.config.get('clinical_significance_threshold', 1.0)
        
        # Alert thresholds
        self.alert_thresholds = {
            'critical_decline': -2.0,  # 2+ point decline in primary score
            'moderate_decline': -1.0,
            'significant_improvement': 2.0,
            'crisis_threshold': 2.0,  # For crisis-related outcomes
            'instability_threshold': 3.0  # Standard deviations for instability
        }
        
        # Outcome weights and priorities
        self.outcome_priorities = {
            OutcomeType.CRISIS_FREQUENCY: 1.0,
            OutcomeType.SYMPTOM_SEVERITY: 0.9,
            OutcomeType.FUNCTIONAL_IMPROVEMENT: 0.8,
            OutcomeType.QUALITY_OF_LIFE: 0.7,
            OutcomeType.MOOD_STABILITY: 0.6,
            OutcomeType.ANXIETY_LEVELS: 0.6,
            OutcomeType.SOCIAL_FUNCTIONING: 0.5,
            OutcomeType.SLEEP_QUALITY: 0.4,
            OutcomeType.WORK_PERFORMANCE: 0.4,
            OutcomeType.COPING_SKILLS: 0.3,
            OutcomeType.THERAPEUTIC_ALLIANCE: 0.3,
            OutcomeType.MEDICATION_ADHERENCE: 0.2
        }
        
        # Analytics models
        self.trend_models = {}  # outcome_type -> trained models
        self.risk_assessment_model = None
        
        # Performance tracking
        self.tracking_metrics = defaultdict(list)
        
        # Load existing data
        self._load_persisted_data()
    
    async def record_outcome_measurement(self,
                                       user_id: str,
                                       outcome_type: OutcomeType,
                                       primary_score: float,
                                       context: Dict[str, Any] = None,
                                       secondary_scores: Dict[str, float] = None,
                                       data_source: str = "user_report") -> OutcomeMeasurement:
        """Record a new outcome measurement for a user"""
        
        try:
            self.logger.info(f"Recording {outcome_type.value} measurement for user {user_id}")
            
            # Create measurement
            measurement = OutcomeMeasurement(
                measurement_id=f"{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                outcome_type=outcome_type,
                measurement_date=datetime.now(),
                primary_score=max(0.0, min(10.0, primary_score)),  # Clamp to 0-10 scale
                secondary_scores=secondary_scores or {},
                measurement_context=context or {},
                data_source=data_source,
                processed_at=datetime.now()
            )
            
            # Get or create user profile
            profile = await self._get_or_create_user_profile(user_id)
            
            # Set baseline if this is the first measurement
            if outcome_type not in profile.baseline_measurements:
                measurement.is_baseline = True
                profile.baseline_measurements[outcome_type] = measurement
                if profile.baseline_date is None:
                    profile.baseline_date = measurement.measurement_date
                    
                self.logger.info(f"Set baseline {outcome_type.value} for user {user_id}: {primary_score}")
            else:
                # Calculate days since baseline
                baseline_date = profile.baseline_measurements[outcome_type].measurement_date
                measurement.days_since_baseline = (measurement.measurement_date - baseline_date).days
            
            # Store measurement
            self.outcome_measurements[measurement.measurement_id] = measurement
            profile.current_measurements[outcome_type] = measurement
            profile.measurement_history[outcome_type].append(measurement)
            profile.last_updated = datetime.now()
            
            # Perform analysis
            await self._analyze_measurement(measurement, profile)
            
            # Check for alerts
            alerts = await self._check_for_alerts(measurement, profile)
            
            # Update trend analysis
            await self._update_trend_analysis(user_id, outcome_type)
            
            # Update overall profile assessment
            await self._update_profile_assessment(profile)
            
            # Store any generated alerts
            for alert in alerts:
                self.alerts[alert.alert_id] = alert
                measurement.alert_generated = True
            
            return measurement
            
        except Exception as e:
            self.logger.error(f"Error recording outcome measurement: {str(e)}")
            raise
    
    async def _get_or_create_user_profile(self, user_id: str) -> UserOutcomeProfile:
        """Get existing user profile or create new one"""
        
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        profile = UserOutcomeProfile(
            user_id=user_id,
            profile_created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.user_profiles[user_id] = profile
        self.logger.info(f"Created new outcome profile for user {user_id}")
        
        return profile
    
    async def _analyze_measurement(self, measurement: OutcomeMeasurement, profile: UserOutcomeProfile):
        """Analyze a new measurement in context of user history"""
        
        outcome_type = measurement.outcome_type
        
        # Get historical measurements
        history = profile.measurement_history[outcome_type]
        
        if len(history) < 2:
            return  # Need at least 2 measurements for analysis
        
        # Calculate change from previous measurement
        previous_measurement = history[-2]  # Second to last
        change = measurement.primary_score - previous_measurement.primary_score
        
        # Calculate change from baseline
        if outcome_type in profile.baseline_measurements:
            baseline = profile.baseline_measurements[outcome_type]
            baseline_change = measurement.primary_score - baseline.primary_score
            
            # Store change metrics in measurement context
            measurement.measurement_context['change_from_previous'] = change
            measurement.measurement_context['change_from_baseline'] = baseline_change
            measurement.measurement_context['percent_change_from_baseline'] = (baseline_change / baseline.primary_score * 100) if baseline.primary_score > 0 else 0
        
        # Calculate volatility/stability
        if len(history) >= 5:
            recent_scores = [m.primary_score for m in history[-5:]]
            volatility = np.std(recent_scores)
            measurement.measurement_context['recent_volatility'] = volatility
        
        self.logger.debug(f"Analyzed {outcome_type.value} measurement for user {measurement.user_id}: change={change:.2f}")
    
    async def _check_for_alerts(self, measurement: OutcomeMeasurement, profile: UserOutcomeProfile) -> List[OutcomeAlert]:
        """Check for alert conditions based on new measurement"""
        
        alerts = []
        outcome_type = measurement.outcome_type
        user_id = measurement.user_id
        
        # Get measurement context
        context = measurement.measurement_context
        
        # Critical decline alert
        change_from_previous = context.get('change_from_previous', 0)
        if change_from_previous <= self.alert_thresholds['critical_decline']:
            alerts.append(OutcomeAlert(
                alert_id=f"critical_decline_{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                severity=AlertSeverity.CRITICAL,
                alert_type="critical_decline",
                title=f"Critical Decline in {outcome_type.value.replace('_', ' ').title()}",
                description=f"User showed a decline of {abs(change_from_previous):.1f} points in {outcome_type.value}",
                affected_outcomes=[outcome_type],
                supporting_data={
                    'current_score': measurement.primary_score,
                    'change': change_from_previous,
                    'measurement_date': measurement.measurement_date.isoformat()
                },
                requires_immediate_attention=True,
                recommended_actions=[
                    "Review recent interventions and life events",
                    "Consider immediate clinical assessment",
                    "Implement crisis prevention protocols if applicable"
                ]
            ))
        
        # Crisis threshold alert (for crisis-related outcomes)
        if outcome_type in [OutcomeType.CRISIS_FREQUENCY, OutcomeType.SYMPTOM_SEVERITY]:
            if measurement.primary_score >= 8.0:  # High severity
                alerts.append(OutcomeAlert(
                    alert_id=f"crisis_threshold_{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    severity=AlertSeverity.EMERGENCY,
                    alert_type="crisis_threshold",
                    title=f"Crisis Threshold Reached: {outcome_type.value.replace('_', ' ').title()}",
                    description=f"User reported high severity score: {measurement.primary_score}/10",
                    affected_outcomes=[outcome_type],
                    supporting_data={
                        'current_score': measurement.primary_score,
                        'threshold': 8.0,
                        'measurement_date': measurement.measurement_date.isoformat()
                    },
                    requires_immediate_attention=True,
                    recommended_actions=[
                        "Immediate clinical intervention required",
                        "Activate crisis response protocols",
                        "Assess safety and implement protective measures"
                    ]
                ))
        
        # Instability alert
        volatility = context.get('recent_volatility', 0)
        if volatility > self.alert_thresholds['instability_threshold']:
            alerts.append(OutcomeAlert(
                alert_id=f"instability_{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                severity=AlertSeverity.WARNING,
                alert_type="instability",
                title=f"Instability Detected in {outcome_type.value.replace('_', ' ').title()}",
                description=f"High variability in recent measurements (SD: {volatility:.2f})",
                affected_outcomes=[outcome_type],
                supporting_data={
                    'volatility': volatility,
                    'threshold': self.alert_thresholds['instability_threshold'],
                    'measurement_date': measurement.measurement_date.isoformat()
                },
                recommended_actions=[
                    "Review medication adherence and dosage",
                    "Assess environmental stressors",
                    "Consider stabilization interventions"
                ]
            ))
        
        # Significant improvement alert
        if change_from_previous >= self.alert_thresholds['significant_improvement']:
            alerts.append(OutcomeAlert(
                alert_id=f"improvement_{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                severity=AlertSeverity.INFO,
                alert_type="significant_improvement",
                title=f"Significant Improvement in {outcome_type.value.replace('_', ' ').title()}",
                description=f"User showed improvement of {change_from_previous:.1f} points",
                affected_outcomes=[outcome_type],
                supporting_data={
                    'current_score': measurement.primary_score,
                    'improvement': change_from_previous,
                    'measurement_date': measurement.measurement_date.isoformat()
                },
                recommended_actions=[
                    "Acknowledge progress with user",
                    "Identify factors contributing to improvement",
                    "Reinforce successful strategies"
                ]
            ))
        
        # Log alerts generated
        if alerts:
            self.logger.warning(f"Generated {len(alerts)} alerts for user {user_id} based on {outcome_type.value} measurement")
        
        return alerts
    
    async def _update_trend_analysis(self, user_id: str, outcome_type: OutcomeType):
        """Update trend analysis for a specific outcome"""
        
        profile = self.user_profiles[user_id]
        history = profile.measurement_history[outcome_type]
        
        if len(history) < self.min_measurements_for_trend:
            return
        
        # Prepare data for analysis
        dates = [m.measurement_date for m in history]
        scores = [m.primary_score for m in history]
        
        # Convert dates to numeric (days since first measurement)
        first_date = dates[0]
        numeric_dates = [(d - first_date).days for d in dates]
        
        # Perform linear regression
        X = np.array(numeric_dates).reshape(-1, 1)
        y = np.array(scores)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        slope = model.coef_[0]
        r_squared = r2_score(y, y_pred)
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Minimal slope
            trend_direction = TrendDirection.STABLE
        elif slope > 0.05:
            trend_direction = TrendDirection.IMPROVING
        elif slope < -0.05:
            trend_direction = TrendDirection.DECLINING
        else:
            # Check for fluctuation
            score_changes = [scores[i] - scores[i-1] for i in range(1, len(scores))]
            sign_changes = sum(1 for i in range(1, len(score_changes)) if score_changes[i] * score_changes[i-1] < 0)
            
            if sign_changes >= len(score_changes) * 0.5:
                trend_direction = TrendDirection.FLUCTUATING
            else:
                trend_direction = TrendDirection.STABLE
        
        # Calculate trend strength
        trend_strength = min(1.0, abs(slope) * len(history) / 10.0)  # Normalize
        
        # Detect significant changes
        significant_changes = []
        for i in range(1, len(scores)):
            change = scores[i] - scores[i-1]
            if abs(change) >= self.clinical_significance_threshold:
                significant_changes.append({
                    'date': dates[i].isoformat(),
                    'change': change,
                    'before_score': scores[i-1],
                    'after_score': scores[i]
                })
        
        # Make predictions
        future_days = 30
        future_X = np.array([numeric_dates[-1] + future_days]).reshape(-1, 1)
        projected_value = model.predict(future_X)[0]
        
        # Calculate prediction confidence based on R²
        projection_confidence = max(0.1, r_squared * 0.8)  # Conservative confidence
        
        # Create trend analysis
        analysis = TrendAnalysis(
            outcome_type=outcome_type,
            user_id=user_id,
            analysis_period_days=(dates[-1] - dates[0]).days,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            slope=slope,
            r_squared=r_squared,
            mean_change=np.mean([scores[i] - scores[i-1] for i in range(1, len(scores))]),
            standard_deviation=np.std(scores),
            coefficient_of_variation=np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            significant_changes=significant_changes,
            change_point_dates=[datetime.fromisoformat(change['date']) for change in significant_changes],
            projected_value_30_days=max(0.0, min(10.0, projected_value)),
            projection_confidence=projection_confidence,
            data_points_analyzed=len(history),
            analysis_quality=min(1.0, (len(history) / 10.0) * r_squared)
        )
        
        # Store analysis
        analysis_id = f"{user_id}_{outcome_type.value}_{int(datetime.now().timestamp())}"
        self.trend_analyses[analysis_id] = analysis
        
        # Update profile
        profile.trend_analysis[outcome_type] = asdict(analysis)
        
        self.logger.debug(f"Updated trend analysis for {outcome_type.value} (user {user_id}): {trend_direction.value}, strength={trend_strength:.2f}")
    
    async def _update_profile_assessment(self, profile: UserOutcomeProfile):
        """Update overall profile assessment and risk level"""
        
        # Calculate overall trajectory
        trend_scores = []
        for outcome_type, analysis in profile.trend_analysis.items():
            if analysis['trend_direction'] == TrendDirection.IMPROVING.value:
                trend_scores.append(1.0)
            elif analysis['trend_direction'] == TrendDirection.STABLE.value:
                trend_scores.append(0.0)
            elif analysis['trend_direction'] == TrendDirection.DECLINING.value:
                trend_scores.append(-1.0)
            else:  # Fluctuating or unknown
                trend_scores.append(-0.5)
        
        if trend_scores:
            avg_trend = np.mean(trend_scores)
            if avg_trend > 0.3:
                profile.overall_trajectory = TrendDirection.IMPROVING
            elif avg_trend < -0.3:
                profile.overall_trajectory = TrendDirection.DECLINING
            elif abs(avg_trend) < 0.1:
                profile.overall_trajectory = TrendDirection.STABLE
            else:
                profile.overall_trajectory = TrendDirection.FLUCTUATING
        
        # Assess risk level
        risk_indicators = []
        
        # Check for high-risk outcome scores
        for outcome_type, measurement in profile.current_measurements.items():
            if outcome_type in [OutcomeType.CRISIS_FREQUENCY, OutcomeType.SYMPTOM_SEVERITY]:
                if measurement.primary_score >= 8.0:
                    risk_indicators.append(f"High {outcome_type.value}")
                elif measurement.primary_score >= 6.0:
                    risk_indicators.append(f"Moderate {outcome_type.value}")
        
        # Check for declining trends in critical areas
        critical_outcomes = [OutcomeType.CRISIS_FREQUENCY, OutcomeType.SYMPTOM_SEVERITY, OutcomeType.FUNCTIONAL_IMPROVEMENT]
        for outcome_type in critical_outcomes:
            if outcome_type in profile.trend_analysis:
                analysis = profile.trend_analysis[outcome_type]
                if analysis['trend_direction'] == TrendDirection.DECLINING.value and analysis['trend_strength'] > 0.5:
                    risk_indicators.append(f"Declining {outcome_type.value}")
        
        # Determine overall risk level
        if len(risk_indicators) == 0:
            profile.current_risk_level = "low"
        elif len(risk_indicators) == 1 and not any("High" in r for r in risk_indicators):
            profile.current_risk_level = "moderate"
        elif len(risk_indicators) <= 2 and not any("High" in r for r in risk_indicators):
            profile.current_risk_level = "high"
        else:
            profile.current_risk_level = "critical"
        
        profile.risk_factors = risk_indicators
        
        # Calculate data quality metrics
        total_possible_measurements = len(OutcomeType) * max(1, (datetime.now() - profile.profile_created_at).days // 7)  # Weekly measurements expected
        actual_measurements = sum(len(measurements) for measurements in profile.measurement_history.values())
        
        profile.data_completeness = min(1.0, actual_measurements / max(1, total_possible_measurements))
        
        # Calculate measurement reliability (based on data source diversity and consistency)
        reliable_measurements = sum(
            1 for measurements in profile.measurement_history.values()
            for m in measurements
            if m.confidence >= 0.7
        )
        
        profile.measurement_reliability = reliable_measurements / max(1, actual_measurements) if actual_measurements > 0 else 0.0
        
        # Overall profile confidence
        profile.profile_confidence = (profile.data_completeness + profile.measurement_reliability) / 2
    
    async def get_user_outcome_summary(self, user_id: str, include_predictions: bool = True) -> Dict[str, Any]:
        """Get comprehensive outcome summary for a user"""
        
        if user_id not in self.user_profiles:
            return {"error": f"No outcome profile found for user {user_id}"}
        
        profile = self.user_profiles[user_id]
        
        # Current status
        current_status = {}
        for outcome_type, measurement in profile.current_measurements.items():
            current_status[outcome_type.value] = {
                'score': measurement.primary_score,
                'date': measurement.measurement_date.isoformat(),
                'days_since_baseline': measurement.days_since_baseline,
                'change_from_baseline': measurement.measurement_context.get('change_from_baseline', 0)
            }
        
        # Trend summaries
        trend_summaries = {}
        for outcome_type, analysis in profile.trend_analysis.items():
            trend_summaries[outcome_type.value] = {
                'direction': analysis['trend_direction'],
                'strength': analysis['trend_strength'],
                'projected_30_days': analysis['projected_value_30_days'],
                'confidence': analysis['projection_confidence']
            }
        
        # Recent alerts
        user_alerts = [
            {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'detected_at': alert.detected_at.isoformat(),
                'requires_attention': alert.requires_immediate_attention,
                'acknowledged': alert.acknowledged
            }
            for alert in self.alerts.values()
            if alert.user_id == user_id and (datetime.now() - alert.detected_at).days <= 30
        ]
        
        # Calculate improvement scores
        improvement_scores = {}
        for outcome_type, measurement in profile.current_measurements.items():
            if outcome_type in profile.baseline_measurements:
                baseline = profile.baseline_measurements[outcome_type].primary_score
                current = measurement.primary_score
                
                # For outcomes where lower is better (like symptom severity)
                if outcome_type in [OutcomeType.SYMPTOM_SEVERITY, OutcomeType.CRISIS_FREQUENCY, OutcomeType.ANXIETY_LEVELS]:
                    improvement = baseline - current  # Positive improvement = lower score
                else:
                    improvement = current - baseline  # Positive improvement = higher score
                
                improvement_scores[outcome_type.value] = improvement
        
        summary = {
            'user_id': user_id,
            'profile_created_at': profile.profile_created_at.isoformat(),
            'last_updated': profile.last_updated.isoformat(),
            'overall_trajectory': profile.overall_trajectory.value,
            'current_risk_level': profile.current_risk_level,
            'data_quality': {
                'completeness': profile.data_completeness,
                'reliability': profile.measurement_reliability,
                'confidence': profile.profile_confidence
            },
            'current_status': current_status,
            'improvement_scores': improvement_scores,
            'trend_analysis': trend_summaries,
            'recent_alerts': user_alerts,
            'risk_factors': profile.risk_factors,
            'protective_factors': profile.protective_factors,
            'milestones_achieved': profile.milestones_achieved,
            'measurement_counts': {
                outcome_type.value: len(measurements)
                for outcome_type, measurements in profile.measurement_history.items()
            }
        }
        
        # Add predictions if requested
        if include_predictions and profile.predicted_outcomes:
            summary['predictions'] = profile.predicted_outcomes
            summary['confidence_intervals'] = profile.confidence_intervals
        
        return summary
    
    async def generate_outcome_report(self,
                                    user_id: Optional[str] = None,
                                    outcome_types: Optional[List[OutcomeType]] = None,
                                    time_period_days: int = 90) -> Dict[str, Any]:
        """Generate comprehensive outcome report"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_days)
        
        if user_id:
            # Single user report
            if user_id not in self.user_profiles:
                return {"error": f"No data available for user {user_id}"}
            
            profiles_to_analyze = {user_id: self.user_profiles[user_id]}
        else:
            # System-wide report
            profiles_to_analyze = self.user_profiles
        
        # Filter by outcome types if specified
        if outcome_types is None:
            outcome_types = list(OutcomeType)
        
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'time_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': time_period_days
            },
            'scope': {
                'user_specific': user_id is not None,
                'user_id': user_id,
                'outcome_types_analyzed': [ot.value for ot in outcome_types],
                'total_users_analyzed': len(profiles_to_analyze)
            },
            'summary_statistics': {},
            'trend_analysis': {},
            'risk_assessment': {},
            'alerts_summary': {},
            'recommendations': []
        }
        
        # Collect all relevant measurements
        all_measurements = []
        for profile in profiles_to_analyze.values():
            for outcome_type in outcome_types:
                if outcome_type in profile.measurement_history:
                    measurements = [
                        m for m in profile.measurement_history[outcome_type]
                        if start_date <= m.measurement_date <= end_date
                    ]
                    all_measurements.extend(measurements)
        
        if not all_measurements:
            report['summary_statistics'] = {'message': 'No measurements found in the specified time period'}
            return report
        
        # Summary statistics
        measurements_by_outcome = defaultdict(list)
        for measurement in all_measurements:
            measurements_by_outcome[measurement.outcome_type].append(measurement)
        
        summary_stats = {}
        for outcome_type, measurements in measurements_by_outcome.items():
            scores = [m.primary_score for m in measurements]
            summary_stats[outcome_type.value] = {
                'count': len(measurements),
                'mean': np.mean(scores),
                'median': np.median(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'trend': self._calculate_overall_trend(measurements)
            }
        
        report['summary_statistics'] = summary_stats
        
        # Trend analysis summary
        trend_summary = {}
        for outcome_type in outcome_types:
            trends = []
            for profile in profiles_to_analyze.values():
                if outcome_type in profile.trend_analysis:
                    trends.append(profile.trend_analysis[outcome_type])
            
            if trends:
                trend_directions = [t['trend_direction'] for t in trends]
                trend_summary[outcome_type.value] = {
                    'improving_users': trend_directions.count(TrendDirection.IMPROVING.value),
                    'stable_users': trend_directions.count(TrendDirection.STABLE.value),
                    'declining_users': trend_directions.count(TrendDirection.DECLINING.value),
                    'fluctuating_users': trend_directions.count(TrendDirection.FLUCTUATING.value),
                    'average_trend_strength': np.mean([t['trend_strength'] for t in trends])
                }
        
        report['trend_analysis'] = trend_summary
        
        # Risk assessment
        risk_levels = [profile.current_risk_level for profile in profiles_to_analyze.values()]
        report['risk_assessment'] = {
            'low_risk_users': risk_levels.count('low'),
            'moderate_risk_users': risk_levels.count('moderate'),
            'high_risk_users': risk_levels.count('high'),
            'critical_risk_users': risk_levels.count('critical'),
            'overall_risk_distribution': {
                level: risk_levels.count(level) / len(risk_levels) 
                for level in ['low', 'moderate', 'high', 'critical']
            } if risk_levels else {}
        }
        
        # Alerts summary
        period_alerts = [
            alert for alert in self.alerts.values()
            if start_date <= alert.detected_at <= end_date and
            (not user_id or alert.user_id == user_id)
        ]
        
        alerts_by_severity = defaultdict(int)
        for alert in period_alerts:
            alerts_by_severity[alert.severity.value] += 1
        
        report['alerts_summary'] = {
            'total_alerts': len(period_alerts),
            'by_severity': dict(alerts_by_severity),
            'unresolved_alerts': len([a for a in period_alerts if not a.resolved]),
            'critical_unresolved': len([
                a for a in period_alerts 
                if not a.resolved and a.severity == AlertSeverity.CRITICAL
            ])
        }
        
        # Generate recommendations
        recommendations = await self._generate_report_recommendations(
            profiles_to_analyze, measurements_by_outcome, period_alerts
        )
        report['recommendations'] = recommendations
        
        return report
    
    def _calculate_overall_trend(self, measurements: List[OutcomeMeasurement]) -> str:
        """Calculate overall trend for a list of measurements"""
        
        if len(measurements) < 3:
            return 'insufficient_data'
        
        # Sort by date
        sorted_measurements = sorted(measurements, key=lambda m: m.measurement_date)
        scores = [m.primary_score for m in sorted_measurements]
        
        # Simple linear regression
        X = np.arange(len(scores)).reshape(-1, 1)
        y = np.array(scores)
        
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        if slope > 0.05:
            return 'improving'
        elif slope < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    async def _generate_report_recommendations(self,
                                             profiles: Dict[str, UserOutcomeProfile],
                                             measurements_by_outcome: Dict[OutcomeType, List[OutcomeMeasurement]],
                                             alerts: List[OutcomeAlert]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on report data"""
        
        recommendations = []
        
        # High-risk users recommendation
        high_risk_users = [
            user_id for user_id, profile in profiles.items()
            if profile.current_risk_level in ['high', 'critical']
        ]
        
        if high_risk_users:
            recommendations.append({
                'type': 'risk_management',
                'priority': 'high',
                'title': 'High-Risk Users Require Attention',
                'description': f'{len(high_risk_users)} users are currently at high or critical risk',
                'affected_users': len(high_risk_users),
                'recommended_actions': [
                    'Conduct immediate clinical review for critical risk users',
                    'Implement enhanced monitoring protocols',
                    'Review and adjust intervention strategies'
                ]
            })
        
        # Declining outcomes recommendation
        declining_outcomes = []
        for outcome_type, measurements in measurements_by_outcome.items():
            if len(measurements) >= 10:  # Sufficient data
                trend = self._calculate_overall_trend(measurements)
                if trend == 'declining':
                    declining_outcomes.append(outcome_type.value)
        
        if declining_outcomes:
            recommendations.append({
                'type': 'outcome_improvement',
                'priority': 'medium',
                'title': 'Declining Outcomes Detected',
                'description': f'System-wide decline observed in: {", ".join(declining_outcomes)}',
                'affected_outcomes': declining_outcomes,
                'recommended_actions': [
                    'Review intervention protocols for affected outcome areas',
                    'Analyze potential systemic causes',
                    'Implement targeted improvement strategies'
                ]
            })
        
        # Data quality recommendation
        low_quality_profiles = [
            user_id for user_id, profile in profiles.items()
            if profile.data_completeness < 0.5 or profile.measurement_reliability < 0.6
        ]
        
        if len(low_quality_profiles) > len(profiles) * 0.3:  # More than 30% have poor data quality
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'title': 'Improve Data Collection Quality',
                'description': f'{len(low_quality_profiles)} users have incomplete or unreliable outcome data',
                'recommended_actions': [
                    'Implement automated measurement reminders',
                    'Simplify data collection processes',
                    'Train users on importance of consistent reporting'
                ]
            })
        
        # Unresolved alerts recommendation
        critical_unresolved = len([
            alert for alert in alerts 
            if not alert.resolved and alert.severity == AlertSeverity.CRITICAL
        ])
        
        if critical_unresolved > 0:
            recommendations.append({
                'type': 'alert_management',
                'priority': 'critical',
                'title': 'Unresolved Critical Alerts',
                'description': f'{critical_unresolved} critical alerts remain unresolved',
                'recommended_actions': [
                    'Immediate review of all unresolved critical alerts',
                    'Implement alert escalation procedures',
                    'Ensure proper clinical follow-up protocols'
                ]
            })
        
        return recommendations
    
    async def predict_future_outcomes(self,
                                    user_id: str,
                                    outcome_types: Optional[List[OutcomeType]] = None,
                                    prediction_horizon_days: int = 30) -> Dict[str, Any]:
        """Predict future outcomes for a user using historical data"""
        
        if user_id not in self.user_profiles:
            return {"error": f"No profile found for user {user_id}"}
        
        profile = self.user_profiles[user_id]
        
        if outcome_types is None:
            outcome_types = list(profile.measurement_history.keys())
        
        predictions = {}
        
        for outcome_type in outcome_types:
            if outcome_type not in profile.measurement_history:
                continue
            
            history = profile.measurement_history[outcome_type]
            
            if len(history) < 5:  # Need sufficient data for prediction
                predictions[outcome_type.value] = {
                    'error': 'Insufficient historical data for prediction'
                }
                continue
            
            # Prepare data
            dates = [m.measurement_date for m in history]
            scores = [m.primary_score for m in history]
            
            # Convert to numeric format
            base_date = dates[0]
            numeric_dates = [(d - base_date).days for d in dates]
            
            # Build prediction model
            X = np.array(numeric_dates).reshape(-1, 1)
            y = np.array(scores)
            
            # Use polynomial features for better prediction
            from sklearn.preprocessing import PolynomialFeatures
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Make predictions
            future_date = numeric_dates[-1] + prediction_horizon_days
            future_X = np.array([[future_date]])
            future_X_poly = poly_features.transform(future_X)
            
            predicted_score = model.predict(future_X_poly)[0]
            predicted_score = max(0.0, min(10.0, predicted_score))  # Clamp to valid range
            
            # Calculate confidence interval
            y_pred = model.predict(X_poly)
            residuals = y - y_pred
            prediction_std = np.std(residuals)
            confidence_interval = (
                max(0.0, predicted_score - 1.96 * prediction_std),
                min(10.0, predicted_score + 1.96 * prediction_std)
            )
            
            # Model quality metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            predictions[outcome_type.value] = {
                'predicted_score': predicted_score,
                'confidence_interval': confidence_interval,
                'prediction_date': (datetime.now() + timedelta(days=prediction_horizon_days)).isoformat(),
                'model_quality': {
                    'r_squared': r2,
                    'rmse': rmse,
                    'data_points_used': len(history)
                },
                'trend_indication': 'improving' if predicted_score > scores[-1] else 'declining' if predicted_score < scores[-1] else 'stable'
            }
        
        # Store predictions in profile
        profile.predicted_outcomes = predictions
        profile.confidence_intervals = {
            outcome: pred['confidence_interval']
            for outcome, pred in predictions.items()
            if 'confidence_interval' in pred
        }
        
        return {
            'user_id': user_id,
            'prediction_horizon_days': prediction_horizon_days,
            'predictions': predictions,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_active_alerts(self, user_id: Optional[str] = None, severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Get active (unresolved) alerts"""
        
        alerts = []
        
        for alert in self.alerts.values():
            # Filter by user if specified
            if user_id and alert.user_id != user_id:
                continue
            
            # Filter by severity if specified
            if severity and alert.severity != severity:
                continue
            
            # Only include unresolved alerts
            if alert.resolved:
                continue
            
            alerts.append({
                'alert_id': alert.alert_id,
                'user_id': alert.user_id,
                'severity': alert.severity.value,
                'alert_type': alert.alert_type,
                'title': alert.title,
                'description': alert.description,
                'affected_outcomes': [ot.value for ot in alert.affected_outcomes],
                'detected_at': alert.detected_at.isoformat(),
                'requires_immediate_attention': alert.requires_immediate_attention,
                'recommended_actions': alert.recommended_actions,
                'acknowledged': alert.acknowledged,
                'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            })
        
        # Sort by severity and date
        severity_order = {
            AlertSeverity.EMERGENCY.value: 0,
            AlertSeverity.CRITICAL.value: 1,
            AlertSeverity.WARNING.value: 2,
            AlertSeverity.INFO.value: 3
        }
        
        alerts.sort(key=lambda a: (severity_order.get(a['severity'], 4), a['detected_at']), reverse=True)
        
        return alerts
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> bool:
        """Acknowledge an alert"""
        
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        if notes:
            alert.resolution_notes = notes
        
        self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str) -> bool:
        """Mark an alert as resolved"""
        
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolution_notes = resolution_notes
        
        # Also acknowledge if not already done
        if not alert.acknowledged:
            alert.acknowledged = True
            alert.acknowledged_by = resolved_by
            alert.acknowledged_at = datetime.now()
        
        self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def _load_persisted_data(self):
        """Load persisted outcome data"""
        
        try:
            data_dir = self.config.get('data_directory', 'src/data/adaptive_learning/outcome_data')
            
            # Load user profiles
            profiles_file = f"{data_dir}/outcome_profiles.pkl"
            if os.path.exists(profiles_file):
                with open(profiles_file, 'rb') as f:
                    saved_profiles = pickle.load(f)
                    
                # Convert back to objects
                for user_id, profile_data in saved_profiles.items():
                    # Convert datetime strings back to datetime objects
                    profile_data['profile_created_at'] = datetime.fromisoformat(profile_data['profile_created_at'])
                    profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                    if profile_data.get('baseline_date'):
                        profile_data['baseline_date'] = datetime.fromisoformat(profile_data['baseline_date'])
                    
                    self.user_profiles[user_id] = UserOutcomeProfile(**profile_data)
                
                self.logger.info(f"Loaded {len(self.user_profiles)} outcome profiles")
            
            # Load measurements
            measurements_file = f"{data_dir}/measurements.pkl"
            if os.path.exists(measurements_file):
                with open(measurements_file, 'rb') as f:
                    saved_measurements = pickle.load(f)
                    
                # Convert back to objects
                for measurement_id, measurement_data in saved_measurements.items():
                    measurement_data['measurement_date'] = datetime.fromisoformat(measurement_data['measurement_date'])
                    if measurement_data.get('processed_at'):
                        measurement_data['processed_at'] = datetime.fromisoformat(measurement_data['processed_at'])
                    
                    # Convert outcome_type back to enum
                    measurement_data['outcome_type'] = OutcomeType(measurement_data['outcome_type'])
                    
                    self.outcome_measurements[measurement_id] = OutcomeMeasurement(**measurement_data)
                
                self.logger.info(f"Loaded {len(self.outcome_measurements)} outcome measurements")
            
            # Load alerts
            alerts_file = f"{data_dir}/alerts.pkl"
            if os.path.exists(alerts_file):
                with open(alerts_file, 'rb') as f:
                    saved_alerts = pickle.load(f)
                    
                for alert_id, alert_data in saved_alerts.items():
                    # Convert datetime strings and enums
                    alert_data['detected_at'] = datetime.fromisoformat(alert_data['detected_at'])
                    if alert_data.get('acknowledged_at'):
                        alert_data['acknowledged_at'] = datetime.fromisoformat(alert_data['acknowledged_at'])
                    
                    alert_data['severity'] = AlertSeverity(alert_data['severity'])
                    alert_data['affected_outcomes'] = [OutcomeType(ot) for ot in alert_data['affected_outcomes']]
                    
                    self.alerts[alert_id] = OutcomeAlert(**alert_data)
                
                self.logger.info(f"Loaded {len(self.alerts)} outcome alerts")
                
        except Exception as e:
            self.logger.warning(f"Could not load persisted outcome data: {str(e)}")
    
    def save_data(self):
        """Save outcome data to disk"""
        
        try:
            data_dir = self.config.get('data_directory', 'src/data/adaptive_learning/outcome_data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save user profiles
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profile_dict = asdict(profile)
                # Convert datetime objects to strings
                profile_dict['profile_created_at'] = profile.profile_created_at.isoformat()
                profile_dict['last_updated'] = profile.last_updated.isoformat()
                if profile.baseline_date:
                    profile_dict['baseline_date'] = profile.baseline_date.isoformat()
                profiles_data[user_id] = profile_dict
            
            with open(f"{data_dir}/outcome_profiles.pkl", 'wb') as f:
                pickle.dump(profiles_data, f)
            
            # Save measurements
            measurements_data = {}
            for measurement_id, measurement in self.outcome_measurements.items():
                measurement_dict = asdict(measurement)
                measurement_dict['measurement_date'] = measurement.measurement_date.isoformat()
                if measurement.processed_at:
                    measurement_dict['processed_at'] = measurement.processed_at.isoformat()
                measurement_dict['outcome_type'] = measurement.outcome_type.value
                measurements_data[measurement_id] = measurement_dict
            
            with open(f"{data_dir}/measurements.pkl", 'wb') as f:
                pickle.dump(measurements_data, f)
            
            # Save alerts
            alerts_data = {}
            for alert_id, alert in self.alerts.items():
                alert_dict = asdict(alert)
                alert_dict['detected_at'] = alert.detected_at.isoformat()
                if alert.acknowledged_at:
                    alert_dict['acknowledged_at'] = alert.acknowledged_at.isoformat()
                alert_dict['severity'] = alert.severity.value
                alert_dict['affected_outcomes'] = [ot.value for ot in alert.affected_outcomes]
                alerts_data[alert_id] = alert_dict
            
            with open(f"{data_dir}/alerts.pkl", 'wb') as f:
                pickle.dump(alerts_data, f)
            
            self.logger.info("Outcome data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving outcome data: {str(e)}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        
        return {
            'total_users_tracked': len(self.user_profiles),
            'total_measurements': len(self.outcome_measurements),
            'total_alerts': len(self.alerts),
            'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
            'measurements_by_outcome': {
                ot.value: len([m for m in self.outcome_measurements.values() if m.outcome_type == ot])
                for ot in OutcomeType
            },
            'risk_distribution': {
                'low': len([p for p in self.user_profiles.values() if p.current_risk_level == 'low']),
                'moderate': len([p for p in self.user_profiles.values() if p.current_risk_level == 'moderate']),
                'high': len([p for p in self.user_profiles.values() if p.current_risk_level == 'high']),
                'critical': len([p for p in self.user_profiles.values() if p.current_risk_level == 'critical'])
            },
            'data_quality': {
                'average_completeness': np.mean([p.data_completeness for p in self.user_profiles.values()]) if self.user_profiles else 0.0,
                'average_reliability': np.mean([p.measurement_reliability for p in self.user_profiles.values()]) if self.user_profiles else 0.0
            },
            'recent_activity': {
                'measurements_last_7_days': len([
                    m for m in self.outcome_measurements.values()
                    if (datetime.now() - m.measurement_date).days <= 7
                ]),
                'alerts_last_7_days': len([
                    a for a in self.alerts.values()
                    if (datetime.now() - a.detected_at).days <= 7
                ])
            }
        }
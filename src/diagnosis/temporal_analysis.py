"""
Temporal Analysis Module for Longitudinal Symptom Tracking

This module implements advanced temporal analysis capabilities for tracking
symptom progression, behavioral patterns, and therapeutic response over time.
Based on the improvements outlined in the comprehensive enhancement plan.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np
from collections import defaultdict, deque

from ..database.central_vector_db import CentralVectorDB
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class SymptomEntry:
    """Data structure for individual symptom entries"""
    symptom_type: str
    intensity: float  # 0.0 to 1.0 scale
    timestamp: datetime
    context: str
    triggers: List[str]
    user_id: str
    session_id: str
    metadata: Dict[str, Any]

@dataclass 
class BehavioralPattern:
    """Data structure for behavioral patterns"""
    pattern_type: str
    confidence: float
    frequency: str  # daily, weekly, monthly
    description: str
    first_observed: datetime
    last_observed: datetime
    correlation_score: float
    associated_symptoms: List[str]

@dataclass
class InterventionOutcome:
    """Data structure for tracking intervention effectiveness"""
    intervention_type: str
    applied_at: datetime
    symptom_before: float
    symptom_after: float
    effectiveness_score: float
    duration_days: int
    user_feedback: Optional[str]
    notes: str

class TemporalAnalysisEngine:
    """
    Advanced temporal analysis engine for tracking symptom progression
    and behavioral patterns over time.
    """
    
    def __init__(self, vector_db: Optional[CentralVectorDB] = None):
        """Initialize the temporal analysis engine"""
        self.vector_db = vector_db
        self.logger = get_logger(__name__)
        
        # In-memory caches for performance
        self.symptom_cache = defaultdict(deque)  # user_id -> deque of symptoms
        self.pattern_cache = defaultdict(list)   # user_id -> list of patterns
        self.intervention_cache = defaultdict(list)  # user_id -> list of interventions
        
        # Configuration
        self.cache_size = 1000  # Max symptoms to keep in memory per user
        self.pattern_detection_window = 30  # Days to look back for pattern detection
        self.min_pattern_occurrences = 3  # Minimum occurrences to consider a pattern
        
    async def record_symptom(self, 
                            user_id: str,
                            symptom_type: str,
                            intensity: float,
                            context: str = "",
                            triggers: List[str] = None,
                            session_id: str = "default",
                            metadata: Dict[str, Any] = None) -> bool:
        """
        Record a new symptom entry with temporal tracking
        
        Args:
            user_id: User identifier
            symptom_type: Type of symptom (e.g., "anxiety", "depression", "stress")
            intensity: Symptom intensity on 0.0-1.0 scale
            context: Contextual information about the symptom
            triggers: List of identified triggers
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            symptom_entry = SymptomEntry(
                symptom_type=symptom_type,
                intensity=float(intensity),
                timestamp=datetime.now(),
                context=context,
                triggers=triggers or [],
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            
            # Add to cache
            self.symptom_cache[user_id].append(symptom_entry)
            
            # Maintain cache size
            if len(self.symptom_cache[user_id]) > self.cache_size:
                self.symptom_cache[user_id].popleft()
            
            # Store in vector database if available
            if self.vector_db:
                await self._store_symptom_in_db(symptom_entry)
            
            # Trigger pattern analysis
            await self._analyze_patterns_for_user(user_id)
            
            self.logger.info(f"Recorded symptom: {symptom_type} (intensity: {intensity}) for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording symptom: {str(e)}")
            return False
    
    async def get_symptom_progression(self, 
                                    user_id: str,
                                    symptom_type: str = None,
                                    days_back: int = 30) -> Dict[str, Any]:
        """
        Get symptom progression analysis for a user
        
        Args:
            user_id: User identifier
            symptom_type: Specific symptom type to analyze (None for all)
            days_back: Number of days to analyze
            
        Returns:
            Symptom progression data and analysis
        """
        try:
            # Get symptoms from cache and database
            symptoms = await self._get_symptoms_for_period(user_id, days_back)
            
            if symptom_type:
                symptoms = [s for s in symptoms if s.symptom_type == symptom_type]
            
            if not symptoms:
                return {
                    "user_id": user_id,
                    "symptom_type": symptom_type,
                    "days_analyzed": days_back,
                    "total_entries": 0,
                    "progression": {},
                    "trends": {},
                    "insights": []
                }
            
            # Analyze progression
            progression_data = self._analyze_symptom_progression(symptoms)
            trends = self._calculate_trends(symptoms)
            insights = self._generate_progression_insights(symptoms, trends)
            
            return {
                "user_id": user_id,
                "symptom_type": symptom_type,
                "days_analyzed": days_back,
                "total_entries": len(symptoms),
                "progression": progression_data,
                "trends": trends,
                "insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing symptom progression: {str(e)}")
            return {"error": str(e)}
    
    async def detect_behavioral_patterns(self, 
                                       user_id: str,
                                       min_confidence: float = 0.7) -> List[BehavioralPattern]:
        """
        Detect behavioral patterns for a user
        
        Args:
            user_id: User identifier
            min_confidence: Minimum confidence threshold for patterns
            
        Returns:
            List of detected behavioral patterns
        """
        try:
            # Check cache first
            cached_patterns = self.pattern_cache.get(user_id, [])
            recent_patterns = [p for p in cached_patterns 
                             if (datetime.now() - p.last_observed).days <= 7]
            
            if recent_patterns:
                return [p for p in recent_patterns if p.confidence >= min_confidence]
            
            # Analyze patterns
            symptoms = await self._get_symptoms_for_period(user_id, self.pattern_detection_window)
            patterns = await self._detect_patterns(symptoms)
            
            # Filter by confidence
            high_confidence_patterns = [p for p in patterns if p.confidence >= min_confidence]
            
            # Update cache
            self.pattern_cache[user_id] = patterns
            
            return high_confidence_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral patterns: {str(e)}")
            return []
    
    async def predict_symptom_trajectory(self, 
                                       user_id: str,
                                       symptom_type: str,
                                       prediction_days: int = 7) -> Dict[str, Any]:
        """
        Predict symptom trajectory using historical data and patterns
        
        Args:
            user_id: User identifier
            symptom_type: Type of symptom to predict
            prediction_days: Number of days to predict ahead
            
        Returns:
            Prediction data with confidence intervals
        """
        try:
            # Get historical data
            symptoms = await self._get_symptoms_for_period(user_id, 60)  # 60 days of data
            symptom_data = [s for s in symptoms if s.symptom_type == symptom_type]
            
            if len(symptom_data) < 5:  # Need minimum data points
                return {
                    "error": "Insufficient data for prediction",
                    "minimum_required": 5,
                    "available": len(symptom_data)
                }
            
            # Extract time series data
            timestamps = [s.timestamp for s in symptom_data]
            intensities = [s.intensity for s in symptom_data]
            
            # Simple linear regression for trend
            prediction = self._predict_trajectory(timestamps, intensities, prediction_days)
            
            # Get patterns for context
            patterns = await self.detect_behavioral_patterns(user_id)
            relevant_patterns = [p for p in patterns if symptom_type in p.associated_symptoms]
            
            return {
                "user_id": user_id,
                "symptom_type": symptom_type,
                "prediction_days": prediction_days,
                "current_intensity": intensities[-1] if intensities else 0,
                "predicted_trajectory": prediction,
                "confidence": self._calculate_prediction_confidence(symptom_data),
                "relevant_patterns": [asdict(p) for p in relevant_patterns],
                "recommendation": self._generate_trajectory_recommendation(prediction, relevant_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting symptom trajectory: {str(e)}")
            return {"error": str(e)}
    
    async def track_intervention_effectiveness(self,
                                             user_id: str,
                                             intervention_type: str,
                                             target_symptom: str,
                                             baseline_period_days: int = 7,
                                             followup_period_days: int = 14) -> Dict[str, Any]:
        """
        Track the effectiveness of an intervention
        
        Args:
            user_id: User identifier
            intervention_type: Type of intervention applied
            target_symptom: Target symptom being treated
            baseline_period_days: Days to look back for baseline
            followup_period_days: Days to track after intervention
            
        Returns:
            Intervention effectiveness analysis
        """
        try:
            intervention_date = datetime.now()
            
            # Get baseline data
            baseline_start = intervention_date - timedelta(days=baseline_period_days)
            baseline_symptoms = await self._get_symptoms_for_period(
                user_id, baseline_period_days, end_date=intervention_date
            )
            baseline_symptoms = [s for s in baseline_symptoms if s.symptom_type == target_symptom]
            
            # Calculate baseline average
            baseline_avg = np.mean([s.intensity for s in baseline_symptoms]) if baseline_symptoms else 0
            
            # Create intervention record
            intervention = InterventionOutcome(
                intervention_type=intervention_type,
                applied_at=intervention_date,
                symptom_before=baseline_avg,
                symptom_after=0,  # Will be updated as data comes in
                effectiveness_score=0,  # Will be calculated
                duration_days=followup_period_days,
                user_feedback=None,
                notes=f"Intervention started for {target_symptom}"
            )
            
            # Add to tracking
            self.intervention_cache[user_id].append(intervention)
            
            return {
                "intervention_id": len(self.intervention_cache[user_id]) - 1,
                "user_id": user_id,
                "intervention_type": intervention_type,
                "target_symptom": target_symptom,
                "baseline_intensity": baseline_avg,
                "baseline_data_points": len(baseline_symptoms),
                "tracking_started": intervention_date.isoformat(),
                "followup_period_days": followup_period_days,
                "status": "tracking_started"
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking intervention: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _store_symptom_in_db(self, symptom: SymptomEntry):
        """Store symptom entry in vector database"""
        if not self.vector_db:
            return
        
        try:
            document = {
                "type": "symptom_entry",
                "user_id": symptom.user_id,
                "symptom_type": symptom.symptom_type,
                "intensity": symptom.intensity,
                "context": symptom.context,
                "triggers": symptom.triggers,
                "timestamp": symptom.timestamp.isoformat(),
                "session_id": symptom.session_id,
                "metadata": symptom.metadata
            }
            
            await self.vector_db.add_document(
                document=json.dumps(document),
                metadata=document,
                doc_id=f"symptom_{symptom.user_id}_{int(symptom.timestamp.timestamp())}"
            )
            
        except Exception as e:
            self.logger.error(f"Error storing symptom in database: {str(e)}")
    
    async def _get_symptoms_for_period(self, 
                                     user_id: str,
                                     days_back: int,
                                     end_date: datetime = None) -> List[SymptomEntry]:
        """Get symptoms for a specific time period"""
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get from cache first
        cached_symptoms = self.symptom_cache.get(user_id, [])
        period_symptoms = [
            s for s in cached_symptoms 
            if start_date <= s.timestamp <= end_date
        ]
        
        # TODO: Also query vector database for older data not in cache
        
        return sorted(period_symptoms, key=lambda x: x.timestamp)
    
    def _analyze_symptom_progression(self, symptoms: List[SymptomEntry]) -> Dict[str, Any]:
        """Analyze symptom progression over time"""
        if not symptoms:
            return {}
        
        # Group by symptom type
        by_type = defaultdict(list)
        for symptom in symptoms:
            by_type[symptom.symptom_type].append(symptom)
        
        progression = {}
        for symptom_type, entries in by_type.items():
            intensities = [e.intensity for e in entries]
            timestamps = [e.timestamp for e in entries]
            
            progression[symptom_type] = {
                "count": len(entries),
                "average_intensity": np.mean(intensities),
                "intensity_std": np.std(intensities),
                "min_intensity": min(intensities),
                "max_intensity": max(intensities),
                "first_recorded": min(timestamps).isoformat(),
                "last_recorded": max(timestamps).isoformat(),
                "trend": self._calculate_linear_trend(timestamps, intensities)
            }
        
        return progression
    
    def _calculate_trends(self, symptoms: List[SymptomEntry]) -> Dict[str, Any]:
        """Calculate trend analysis for symptoms"""
        if len(symptoms) < 2:
            return {"trend": "insufficient_data"}
        
        # Overall trend
        intensities = [s.intensity for s in symptoms]
        timestamps = [s.timestamp.timestamp() for s in symptoms]
        
        # Simple linear regression
        slope = self._calculate_linear_trend(timestamps, intensities)
        
        # Weekly averages
        weekly_averages = self._calculate_weekly_averages(symptoms)
        
        return {
            "overall_slope": slope,
            "direction": "improving" if slope < -0.01 else "worsening" if slope > 0.01 else "stable",
            "weekly_averages": weekly_averages,
            "volatility": np.std(intensities) if len(intensities) > 1 else 0
        }
    
    def _calculate_linear_trend(self, timestamps: List, intensities: List[float]) -> float:
        """Calculate linear trend (slope) of intensities over time"""
        if len(timestamps) < 2:
            return 0
        
        x = np.array([t.timestamp() if hasattr(t, 'timestamp') else t for t in timestamps])
        y = np.array(intensities)
        
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        # Normalize timestamps to days from first timestamp
        x = (x - x[0]) / 86400  # Convert to days
        
        # Calculate slope using least squares
        try:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
            return 0
    
    def _calculate_weekly_averages(self, symptoms: List[SymptomEntry]) -> List[Dict[str, Any]]:
        """Calculate weekly averages for symptoms"""
        if not symptoms:
            return []
        
        # Group by week
        weekly_data = defaultdict(list)
        for symptom in symptoms:
            week_start = symptom.timestamp - timedelta(days=symptom.timestamp.weekday())
            week_key = week_start.strftime("%Y-%W")
            weekly_data[week_key].append(symptom.intensity)
        
        # Calculate averages
        weekly_averages = []
        for week, intensities in sorted(weekly_data.items()):
            weekly_averages.append({
                "week": week,
                "average_intensity": np.mean(intensities),
                "count": len(intensities)
            })
        
        return weekly_averages
    
    def _generate_progression_insights(self, 
                                     symptoms: List[SymptomEntry],
                                     trends: Dict[str, Any]) -> List[str]:
        """Generate insights based on symptom progression"""
        insights = []
        
        if not symptoms:
            return insights
        
        # Trend insights
        direction = trends.get("direction", "stable")
        if direction == "improving":
            insights.append("Your symptoms show a positive trend of improvement over time.")
        elif direction == "worsening":
            insights.append("Your symptoms appear to be increasing in intensity over time.")
        else:
            insights.append("Your symptoms are relatively stable over the analyzed period.")
        
        # Volatility insights
        volatility = trends.get("volatility", 0)
        if volatility > 0.3:
            insights.append("Your symptom intensity shows high variability, suggesting external factors may be influencing your condition.")
        elif volatility < 0.1:
            insights.append("Your symptoms show consistent patterns, which may help in identifying effective interventions.")
        
        # Frequency insights
        total_days = (max(s.timestamp for s in symptoms) - min(s.timestamp for s in symptoms)).days + 1
        frequency = len(symptoms) / max(total_days, 1)
        
        if frequency > 1:
            insights.append("You're tracking symptoms frequently, which provides good data for analysis.")
        elif frequency < 0.2:
            insights.append("Consider tracking symptoms more regularly for better pattern recognition.")
        
        # Pattern insights
        symptom_types = set(s.symptom_type for s in symptoms)
        if len(symptom_types) > 3:
            insights.append("You're experiencing multiple types of symptoms. Consider discussing holistic treatment approaches.")
        
        return insights
    
    async def _analyze_patterns_for_user(self, user_id: str):
        """Analyze patterns for a specific user (background task)"""
        try:
            symptoms = await self._get_symptoms_for_period(user_id, self.pattern_detection_window)
            if len(symptoms) >= self.min_pattern_occurrences:
                patterns = await self._detect_patterns(symptoms)
                self.pattern_cache[user_id] = patterns
        except Exception as e:
            self.logger.error(f"Error analyzing patterns for user {user_id}: {str(e)}")
    
    async def _detect_patterns(self, symptoms: List[SymptomEntry]) -> List[BehavioralPattern]:
        """Detect behavioral patterns from symptom data"""
        patterns = []
        
        if len(symptoms) < self.min_pattern_occurrences:
            return patterns
        
        # Time-based patterns
        patterns.extend(self._detect_time_patterns(symptoms))
        
        # Trigger-based patterns
        patterns.extend(self._detect_trigger_patterns(symptoms))
        
        # Intensity patterns
        patterns.extend(self._detect_intensity_patterns(symptoms))
        
        return patterns
    
    def _detect_time_patterns(self, symptoms: List[SymptomEntry]) -> List[BehavioralPattern]:
        """Detect time-based patterns (daily, weekly cycles)"""
        patterns = []
        
        # Hour of day patterns
        hour_intensity = defaultdict(list)
        for symptom in symptoms:
            hour_intensity[symptom.timestamp.hour].append(symptom.intensity)
        
        # Find peak hours
        hourly_avg = {hour: np.mean(intensities) for hour, intensities in hour_intensity.items()}
        if hourly_avg:
            peak_hour = max(hourly_avg, key=hourly_avg.get)
            if hourly_avg[peak_hour] > 0.6:  # High intensity threshold
                patterns.append(BehavioralPattern(
                    pattern_type="time_of_day",
                    confidence=min(hourly_avg[peak_hour], 1.0),
                    frequency="daily",
                    description=f"Symptoms tend to peak around {peak_hour}:00",
                    first_observed=min(s.timestamp for s in symptoms),
                    last_observed=max(s.timestamp for s in symptoms),
                    correlation_score=hourly_avg[peak_hour],
                    associated_symptoms=list(set(s.symptom_type for s in symptoms))
                ))
        
        return patterns
    
    def _detect_trigger_patterns(self, symptoms: List[SymptomEntry]) -> List[BehavioralPattern]:
        """Detect trigger-based patterns"""
        patterns = []
        
        # Count trigger occurrences
        trigger_counts = defaultdict(int)
        trigger_intensities = defaultdict(list)
        
        for symptom in symptoms:
            for trigger in symptom.triggers:
                trigger_counts[trigger] += 1
                trigger_intensities[trigger].append(symptom.intensity)
        
        # Identify significant triggers
        for trigger, count in trigger_counts.items():
            if count >= self.min_pattern_occurrences:
                avg_intensity = np.mean(trigger_intensities[trigger])
                if avg_intensity > 0.5:  # Significant intensity
                    patterns.append(BehavioralPattern(
                        pattern_type="trigger_correlation",
                        confidence=min(avg_intensity * (count / len(symptoms)), 1.0),
                        frequency="variable",
                        description=f"'{trigger}' is associated with increased symptoms",
                        first_observed=min(s.timestamp for s in symptoms if trigger in s.triggers),
                        last_observed=max(s.timestamp for s in symptoms if trigger in s.triggers),
                        correlation_score=avg_intensity,
                        associated_symptoms=list(set(s.symptom_type for s in symptoms if trigger in s.triggers))
                    ))
        
        return patterns
    
    def _detect_intensity_patterns(self, symptoms: List[SymptomEntry]) -> List[BehavioralPattern]:
        """Detect intensity fluctuation patterns"""
        patterns = []
        
        if len(symptoms) < 5:
            return patterns
        
        intensities = [s.intensity for s in symptoms]
        
        # Check for cyclical patterns
        high_intensity_days = sum(1 for i in intensities if i > 0.7)
        low_intensity_days = sum(1 for i in intensities if i < 0.3)
        
        if high_intensity_days > len(symptoms) * 0.3:
            patterns.append(BehavioralPattern(
                pattern_type="high_intensity_episodes",
                confidence=high_intensity_days / len(symptoms),
                frequency="regular",
                description="Frequent high-intensity symptom episodes",
                first_observed=min(s.timestamp for s in symptoms),
                last_observed=max(s.timestamp for s in symptoms),
                correlation_score=high_intensity_days / len(symptoms),
                associated_symptoms=list(set(s.symptom_type for s in symptoms))
            ))
        
        return patterns
    
    def _predict_trajectory(self, 
                          timestamps: List[datetime],
                          intensities: List[float],
                          prediction_days: int) -> Dict[str, Any]:
        """Predict symptom trajectory using historical data"""
        if len(timestamps) < 3:
            return {"error": "Insufficient data for prediction"}
        
        # Convert to numerical format
        x = np.array([(t - timestamps[0]).days for t in timestamps])
        y = np.array(intensities)
        
        # Fit linear trend
        try:
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            
            # Generate predictions
            future_days = np.arange(x[-1] + 1, x[-1] + prediction_days + 1)
            predictions = slope * future_days + intercept
            
            # Clamp predictions to valid range [0, 1]
            predictions = np.clip(predictions, 0, 1)
            
            return {
                "slope": float(slope),
                "current_trend": "improving" if slope < 0 else "worsening" if slope > 0 else "stable",
                "predictions": [
                    {
                        "day": int(day),
                        "predicted_intensity": float(pred),
                        "date": (timestamps[0] + timedelta(days=int(day))).isoformat()
                    }
                    for day, pred in zip(future_days, predictions)
                ]
            }
        except Exception as e:
            return {"error": f"Prediction calculation failed: {str(e)}"}
    
    def _calculate_prediction_confidence(self, symptoms: List[SymptomEntry]) -> float:
        """Calculate confidence in trajectory prediction"""
        if len(symptoms) < 5:
            return 0.1
        
        # Base confidence on data consistency and volume
        intensities = [s.intensity for s in symptoms]
        consistency = 1 - np.std(intensities)  # Lower std = higher consistency
        volume_factor = min(len(symptoms) / 30, 1)  # 30 data points = full confidence
        
        return max(0.1, min(0.9, consistency * volume_factor))
    
    def _generate_trajectory_recommendation(self, 
                                          prediction: Dict[str, Any],
                                          patterns: List[BehavioralPattern]) -> str:
        """Generate recommendations based on trajectory prediction"""
        if "error" in prediction:
            return "Continue tracking symptoms for better predictions."
        
        trend = prediction.get("current_trend", "stable")
        
        if trend == "improving":
            return "Your symptoms show a positive trend. Continue current strategies and consider maintaining lifestyle factors that may be contributing to improvement."
        elif trend == "worsening":
            rec = "Your symptoms show a concerning trend. "
            if patterns:
                trigger_patterns = [p for p in patterns if p.pattern_type == "trigger_correlation"]
                if trigger_patterns:
                    rec += f"Consider addressing triggers like: {', '.join(p.description for p in trigger_patterns[:2])}. "
            rec += "Consider reaching out to a healthcare professional for additional support."
            return rec
        else:
            return "Your symptoms are relatively stable. Continue monitoring and consider identifying potential improvement strategies."


class SymptomFrequencyMapper:
    """Maps and analyzes symptom frequency patterns"""
    
    def __init__(self, temporal_engine: TemporalAnalysisEngine):
        self.temporal_engine = temporal_engine
        self.logger = get_logger(__name__)
    
    async def analyze_frequency_patterns(self, 
                                       user_id: str,
                                       days_back: int = 30) -> Dict[str, Any]:
        """Analyze frequency patterns for all symptoms"""
        try:
            symptoms = await self.temporal_engine._get_symptoms_for_period(user_id, days_back)
            
            # Group by symptom type
            frequency_map = defaultdict(list)
            for symptom in symptoms:
                frequency_map[symptom.symptom_type].append(symptom.timestamp)
            
            analysis = {}
            for symptom_type, timestamps in frequency_map.items():
                analysis[symptom_type] = self._analyze_symptom_frequency(timestamps, days_back)
            
            return {
                "user_id": user_id,
                "analysis_period_days": days_back,
                "frequency_analysis": analysis,
                "overall_patterns": self._identify_overall_patterns(analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing frequency patterns: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_symptom_frequency(self, 
                                 timestamps: List[datetime],
                                 total_days: int) -> Dict[str, Any]:
        """Analyze frequency for a specific symptom"""
        if not timestamps:
            return {"frequency": 0, "pattern": "none"}
        
        # Calculate basic frequency
        frequency = len(timestamps) / total_days
        
        # Analyze intervals between occurrences
        if len(timestamps) > 1:
            sorted_timestamps = sorted(timestamps)
            intervals = []
            for i in range(1, len(sorted_timestamps)):
                interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).days
                intervals.append(interval)
            
            avg_interval = np.mean(intervals)
            interval_std = np.std(intervals) if len(intervals) > 1 else 0
        else:
            avg_interval = total_days
            interval_std = 0
        
        # Determine pattern type
        if frequency > 0.8:
            pattern = "daily"
        elif frequency > 0.3:
            pattern = "frequent"
        elif frequency > 0.1:
            pattern = "occasional"
        else:
            pattern = "rare"
        
        return {
            "frequency": round(frequency, 3),
            "occurrences": len(timestamps),
            "pattern": pattern,
            "average_interval_days": round(avg_interval, 1),
            "interval_consistency": round(1 - (interval_std / max(avg_interval, 1)), 2),
            "first_occurrence": min(timestamps).isoformat(),
            "last_occurrence": max(timestamps).isoformat()
        }
    
    def _identify_overall_patterns(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify overall patterns across all symptoms"""
        if not analysis:
            return {}
        
        # Find most frequent symptoms
        frequent_symptoms = [
            symptom for symptom, data in analysis.items()
            if data.get("frequency", 0) > 0.3
        ]
        
        # Find clustered symptoms (occurring together)
        clustered_patterns = []
        # TODO: Implement clustering analysis
        
        return {
            "most_frequent_symptoms": frequent_symptoms,
            "total_symptom_types": len(analysis),
            "average_frequency": np.mean([data.get("frequency", 0) for data in analysis.values()]),
            "clustered_patterns": clustered_patterns
        }
"""
Population Health Analytics and Insights
Analyzes mental health trends across populations while preserving privacy
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import hashlib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PopulationSegment(Enum):
    AGE_ADOLESCENT = "age_13_17"
    AGE_YOUNG_ADULT = "age_18_25"
    AGE_ADULT = "age_26_64"
    AGE_SENIOR = "age_65_plus"
    GENDER_MALE = "male"
    GENDER_FEMALE = "female"
    GENDER_OTHER = "other"
    SEVERITY_MILD = "mild"
    SEVERITY_MODERATE = "moderate"
    SEVERITY_SEVERE = "severe"
    SOCIOECONOMIC_LOW = "low_ses"
    SOCIOECONOMIC_MIDDLE = "middle_ses"
    SOCIOECONOMIC_HIGH = "high_ses"


class HealthMetric(Enum):
    TREATMENT_ENGAGEMENT = "treatment_engagement"
    SYMPTOM_IMPROVEMENT = "symptom_improvement"
    CRISIS_INCIDENTS = "crisis_incidents"
    DROPOUT_RATE = "dropout_rate"
    RECOVERY_TIME = "recovery_time"
    RELAPSE_RATE = "relapse_rate"
    QUALITY_OF_LIFE = "quality_of_life"
    FUNCTIONAL_IMPROVEMENT = "functional_improvement"


@dataclass
class AnonymizedPatientRecord:
    """Anonymized patient record for population analysis"""
    record_id: str  # Hash-based anonymous ID
    age_group: PopulationSegment
    gender: PopulationSegment
    diagnosis_category: str
    severity_level: PopulationSegment
    socioeconomic_status: PopulationSegment
    treatment_type: str
    treatment_duration_days: int
    session_count: int
    engagement_score: float
    baseline_assessment: Dict[str, float]
    outcome_assessment: Dict[str, float]
    treatment_completion: bool
    crisis_incidents: int
    adverse_events: int
    geographical_region: str  # Broad region, not specific location
    created_date: datetime
    outcome_date: Optional[datetime] = None


@dataclass
class PopulationInsight:
    """Population health insight"""
    insight_id: str
    title: str
    description: str
    population_segment: Optional[PopulationSegment]
    metric: HealthMetric
    finding_type: str  # trend, anomaly, correlation, comparison
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    time_period: Tuple[datetime, datetime]
    clinical_relevance: str  # high, medium, low
    actionable_recommendations: List[str]
    data_source: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric: HealthMetric
    time_period: str
    trend_direction: str  # increasing, decreasing, stable
    trend_magnitude: float
    seasonal_patterns: Dict[str, float]
    change_points: List[datetime]
    forecast: List[Tuple[datetime, float, float]]  # date, value, confidence
    statistical_tests: Dict[str, Any]


class PrivacyPreservingAnalyzer:
    """Analyzes population data while preserving individual privacy"""
    
    def __init__(self, min_group_size: int = 10, noise_factor: float = 0.05):
        self.min_group_size = min_group_size
        self.noise_factor = noise_factor
        
    def anonymize_record(self, patient_data: Dict[str, Any]) -> AnonymizedPatientRecord:
        """Convert patient data to anonymized record"""
        # Create anonymous ID using hash
        identifiers = f"{patient_data.get('original_id', '')}{patient_data.get('dob', '')}"
        record_id = hashlib.sha256(identifiers.encode()).hexdigest()[:16]
        
        # Categorize age
        age = patient_data.get('age', 0)
        if 13 <= age <= 17:
            age_group = PopulationSegment.AGE_ADOLESCENT
        elif 18 <= age <= 25:
            age_group = PopulationSegment.AGE_YOUNG_ADULT
        elif 26 <= age <= 64:
            age_group = PopulationSegment.AGE_ADULT
        else:
            age_group = PopulationSegment.AGE_SENIOR
            
        # Categorize gender
        gender = patient_data.get('gender', '').lower()
        if gender in ['male', 'm']:
            gender_segment = PopulationSegment.GENDER_MALE
        elif gender in ['female', 'f']:
            gender_segment = PopulationSegment.GENDER_FEMALE
        else:
            gender_segment = PopulationSegment.GENDER_OTHER
            
        # Categorize severity
        severity_score = patient_data.get('severity_score', 0)
        if severity_score <= 3:
            severity_level = PopulationSegment.SEVERITY_MILD
        elif severity_score <= 6:
            severity_level = PopulationSegment.SEVERITY_MODERATE
        else:
            severity_level = PopulationSegment.SEVERITY_SEVERE
            
        # Categorize socioeconomic status (simplified)
        income = patient_data.get('household_income', 0)
        if income < 30000:
            ses = PopulationSegment.SOCIOECONOMIC_LOW
        elif income < 80000:
            ses = PopulationSegment.SOCIOECONOMIC_MIDDLE
        else:
            ses = PopulationSegment.SOCIOECONOMIC_HIGH
            
        return AnonymizedPatientRecord(
            record_id=record_id,
            age_group=age_group,
            gender=gender_segment,
            diagnosis_category=patient_data.get('diagnosis_category', 'unknown'),
            severity_level=severity_level,
            socioeconomic_status=ses,
            treatment_type=patient_data.get('treatment_type', 'unknown'),
            treatment_duration_days=patient_data.get('treatment_duration_days', 0),
            session_count=patient_data.get('session_count', 0),
            engagement_score=patient_data.get('engagement_score', 0),
            baseline_assessment=patient_data.get('baseline_assessment', {}),
            outcome_assessment=patient_data.get('outcome_assessment', {}),
            treatment_completion=patient_data.get('treatment_completion', False),
            crisis_incidents=patient_data.get('crisis_incidents', 0),
            adverse_events=patient_data.get('adverse_events', 0),
            geographical_region=patient_data.get('geographical_region', 'unknown'),
            created_date=datetime.fromisoformat(patient_data.get('created_date', datetime.now().isoformat())),
            outcome_date=datetime.fromisoformat(patient_data['outcome_date']) if patient_data.get('outcome_date') else None
        )
        
    def add_differential_privacy_noise(self, value: float) -> float:
        """Add noise for differential privacy"""
        noise = np.random.normal(0, self.noise_factor * abs(value))
        return value + noise
    
    def check_group_size(self, group_size: int) -> bool:
        """Check if group size meets minimum threshold"""
        return group_size >= self.min_group_size
        
    def aggregate_with_privacy(self, values: List[float], operation: str = 'mean') -> Optional[float]:
        """Perform aggregation with privacy protection"""
        if len(values) < self.min_group_size:
            return None
            
        if operation == 'mean':
            result = np.mean(values)
        elif operation == 'median':
            result = np.median(values)
        elif operation == 'sum':
            result = np.sum(values)
        elif operation == 'count':
            result = len(values)
        else:
            result = np.mean(values)
            
        return self.add_differential_privacy_noise(result)


class PopulationHealthAnalyzer:
    """Main population health analytics engine"""
    
    def __init__(self):
        self.privacy_analyzer = PrivacyPreservingAnalyzer()
        self.population_data: List[AnonymizedPatientRecord] = []
        self.insights_cache: List[PopulationInsight] = []
        
    async def load_population_data(self, raw_data: List[Dict[str, Any]]):
        """Load and anonymize population data"""
        logger.info(f"Loading {len(raw_data)} patient records for population analysis")
        
        self.population_data = []
        for record in raw_data:
            try:
                anonymized_record = self.privacy_analyzer.anonymize_record(record)
                self.population_data.append(anonymized_record)
            except Exception as e:
                logger.warning(f"Failed to anonymize record: {e}")
                
        logger.info(f"Successfully loaded {len(self.population_data)} anonymized records")
        
    async def analyze_treatment_outcomes_by_segment(self, segment: PopulationSegment) -> Dict[str, Any]:
        """Analyze treatment outcomes for a specific population segment"""
        segment_records = [r for r in self.population_data if 
                          r.age_group == segment or r.gender == segment or 
                          r.severity_level == segment or r.socioeconomic_status == segment]
        
        if not self.privacy_analyzer.check_group_size(len(segment_records)):
            return {"error": "Insufficient data for analysis (privacy protection)"}
            
        # Calculate outcome metrics
        completion_rate = self.privacy_analyzer.aggregate_with_privacy(
            [float(r.treatment_completion) for r in segment_records], 'mean'
        )
        
        engagement_scores = [r.engagement_score for r in segment_records if r.engagement_score > 0]
        avg_engagement = self.privacy_analyzer.aggregate_with_privacy(engagement_scores, 'mean')
        
        session_counts = [r.session_count for r in segment_records if r.session_count > 0]
        avg_sessions = self.privacy_analyzer.aggregate_with_privacy(session_counts, 'mean')
        
        crisis_incidents = [r.crisis_incidents for r in segment_records]
        avg_crisis_rate = self.privacy_analyzer.aggregate_with_privacy(crisis_incidents, 'mean')
        
        # Calculate improvement scores
        improvement_scores = []
        for record in segment_records:
            if record.baseline_assessment and record.outcome_assessment:
                baseline_phq9 = record.baseline_assessment.get('phq9', 0)
                outcome_phq9 = record.outcome_assessment.get('phq9', 0)
                if baseline_phq9 > 0:
                    improvement = (baseline_phq9 - outcome_phq9) / baseline_phq9
                    improvement_scores.append(improvement)
                    
        avg_improvement = self.privacy_analyzer.aggregate_with_privacy(improvement_scores, 'mean')
        
        return {
            "segment": segment.value,
            "sample_size": len(segment_records),
            "completion_rate": completion_rate,
            "average_engagement": avg_engagement,
            "average_sessions": avg_sessions,
            "average_crisis_rate": avg_crisis_rate,
            "average_symptom_improvement": avg_improvement,
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    async def compare_segments(self, segment1: PopulationSegment, 
                             segment2: PopulationSegment, 
                             metric: HealthMetric) -> Dict[str, Any]:
        """Compare outcomes between two population segments"""
        
        # Get data for each segment
        records1 = [r for r in self.population_data if 
                   r.age_group == segment1 or r.gender == segment1 or 
                   r.severity_level == segment1 or r.socioeconomic_status == segment1]
        
        records2 = [r for r in self.population_data if 
                   r.age_group == segment2 or r.gender == segment2 or 
                   r.severity_level == segment2 or r.socioeconomic_status == segment2]
        
        if (not self.privacy_analyzer.check_group_size(len(records1)) or 
            not self.privacy_analyzer.check_group_size(len(records2))):
            return {"error": "Insufficient data for comparison (privacy protection)"}
            
        # Extract metric values
        values1 = self._extract_metric_values(records1, metric)
        values2 = self._extract_metric_values(records2, metric)
        
        if not values1 or not values2:
            return {"error": f"No data available for metric {metric.value}"}
            
        # Statistical comparison
        try:
            # t-test for continuous metrics
            if metric in [HealthMetric.SYMPTOM_IMPROVEMENT, HealthMetric.TREATMENT_ENGAGEMENT, 
                         HealthMetric.QUALITY_OF_LIFE, HealthMetric.RECOVERY_TIME]:
                statistic, p_value = stats.ttest_ind(values1, values2)
                test_type = "t-test"
            else:
                # Chi-square for categorical metrics
                statistic, p_value = stats.chi2_contingency([values1, values2])[:2]
                test_type = "chi-square"
                
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                                (len(values2) - 1) * np.var(values2)) / 
                               (len(values1) + len(values2) - 2))
            effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
            
            # Add noise for privacy
            mean1 = self.privacy_analyzer.add_differential_privacy_noise(np.mean(values1))
            mean2 = self.privacy_analyzer.add_differential_privacy_noise(np.mean(values2))
            
            return {
                "segment1": segment1.value,
                "segment2": segment2.value,
                "metric": metric.value,
                "segment1_mean": mean1,
                "segment2_mean": mean2,
                "segment1_size": len(records1),
                "segment2_size": len(records2),
                "statistical_test": test_type,
                "test_statistic": statistic,
                "p_value": p_value,
                "effect_size": effect_size,
                "statistically_significant": p_value < 0.05,
                "clinically_significant": abs(effect_size) > 0.5,
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Statistical comparison failed: {e}")
            return {"error": "Statistical analysis failed"}
            
    async def identify_health_disparities(self) -> List[PopulationInsight]:
        """Identify health disparities across population segments"""
        disparities = []
        
        segments_to_compare = [
            (PopulationSegment.SOCIOECONOMIC_LOW, PopulationSegment.SOCIOECONOMIC_HIGH),
            (PopulationSegment.GENDER_MALE, PopulationSegment.GENDER_FEMALE),
            (PopulationSegment.AGE_ADOLESCENT, PopulationSegment.AGE_ADULT),
            (PopulationSegment.SEVERITY_MILD, PopulationSegment.SEVERITY_SEVERE)
        ]
        
        metrics_to_analyze = [
            HealthMetric.TREATMENT_ENGAGEMENT,
            HealthMetric.SYMPTOM_IMPROVEMENT,
            HealthMetric.DROPOUT_RATE,
            HealthMetric.CRISIS_INCIDENTS
        ]
        
        for segment1, segment2 in segments_to_compare:
            for metric in metrics_to_analyze:
                try:
                    comparison = await self.compare_segments(segment1, segment2, metric)
                    
                    if "error" not in comparison and comparison.get("statistically_significant"):
                        # Determine if this represents a disparity
                        effect_size = abs(comparison.get("effect_size", 0))
                        
                        if effect_size > 0.5:  # Medium to large effect
                            insight = PopulationInsight(
                                insight_id=f"disparity_{segment1.value}_{segment2.value}_{metric.value}",
                                title=f"Health Disparity: {metric.value.replace('_', ' ').title()}",
                                description=f"Significant difference in {metric.value} between {segment1.value} and {segment2.value} populations",
                                population_segment=None,
                                metric=metric,
                                finding_type="disparity",
                                statistical_significance=comparison["p_value"],
                                effect_size=comparison["effect_size"],
                                confidence_interval=(comparison["segment1_mean"] - comparison["segment2_mean"] - 0.1,
                                                   comparison["segment1_mean"] - comparison["segment2_mean"] + 0.1),
                                sample_size=comparison["segment1_size"] + comparison["segment2_size"],
                                time_period=(datetime.utcnow() - timedelta(days=365), datetime.utcnow()),
                                clinical_relevance="high" if effect_size > 0.8 else "medium",
                                actionable_recommendations=self._generate_disparity_recommendations(
                                    segment1, segment2, metric, comparison),
                                data_source="population_health_analysis"
                            )
                            disparities.append(insight)
                            
                except Exception as e:
                    logger.error(f"Failed to analyze disparity for {segment1} vs {segment2} on {metric}: {e}")
                    
        return disparities
        
    async def analyze_treatment_effectiveness(self, treatment_type: str) -> Dict[str, Any]:
        """Analyze effectiveness of a specific treatment type"""
        treatment_records = [r for r in self.population_data if r.treatment_type == treatment_type]
        
        if not self.privacy_analyzer.check_group_size(len(treatment_records)):
            return {"error": "Insufficient data for analysis (privacy protection)"}
            
        # Calculate effectiveness metrics
        completion_rates = [float(r.treatment_completion) for r in treatment_records]
        avg_completion = self.privacy_analyzer.aggregate_with_privacy(completion_rates, 'mean')
        
        engagement_scores = [r.engagement_score for r in treatment_records if r.engagement_score > 0]
        avg_engagement = self.privacy_analyzer.aggregate_with_privacy(engagement_scores, 'mean')
        
        # Symptom improvement
        improvement_scores = []
        for record in treatment_records:
            if record.baseline_assessment and record.outcome_assessment:
                baseline_total = sum(record.baseline_assessment.values())
                outcome_total = sum(record.outcome_assessment.values())
                if baseline_total > 0:
                    improvement = (baseline_total - outcome_total) / baseline_total
                    improvement_scores.append(improvement)
                    
        avg_improvement = self.privacy_analyzer.aggregate_with_privacy(improvement_scores, 'mean')
        
        # Duration analysis
        durations = [r.treatment_duration_days for r in treatment_records if r.treatment_duration_days > 0]
        avg_duration = self.privacy_analyzer.aggregate_with_privacy(durations, 'mean')
        
        # Crisis incidents
        crisis_rates = [r.crisis_incidents for r in treatment_records]
        avg_crisis_rate = self.privacy_analyzer.aggregate_with_privacy(crisis_rates, 'mean')
        
        return {
            "treatment_type": treatment_type,
            "sample_size": len(treatment_records),
            "completion_rate": avg_completion,
            "average_engagement": avg_engagement,
            "average_improvement": avg_improvement,
            "average_duration_days": avg_duration,
            "average_crisis_rate": avg_crisis_rate,
            "effectiveness_score": self._calculate_effectiveness_score(
                avg_completion, avg_engagement, avg_improvement, avg_crisis_rate),
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    async def generate_population_trends(self, time_period_days: int = 365) -> List[TrendAnalysis]:
        """Generate trend analysis for population health metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
        recent_records = [r for r in self.population_data if r.created_date >= cutoff_date]
        
        if not self.privacy_analyzer.check_group_size(len(recent_records)):
            return []
            
        trends = []
        
        # Group records by month
        monthly_data = defaultdict(list)
        for record in recent_records:
            month_key = record.created_date.strftime("%Y-%m")
            monthly_data[month_key].append(record)
            
        # Analyze trends for key metrics
        metrics_to_analyze = [
            HealthMetric.TREATMENT_ENGAGEMENT,
            HealthMetric.SYMPTOM_IMPROVEMENT,
            HealthMetric.DROPOUT_RATE,
            HealthMetric.CRISIS_INCIDENTS
        ]
        
        for metric in metrics_to_analyze:
            try:
                monthly_values = []
                months = []
                
                for month in sorted(monthly_data.keys()):
                    if self.privacy_analyzer.check_group_size(len(monthly_data[month])):
                        metric_values = self._extract_metric_values(monthly_data[month], metric)
                        if metric_values:
                            avg_value = self.privacy_analyzer.aggregate_with_privacy(metric_values, 'mean')
                            if avg_value is not None:
                                monthly_values.append(avg_value)
                                months.append(month)
                                
                if len(monthly_values) >= 3:  # Need at least 3 months for trend
                    # Simple trend analysis
                    x = range(len(monthly_values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, monthly_values)
                    
                    trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    
                    trend = TrendAnalysis(
                        metric=metric,
                        time_period=f"{time_period_days}_days",
                        trend_direction=trend_direction,
                        trend_magnitude=abs(slope),
                        seasonal_patterns={},  # Simplified for now
                        change_points=[],      # Would need more sophisticated analysis
                        forecast=[],           # Would need time series forecasting
                        statistical_tests={
                            "linear_regression_slope": slope,
                            "r_squared": r_value**2,
                            "p_value": p_value
                        }
                    )
                    trends.append(trend)
                    
            except Exception as e:
                logger.error(f"Failed to analyze trend for {metric}: {e}")
                
        return trends
        
    def _extract_metric_values(self, records: List[AnonymizedPatientRecord], 
                             metric: HealthMetric) -> List[float]:
        """Extract values for a specific metric from records"""
        values = []
        
        for record in records:
            if metric == HealthMetric.TREATMENT_ENGAGEMENT:
                values.append(record.engagement_score)
            elif metric == HealthMetric.SYMPTOM_IMPROVEMENT:
                if record.baseline_assessment and record.outcome_assessment:
                    baseline = sum(record.baseline_assessment.values())
                    outcome = sum(record.outcome_assessment.values())
                    if baseline > 0:
                        improvement = (baseline - outcome) / baseline
                        values.append(improvement)
            elif metric == HealthMetric.CRISIS_INCIDENTS:
                values.append(record.crisis_incidents)
            elif metric == HealthMetric.DROPOUT_RATE:
                values.append(float(not record.treatment_completion))
            elif metric == HealthMetric.RECOVERY_TIME:
                if record.outcome_date:
                    duration = (record.outcome_date - record.created_date).days
                    values.append(duration)
                    
        return values
        
    def _calculate_effectiveness_score(self, completion_rate: Optional[float], 
                                     engagement: Optional[float], 
                                     improvement: Optional[float], 
                                     crisis_rate: Optional[float]) -> float:
        """Calculate overall treatment effectiveness score"""
        scores = []
        weights = []
        
        if completion_rate is not None:
            scores.append(completion_rate)
            weights.append(0.3)
            
        if engagement is not None:
            scores.append(engagement / 10.0)  # Normalize to 0-1
            weights.append(0.2)
            
        if improvement is not None:
            scores.append(max(0, improvement))  # Only positive improvements
            weights.append(0.4)
            
        if crisis_rate is not None:
            scores.append(max(0, 1 - crisis_rate))  # Lower crisis rate is better
            weights.append(0.1)
            
        if not scores:
            return 0.0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return min(1.0, max(0.0, weighted_score))
        
    def _generate_disparity_recommendations(self, segment1: PopulationSegment, 
                                          segment2: PopulationSegment, 
                                          metric: HealthMetric, 
                                          comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations to address health disparities"""
        recommendations = []
        
        # Determine which segment is performing worse
        worse_segment = segment1 if comparison["segment1_mean"] < comparison["segment2_mean"] else segment2
        
        if metric == HealthMetric.TREATMENT_ENGAGEMENT:
            recommendations.append(f"Develop targeted engagement strategies for {worse_segment.value} population")
            recommendations.append("Consider cultural competency training for providers")
            recommendations.append("Implement peer support programs")
            
        elif metric == HealthMetric.SYMPTOM_IMPROVEMENT:
            recommendations.append(f"Tailor treatment approaches for {worse_segment.value} population")
            recommendations.append("Investigate barriers to treatment effectiveness")
            recommendations.append("Consider supplemental interventions")
            
        elif metric == HealthMetric.DROPOUT_RATE:
            recommendations.append(f"Address barriers to treatment continuation for {worse_segment.value}")
            recommendations.append("Implement retention-focused interventions")
            recommendations.append("Improve accessibility and convenience")
            
        elif metric == HealthMetric.CRISIS_INCIDENTS:
            recommendations.append(f"Enhance crisis prevention for {worse_segment.value} population")
            recommendations.append("Implement early warning systems")
            recommendations.append("Strengthen safety planning protocols")
            
        return recommendations
        
    async def get_population_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for population health dashboard"""
        if not self.population_data:
            return {"error": "No population data available"}
            
        total_records = len(self.population_data)
        
        # Overall metrics
        completion_rates = [float(r.treatment_completion) for r in self.population_data]
        overall_completion = self.privacy_analyzer.aggregate_with_privacy(completion_rates, 'mean')
        
        engagement_scores = [r.engagement_score for r in self.population_data if r.engagement_score > 0]
        overall_engagement = self.privacy_analyzer.aggregate_with_privacy(engagement_scores, 'mean')
        
        crisis_incidents = [r.crisis_incidents for r in self.population_data]
        overall_crisis_rate = self.privacy_analyzer.aggregate_with_privacy(crisis_incidents, 'mean')
        
        # Demographic distribution
        age_distribution = Counter([r.age_group.value for r in self.population_data])
        gender_distribution = Counter([r.gender.value for r in self.population_data])
        severity_distribution = Counter([r.severity_level.value for r in self.population_data])
        
        # Treatment type distribution
        treatment_distribution = Counter([r.treatment_type for r in self.population_data])
        
        return {
            "total_population": total_records,
            "overall_metrics": {
                "completion_rate": overall_completion,
                "engagement_score": overall_engagement,
                "crisis_rate": overall_crisis_rate
            },
            "demographics": {
                "age_distribution": dict(age_distribution),
                "gender_distribution": dict(gender_distribution),
                "severity_distribution": dict(severity_distribution)
            },
            "treatment_distribution": dict(treatment_distribution),
            "data_freshness": datetime.utcnow().isoformat(),
            "privacy_protected": True
        }
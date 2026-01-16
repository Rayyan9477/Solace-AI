"""
Solace-AI Therapy Service - Progress Tracking and Outcome Measurement.
Evidence-based progress monitoring using validated instruments and outcome tracking.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import SeverityLevel, ResponseStatus, OutcomeInstrument

logger = structlog.get_logger(__name__)


class MeasureType(str, Enum):
    """Types of outcome measures."""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    WELLBEING = "wellbeing"
    FUNCTIONING = "functioning"
    QUALITY_OF_LIFE = "quality_of_life"
    SYMPTOM_SEVERITY = "symptom_severity"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    SESSION_RATING = "session_rating"


class InstrumentType(str, Enum):
    """Validated assessment instruments."""
    PHQ9 = "phq9"
    GAD7 = "gad7"
    WHO5 = "who5"
    ORS = "ors"
    SRS = "srs"
    PCL5 = "pcl5"
    WHODAS = "whodas"
    CUSTOM = "custom"


class ChangeCategory(str, Enum):
    """Clinical change categories."""
    RECOVERED = "recovered"
    IMPROVED = "improved"
    NO_CHANGE = "no_change"
    DETERIORATED = "deteriorated"


@dataclass
class MeasureScore:
    """Single measurement score."""
    score_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    instrument: InstrumentType = InstrumentType.CUSTOM
    measure_type: MeasureType = MeasureType.SYMPTOM_SEVERITY
    raw_score: float = 0.0
    normalized_score: float = 0.0
    severity_label: str = ""
    item_responses: dict[str, int] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""


@dataclass
class ProgressMetric:
    """Progress metric over time."""
    metric_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    measure_type: MeasureType = MeasureType.SYMPTOM_SEVERITY
    baseline_score: float = 0.0
    current_score: float = 0.0
    target_score: float | None = None
    percent_change: float = 0.0
    change_category: ChangeCategory = ChangeCategory.NO_CHANGE
    clinically_significant: bool = False
    reliable_change: bool = False
    scores_history: list[MeasureScore] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class OutcomeReport:
    """Comprehensive outcome report."""
    report_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    plan_id: UUID | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reporting_period_days: int = 30
    metrics: list[ProgressMetric] = field(default_factory=list)
    sessions_completed: int = 0
    homework_completion_rate: float = 0.0
    engagement_score: float = 0.0
    overall_change: ChangeCategory = ChangeCategory.NO_CHANGE
    recommendations: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


class InstrumentConfig:
    """Configuration for validated instruments."""
    INSTRUMENTS = {
        InstrumentType.PHQ9: {
            "name": "Patient Health Questionnaire-9",
            "measure_type": MeasureType.DEPRESSION,
            "min_score": 0, "max_score": 27, "items": 9,
            "severity_thresholds": [(5, "minimal"), (10, "mild"), (15, "moderate"), (20, "moderately_severe"), (27, "severe")],
            "rci": 6.0, "clinical_cutoff": 10,
        },
        InstrumentType.GAD7: {
            "name": "Generalized Anxiety Disorder-7",
            "measure_type": MeasureType.ANXIETY,
            "min_score": 0, "max_score": 21, "items": 7,
            "severity_thresholds": [(5, "minimal"), (10, "mild"), (15, "moderate"), (21, "severe")],
            "rci": 4.0, "clinical_cutoff": 10,
        },
        InstrumentType.WHO5: {
            "name": "WHO-5 Well-Being Index",
            "measure_type": MeasureType.WELLBEING,
            "min_score": 0, "max_score": 100, "items": 5,
            "severity_thresholds": [(28, "low"), (50, "moderate"), (100, "high")],
            "rci": 10.0, "clinical_cutoff": 50, "higher_better": True,
        },
        InstrumentType.ORS: {
            "name": "Outcome Rating Scale",
            "measure_type": MeasureType.FUNCTIONING,
            "min_score": 0, "max_score": 40, "items": 4,
            "severity_thresholds": [(25, "clinical"), (40, "non_clinical")],
            "rci": 5.0, "clinical_cutoff": 25, "higher_better": True,
        },
        InstrumentType.SRS: {
            "name": "Session Rating Scale",
            "measure_type": MeasureType.THERAPEUTIC_ALLIANCE,
            "min_score": 0, "max_score": 40, "items": 4,
            "severity_thresholds": [(36, "at_risk"), (40, "good")],
            "rci": 5.0, "clinical_cutoff": 36, "higher_better": True,
        },
    }

    @classmethod
    def get_config(cls, instrument: InstrumentType) -> dict[str, Any]:
        """Get instrument configuration."""
        return cls.INSTRUMENTS.get(instrument, {})

    @classmethod
    def get_severity_label(cls, instrument: InstrumentType, score: float) -> str:
        """Get severity label for score."""
        config = cls.get_config(instrument)
        thresholds = config.get("severity_thresholds", [])
        label = "unknown"
        for threshold, name in thresholds:
            if score <= threshold:
                label = name
                break
        return label


class ProgressTrackerSettings(BaseSettings):
    """Progress tracker configuration."""
    measurement_frequency_days: int = Field(default=7)
    enable_automated_alerts: bool = Field(default=True)
    deterioration_threshold: float = Field(default=-0.2)
    improvement_threshold: float = Field(default=0.2)
    enable_reliable_change: bool = Field(default=True)
    report_period_days: int = Field(default=30)
    model_config = SettingsConfigDict(env_prefix="PROGRESS_TRACKER_", env_file=".env", extra="ignore")


class ProgressTracker:
    """
    Tracks therapy progress and outcomes.

    Uses validated instruments and evidence-based metrics to monitor
    treatment progress, detect deterioration, and generate outcome reports.
    """

    def __init__(self, settings: ProgressTrackerSettings | None = None) -> None:
        self._settings = settings or ProgressTrackerSettings()
        self._scores: dict[UUID, list[MeasureScore]] = {}
        self._metrics: dict[UUID, dict[MeasureType, ProgressMetric]] = {}
        self._stats = {"scores_recorded": 0, "alerts_generated": 0, "reports_generated": 0}
        logger.info("progress_tracker_initialized", automated_alerts=self._settings.enable_automated_alerts)

    def record_score(
        self,
        user_id: UUID,
        instrument: InstrumentType,
        raw_score: float,
        session_id: UUID | None = None,
        item_responses: dict[str, int] | None = None,
    ) -> MeasureScore:
        """
        Record an assessment score.

        Args:
            user_id: User identifier
            instrument: Assessment instrument used
            raw_score: Raw score value
            session_id: Associated session ID
            item_responses: Individual item responses

        Returns:
            Recorded score with analysis
        """
        config = InstrumentConfig.get_config(instrument)
        min_score = config.get("min_score", 0)
        max_score = config.get("max_score", 100)
        clamped_score = max(min_score, min(max_score, raw_score))
        normalized = (clamped_score - min_score) / (max_score - min_score) if max_score > min_score else 0
        severity = InstrumentConfig.get_severity_label(instrument, clamped_score)
        score = MeasureScore(
            user_id=user_id, session_id=session_id, instrument=instrument,
            measure_type=config.get("measure_type", MeasureType.SYMPTOM_SEVERITY),
            raw_score=clamped_score, normalized_score=normalized, severity_label=severity,
            item_responses=item_responses or {},
        )
        if user_id not in self._scores:
            self._scores[user_id] = []
        self._scores[user_id].append(score)
        self._stats["scores_recorded"] += 1
        self._update_progress_metric(user_id, score)
        logger.debug("score_recorded", user_id=str(user_id), instrument=instrument.value, score=clamped_score, severity=severity)
        return score

    def _update_progress_metric(self, user_id: UUID, score: MeasureScore) -> None:
        """Update progress metric with new score."""
        if user_id not in self._metrics:
            self._metrics[user_id] = {}
        measure_type = score.measure_type
        if measure_type not in self._metrics[user_id]:
            self._metrics[user_id][measure_type] = ProgressMetric(
                user_id=user_id, measure_type=measure_type, baseline_score=score.raw_score, current_score=score.raw_score,
            )
        metric = self._metrics[user_id][measure_type]
        metric.current_score = score.raw_score
        metric.scores_history.append(score)
        metric.last_updated = datetime.now(timezone.utc)
        self._calculate_change(metric, score.instrument)

    def _calculate_change(self, metric: ProgressMetric, instrument: InstrumentType) -> None:
        """Calculate change statistics for metric."""
        if metric.baseline_score == 0:
            metric.percent_change = 0.0
            return
        config = InstrumentConfig.get_config(instrument)
        higher_better = config.get("higher_better", False)
        raw_change = metric.current_score - metric.baseline_score
        if higher_better:
            metric.percent_change = raw_change / metric.baseline_score if metric.baseline_score != 0 else 0
        else:
            metric.percent_change = -raw_change / metric.baseline_score if metric.baseline_score != 0 else 0
        rci = config.get("rci", 5.0)
        metric.reliable_change = abs(raw_change) >= rci if self._settings.enable_reliable_change else False
        clinical_cutoff = config.get("clinical_cutoff")
        if clinical_cutoff is not None:
            if higher_better:
                crossed_cutoff = metric.baseline_score < clinical_cutoff <= metric.current_score
            else:
                crossed_cutoff = metric.baseline_score >= clinical_cutoff > metric.current_score
            metric.clinically_significant = metric.reliable_change and crossed_cutoff
        metric.change_category = self._categorize_change(metric, higher_better)

    def _categorize_change(self, metric: ProgressMetric, higher_better: bool) -> ChangeCategory:
        """Categorize the change."""
        if metric.clinically_significant and metric.percent_change > 0:
            return ChangeCategory.RECOVERED
        elif metric.percent_change >= self._settings.improvement_threshold:
            return ChangeCategory.IMPROVED
        elif metric.percent_change <= self._settings.deterioration_threshold:
            return ChangeCategory.DETERIORATED
        return ChangeCategory.NO_CHANGE

    def get_user_scores(
        self,
        user_id: UUID,
        instrument: InstrumentType | None = None,
        days: int | None = None,
    ) -> list[MeasureScore]:
        """Get scores for a user."""
        scores = self._scores.get(user_id, [])
        if instrument:
            scores = [s for s in scores if s.instrument == instrument]
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            scores = [s for s in scores if s.recorded_at >= cutoff]
        return sorted(scores, key=lambda x: x.recorded_at, reverse=True)

    def get_progress_metric(self, user_id: UUID, measure_type: MeasureType) -> ProgressMetric | None:
        """Get progress metric for a user and measure type."""
        return self._metrics.get(user_id, {}).get(measure_type)

    def get_all_metrics(self, user_id: UUID) -> list[ProgressMetric]:
        """Get all progress metrics for a user."""
        return list(self._metrics.get(user_id, {}).values())

    def check_deterioration(self, user_id: UUID) -> list[dict[str, Any]]:
        """Check for deterioration across all metrics."""
        alerts = []
        metrics = self._metrics.get(user_id, {})
        for measure_type, metric in metrics.items():
            if metric.change_category == ChangeCategory.DETERIORATED:
                alert = {
                    "type": "deterioration",
                    "measure": measure_type.value,
                    "percent_change": metric.percent_change,
                    "baseline": metric.baseline_score,
                    "current": metric.current_score,
                    "reliable": metric.reliable_change,
                }
                alerts.append(alert)
                if self._settings.enable_automated_alerts:
                    self._stats["alerts_generated"] += 1
                    logger.warning("deterioration_detected", user_id=str(user_id), measure=measure_type.value, change=metric.percent_change)
        return alerts

    def check_measurement_due(self, user_id: UUID, instrument: InstrumentType) -> bool:
        """Check if a measurement is due."""
        scores = self.get_user_scores(user_id, instrument=instrument)
        if not scores:
            return True
        last_score = scores[0]
        days_since = (datetime.now(timezone.utc) - last_score.recorded_at).days
        return days_since >= self._settings.measurement_frequency_days

    def generate_outcome_report(
        self,
        user_id: UUID,
        plan_id: UUID | None = None,
        period_days: int | None = None,
        sessions_completed: int = 0,
        homework_rate: float = 0.0,
        engagement_score: float = 0.0,
    ) -> OutcomeReport:
        """
        Generate comprehensive outcome report.

        Args:
            user_id: User identifier
            plan_id: Treatment plan ID
            period_days: Reporting period in days
            sessions_completed: Number of sessions completed
            homework_rate: Homework completion rate
            engagement_score: Overall engagement score

        Returns:
            Generated outcome report
        """
        period = period_days or self._settings.report_period_days
        metrics = self.get_all_metrics(user_id)
        recommendations = self._generate_recommendations(metrics, homework_rate, engagement_score)
        risk_flags = self._identify_risk_flags(metrics)
        overall = self._determine_overall_change(metrics)
        report = OutcomeReport(
            user_id=user_id, plan_id=plan_id, reporting_period_days=period,
            metrics=metrics, sessions_completed=sessions_completed,
            homework_completion_rate=homework_rate, engagement_score=engagement_score,
            overall_change=overall, recommendations=recommendations, risk_flags=risk_flags,
        )
        self._stats["reports_generated"] += 1
        logger.info("outcome_report_generated", user_id=str(user_id), overall_change=overall.value)
        return report

    def _generate_recommendations(
        self,
        metrics: list[ProgressMetric],
        homework_rate: float,
        engagement_score: float,
    ) -> list[str]:
        """Generate treatment recommendations."""
        recommendations = []
        for metric in metrics:
            if metric.change_category == ChangeCategory.DETERIORATED:
                recommendations.append(f"Review treatment approach for {metric.measure_type.value} - deterioration detected")
            elif metric.change_category == ChangeCategory.NO_CHANGE:
                recommendations.append(f"Consider treatment modifications for {metric.measure_type.value} - no significant change")
            elif metric.clinically_significant:
                recommendations.append(f"Maintain current approach for {metric.measure_type.value} - clinically significant improvement")
        if homework_rate < 0.5:
            recommendations.append("Address homework completion barriers - below target rate")
        if engagement_score < 0.6:
            recommendations.append("Explore engagement concerns - low engagement score")
        return recommendations

    def _identify_risk_flags(self, metrics: list[ProgressMetric]) -> list[str]:
        """Identify risk flags from metrics."""
        flags = []
        for metric in metrics:
            if metric.change_category == ChangeCategory.DETERIORATED and metric.reliable_change:
                flags.append(f"Reliable deterioration in {metric.measure_type.value}")
            if metric.measure_type == MeasureType.DEPRESSION:
                if metric.current_score >= 20:
                    flags.append("Severe depression score - consider safety assessment")
            if metric.measure_type == MeasureType.THERAPEUTIC_ALLIANCE:
                if metric.current_score < 30:
                    flags.append("Low therapeutic alliance - risk of dropout")
        return flags

    def _determine_overall_change(self, metrics: list[ProgressMetric]) -> ChangeCategory:
        """Determine overall change category."""
        if not metrics:
            return ChangeCategory.NO_CHANGE
        recovered = sum(1 for m in metrics if m.change_category == ChangeCategory.RECOVERED)
        improved = sum(1 for m in metrics if m.change_category == ChangeCategory.IMPROVED)
        deteriorated = sum(1 for m in metrics if m.change_category == ChangeCategory.DETERIORATED)
        if recovered > 0 and deteriorated == 0:
            return ChangeCategory.RECOVERED
        elif improved > deteriorated:
            return ChangeCategory.IMPROVED
        elif deteriorated > improved:
            return ChangeCategory.DETERIORATED
        return ChangeCategory.NO_CHANGE

    def set_baseline(self, user_id: UUID, measure_type: MeasureType, score: float) -> bool:
        """Manually set baseline score."""
        if user_id not in self._metrics:
            self._metrics[user_id] = {}
        if measure_type not in self._metrics[user_id]:
            self._metrics[user_id][measure_type] = ProgressMetric(user_id=user_id, measure_type=measure_type)
        self._metrics[user_id][measure_type].baseline_score = score
        return True

    def set_target(self, user_id: UUID, measure_type: MeasureType, target: float) -> bool:
        """Set target score for metric."""
        metric = self._metrics.get(user_id, {}).get(measure_type)
        if not metric:
            return False
        metric.target_score = target
        return True

    def get_trend(self, user_id: UUID, measure_type: MeasureType, window: int = 5) -> dict[str, Any]:
        """Get score trend for a measure."""
        metric = self._metrics.get(user_id, {}).get(measure_type)
        if not metric or len(metric.scores_history) < 2:
            return {"trend": "insufficient_data", "scores": []}
        recent = metric.scores_history[-window:] if len(metric.scores_history) >= window else metric.scores_history
        scores = [s.raw_score for s in recent]
        if len(scores) >= 2:
            diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > 0.5:
                trend = "increasing"
            elif avg_diff < -0.5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        return {"trend": trend, "scores": scores, "avg_change": avg_diff if len(scores) >= 2 else 0}

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "users_tracked": len(self._scores),
            "total_scores": self._stats["scores_recorded"],
            "alerts_generated": self._stats["alerts_generated"],
            "reports_generated": self._stats["reports_generated"],
        }

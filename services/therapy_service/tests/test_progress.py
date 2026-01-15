"""
Unit tests for Progress Tracker.
Tests outcome measurement, progress monitoring, and clinical change detection.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.domain.progress import (
    ProgressTracker,
    ProgressTrackerSettings,
    MeasureScore,
    ProgressMetric,
    OutcomeReport,
    MeasureType,
    InstrumentType,
    ChangeCategory,
    InstrumentConfig,
)


class TestProgressTrackerSettings:
    """Tests for ProgressTrackerSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = ProgressTrackerSettings()
        assert settings.measurement_frequency_days == 7
        assert settings.enable_automated_alerts is True
        assert settings.deterioration_threshold == -0.2
        assert settings.improvement_threshold == 0.2

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = ProgressTrackerSettings(
            measurement_frequency_days=14,
            enable_automated_alerts=False,
        )
        assert settings.measurement_frequency_days == 14
        assert settings.enable_automated_alerts is False


class TestInstrumentConfig:
    """Tests for instrument configuration."""

    def test_get_phq9_config(self) -> None:
        """Test PHQ-9 configuration."""
        config = InstrumentConfig.get_config(InstrumentType.PHQ9)
        assert config["name"] == "Patient Health Questionnaire-9"
        assert config["max_score"] == 27
        assert config["items"] == 9

    def test_get_gad7_config(self) -> None:
        """Test GAD-7 configuration."""
        config = InstrumentConfig.get_config(InstrumentType.GAD7)
        assert config["name"] == "Generalized Anxiety Disorder-7"
        assert config["max_score"] == 21

    def test_severity_label_phq9(self) -> None:
        """Test PHQ-9 severity labeling."""
        assert InstrumentConfig.get_severity_label(InstrumentType.PHQ9, 3) == "minimal"
        assert InstrumentConfig.get_severity_label(InstrumentType.PHQ9, 8) == "mild"
        assert InstrumentConfig.get_severity_label(InstrumentType.PHQ9, 12) == "moderate"
        assert InstrumentConfig.get_severity_label(InstrumentType.PHQ9, 18) == "moderately_severe"
        assert InstrumentConfig.get_severity_label(InstrumentType.PHQ9, 25) == "severe"

    def test_severity_label_gad7(self) -> None:
        """Test GAD-7 severity labeling."""
        assert InstrumentConfig.get_severity_label(InstrumentType.GAD7, 4) == "minimal"
        assert InstrumentConfig.get_severity_label(InstrumentType.GAD7, 12) == "moderate"


class TestProgressTracker:
    """Tests for ProgressTracker functionality."""

    def test_tracker_initialization(self) -> None:
        """Test tracker initializes correctly."""
        tracker = ProgressTracker()
        stats = tracker.get_statistics()
        assert stats["users_tracked"] == 0
        assert stats["total_scores"] == 0

    def test_record_score_phq9(self) -> None:
        """Test recording PHQ-9 score."""
        tracker = ProgressTracker()
        user_id = uuid4()
        score = tracker.record_score(
            user_id=user_id,
            instrument=InstrumentType.PHQ9,
            raw_score=15,
        )
        assert score.user_id == user_id
        assert score.raw_score == 15
        assert score.severity_label == "moderate"
        assert score.measure_type == MeasureType.DEPRESSION

    def test_record_score_gad7(self) -> None:
        """Test recording GAD-7 score."""
        tracker = ProgressTracker()
        score = tracker.record_score(
            user_id=uuid4(),
            instrument=InstrumentType.GAD7,
            raw_score=10,
        )
        assert score.raw_score == 10
        assert score.measure_type == MeasureType.ANXIETY

    def test_score_normalization(self) -> None:
        """Test score normalization."""
        tracker = ProgressTracker()
        score = tracker.record_score(
            user_id=uuid4(),
            instrument=InstrumentType.PHQ9,
            raw_score=13.5,
        )
        # 13.5 / 27 = 0.5
        assert abs(score.normalized_score - 0.5) < 0.01

    def test_score_clamping(self) -> None:
        """Test scores are clamped to valid range."""
        tracker = ProgressTracker()
        score = tracker.record_score(
            user_id=uuid4(),
            instrument=InstrumentType.PHQ9,
            raw_score=100,  # Over max
        )
        assert score.raw_score == 27  # Clamped to max

    def test_get_user_scores(self) -> None:
        """Test getting scores for a user."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.GAD7, raw_score=10)
        scores = tracker.get_user_scores(user_id)
        assert len(scores) == 2

    def test_get_user_scores_by_instrument(self) -> None:
        """Test filtering scores by instrument."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.GAD7, raw_score=10)
        scores = tracker.get_user_scores(user_id, instrument=InstrumentType.PHQ9)
        assert len(scores) == 1
        assert scores[0].instrument == InstrumentType.PHQ9

    def test_progress_metric_creation(self) -> None:
        """Test progress metric is created on first score."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric is not None
        assert metric.baseline_score == 15
        assert metric.current_score == 15

    def test_progress_metric_update(self) -> None:
        """Test progress metric updates with new scores."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.baseline_score == 15
        assert metric.current_score == 10

    def test_change_detection_improved(self) -> None:
        """Test improvement detection."""
        tracker = ProgressTracker(ProgressTrackerSettings(improvement_threshold=0.2))
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=20)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.change_category == ChangeCategory.IMPROVED

    def test_change_detection_deteriorated(self) -> None:
        """Test deterioration detection."""
        tracker = ProgressTracker(ProgressTrackerSettings(deterioration_threshold=-0.2))
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=20)
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.change_category == ChangeCategory.DETERIORATED

    def test_check_deterioration(self) -> None:
        """Test deterioration checking."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=20)
        alerts = tracker.check_deterioration(user_id)
        assert len(alerts) > 0
        assert alerts[0]["type"] == "deterioration"

    def test_check_measurement_due(self) -> None:
        """Test measurement due checking."""
        tracker = ProgressTracker(ProgressTrackerSettings(measurement_frequency_days=7))
        user_id = uuid4()
        # No scores yet
        assert tracker.check_measurement_due(user_id, InstrumentType.PHQ9) is True
        # Record score
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)
        # Should not be due immediately
        assert tracker.check_measurement_due(user_id, InstrumentType.PHQ9) is False

    def test_generate_outcome_report(self) -> None:
        """Test outcome report generation."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=20)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.GAD7, raw_score=12)
        report = tracker.generate_outcome_report(
            user_id=user_id,
            sessions_completed=5,
            homework_rate=0.8,
            engagement_score=0.75,
        )
        assert report.user_id == user_id
        assert report.sessions_completed == 5
        assert len(report.metrics) > 0
        assert len(report.recommendations) > 0

    def test_set_baseline(self) -> None:
        """Test manually setting baseline."""
        tracker = ProgressTracker()
        user_id = uuid4()
        result = tracker.set_baseline(user_id, MeasureType.DEPRESSION, 18.0)
        assert result is True
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.baseline_score == 18.0

    def test_set_target(self) -> None:
        """Test setting target score."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        result = tracker.set_target(user_id, MeasureType.DEPRESSION, 5.0)
        assert result is True
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.target_score == 5.0

    def test_get_trend(self) -> None:
        """Test getting score trend."""
        tracker = ProgressTracker()
        user_id = uuid4()
        # Record decreasing scores (improvement for PHQ-9)
        for score in [20, 18, 15, 12, 10]:
            tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=score)
        trend = tracker.get_trend(user_id, MeasureType.DEPRESSION)
        assert trend["trend"] == "decreasing"

    def test_get_all_metrics(self) -> None:
        """Test getting all metrics for a user."""
        tracker = ProgressTracker()
        user_id = uuid4()
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=15)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.GAD7, raw_score=12)
        metrics = tracker.get_all_metrics(user_id)
        assert len(metrics) == 2

    def test_reliable_change_calculation(self) -> None:
        """Test reliable change index calculation."""
        tracker = ProgressTracker(ProgressTrackerSettings(enable_reliable_change=True))
        user_id = uuid4()
        # PHQ-9 RCI is 6.0
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=20)
        tracker.record_score(user_id=user_id, instrument=InstrumentType.PHQ9, raw_score=10)  # Change of 10 > RCI
        metric = tracker.get_progress_metric(user_id, MeasureType.DEPRESSION)
        assert metric.reliable_change is True

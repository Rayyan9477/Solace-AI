"""
Unit tests for analytics aggregations module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from aggregations import (
    AggregationWindow,
    MetricType,
    TimeWindow,
    MetricValue,
    AggregatedMetric,
    MetricBucket,
    MetricsStore,
    AnalyticsAggregator,
    CountAggregator,
    SumAggregator,
    AverageAggregator,
    PercentileAggregator,
)


class TestAggregationWindow:
    """Tests for AggregationWindow enum."""

    def test_window_values(self):
        """Test window enum values."""
        assert AggregationWindow.MINUTE.value == "minute"
        assert AggregationWindow.HOUR.value == "hour"
        assert AggregationWindow.DAY.value == "day"


class TestTimeWindow:
    """Tests for TimeWindow dataclass."""

    def test_for_minute(self):
        """Test creating minute window."""
        dt = datetime(2026, 1, 20, 10, 30, 45, tzinfo=timezone.utc)
        window = TimeWindow.for_minute(dt)

        assert window.start == datetime(2026, 1, 20, 10, 30, 0, tzinfo=timezone.utc)
        assert window.end == datetime(2026, 1, 20, 10, 31, 0, tzinfo=timezone.utc)
        assert window.window_type == AggregationWindow.MINUTE

    def test_for_hour(self):
        """Test creating hour window."""
        dt = datetime(2026, 1, 20, 10, 30, 45, tzinfo=timezone.utc)
        window = TimeWindow.for_hour(dt)

        assert window.start == datetime(2026, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        assert window.end == datetime(2026, 1, 20, 11, 0, 0, tzinfo=timezone.utc)
        assert window.window_type == AggregationWindow.HOUR

    def test_for_day(self):
        """Test creating day window."""
        dt = datetime(2026, 1, 20, 10, 30, 45, tzinfo=timezone.utc)
        window = TimeWindow.for_day(dt)

        assert window.start == datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc)
        assert window.end == datetime(2026, 1, 21, 0, 0, 0, tzinfo=timezone.utc)
        assert window.window_type == AggregationWindow.DAY

    def test_contains(self):
        """Test window contains check."""
        dt = datetime(2026, 1, 20, 10, 30, 0, tzinfo=timezone.utc)
        window = TimeWindow.for_hour(dt)

        assert window.contains(datetime(2026, 1, 20, 10, 15, 0, tzinfo=timezone.utc))
        assert window.contains(datetime(2026, 1, 20, 10, 59, 59, tzinfo=timezone.utc))
        assert not window.contains(datetime(2026, 1, 20, 11, 0, 0, tzinfo=timezone.utc))

    def test_to_key(self):
        """Test window key generation."""
        dt = datetime(2026, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow.for_hour(dt)

        key = window.to_key()
        assert "hour:" in key
        assert "2026-01-20" in key


class TestMetricBucket:
    """Tests for MetricBucket class."""

    @pytest.fixture
    def bucket(self):
        """Create a metric bucket."""
        window = TimeWindow.for_hour(datetime.now(timezone.utc))
        return MetricBucket(metric_name="test_metric", window=window)

    @pytest.mark.asyncio
    async def test_add_single_value(self, bucket):
        """Test adding a single value."""
        await bucket.add(Decimal("10"))

        assert bucket.count == 1
        assert bucket.sum_value == Decimal("10")
        assert bucket.min_value == Decimal("10")
        assert bucket.max_value == Decimal("10")

    @pytest.mark.asyncio
    async def test_add_multiple_values(self, bucket):
        """Test adding multiple values."""
        await bucket.add(Decimal("10"))
        await bucket.add(Decimal("20"))
        await bucket.add(Decimal("30"))

        assert bucket.count == 3
        assert bucket.sum_value == Decimal("60")
        assert bucket.min_value == Decimal("10")
        assert bucket.max_value == Decimal("30")

    @pytest.mark.asyncio
    async def test_to_aggregated(self, bucket):
        """Test converting to aggregated metric."""
        await bucket.add(Decimal("10"))
        await bucket.add(Decimal("20"))

        aggregated = bucket.to_aggregated()

        assert aggregated.metric_name == "test_metric"
        assert aggregated.count == 2
        assert aggregated.sum_value == Decimal("30")
        assert aggregated.avg_value == Decimal("15")


class TestMetricsStore:
    """Tests for MetricsStore class."""

    @pytest.mark.asyncio
    async def test_record_metric(self, metrics_store):
        """Test recording a metric."""
        await metrics_store.record("test.metric", Decimal("100"))

        stats = await metrics_store.get_statistics()
        assert stats["total_records"] == 1

    @pytest.mark.asyncio
    async def test_record_counter(self, metrics_store):
        """Test recording counter metrics."""
        await metrics_store.record_counter("test.counter")
        await metrics_store.record_counter("test.counter")
        await metrics_store.record_counter("test.counter")

        stats = await metrics_store.get_statistics()
        assert stats["total_records"] == 3

    @pytest.mark.asyncio
    async def test_record_gauge(self, metrics_store):
        """Test recording gauge metrics."""
        await metrics_store.record_gauge("test.gauge", Decimal("50.5"))

        metrics = await metrics_store.get_aggregated("test.gauge", window_type=AggregationWindow.MINUTE)
        assert len(metrics) == 1
        assert metrics[0].sum_value == Decimal("50.5")

    @pytest.mark.asyncio
    async def test_record_timing(self, metrics_store):
        """Test recording timing metrics."""
        await metrics_store.record_timing("test.timing", 150)

        metrics = await metrics_store.get_aggregated("test.timing", window_type=AggregationWindow.MINUTE)
        assert len(metrics) == 1
        assert metrics[0].sum_value == Decimal("150")

    @pytest.mark.asyncio
    async def test_record_with_labels(self, metrics_store):
        """Test recording metrics with labels."""
        labels = {"service": "test", "environment": "dev"}
        await metrics_store.record("test.labeled", Decimal("100"), labels=labels)

        metrics = await metrics_store.get_aggregated("test.labeled", window_type=AggregationWindow.MINUTE, labels=labels)
        assert len(metrics) == 1
        assert metrics[0].labels == labels

    @pytest.mark.asyncio
    async def test_get_aggregated_empty(self, metrics_store):
        """Test getting aggregated metrics when none exist."""
        metrics = await metrics_store.get_aggregated("nonexistent.metric")
        assert metrics == []

    @pytest.mark.asyncio
    async def test_get_metric_names(self, metrics_store):
        """Test getting all metric names."""
        await metrics_store.record("metric.one", Decimal("1"))
        await metrics_store.record("metric.two", Decimal("2"))

        names = await metrics_store.get_metric_names()
        assert "metric.one" in names
        assert "metric.two" in names


class TestAnalyticsAggregator:
    """Tests for AnalyticsAggregator class."""

    @pytest.mark.asyncio
    async def test_track_session_event(self, analytics_aggregator, sample_user_id):
        """Test tracking session events."""
        await analytics_aggregator.track_session_event(
            event_type="session.started",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={"duration_seconds": 300},
        )

        summary = await analytics_aggregator.get_event_summary()
        assert summary.get("session.started") == 1

    @pytest.mark.asyncio
    async def test_track_safety_event(self, analytics_aggregator):
        """Test tracking safety events."""
        await analytics_aggregator.track_safety_event(
            risk_level="HIGH",
            detection_layer=2,
            metadata={"recommended_action": "escalate"},
        )

        metrics = await analytics_aggregator.store.get_aggregated(
            "safety.crisis_events", window_type=AggregationWindow.MINUTE
        )
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_track_therapy_event(self, analytics_aggregator):
        """Test tracking therapy events."""
        await analytics_aggregator.track_therapy_event(
            modality="CBT",
            technique="cognitive_restructuring",
            engagement_score=Decimal("0.85"),
        )

        metrics = await analytics_aggregator.store.get_aggregated(
            "therapy.interventions", window_type=AggregationWindow.MINUTE
        )
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_track_diagnosis_event(self, analytics_aggregator):
        """Test tracking diagnosis events."""
        await analytics_aggregator.track_diagnosis_event(
            assessment_type="diagnosis.completed",
            severity="MILD",
            stepped_care_level=2,
        )

        metrics = await analytics_aggregator.store.get_aggregated(
            "diagnosis.assessments", window_type=AggregationWindow.MINUTE
        )
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, analytics_aggregator, sample_user_id):
        """Test getting dashboard metrics."""
        await analytics_aggregator.track_session_event(
            event_type="session.started",
            user_id=sample_user_id,
            session_id=uuid4(),
            metadata={},
        )

        dashboard = await analytics_aggregator.get_dashboard_metrics()

        assert "sessions_last_hour" in dashboard
        assert "safety_checks_last_hour" in dashboard
        assert "crisis_events_last_24h" in dashboard
        assert "generated_at" in dashboard


class TestAggregators:
    """Tests for aggregator strategy classes."""

    @pytest.mark.asyncio
    async def test_count_aggregator(self):
        """Test count aggregation."""
        aggregator = CountAggregator()
        values = ["a", "b", "c", "d", "e"]

        result = await aggregator.aggregate(values)
        assert result == Decimal("5")

    @pytest.mark.asyncio
    async def test_sum_aggregator(self):
        """Test sum aggregation."""
        aggregator = SumAggregator()
        values = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = await aggregator.aggregate(values)
        assert result == Decimal("60")

    @pytest.mark.asyncio
    async def test_average_aggregator(self):
        """Test average aggregation."""
        aggregator = AverageAggregator()
        values = [Decimal("10"), Decimal("20"), Decimal("30")]

        result = await aggregator.aggregate(values)
        assert result == Decimal("20")

    @pytest.mark.asyncio
    async def test_average_aggregator_empty(self):
        """Test average aggregation with empty list."""
        aggregator = AverageAggregator()

        result = await aggregator.aggregate([])
        assert result == Decimal("0")

    @pytest.mark.asyncio
    async def test_percentile_aggregator(self):
        """Test percentile aggregation."""
        aggregator = PercentileAggregator(percentile=50)
        values = [Decimal(str(i)) for i in range(1, 101)]

        result = await aggregator.aggregate(values)
        # Percentile index is: int(100 * 50 / 100) = 50, so values[50] = 51
        assert result == Decimal("51")


class TestAggregatedMetric:
    """Tests for AggregatedMetric model."""

    def test_computed_avg(self):
        """Test computed average property."""
        metric = AggregatedMetric(
            metric_name="test",
            window=AggregationWindow.HOUR,
            window_start=datetime.now(timezone.utc),
            window_end=datetime.now(timezone.utc) + timedelta(hours=1),
            count=4,
            sum_value=Decimal("100"),
        )

        assert metric.computed_avg == Decimal("25")

    def test_computed_avg_zero_count(self):
        """Test computed average with zero count."""
        metric = AggregatedMetric(
            metric_name="test",
            window=AggregationWindow.HOUR,
            window_start=datetime.now(timezone.utc),
            window_end=datetime.now(timezone.utc) + timedelta(hours=1),
            count=0,
            sum_value=Decimal("0"),
        )

        assert metric.computed_avg is None

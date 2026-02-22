"""
Solace-AI Analytics Service - Metrics Aggregation.

Time-series aggregation engine for real-time and historical analytics.
Supports multiple aggregation windows and metric types.

Architecture Layer: Domain
Principles: Strategy Pattern, Immutable Data, Thread-Safe Aggregations
"""
from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class AggregationWindow(str, Enum):
    """Time windows for metric aggregation."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class MetricType(str, Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass(frozen=True)
class TimeWindow:
    """Immutable time window for aggregation."""
    start: datetime
    end: datetime
    window_type: AggregationWindow

    @classmethod
    def for_minute(cls, dt: datetime) -> TimeWindow:
        """Create minute window containing given datetime."""
        start = dt.replace(second=0, microsecond=0)
        return cls(start=start, end=start + timedelta(minutes=1), window_type=AggregationWindow.MINUTE)

    @classmethod
    def for_hour(cls, dt: datetime) -> TimeWindow:
        """Create hour window containing given datetime."""
        start = dt.replace(minute=0, second=0, microsecond=0)
        return cls(start=start, end=start + timedelta(hours=1), window_type=AggregationWindow.HOUR)

    @classmethod
    def for_day(cls, dt: datetime) -> TimeWindow:
        """Create day window containing given datetime."""
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return cls(start=start, end=start + timedelta(days=1), window_type=AggregationWindow.DAY)

    def contains(self, dt: datetime) -> bool:
        """Check if datetime falls within this window."""
        return self.start <= dt < self.end

    def to_key(self) -> str:
        """Generate unique key for this window."""
        return f"{self.window_type.value}:{self.start.isoformat()}"


class MetricValue(BaseModel):
    """Value container for a single metric data point."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    value: Decimal
    labels: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class AggregatedMetric(BaseModel):
    """Aggregated metric over a time window."""
    metric_name: str
    window: AggregationWindow
    window_start: datetime
    window_end: datetime
    count: int = Field(ge=0)
    sum_value: Decimal = Field(default=Decimal("0"))
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    avg_value: Decimal | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)

    @property
    def computed_avg(self) -> Decimal | None:
        """Compute average from sum and count."""
        if self.count == 0:
            return None
        return self.sum_value / Decimal(self.count)


@dataclass
class MetricBucket:
    """Mutable bucket for collecting metrics within a window."""
    metric_name: str
    window: TimeWindow
    labels: dict[str, str] = field(default_factory=dict)
    count: int = 0
    sum_value: Decimal = field(default_factory=lambda: Decimal("0"))
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    values: list[Decimal] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def add(self, value: Decimal) -> None:
        """Add a value to the bucket."""
        async with self._lock:
            self.count += 1
            self.sum_value += value
            self.min_value = value if self.min_value is None else min(self.min_value, value)
            self.max_value = value if self.max_value is None else max(self.max_value, value)
            self.values.append(value)

    def to_aggregated(self) -> AggregatedMetric:
        """Convert bucket to immutable aggregated metric."""
        return AggregatedMetric(
            metric_name=self.metric_name,
            window=self.window.window_type,
            window_start=self.window.start,
            window_end=self.window.end,
            count=self.count,
            sum_value=self.sum_value,
            min_value=self.min_value,
            max_value=self.max_value,
            avg_value=self.sum_value / Decimal(self.count) if self.count > 0 else None,
            labels=self.labels,
        )


class MetricAggregator(ABC, Generic[T]):
    """Abstract base for metric aggregation strategies."""

    @abstractmethod
    async def aggregate(self, values: list[T]) -> Decimal:
        """Aggregate values into a single metric."""


class CountAggregator(MetricAggregator[Any]):
    """Count the number of values."""

    async def aggregate(self, values: list[Any]) -> Decimal:
        return Decimal(len(values))


class SumAggregator(MetricAggregator[Decimal]):
    """Sum all values."""

    async def aggregate(self, values: list[Decimal]) -> Decimal:
        return sum(values, Decimal("0"))


class AverageAggregator(MetricAggregator[Decimal]):
    """Calculate average of values."""

    async def aggregate(self, values: list[Decimal]) -> Decimal:
        if not values:
            return Decimal("0")
        return sum(values, Decimal("0")) / Decimal(len(values))


class PercentileAggregator(MetricAggregator[Decimal]):
    """Calculate percentile of values."""

    def __init__(self, percentile: int = 95) -> None:
        self.percentile = percentile

    async def aggregate(self, values: list[Decimal]) -> Decimal:
        if not values:
            return Decimal("0")
        sorted_values = sorted(values)
        # Nearest-rank percentile: ceil(P/100 * N) clamped to valid range
        index = min(math.ceil(self.percentile / 100 * len(sorted_values)), len(sorted_values)) - 1
        return sorted_values[max(index, 0)]


class MetricsStore:
    """In-memory metrics store with time-based windowing."""

    def __init__(self, max_windows_per_metric: int = 1000) -> None:
        self._buckets: dict[str, dict[str, MetricBucket]] = defaultdict(dict)
        self._max_windows = max_windows_per_metric
        self._lock = asyncio.Lock()
        self._stats = {"total_records": 0, "metrics_tracked": 0}
        logger.info("metrics_store_initialized", max_windows=max_windows_per_metric)

    async def record(
        self,
        metric_name: str,
        value: Decimal,
        timestamp: datetime | None = None,
        labels: dict[str, str] | None = None,
        window_type: AggregationWindow = AggregationWindow.MINUTE,
    ) -> None:
        """Record a metric value."""
        ts = timestamp or datetime.now(timezone.utc)
        labels = labels or {}

        window = self._create_window(ts, window_type)
        bucket_key = f"{metric_name}:{window.to_key()}:{self._labels_key(labels)}"

        async with self._lock:
            if metric_name not in self._buckets:
                self._buckets[metric_name] = {}
                self._stats["metrics_tracked"] += 1

            if bucket_key not in self._buckets[metric_name]:
                self._buckets[metric_name][bucket_key] = MetricBucket(
                    metric_name=metric_name, window=window, labels=labels
                )
                await self._prune_old_windows(metric_name)

        bucket = self._buckets[metric_name][bucket_key]
        await bucket.add(value)
        self._stats["total_records"] += 1

    async def record_counter(self, metric_name: str, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        await self.record(metric_name, Decimal("1"), labels=labels)

    async def record_gauge(
        self, metric_name: str, value: Decimal, labels: dict[str, str] | None = None
    ) -> None:
        """Record a gauge metric value."""
        await self.record(metric_name, value, labels=labels)

    async def record_timing(
        self, metric_name: str, duration_ms: int, labels: dict[str, str] | None = None
    ) -> None:
        """Record a timing metric in milliseconds."""
        await self.record(metric_name, Decimal(duration_ms), labels=labels)

    async def get_aggregated(
        self,
        metric_name: str,
        window_type: AggregationWindow = AggregationWindow.HOUR,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[AggregatedMetric]:
        """Get aggregated metrics for a time range."""
        if metric_name not in self._buckets:
            return []

        results: list[AggregatedMetric] = []
        end_time = end_time or datetime.now(timezone.utc)
        start_time = start_time or (end_time - timedelta(hours=24))

        for bucket in self._buckets[metric_name].values():
            if bucket.window.window_type != window_type:
                continue
            if bucket.window.end < start_time or bucket.window.start > end_time:
                continue
            if labels and not self._labels_match(bucket.labels, labels):
                continue
            results.append(bucket.to_aggregated())

        return sorted(results, key=lambda m: m.window_start)

    async def get_latest(
        self, metric_name: str, labels: dict[str, str] | None = None
    ) -> AggregatedMetric | None:
        """Get the most recent aggregated metric."""
        metrics = await self.get_aggregated(metric_name, labels=labels)
        return metrics[-1] if metrics else None

    async def get_metric_names(self) -> list[str]:
        """Get all tracked metric names."""
        return list(self._buckets.keys())

    async def get_statistics(self) -> dict[str, Any]:
        """Get store statistics."""
        return {
            **self._stats,
            "buckets_by_metric": {k: len(v) for k, v in self._buckets.items()},
        }

    def _create_window(self, dt: datetime, window_type: AggregationWindow) -> TimeWindow:
        """Create appropriate time window for datetime."""
        if window_type == AggregationWindow.MINUTE:
            return TimeWindow.for_minute(dt)
        elif window_type == AggregationWindow.HOUR:
            return TimeWindow.for_hour(dt)
        else:
            return TimeWindow.for_day(dt)

    def _labels_key(self, labels: dict[str, str]) -> str:
        """Generate key from labels dict."""
        if not labels:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _labels_match(self, bucket_labels: dict[str, str], filter_labels: dict[str, str]) -> bool:
        """Check if bucket labels match filter."""
        return all(bucket_labels.get(k) == v for k, v in filter_labels.items())

    async def _prune_old_windows(self, metric_name: str) -> None:
        """Remove oldest windows if over limit."""
        if len(self._buckets[metric_name]) > self._max_windows:
            sorted_keys = sorted(
                self._buckets[metric_name].keys(),
                key=lambda k: self._buckets[metric_name][k].window.start,
            )
            for key in sorted_keys[: len(sorted_keys) - self._max_windows]:
                del self._buckets[metric_name][key]


class AnalyticsAggregator:
    """High-level analytics aggregation service."""

    def __init__(self, store: MetricsStore | None = None) -> None:
        self._store = store or MetricsStore()
        self._event_counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        logger.info("analytics_aggregator_initialized")

    @property
    def store(self) -> MetricsStore:
        """Get underlying metrics store."""
        return self._store

    async def track_session_event(
        self, event_type: str, user_id: UUID, session_id: UUID | None, metadata: dict[str, Any]
    ) -> None:
        """Track a session-related event."""
        labels = {"event_type": event_type}
        await self._store.record_counter(f"events.{event_type}", labels=labels)

        async with self._lock:
            self._event_counts[event_type] += 1

        if "duration_seconds" in metadata:
            await self._store.record_timing(
                "session.duration_ms",
                int(metadata["duration_seconds"] * 1000),
                labels={"event_type": event_type},
            )

        if "generation_time_ms" in metadata:
            await self._store.record_timing(
                "response.generation_time_ms",
                metadata["generation_time_ms"],
                labels={"model": metadata.get("model_used", "unknown")},
            )

        logger.debug("session_event_tracked", event_type=event_type, user_id=str(user_id))

    async def track_safety_event(
        self, risk_level: str, detection_layer: int, metadata: dict[str, Any]
    ) -> None:
        """Track a safety-related event."""
        labels = {"risk_level": risk_level, "detection_layer": str(detection_layer)}
        await self._store.record_counter("safety.assessments", labels=labels)

        if risk_level in ("HIGH", "CRITICAL"):
            await self._store.record_counter("safety.crisis_events", labels=labels)

        logger.info("safety_event_tracked", risk_level=risk_level, detection_layer=detection_layer)

    async def track_therapy_event(
        self, modality: str, technique: str, engagement_score: Decimal | None
    ) -> None:
        """Track a therapy-related event."""
        labels = {"modality": modality, "technique": technique}
        await self._store.record_counter("therapy.interventions", labels=labels)

        if engagement_score is not None:
            await self._store.record_gauge("therapy.engagement_score", engagement_score, labels=labels)

    async def track_diagnosis_event(
        self, assessment_type: str, severity: str, stepped_care_level: int
    ) -> None:
        """Track a diagnosis-related event."""
        labels = {"assessment_type": assessment_type, "severity": severity}
        await self._store.record_counter("diagnosis.assessments", labels=labels)
        await self._store.record_gauge(
            "diagnosis.stepped_care_level",
            Decimal(stepped_care_level),
            labels={"assessment_type": assessment_type},
        )

    async def get_event_summary(self) -> dict[str, int]:
        """Get summary of event counts."""
        async with self._lock:
            return dict(self._event_counts)

    async def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get metrics for analytics dashboard."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        session_metrics = await self._store.get_aggregated(
            "events.session.started", start_time=hour_ago
        )
        safety_metrics = await self._store.get_aggregated(
            "safety.assessments", start_time=hour_ago
        )
        crisis_metrics = await self._store.get_aggregated(
            "safety.crisis_events", start_time=day_ago
        )

        return {
            "sessions_last_hour": sum(m.count for m in session_metrics),
            "safety_checks_last_hour": sum(m.count for m in safety_metrics),
            "crisis_events_last_24h": sum(m.count for m in crisis_metrics),
            "event_totals": await self.get_event_summary(),
            "generated_at": now.isoformat(),
        }

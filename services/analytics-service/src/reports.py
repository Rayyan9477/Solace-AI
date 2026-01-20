"""
Solace-AI Analytics Service - Report Generation.

Generates clinical, operational, and compliance reports from aggregated analytics.
Supports multiple output formats and scheduled report generation.

Architecture Layer: Domain
Principles: Strategy Pattern, Template Method, Immutable Reports
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict
import structlog

try:
    from .aggregations import AnalyticsAggregator, AggregatedMetric, AggregationWindow
except ImportError:
    from aggregations import AnalyticsAggregator, AggregatedMetric, AggregationWindow

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ReportType(str, Enum):
    """Types of analytics reports."""
    SESSION_SUMMARY = "session_summary"
    SAFETY_OVERVIEW = "safety_overview"
    CLINICAL_OUTCOMES = "clinical_outcomes"
    ENGAGEMENT_METRICS = "engagement_metrics"
    OPERATIONAL_HEALTH = "operational_health"
    COMPLIANCE_AUDIT = "compliance_audit"


class ReportFormat(str, Enum):
    """Output formats for reports."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportPeriod(str, Enum):
    """Standard reporting periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportStatus(str, Enum):
    """Status of report generation."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ReportTimeRange:
    """Immutable time range for report data."""
    start: datetime
    end: datetime
    period: ReportPeriod

    @classmethod
    def for_last_hour(cls) -> ReportTimeRange:
        """Create time range for last hour."""
        now = datetime.now(timezone.utc)
        return cls(start=now - timedelta(hours=1), end=now, period=ReportPeriod.HOURLY)

    @classmethod
    def for_last_day(cls) -> ReportTimeRange:
        """Create time range for last 24 hours."""
        now = datetime.now(timezone.utc)
        return cls(start=now - timedelta(days=1), end=now, period=ReportPeriod.DAILY)

    @classmethod
    def for_last_week(cls) -> ReportTimeRange:
        """Create time range for last 7 days."""
        now = datetime.now(timezone.utc)
        return cls(start=now - timedelta(weeks=1), end=now, period=ReportPeriod.WEEKLY)

    @classmethod
    def for_last_month(cls) -> ReportTimeRange:
        """Create time range for last 30 days."""
        now = datetime.now(timezone.utc)
        return cls(start=now - timedelta(days=30), end=now, period=ReportPeriod.MONTHLY)

    @classmethod
    def custom(cls, start: datetime, end: datetime) -> ReportTimeRange:
        """Create custom time range."""
        return cls(start=start, end=end, period=ReportPeriod.CUSTOM)

    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return (self.end - self.start).total_seconds() / 3600


class ReportSection(BaseModel):
    """Individual section within a report."""
    title: str
    description: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True)


class Report(BaseModel):
    """Complete analytics report."""
    report_id: UUID = Field(default_factory=uuid4)
    report_type: ReportType
    title: str
    description: str = ""
    time_range_start: datetime
    time_range_end: datetime
    period: ReportPeriod
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ReportStatus = ReportStatus.COMPLETED
    sections: list[ReportSection] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class ReportGenerator(ABC):
    """Abstract base for report generators."""

    @property
    @abstractmethod
    def report_type(self) -> ReportType:
        """Return the type of report this generator creates."""

    @abstractmethod
    async def generate(
        self, aggregator: AnalyticsAggregator, time_range: ReportTimeRange
    ) -> Report:
        """Generate report from aggregated data."""


class SessionSummaryReportGenerator(ReportGenerator):
    """Generate session summary reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.SESSION_SUMMARY

    async def generate(
        self, aggregator: AnalyticsAggregator, time_range: ReportTimeRange
    ) -> Report:
        """Generate session summary report."""
        store = aggregator.store

        session_starts = await store.get_aggregated(
            "events.session.started",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )
        session_ends = await store.get_aggregated(
            "events.session.ended",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )
        durations = await store.get_aggregated(
            "session.duration_ms",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )

        total_sessions = sum(m.count for m in session_starts)
        completed_sessions = sum(m.count for m in session_ends)
        avg_duration_ms = (
            sum(m.sum_value for m in durations) / Decimal(max(1, sum(m.count for m in durations)))
            if durations else Decimal("0")
        )

        overview_section = ReportSection(
            title="Session Overview",
            description="Summary of session activity during the reporting period",
            data={
                "total_sessions_started": total_sessions,
                "total_sessions_completed": completed_sessions,
                "completion_rate": f"{(completed_sessions / max(1, total_sessions)) * 100:.1f}%",
                "average_duration_minutes": float(avg_duration_ms / Decimal("60000")),
            },
        )

        hourly_data = [
            {
                "hour": m.window_start.strftime("%Y-%m-%d %H:00"),
                "sessions": m.count,
            }
            for m in session_starts
        ]

        trends_section = ReportSection(
            title="Session Trends",
            description="Hourly session distribution",
            tables=[{"name": "hourly_sessions", "data": hourly_data}],
        )

        return Report(
            report_type=self.report_type,
            title="Session Summary Report",
            description=f"Session activity from {time_range.start.date()} to {time_range.end.date()}",
            time_range_start=time_range.start,
            time_range_end=time_range.end,
            period=time_range.period,
            sections=[overview_section, trends_section],
            summary={
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "avg_duration_minutes": float(avg_duration_ms / Decimal("60000")),
            },
        )


class SafetyOverviewReportGenerator(ReportGenerator):
    """Generate safety overview reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.SAFETY_OVERVIEW

    async def generate(
        self, aggregator: AnalyticsAggregator, time_range: ReportTimeRange
    ) -> Report:
        """Generate safety overview report."""
        store = aggregator.store

        safety_checks = await store.get_aggregated(
            "safety.assessments",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )
        crisis_events = await store.get_aggregated(
            "safety.crisis_events",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )

        total_checks = sum(m.count for m in safety_checks)
        total_crises = sum(m.count for m in crisis_events)

        overview_section = ReportSection(
            title="Safety Overview",
            description="Summary of safety assessments and crisis events",
            data={
                "total_safety_checks": total_checks,
                "total_crisis_events": total_crises,
                "crisis_rate": f"{(total_crises / max(1, total_checks)) * 100:.2f}%",
            },
        )

        risk_distribution: dict[str, int] = {}
        for metric in safety_checks:
            level = metric.labels.get("risk_level", "unknown")
            risk_distribution[level] = risk_distribution.get(level, 0) + metric.count

        distribution_section = ReportSection(
            title="Risk Level Distribution",
            description="Distribution of assessed risk levels",
            data=risk_distribution,
        )

        return Report(
            report_type=self.report_type,
            title="Safety Overview Report",
            description=f"Safety metrics from {time_range.start.date()} to {time_range.end.date()}",
            time_range_start=time_range.start,
            time_range_end=time_range.end,
            period=time_range.period,
            sections=[overview_section, distribution_section],
            summary={
                "total_checks": total_checks,
                "total_crises": total_crises,
                "risk_distribution": risk_distribution,
            },
        )


class ClinicalOutcomesReportGenerator(ReportGenerator):
    """Generate clinical outcomes reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.CLINICAL_OUTCOMES

    async def generate(
        self, aggregator: AnalyticsAggregator, time_range: ReportTimeRange
    ) -> Report:
        """Generate clinical outcomes report."""
        store = aggregator.store

        diagnosis_metrics = await store.get_aggregated(
            "diagnosis.assessments",
            window_type=AggregationWindow.DAY,
            start_time=time_range.start,
            end_time=time_range.end,
        )
        therapy_metrics = await store.get_aggregated(
            "therapy.interventions",
            window_type=AggregationWindow.DAY,
            start_time=time_range.start,
            end_time=time_range.end,
        )
        engagement_metrics = await store.get_aggregated(
            "therapy.engagement_score",
            window_type=AggregationWindow.DAY,
            start_time=time_range.start,
            end_time=time_range.end,
        )

        total_diagnoses = sum(m.count for m in diagnosis_metrics)
        total_interventions = sum(m.count for m in therapy_metrics)
        avg_engagement = (
            sum(m.sum_value for m in engagement_metrics) / Decimal(max(1, sum(m.count for m in engagement_metrics)))
            if engagement_metrics else Decimal("0")
        )

        overview_section = ReportSection(
            title="Clinical Activity Overview",
            description="Summary of diagnostic and therapeutic activities",
            data={
                "total_diagnostic_assessments": total_diagnoses,
                "total_therapeutic_interventions": total_interventions,
                "average_engagement_score": float(avg_engagement),
            },
        )

        modality_usage: dict[str, int] = {}
        for metric in therapy_metrics:
            modality = metric.labels.get("modality", "unknown")
            modality_usage[modality] = modality_usage.get(modality, 0) + metric.count

        modality_section = ReportSection(
            title="Therapy Modality Usage",
            description="Distribution of therapeutic modalities used",
            data=modality_usage,
        )

        return Report(
            report_type=self.report_type,
            title="Clinical Outcomes Report",
            description=f"Clinical activity from {time_range.start.date()} to {time_range.end.date()}",
            time_range_start=time_range.start,
            time_range_end=time_range.end,
            period=time_range.period,
            sections=[overview_section, modality_section],
            summary={
                "total_diagnoses": total_diagnoses,
                "total_interventions": total_interventions,
                "avg_engagement": float(avg_engagement),
                "modality_distribution": modality_usage,
            },
        )


class OperationalHealthReportGenerator(ReportGenerator):
    """Generate operational health reports."""

    @property
    def report_type(self) -> ReportType:
        return ReportType.OPERATIONAL_HEALTH

    async def generate(
        self, aggregator: AnalyticsAggregator, time_range: ReportTimeRange
    ) -> Report:
        """Generate operational health report."""
        store = aggregator.store

        response_times = await store.get_aggregated(
            "response.generation_time_ms",
            window_type=AggregationWindow.HOUR,
            start_time=time_range.start,
            end_time=time_range.end,
        )

        total_responses = sum(m.count for m in response_times)
        total_time_ms = sum(m.sum_value for m in response_times)
        avg_response_time = total_time_ms / Decimal(max(1, total_responses)) if response_times else Decimal("0")
        p95_response_time = max((m.max_value or Decimal("0")) for m in response_times) if response_times else Decimal("0")

        performance_section = ReportSection(
            title="System Performance",
            description="Response time and throughput metrics",
            data={
                "total_responses_generated": total_responses,
                "average_response_time_ms": float(avg_response_time),
                "p95_response_time_ms": float(p95_response_time),
            },
        )

        event_summary = await aggregator.get_event_summary()
        throughput_section = ReportSection(
            title="Event Throughput",
            description="Event processing volume by type",
            data=event_summary,
        )

        return Report(
            report_type=self.report_type,
            title="Operational Health Report",
            description=f"System performance from {time_range.start.date()} to {time_range.end.date()}",
            time_range_start=time_range.start,
            time_range_end=time_range.end,
            period=time_range.period,
            sections=[performance_section, throughput_section],
            summary={
                "total_responses": total_responses,
                "avg_response_time_ms": float(avg_response_time),
                "p95_response_time_ms": float(p95_response_time),
            },
        )


class ReportService:
    """Service for generating and managing analytics reports."""

    def __init__(self, aggregator: AnalyticsAggregator) -> None:
        self._aggregator = aggregator
        self._generators: dict[ReportType, ReportGenerator] = {
            ReportType.SESSION_SUMMARY: SessionSummaryReportGenerator(),
            ReportType.SAFETY_OVERVIEW: SafetyOverviewReportGenerator(),
            ReportType.CLINICAL_OUTCOMES: ClinicalOutcomesReportGenerator(),
            ReportType.OPERATIONAL_HEALTH: OperationalHealthReportGenerator(),
        }
        self._report_cache: dict[str, Report] = {}
        self._stats = {"reports_generated": 0, "cache_hits": 0}
        logger.info("report_service_initialized", available_reports=list(self._generators.keys()))

    async def generate_report(
        self,
        report_type: ReportType,
        time_range: ReportTimeRange | None = None,
        use_cache: bool = True,
    ) -> Report:
        """Generate a report of the specified type."""
        time_range = time_range or ReportTimeRange.for_last_day()
        cache_key = f"{report_type.value}:{time_range.start.isoformat()}:{time_range.end.isoformat()}"

        if use_cache and cache_key in self._report_cache:
            self._stats["cache_hits"] += 1
            logger.debug("report_cache_hit", report_type=report_type.value)
            return self._report_cache[cache_key]

        generator = self._generators.get(report_type)
        if not generator:
            raise ValueError(f"No generator for report type: {report_type}")

        logger.info("generating_report", report_type=report_type.value, period=time_range.period.value)
        report = await generator.generate(self._aggregator, time_range)
        self._report_cache[cache_key] = report
        self._stats["reports_generated"] += 1

        logger.info("report_generated", report_id=str(report.report_id), sections=len(report.sections))
        return report

    async def get_available_report_types(self) -> list[ReportType]:
        """Get list of available report types."""
        return list(self._generators.keys())

    async def get_statistics(self) -> dict[str, Any]:
        """Get report service statistics."""
        return {
            **self._stats,
            "cached_reports": len(self._report_cache),
            "available_types": [rt.value for rt in self._generators.keys()],
        }

    def clear_cache(self) -> None:
        """Clear the report cache."""
        self._report_cache.clear()
        logger.info("report_cache_cleared")

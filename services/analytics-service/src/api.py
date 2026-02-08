"""
Solace-AI Analytics Service - API Endpoints.

REST API for analytics queries, report generation, and metrics access.
Implements query validation, caching, and rate limiting.

Architecture Layer: Infrastructure (API)
Principles: Clean API Design, Request Validation, Response Caching
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field, field_validator
import structlog

try:
    from .aggregations import (
        AggregationWindow,
        AggregatedMetric,
        AnalyticsAggregator,
        MetricsStore,
    )
    from .reports import (
        Report,
        ReportService,
        ReportType,
        ReportTimeRange,
        ReportPeriod,
    )
    from .consumer import AnalyticsConsumer
except ImportError:
    from aggregations import (
        AggregationWindow,
        AggregatedMetric,
        AnalyticsAggregator,
        MetricsStore,
    )
    from reports import (
        Report,
        ReportService,
        ReportType,
        ReportTimeRange,
        ReportPeriod,
    )
    from consumer import AnalyticsConsumer

from solace_security.middleware import (
    AuthenticatedUser,
    AuthenticatedService,
    get_current_user,
    get_current_service,
    require_roles,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


_analytics_aggregator: AnalyticsAggregator | None = None
_report_service: ReportService | None = None
_analytics_consumer: AnalyticsConsumer | None = None


def get_aggregator() -> AnalyticsAggregator:
    """Dependency to get analytics aggregator."""
    if _analytics_aggregator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics service not initialized",
        )
    return _analytics_aggregator


def get_report_service() -> ReportService:
    """Dependency to get report service."""
    if _report_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report service not initialized",
        )
    return _report_service


def get_consumer() -> AnalyticsConsumer:
    """Dependency to get analytics consumer."""
    if _analytics_consumer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics consumer not initialized",
        )
    return _analytics_consumer


def set_dependencies(
    aggregator: AnalyticsAggregator,
    report_service: ReportService,
    consumer: AnalyticsConsumer,
) -> None:
    """Set global dependencies for API routes."""
    global _analytics_aggregator, _report_service, _analytics_consumer
    _analytics_aggregator = aggregator
    _report_service = report_service
    _analytics_consumer = consumer


class TimeRangeRequest(BaseModel):
    """Request model for time range specification."""
    start_time: datetime | None = Field(
        default=None,
        description="Start of time range (UTC). Defaults to 24 hours ago.",
    )
    end_time: datetime | None = Field(
        default=None,
        description="End of time range (UTC). Defaults to now.",
    )
    period: Literal["hourly", "daily", "weekly", "monthly"] = Field(
        default="daily",
        description="Reporting period.",
    )

    def to_report_time_range(self) -> ReportTimeRange:
        """Convert to ReportTimeRange."""
        now = datetime.now(timezone.utc)
        start = self.start_time or (now - timedelta(days=1))
        end = self.end_time or now
        period_map = {
            "hourly": ReportPeriod.HOURLY,
            "daily": ReportPeriod.DAILY,
            "weekly": ReportPeriod.WEEKLY,
            "monthly": ReportPeriod.MONTHLY,
        }
        return ReportTimeRange(start=start, end=end, period=period_map[self.period])


class MetricQueryRequest(BaseModel):
    """Request model for metric queries."""
    metric_name: str = Field(min_length=1, max_length=200)
    window: Literal["minute", "hour", "day"] = Field(default="hour")
    start_time: datetime | None = None
    end_time: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class MetricResponse(BaseModel):
    """Response model for metric data."""
    metric_name: str
    window: str
    window_start: datetime
    window_end: datetime
    count: int
    sum_value: float
    min_value: float | None
    max_value: float | None
    avg_value: float | None
    labels: dict[str, str]


class MetricsListResponse(BaseModel):
    """Response model for metrics list."""
    metrics: list[MetricResponse]
    total: int
    query_time_ms: int


class DashboardResponse(BaseModel):
    """Response model for dashboard metrics."""
    sessions_last_hour: int
    safety_checks_last_hour: int
    crisis_events_last_24h: int
    event_totals: dict[str, int]
    generated_at: str


class ReportResponse(BaseModel):
    """Response model for generated report."""
    report_id: UUID
    report_type: str
    title: str
    description: str
    time_range_start: datetime
    time_range_end: datetime
    period: str
    generated_at: datetime
    status: str
    sections: list[dict[str, Any]]
    summary: dict[str, Any]


class ReportTypesResponse(BaseModel):
    """Response model for available report types."""
    available_types: list[str]


class ConsumerStatsResponse(BaseModel):
    """Response model for consumer statistics."""
    config: dict[str, Any]
    metrics: dict[str, Any]
    running: bool
    queue_size: int
    current_batch_size: int


class ServiceStatsResponse(BaseModel):
    """Response model for service statistics."""
    aggregator: dict[str, Any]
    report_service: dict[str, Any]
    consumer: dict[str, Any]


class IngestEventRequest(BaseModel):
    """Request model for event ingestion."""
    event_type: str = Field(min_length=1, max_length=100)
    user_id: UUID
    session_id: UUID | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Validate event type format."""
        valid_prefixes = ("session.", "safety.", "diagnosis.", "therapy.", "memory.", "personality.", "system.")
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(f"Event type must start with one of: {valid_prefixes}")
        return v


class IngestEventResponse(BaseModel):
    """Response model for event ingestion."""
    accepted: bool
    event_id: str | None = None
    message: str


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    aggregator: AnalyticsAggregator = Depends(get_aggregator),
) -> DashboardResponse:
    """Get dashboard metrics summary."""
    logger.info("dashboard_request")
    start_time = datetime.now(timezone.utc)

    metrics = await aggregator.get_dashboard_metrics()

    query_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
    logger.info("dashboard_response", query_time_ms=query_time)

    return DashboardResponse(
        sessions_last_hour=metrics["sessions_last_hour"],
        safety_checks_last_hour=metrics["safety_checks_last_hour"],
        crisis_events_last_24h=metrics["crisis_events_last_24h"],
        event_totals=metrics["event_totals"],
        generated_at=metrics["generated_at"],
    )


@router.post("/metrics/query", response_model=MetricsListResponse)
async def query_metrics(
    request: MetricQueryRequest,
    aggregator: AnalyticsAggregator = Depends(get_aggregator),
) -> MetricsListResponse:
    """Query aggregated metrics."""
    logger.info("metrics_query", metric_name=request.metric_name, window=request.window)
    start_time = datetime.now(timezone.utc)

    window_map = {
        "minute": AggregationWindow.MINUTE,
        "hour": AggregationWindow.HOUR,
        "day": AggregationWindow.DAY,
    }

    now = datetime.now(timezone.utc)
    query_start = request.start_time or (now - timedelta(hours=24))
    query_end = request.end_time or now

    metrics = await aggregator.store.get_aggregated(
        metric_name=request.metric_name,
        window_type=window_map[request.window],
        start_time=query_start,
        end_time=query_end,
        labels=request.labels if request.labels else None,
    )

    response_metrics = [
        MetricResponse(
            metric_name=m.metric_name,
            window=m.window.value,
            window_start=m.window_start,
            window_end=m.window_end,
            count=m.count,
            sum_value=float(m.sum_value),
            min_value=float(m.min_value) if m.min_value is not None else None,
            max_value=float(m.max_value) if m.max_value is not None else None,
            avg_value=float(m.avg_value) if m.avg_value is not None else None,
            labels=m.labels,
        )
        for m in metrics
    ]

    query_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
    logger.info("metrics_query_complete", results=len(response_metrics), query_time_ms=query_time)

    return MetricsListResponse(
        metrics=response_metrics,
        total=len(response_metrics),
        query_time_ms=query_time,
    )


@router.get("/metrics/names", response_model=list[str])
async def list_metric_names(
    aggregator: AnalyticsAggregator = Depends(get_aggregator),
) -> list[str]:
    """List all available metric names."""
    logger.info("list_metric_names")
    return await aggregator.store.get_metric_names()


@router.get("/reports/types", response_model=ReportTypesResponse)
async def list_report_types(
    report_service: ReportService = Depends(get_report_service),
) -> ReportTypesResponse:
    """List available report types."""
    types = await report_service.get_available_report_types()
    return ReportTypesResponse(available_types=[t.value for t in types])


@router.post("/reports/generate", response_model=ReportResponse)
async def generate_report(
    report_type: str = Query(description="Type of report to generate"),
    start_time: datetime | None = Query(default=None, description="Start time (UTC)"),
    end_time: datetime | None = Query(default=None, description="End time (UTC)"),
    use_cache: bool = Query(default=True, description="Use cached report if available"),
    report_service: ReportService = Depends(get_report_service),
) -> ReportResponse:
    """Generate an analytics report."""
    try:
        rt = ReportType(report_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid report type: {report_type}. Valid types: {[t.value for t in ReportType]}",
        )

    logger.info("report_generation_request", report_type=report_type)

    now = datetime.now(timezone.utc)
    time_range = ReportTimeRange.custom(
        start=start_time or (now - timedelta(days=1)),
        end=end_time or now,
    )

    report = await report_service.generate_report(rt, time_range, use_cache=use_cache)

    logger.info("report_generated", report_id=str(report.report_id), sections=len(report.sections))

    return ReportResponse(
        report_id=report.report_id,
        report_type=report.report_type.value,
        title=report.title,
        description=report.description,
        time_range_start=report.time_range_start,
        time_range_end=report.time_range_end,
        period=report.period.value,
        generated_at=report.generated_at,
        status=report.status.value,
        sections=[s.model_dump() for s in report.sections],
        summary=report.summary,
    )


@router.get("/reports/{report_type}", response_model=ReportResponse)
async def get_report(
    report_type: str,
    period: Literal["hourly", "daily", "weekly", "monthly"] = Query(default="daily"),
    report_service: ReportService = Depends(get_report_service),
) -> ReportResponse:
    """Get a standard period report by type."""
    try:
        rt = ReportType(report_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid report type: {report_type}",
        )

    period_map = {
        "hourly": ReportTimeRange.for_last_hour,
        "daily": ReportTimeRange.for_last_day,
        "weekly": ReportTimeRange.for_last_week,
        "monthly": ReportTimeRange.for_last_month,
    }

    time_range = period_map[period]()
    report = await report_service.generate_report(rt, time_range)

    return ReportResponse(
        report_id=report.report_id,
        report_type=report.report_type.value,
        title=report.title,
        description=report.description,
        time_range_start=report.time_range_start,
        time_range_end=report.time_range_end,
        period=report.period.value,
        generated_at=report.generated_at,
        status=report.status.value,
        sections=[s.model_dump() for s in report.sections],
        summary=report.summary,
    )


@router.post("/events/ingest", response_model=IngestEventResponse)
async def ingest_event(
    request: IngestEventRequest,
    consumer: AnalyticsConsumer = Depends(get_consumer),
) -> IngestEventResponse:
    """Ingest an event for analytics processing."""
    from uuid import uuid4

    event_id = uuid4()
    event_data = {
        "event_type": request.event_type,
        "user_id": str(request.user_id),
        "session_id": str(request.session_id) if request.session_id else None,
        "metadata": {
            "event_id": str(event_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(uuid4()),
            "source_service": "api-ingest",
            **request.metadata,
        },
        **request.payload,
    }

    success = await consumer.process_event(event_data)

    if success:
        logger.info("event_ingested", event_id=str(event_id), event_type=request.event_type)
        return IngestEventResponse(
            accepted=True,
            event_id=str(event_id),
            message="Event accepted for processing",
        )
    else:
        logger.warning("event_ingest_failed", event_type=request.event_type)
        return IngestEventResponse(
            accepted=False,
            message="Event processing failed",
        )


@router.get("/consumer/stats", response_model=ConsumerStatsResponse)
async def get_consumer_stats(
    consumer: AnalyticsConsumer = Depends(get_consumer),
) -> ConsumerStatsResponse:
    """Get consumer statistics."""
    stats = await consumer.get_statistics()
    return ConsumerStatsResponse(**stats)


@router.get("/stats", response_model=ServiceStatsResponse)
async def get_service_stats(
    aggregator: AnalyticsAggregator = Depends(get_aggregator),
    report_service: ReportService = Depends(get_report_service),
    consumer: AnalyticsConsumer = Depends(get_consumer),
) -> ServiceStatsResponse:
    """Get overall service statistics."""
    return ServiceStatsResponse(
        aggregator=await aggregator.store.get_statistics(),
        report_service=await report_service.get_statistics(),
        consumer=await consumer.get_statistics(),
    )


@router.get("/health")
async def analytics_health() -> dict[str, Any]:
    """Analytics-specific health check."""
    status_info: dict[str, Any] = {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

    if _analytics_aggregator is not None:
        status_info["aggregator"] = "initialized"
        status_info["metrics_tracked"] = (await _analytics_aggregator.store.get_statistics())["metrics_tracked"]
    else:
        status_info["aggregator"] = "not_initialized"
        status_info["status"] = "degraded"

    if _analytics_consumer is not None:
        status_info["consumer"] = "initialized"
        status_info["consumer_running"] = (await _analytics_consumer.get_statistics())["running"]
    else:
        status_info["consumer"] = "not_initialized"

    return status_info

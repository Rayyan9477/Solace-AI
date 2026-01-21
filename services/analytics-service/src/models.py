"""
Solace-AI Analytics Service - Data Models.

Data transfer objects for analytics storage operations.
These models represent the storage schema for events, metrics, and aggregations.

Architecture Layer: Infrastructure
Principles: Data Transfer Objects, Immutable Records, Type Safety
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TableName(str, Enum):
    """Analytics table names."""
    EVENTS = "analytics_events"
    METRICS = "analytics_metrics"
    SESSIONS = "analytics_sessions"
    AGGREGATIONS = "analytics_aggregations"


class AnalyticsEvent(BaseModel):
    """Analytics event for storage."""
    event_id: UUID
    event_type: str
    category: str
    user_id: UUID
    session_id: UUID | None = None
    timestamp: datetime
    correlation_id: UUID
    source_service: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetricRecord(BaseModel):
    """Metric record for storage."""
    metric_id: UUID
    metric_name: str
    value: Decimal
    labels: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    window_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionRecord(BaseModel):
    """Session record for storage."""
    session_id: UUID
    user_id: UUID
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    message_count: int = 0
    channel: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregationRecord(BaseModel):
    """Pre-computed aggregation record."""
    aggregation_id: UUID
    metric_name: str
    window_type: str
    window_start: datetime
    window_end: datetime
    count: int
    sum_value: Decimal
    min_value: Decimal | None = None
    max_value: Decimal | None = None
    avg_value: Decimal | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

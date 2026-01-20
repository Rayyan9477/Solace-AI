"""
Pytest configuration and fixtures for analytics service tests.
"""
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from aggregations import (
    MetricsStore,
    AnalyticsAggregator,
    AggregationWindow,
    MetricBucket,
    TimeWindow,
)
from reports import ReportService, ReportTimeRange
from consumer import AnalyticsConsumer, ConsumerConfig, AnalyticsEvent, EventCategory


@pytest.fixture
def metrics_store():
    """Create a metrics store for testing."""
    return MetricsStore(max_windows_per_metric=100)


@pytest.fixture
def analytics_aggregator(metrics_store):
    """Create an analytics aggregator with a metrics store."""
    return AnalyticsAggregator(store=metrics_store)


@pytest.fixture
def report_service(analytics_aggregator):
    """Create a report service."""
    return ReportService(analytics_aggregator)


@pytest.fixture
def consumer_config():
    """Create consumer configuration."""
    return ConsumerConfig(
        group_id="test-analytics",
        batch_size=10,
        batch_timeout_ms=1000,
    )


@pytest.fixture
def analytics_consumer(analytics_aggregator, consumer_config):
    """Create an analytics consumer."""
    return AnalyticsConsumer(aggregator=analytics_aggregator, config=consumer_config)


@pytest.fixture
def sample_user_id():
    """Generate a sample user ID."""
    return uuid4()


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID."""
    return uuid4()


@pytest.fixture
def sample_session_event(sample_user_id, sample_session_id):
    """Create a sample session event."""
    return {
        "event_type": "session.started",
        "user_id": str(sample_user_id),
        "session_id": str(sample_session_id),
        "metadata": {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(uuid4()),
            "source_service": "orchestrator-service",
        },
        "session_number": 1,
        "channel": "web",
    }


@pytest.fixture
def sample_safety_event(sample_user_id, sample_session_id):
    """Create a sample safety event."""
    return {
        "event_type": "safety.assessment.completed",
        "user_id": str(sample_user_id),
        "session_id": str(sample_session_id),
        "metadata": {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(uuid4()),
            "source_service": "safety-service",
        },
        "risk_level": "LOW",
        "risk_score": "0.2",
        "detection_layer": 1,
        "recommended_action": "continue",
    }


@pytest.fixture
def sample_therapy_event(sample_user_id, sample_session_id):
    """Create a sample therapy event."""
    return {
        "event_type": "therapy.intervention.delivered",
        "user_id": str(sample_user_id),
        "session_id": str(sample_session_id),
        "metadata": {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(uuid4()),
            "source_service": "therapy-service",
        },
        "modality": "CBT",
        "technique": "cognitive_restructuring",
        "user_engagement_score": "0.85",
    }


@pytest.fixture
def sample_diagnosis_event(sample_user_id, sample_session_id):
    """Create a sample diagnosis event."""
    return {
        "event_type": "diagnosis.completed",
        "user_id": str(sample_user_id),
        "session_id": str(sample_session_id),
        "metadata": {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": str(uuid4()),
            "source_service": "diagnosis-service",
        },
        "primary_hypothesis": {
            "condition_code": "F32.0",
            "condition_name": "Major Depressive Disorder",
            "confidence": "MODERATE",
            "severity": "MILD",
        },
        "stepped_care_level": 2,
    }


@pytest.fixture
def time_range_last_hour():
    """Create a time range for the last hour."""
    return ReportTimeRange.for_last_hour()


@pytest.fixture
def time_range_last_day():
    """Create a time range for the last day."""
    return ReportTimeRange.for_last_day()

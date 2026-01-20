"""
Unit tests for analytics API endpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from api import (
    router,
    set_dependencies,
    TimeRangeRequest,
    MetricQueryRequest,
    IngestEventRequest,
)
from aggregations import AnalyticsAggregator, MetricsStore
from reports import ReportService, ReportType
from consumer import AnalyticsConsumer, ConsumerConfig


@pytest.fixture
def app(analytics_aggregator, report_service, analytics_consumer):
    """Create a FastAPI app with analytics routes."""
    app = FastAPI()
    app.include_router(router)
    set_dependencies(analytics_aggregator, report_service, analytics_consumer)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestTimeRangeRequest:
    """Tests for TimeRangeRequest model."""

    def test_default_values(self):
        """Test default time range values."""
        request = TimeRangeRequest()

        assert request.start_time is None
        assert request.end_time is None
        assert request.period == "daily"

    def test_to_report_time_range(self):
        """Test conversion to ReportTimeRange."""
        now = datetime.now(timezone.utc)
        request = TimeRangeRequest(
            start_time=now - timedelta(hours=2),
            end_time=now,
            period="hourly",
        )

        time_range = request.to_report_time_range()

        assert time_range.start == request.start_time
        assert time_range.end == request.end_time


class TestMetricQueryRequest:
    """Tests for MetricQueryRequest model."""

    def test_valid_request(self):
        """Test creating valid request."""
        request = MetricQueryRequest(
            metric_name="test.metric",
            window="hour",
            labels={"service": "test"},
        )

        assert request.metric_name == "test.metric"
        assert request.window == "hour"


class TestIngestEventRequest:
    """Tests for IngestEventRequest model."""

    def test_valid_request(self):
        """Test creating valid request."""
        request = IngestEventRequest(
            event_type="session.started",
            user_id=uuid4(),
            session_id=uuid4(),
            payload={"test": "data"},
        )

        assert request.event_type == "session.started"

    def test_invalid_event_type(self):
        """Test invalid event type."""
        with pytest.raises(ValueError):
            IngestEventRequest(
                event_type="invalid_prefix",
                user_id=uuid4(),
            )


class TestDashboardEndpoint:
    """Tests for dashboard endpoint."""

    def test_get_dashboard(self, client):
        """Test getting dashboard metrics."""
        response = client.get("/api/v1/analytics/dashboard")

        assert response.status_code == 200
        data = response.json()
        assert "sessions_last_hour" in data
        assert "safety_checks_last_hour" in data
        assert "generated_at" in data


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_query_metrics(self, client):
        """Test querying metrics."""
        response = client.post(
            "/api/v1/analytics/metrics/query",
            json={
                "metric_name": "test.metric",
                "window": "hour",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "total" in data
        assert "query_time_ms" in data

    def test_list_metric_names(self, client, analytics_aggregator):
        """Test listing metric names."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            analytics_aggregator.store.record("test.metric.one", Decimal("1"))
        )

        response = client.get("/api/v1/analytics/metrics/names")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestReportEndpoints:
    """Tests for report endpoints."""

    def test_list_report_types(self, client):
        """Test listing available report types."""
        response = client.get("/api/v1/analytics/reports/types")

        assert response.status_code == 200
        data = response.json()
        assert "available_types" in data
        assert "session_summary" in data["available_types"]

    def test_generate_report(self, client):
        """Test generating a report."""
        response = client.post(
            "/api/v1/analytics/reports/generate?report_type=session_summary"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "session_summary"
        assert "sections" in data
        assert "summary" in data

    def test_generate_report_invalid_type(self, client):
        """Test generating report with invalid type."""
        response = client.post(
            "/api/v1/analytics/reports/generate?report_type=invalid_type"
        )

        assert response.status_code == 400

    def test_get_report_by_type(self, client):
        """Test getting standard period report."""
        response = client.get("/api/v1/analytics/reports/session_summary?period=daily")

        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "session_summary"

    def test_get_report_invalid_type(self, client):
        """Test getting report with invalid type."""
        response = client.get("/api/v1/analytics/reports/invalid_type")

        assert response.status_code == 400


class TestEventIngestionEndpoint:
    """Tests for event ingestion endpoint."""

    def test_ingest_valid_event(self, client):
        """Test ingesting a valid event."""
        user_id = str(uuid4())
        response = client.post(
            "/api/v1/analytics/events/ingest",
            json={
                "event_type": "session.started",
                "user_id": user_id,
                "payload": {"channel": "web"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is True
        assert data["event_id"] is not None

    def test_ingest_event_with_session(self, client):
        """Test ingesting event with session ID."""
        user_id = str(uuid4())
        session_id = str(uuid4())
        response = client.post(
            "/api/v1/analytics/events/ingest",
            json={
                "event_type": "session.message.received",
                "user_id": user_id,
                "session_id": session_id,
                "payload": {"content_length": 100},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is True


class TestConsumerStatsEndpoint:
    """Tests for consumer stats endpoint."""

    def test_get_consumer_stats(self, client):
        """Test getting consumer statistics."""
        response = client.get("/api/v1/analytics/consumer/stats")

        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "metrics" in data
        assert "running" in data


class TestServiceStatsEndpoint:
    """Tests for service stats endpoint."""

    def test_get_service_stats(self, client):
        """Test getting overall service statistics."""
        response = client.get("/api/v1/analytics/stats")

        assert response.status_code == 200
        data = response.json()
        assert "aggregator" in data
        assert "report_service" in data
        assert "consumer" in data


class TestHealthEndpoint:
    """Tests for analytics health endpoint."""

    def test_analytics_health(self, client):
        """Test analytics-specific health check."""
        response = client.get("/api/v1/analytics/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestQueryWithTimeRange:
    """Tests for metrics query with time ranges."""

    def test_query_with_custom_time_range(self, client):
        """Test querying metrics with custom time range."""
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=6)).isoformat()
        end = now.isoformat()

        response = client.post(
            "/api/v1/analytics/metrics/query",
            json={
                "metric_name": "test.metric",
                "window": "hour",
                "start_time": start,
                "end_time": end,
            },
        )

        assert response.status_code == 200

    def test_query_with_labels(self, client):
        """Test querying metrics with labels."""
        response = client.post(
            "/api/v1/analytics/metrics/query",
            json={
                "metric_name": "test.metric",
                "window": "hour",
                "labels": {"service": "test", "environment": "dev"},
            },
        )

        assert response.status_code == 200


class TestReportWithParameters:
    """Tests for report generation with parameters."""

    def test_generate_report_with_time_range(self, client):
        """Test generating report with custom time range."""
        from urllib.parse import quote
        now = datetime.now(timezone.utc)
        start = quote((now - timedelta(days=7)).isoformat())
        end = quote(now.isoformat())

        response = client.post(
            f"/api/v1/analytics/reports/generate?report_type=session_summary&start_time={start}&end_time={end}"
        )

        assert response.status_code == 200

    def test_generate_report_no_cache(self, client):
        """Test generating report without cache."""
        response = client.post(
            "/api/v1/analytics/reports/generate?report_type=safety_overview&use_cache=false"
        )

        assert response.status_code == 200

    def test_generate_all_report_types(self, client):
        """Test generating all supported report types."""
        report_types = ["session_summary", "safety_overview", "clinical_outcomes", "operational_health"]

        for report_type in report_types:
            response = client.post(
                f"/api/v1/analytics/reports/generate?report_type={report_type}"
            )
            assert response.status_code == 200, f"Failed for {report_type}"

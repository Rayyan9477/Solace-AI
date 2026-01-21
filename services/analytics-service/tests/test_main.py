"""
Unit tests for analytics main module and observability configuration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestObservabilityConfig:
    """Tests for ObservabilityConfig model."""

    def test_default_values(self):
        """Test default observability configuration."""
        from main import ObservabilityConfig

        config = ObservabilityConfig()

        assert config.prometheus_enabled is True
        assert config.prometheus_endpoint == "/metrics"
        assert config.otel_enabled is False
        assert config.otel_service_name == "analytics-service"
        assert config.otel_exporter_otlp_endpoint == "http://localhost:4317"

    def test_custom_values(self):
        """Test custom observability configuration."""
        from main import ObservabilityConfig

        config = ObservabilityConfig(
            prometheus_enabled=False,
            prometheus_endpoint="/custom-metrics",
            otel_enabled=True,
            otel_service_name="custom-service",
            otel_exporter_otlp_endpoint="http://otel-collector:4317",
        )

        assert config.prometheus_enabled is False
        assert config.prometheus_endpoint == "/custom-metrics"
        assert config.otel_enabled is True
        assert config.otel_service_name == "custom-service"
        assert config.otel_exporter_otlp_endpoint == "http://otel-collector:4317"


class TestAnalyticsServiceSettings:
    """Tests for AnalyticsServiceSettings model."""

    def test_default_settings(self):
        """Test default analytics service settings."""
        from main import AnalyticsServiceSettings

        settings = AnalyticsServiceSettings()

        assert settings.service.name == "analytics-service"
        assert settings.service.env == "development"
        assert settings.service.port == 8009
        assert settings.metrics.max_windows_per_metric == 1000
        assert settings.consumer.group_id == "analytics-service"
        assert settings.observability.prometheus_enabled is True

    def test_load_settings(self):
        """Test loading settings from environment."""
        from main import AnalyticsServiceSettings

        settings = AnalyticsServiceSettings.load()

        assert settings is not None
        assert settings.service.name == "analytics-service"


class TestServiceConfig:
    """Tests for ServiceConfig model."""

    def test_default_values(self):
        """Test default service configuration."""
        from main import ServiceConfig

        config = ServiceConfig()

        assert config.name == "analytics-service"
        assert config.env == "development"
        assert config.host == "0.0.0.0"
        assert config.port == 8009
        assert config.log_level == "INFO"

    def test_port_validation(self):
        """Test port validation."""
        from main import ServiceConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ServiceConfig(port=0)

        with pytest.raises(ValidationError):
            ServiceConfig(port=70000)


class TestMetricsConfig:
    """Tests for MetricsConfig model."""

    def test_default_values(self):
        """Test default metrics configuration."""
        from main import MetricsConfig

        config = MetricsConfig()

        assert config.max_windows_per_metric == 1000
        assert config.default_retention_hours == 168


class TestConsumerConfigSettings:
    """Tests for ConsumerConfig settings model."""

    def test_default_values(self):
        """Test default consumer configuration."""
        from main import ConsumerConfig

        config = ConsumerConfig()

        assert config.group_id == "analytics-service"
        assert config.batch_size == 100
        assert config.batch_timeout_ms == 5000
        assert config.kafka_enabled is False
        assert config.kafka_bootstrap_servers == "localhost:9092"


class TestSetupPrometheus:
    """Tests for Prometheus setup function."""

    def test_prometheus_disabled(self):
        """Test Prometheus setup when disabled."""
        from main import _setup_prometheus, AnalyticsServiceSettings, ObservabilityConfig

        app = FastAPI()
        settings = AnalyticsServiceSettings()
        settings.observability = ObservabilityConfig(prometheus_enabled=False)

        _setup_prometheus(app, settings)

        routes = [route.path for route in app.routes]
        assert "/metrics" not in routes

    def test_prometheus_enabled_without_package(self):
        """Test Prometheus setup when package not installed."""
        from main import _setup_prometheus, AnalyticsServiceSettings

        app = FastAPI()
        settings = AnalyticsServiceSettings()

        with patch.dict('sys.modules', {'prometheus_fastapi_instrumentator': None}):
            with patch('main.logger') as mock_logger:
                _setup_prometheus(app, settings)


class TestSetupOpenTelemetry:
    """Tests for OpenTelemetry setup function."""

    def test_otel_disabled(self):
        """Test OpenTelemetry setup when disabled."""
        from main import _setup_opentelemetry, AnalyticsServiceSettings

        app = FastAPI()
        settings = AnalyticsServiceSettings()

        _setup_opentelemetry(app, settings)

    def test_otel_enabled_without_package(self):
        """Test OpenTelemetry setup when package not installed."""
        from main import _setup_opentelemetry, AnalyticsServiceSettings, ObservabilityConfig

        app = FastAPI()
        settings = AnalyticsServiceSettings()
        settings.observability = ObservabilityConfig(otel_enabled=True)

        with patch.dict('sys.modules', {'opentelemetry': None}):
            with patch('main.logger') as mock_logger:
                _setup_opentelemetry(app, settings)


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi(self):
        """Test that create_app returns FastAPI instance."""
        from main import create_app

        app = create_app()

        assert isinstance(app, FastAPI)
        assert app.title == "Solace-AI Analytics Service"
        assert app.version == "1.0.0"

    def test_app_has_health_endpoints(self):
        """Test that app has health endpoints."""
        from main import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_app_has_root_endpoint(self):
        """Test that app has root endpoint."""
        from main import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "analytics-service"
        assert data["status"] == "running"

    def test_ready_endpoint_before_initialization(self):
        """Test ready endpoint before services are initialized."""
        from main import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/ready")
        assert response.status_code == 200


class TestDependencyGetters:
    """Tests for dependency getter functions."""

    def test_get_analytics_aggregator_not_initialized(self):
        """Test getting aggregator when not initialized."""
        import main
        original = main._analytics_aggregator
        main._analytics_aggregator = None

        try:
            with pytest.raises(RuntimeError, match="Analytics aggregator not initialized"):
                main.get_analytics_aggregator()
        finally:
            main._analytics_aggregator = original

    def test_get_report_service_not_initialized(self):
        """Test getting report service when not initialized."""
        import main
        original = main._report_service
        main._report_service = None

        try:
            with pytest.raises(RuntimeError, match="Report service not initialized"):
                main.get_report_service()
        finally:
            main._report_service = original

    def test_get_analytics_consumer_not_initialized(self):
        """Test getting consumer when not initialized."""
        import main
        original = main._analytics_consumer
        main._analytics_consumer = None

        try:
            with pytest.raises(RuntimeError, match="Analytics consumer not initialized"):
                main.get_analytics_consumer()
        finally:
            main._analytics_consumer = original


class TestCreateServices:
    """Tests for _create_services function."""

    def test_create_services(self):
        """Test creating services tuple."""
        from main import _create_services, AnalyticsServiceSettings
        from aggregations import AnalyticsAggregator
        from reports import ReportService
        from consumer import AnalyticsConsumer

        settings = AnalyticsServiceSettings()
        aggregator, report_service, consumer = _create_services(settings)

        assert isinstance(aggregator, AnalyticsAggregator)
        assert isinstance(report_service, ReportService)
        assert isinstance(consumer, AnalyticsConsumer)

    def test_create_services_respects_config(self):
        """Test that create_services uses configuration values."""
        from main import _create_services, AnalyticsServiceSettings, MetricsConfig, ConsumerConfig as SettingsConsumerConfig

        settings = AnalyticsServiceSettings()
        settings.metrics = MetricsConfig(max_windows_per_metric=500)
        settings.consumer = SettingsConsumerConfig(group_id="custom-group", batch_size=50)

        aggregator, report_service, consumer = _create_services(settings)

        assert aggregator.store._max_windows == 500

"""
Unit tests for analytics service configuration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import timedelta
from pydantic import ValidationError

from config import (
    Environment,
    ServiceConfiguration,
    ClickHouseConfig,
    MetricsStoreConfig,
    ConsumerConfiguration,
    ReportConfig,
    AggregationConfig,
    ObservabilityConfig,
    AlertConfig,
    AnalyticsServiceConfig,
    get_config,
    reset_config,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_all_environments_exist(self):
        """Test all expected environments exist."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestServiceConfiguration:
    """Tests for ServiceConfiguration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServiceConfiguration()

        assert config.name == "analytics-service"
        assert config.version == "1.0.0"
        assert config.env == Environment.DEVELOPMENT
        assert config.host == "0.0.0.0"
        assert config.port == 8009
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.workers == 1

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValidationError):
            ServiceConfiguration(port=0)

        with pytest.raises(ValidationError):
            ServiceConfiguration(port=70000)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServiceConfiguration(
            name="custom-analytics",
            env=Environment.PRODUCTION,
            port=9009,
            log_level="ERROR",
            workers=8,
        )

        assert config.name == "custom-analytics"
        assert config.env == Environment.PRODUCTION
        assert config.port == 9009
        assert config.workers == 8


class TestClickHouseConfig:
    """Tests for ClickHouseConfig."""

    def test_default_values(self):
        """Test default ClickHouse configuration."""
        config = ClickHouseConfig()

        assert config.enabled is False
        assert config.host == "localhost"
        assert config.port == 8123
        assert config.database == "solace_analytics"
        assert config.username == "default"
        assert config.password == ""
        assert config.secure is False
        assert config.verify is True
        assert config.connect_timeout == 10.0
        assert config.query_timeout == 300.0
        assert config.max_connections == 10
        assert config.compression is True

    def test_custom_values(self):
        """Test custom ClickHouse configuration."""
        config = ClickHouseConfig(
            enabled=True,
            host="clickhouse.example.com",
            port=8443,
            database="analytics_prod",
            username="admin",
            password="secret",
            secure=True,
            max_connections=50,
        )

        assert config.enabled is True
        assert config.host == "clickhouse.example.com"
        assert config.secure is True
        assert config.max_connections == 50


class TestMetricsStoreConfig:
    """Tests for MetricsStoreConfig."""

    def test_default_values(self):
        """Test default metrics store configuration."""
        config = MetricsStoreConfig()

        assert config.max_windows_per_metric == 1000
        assert config.default_retention_hours == 168
        assert config.aggregation_interval_seconds == 60
        assert config.flush_interval_seconds == 300
        assert config.max_labels_per_metric == 10

    def test_validation(self):
        """Test validation constraints."""
        with pytest.raises(ValidationError):
            MetricsStoreConfig(max_windows_per_metric=50)

        with pytest.raises(ValidationError):
            MetricsStoreConfig(default_retention_hours=0)


class TestConsumerConfiguration:
    """Tests for ConsumerConfiguration."""

    def test_default_values(self):
        """Test default consumer configuration."""
        config = ConsumerConfiguration()

        assert config.group_id == "analytics-service"
        assert len(config.topics) > 0
        assert "solace.sessions" in config.topics
        assert config.batch_size == 100
        assert config.batch_timeout_ms == 5000
        assert config.kafka_enabled is False
        assert config.auto_offset_reset == "latest"

    def test_custom_values(self):
        """Test custom consumer configuration."""
        config = ConsumerConfiguration(
            group_id="custom-consumer",
            topics=["topic1", "topic2"],
            batch_size=50,
            kafka_enabled=True,
            kafka_bootstrap_servers="kafka:9092",
            auto_offset_reset="earliest",
        )

        assert config.group_id == "custom-consumer"
        assert len(config.topics) == 2
        assert config.kafka_enabled is True
        assert config.auto_offset_reset == "earliest"


class TestReportConfig:
    """Tests for ReportConfig."""

    def test_default_values(self):
        """Test default report configuration."""
        config = ReportConfig()

        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300
        assert config.max_report_rows == 10000
        assert "json" in config.export_formats
        assert "csv" in config.export_formats

    def test_custom_values(self):
        """Test custom report configuration."""
        config = ReportConfig(
            cache_ttl_seconds=600,
            max_report_rows=50000,
            export_formats=["json", "xlsx"],
        )

        assert config.cache_ttl_seconds == 600
        assert config.max_report_rows == 50000
        assert "xlsx" in config.export_formats


class TestAggregationConfig:
    """Tests for AggregationConfig."""

    def test_default_values(self):
        """Test default aggregation configuration."""
        config = AggregationConfig()

        assert "minute" in config.windows
        assert "hour" in config.windows
        assert "day" in config.windows
        assert config.rollup_enabled is True
        assert 95 in config.percentiles
        assert 99 in config.percentiles

    def test_custom_values(self):
        """Test custom aggregation configuration."""
        config = AggregationConfig(
            windows=["hour", "day"],
            rollup_enabled=False,
            percentiles=[50, 90, 99],
        )

        assert len(config.windows) == 2
        assert config.rollup_enabled is False
        assert len(config.percentiles) == 3


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self):
        """Test default observability configuration."""
        config = ObservabilityConfig()

        assert config.prometheus_enabled is True
        assert config.prometheus_endpoint == "/metrics"
        assert config.otel_enabled is False
        assert config.otel_service_name == "analytics-service"
        assert config.log_format == "json"
        assert config.trace_sample_rate == 0.1

    def test_custom_values(self):
        """Test custom observability configuration."""
        config = ObservabilityConfig(
            otel_enabled=True,
            otel_exporter_endpoint="http://otel-collector:4317",
            trace_sample_rate=0.5,
        )

        assert config.otel_enabled is True
        assert config.trace_sample_rate == 0.5


class TestAlertConfig:
    """Tests for AlertConfig."""

    def test_default_values(self):
        """Test default alert configuration."""
        config = AlertConfig()

        assert config.enabled is False
        assert config.check_interval_seconds == 60
        assert config.notification_webhook is None
        assert "error_rate" in config.thresholds
        assert config.thresholds["error_rate"] == 0.05

    def test_custom_values(self):
        """Test custom alert configuration."""
        config = AlertConfig(
            enabled=True,
            notification_webhook="https://hooks.example.com/alert",
            thresholds={"error_rate": 0.01, "latency_p99_ms": 1000},
        )

        assert config.enabled is True
        assert config.notification_webhook is not None
        assert config.thresholds["error_rate"] == 0.01


class TestAnalyticsServiceConfig:
    """Tests for AnalyticsServiceConfig aggregate."""

    def test_default_configuration(self):
        """Test default aggregate configuration."""
        config = AnalyticsServiceConfig()

        assert config.service.name == "analytics-service"
        assert config.clickhouse.enabled is False
        assert config.consumer.kafka_enabled is False
        assert config.observability.prometheus_enabled is True

    def test_is_production(self):
        """Test production environment check."""
        config = AnalyticsServiceConfig()
        assert config.is_production() is False

        config.service.env = Environment.PRODUCTION
        assert config.is_production() is True

    def test_get_retention_timedelta(self):
        """Test retention timedelta calculation."""
        config = AnalyticsServiceConfig()
        retention = config.get_retention_timedelta()

        assert isinstance(retention, timedelta)
        assert retention.total_seconds() == 168 * 3600

    def test_load_configuration(self):
        """Test configuration loading."""
        config = AnalyticsServiceConfig.load()

        assert config is not None
        assert config.service.name == "analytics-service"


class TestConfigSingleton:
    """Tests for configuration singleton."""

    def test_get_config_returns_instance(self):
        """Test get_config returns configuration."""
        reset_config()
        config = get_config()
        assert config is not None
        assert isinstance(config, AnalyticsServiceConfig)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns singleton."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self):
        """Test configuration reset."""
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

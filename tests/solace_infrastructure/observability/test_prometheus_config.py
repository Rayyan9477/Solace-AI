"""Unit tests for Prometheus configuration module."""
from __future__ import annotations

import pytest

from solace_infrastructure.observability.prometheus_config import (
    ScrapeProtocol,
    ServiceDiscoveryType,
    MetricRelabelAction,
    PrometheusSettings,
    StaticTarget,
    RelabelConfig,
    ScrapeJob,
    SolacePrometheusScrapeJobs,
    RemoteWriteConfig,
    PrometheusConfigGenerator,
    create_prometheus_config,
)


class TestScrapeProtocol:
    """Tests for ScrapeProtocol enum."""

    def test_http_value(self) -> None:
        """Test HTTP protocol value."""
        assert ScrapeProtocol.HTTP.value == "http"

    def test_https_value(self) -> None:
        """Test HTTPS protocol value."""
        assert ScrapeProtocol.HTTPS.value == "https"


class TestServiceDiscoveryType:
    """Tests for ServiceDiscoveryType enum."""

    def test_static_value(self) -> None:
        """Test static discovery type."""
        assert ServiceDiscoveryType.STATIC.value == "static"

    def test_kubernetes_value(self) -> None:
        """Test kubernetes discovery type."""
        assert ServiceDiscoveryType.KUBERNETES.value == "kubernetes"


class TestMetricRelabelAction:
    """Tests for MetricRelabelAction enum."""

    def test_replace_action(self) -> None:
        """Test replace action value."""
        assert MetricRelabelAction.REPLACE.value == "replace"

    def test_keep_action(self) -> None:
        """Test keep action value."""
        assert MetricRelabelAction.KEEP.value == "keep"

    def test_drop_action(self) -> None:
        """Test drop action value."""
        assert MetricRelabelAction.DROP.value == "drop"


class TestPrometheusSettings:
    """Tests for PrometheusSettings."""

    def test_default_scrape_interval(self) -> None:
        """Test default scrape interval."""
        settings = PrometheusSettings()
        assert settings.global_scrape_interval == "15s"

    def test_default_scrape_timeout(self) -> None:
        """Test default scrape timeout."""
        settings = PrometheusSettings()
        assert settings.global_scrape_timeout == "10s"

    def test_default_retention_time(self) -> None:
        """Test default retention time."""
        settings = PrometheusSettings()
        assert settings.retention_time == "15d"

    def test_remote_write_disabled_by_default(self) -> None:
        """Test remote write is disabled by default."""
        settings = PrometheusSettings()
        assert settings.remote_write_enabled is False


class TestStaticTarget:
    """Tests for StaticTarget dataclass."""

    def test_target_creation(self) -> None:
        """Test creating a static target."""
        target = StaticTarget("localhost:8080", {"service": "test"})
        assert target.address == "localhost:8080"
        assert target.labels["service"] == "test"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        target = StaticTarget("localhost:8080", {"service": "test"})
        result = target.to_dict()
        assert result["targets"] == ["localhost:8080"]
        assert result["labels"]["service"] == "test"


class TestRelabelConfig:
    """Tests for RelabelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = RelabelConfig()
        assert config.separator == ";"
        assert config.regex == "(.*)"
        assert config.replacement == "$1"
        assert config.action == MetricRelabelAction.REPLACE

    def test_to_dict_with_source_labels(self) -> None:
        """Test to_dict with source labels."""
        config = RelabelConfig(source_labels=["__address__"], target_label="instance")
        result = config.to_dict()
        assert result["source_labels"] == ["__address__"]
        assert result["target_label"] == "instance"


class TestScrapeJob:
    """Tests for ScrapeJob dataclass."""

    def test_job_creation(self) -> None:
        """Test creating a scrape job."""
        job = ScrapeJob("test-job")
        assert job.job_name == "test-job"
        assert job.scrape_interval == "15s"

    def test_job_with_targets(self) -> None:
        """Test job with static targets."""
        target = StaticTarget("localhost:8080")
        job = ScrapeJob("test-job", static_targets=[target])
        assert len(job.static_targets) == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        target = StaticTarget("localhost:8080")
        job = ScrapeJob("test-job", static_targets=[target])
        result = job.to_dict()
        assert result["job_name"] == "test-job"
        assert "static_configs" in result


class TestSolacePrometheusScrapeJobs:
    """Tests for SolacePrometheusScrapeJobs factory."""

    def test_orchestrator_job(self) -> None:
        """Test orchestrator job creation."""
        job = SolacePrometheusScrapeJobs.orchestrator_job()
        assert job.job_name == "solace-orchestrator"
        assert len(job.static_targets) > 0

    def test_safety_service_job(self) -> None:
        """Test safety service job with faster scrape."""
        job = SolacePrometheusScrapeJobs.safety_service_job()
        assert job.job_name == "solace-safety"
        assert job.scrape_interval == "5s"

    def test_all_jobs_returns_list(self) -> None:
        """Test all_jobs returns complete list."""
        jobs = SolacePrometheusScrapeJobs.all_jobs()
        assert len(jobs) >= 5
        job_names = [j.job_name for j in jobs]
        assert "solace-orchestrator" in job_names
        assert "solace-safety" in job_names

    def test_infrastructure_jobs(self) -> None:
        """Test infrastructure jobs."""
        jobs = SolacePrometheusScrapeJobs.infrastructure_jobs()
        assert len(jobs) == 4
        job_names = [j.job_name for j in jobs]
        assert "kafka" in job_names
        assert "redis" in job_names


class TestRemoteWriteConfig:
    """Tests for RemoteWriteConfig dataclass."""

    def test_remote_write_creation(self) -> None:
        """Test creating remote write config."""
        config = RemoteWriteConfig(url="http://cortex:9009/api/v1/push")
        assert config.url == "http://cortex:9009/api/v1/push"
        assert config.name == "cortex"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        config = RemoteWriteConfig(url="http://cortex:9009/api/v1/push")
        result = config.to_dict()
        assert result["url"] == "http://cortex:9009/api/v1/push"
        assert "queue_config" in result


class TestPrometheusConfigGenerator:
    """Tests for PrometheusConfigGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be initialized."""
        generator = PrometheusConfigGenerator()
        assert generator is not None

    def test_generate_global_config(self) -> None:
        """Test generate global config."""
        generator = PrometheusConfigGenerator()
        config = generator.generate_global_config()
        assert "scrape_interval" in config
        assert "external_labels" in config

    def test_generate_scrape_configs(self) -> None:
        """Test generate scrape configs."""
        generator = PrometheusConfigGenerator()
        configs = generator.generate_scrape_configs()
        assert len(configs) > 0
        assert all("job_name" in c for c in configs)

    def test_generate_full_config(self) -> None:
        """Test generate full config."""
        generator = PrometheusConfigGenerator()
        config = generator.generate_full_config()
        assert "global" in config
        assert "scrape_configs" in config


class TestCreatePrometheusConfig:
    """Tests for create_prometheus_config factory."""

    def test_create_config_returns_dict(self) -> None:
        """Test factory returns dictionary."""
        config = create_prometheus_config()
        assert isinstance(config, dict)

    def test_create_config_has_required_sections(self) -> None:
        """Test config has required sections."""
        config = create_prometheus_config()
        assert "global" in config
        assert "scrape_configs" in config

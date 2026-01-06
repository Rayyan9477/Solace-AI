"""Solace-AI Prometheus Configuration - Scrape targets, jobs, and service discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class ScrapeProtocol(str, Enum):
    """Supported scrape protocols."""
    HTTP = "http"
    HTTPS = "https"


class ServiceDiscoveryType(str, Enum):
    """Service discovery mechanisms."""
    STATIC = "static"
    KUBERNETES = "kubernetes"
    CONSUL = "consul"
    DNS = "dns"


class MetricRelabelAction(str, Enum):
    """Metric relabeling actions."""
    REPLACE = "replace"
    KEEP = "keep"
    DROP = "drop"
    LABELDROP = "labeldrop"
    LABELKEEP = "labelkeep"
    HASHMOD = "hashmod"


class PrometheusSettings(BaseSettings):
    """Prometheus configuration from environment."""
    global_scrape_interval: str = Field(default="15s")
    global_scrape_timeout: str = Field(default="10s")
    global_evaluation_interval: str = Field(default="15s")
    external_labels: dict[str, str] = Field(default_factory=lambda: {"cluster": "solace-ai", "env": "production"})
    remote_write_enabled: bool = Field(default=False)
    remote_write_url: str | None = Field(default=None)
    retention_time: str = Field(default="15d")
    retention_size: str = Field(default="50GB")
    model_config = SettingsConfigDict(env_prefix="PROMETHEUS_", env_file=".env", extra="ignore")


@dataclass
class StaticTarget:
    """Static scrape target configuration."""
    address: str
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"targets": [self.address], "labels": self.labels}


@dataclass
class RelabelConfig:
    """Prometheus relabeling configuration."""
    source_labels: list[str] = field(default_factory=list)
    separator: str = ";"
    target_label: str = ""
    regex: str = "(.*)"
    replacement: str = "$1"
    action: MetricRelabelAction = MetricRelabelAction.REPLACE

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"action": self.action.value}
        if self.source_labels:
            result["source_labels"] = self.source_labels
        if self.target_label:
            result["target_label"] = self.target_label
        if self.regex != "(.*)":
            result["regex"] = self.regex
        if self.replacement != "$1":
            result["replacement"] = self.replacement
        return result


@dataclass
class ScrapeJob:
    """Prometheus scrape job configuration."""
    job_name: str
    scrape_interval: str = "15s"
    scrape_timeout: str = "10s"
    scheme: ScrapeProtocol = ScrapeProtocol.HTTP
    metrics_path: str = "/metrics"
    static_targets: list[StaticTarget] = field(default_factory=list)
    relabel_configs: list[RelabelConfig] = field(default_factory=list)
    metric_relabel_configs: list[RelabelConfig] = field(default_factory=list)
    honor_labels: bool = False
    honor_timestamps: bool = True
    service_discovery: ServiceDiscoveryType = ServiceDiscoveryType.STATIC

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "job_name": self.job_name, "scrape_interval": self.scrape_interval,
            "scrape_timeout": self.scrape_timeout, "scheme": self.scheme.value,
            "metrics_path": self.metrics_path, "honor_labels": self.honor_labels,
            "honor_timestamps": self.honor_timestamps,
        }
        if self.static_targets:
            config["static_configs"] = [t.to_dict() for t in self.static_targets]
        if self.relabel_configs:
            config["relabel_configs"] = [r.to_dict() for r in self.relabel_configs]
        if self.metric_relabel_configs:
            config["metric_relabel_configs"] = [r.to_dict() for r in self.metric_relabel_configs]
        return config


class SolacePrometheusScrapeJobs:
    """Factory for Solace-AI Prometheus scrape job definitions."""

    @staticmethod
    def orchestrator_job() -> ScrapeJob:
        """Orchestrator service metrics."""
        return ScrapeJob(
            job_name="solace-orchestrator",
            static_targets=[StaticTarget("orchestrator:8000", {"service": "orchestrator", "component": "core"})],
            relabel_configs=[RelabelConfig(source_labels=["__address__"], target_label="instance",
                                          regex="([^:]+).*", replacement="${1}")],
        )

    @staticmethod
    def safety_service_job() -> ScrapeJob:
        """Safety service metrics with high priority."""
        return ScrapeJob(
            job_name="solace-safety", scrape_interval="5s",
            static_targets=[StaticTarget("safety-service:8001", {"service": "safety", "component": "critical"})],
        )

    @staticmethod
    def diagnosis_service_job() -> ScrapeJob:
        """Diagnosis service metrics."""
        return ScrapeJob(
            job_name="solace-diagnosis",
            static_targets=[StaticTarget("diagnosis-service:8002", {"service": "diagnosis", "component": "clinical"})],
        )

    @staticmethod
    def therapy_service_job() -> ScrapeJob:
        """Therapy service metrics."""
        return ScrapeJob(
            job_name="solace-therapy",
            static_targets=[StaticTarget("therapy-service:8003", {"service": "therapy", "component": "clinical"})],
        )

    @staticmethod
    def memory_service_job() -> ScrapeJob:
        """Memory service metrics."""
        return ScrapeJob(
            job_name="solace-memory",
            static_targets=[StaticTarget("memory-service:8004", {"service": "memory", "component": "storage"})],
        )

    @staticmethod
    def infrastructure_jobs() -> list[ScrapeJob]:
        """Infrastructure component metrics (Kafka, Redis, PostgreSQL, Weaviate)."""
        return [
            ScrapeJob("kafka", static_targets=[StaticTarget("kafka:9308", {"service": "kafka"})]),
            ScrapeJob("redis", static_targets=[StaticTarget("redis-exporter:9121", {"service": "redis"})]),
            ScrapeJob("postgres", static_targets=[StaticTarget("postgres-exporter:9187", {"service": "postgres"})]),
            ScrapeJob("weaviate", static_targets=[StaticTarget("weaviate:2112", {"service": "weaviate"})]),
        ]

    @classmethod
    def all_jobs(cls) -> list[ScrapeJob]:
        """All Solace-AI scrape jobs."""
        return [cls.orchestrator_job(), cls.safety_service_job(), cls.diagnosis_service_job(),
                cls.therapy_service_job(), cls.memory_service_job()] + cls.infrastructure_jobs()


@dataclass
class RemoteWriteConfig:
    """Prometheus remote write configuration for long-term storage."""
    url: str
    name: str = "cortex"
    remote_timeout: str = "30s"
    queue_capacity: int = 10000
    max_shards: int = 50
    min_shards: int = 1
    max_samples_per_send: int = 500
    batch_send_deadline: str = "5s"
    write_relabel_configs: list[RelabelConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "url": self.url, "name": self.name, "remote_timeout": self.remote_timeout,
            "queue_config": {"capacity": self.queue_capacity, "max_shards": self.max_shards,
                            "min_shards": self.min_shards, "max_samples_per_send": self.max_samples_per_send,
                            "batch_send_deadline": self.batch_send_deadline},
        }
        if self.write_relabel_configs:
            config["write_relabel_configs"] = [r.to_dict() for r in self.write_relabel_configs]
        return config


class PrometheusConfigGenerator:
    """Generates complete Prometheus configuration."""

    def __init__(self, settings: PrometheusSettings | None = None) -> None:
        self._settings = settings or PrometheusSettings()

    def generate_global_config(self) -> dict[str, Any]:
        """Generate global configuration section."""
        return {
            "scrape_interval": self._settings.global_scrape_interval,
            "scrape_timeout": self._settings.global_scrape_timeout,
            "evaluation_interval": self._settings.global_evaluation_interval,
            "external_labels": self._settings.external_labels,
        }

    def generate_scrape_configs(self, jobs: list[ScrapeJob] | None = None) -> list[dict[str, Any]]:
        """Generate scrape configurations for all jobs."""
        jobs = jobs or SolacePrometheusScrapeJobs.all_jobs()
        return [job.to_dict() for job in jobs]

    def generate_remote_write(self, config: RemoteWriteConfig | None = None) -> list[dict[str, Any]]:
        """Generate remote write configuration if enabled."""
        if not self._settings.remote_write_enabled or not self._settings.remote_write_url:
            return []
        cfg = config or RemoteWriteConfig(url=self._settings.remote_write_url)
        return [cfg.to_dict()]

    def generate_full_config(self, jobs: list[ScrapeJob] | None = None,
                            remote_write: RemoteWriteConfig | None = None) -> dict[str, Any]:
        """Generate complete Prometheus configuration."""
        config: dict[str, Any] = {"global": self.generate_global_config(),
                                   "scrape_configs": self.generate_scrape_configs(jobs)}
        remote = self.generate_remote_write(remote_write)
        if remote:
            config["remote_write"] = remote
        logger.info("prometheus_config_generated", jobs=len(config["scrape_configs"]),
                   remote_write=bool(remote))
        return config


def create_prometheus_config(settings: PrometheusSettings | None = None) -> dict[str, Any]:
    """Create default Prometheus configuration for Solace-AI."""
    generator = PrometheusConfigGenerator(settings)
    return generator.generate_full_config()

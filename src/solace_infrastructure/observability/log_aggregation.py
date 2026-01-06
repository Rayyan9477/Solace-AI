"""Solace-AI Log Aggregation - ELK/Loki pipeline, shipping, and retention configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class LogBackend(str, Enum):
    """Supported log aggregation backends."""
    LOKI = "loki"
    ELASTICSEARCH = "elasticsearch"


class LogAggregationLevel(str, Enum):
    """Log levels for aggregation filtering."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetentionTier(str, Enum):
    """Log retention tiers for lifecycle management."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DELETE = "delete"


class LogAggregationSettings(BaseSettings):
    """Log aggregation configuration from environment."""
    backend: LogBackend = Field(default=LogBackend.LOKI)
    loki_url: str = Field(default="http://loki:3100")
    elasticsearch_url: str = Field(default="http://elasticsearch:9200")
    retention_days_hot: int = Field(default=7, ge=1)
    retention_days_warm: int = Field(default=30, ge=1)
    retention_days_cold: int = Field(default=90, ge=1)
    batch_size: int = Field(default=1000, ge=100)
    batch_wait_seconds: int = Field(default=1, ge=1)
    max_retries: int = Field(default=3, ge=1)
    index_prefix: str = Field(default="solace")
    enable_structured_metadata: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="LOG_AGGREGATION_", env_file=".env", extra="ignore")


@dataclass
class LogLabel:
    """Label for log stream identification."""
    name: str
    value: str
    static: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "static": self.static}


@dataclass
class LogPipeline:
    """Log processing pipeline configuration."""
    name: str
    match_labels: dict[str, str]
    stages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "selector": self.match_labels, "stages": self.stages}


@dataclass
class RetentionPolicy:
    """Log retention policy configuration."""
    name: str
    tier: RetentionTier
    min_age: str
    max_age: str
    delete_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        policy: dict[str, Any] = {"name": self.name, "tier": self.tier.value,
                                   "min_age": self.min_age, "max_age": self.max_age}
        if self.delete_after:
            policy["delete_after"] = self.delete_after
        return policy


class SolaceLogPipelines:
    """Factory for Solace-AI log pipeline definitions."""

    @staticmethod
    def json_parser_stage() -> dict[str, Any]:
        """JSON parsing stage for structured logs."""
        return {"json": {"expressions": {"level": "level", "message": "event", "service": "service",
                                         "correlation_id": "correlation_id", "timestamp": "timestamp"}}}

    @staticmethod
    def timestamp_stage() -> dict[str, Any]:
        """Timestamp extraction stage."""
        return {"timestamp": {"source": "timestamp", "format": "RFC3339Nano"}}

    @staticmethod
    def labels_stage() -> dict[str, Any]:
        """Label extraction from parsed JSON."""
        return {"labels": {"level": "", "service": "", "correlation_id": ""}}

    @staticmethod
    def application_logs_pipeline() -> LogPipeline:
        """Application logs processing pipeline."""
        return LogPipeline(
            name="solace-application", match_labels={"job": "solace-*"},
            stages=[SolaceLogPipelines.json_parser_stage(), SolaceLogPipelines.timestamp_stage(),
                    SolaceLogPipelines.labels_stage()],
        )

    @staticmethod
    def safety_logs_pipeline() -> LogPipeline:
        """Safety service logs - high priority with extended retention."""
        return LogPipeline(
            name="solace-safety", match_labels={"job": "solace-safety"},
            stages=[SolaceLogPipelines.json_parser_stage(), SolaceLogPipelines.timestamp_stage(),
                    {"labels": {"level": "", "service": "", "correlation_id": "", "severity": "", "crisis_id": ""}}],
        )

    @staticmethod
    def audit_logs_pipeline() -> LogPipeline:
        """Audit logs - HIPAA compliance, never deleted."""
        return LogPipeline(
            name="solace-audit", match_labels={"job": "solace-audit"},
            stages=[SolaceLogPipelines.json_parser_stage(),
                    {"labels": {"action": "", "user_id": "", "resource": "", "result": ""}}],
        )

    @classmethod
    def all_pipelines(cls) -> list[LogPipeline]:
        """All Solace-AI log pipelines."""
        return [cls.application_logs_pipeline(), cls.safety_logs_pipeline(), cls.audit_logs_pipeline()]


@dataclass
class LokiConfig:
    """Loki-specific configuration."""
    url: str
    tenant_id: str = "solace"
    batch_size: int = 1000
    batch_wait: str = "1s"
    external_labels: dict[str, str] = field(default_factory=lambda: {"cluster": "solace-ai"})

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "tenant_id": self.tenant_id,
                "batchwait": self.batch_wait, "batchsize": self.batch_size,
                "external_labels": self.external_labels}


@dataclass
class ElasticsearchConfig:
    """Elasticsearch-specific configuration."""
    url: str
    index_prefix: str = "solace"
    number_of_shards: int = 5
    number_of_replicas: int = 1
    refresh_interval: str = "5s"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "index_prefix": self.index_prefix,
                "settings": {"number_of_shards": self.number_of_shards,
                            "number_of_replicas": self.number_of_replicas,
                            "refresh_interval": self.refresh_interval}}


@dataclass
class IndexTemplate:
    """Elasticsearch index template for log indices."""
    name: str
    index_patterns: list[str]
    settings: dict[str, Any] = field(default_factory=dict)
    mappings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "index_patterns": self.index_patterns,
                "template": {"settings": self.settings, "mappings": self.mappings}}


class SolaceIndexTemplates:
    """Factory for Solace-AI Elasticsearch index templates."""

    @staticmethod
    def application_logs_template() -> IndexTemplate:
        """Application logs index template."""
        return IndexTemplate(
            name="solace-logs", index_patterns=["solace-logs-*"],
            settings={"number_of_shards": 5, "number_of_replicas": 1, "index.lifecycle.name": "solace-logs-policy"},
            mappings={"properties": {
                "timestamp": {"type": "date"}, "level": {"type": "keyword"}, "service": {"type": "keyword"},
                "message": {"type": "text"}, "correlation_id": {"type": "keyword"},
            }},
        )

    @staticmethod
    def audit_logs_template() -> IndexTemplate:
        """Audit logs index template - HIPAA compliant, immutable."""
        return IndexTemplate(
            name="solace-audit", index_patterns=["solace-audit-*"],
            settings={"number_of_shards": 3, "number_of_replicas": 2,
                     "index.lifecycle.name": "solace-audit-policy", "index.blocks.write": False},
            mappings={"properties": {
                "timestamp": {"type": "date"}, "action": {"type": "keyword"}, "user_id": {"type": "keyword"},
                "resource": {"type": "keyword"}, "result": {"type": "keyword"}, "details": {"type": "object"},
            }},
        )

    @classmethod
    def all_templates(cls) -> list[IndexTemplate]:
        """All Solace-AI index templates."""
        return [cls.application_logs_template(), cls.audit_logs_template()]


class LogAggregationConfigGenerator:
    """Generates complete log aggregation configuration."""

    def __init__(self, settings: LogAggregationSettings | None = None) -> None:
        self._settings = settings or LogAggregationSettings()

    def generate_retention_policies(self) -> list[RetentionPolicy]:
        """Generate retention policies based on settings."""
        return [
            RetentionPolicy("hot", RetentionTier.HOT, "0d", f"{self._settings.retention_days_hot}d"),
            RetentionPolicy("warm", RetentionTier.WARM, f"{self._settings.retention_days_hot}d",
                           f"{self._settings.retention_days_warm}d"),
            RetentionPolicy("cold", RetentionTier.COLD, f"{self._settings.retention_days_warm}d",
                           f"{self._settings.retention_days_cold}d", f"{self._settings.retention_days_cold}d"),
        ]

    def generate_loki_config(self) -> dict[str, Any]:
        """Generate Loki configuration."""
        loki = LokiConfig(url=self._settings.loki_url)
        pipelines = SolaceLogPipelines.all_pipelines()
        return {"client": loki.to_dict(), "pipelines": [p.to_dict() for p in pipelines],
                "retention": [r.to_dict() for r in self.generate_retention_policies()]}

    def generate_elasticsearch_config(self) -> dict[str, Any]:
        """Generate Elasticsearch configuration."""
        es = ElasticsearchConfig(url=self._settings.elasticsearch_url, index_prefix=self._settings.index_prefix)
        templates = SolaceIndexTemplates.all_templates()
        return {"client": es.to_dict(), "index_templates": [t.to_dict() for t in templates],
                "ilm_policies": self._generate_ilm_policies()}

    def _generate_ilm_policies(self) -> list[dict[str, Any]]:
        """Generate ILM policies for Elasticsearch."""
        return [
            {"name": "solace-logs-policy", "phases": {
                "hot": {"min_age": "0ms", "actions": {"rollover": {"max_size": "50gb", "max_age": "1d"}}},
                "warm": {"min_age": f"{self._settings.retention_days_hot}d", "actions": {"shrink": {"number_of_shards": 1}}},
                "delete": {"min_age": f"{self._settings.retention_days_cold}d", "actions": {"delete": {}}},
            }},
            {"name": "solace-audit-policy", "phases": {
                "hot": {"min_age": "0ms", "actions": {"rollover": {"max_size": "10gb", "max_age": "30d"}}},
            }},
        ]

    def generate_full_config(self) -> dict[str, Any]:
        """Generate complete log aggregation configuration."""
        config: dict[str, Any] = {"backend": self._settings.backend.value}
        if self._settings.backend == LogBackend.LOKI:
            config["loki"] = self.generate_loki_config()
        else:
            config["elasticsearch"] = self.generate_elasticsearch_config()
        logger.info("log_aggregation_config_generated", backend=self._settings.backend.value,
                   retention_cold_days=self._settings.retention_days_cold)
        return config


def create_log_aggregation_config(settings: LogAggregationSettings | None = None) -> dict[str, Any]:
    """Create default log aggregation configuration for Solace-AI."""
    generator = LogAggregationConfigGenerator(settings)
    return generator.generate_full_config()

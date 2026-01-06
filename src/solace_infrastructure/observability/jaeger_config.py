"""Solace-AI Jaeger Configuration - Distributed tracing setup and sampling strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class SamplingType(str, Enum):
    """Jaeger sampling strategy types."""
    CONST = "const"
    PROBABILISTIC = "probabilistic"
    RATE_LIMITING = "ratelimiting"
    REMOTE = "remote"


class SpanStorageType(str, Enum):
    """Jaeger span storage backends."""
    MEMORY = "memory"
    ELASTICSEARCH = "elasticsearch"
    CASSANDRA = "cassandra"
    BADGER = "badger"
    GRPC = "grpc"


class JaegerSettings(BaseSettings):
    """Jaeger configuration from environment."""
    service_name: str = Field(default="solace-ai")
    agent_host: str = Field(default="jaeger-agent")
    agent_port: int = Field(default=6831, ge=1, le=65535)
    collector_endpoint: str | None = Field(default=None)
    sampling_type: SamplingType = Field(default=SamplingType.PROBABILISTIC)
    sampling_param: float = Field(default=0.1, ge=0.0, le=1.0)
    propagation_format: str = Field(default="jaeger")
    reporter_flush_interval: int = Field(default=1000, ge=100)
    reporter_queue_size: int = Field(default=100, ge=10)
    storage_type: SpanStorageType = Field(default=SpanStorageType.ELASTICSEARCH)
    es_server_urls: str = Field(default="http://elasticsearch:9200")
    span_retention_days: int = Field(default=7, ge=1)
    model_config = SettingsConfigDict(env_prefix="JAEGER_", env_file=".env", extra="ignore")


@dataclass
class SamplingStrategy:
    """Jaeger sampling strategy configuration."""
    strategy_type: SamplingType
    param: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.strategy_type.value, "param": self.param}


@dataclass
class OperationSampling:
    """Per-operation sampling configuration."""
    operation: str
    strategy: SamplingStrategy

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, **self.strategy.to_dict()}


@dataclass
class ServiceSamplingConfig:
    """Service-level sampling configuration."""
    service: str
    default_strategy: SamplingStrategy
    operation_strategies: list[OperationSampling] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {"service": self.service, "default_strategy": self.default_strategy.to_dict()}
        if self.operation_strategies:
            config["operation_strategies"] = [o.to_dict() for o in self.operation_strategies]
        return config


class SolaceTracingConfig:
    """Factory for Solace-AI tracing configurations."""

    @staticmethod
    def orchestrator_sampling() -> ServiceSamplingConfig:
        """Orchestrator service sampling - moderate rate."""
        return ServiceSamplingConfig(
            service="solace-orchestrator",
            default_strategy=SamplingStrategy(SamplingType.PROBABILISTIC, 0.1),
            operation_strategies=[
                OperationSampling("process_message", SamplingStrategy(SamplingType.PROBABILISTIC, 0.2)),
                OperationSampling("route_to_agent", SamplingStrategy(SamplingType.PROBABILISTIC, 0.15)),
            ],
        )

    @staticmethod
    def safety_sampling() -> ServiceSamplingConfig:
        """Safety service sampling - HIGH rate for critical monitoring."""
        return ServiceSamplingConfig(
            service="solace-safety",
            default_strategy=SamplingStrategy(SamplingType.CONST, 1.0),
            operation_strategies=[
                OperationSampling("crisis_detection", SamplingStrategy(SamplingType.CONST, 1.0)),
                OperationSampling("escalation", SamplingStrategy(SamplingType.CONST, 1.0)),
            ],
        )

    @staticmethod
    def memory_sampling() -> ServiceSamplingConfig:
        """Memory service sampling - lower rate for high-volume ops."""
        return ServiceSamplingConfig(
            service="solace-memory",
            default_strategy=SamplingStrategy(SamplingType.PROBABILISTIC, 0.05),
            operation_strategies=[
                OperationSampling("context_assembly", SamplingStrategy(SamplingType.PROBABILISTIC, 0.1)),
                OperationSampling("vector_search", SamplingStrategy(SamplingType.PROBABILISTIC, 0.05)),
            ],
        )

    @staticmethod
    def llm_sampling() -> ServiceSamplingConfig:
        """LLM service sampling - moderate for cost tracking."""
        return ServiceSamplingConfig(
            service="solace-llm",
            default_strategy=SamplingStrategy(SamplingType.PROBABILISTIC, 0.2),
        )

    @classmethod
    def all_services(cls) -> list[ServiceSamplingConfig]:
        """All Solace-AI service sampling configurations."""
        return [cls.orchestrator_sampling(), cls.safety_sampling(), cls.memory_sampling(), cls.llm_sampling()]


@dataclass
class CollectorConfig:
    """Jaeger collector configuration."""
    grpc_port: int = 14250
    http_port: int = 14268
    zipkin_port: int = 9411
    queue_size: int = 2000
    num_workers: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {"grpc_port": self.grpc_port, "http_port": self.http_port, "zipkin_port": self.zipkin_port,
                "queue_size": self.queue_size, "num_workers": self.num_workers}


@dataclass
class AgentConfig:
    """Jaeger agent configuration."""
    host: str = "0.0.0.0"
    compact_port: int = 6831
    binary_port: int = 6832
    http_port: int = 5778
    processor_queue_size: int = 1000
    processor_workers: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "compact_port": self.compact_port, "binary_port": self.binary_port,
                "http_port": self.http_port, "processor_queue_size": self.processor_queue_size,
                "processor_workers": self.processor_workers}


class JaegerConfigGenerator:
    """Generates complete Jaeger configuration."""

    def __init__(self, settings: JaegerSettings | None = None) -> None:
        self._settings = settings or JaegerSettings()

    def generate_sampling_config(self, services: list[ServiceSamplingConfig] | None = None) -> dict[str, Any]:
        """Generate sampling configuration for all services."""
        services = services or SolaceTracingConfig.all_services()
        return {"default_strategy": {"type": SamplingType.PROBABILISTIC.value, "param": 0.1},
                "services": {s.service: s.to_dict() for s in services}}

    def generate_collector_config(self, config: CollectorConfig | None = None) -> dict[str, Any]:
        """Generate collector configuration."""
        cfg = config or CollectorConfig()
        return cfg.to_dict()

    def generate_agent_config(self, config: AgentConfig | None = None) -> dict[str, Any]:
        """Generate agent configuration."""
        cfg = config or AgentConfig()
        return cfg.to_dict()

    def generate_storage_config(self) -> dict[str, Any]:
        """Generate storage configuration based on settings."""
        base: dict[str, Any] = {"type": self._settings.storage_type.value}
        if self._settings.storage_type == SpanStorageType.ELASTICSEARCH:
            base["elasticsearch"] = {"server_urls": self._settings.es_server_urls,
                                      "index_prefix": "jaeger", "num_shards": 5, "num_replicas": 1,
                                      "max_span_age": f"{self._settings.span_retention_days}d"}
        return base

    def generate_full_config(self, services: list[ServiceSamplingConfig] | None = None,
                            collector: CollectorConfig | None = None,
                            agent: AgentConfig | None = None) -> dict[str, Any]:
        """Generate complete Jaeger configuration."""
        config = {
            "service_name": self._settings.service_name,
            "sampling": self.generate_sampling_config(services),
            "collector": self.generate_collector_config(collector),
            "agent": self.generate_agent_config(agent),
            "storage": self.generate_storage_config(),
            "reporter": {"flush_interval": self._settings.reporter_flush_interval,
                        "queue_size": self._settings.reporter_queue_size},
        }
        logger.info("jaeger_config_generated", service=self._settings.service_name,
                   sampling_type=self._settings.sampling_type.value)
        return config


def create_jaeger_config(settings: JaegerSettings | None = None) -> dict[str, Any]:
    """Create default Jaeger configuration for Solace-AI."""
    generator = JaegerConfigGenerator(settings)
    return generator.generate_full_config()

"""Unit tests for Jaeger configuration module."""
from __future__ import annotations

import pytest

from solace_infrastructure.observability.jaeger_config import (
    SamplingType,
    SpanStorageType,
    JaegerSettings,
    SamplingStrategy,
    OperationSampling,
    ServiceSamplingConfig,
    SolaceTracingConfig,
    CollectorConfig,
    AgentConfig,
    JaegerConfigGenerator,
    create_jaeger_config,
)


class TestSamplingType:
    """Tests for SamplingType enum."""

    def test_const_value(self) -> None:
        """Test const sampling type."""
        assert SamplingType.CONST.value == "const"

    def test_probabilistic_value(self) -> None:
        """Test probabilistic sampling type."""
        assert SamplingType.PROBABILISTIC.value == "probabilistic"

    def test_rate_limiting_value(self) -> None:
        """Test rate limiting sampling type."""
        assert SamplingType.RATE_LIMITING.value == "ratelimiting"


class TestSpanStorageType:
    """Tests for SpanStorageType enum."""

    def test_memory_value(self) -> None:
        """Test memory storage type."""
        assert SpanStorageType.MEMORY.value == "memory"

    def test_elasticsearch_value(self) -> None:
        """Test elasticsearch storage type."""
        assert SpanStorageType.ELASTICSEARCH.value == "elasticsearch"


class TestJaegerSettings:
    """Tests for JaegerSettings."""

    def test_default_service_name(self) -> None:
        """Test default service name."""
        settings = JaegerSettings()
        assert settings.service_name == "solace-ai"

    def test_default_agent_port(self) -> None:
        """Test default agent port."""
        settings = JaegerSettings()
        assert settings.agent_port == 6831

    def test_default_sampling_type(self) -> None:
        """Test default sampling type."""
        settings = JaegerSettings()
        assert settings.sampling_type == SamplingType.PROBABILISTIC

    def test_default_sampling_param(self) -> None:
        """Test default sampling param."""
        settings = JaegerSettings()
        assert settings.sampling_param == 0.1


class TestSamplingStrategy:
    """Tests for SamplingStrategy dataclass."""

    def test_strategy_creation(self) -> None:
        """Test creating a sampling strategy."""
        strategy = SamplingStrategy(SamplingType.PROBABILISTIC, 0.5)
        assert strategy.strategy_type == SamplingType.PROBABILISTIC
        assert strategy.param == 0.5

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        strategy = SamplingStrategy(SamplingType.CONST, 1.0)
        result = strategy.to_dict()
        assert result["type"] == "const"
        assert result["param"] == 1.0


class TestOperationSampling:
    """Tests for OperationSampling dataclass."""

    def test_operation_creation(self) -> None:
        """Test creating operation sampling."""
        strategy = SamplingStrategy(SamplingType.PROBABILISTIC, 0.1)
        operation = OperationSampling("test_op", strategy)
        assert operation.operation == "test_op"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        strategy = SamplingStrategy(SamplingType.PROBABILISTIC, 0.1)
        operation = OperationSampling("test_op", strategy)
        result = operation.to_dict()
        assert result["operation"] == "test_op"
        assert result["type"] == "probabilistic"


class TestServiceSamplingConfig:
    """Tests for ServiceSamplingConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating service config."""
        strategy = SamplingStrategy(SamplingType.PROBABILISTIC, 0.1)
        config = ServiceSamplingConfig("test-service", strategy)
        assert config.service == "test-service"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        strategy = SamplingStrategy(SamplingType.PROBABILISTIC, 0.1)
        config = ServiceSamplingConfig("test-service", strategy)
        result = config.to_dict()
        assert result["service"] == "test-service"
        assert "default_strategy" in result


class TestSolaceTracingConfig:
    """Tests for SolaceTracingConfig factory."""

    def test_orchestrator_sampling(self) -> None:
        """Test orchestrator sampling config."""
        config = SolaceTracingConfig.orchestrator_sampling()
        assert config.service == "solace-orchestrator"

    def test_safety_sampling_full_rate(self) -> None:
        """Test safety service has full sampling."""
        config = SolaceTracingConfig.safety_sampling()
        assert config.service == "solace-safety"
        assert config.default_strategy.param == 1.0

    def test_memory_sampling_low_rate(self) -> None:
        """Test memory service has low sampling."""
        config = SolaceTracingConfig.memory_sampling()
        assert config.service == "solace-memory"
        assert config.default_strategy.param == 0.05

    def test_all_services(self) -> None:
        """Test all_services returns complete list."""
        configs = SolaceTracingConfig.all_services()
        assert len(configs) == 4
        services = [c.service for c in configs]
        assert "solace-safety" in services


class TestCollectorConfig:
    """Tests for CollectorConfig dataclass."""

    def test_default_ports(self) -> None:
        """Test default collector ports."""
        config = CollectorConfig()
        assert config.grpc_port == 14250
        assert config.http_port == 14268

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        config = CollectorConfig()
        result = config.to_dict()
        assert result["grpc_port"] == 14250


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_ports(self) -> None:
        """Test default agent ports."""
        config = AgentConfig()
        assert config.compact_port == 6831
        assert config.binary_port == 6832

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        config = AgentConfig()
        result = config.to_dict()
        assert result["compact_port"] == 6831


class TestJaegerConfigGenerator:
    """Tests for JaegerConfigGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be initialized."""
        generator = JaegerConfigGenerator()
        assert generator is not None

    def test_generate_sampling_config(self) -> None:
        """Test generate sampling config."""
        generator = JaegerConfigGenerator()
        config = generator.generate_sampling_config()
        assert "default_strategy" in config
        assert "services" in config

    def test_generate_collector_config(self) -> None:
        """Test generate collector config."""
        generator = JaegerConfigGenerator()
        config = generator.generate_collector_config()
        assert "grpc_port" in config

    def test_generate_storage_config(self) -> None:
        """Test generate storage config."""
        generator = JaegerConfigGenerator()
        config = generator.generate_storage_config()
        assert "type" in config

    def test_generate_full_config(self) -> None:
        """Test generate full config."""
        generator = JaegerConfigGenerator()
        config = generator.generate_full_config()
        assert "sampling" in config
        assert "collector" in config
        assert "storage" in config


class TestCreateJaegerConfig:
    """Tests for create_jaeger_config factory."""

    def test_create_returns_dict(self) -> None:
        """Test factory returns dictionary."""
        config = create_jaeger_config()
        assert isinstance(config, dict)

    def test_create_has_required_sections(self) -> None:
        """Test config has required sections."""
        config = create_jaeger_config()
        assert "service_name" in config
        assert "sampling" in config
        assert "collector" in config

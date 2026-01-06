"""Unit tests for log aggregation module."""
from __future__ import annotations

import pytest

from solace_infrastructure.observability.log_aggregation import (
    LogBackend,
    LogAggregationLevel,
    RetentionTier,
    LogAggregationSettings,
    LogLabel,
    LogPipeline,
    RetentionPolicy,
    SolaceLogPipelines,
    LokiConfig,
    ElasticsearchConfig,
    IndexTemplate,
    SolaceIndexTemplates,
    LogAggregationConfigGenerator,
    create_log_aggregation_config,
)


class TestLogBackend:
    """Tests for LogBackend enum."""

    def test_loki_value(self) -> None:
        """Test loki backend."""
        assert LogBackend.LOKI.value == "loki"

    def test_elasticsearch_value(self) -> None:
        """Test elasticsearch backend."""
        assert LogBackend.ELASTICSEARCH.value == "elasticsearch"


class TestLogAggregationLevel:
    """Tests for LogAggregationLevel enum."""

    def test_debug_value(self) -> None:
        """Test debug level."""
        assert LogAggregationLevel.DEBUG.value == "debug"

    def test_error_value(self) -> None:
        """Test error level."""
        assert LogAggregationLevel.ERROR.value == "error"


class TestRetentionTier:
    """Tests for RetentionTier enum."""

    def test_hot_value(self) -> None:
        """Test hot tier."""
        assert RetentionTier.HOT.value == "hot"

    def test_warm_value(self) -> None:
        """Test warm tier."""
        assert RetentionTier.WARM.value == "warm"

    def test_cold_value(self) -> None:
        """Test cold tier."""
        assert RetentionTier.COLD.value == "cold"


class TestLogAggregationSettings:
    """Tests for LogAggregationSettings."""

    def test_default_backend(self) -> None:
        """Test default backend is Loki."""
        settings = LogAggregationSettings()
        assert settings.backend == LogBackend.LOKI

    def test_default_retention_hot(self) -> None:
        """Test default hot retention."""
        settings = LogAggregationSettings()
        assert settings.retention_days_hot == 7

    def test_default_retention_cold(self) -> None:
        """Test default cold retention."""
        settings = LogAggregationSettings()
        assert settings.retention_days_cold == 90

    def test_default_batch_size(self) -> None:
        """Test default batch size."""
        settings = LogAggregationSettings()
        assert settings.batch_size == 1000


class TestLogLabel:
    """Tests for LogLabel dataclass."""

    def test_label_creation(self) -> None:
        """Test creating a log label."""
        label = LogLabel("service", "solace-api")
        assert label.name == "service"
        assert label.value == "solace-api"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        label = LogLabel("service", "solace-api", static=True)
        result = label.to_dict()
        assert result["name"] == "service"
        assert result["static"] is True


class TestLogPipeline:
    """Tests for LogPipeline dataclass."""

    def test_pipeline_creation(self) -> None:
        """Test creating a log pipeline."""
        pipeline = LogPipeline("test", {"job": "test"})
        assert pipeline.name == "test"
        assert pipeline.match_labels["job"] == "test"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        pipeline = LogPipeline("test", {"job": "test"}, [{"json": {}}])
        result = pipeline.to_dict()
        assert result["name"] == "test"
        assert "selector" in result
        assert len(result["stages"]) == 1


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_policy_creation(self) -> None:
        """Test creating a retention policy."""
        policy = RetentionPolicy("hot", RetentionTier.HOT, "0d", "7d")
        assert policy.name == "hot"
        assert policy.tier == RetentionTier.HOT

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        policy = RetentionPolicy("hot", RetentionTier.HOT, "0d", "7d")
        result = policy.to_dict()
        assert result["name"] == "hot"
        assert result["tier"] == "hot"


class TestSolaceLogPipelines:
    """Tests for SolaceLogPipelines factory."""

    def test_json_parser_stage(self) -> None:
        """Test JSON parser stage."""
        stage = SolaceLogPipelines.json_parser_stage()
        assert "json" in stage

    def test_application_logs_pipeline(self) -> None:
        """Test application logs pipeline."""
        pipeline = SolaceLogPipelines.application_logs_pipeline()
        assert pipeline.name == "solace-application"
        assert len(pipeline.stages) > 0

    def test_safety_logs_pipeline(self) -> None:
        """Test safety logs pipeline."""
        pipeline = SolaceLogPipelines.safety_logs_pipeline()
        assert pipeline.name == "solace-safety"

    def test_audit_logs_pipeline(self) -> None:
        """Test audit logs pipeline."""
        pipeline = SolaceLogPipelines.audit_logs_pipeline()
        assert pipeline.name == "solace-audit"

    def test_all_pipelines(self) -> None:
        """Test all_pipelines returns complete list."""
        pipelines = SolaceLogPipelines.all_pipelines()
        assert len(pipelines) == 3
        names = [p.name for p in pipelines]
        assert "solace-audit" in names


class TestLokiConfig:
    """Tests for LokiConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating Loki config."""
        config = LokiConfig(url="http://loki:3100")
        assert config.url == "http://loki:3100"
        assert config.tenant_id == "solace"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        config = LokiConfig(url="http://loki:3100")
        result = config.to_dict()
        assert result["url"] == "http://loki:3100"
        assert "external_labels" in result


class TestElasticsearchConfig:
    """Tests for ElasticsearchConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating Elasticsearch config."""
        config = ElasticsearchConfig(url="http://es:9200")
        assert config.url == "http://es:9200"
        assert config.index_prefix == "solace"

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        config = ElasticsearchConfig(url="http://es:9200")
        result = config.to_dict()
        assert result["url"] == "http://es:9200"
        assert "settings" in result


class TestIndexTemplate:
    """Tests for IndexTemplate dataclass."""

    def test_template_creation(self) -> None:
        """Test creating an index template."""
        template = IndexTemplate("test", ["test-*"])
        assert template.name == "test"
        assert template.index_patterns == ["test-*"]

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        template = IndexTemplate("test", ["test-*"], {"number_of_shards": 3})
        result = template.to_dict()
        assert result["name"] == "test"
        assert "template" in result


class TestSolaceIndexTemplates:
    """Tests for SolaceIndexTemplates factory."""

    def test_application_logs_template(self) -> None:
        """Test application logs template."""
        template = SolaceIndexTemplates.application_logs_template()
        assert template.name == "solace-logs"
        assert "solace-logs-*" in template.index_patterns

    def test_audit_logs_template(self) -> None:
        """Test audit logs template."""
        template = SolaceIndexTemplates.audit_logs_template()
        assert template.name == "solace-audit"

    def test_all_templates(self) -> None:
        """Test all_templates returns complete list."""
        templates = SolaceIndexTemplates.all_templates()
        assert len(templates) == 2


class TestLogAggregationConfigGenerator:
    """Tests for LogAggregationConfigGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test generator can be initialized."""
        generator = LogAggregationConfigGenerator()
        assert generator is not None

    def test_generate_retention_policies(self) -> None:
        """Test generate retention policies."""
        generator = LogAggregationConfigGenerator()
        policies = generator.generate_retention_policies()
        assert len(policies) == 3
        tiers = [p.tier for p in policies]
        assert RetentionTier.HOT in tiers

    def test_generate_loki_config(self) -> None:
        """Test generate Loki config."""
        generator = LogAggregationConfigGenerator()
        config = generator.generate_loki_config()
        assert "client" in config
        assert "pipelines" in config

    def test_generate_elasticsearch_config(self) -> None:
        """Test generate Elasticsearch config."""
        settings = LogAggregationSettings(backend=LogBackend.ELASTICSEARCH)
        generator = LogAggregationConfigGenerator(settings)
        config = generator.generate_elasticsearch_config()
        assert "client" in config
        assert "index_templates" in config

    def test_generate_full_config_loki(self) -> None:
        """Test generate full config for Loki."""
        generator = LogAggregationConfigGenerator()
        config = generator.generate_full_config()
        assert config["backend"] == "loki"
        assert "loki" in config

    def test_generate_full_config_elasticsearch(self) -> None:
        """Test generate full config for Elasticsearch."""
        settings = LogAggregationSettings(backend=LogBackend.ELASTICSEARCH)
        generator = LogAggregationConfigGenerator(settings)
        config = generator.generate_full_config()
        assert config["backend"] == "elasticsearch"
        assert "elasticsearch" in config


class TestCreateLogAggregationConfig:
    """Tests for create_log_aggregation_config factory."""

    def test_create_returns_dict(self) -> None:
        """Test factory returns dictionary."""
        config = create_log_aggregation_config()
        assert isinstance(config, dict)

    def test_create_has_backend(self) -> None:
        """Test config has backend field."""
        config = create_log_aggregation_config()
        assert "backend" in config

"""Solace-AI Infrastructure Library - Database clients, caching, and observability utilities.

Exports are lazy (PEP 562): ``import solace_infrastructure`` loads nothing; a name is
imported from its submodule on first attribute access.
"""
from __future__ import annotations

import importlib
from typing import Any

# module -> public names re-exported unchanged
_MODULE_NAMES: dict[str, tuple[str, ...]] = {
    "solace_infrastructure.kafka": (
        # Topics
        "CleanupPolicy", "CompressionCodec", "TimestampType", "TopicPriority", "TopicDefinition",
        "TopicAdminSettings", "TopicMetadata", "TopicOperationResult", "TopicValidator",
        "KafkaAdminAdapter", "TopicManager", "create_topic_manager",
        # Schemas
        "SchemaType", "CompatibilityLevel", "SchemaFormat", "SchemaVersion", "SchemaMetadata",
        "CompatibilityResult", "SchemaRegistrySettings", "SchemaDefinition", "SchemaValidator",
        "JsonSchemaValidator", "AvroSchemaValidator", "SchemaCache", "SchemaRegistryAdapter",
        "SchemaManager", "create_schema_manager",
        # Partitioning
        "PartitionStrategy", "PartitionResult", "PartitionerConfig", "Partitioner",
        "RoundRobinPartitioner", "HashKeyPartitioner", "UserAffinityPartitioner",
        "PriorityPartitioner", "StickyPartitioner", "PartitionerFactory", "TopicPartitionRouter",
        "create_partition_router",
        # Retention
        "RetentionType", "ComplianceCategory", "RetentionPriority", "RetentionMetrics",
        "RetentionPolicy", "TopicRetentionAssignment", "RetentionManager", "PRESET_POLICIES",
        "create_retention_manager", "get_hipaa_policy",
        # Monitoring
        "AlertSeverity", "BrokerMetrics", "ConsumerGroupMetrics", "ConsumerLag", "KafkaAlert",
        "ClusterHealth", "MonitoringSettings", "MetricsCollector", "KafkaMonitorAdapter",
        "KafkaMonitor", "create_kafka_monitor",
    ),
    "solace_infrastructure.postgres": (
        "PostgresClient", "PostgresRepository", "PostgresSettings", "IsolationLevel", "QueryResult",
        "create_postgres_client",
    ),
    "solace_infrastructure.redis": (
        "RedisClient", "RedisSettings", "RedisMode", "CacheEntry", "create_redis_client",
    ),
    "solace_infrastructure.weaviate": (
        "WeaviateClient", "WeaviateSettings", "CollectionConfig", "PropertyConfig", "PropertyDataType",
        "VectorDistanceMetric", "SearchResult", "create_weaviate_client",
    ),
    "solace_infrastructure.health": (
        "HealthMonitor", "HealthChecker", "ClientHealthChecker", "CallableHealthChecker",
        "HealthCheckResult", "ComponentHealth", "HealthStatus", "ComponentType", "HealthCheckable",
        "create_health_monitor",
    ),
    "solace_infrastructure.observability_core": (
        "ObservabilitySettings", "LogLevel", "MetricType", "MetricValue", "MetricsRegistry", "Span",
        "Tracer", "configure_logging", "get_correlation_id", "set_correlation_id", "get_trace_context",
        "set_trace_context", "add_log_context", "get_metrics_registry", "get_tracer", "traced", "timed",
        "counted",
    ),
    "solace_infrastructure.observability": (
        # Prometheus
        "ScrapeProtocol", "ServiceDiscoveryType", "MetricRelabelAction", "PrometheusSettings",
        "StaticTarget", "RelabelConfig", "ScrapeJob", "SolacePrometheusScrapeJobs", "RemoteWriteConfig",
        "PrometheusConfigGenerator", "create_prometheus_config",
        # Grafana
        "PanelType", "DatasourceType", "RefreshInterval", "GrafanaSettings", "PanelThreshold",
        "PanelTarget", "Panel", "DashboardRow", "Dashboard", "SolaceDashboards",
        "GrafanaDashboardGenerator", "create_grafana_dashboards",
        # Jaeger
        "SamplingType", "SpanStorageType", "JaegerSettings", "SamplingStrategy", "OperationSampling",
        "ServiceSamplingConfig", "SolaceTracingConfig", "CollectorConfig", "AgentConfig",
        "JaegerConfigGenerator", "create_jaeger_config",
        # Alerting
        "ReceiverType", "AlertManagerSettings", "AlertLabel", "AlertRule", "AlertRuleGroup",
        "SolaceAlertRules", "Receiver", "Route", "InhibitRule", "AlertManagerConfigGenerator",
        "create_alertmanager_config", "create_prometheus_alerting_rules",
        # Log aggregation
        "LogBackend", "RetentionTier", "LogAggregationSettings", "LogLabel", "LogPipeline",
        "SolaceLogPipelines", "LokiConfig", "ElasticsearchConfig", "IndexTemplate",
        "SolaceIndexTemplates", "LogAggregationConfigGenerator", "create_log_aggregation_config",
    ),
    "solace_infrastructure.database": (
        # Base models
        "Base", "AuditableModel", "TenantModel", "TimestampMixin", "SoftDeleteMixin", "VersionMixin",
        "AuditMixin", "TenantMixin", "ModelState", "create_all_tables", "create_all_tables_async",
        # Migrations
        "MigrationRunner", "MigrationSettings", "MigrationDirection", "MigrationState", "MigrationInfo",
        "MigrationResult", "create_migration_runner",
        # Seed data
        "SeedDataLoader", "SeedSettings", "SeedResult", "SeedCategory", "Environment",
        "create_seed_loader",
        # Weaviate schema
        "WeaviateSchemaManager", "WeaviateSchemaSettings", "SolaceCollections", "CollectionDefinition",
        "CollectionType", "setup_weaviate_schema",
        # Redis setup
        "RedisSetupManager", "RedisSetupSettings", "RedisKeyBuilder", "RedisNamespace", "MemoryTier",
        "TTLPolicy", "setup_redis",
    ),
}

# public name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    name: (module, name) for module, names in _MODULE_NAMES.items() for name in names
}
_EXPORTS.update({
    "KafkaTopicMetrics": ("solace_infrastructure.kafka", "TopicMetrics"),
    "ObsAlertSeverity": ("solace_infrastructure.observability", "AlertSeverity"),
    "LogAggregationLevel": ("solace_infrastructure.observability", "LogLevel"),
    "LogRetentionPolicy": ("solace_infrastructure.observability", "RetentionPolicy"),
    "DBBaseModel": ("solace_infrastructure.database", "BaseModel"),
})

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

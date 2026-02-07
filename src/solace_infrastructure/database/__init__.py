"""Solace-AI Database Infrastructure - Schema management and initialization.

This module provides database infrastructure components for Solace-AI:
- Base SQLAlchemy models with enterprise patterns
- Centralized schema registry for entity management
- Unified connection pool management
- Alembic migration orchestration
- Seed data loading with dependency resolution
- Weaviate vector database schema management
- Redis namespace configuration

Usage:
    from solace_infrastructure.database import (
        # Base models
        Base, BaseModel, AuditableModel, TenantModel,
        TimestampMixin, SoftDeleteMixin, VersionMixin, AuditMixin,

        # Schema registry
        SchemaRegistry,

        # Connection pooling
        ConnectionPoolManager, ConnectionPoolConfig, get_connection_pool,

        # Migrations
        MigrationRunner, MigrationSettings, create_migration_runner,

        # Seed data
        SeedDataLoader, SeedSettings, create_seed_loader,

        # Weaviate
        WeaviateSchemaManager, SolaceCollections, setup_weaviate_schema,

        # Redis
        RedisSetupManager, RedisKeyBuilder, setup_redis,
    )
"""

from solace_infrastructure.database.base_models import (
    Base,
    BaseModel,
    AuditableModel,
    TenantModel,
    TimestampMixin,
    SoftDeleteMixin,
    VersionMixin,
    AuditMixin,
    TenantMixin,
    UserProfileBase,
    SessionBase,
    ClinicalBase,
    SafetyEventBase,
    ConfigurationBase,
    ModelState,
    create_all_tables,
    create_all_tables_async,
    get_model_table_name,
)

from solace_infrastructure.database.migrations_runner import (
    MigrationRunner,
    MigrationSettings,
    MigrationDirection,
    MigrationState,
    MigrationInfo,
    MigrationResult,
    create_migration_runner,
)

from solace_infrastructure.database.seed_data import (
    SeedDataLoader,
    SeedSettings,
    SeedResult,
    SeedBatch,
    SeedCategory,
    Environment,
    BaseSeedProvider,
    SystemConfigSeedProvider,
    ClinicalReferenceSeedProvider,
    TherapyTechniqueSeedProvider,
    SafetyResourceSeedProvider,
    create_seed_loader,
)

from solace_infrastructure.database.weaviate_schema import (
    WeaviateSchemaManager,
    WeaviateSchemaSettings,
    SolaceCollections,
    CollectionDefinition,
    CollectionType,
    SchemaVersion,
    CURRENT_SCHEMA_VERSION,
    setup_weaviate_schema,
)

from solace_infrastructure.database.redis_setup import (
    RedisSetupManager,
    RedisSetupSettings,
    RedisKeyBuilder,
    RedisNamespace,
    MemoryTier,
    TTLPolicy,
    setup_redis,
)

from solace_infrastructure.database.connection_manager import (
    ConnectionPoolManager,
    ConnectionPoolConfig,
    get_connection_pool,
)

from solace_infrastructure.database.schema_registry import SchemaRegistry

__all__ = [
    # Base models
    "Base",
    "BaseModel",
    "AuditableModel",
    "TenantModel",
    "TimestampMixin",
    "SoftDeleteMixin",
    "VersionMixin",
    "AuditMixin",
    "TenantMixin",
    "UserProfileBase",
    "SessionBase",
    "ClinicalBase",
    "SafetyEventBase",
    "ConfigurationBase",
    "ModelState",
    "create_all_tables",
    "create_all_tables_async",
    "get_model_table_name",
    # Schema registry
    "SchemaRegistry",
    # Connection pooling
    "ConnectionPoolManager",
    "ConnectionPoolConfig",
    "get_connection_pool",
    # Migrations
    "MigrationRunner",
    "MigrationSettings",
    "MigrationDirection",
    "MigrationState",
    "MigrationInfo",
    "MigrationResult",
    "create_migration_runner",
    # Seed data
    "SeedDataLoader",
    "SeedSettings",
    "SeedResult",
    "SeedBatch",
    "SeedCategory",
    "Environment",
    "BaseSeedProvider",
    "SystemConfigSeedProvider",
    "ClinicalReferenceSeedProvider",
    "TherapyTechniqueSeedProvider",
    "SafetyResourceSeedProvider",
    "create_seed_loader",
    # Weaviate schema
    "WeaviateSchemaManager",
    "WeaviateSchemaSettings",
    "SolaceCollections",
    "CollectionDefinition",
    "CollectionType",
    "SchemaVersion",
    "CURRENT_SCHEMA_VERSION",
    "setup_weaviate_schema",
    # Redis setup
    "RedisSetupManager",
    "RedisSetupSettings",
    "RedisKeyBuilder",
    "RedisNamespace",
    "MemoryTier",
    "TTLPolicy",
    "setup_redis",
]

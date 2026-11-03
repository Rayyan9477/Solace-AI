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

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from solace_infrastructure.database.base_models import Base
    from solace_infrastructure.database.base_models import BaseModel
    from solace_infrastructure.database.base_models import AuditableModel
    from solace_infrastructure.database.base_models import TenantModel
    from solace_infrastructure.database.base_models import TimestampMixin
    from solace_infrastructure.database.base_models import SoftDeleteMixin
    from solace_infrastructure.database.base_models import VersionMixin
    from solace_infrastructure.database.base_models import AuditMixin
    from solace_infrastructure.database.base_models import TenantMixin
    from solace_infrastructure.database.base_models import UserProfileBase
    from solace_infrastructure.database.base_models import SessionBase
    from solace_infrastructure.database.base_models import ClinicalBase
    from solace_infrastructure.database.base_models import SafetyEventBase
    from solace_infrastructure.database.base_models import ConfigurationBase
    from solace_infrastructure.database.base_models import ModelState
    from solace_infrastructure.database.base_models import create_all_tables
    from solace_infrastructure.database.base_models import create_all_tables_async
    from solace_infrastructure.database.base_models import get_model_table_name
    from solace_infrastructure.database.base_models import configure_phi_encryption
    from solace_infrastructure.database.base_models import get_phi_encryptor
    from solace_infrastructure.database.migrations_runner import MigrationRunner
    from solace_infrastructure.database.migrations_runner import MigrationSettings
    from solace_infrastructure.database.migrations_runner import MigrationDirection
    from solace_infrastructure.database.migrations_runner import MigrationState
    from solace_infrastructure.database.migrations_runner import MigrationInfo
    from solace_infrastructure.database.migrations_runner import MigrationResult
    from solace_infrastructure.database.migrations_runner import create_migration_runner
    from solace_infrastructure.database.seed_data import SeedDataLoader
    from solace_infrastructure.database.seed_data import SeedSettings
    from solace_infrastructure.database.seed_data import SeedResult
    from solace_infrastructure.database.seed_data import SeedBatch
    from solace_infrastructure.database.seed_data import SeedCategory
    from solace_infrastructure.database.seed_data import Environment
    from solace_infrastructure.database.seed_data import BaseSeedProvider
    from solace_infrastructure.database.seed_data import SystemConfigSeedProvider
    from solace_infrastructure.database.seed_data import ClinicalReferenceSeedProvider
    from solace_infrastructure.database.seed_data import TherapyTechniqueSeedProvider
    from solace_infrastructure.database.seed_data import SafetyResourceSeedProvider
    from solace_infrastructure.database.seed_data import create_seed_loader
    from solace_infrastructure.database.weaviate_schema import WeaviateSchemaManager
    from solace_infrastructure.database.weaviate_schema import WeaviateSchemaSettings
    from solace_infrastructure.database.weaviate_schema import SolaceCollections
    from solace_infrastructure.database.weaviate_schema import CollectionDefinition
    from solace_infrastructure.database.weaviate_schema import CollectionType
    from solace_infrastructure.database.weaviate_schema import SchemaVersion
    from solace_infrastructure.database.weaviate_schema import CURRENT_SCHEMA_VERSION
    from solace_infrastructure.database.weaviate_schema import setup_weaviate_schema
    from solace_infrastructure.database.redis_setup import RedisSetupManager
    from solace_infrastructure.database.redis_setup import RedisSetupSettings
    from solace_infrastructure.database.redis_setup import RedisKeyBuilder
    from solace_infrastructure.database.redis_setup import RedisNamespace
    from solace_infrastructure.database.redis_setup import MemoryTier
    from solace_infrastructure.database.redis_setup import TTLPolicy
    from solace_infrastructure.database.redis_setup import setup_redis
    from solace_infrastructure.database.connection_manager import ConnectionPoolManager
    from solace_infrastructure.database.connection_manager import ConnectionPoolConfig
    from solace_infrastructure.database.connection_manager import get_connection_pool
    from solace_infrastructure.database.schema_registry import SchemaRegistry

# Lazy exports (PEP 562): importing a database submodule no longer loads weaviate,
# alembic, redis or seed modules. Names resolve on first attribute access.
_EXPORTS: dict[str, str] = {
    "Base": "base_models",
    "BaseModel": "base_models",
    "AuditableModel": "base_models",
    "TenantModel": "base_models",
    "TimestampMixin": "base_models",
    "SoftDeleteMixin": "base_models",
    "VersionMixin": "base_models",
    "AuditMixin": "base_models",
    "TenantMixin": "base_models",
    "UserProfileBase": "base_models",
    "SessionBase": "base_models",
    "ClinicalBase": "base_models",
    "SafetyEventBase": "base_models",
    "ConfigurationBase": "base_models",
    "ModelState": "base_models",
    "create_all_tables": "base_models",
    "create_all_tables_async": "base_models",
    "get_model_table_name": "base_models",
    "configure_phi_encryption": "base_models",
    "get_phi_encryptor": "base_models",
    "MigrationRunner": "migrations_runner",
    "MigrationSettings": "migrations_runner",
    "MigrationDirection": "migrations_runner",
    "MigrationState": "migrations_runner",
    "MigrationInfo": "migrations_runner",
    "MigrationResult": "migrations_runner",
    "create_migration_runner": "migrations_runner",
    "SeedDataLoader": "seed_data",
    "SeedSettings": "seed_data",
    "SeedResult": "seed_data",
    "SeedBatch": "seed_data",
    "SeedCategory": "seed_data",
    "Environment": "seed_data",
    "BaseSeedProvider": "seed_data",
    "SystemConfigSeedProvider": "seed_data",
    "ClinicalReferenceSeedProvider": "seed_data",
    "TherapyTechniqueSeedProvider": "seed_data",
    "SafetyResourceSeedProvider": "seed_data",
    "create_seed_loader": "seed_data",
    "WeaviateSchemaManager": "weaviate_schema",
    "WeaviateSchemaSettings": "weaviate_schema",
    "SolaceCollections": "weaviate_schema",
    "CollectionDefinition": "weaviate_schema",
    "CollectionType": "weaviate_schema",
    "SchemaVersion": "weaviate_schema",
    "CURRENT_SCHEMA_VERSION": "weaviate_schema",
    "setup_weaviate_schema": "weaviate_schema",
    "RedisSetupManager": "redis_setup",
    "RedisSetupSettings": "redis_setup",
    "RedisKeyBuilder": "redis_setup",
    "RedisNamespace": "redis_setup",
    "MemoryTier": "redis_setup",
    "TTLPolicy": "redis_setup",
    "setup_redis": "redis_setup",
    "ConnectionPoolManager": "connection_manager",
    "ConnectionPoolConfig": "connection_manager",
    "get_connection_pool": "connection_manager",
    "SchemaRegistry": "schema_registry",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


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
    "configure_phi_encryption",
    "get_phi_encryptor",
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

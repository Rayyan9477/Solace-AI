"""
Solace-AI Safety Service - Infrastructure Layer.
Repository implementations, database clients, and observability.
"""
from .repository import (
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
    SafetyAssessmentRepository,
    SafetyPlanRepository,
    SafetyIncidentRepository,
    UserRiskProfileRepository,
    SafetyRepositoryFactory,
    get_repository_factory,
    reset_repositories,
)

from .database import (
    DatabaseConfig,
    ContraindicationRuleRecord,
    ContraindicationRepository,
    get_contraindication_repository,
    close_contraindication_repository,
    # Backward compatibility aliases
    ContraindicationDBConfig,
    ContraindicationRuleDTO,
    ContraindicationDatabase,
    get_contraindication_db,
    close_contraindication_db,
)

from .telemetry import (
    TelemetryConfig,
    Telemetry,
    get_telemetry,
    traced,
    # Backward compatibility alias
    SafetyServiceTelemetry,
)

__all__ = [
    # Repository
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
    "SafetyAssessmentRepository",
    "SafetyPlanRepository",
    "SafetyIncidentRepository",
    "UserRiskProfileRepository",
    "SafetyRepositoryFactory",
    "get_repository_factory",
    "reset_repositories",
    # Database
    "DatabaseConfig",
    "ContraindicationRuleRecord",
    "ContraindicationRepository",
    "get_contraindication_repository",
    "close_contraindication_repository",
    "ContraindicationDBConfig",
    "ContraindicationRuleDTO",
    "ContraindicationDatabase",
    "get_contraindication_db",
    "close_contraindication_db",
    # Telemetry
    "TelemetryConfig",
    "Telemetry",
    "SafetyServiceTelemetry",
    "get_telemetry",
    "traced",
]

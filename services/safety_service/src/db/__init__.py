"""
DEPRECATED: This module has been moved to safety_service.src.infrastructure.database.
This file exists only for backward compatibility. Import from infrastructure instead.
"""

import warnings

warnings.warn(
    "safety_service.src.db is deprecated. Use safety_service.src.infrastructure.database instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from services.safety_service.src.infrastructure.database import (
    DatabaseConfig as ContraindicationDBConfig,
    ContraindicationRuleRecord as ContraindicationRuleDTO,
    ContraindicationRepository as ContraindicationDatabase,
    get_contraindication_repository as get_contraindication_db,
    close_contraindication_repository as close_contraindication_db,
)

__all__ = [
    "ContraindicationDBConfig",
    "ContraindicationRuleDTO",
    "ContraindicationDatabase",
    "get_contraindication_db",
    "close_contraindication_db",
]

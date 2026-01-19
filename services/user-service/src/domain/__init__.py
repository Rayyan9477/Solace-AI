"""
Solace-AI User Service - Domain Layer.

Public API for the User Service domain layer.
Exports domain entities, value objects, and services.
"""
from .entities import User, UserPreferences
from .value_objects import (
    UserRole,
    AccountStatus,
    ConsentType,
    ConsentRecord,
    EmailAddress,
    DisplayName,
    Timezone,
    Locale,
    PasswordPolicy,
)
from .consent import (
    ConsentService,
    ConsentRepository,
    ConsentGrantResult,
    ConsentRevokeResult,
    ConsentVerificationResult,
)

__all__ = [
    # Entities
    "User",
    "UserPreferences",
    # Value Objects
    "UserRole",
    "AccountStatus",
    "ConsentType",
    "ConsentRecord",
    "EmailAddress",
    "DisplayName",
    "Timezone",
    "Locale",
    "PasswordPolicy",
    # Domain Services
    "ConsentService",
    "ConsentRepository",
    # Results
    "ConsentGrantResult",
    "ConsentRevokeResult",
    "ConsentVerificationResult",
]

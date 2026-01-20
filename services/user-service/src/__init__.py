"""
Solace-AI User Service.

Production-ready user management microservice following Clean Architecture principles.

Architecture:
    - Domain Layer: Core business logic (entities, value objects, domain services)
    - Infrastructure Layer: External dependencies (JWT, password, tokens, encryption)

Usage:
    # Domain layer
    from src.domain import User, UserPreferences, EmailAddress, UserRole
    from src.domain import ConsentService, ConsentType

    # Infrastructure layer
    from src.infrastructure import create_jwt_service, create_password_service
    from src.infrastructure import create_token_service, create_encryption_service

    # Configuration
    from src.config import UserServiceSettings

    # Events
    from src.events import UserCreatedEvent, UserUpdatedEvent
"""
__version__ = "1.0.0"

# Re-export commonly used components for convenience
from .config import UserServiceSettings
from .events import (
    DomainEvent,
    UserCreatedEvent,
    UserUpdatedEvent,
    UserDeletedEvent,
    ConsentGrantedEvent,
    ConsentRevokedEvent,
    LoginSuccessfulEvent,
    LoginFailedEvent,
    AccountLockedEvent,
    PreferencesUpdatedEvent,
)

__all__ = [
    "__version__",
    # Config
    "UserServiceSettings",
    # Events
    "DomainEvent",
    "UserCreatedEvent",
    "UserUpdatedEvent",
    "UserDeletedEvent",
    "ConsentGrantedEvent",
    "ConsentRevokedEvent",
    "LoginSuccessfulEvent",
    "LoginFailedEvent",
    "AccountLockedEvent",
    "PreferencesUpdatedEvent",
]

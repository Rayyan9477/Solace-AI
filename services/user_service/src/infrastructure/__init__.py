"""
Solace-AI User Service - Infrastructure Layer.
Repository implementations and external integrations.
"""
from .repository import (
    UserRepository,
    UserPreferencesRepository,
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    UserRepositoryFactory,
    UserRepositoryConfig,
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
)

__all__ = [
    "UserRepository",
    "UserPreferencesRepository",
    "InMemoryUserRepository",
    "InMemoryUserPreferencesRepository",
    "UserRepositoryFactory",
    "UserRepositoryConfig",
    "RepositoryError",
    "EntityNotFoundError",
    "DuplicateEntityError",
]

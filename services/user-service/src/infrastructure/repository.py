"""
Solace-AI User Service - Repository Layer.

Repository abstractions and implementations for user data persistence.
Follows Repository pattern for separating domain logic from data access.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion, Interface Segregation
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..domain.entities import User, UserPreferences
from ..domain.value_objects import ConsentRecord

logger = structlog.get_logger(__name__)


# --- Configuration ---


class RepositoryConfig(BaseSettings):
    """Repository configuration settings."""

    database_url: str = Field(
        default="postgresql://localhost:5432/solace_users",
        description="Database connection URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    echo_sql: bool = Field(default=False, description="Echo SQL statements")
    use_postgres: bool = Field(
        default=False,
        description="Use PostgreSQL repositories",
    )
    db_schema: str = Field(default="public", description="PostgreSQL schema name")

    model_config = SettingsConfigDict(
        env_prefix="USER_REPO_",
        env_file=".env",
        extra="ignore",
    )


# --- Exceptions ---


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Entity not found in repository."""

    def __init__(self, entity_type: str, entity_id: UUID | str) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID {entity_id} not found")


class DuplicateEntityError(RepositoryError):
    """Duplicate entity in repository."""

    def __init__(self, entity_type: str, identifier: str) -> None:
        self.entity_type = entity_type
        self.identifier = identifier
        super().__init__(f"{entity_type} with identifier {identifier} already exists")


class ConcurrencyError(RepositoryError):
    """Optimistic concurrency violation."""

    def __init__(
        self,
        entity_type: str,
        entity_id: UUID | str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrency conflict for {entity_type} {entity_id}: "
            f"expected version {expected_version}, found {actual_version}"
        )


# --- Abstract Repositories ---


class UserRepository(ABC):
    """
    Abstract repository for user persistence.

    Defines contract for user data access operations.
    Implementations may use different storage backends
    (PostgreSQL, MongoDB, etc.). For testing, use InMemory repos from tests/fixtures.py.
    """

    @abstractmethod
    async def save(self, user: User) -> User:
        """
        Save a new user.

        Args:
            user: User entity to save

        Returns:
            Saved user with any generated fields

        Raises:
            DuplicateEntityError: If email already exists
        """
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> User | None:
        """
        Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User entity or None if not found
        """
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """
        Get user by email address.

        Args:
            email: User email (case-insensitive)

        Returns:
            User entity or None if not found
        """
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        """
        Update an existing user.

        Args:
            user: User entity with updates

        Returns:
            Updated user entity

        Raises:
            EntityNotFoundError: If user doesn't exist
        """
        pass

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """
        Delete a user by ID (hard delete).

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[User]:
        """
        List users with pagination.

        Args:
            limit: Maximum users to return
            offset: Number of users to skip
            include_deleted: Include soft-deleted users

        Returns:
            List of users
        """
        pass

    @abstractmethod
    async def count(self, include_deleted: bool = False) -> int:
        """
        Count total users.

        Args:
            include_deleted: Include soft-deleted users

        Returns:
            Total count
        """
        pass

    @abstractmethod
    async def find_on_call_clinicians(self) -> list[User]:
        """
        Find all clinicians currently on-call for crisis notifications.

        Returns:
            List of on-call clinician users with contact information.
        """
        pass

    @abstractmethod
    async def find_by_role(self, role: str, active_only: bool = True) -> list[User]:
        """
        Find users by role.

        Args:
            role: User role to filter by
            active_only: Only return active accounts

        Returns:
            List of users with the specified role.
        """
        pass

    @abstractmethod
    async def assign_patient_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Assign a patient to a clinician."""
        pass

    @abstractmethod
    async def unassign_patient_from_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Remove a patient assignment from a clinician."""
        pass

    @abstractmethod
    async def is_patient_assigned_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Check if a specific patient is assigned to a specific clinician."""
        pass

    @abstractmethod
    async def get_assigned_patients(self, clinician_id: UUID) -> list[UUID]:
        """Get all patient IDs assigned to a clinician."""
        pass

    @abstractmethod
    async def get_assigned_clinician(self, patient_id: UUID) -> UUID | None:
        """Get the clinician assigned to a patient, if any."""
        pass


class UserPreferencesRepository(ABC):
    """
    Abstract repository for user preferences persistence.

    Defines contract for preferences data access operations.
    """

    @abstractmethod
    async def save(self, preferences: UserPreferences) -> UserPreferences:
        """
        Save user preferences.

        Args:
            preferences: Preferences entity to save

        Returns:
            Saved preferences
        """
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> UserPreferences | None:
        """
        Get preferences for a user.

        Args:
            user_id: User identifier

        Returns:
            Preferences entity or None
        """
        pass

    @abstractmethod
    async def update(self, preferences: UserPreferences) -> UserPreferences:
        """
        Update existing preferences.

        Args:
            preferences: Preferences entity with updates

        Returns:
            Updated preferences
        """
        pass

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """
        Delete preferences for a user.

        Args:
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        pass


class ConsentRepository(ABC):
    """
    Abstract repository for consent records persistence.

    Defines contract for consent audit trail operations.
    Consent records are immutable - only append operations.
    """

    @abstractmethod
    async def save(self, consent: ConsentRecord) -> ConsentRecord:
        """
        Save a consent record.

        Args:
            consent: Consent record to save

        Returns:
            Saved consent record
        """
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: UUID) -> list[ConsentRecord]:
        """
        Get all consent records for a user.

        Args:
            user_id: User identifier

        Returns:
            List of consent records, ordered by timestamp
        """
        pass

    @abstractmethod
    async def get_by_id(self, consent_id: UUID) -> ConsentRecord | None:
        """
        Get consent record by ID.

        Args:
            consent_id: Consent record identifier

        Returns:
            Consent record or None
        """
        pass


# --- Repository Factory ---


class RepositoryFactory:
    """
    Factory for creating repository instances.

    Supports different storage backends based on configuration.
    Uses singleton pattern for repository instances.
    """

    def __init__(
        self,
        config: RepositoryConfig | None = None,
        postgres_client: Any | None = None,
    ) -> None:
        self._config = config or RepositoryConfig()
        self._postgres_client = postgres_client
        self._user_repo: UserRepository | None = None
        self._prefs_repo: UserPreferencesRepository | None = None
        self._consent_repo: ConsentRepository | None = None

    def get_user_repository(self) -> UserRepository:
        """Get or create user repository."""
        if self._user_repo is None:
            if self._config.use_postgres and self._postgres_client is not None:
                from .postgres_repository import PostgresUserRepository
                self._user_repo = PostgresUserRepository(
                    self._postgres_client,
                    schema=self._config.db_schema,
                )
                logger.info("user_repository_created", type="postgres")
            else:
                raise RepositoryError(
                    "PostgreSQL is required. For tests, use InMemory repos from tests/fixtures.py"
                )
        return self._user_repo

    def get_preferences_repository(self) -> UserPreferencesRepository:
        """Get or create preferences repository."""
        if self._prefs_repo is None:
            if self._config.use_postgres and self._postgres_client is not None:
                from .postgres_repository import PostgresUserPreferencesRepository
                self._prefs_repo = PostgresUserPreferencesRepository(
                    self._postgres_client,
                    schema=self._config.db_schema,
                )
                logger.info("preferences_repository_created", type="postgres")
            else:
                raise RepositoryError(
                    "PostgreSQL is required. For tests, use InMemory repos from tests/fixtures.py"
                )
        return self._prefs_repo

    def get_consent_repository(self) -> ConsentRepository:
        """Get or create consent repository."""
        if self._consent_repo is None:
            if self._config.use_postgres and self._postgres_client is not None:
                from .postgres_repository import PostgresConsentRepository
                self._consent_repo = PostgresConsentRepository(
                    self._postgres_client,
                    schema=self._config.db_schema,
                )
                logger.info("consent_repository_created", type="postgres")
            else:
                raise RepositoryError(
                    "PostgreSQL is required. For tests, use InMemory repos from tests/fixtures.py"
                )
        return self._consent_repo

    def reset(self) -> None:
        """Reset all repositories (for testing)."""
        self._user_repo = None
        self._prefs_repo = None
        self._consent_repo = None


# --- Module-Level Singleton ---


_factory: RepositoryFactory | None = None


def get_repository_factory(
    config: RepositoryConfig | None = None,
    postgres_client: Any | None = None,
) -> RepositoryFactory:
    """
    Get singleton repository factory.

    Args:
        config: Optional configuration (only used on first call)
        postgres_client: Optional PostgreSQL client for database connections

    Returns:
        Repository factory instance
    """
    global _factory
    if _factory is None:
        _factory = RepositoryFactory(config, postgres_client)
    return _factory


def reset_repositories() -> None:
    """Reset all repositories (for testing)."""
    global _factory
    _factory = None

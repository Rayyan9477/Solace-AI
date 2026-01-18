"""
Solace-AI User Service - Repository Layer.
Repository abstraction and implementations for user data persistence.
"""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..domain.service import User, UserPreferences, ConsentRecord

logger = structlog.get_logger(__name__)


class UserRepositoryConfig(BaseSettings):
    """Repository configuration settings."""
    database_url: str = Field(default="postgresql://localhost:5432/solace_users")
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    echo_sql: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix="USER_REPO_", env_file=".env", extra="ignore")


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Entity not found in repository."""
    def __init__(self, entity_type: str, entity_id: UUID) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID {entity_id} not found")


class DuplicateEntityError(RepositoryError):
    """Duplicate entity in repository."""
    def __init__(self, entity_type: str, identifier: str) -> None:
        self.entity_type = entity_type
        self.identifier = identifier
        super().__init__(f"{entity_type} with identifier {identifier} already exists")


class UserRepository(ABC):
    """Abstract repository for users."""

    @abstractmethod
    async def save(self, user: User) -> User:
        """Save a user."""
        pass

    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> User | None:
        """Get user by ID."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        pass

    @abstractmethod
    async def update(self, user: User) -> User:
        """Update an existing user."""
        pass

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete a user by ID."""
        pass

    @abstractmethod
    async def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """List users with pagination."""
        pass

    @abstractmethod
    async def get_consent_records(self, user_id: UUID) -> list[ConsentRecord]:
        """Get consent records for a user."""
        pass

    @abstractmethod
    async def save_consent(self, consent: ConsentRecord) -> ConsentRecord:
        """Save a consent record."""
        pass


class UserPreferencesRepository(ABC):
    """Abstract repository for user preferences."""

    @abstractmethod
    async def save(self, preferences: UserPreferences) -> UserPreferences:
        """Save user preferences."""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> UserPreferences | None:
        """Get preferences for a user."""
        pass

    @abstractmethod
    async def update(self, preferences: UserPreferences) -> UserPreferences:
        """Update existing preferences."""
        pass

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete preferences for a user."""
        pass


class InMemoryUserRepository(UserRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._users: dict[UUID, User] = {}
        self._users_by_email: dict[str, UUID] = {}
        self._consent_records: dict[UUID, list[ConsentRecord]] = {}
        self._lock = asyncio.Lock()

    async def save(self, user: User) -> User:
        async with self._lock:
            if user.email.lower() in self._users_by_email:
                raise DuplicateEntityError("User", user.email)
            self._users[user.user_id] = user
            self._users_by_email[user.email.lower()] = user.user_id
            logger.debug("user_saved", user_id=str(user.user_id), email=user.email)
            return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        user = self._users.get(user_id)
        if user and user.deleted_at is None:
            return user
        return None

    async def get_by_email(self, email: str) -> User | None:
        user_id = self._users_by_email.get(email.lower())
        if user_id:
            return await self.get_by_id(user_id)
        return None

    async def update(self, user: User) -> User:
        async with self._lock:
            if user.user_id not in self._users:
                raise EntityNotFoundError("User", user.user_id)
            old_user = self._users[user.user_id]
            if old_user.email.lower() != user.email.lower():
                del self._users_by_email[old_user.email.lower()]
                self._users_by_email[user.email.lower()] = user.user_id
            self._users[user.user_id] = user
            logger.debug("user_updated", user_id=str(user.user_id))
            return user

    async def delete(self, user_id: UUID) -> bool:
        async with self._lock:
            if user_id not in self._users:
                return False
            user = self._users[user_id]
            del self._users_by_email[user.email.lower()]
            del self._users[user_id]
            logger.debug("user_deleted", user_id=str(user_id))
            return True

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        users = [u for u in self._users.values() if u.deleted_at is None]
        users.sort(key=lambda u: u.created_at, reverse=True)
        return users[offset:offset + limit]

    async def get_consent_records(self, user_id: UUID) -> list[ConsentRecord]:
        return self._consent_records.get(user_id, [])

    async def save_consent(self, consent: ConsentRecord) -> ConsentRecord:
        async with self._lock:
            if consent.user_id not in self._consent_records:
                self._consent_records[consent.user_id] = []
            self._consent_records[consent.user_id].append(consent)
            logger.debug("consent_saved", user_id=str(consent.user_id), consent_type=consent.consent_type)
            return consent

    async def count_users(self) -> int:
        """Count total users."""
        return len([u for u in self._users.values() if u.deleted_at is None])

    async def search_users(self, query: str, limit: int = 20) -> list[User]:
        """Search users by email or display name."""
        query_lower = query.lower()
        matches = [
            u for u in self._users.values()
            if u.deleted_at is None and (query_lower in u.email.lower() or query_lower in u.display_name.lower())
        ]
        return matches[:limit]


class InMemoryUserPreferencesRepository(UserPreferencesRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._preferences: dict[UUID, UserPreferences] = {}
        self._lock = asyncio.Lock()

    async def save(self, preferences: UserPreferences) -> UserPreferences:
        async with self._lock:
            self._preferences[preferences.user_id] = preferences
            logger.debug("preferences_saved", user_id=str(preferences.user_id))
            return preferences

    async def get_by_user(self, user_id: UUID) -> UserPreferences | None:
        return self._preferences.get(user_id)

    async def update(self, preferences: UserPreferences) -> UserPreferences:
        async with self._lock:
            self._preferences[preferences.user_id] = preferences
            logger.debug("preferences_updated", user_id=str(preferences.user_id))
            return preferences

    async def delete(self, user_id: UUID) -> bool:
        async with self._lock:
            if user_id in self._preferences:
                del self._preferences[user_id]
                return True
            return False

    async def get_users_with_notification_enabled(self, channel: str) -> list[UUID]:
        """Get users with specific notification channel enabled."""
        users = []
        for user_id, prefs in self._preferences.items():
            if channel == "email" and prefs.notification_email:
                users.append(user_id)
            elif channel == "sms" and prefs.notification_sms:
                users.append(user_id)
            elif channel == "push" and prefs.notification_push:
                users.append(user_id)
        return users


class UserRepositoryFactory:
    """Factory for creating repository instances."""

    def __init__(self, config: UserRepositoryConfig | None = None) -> None:
        self._config = config or UserRepositoryConfig()
        self._user_repo: UserRepository | None = None
        self._prefs_repo: UserPreferencesRepository | None = None

    def get_user_repository(self) -> UserRepository:
        """Get or create user repository."""
        if self._user_repo is None:
            self._user_repo = InMemoryUserRepository()
            logger.info("user_repository_created", type="in_memory")
        return self._user_repo

    def get_preferences_repository(self) -> UserPreferencesRepository:
        """Get or create preferences repository."""
        if self._prefs_repo is None:
            self._prefs_repo = InMemoryUserPreferencesRepository()
            logger.info("preferences_repository_created", type="in_memory")
        return self._prefs_repo

    def reset(self) -> None:
        """Reset all repositories (for testing)."""
        self._user_repo = None
        self._prefs_repo = None


_factory: UserRepositoryFactory | None = None


def get_repository_factory() -> UserRepositoryFactory:
    """Get singleton repository factory."""
    global _factory
    if _factory is None:
        _factory = UserRepositoryFactory()
    return _factory


def reset_repositories() -> None:
    """Reset all repositories (for testing)."""
    global _factory
    _factory = None

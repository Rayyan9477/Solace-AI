"""
Solace-AI Personality Service - Repository Infrastructure.
Persistence layer implementing repository pattern for personality domain entities.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from uuid import UUID
import structlog

from ..domain.entities import PersonalityProfile, TraitAssessment, ProfileSnapshot

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Abstract base repository interface."""
    @abstractmethod
    async def get(self, entity_id: UUID) -> T | None: ...
    @abstractmethod
    async def save(self, entity: T) -> T: ...
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool: ...


class PersonalityRepositoryPort(ABC):
    """Abstract port for personality persistence."""
    @abstractmethod
    async def save_profile(self, profile: PersonalityProfile) -> None: ...
    @abstractmethod
    async def get_profile(self, profile_id: UUID) -> PersonalityProfile | None: ...
    @abstractmethod
    async def get_profile_by_user(self, user_id: UUID) -> PersonalityProfile | None: ...
    @abstractmethod
    async def list_profiles(self, limit: int = 100, offset: int = 0) -> list[PersonalityProfile]: ...
    @abstractmethod
    async def delete_profile(self, profile_id: UUID) -> bool: ...
    @abstractmethod
    async def save_assessment(self, assessment: TraitAssessment) -> None: ...
    @abstractmethod
    async def get_assessment(self, assessment_id: UUID) -> TraitAssessment | None: ...
    @abstractmethod
    async def list_user_assessments(self, user_id: UUID, limit: int = 10) -> list[TraitAssessment]: ...
    @abstractmethod
    async def save_snapshot(self, snapshot: ProfileSnapshot) -> None: ...
    @abstractmethod
    async def get_snapshot(self, snapshot_id: UUID) -> ProfileSnapshot | None: ...
    @abstractmethod
    async def list_snapshots(self, profile_id: UUID, limit: int = 10) -> list[ProfileSnapshot]: ...
    @abstractmethod
    async def delete_user_data(self, user_id: UUID) -> int: ...
    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]: ...


class UnitOfWork:
    """Unit of Work pattern for coordinating repository transactions."""
    def __init__(self, repository: PersonalityRepositoryPort) -> None:
        self.repository = repository
        self._committed = False
        self._pending_profiles: list[PersonalityProfile] = []
        self._pending_assessments: list[TraitAssessment] = []
        self._pending_snapshots: list[ProfileSnapshot] = []

    def add_profile(self, profile: PersonalityProfile) -> None:
        self._pending_profiles.append(profile)

    def add_assessment(self, assessment: TraitAssessment) -> None:
        self._pending_assessments.append(assessment)

    def add_snapshot(self, snapshot: ProfileSnapshot) -> None:
        self._pending_snapshots.append(snapshot)

    async def commit(self) -> None:
        for profile in self._pending_profiles:
            await self.repository.save_profile(profile)
        for assessment in self._pending_assessments:
            await self.repository.save_assessment(assessment)
        for snapshot in self._pending_snapshots:
            await self.repository.save_snapshot(snapshot)
        self._pending_profiles.clear()
        self._pending_assessments.clear()
        self._pending_snapshots.clear()
        self._committed = True

    async def rollback(self) -> None:
        self._pending_profiles.clear()
        self._pending_assessments.clear()
        self._pending_snapshots.clear()

    async def __aenter__(self) -> UnitOfWork:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            await self.rollback()
        elif not self._committed:
            await self.commit()


class RepositoryFactory:
    """Factory for creating repository instances."""
    _instance: PersonalityRepositoryPort | None = None
    _postgres_client: Any | None = None
    _use_postgres: bool = False
    _schema: str = "public"

    @classmethod
    def configure(
        cls,
        use_postgres: bool = False,
        postgres_client: Any | None = None,
        schema: str = "public",
    ) -> None:
        """Configure the factory for creating repositories."""
        cls._use_postgres = use_postgres
        cls._postgres_client = postgres_client
        cls._schema = schema
        cls._instance = None  # Reset to pick up new config

    @classmethod
    def create_postgres(cls, client: Any, schema: str = "public") -> PersonalityRepositoryPort:
        """Create PostgreSQL repository."""
        from .postgres_repository import PostgresPersonalityRepository
        return PostgresPersonalityRepository(client, schema=schema)

    @classmethod
    def get_default(cls) -> PersonalityRepositoryPort:
        """Get default repository instance (singleton).

        Requires PostgreSQL configuration. For tests, use
        InMemoryPersonalityRepository from tests/fixtures.py.
        """
        if cls._instance is None:
            env = os.getenv("ENVIRONMENT", "development")
            if env == "test":
                raise RuntimeError(
                    "For tests, use InMemoryPersonalityRepository from tests/fixtures.py"
                )
            if cls._use_postgres and cls._postgres_client is not None:
                from .postgres_repository import PostgresPersonalityRepository
                cls._instance = PostgresPersonalityRepository(
                    cls._postgres_client,
                    schema=cls._schema,
                )
                logger.info("personality_repository_created", type="postgres")
            else:
                try:
                    from solace_infrastructure.database import ConnectionPoolManager
                    from .postgres_repository import PostgresPersonalityRepository
                    cls._instance = PostgresPersonalityRepository(
                        ConnectionPoolManager,
                        schema=cls._schema,
                    )
                    logger.info("personality_repository_created", type="postgres_auto")
                except (ImportError, Exception) as e:
                    raise RuntimeError(
                        "PostgreSQL is required. Set POSTGRES_* environment variables."
                    ) from e
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._postgres_client = None
        cls._use_postgres = False
        cls._schema = "public"

    @classmethod
    def create_unit_of_work(cls, repository: PersonalityRepositoryPort | None = None) -> UnitOfWork:
        return UnitOfWork(repository or cls.get_default())

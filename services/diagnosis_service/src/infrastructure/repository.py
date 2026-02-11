"""
Solace-AI Diagnosis Service - Repository Layer.
Provides persistence abstraction for diagnosis entities.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID
import structlog

from ..domain.entities import (
    DiagnosisSessionEntity, SymptomEntity, DiagnosisRecordEntity,
)

logger = structlog.get_logger(__name__)


class DiagnosisRepositoryPort(ABC):
    """Abstract port for diagnosis persistence."""

    @abstractmethod
    async def save_session(self, session: DiagnosisSessionEntity) -> None:
        """Save a diagnosis session."""

    @abstractmethod
    async def get_session(self, session_id: UUID) -> DiagnosisSessionEntity | None:
        """Get a session by ID."""

    @abstractmethod
    async def get_active_session(self, user_id: UUID) -> DiagnosisSessionEntity | None:
        """Get active session for user."""

    @abstractmethod
    async def list_user_sessions(self, user_id: UUID, limit: int = 10) -> list[DiagnosisSessionEntity]:
        """List sessions for a user."""

    @abstractmethod
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session."""

    @abstractmethod
    async def save_record(self, record: DiagnosisRecordEntity) -> None:
        """Save a diagnosis record."""

    @abstractmethod
    async def get_record(self, record_id: UUID) -> DiagnosisRecordEntity | None:
        """Get a diagnosis record by ID."""

    @abstractmethod
    async def list_user_records(self, user_id: UUID, limit: int = 10) -> list[DiagnosisRecordEntity]:
        """List diagnosis records for a user."""

    @abstractmethod
    async def delete_user_data(self, user_id: UUID) -> int:
        """Delete all data for a user (GDPR)."""

    @abstractmethod
    async def get_symptom_history(self, user_id: UUID, symptom_name: str) -> list[SymptomEntity]:
        """Get history of a specific symptom for user."""

    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics."""


class RepositoryFactory:
    """Factory for creating repository instances."""

    _instance: DiagnosisRepositoryPort | None = None
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
    def create_postgres(cls, client: Any, schema: str = "public") -> DiagnosisRepositoryPort:
        """Create PostgreSQL repository."""
        from .postgres_repository import PostgresDiagnosisRepository
        return PostgresDiagnosisRepository(client, schema=schema)

    @classmethod
    def get_default(cls) -> DiagnosisRepositoryPort:
        """Get default repository instance (singleton).

        Requires PostgreSQL configuration. In-memory repositories are only
        available through test fixtures.
        Uses ConnectionPoolManager for pool management.
        """
        if cls._instance is None:
            env = os.getenv("ENVIRONMENT", "development")
            if env == "test":
                raise RuntimeError(
                    "For tests, use InMemoryDiagnosisRepository from tests/fixtures.py"
                )
            # Explicit postgres configuration takes priority
            if cls._use_postgres and cls._postgres_client is not None:
                from .postgres_repository import PostgresDiagnosisRepository
                cls._instance = PostgresDiagnosisRepository(
                    cls._postgres_client,
                    schema=cls._schema,
                )
                logger.info("diagnosis_repository_created", type="postgres")
            else:
                # Auto-detect postgres via ConnectionPoolManager
                try:
                    from solace_infrastructure.database import ConnectionPoolManager
                    from .postgres_repository import PostgresDiagnosisRepository
                    cls._instance = PostgresDiagnosisRepository(
                        ConnectionPoolManager,
                        schema=cls._schema,
                    )
                    logger.info("diagnosis_repository_created", type="postgres_auto")
                except (ImportError, Exception) as e:
                    raise RuntimeError(
                        "PostgreSQL is required. Set POSTGRES_* environment variables. "
                        "For testing, set ENVIRONMENT=test and use test fixtures."
                    ) from e
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        cls._postgres_client = None
        cls._use_postgres = False
        cls._schema = "public"

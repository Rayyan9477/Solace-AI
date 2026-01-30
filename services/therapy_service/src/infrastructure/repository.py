"""
Solace-AI Therapy Service - Repository Infrastructure.
Persistence layer implementing repository pattern for therapy domain entities.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Generic, TypeVar
from uuid import UUID
import structlog

from ..domain.entities import TreatmentPlanEntity, TherapySessionEntity
from ..domain.value_objects import Technique, OutcomeMeasure

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


class TreatmentPlanRepository(Repository[TreatmentPlanEntity]):
    """Repository for treatment plan persistence."""

    def __init__(self) -> None:
        self._storage: dict[UUID, TreatmentPlanEntity] = {}
        self._user_index: dict[UUID, list[UUID]] = {}

    async def get(self, plan_id: UUID) -> TreatmentPlanEntity | None:
        """Get treatment plan by ID."""
        return self._storage.get(plan_id)

    async def save(self, plan: TreatmentPlanEntity) -> TreatmentPlanEntity:
        """Save treatment plan."""
        self._storage[plan.plan_id] = plan
        if plan.user_id not in self._user_index:
            self._user_index[plan.user_id] = []
        if plan.plan_id not in self._user_index[plan.user_id]:
            self._user_index[plan.user_id].append(plan.plan_id)
        return plan

    async def delete(self, plan_id: UUID) -> bool:
        """Delete treatment plan."""
        if plan_id in self._storage:
            plan = self._storage.pop(plan_id)
            if plan.user_id in self._user_index:
                self._user_index[plan.user_id] = [pid for pid in self._user_index[plan.user_id] if pid != plan_id]
            return True
        return False

    async def get_by_user(self, user_id: UUID) -> list[TreatmentPlanEntity]:
        """Get all treatment plans for a user."""
        plan_ids = self._user_index.get(user_id, [])
        return [self._storage[pid] for pid in plan_ids if pid in self._storage]

    async def get_active_by_user(self, user_id: UUID) -> list[TreatmentPlanEntity]:
        """Get active treatment plans for a user."""
        return [p for p in await self.get_by_user(user_id) if p.is_active]

    async def find_by_criteria(self, user_id: UUID | None = None, modality: str | None = None,
                               is_active: bool | None = None, min_sessions: int | None = None) -> list[TreatmentPlanEntity]:
        """Find treatment plans by criteria."""
        results = list(self._storage.values())
        if user_id:
            results = [p for p in results if p.user_id == user_id]
        if modality:
            results = [p for p in results if p.primary_modality.value == modality]
        if is_active is not None:
            results = [p for p in results if p.is_active == is_active]
        if min_sessions is not None:
            results = [p for p in results if p.total_sessions_completed >= min_sessions]
        return results

    async def count(self) -> int:
        """Count total treatment plans."""
        return len(self._storage)


class TherapySessionRepository(Repository[TherapySessionEntity]):
    """Repository for therapy session persistence."""

    def __init__(self) -> None:
        self._storage: dict[UUID, TherapySessionEntity] = {}
        self._user_index: dict[UUID, list[UUID]] = {}
        self._plan_index: dict[UUID, list[UUID]] = {}

    async def get(self, session_id: UUID) -> TherapySessionEntity | None:
        """Get therapy session by ID."""
        return self._storage.get(session_id)

    async def save(self, session: TherapySessionEntity) -> TherapySessionEntity:
        """Save therapy session."""
        self._storage[session.session_id] = session
        if session.user_id not in self._user_index:
            self._user_index[session.user_id] = []
        if session.session_id not in self._user_index[session.user_id]:
            self._user_index[session.user_id].append(session.session_id)
        if session.treatment_plan_id not in self._plan_index:
            self._plan_index[session.treatment_plan_id] = []
        if session.session_id not in self._plan_index[session.treatment_plan_id]:
            self._plan_index[session.treatment_plan_id].append(session.session_id)
        return session

    async def delete(self, session_id: UUID) -> bool:
        """Delete therapy session."""
        if session_id in self._storage:
            session = self._storage.pop(session_id)
            if session.user_id in self._user_index:
                self._user_index[session.user_id] = [sid for sid in self._user_index[session.user_id] if sid != session_id]
            if session.treatment_plan_id in self._plan_index:
                self._plan_index[session.treatment_plan_id] = [sid for sid in self._plan_index[session.treatment_plan_id] if sid != session_id]
            return True
        return False

    async def get_by_user(self, user_id: UUID, limit: int = 100, offset: int = 0) -> list[TherapySessionEntity]:
        """Get therapy sessions for a user."""
        session_ids = self._user_index.get(user_id, [])
        sessions = sorted([self._storage[sid] for sid in session_ids if sid in self._storage], key=lambda s: s.started_at, reverse=True)
        return sessions[offset:offset + limit]

    async def get_by_plan(self, plan_id: UUID, limit: int = 100, offset: int = 0) -> list[TherapySessionEntity]:
        """Get therapy sessions for a treatment plan."""
        session_ids = self._plan_index.get(plan_id, [])
        sessions = sorted([self._storage[sid] for sid in session_ids if sid in self._storage], key=lambda s: s.session_number)
        return sessions[offset:offset + limit]

    async def get_active_by_user(self, user_id: UUID) -> TherapySessionEntity | None:
        """Get active therapy session for a user."""
        for sid in self._user_index.get(user_id, []):
            session = self._storage.get(sid)
            if session and session.is_active:
                return session
        return None

    async def get_recent_sessions(self, user_id: UUID, days: int = 30) -> list[TherapySessionEntity]:
        """Get recent therapy sessions."""
        cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
        return [s for s in await self.get_by_user(user_id) if s.started_at >= cutoff]

    async def count_by_plan(self, plan_id: UUID) -> int:
        """Count sessions for a treatment plan."""
        return len(self._plan_index.get(plan_id, []))

    async def count(self) -> int:
        """Count total therapy sessions."""
        return len(self._storage)


class TechniqueRepository:
    """Repository for therapeutic technique lookup."""

    def __init__(self) -> None:
        self._techniques: dict[UUID, Technique] = {}
        self._name_index: dict[str, UUID] = {}
        self._modality_index: dict[str, list[UUID]] = {}

    async def get(self, technique_id: UUID) -> Technique | None:
        """Get technique by ID."""
        return self._techniques.get(technique_id)

    async def get_by_name(self, name: str) -> Technique | None:
        """Get technique by name."""
        technique_id = self._name_index.get(name.lower())
        return self._techniques.get(technique_id) if technique_id else None

    async def save(self, technique: Technique) -> Technique:
        """Save technique."""
        self._techniques[technique.technique_id] = technique
        self._name_index[technique.name.lower()] = technique.technique_id
        modality_key = technique.modality.value
        if modality_key not in self._modality_index:
            self._modality_index[modality_key] = []
        if technique.technique_id not in self._modality_index[modality_key]:
            self._modality_index[modality_key].append(technique.technique_id)
        return technique

    async def delete(self, technique_id: UUID) -> bool:
        """Delete technique."""
        if technique_id in self._techniques:
            technique = self._techniques.pop(technique_id)
            self._name_index.pop(technique.name.lower(), None)
            modality_key = technique.modality.value
            if modality_key in self._modality_index:
                self._modality_index[modality_key] = [tid for tid in self._modality_index[modality_key] if tid != technique_id]
            return True
        return False

    async def get_by_modality(self, modality: str) -> list[Technique]:
        """Get techniques by modality."""
        return [self._techniques[tid] for tid in self._modality_index.get(modality, []) if tid in self._techniques]

    async def search(self, modality: str | None = None, category: str | None = None,
                     difficulty: str | None = None, max_duration: int | None = None) -> list[Technique]:
        """Search techniques by criteria."""
        results = list(self._techniques.values())
        if modality:
            results = [t for t in results if t.modality.value == modality]
        if category:
            results = [t for t in results if t.category.value == category]
        if difficulty:
            results = [t for t in results if t.difficulty.value == difficulty]
        if max_duration:
            results = [t for t in results if t.duration_minutes <= max_duration]
        return results

    async def count(self) -> int:
        """Count total techniques."""
        return len(self._techniques)


class OutcomeMeasureRepository:
    """Repository for outcome measure persistence."""

    def __init__(self) -> None:
        self._measures: dict[UUID, OutcomeMeasure] = {}
        self._session_index: dict[UUID, list[UUID]] = {}
        self._user_instrument_index: dict[tuple[UUID, str], list[UUID]] = {}

    async def get(self, measure_id: UUID) -> OutcomeMeasure | None:
        """Get outcome measure by ID."""
        return self._measures.get(measure_id)

    async def save(self, measure: OutcomeMeasure, user_id: UUID) -> OutcomeMeasure:
        """Save outcome measure."""
        self._measures[measure.measure_id] = measure
        if measure.session_id:
            if measure.session_id not in self._session_index:
                self._session_index[measure.session_id] = []
            if measure.measure_id not in self._session_index[measure.session_id]:
                self._session_index[measure.session_id].append(measure.measure_id)
        index_key = (user_id, measure.instrument.value)
        if index_key not in self._user_instrument_index:
            self._user_instrument_index[index_key] = []
        if measure.measure_id not in self._user_instrument_index[index_key]:
            self._user_instrument_index[index_key].append(measure.measure_id)
        return measure

    async def delete(self, measure_id: UUID) -> bool:
        """Delete outcome measure."""
        if measure_id in self._measures:
            measure = self._measures.pop(measure_id)
            if measure.session_id and measure.session_id in self._session_index:
                self._session_index[measure.session_id] = [mid for mid in self._session_index[measure.session_id] if mid != measure_id]
            return True
        return False

    async def get_by_session(self, session_id: UUID) -> list[OutcomeMeasure]:
        """Get outcome measures for a session."""
        return [self._measures[mid] for mid in self._session_index.get(session_id, []) if mid in self._measures]

    async def get_history(self, user_id: UUID, instrument: str, limit: int = 50) -> list[OutcomeMeasure]:
        """Get outcome measure history for user and instrument."""
        measure_ids = self._user_instrument_index.get((user_id, instrument), [])
        measures = sorted([self._measures[mid] for mid in measure_ids if mid in self._measures], key=lambda m: m.recorded_at, reverse=True)
        return measures[:limit]

    async def get_latest(self, user_id: UUID, instrument: str) -> OutcomeMeasure | None:
        """Get latest outcome measure for user and instrument."""
        history = await self.get_history(user_id, instrument, limit=1)
        return history[0] if history else None

    async def count(self) -> int:
        """Count total outcome measures."""
        return len(self._measures)


class UnitOfWork:
    """Unit of Work pattern for coordinating repository transactions."""

    def __init__(self, treatment_plans: TreatmentPlanRepository, sessions: TherapySessionRepository,
                 techniques: TechniqueRepository, outcomes: OutcomeMeasureRepository) -> None:
        self.treatment_plans, self.sessions = treatment_plans, sessions
        self.techniques, self.outcomes = techniques, outcomes
        self._committed = False

    async def commit(self) -> None:
        """Commit all changes."""
        self._committed = True

    async def rollback(self) -> None:
        """Rollback changes (no-op for in-memory)."""
        pass

    async def __aenter__(self) -> UnitOfWork:
        """Enter context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if exc_type is not None:
            await self.rollback()
        elif not self._committed:
            await self.commit()


def create_unit_of_work(backend: str = "memory", **kwargs: Any) -> UnitOfWork:
    """Factory to create UnitOfWork with the configured backend.

    Args:
        backend: Storage backend type ("memory" or "postgres").
        **kwargs: Backend-specific configuration (e.g., postgres_client for postgres).

    Returns:
        Configured UnitOfWork with appropriate repositories.
    """
    if backend == "postgres":
        logger.info("therapy_repository_backend", backend="postgres")
        # PostgreSQL backend - import here to avoid hard dependency
        try:
            from .postgres_repository import (
                PostgresTreatmentPlanRepository,
                PostgresTherapySessionRepository,
                PostgresTechniqueRepository,
                PostgresOutcomeMeasureRepository,
            )
            client = kwargs["postgres_client"]
            return UnitOfWork(
                treatment_plans=PostgresTreatmentPlanRepository(client),
                sessions=PostgresTherapySessionRepository(client),
                techniques=PostgresTechniqueRepository(client),
                outcomes=PostgresOutcomeMeasureRepository(client),
            )
        except (ImportError, KeyError) as e:
            logger.warning("postgres_backend_unavailable", error=str(e), fallback="memory")

    logger.info("therapy_repository_backend", backend="memory")
    return UnitOfWork(
        treatment_plans=TreatmentPlanRepository(),
        sessions=TherapySessionRepository(),
        techniques=TechniqueRepository(),
        outcomes=OutcomeMeasureRepository(),
    )

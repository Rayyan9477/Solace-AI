"""
Solace-AI Safety Service - Repository Layer.
Repository abstraction and implementations for safety data persistence.
"""
from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from typing import Any, Generic, TypeVar
from uuid import UUID
import structlog

from ..domain.entities import (
    SafetyAssessment, SafetyPlan, SafetyIncident, UserRiskProfile,
    SafetyPlanStatus, IncidentStatus, IncidentSeverity,
)
from ..config import SafetyRepositoryConfig

logger = structlog.get_logger(__name__)
T = TypeVar("T")


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
    def __init__(self, entity_type: str, entity_id: UUID) -> None:
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID {entity_id} already exists")


class SafetyAssessmentRepository(ABC):
    """Abstract repository for safety assessments."""

    @abstractmethod
    async def save(self, assessment: SafetyAssessment) -> SafetyAssessment:
        """Save a safety assessment."""
        pass

    @abstractmethod
    async def get_by_id(self, assessment_id: UUID) -> SafetyAssessment | None:
        """Get assessment by ID."""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyAssessment]:
        """Get assessments for a user."""
        pass

    @abstractmethod
    async def get_by_session(self, session_id: UUID) -> list[SafetyAssessment]:
        """Get assessments for a session."""
        pass

    @abstractmethod
    async def get_recent(self, user_id: UUID, hours: int = 24) -> list[SafetyAssessment]:
        """Get recent assessments for user within time window."""
        pass

    @abstractmethod
    async def count_by_crisis_level(self, user_id: UUID, crisis_level: str) -> int:
        """Count assessments by crisis level for a user."""
        pass


class SafetyPlanRepository(ABC):
    """Abstract repository for safety plans."""

    @abstractmethod
    async def save(self, plan: SafetyPlan) -> SafetyPlan:
        """Save a safety plan."""
        pass

    @abstractmethod
    async def get_by_id(self, plan_id: UUID) -> SafetyPlan | None:
        """Get plan by ID."""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> list[SafetyPlan]:
        """Get all plans for a user."""
        pass

    @abstractmethod
    async def get_active_by_user(self, user_id: UUID) -> SafetyPlan | None:
        """Get active plan for a user."""
        pass

    @abstractmethod
    async def update(self, plan: SafetyPlan) -> SafetyPlan:
        """Update an existing plan."""
        pass

    @abstractmethod
    async def delete(self, plan_id: UUID) -> bool:
        """Delete a plan by ID."""
        pass


class SafetyIncidentRepository(ABC):
    """Abstract repository for safety incidents."""

    @abstractmethod
    async def save(self, incident: SafetyIncident) -> SafetyIncident:
        """Save a safety incident."""
        pass

    @abstractmethod
    async def get_by_id(self, incident_id: UUID) -> SafetyIncident | None:
        """Get incident by ID."""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyIncident]:
        """Get incidents for a user."""
        pass

    @abstractmethod
    async def get_open(self, user_id: UUID | None = None) -> list[SafetyIncident]:
        """Get open incidents, optionally filtered by user."""
        pass

    @abstractmethod
    async def get_by_status(self, status: IncidentStatus) -> list[SafetyIncident]:
        """Get incidents by status."""
        pass

    @abstractmethod
    async def update(self, incident: SafetyIncident) -> SafetyIncident:
        """Update an existing incident."""
        pass


class UserRiskProfileRepository(ABC):
    """Abstract repository for user risk profiles."""

    @abstractmethod
    async def save(self, profile: UserRiskProfile) -> UserRiskProfile:
        """Save a risk profile."""
        pass

    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> UserRiskProfile | None:
        """Get profile for a user."""
        pass

    @abstractmethod
    async def get_or_create(self, user_id: UUID) -> UserRiskProfile:
        """Get existing profile or create new one."""
        pass

    @abstractmethod
    async def update(self, profile: UserRiskProfile) -> UserRiskProfile:
        """Update an existing profile."""
        pass

    @abstractmethod
    async def get_high_risk_users(self) -> list[UserRiskProfile]:
        """Get all users flagged as high risk."""
        pass


class InMemorySafetyAssessmentRepository(SafetyAssessmentRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._assessments: dict[UUID, SafetyAssessment] = {}
        self._lock = asyncio.Lock()

    async def save(self, assessment: SafetyAssessment) -> SafetyAssessment:
        async with self._lock:
            self._assessments[assessment.assessment_id] = assessment
            logger.debug("assessment_saved", assessment_id=str(assessment.assessment_id))
            return assessment

    async def get_by_id(self, assessment_id: UUID) -> SafetyAssessment | None:
        return self._assessments.get(assessment_id)

    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyAssessment]:
        assessments = [a for a in self._assessments.values() if a.user_id == user_id]
        assessments.sort(key=lambda a: a.created_at, reverse=True)
        return assessments[:limit]

    async def get_by_session(self, session_id: UUID) -> list[SafetyAssessment]:
        return [a for a in self._assessments.values() if a.session_id == session_id]

    async def get_recent(self, user_id: UUID, hours: int = 24) -> list[SafetyAssessment]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self._assessments.values()
                if a.user_id == user_id and a.created_at >= cutoff]

    async def count_by_crisis_level(self, user_id: UUID, crisis_level: str) -> int:
        return sum(1 for a in self._assessments.values()
                   if a.user_id == user_id and a.crisis_level == crisis_level)


class InMemorySafetyPlanRepository(SafetyPlanRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._plans: dict[UUID, SafetyPlan] = {}
        self._lock = asyncio.Lock()

    async def save(self, plan: SafetyPlan) -> SafetyPlan:
        async with self._lock:
            self._plans[plan.plan_id] = plan
            logger.debug("plan_saved", plan_id=str(plan.plan_id))
            return plan

    async def get_by_id(self, plan_id: UUID) -> SafetyPlan | None:
        return self._plans.get(plan_id)

    async def get_by_user(self, user_id: UUID) -> list[SafetyPlan]:
        return [p for p in self._plans.values() if p.user_id == user_id]

    async def get_active_by_user(self, user_id: UUID) -> SafetyPlan | None:
        for plan in self._plans.values():
            if plan.user_id == user_id and plan.status == SafetyPlanStatus.ACTIVE:
                return plan
        return None

    async def update(self, plan: SafetyPlan) -> SafetyPlan:
        async with self._lock:
            if plan.plan_id not in self._plans:
                raise EntityNotFoundError("SafetyPlan", plan.plan_id)
            self._plans[plan.plan_id] = plan
            return plan

    async def delete(self, plan_id: UUID) -> bool:
        async with self._lock:
            if plan_id in self._plans:
                del self._plans[plan_id]
                return True
            return False


class InMemorySafetyIncidentRepository(SafetyIncidentRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._incidents: dict[UUID, SafetyIncident] = {}
        self._lock = asyncio.Lock()

    async def save(self, incident: SafetyIncident) -> SafetyIncident:
        async with self._lock:
            self._incidents[incident.incident_id] = incident
            logger.debug("incident_saved", incident_id=str(incident.incident_id))
            return incident

    async def get_by_id(self, incident_id: UUID) -> SafetyIncident | None:
        return self._incidents.get(incident_id)

    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyIncident]:
        incidents = [i for i in self._incidents.values() if i.user_id == user_id]
        incidents.sort(key=lambda i: i.created_at, reverse=True)
        return incidents[:limit]

    async def get_open(self, user_id: UUID | None = None) -> list[SafetyIncident]:
        open_statuses = {IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED, IncidentStatus.IN_PROGRESS}
        incidents = [i for i in self._incidents.values() if i.status in open_statuses]
        if user_id:
            incidents = [i for i in incidents if i.user_id == user_id]
        return incidents

    async def get_by_status(self, status: IncidentStatus) -> list[SafetyIncident]:
        return [i for i in self._incidents.values() if i.status == status]

    async def update(self, incident: SafetyIncident) -> SafetyIncident:
        async with self._lock:
            if incident.incident_id not in self._incidents:
                raise EntityNotFoundError("SafetyIncident", incident.incident_id)
            self._incidents[incident.incident_id] = incident
            return incident


class InMemoryUserRiskProfileRepository(UserRiskProfileRepository):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        self._profiles: dict[UUID, UserRiskProfile] = {}
        self._lock = asyncio.Lock()

    async def save(self, profile: UserRiskProfile) -> UserRiskProfile:
        async with self._lock:
            self._profiles[profile.user_id] = profile
            logger.debug("profile_saved", user_id=str(profile.user_id))
            return profile

    async def get_by_user(self, user_id: UUID) -> UserRiskProfile | None:
        return self._profiles.get(user_id)

    async def get_or_create(self, user_id: UUID) -> UserRiskProfile:
        async with self._lock:
            if user_id not in self._profiles:
                self._profiles[user_id] = UserRiskProfile(user_id=user_id)
            return self._profiles[user_id]

    async def update(self, profile: UserRiskProfile) -> UserRiskProfile:
        async with self._lock:
            self._profiles[profile.user_id] = profile
            return profile

    async def get_high_risk_users(self) -> list[UserRiskProfile]:
        return [p for p in self._profiles.values() if p.high_risk_flag]


class SafetyRepositoryFactory:
    """Factory for creating repository instances."""

    def __init__(self, config: SafetyRepositoryConfig | None = None) -> None:
        self._config = config or SafetyRepositoryConfig()
        self._assessment_repo: SafetyAssessmentRepository | None = None
        self._plan_repo: SafetyPlanRepository | None = None
        self._incident_repo: SafetyIncidentRepository | None = None
        self._profile_repo: UserRiskProfileRepository | None = None

    def get_assessment_repository(self) -> SafetyAssessmentRepository:
        """Get or create assessment repository."""
        if self._assessment_repo is None:
            self._assessment_repo = InMemorySafetyAssessmentRepository()
        return self._assessment_repo

    def get_plan_repository(self) -> SafetyPlanRepository:
        """Get or create safety plan repository."""
        if self._plan_repo is None:
            self._plan_repo = InMemorySafetyPlanRepository()
        return self._plan_repo

    def get_incident_repository(self) -> SafetyIncidentRepository:
        """Get or create incident repository."""
        if self._incident_repo is None:
            self._incident_repo = InMemorySafetyIncidentRepository()
        return self._incident_repo

    def get_profile_repository(self) -> UserRiskProfileRepository:
        """Get or create profile repository."""
        if self._profile_repo is None:
            self._profile_repo = InMemoryUserRiskProfileRepository()
        return self._profile_repo


_factory: SafetyRepositoryFactory | None = None


def get_repository_factory() -> SafetyRepositoryFactory:
    """Get singleton repository factory."""
    global _factory
    if _factory is None:
        _factory = SafetyRepositoryFactory()
    return _factory


def reset_repositories() -> None:
    """Reset all repositories (for testing)."""
    global _factory
    _factory = None

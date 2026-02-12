"""
Test fixtures for Solace-AI Safety Service.
In-memory repository implementations for use in tests only.

These classes were moved from production code to keep the production
repository layer strictly PostgreSQL-backed.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone, timedelta
from uuid import UUID

import structlog

from services.safety_service.src.domain.entities import (
    SafetyAssessment, SafetyPlan, SafetyIncident, UserRiskProfile,
    SafetyPlanStatus, IncidentStatus, IncidentSeverity,
)
from services.safety_service.src.infrastructure.repository import (
    SafetyAssessmentRepository,
    SafetyPlanRepository,
    SafetyIncidentRepository,
    UserRiskProfileRepository,
    EntityNotFoundError,
)

logger = structlog.get_logger(__name__)


class InMemorySafetyAssessmentRepository(SafetyAssessmentRepository):
    """In-memory implementation for testing."""

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
    """In-memory implementation for testing."""

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
    """In-memory implementation for testing."""

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
    """In-memory implementation for testing."""

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

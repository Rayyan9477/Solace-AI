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
_postgres_client: Any | None = None


def configure_repository_factory(
    config: SafetyRepositoryConfig | None = None,
    postgres_client: Any | None = None,
) -> None:
    """Configure the repository factory with optional PostgreSQL client."""
    global _factory, _postgres_client
    _factory = None  # Reset to pick up new config
    _postgres_client = postgres_client


def get_repository_factory(
    config: SafetyRepositoryConfig | None = None,
) -> SafetyRepositoryFactory:
    """Get singleton repository factory.

    Uses PostgreSQL repositories when config.use_postgres is True
    and a PostgresClient has been configured via configure_repository_factory().
    """
    global _factory, _postgres_client
    if _factory is None:
        repo_config = config or SafetyRepositoryConfig()
        if repo_config.use_postgres and _postgres_client is not None:
            _factory = PostgresSafetyRepositoryFactory(
                config=repo_config,
                postgres_client=_postgres_client,
                schema=repo_config.db_schema,
            )
            logger.info("safety_repository_factory_created", type="postgres")
        else:
            _factory = SafetyRepositoryFactory(config=repo_config)
            logger.info("safety_repository_factory_created", type="in_memory")
    return _factory


def reset_repositories() -> None:
    """Reset all repositories (for testing)."""
    global _factory, _postgres_client
    _factory = None
    _postgres_client = None


# =============================================================================
# PostgreSQL Repository Implementations
# =============================================================================

try:
    from solace_infrastructure.postgres import PostgresClient, PostgresSettings
    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False
    PostgresClient = None
    PostgresSettings = None


def _serialize_entity(entity: Any) -> dict[str, Any]:
    """Serialize a Pydantic entity to a dictionary for database storage."""
    if hasattr(entity, "model_dump"):
        data = entity.model_dump()
    elif hasattr(entity, "dict"):
        data = entity.dict()
    else:
        data = dict(entity)
    # Convert UUID and datetime to serializable types
    for key, value in data.items():
        if isinstance(value, UUID):
            data[key] = str(value)
        elif isinstance(value, datetime):
            data[key] = value.isoformat()
    return data


class PostgresSafetyAssessmentRepository(SafetyAssessmentRepository):
    """PostgreSQL implementation of safety assessment repository."""

    def __init__(self, client: Any, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.safety_assessments"

    async def save(self, assessment: SafetyAssessment) -> SafetyAssessment:
        """Save a safety assessment to PostgreSQL."""
        import json
        query = f"""
            INSERT INTO {self._table} (
                assessment_id, user_id, session_id, assessment_type, content_assessed,
                risk_score, crisis_level, is_safe, risk_factors, protective_factors,
                trigger_indicators, detection_layers_triggered, recommended_action,
                requires_escalation, requires_human_review, detection_time_ms, context,
                created_at, reviewed_at, reviewed_by, review_notes
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
            )
            ON CONFLICT (assessment_id) DO UPDATE SET
                risk_score = EXCLUDED.risk_score,
                crisis_level = EXCLUDED.crisis_level,
                is_safe = EXCLUDED.is_safe,
                reviewed_at = EXCLUDED.reviewed_at,
                reviewed_by = EXCLUDED.reviewed_by,
                review_notes = EXCLUDED.review_notes
            RETURNING *
        """
        async with self._client.acquire() as conn:
            await conn.fetchrow(
                query,
                assessment.assessment_id,
                assessment.user_id,
                assessment.session_id,
                assessment.assessment_type.value,
                assessment.content_assessed,
                float(assessment.risk_score),
                assessment.crisis_level,
                assessment.is_safe,
                json.dumps(assessment.risk_factors),
                json.dumps(assessment.protective_factors),
                assessment.trigger_indicators,
                assessment.detection_layers_triggered,
                assessment.recommended_action,
                assessment.requires_escalation,
                assessment.requires_human_review,
                assessment.detection_time_ms,
                json.dumps(assessment.context),
                assessment.created_at,
                assessment.reviewed_at,
                assessment.reviewed_by,
                assessment.review_notes,
            )
            logger.debug("assessment_saved_postgres", assessment_id=str(assessment.assessment_id))
            return assessment

    async def get_by_id(self, assessment_id: UUID) -> SafetyAssessment | None:
        """Get assessment by ID from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE assessment_id = $1"
        async with self._client.acquire() as conn:
            row = await conn.fetchrow(query, assessment_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyAssessment]:
        """Get assessments for a user from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_session(self, session_id: UUID) -> list[SafetyAssessment]:
        """Get assessments for a session from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE session_id = $1
            ORDER BY created_at ASC
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, session_id)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_recent(self, user_id: UUID, hours: int = 24) -> list[SafetyAssessment]:
        """Get recent assessments for user within time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND created_at >= $2
            ORDER BY created_at DESC
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, user_id, cutoff)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def count_by_crisis_level(self, user_id: UUID, crisis_level: str) -> int:
        """Count assessments by crisis level for a user."""
        query = f"""
            SELECT COUNT(*) FROM {self._table}
            WHERE user_id = $1 AND crisis_level = $2
        """
        async with self._client.acquire() as conn:
            count = await conn.fetchval(query, user_id, crisis_level)
            return count or 0

    def _row_to_entity(self, row: dict[str, Any]) -> SafetyAssessment:
        """Convert a database row to a SafetyAssessment entity."""
        import json
        return SafetyAssessment(
            assessment_id=row["assessment_id"],
            user_id=row["user_id"],
            session_id=row.get("session_id"),
            assessment_type=AssessmentType(row["assessment_type"]),
            content_assessed=row["content_assessed"],
            risk_score=row["risk_score"],
            crisis_level=row["crisis_level"],
            is_safe=row["is_safe"],
            risk_factors=json.loads(row["risk_factors"]) if isinstance(row["risk_factors"], str) else row["risk_factors"],
            protective_factors=json.loads(row["protective_factors"]) if isinstance(row["protective_factors"], str) else row["protective_factors"],
            trigger_indicators=row["trigger_indicators"],
            detection_layers_triggered=row["detection_layers_triggered"],
            recommended_action=row["recommended_action"],
            requires_escalation=row["requires_escalation"],
            requires_human_review=row["requires_human_review"],
            detection_time_ms=row["detection_time_ms"],
            context=json.loads(row["context"]) if isinstance(row["context"], str) else row["context"],
            created_at=row["created_at"],
            reviewed_at=row.get("reviewed_at"),
            reviewed_by=row.get("reviewed_by"),
            review_notes=row.get("review_notes"),
        )


class PostgresSafetyPlanRepository(SafetyPlanRepository):
    """PostgreSQL implementation of safety plan repository."""

    def __init__(self, client: Any, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.safety_plans"

    async def save(self, plan: SafetyPlan) -> SafetyPlan:
        """Save a safety plan to PostgreSQL."""
        import json
        query = f"""
            INSERT INTO {self._table} (
                plan_id, user_id, status, version, warning_signs, coping_strategies,
                emergency_contacts, safe_environment_actions, reasons_to_live,
                professional_resources, clinician_notes, last_reviewed_at, last_reviewed_by,
                next_review_due, created_at, updated_at, expires_at, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            ON CONFLICT (plan_id) DO UPDATE SET
                status = EXCLUDED.status,
                version = EXCLUDED.version,
                warning_signs = EXCLUDED.warning_signs,
                coping_strategies = EXCLUDED.coping_strategies,
                emergency_contacts = EXCLUDED.emergency_contacts,
                safe_environment_actions = EXCLUDED.safe_environment_actions,
                reasons_to_live = EXCLUDED.reasons_to_live,
                professional_resources = EXCLUDED.professional_resources,
                clinician_notes = EXCLUDED.clinician_notes,
                last_reviewed_at = EXCLUDED.last_reviewed_at,
                last_reviewed_by = EXCLUDED.last_reviewed_by,
                next_review_due = EXCLUDED.next_review_due,
                updated_at = EXCLUDED.updated_at,
                expires_at = EXCLUDED.expires_at,
                metadata = EXCLUDED.metadata
            RETURNING *
        """
        async with self._client.acquire() as conn:
            await conn.fetchrow(
                query,
                plan.plan_id,
                plan.user_id,
                plan.status.value,
                plan.version,
                json.dumps([_serialize_entity(ws) for ws in plan.warning_signs]),
                json.dumps([_serialize_entity(cs) for cs in plan.coping_strategies]),
                json.dumps([_serialize_entity(ec) for ec in plan.emergency_contacts]),
                json.dumps([_serialize_entity(sea) for sea in plan.safe_environment_actions]),
                plan.reasons_to_live,
                json.dumps(plan.professional_resources),
                plan.clinician_notes,
                plan.last_reviewed_at,
                plan.last_reviewed_by,
                plan.next_review_due,
                plan.created_at,
                plan.updated_at,
                plan.expires_at,
                json.dumps(plan.metadata),
            )
            logger.debug("plan_saved_postgres", plan_id=str(plan.plan_id))
            return plan

    async def get_by_id(self, plan_id: UUID) -> SafetyPlan | None:
        """Get plan by ID from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE plan_id = $1"
        async with self._client.acquire() as conn:
            row = await conn.fetchrow(query, plan_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_by_user(self, user_id: UUID) -> list[SafetyPlan]:
        """Get all plans for a user from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1
            ORDER BY created_at DESC
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_active_by_user(self, user_id: UUID) -> SafetyPlan | None:
        """Get active plan for a user from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND status = 'ACTIVE'
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with self._client.acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def update(self, plan: SafetyPlan) -> SafetyPlan:
        """Update an existing plan."""
        return await self.save(plan)

    async def delete(self, plan_id: UUID) -> bool:
        """Delete a plan by ID from PostgreSQL."""
        query = f"DELETE FROM {self._table} WHERE plan_id = $1"
        async with self._client.acquire() as conn:
            result = await conn.execute(query, plan_id)
            return result == "DELETE 1"

    def _row_to_entity(self, row: dict[str, Any]) -> SafetyPlan:
        """Convert a database row to a SafetyPlan entity."""
        import json
        from ..domain.entities import WarningSign, CopingStrategy, EmergencyContact, SafeEnvironmentAction
        warning_signs_data = json.loads(row["warning_signs"]) if isinstance(row["warning_signs"], str) else row["warning_signs"]
        coping_strategies_data = json.loads(row["coping_strategies"]) if isinstance(row["coping_strategies"], str) else row["coping_strategies"]
        emergency_contacts_data = json.loads(row["emergency_contacts"]) if isinstance(row["emergency_contacts"], str) else row["emergency_contacts"]
        safe_env_actions_data = json.loads(row["safe_environment_actions"]) if isinstance(row["safe_environment_actions"], str) else row["safe_environment_actions"]
        professional_resources_data = json.loads(row["professional_resources"]) if isinstance(row["professional_resources"], str) else row["professional_resources"]
        metadata_data = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]

        return SafetyPlan(
            plan_id=row["plan_id"],
            user_id=row["user_id"],
            status=SafetyPlanStatus(row["status"]),
            version=row["version"],
            warning_signs=[WarningSign(**ws) for ws in warning_signs_data],
            coping_strategies=[CopingStrategy(**cs) for cs in coping_strategies_data],
            emergency_contacts=[EmergencyContact(**ec) for ec in emergency_contacts_data],
            safe_environment_actions=[SafeEnvironmentAction(**sea) for sea in safe_env_actions_data],
            reasons_to_live=row["reasons_to_live"],
            professional_resources=professional_resources_data,
            clinician_notes=row.get("clinician_notes"),
            last_reviewed_at=row.get("last_reviewed_at"),
            last_reviewed_by=row.get("last_reviewed_by"),
            next_review_due=row.get("next_review_due"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row.get("expires_at"),
            metadata=metadata_data,
        )


class PostgresSafetyIncidentRepository(SafetyIncidentRepository):
    """PostgreSQL implementation of safety incident repository."""

    def __init__(self, client: Any, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.safety_incidents"

    async def save(self, incident: SafetyIncident) -> SafetyIncident:
        """Save a safety incident to PostgreSQL."""
        import json
        query = f"""
            INSERT INTO {self._table} (
                incident_id, user_id, session_id, assessment_id, escalation_id,
                severity, status, crisis_level, description, trigger_indicators,
                risk_factors, actions_taken, resources_provided, assigned_clinician_id,
                resolution_notes, follow_up_required, follow_up_due, created_at,
                acknowledged_at, resolved_at, closed_at, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )
            ON CONFLICT (incident_id) DO UPDATE SET
                status = EXCLUDED.status,
                actions_taken = EXCLUDED.actions_taken,
                resources_provided = EXCLUDED.resources_provided,
                assigned_clinician_id = EXCLUDED.assigned_clinician_id,
                resolution_notes = EXCLUDED.resolution_notes,
                follow_up_required = EXCLUDED.follow_up_required,
                follow_up_due = EXCLUDED.follow_up_due,
                acknowledged_at = EXCLUDED.acknowledged_at,
                resolved_at = EXCLUDED.resolved_at,
                closed_at = EXCLUDED.closed_at,
                metadata = EXCLUDED.metadata
            RETURNING *
        """
        async with self._client.acquire() as conn:
            await conn.fetchrow(
                query,
                incident.incident_id,
                incident.user_id,
                incident.session_id,
                incident.assessment_id,
                incident.escalation_id,
                incident.severity.value,
                incident.status.value,
                incident.crisis_level,
                incident.description,
                incident.trigger_indicators,
                json.dumps(incident.risk_factors),
                incident.actions_taken,
                incident.resources_provided,
                incident.assigned_clinician_id,
                incident.resolution_notes,
                incident.follow_up_required,
                incident.follow_up_due,
                incident.created_at,
                incident.acknowledged_at,
                incident.resolved_at,
                incident.closed_at,
                json.dumps(incident.metadata),
            )
            logger.debug("incident_saved_postgres", incident_id=str(incident.incident_id))
            return incident

    async def get_by_id(self, incident_id: UUID) -> SafetyIncident | None:
        """Get incident by ID from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE incident_id = $1"
        async with self._client.acquire() as conn:
            row = await conn.fetchrow(query, incident_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_by_user(self, user_id: UUID, limit: int = 100) -> list[SafetyIncident]:
        """Get incidents for a user from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_open(self, user_id: UUID | None = None) -> list[SafetyIncident]:
        """Get open incidents from PostgreSQL."""
        if user_id:
            query = f"""
                SELECT * FROM {self._table}
                WHERE user_id = $1 AND status IN ('OPEN', 'ACKNOWLEDGED', 'IN_PROGRESS')
                ORDER BY created_at DESC
            """
            async with self._client.acquire() as conn:
                rows = await conn.fetch(query, user_id)
        else:
            query = f"""
                SELECT * FROM {self._table}
                WHERE status IN ('OPEN', 'ACKNOWLEDGED', 'IN_PROGRESS')
                ORDER BY created_at DESC
            """
            async with self._client.acquire() as conn:
                rows = await conn.fetch(query)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_status(self, status: IncidentStatus) -> list[SafetyIncident]:
        """Get incidents by status from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE status = $1
            ORDER BY created_at DESC
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query, status.value)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def update(self, incident: SafetyIncident) -> SafetyIncident:
        """Update an existing incident."""
        return await self.save(incident)

    def _row_to_entity(self, row: dict[str, Any]) -> SafetyIncident:
        """Convert a database row to a SafetyIncident entity."""
        import json
        risk_factors_data = json.loads(row["risk_factors"]) if isinstance(row["risk_factors"], str) else row["risk_factors"]
        metadata_data = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]

        return SafetyIncident(
            incident_id=row["incident_id"],
            user_id=row["user_id"],
            session_id=row.get("session_id"),
            assessment_id=row.get("assessment_id"),
            escalation_id=row.get("escalation_id"),
            severity=IncidentSeverity(row["severity"]),
            status=IncidentStatus(row["status"]),
            crisis_level=row["crisis_level"],
            description=row["description"],
            trigger_indicators=row["trigger_indicators"],
            risk_factors=risk_factors_data,
            actions_taken=row["actions_taken"],
            resources_provided=row["resources_provided"],
            assigned_clinician_id=row.get("assigned_clinician_id"),
            resolution_notes=row.get("resolution_notes"),
            follow_up_required=row["follow_up_required"],
            follow_up_due=row.get("follow_up_due"),
            created_at=row["created_at"],
            acknowledged_at=row.get("acknowledged_at"),
            resolved_at=row.get("resolved_at"),
            closed_at=row.get("closed_at"),
            metadata=metadata_data,
        )


class PostgresUserRiskProfileRepository(UserRiskProfileRepository):
    """PostgreSQL implementation of user risk profile repository."""

    def __init__(self, client: Any, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.user_risk_profiles"

    async def save(self, profile: UserRiskProfile) -> UserRiskProfile:
        """Save a risk profile to PostgreSQL."""
        query = f"""
            INSERT INTO {self._table} (
                profile_id, user_id, baseline_risk_level, current_risk_level,
                total_assessments, total_incidents, crisis_events_count, escalations_count,
                last_crisis_at, last_assessment_at, high_risk_flag, recent_escalation,
                active_safety_plan_id, risk_trend, protective_factors_count, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (user_id) DO UPDATE SET
                baseline_risk_level = EXCLUDED.baseline_risk_level,
                current_risk_level = EXCLUDED.current_risk_level,
                total_assessments = EXCLUDED.total_assessments,
                total_incidents = EXCLUDED.total_incidents,
                crisis_events_count = EXCLUDED.crisis_events_count,
                escalations_count = EXCLUDED.escalations_count,
                last_crisis_at = EXCLUDED.last_crisis_at,
                last_assessment_at = EXCLUDED.last_assessment_at,
                high_risk_flag = EXCLUDED.high_risk_flag,
                recent_escalation = EXCLUDED.recent_escalation,
                active_safety_plan_id = EXCLUDED.active_safety_plan_id,
                risk_trend = EXCLUDED.risk_trend,
                protective_factors_count = EXCLUDED.protective_factors_count,
                updated_at = EXCLUDED.updated_at
            RETURNING *
        """
        async with self._client.acquire() as conn:
            await conn.fetchrow(
                query,
                profile.profile_id,
                profile.user_id,
                profile.baseline_risk_level,
                profile.current_risk_level,
                profile.total_assessments,
                profile.total_incidents,
                profile.crisis_events_count,
                profile.escalations_count,
                profile.last_crisis_at,
                profile.last_assessment_at,
                profile.high_risk_flag,
                profile.recent_escalation,
                profile.active_safety_plan_id,
                profile.risk_trend,
                profile.protective_factors_count,
                profile.created_at,
                profile.updated_at,
            )
            logger.debug("profile_saved_postgres", user_id=str(profile.user_id))
            return profile

    async def get_by_user(self, user_id: UUID) -> UserRiskProfile | None:
        """Get profile for a user from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE user_id = $1"
        async with self._client.acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_or_create(self, user_id: UUID) -> UserRiskProfile:
        """Get existing profile or create new one."""
        profile = await self.get_by_user(user_id)
        if profile is not None:
            return profile
        new_profile = UserRiskProfile(user_id=user_id)
        return await self.save(new_profile)

    async def update(self, profile: UserRiskProfile) -> UserRiskProfile:
        """Update an existing profile."""
        return await self.save(profile)

    async def get_high_risk_users(self) -> list[UserRiskProfile]:
        """Get all users flagged as high risk from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE high_risk_flag = TRUE
            ORDER BY updated_at DESC
        """
        async with self._client.acquire() as conn:
            rows = await conn.fetch(query)
            return [self._row_to_entity(dict(row)) for row in rows]

    def _row_to_entity(self, row: dict[str, Any]) -> UserRiskProfile:
        """Convert a database row to a UserRiskProfile entity."""
        return UserRiskProfile(
            profile_id=row["profile_id"],
            user_id=row["user_id"],
            baseline_risk_level=row["baseline_risk_level"],
            current_risk_level=row["current_risk_level"],
            total_assessments=row["total_assessments"],
            total_incidents=row["total_incidents"],
            crisis_events_count=row["crisis_events_count"],
            escalations_count=row["escalations_count"],
            last_crisis_at=row.get("last_crisis_at"),
            last_assessment_at=row.get("last_assessment_at"),
            high_risk_flag=row["high_risk_flag"],
            recent_escalation=row["recent_escalation"],
            active_safety_plan_id=row.get("active_safety_plan_id"),
            risk_trend=row["risk_trend"],
            protective_factors_count=row["protective_factors_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresSafetyRepositoryFactory(SafetyRepositoryFactory):
    """Factory for creating PostgreSQL repository instances."""

    def __init__(
        self,
        config: SafetyRepositoryConfig | None = None,
        postgres_client: Any | None = None,
        schema: str = "public",
    ) -> None:
        super().__init__(config)
        self._postgres_client = postgres_client
        self._schema = schema

    def get_assessment_repository(self) -> SafetyAssessmentRepository:
        """Get PostgreSQL assessment repository."""
        if self._assessment_repo is None:
            if self._postgres_client and _POSTGRES_AVAILABLE:
                self._assessment_repo = PostgresSafetyAssessmentRepository(
                    self._postgres_client, self._schema
                )
            else:
                self._assessment_repo = InMemorySafetyAssessmentRepository()
        return self._assessment_repo

    def get_plan_repository(self) -> SafetyPlanRepository:
        """Get PostgreSQL safety plan repository."""
        if self._plan_repo is None:
            if self._postgres_client and _POSTGRES_AVAILABLE:
                self._plan_repo = PostgresSafetyPlanRepository(
                    self._postgres_client, self._schema
                )
            else:
                self._plan_repo = InMemorySafetyPlanRepository()
        return self._plan_repo

    def get_incident_repository(self) -> SafetyIncidentRepository:
        """Get PostgreSQL incident repository."""
        if self._incident_repo is None:
            if self._postgres_client and _POSTGRES_AVAILABLE:
                self._incident_repo = PostgresSafetyIncidentRepository(
                    self._postgres_client, self._schema
                )
            else:
                self._incident_repo = InMemorySafetyIncidentRepository()
        return self._incident_repo

    def get_profile_repository(self) -> UserRiskProfileRepository:
        """Get PostgreSQL profile repository."""
        if self._profile_repo is None:
            if self._postgres_client and _POSTGRES_AVAILABLE:
                self._profile_repo = PostgresUserRiskProfileRepository(
                    self._postgres_client, self._schema
                )
            else:
                self._profile_repo = InMemoryUserRiskProfileRepository()
        return self._profile_repo


def create_postgres_repository_factory(
    postgres_client: Any,
    config: SafetyRepositoryConfig | None = None,
    schema: str = "public",
) -> PostgresSafetyRepositoryFactory:
    """Create a repository factory with PostgreSQL implementations.

    Args:
        postgres_client: Configured PostgresClient instance
        config: Optional repository configuration
        schema: Database schema name (default: public)

    Returns:
        Factory configured to create PostgreSQL repositories
    """
    return PostgresSafetyRepositoryFactory(
        config=config,
        postgres_client=postgres_client,
        schema=schema,
    )

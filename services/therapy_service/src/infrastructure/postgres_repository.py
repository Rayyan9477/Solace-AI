"""
Solace-AI Therapy Service - PostgreSQL Repository Implementations.

PostgreSQL-backed repositories for therapy domain entity persistence.
Uses asyncpg via PostgresClient for async database operations.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

try:
    from solace_infrastructure.database import ConnectionPoolManager
    from solace_infrastructure.feature_flags import FeatureFlags
except ImportError:
    ConnectionPoolManager = None
    FeatureFlags = None

from ..domain.entities import (
    HomeworkEntity,
    InterventionEntity,
    TherapySessionEntity,
    TreatmentGoalEntity,
    TreatmentPlanEntity,
)
from ..domain.value_objects import OutcomeMeasure, Technique
from ..schemas import (
    DeliveryMode,
    DifficultyLevel,
    GoalStatus,
    HomeworkStatus,
    OutcomeInstrument,
    ResponseStatus,
    RiskLevel,
    SessionPhase,
    SeverityLevel,
    SteppedCareLevel,
    TechniqueCategory,
    TherapyModality,
    TreatmentPhase,
)
from .repository import Repository

if TYPE_CHECKING:
    from solace_infrastructure.postgres import PostgresClient

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Treatment Plan Repository
# ---------------------------------------------------------------------------

class PostgresTreatmentPlanRepository(Repository[TreatmentPlanEntity]):
    """PostgreSQL implementation of treatment plan repository."""

    POOL_NAME = "therapy_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._table = f"{schema}.treatment_plans"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._acquire()
        raise Exception("No database connection available.")

    async def get(self, plan_id: UUID) -> TreatmentPlanEntity | None:
        """Get treatment plan by ID."""
        query = f"SELECT * FROM {self._table} WHERE plan_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, plan_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def save(self, plan: TreatmentPlanEntity) -> TreatmentPlanEntity:
        """Save or update treatment plan."""
        query = f"""
            INSERT INTO {self._table} (
                plan_id, user_id, primary_diagnosis, secondary_diagnoses,
                severity, stepped_care_level, primary_modality, adjunct_modalities,
                current_phase, phase_sessions_completed, total_sessions_completed,
                session_frequency_per_week, response_status, goals,
                skills_acquired, skills_in_progress, contraindications,
                baseline_phq9, latest_phq9, baseline_gad7, latest_gad7,
                created_at, updated_at, target_completion, review_date,
                is_active, termination_reason, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23,
                $24, $25, $26, $27, $28
            )
            ON CONFLICT (plan_id) DO UPDATE SET
                primary_diagnosis = EXCLUDED.primary_diagnosis,
                secondary_diagnoses = EXCLUDED.secondary_diagnoses,
                severity = EXCLUDED.severity,
                stepped_care_level = EXCLUDED.stepped_care_level,
                primary_modality = EXCLUDED.primary_modality,
                adjunct_modalities = EXCLUDED.adjunct_modalities,
                current_phase = EXCLUDED.current_phase,
                phase_sessions_completed = EXCLUDED.phase_sessions_completed,
                total_sessions_completed = EXCLUDED.total_sessions_completed,
                session_frequency_per_week = EXCLUDED.session_frequency_per_week,
                response_status = EXCLUDED.response_status,
                goals = EXCLUDED.goals,
                skills_acquired = EXCLUDED.skills_acquired,
                skills_in_progress = EXCLUDED.skills_in_progress,
                contraindications = EXCLUDED.contraindications,
                baseline_phq9 = EXCLUDED.baseline_phq9,
                latest_phq9 = EXCLUDED.latest_phq9,
                baseline_gad7 = EXCLUDED.baseline_gad7,
                latest_gad7 = EXCLUDED.latest_gad7,
                updated_at = EXCLUDED.updated_at,
                target_completion = EXCLUDED.target_completion,
                review_date = EXCLUDED.review_date,
                is_active = EXCLUDED.is_active,
                termination_reason = EXCLUDED.termination_reason,
                version = EXCLUDED.version
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                plan.plan_id,
                plan.user_id,
                plan.primary_diagnosis,
                json.dumps(plan.secondary_diagnoses),
                plan.severity.value,
                plan.stepped_care_level.value,
                plan.primary_modality.value,
                json.dumps([m.value for m in plan.adjunct_modalities]),
                plan.current_phase.value,
                plan.phase_sessions_completed,
                plan.total_sessions_completed,
                plan.session_frequency_per_week,
                plan.response_status.value,
                json.dumps([self._goal_to_dict(g) for g in plan.goals]),
                json.dumps(plan.skills_acquired),
                json.dumps(plan.skills_in_progress),
                json.dumps(plan.contraindications),
                plan.baseline_phq9,
                plan.latest_phq9,
                plan.baseline_gad7,
                plan.latest_gad7,
                plan.created_at,
                plan.updated_at,
                plan.target_completion,
                plan.review_date,
                plan.is_active,
                plan.termination_reason,
                plan.version,
            )
            logger.debug("treatment_plan_saved_postgres", plan_id=str(plan.plan_id))
            return plan

    async def delete(self, plan_id: UUID) -> bool:
        """Delete treatment plan."""
        query = f"DELETE FROM {self._table} WHERE plan_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, plan_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("treatment_plan_deleted_postgres", plan_id=str(plan_id))
            return deleted

    async def get_by_user(self, user_id: UUID) -> list[TreatmentPlanEntity]:
        """Get all treatment plans for a user."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 ORDER BY created_at DESC
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_active_by_user(self, user_id: UUID) -> list[TreatmentPlanEntity]:
        """Get active treatment plans for a user."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND is_active = true
            ORDER BY created_at DESC
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def find_by_criteria(
        self,
        user_id: UUID | None = None,
        modality: str | None = None,
        is_active: bool | None = None,
        min_sessions: int | None = None,
    ) -> list[TreatmentPlanEntity]:
        """Find treatment plans by criteria."""
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if user_id is not None:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if modality is not None:
            conditions.append(f"primary_modality = ${idx}")
            params.append(modality)
            idx += 1
        if is_active is not None:
            conditions.append(f"is_active = ${idx}")
            params.append(is_active)
            idx += 1
        if min_sessions is not None:
            conditions.append(f"total_sessions_completed >= ${idx}")
            params.append(min_sessions)
            idx += 1

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM {self._table}{where} ORDER BY created_at DESC"
        async with self._acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def count(self) -> int:
        """Count total treatment plans."""
        query = f"SELECT COUNT(*) FROM {self._table}"
        async with self._acquire() as conn:
            return await conn.fetchval(query) or 0

    # -- Serialization helpers --

    @staticmethod
    def _goal_to_dict(goal: TreatmentGoalEntity) -> dict[str, Any]:
        return {
            "goal_id": str(goal.goal_id),
            "description": goal.description,
            "target_date": goal.target_date.isoformat() if goal.target_date else None,
            "status": goal.status.value,
            "progress_percentage": goal.progress_percentage,
            "milestones": goal.milestones,
            "completed_milestones": goal.completed_milestones,
            "notes": goal.notes,
            "created_at": goal.created_at.isoformat(),
            "updated_at": goal.updated_at.isoformat(),
            "version": goal.version,
        }

    @staticmethod
    def _dict_to_goal(data: dict[str, Any]) -> TreatmentGoalEntity:
        return TreatmentGoalEntity(
            goal_id=UUID(data["goal_id"]),
            description=data.get("description", ""),
            target_date=datetime.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            status=GoalStatus(data.get("status", "not_started")),
            progress_percentage=data.get("progress_percentage", 0),
            milestones=data.get("milestones", []),
            completed_milestones=data.get("completed_milestones", []),
            notes=data.get("notes", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            version=data.get("version", 1),
        )

    def _row_to_entity(self, row: dict[str, Any]) -> TreatmentPlanEntity:
        secondary = row.get("secondary_diagnoses", [])
        if isinstance(secondary, str):
            secondary = json.loads(secondary)

        adjunct = row.get("adjunct_modalities", [])
        if isinstance(adjunct, str):
            adjunct = json.loads(adjunct)

        goals_raw = row.get("goals", [])
        if isinstance(goals_raw, str):
            goals_raw = json.loads(goals_raw)

        skills_acq = row.get("skills_acquired", [])
        if isinstance(skills_acq, str):
            skills_acq = json.loads(skills_acq)

        skills_ip = row.get("skills_in_progress", [])
        if isinstance(skills_ip, str):
            skills_ip = json.loads(skills_ip)

        contras = row.get("contraindications", [])
        if isinstance(contras, str):
            contras = json.loads(contras)

        return TreatmentPlanEntity(
            plan_id=row["plan_id"],
            user_id=row["user_id"],
            primary_diagnosis=row.get("primary_diagnosis", ""),
            secondary_diagnoses=secondary,
            severity=SeverityLevel(row["severity"]) if row.get("severity") else SeverityLevel.MODERATE,
            stepped_care_level=SteppedCareLevel(row["stepped_care_level"]) if row.get("stepped_care_level") is not None else SteppedCareLevel.MEDIUM_INTENSITY,
            primary_modality=TherapyModality(row["primary_modality"]) if row.get("primary_modality") else TherapyModality.CBT,
            adjunct_modalities=[TherapyModality(m) for m in adjunct],
            current_phase=TreatmentPhase(row["current_phase"]) if row.get("current_phase") else TreatmentPhase.FOUNDATION,
            phase_sessions_completed=row.get("phase_sessions_completed", 0),
            total_sessions_completed=row.get("total_sessions_completed", 0),
            session_frequency_per_week=row.get("session_frequency_per_week", 1),
            response_status=ResponseStatus(row["response_status"]) if row.get("response_status") else ResponseStatus.NOT_STARTED,
            goals=[self._dict_to_goal(g) for g in goals_raw],
            skills_acquired=skills_acq,
            skills_in_progress=skills_ip,
            contraindications=contras,
            baseline_phq9=row.get("baseline_phq9"),
            latest_phq9=row.get("latest_phq9"),
            baseline_gad7=row.get("baseline_gad7"),
            latest_gad7=row.get("latest_gad7"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            target_completion=row.get("target_completion"),
            review_date=row.get("review_date"),
            is_active=row.get("is_active", True),
            termination_reason=row.get("termination_reason"),
            version=row.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Therapy Session Repository
# ---------------------------------------------------------------------------

class PostgresTherapySessionRepository(Repository[TherapySessionEntity]):
    """PostgreSQL implementation of therapy session repository."""

    POOL_NAME = "therapy_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._table = f"{schema}.therapy_sessions"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._acquire()
        raise Exception("No database connection available.")

    async def get(self, session_id: UUID) -> TherapySessionEntity | None:
        """Get therapy session by ID."""
        query = f"SELECT * FROM {self._table} WHERE session_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, session_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def save(self, session: TherapySessionEntity) -> TherapySessionEntity:
        """Save or update therapy session."""
        query = f"""
            INSERT INTO {self._table} (
                session_id, user_id, treatment_plan_id, session_number,
                current_phase, started_at, ended_at,
                mood_rating_start, mood_rating_end,
                agenda_items, topics_covered, interventions,
                homework_assigned, skills_practiced, insights_gained,
                current_risk, safety_flags, session_rating, alliance_score,
                summary, next_session_focus, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )
            ON CONFLICT (session_id) DO UPDATE SET
                current_phase = EXCLUDED.current_phase,
                ended_at = EXCLUDED.ended_at,
                mood_rating_start = EXCLUDED.mood_rating_start,
                mood_rating_end = EXCLUDED.mood_rating_end,
                agenda_items = EXCLUDED.agenda_items,
                topics_covered = EXCLUDED.topics_covered,
                interventions = EXCLUDED.interventions,
                homework_assigned = EXCLUDED.homework_assigned,
                skills_practiced = EXCLUDED.skills_practiced,
                insights_gained = EXCLUDED.insights_gained,
                current_risk = EXCLUDED.current_risk,
                safety_flags = EXCLUDED.safety_flags,
                session_rating = EXCLUDED.session_rating,
                alliance_score = EXCLUDED.alliance_score,
                summary = EXCLUDED.summary,
                next_session_focus = EXCLUDED.next_session_focus,
                version = EXCLUDED.version
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                session.session_id,
                session.user_id,
                session.treatment_plan_id,
                session.session_number,
                session.current_phase.value,
                session.started_at,
                session.ended_at,
                session.mood_rating_start,
                session.mood_rating_end,
                json.dumps(session.agenda_items),
                json.dumps(session.topics_covered),
                json.dumps([self._intervention_to_dict(i) for i in session.interventions]),
                json.dumps([self._homework_to_dict(h) for h in session.homework_assigned]),
                json.dumps(session.skills_practiced),
                json.dumps(session.insights_gained),
                session.current_risk.value,
                json.dumps(session.safety_flags),
                session.session_rating,
                session.alliance_score,
                session.summary,
                session.next_session_focus,
                session.version,
            )
            logger.debug("therapy_session_saved_postgres", session_id=str(session.session_id))
            return session

    async def delete(self, session_id: UUID) -> bool:
        """Delete therapy session."""
        query = f"DELETE FROM {self._table} WHERE session_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, session_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("therapy_session_deleted_postgres", session_id=str(session_id))
            return deleted

    async def get_by_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[TherapySessionEntity]:
        """Get therapy sessions for a user."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1
            ORDER BY started_at DESC
            LIMIT $2 OFFSET $3
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, limit, offset)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_plan(
        self, plan_id: UUID, limit: int = 100, offset: int = 0
    ) -> list[TherapySessionEntity]:
        """Get therapy sessions for a treatment plan."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE treatment_plan_id = $1
            ORDER BY session_number ASC
            LIMIT $2 OFFSET $3
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, plan_id, limit, offset)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_active_by_user(self, user_id: UUID) -> TherapySessionEntity | None:
        """Get active therapy session for a user."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND ended_at IS NULL
            ORDER BY started_at DESC
            LIMIT 1
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_recent_sessions(
        self, user_id: UUID, days: int = 30
    ) -> list[TherapySessionEntity]:
        """Get recent therapy sessions."""
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND started_at >= $2
            ORDER BY started_at DESC
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, cutoff)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def count_by_plan(self, plan_id: UUID) -> int:
        """Count sessions for a treatment plan."""
        query = f"SELECT COUNT(*) FROM {self._table} WHERE treatment_plan_id = $1"
        async with self._acquire() as conn:
            return await conn.fetchval(query, plan_id) or 0

    async def count(self) -> int:
        """Count total therapy sessions."""
        query = f"SELECT COUNT(*) FROM {self._table}"
        async with self._acquire() as conn:
            return await conn.fetchval(query) or 0

    # -- Serialization helpers --

    @staticmethod
    def _intervention_to_dict(intervention: InterventionEntity) -> dict[str, Any]:
        return {
            "intervention_id": str(intervention.intervention_id),
            "session_id": str(intervention.session_id),
            "technique_id": str(intervention.technique_id),
            "technique_name": intervention.technique_name,
            "modality": intervention.modality.value,
            "phase": intervention.phase.value,
            "started_at": intervention.started_at.isoformat(),
            "completed_at": intervention.completed_at.isoformat() if intervention.completed_at else None,
            "messages_exchanged": intervention.messages_exchanged,
            "completed": intervention.completed,
            "engagement_score": str(intervention.engagement_score),
            "skills_practiced": intervention.skills_practiced,
            "insights_gained": intervention.insights_gained,
            "version": intervention.version,
        }

    @staticmethod
    def _dict_to_intervention(data: dict[str, Any]) -> InterventionEntity:
        return InterventionEntity(
            intervention_id=UUID(data["intervention_id"]),
            session_id=UUID(data["session_id"]),
            technique_id=UUID(data["technique_id"]),
            technique_name=data.get("technique_name", ""),
            modality=TherapyModality(data["modality"]) if data.get("modality") else TherapyModality.CBT,
            phase=SessionPhase(data["phase"]) if data.get("phase") else SessionPhase.WORKING,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now(timezone.utc),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            messages_exchanged=data.get("messages_exchanged", 0),
            completed=data.get("completed", False),
            engagement_score=Decimal(data.get("engagement_score", "0.5")),
            skills_practiced=data.get("skills_practiced", []),
            insights_gained=data.get("insights_gained", []),
            version=data.get("version", 1),
        )

    @staticmethod
    def _homework_to_dict(hw: HomeworkEntity) -> dict[str, Any]:
        return {
            "homework_id": str(hw.homework_id),
            "session_id": str(hw.session_id) if hw.session_id else None,
            "technique_id": str(hw.technique_id) if hw.technique_id else None,
            "title": hw.title,
            "description": hw.description,
            "status": hw.status.value,
            "assigned_at": hw.assigned_at.isoformat(),
            "due_date": hw.due_date.isoformat() if hw.due_date else None,
            "completed_at": hw.completed_at.isoformat() if hw.completed_at else None,
            "completion_notes": hw.completion_notes,
            "rating": hw.rating,
            "version": hw.version,
        }

    @staticmethod
    def _dict_to_homework(data: dict[str, Any]) -> HomeworkEntity:
        return HomeworkEntity(
            homework_id=UUID(data["homework_id"]),
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            technique_id=UUID(data["technique_id"]) if data.get("technique_id") else None,
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=HomeworkStatus(data.get("status", "assigned")),
            assigned_at=datetime.fromisoformat(data["assigned_at"]) if data.get("assigned_at") else datetime.now(timezone.utc),
            due_date=datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            completion_notes=data.get("completion_notes", ""),
            rating=data.get("rating"),
            version=data.get("version", 1),
        )

    def _row_to_entity(self, row: dict[str, Any]) -> TherapySessionEntity:
        agenda = row.get("agenda_items", [])
        if isinstance(agenda, str):
            agenda = json.loads(agenda)

        topics = row.get("topics_covered", [])
        if isinstance(topics, str):
            topics = json.loads(topics)

        interventions_raw = row.get("interventions", [])
        if isinstance(interventions_raw, str):
            interventions_raw = json.loads(interventions_raw)

        homework_raw = row.get("homework_assigned", [])
        if isinstance(homework_raw, str):
            homework_raw = json.loads(homework_raw)

        skills = row.get("skills_practiced", [])
        if isinstance(skills, str):
            skills = json.loads(skills)

        insights = row.get("insights_gained", [])
        if isinstance(insights, str):
            insights = json.loads(insights)

        safety = row.get("safety_flags", [])
        if isinstance(safety, str):
            safety = json.loads(safety)

        return TherapySessionEntity(
            session_id=row["session_id"],
            user_id=row["user_id"],
            treatment_plan_id=row["treatment_plan_id"],
            session_number=row.get("session_number", 1),
            current_phase=SessionPhase(row["current_phase"]) if row.get("current_phase") else SessionPhase.PRE_SESSION,
            started_at=row["started_at"],
            ended_at=row.get("ended_at"),
            mood_rating_start=row.get("mood_rating_start"),
            mood_rating_end=row.get("mood_rating_end"),
            agenda_items=agenda,
            topics_covered=topics,
            interventions=[self._dict_to_intervention(i) for i in interventions_raw],
            homework_assigned=[self._dict_to_homework(h) for h in homework_raw],
            skills_practiced=skills,
            insights_gained=insights,
            current_risk=RiskLevel(row["current_risk"]) if row.get("current_risk") else RiskLevel.NONE,
            safety_flags=safety,
            session_rating=row.get("session_rating"),
            alliance_score=row.get("alliance_score"),
            summary=row.get("summary", ""),
            next_session_focus=row.get("next_session_focus", ""),
            version=row.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Technique Repository
# ---------------------------------------------------------------------------

class PostgresTechniqueRepository:
    """PostgreSQL implementation of technique repository."""

    POOL_NAME = "therapy_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._table = f"{schema}.therapy_techniques"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._acquire()
        raise Exception("No database connection available.")

    async def get(self, technique_id: UUID) -> Technique | None:
        """Get technique by ID."""
        query = f"SELECT * FROM {self._table} WHERE technique_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, technique_id)
            if row is None:
                return None
            return self._row_to_value_object(dict(row))

    async def get_by_name(self, name: str) -> Technique | None:
        """Get technique by name (case-insensitive)."""
        query = f"SELECT * FROM {self._table} WHERE LOWER(name) = LOWER($1)"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, name)
            if row is None:
                return None
            return self._row_to_value_object(dict(row))

    async def save(self, technique: Technique) -> Technique:
        """Save or update technique."""
        query = f"""
            INSERT INTO {self._table} (
                technique_id, name, modality, category, description,
                instructions, duration_minutes, difficulty, delivery_mode,
                requires_homework, contraindications, prerequisites,
                target_symptoms, effectiveness_rating, evidence_level
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            ON CONFLICT (technique_id) DO UPDATE SET
                name = EXCLUDED.name,
                modality = EXCLUDED.modality,
                category = EXCLUDED.category,
                description = EXCLUDED.description,
                instructions = EXCLUDED.instructions,
                duration_minutes = EXCLUDED.duration_minutes,
                difficulty = EXCLUDED.difficulty,
                delivery_mode = EXCLUDED.delivery_mode,
                requires_homework = EXCLUDED.requires_homework,
                contraindications = EXCLUDED.contraindications,
                prerequisites = EXCLUDED.prerequisites,
                target_symptoms = EXCLUDED.target_symptoms,
                effectiveness_rating = EXCLUDED.effectiveness_rating,
                evidence_level = EXCLUDED.evidence_level
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                technique.technique_id,
                technique.name,
                technique.modality.value,
                technique.category.value,
                technique.description,
                technique.instructions,
                technique.duration_minutes,
                technique.difficulty.value,
                technique.delivery_mode.value,
                technique.requires_homework,
                json.dumps(list(technique.contraindications)),
                json.dumps(list(technique.prerequisites)),
                json.dumps(list(technique.target_symptoms)),
                str(technique.effectiveness_rating),
                technique.evidence_level,
            )
            logger.debug("technique_saved_postgres", technique_id=str(technique.technique_id))
            return technique

    async def delete(self, technique_id: UUID) -> bool:
        """Delete technique."""
        query = f"DELETE FROM {self._table} WHERE technique_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, technique_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("technique_deleted_postgres", technique_id=str(technique_id))
            return deleted

    async def get_by_modality(self, modality: str) -> list[Technique]:
        """Get techniques by modality."""
        query = f"SELECT * FROM {self._table} WHERE modality = $1"
        async with self._acquire() as conn:
            rows = await conn.fetch(query, modality)
            return [self._row_to_value_object(dict(row)) for row in rows]

    async def search(
        self,
        modality: str | None = None,
        category: str | None = None,
        difficulty: str | None = None,
        max_duration: int | None = None,
    ) -> list[Technique]:
        """Search techniques by criteria."""
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if modality is not None:
            conditions.append(f"modality = ${idx}")
            params.append(modality)
            idx += 1
        if category is not None:
            conditions.append(f"category = ${idx}")
            params.append(category)
            idx += 1
        if difficulty is not None:
            conditions.append(f"difficulty = ${idx}")
            params.append(difficulty)
            idx += 1
        if max_duration is not None:
            conditions.append(f"duration_minutes <= ${idx}")
            params.append(max_duration)
            idx += 1

        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM {self._table}{where} ORDER BY name ASC"
        async with self._acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_value_object(dict(row)) for row in rows]

    async def count(self) -> int:
        """Count total techniques."""
        query = f"SELECT COUNT(*) FROM {self._table}"
        async with self._acquire() as conn:
            return await conn.fetchval(query) or 0

    def _row_to_value_object(self, row: dict[str, Any]) -> Technique:
        contras = row.get("contraindications", [])
        if isinstance(contras, str):
            contras = json.loads(contras)

        prereqs = row.get("prerequisites", [])
        if isinstance(prereqs, str):
            prereqs = json.loads(prereqs)

        symptoms = row.get("target_symptoms", [])
        if isinstance(symptoms, str):
            symptoms = json.loads(symptoms)

        return Technique(
            technique_id=row["technique_id"],
            name=row["name"],
            modality=TherapyModality(row["modality"]) if row.get("modality") else TherapyModality.CBT,
            category=TechniqueCategory(row["category"]) if row.get("category") else TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description=row.get("description", ""),
            instructions=row.get("instructions", ""),
            duration_minutes=row.get("duration_minutes", 15),
            difficulty=DifficultyLevel(row["difficulty"]) if row.get("difficulty") else DifficultyLevel.BEGINNER,
            delivery_mode=DeliveryMode(row["delivery_mode"]) if row.get("delivery_mode") else DeliveryMode.GUIDED,
            requires_homework=row.get("requires_homework", False),
            contraindications=tuple(contras),
            prerequisites=tuple(prereqs),
            target_symptoms=tuple(symptoms),
            effectiveness_rating=Decimal(str(row.get("effectiveness_rating", "0.7"))),
            evidence_level=row.get("evidence_level", "moderate"),
        )


# ---------------------------------------------------------------------------
# Outcome Measure Repository
# ---------------------------------------------------------------------------

class PostgresOutcomeMeasureRepository:
    """PostgreSQL implementation of outcome measure repository."""

    POOL_NAME = "therapy_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._table = f"{schema}.outcome_measures"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._acquire()
        raise Exception("No database connection available.")

    async def get(self, measure_id: UUID) -> OutcomeMeasure | None:
        """Get outcome measure by ID."""
        query = f"SELECT * FROM {self._table} WHERE measure_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, measure_id)
            if row is None:
                return None
            return self._row_to_value_object(dict(row))

    async def save(self, measure: OutcomeMeasure, user_id: UUID) -> OutcomeMeasure:
        """Save outcome measure."""
        query = f"""
            INSERT INTO {self._table} (
                measure_id, user_id, instrument, raw_score,
                subscale_scores, recorded_at, session_id, notes
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (measure_id) DO UPDATE SET
                raw_score = EXCLUDED.raw_score,
                subscale_scores = EXCLUDED.subscale_scores,
                recorded_at = EXCLUDED.recorded_at,
                session_id = EXCLUDED.session_id,
                notes = EXCLUDED.notes
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                measure.measure_id,
                user_id,
                measure.instrument.value,
                measure.raw_score,
                json.dumps(dict(measure.subscale_scores)),
                measure.recorded_at,
                measure.session_id,
                measure.notes,
            )
            logger.debug("outcome_measure_saved_postgres", measure_id=str(measure.measure_id))
            return measure

    async def delete(self, measure_id: UUID) -> bool:
        """Delete outcome measure."""
        query = f"DELETE FROM {self._table} WHERE measure_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, measure_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("outcome_measure_deleted_postgres", measure_id=str(measure_id))
            return deleted

    async def get_by_session(self, session_id: UUID) -> list[OutcomeMeasure]:
        """Get outcome measures for a session."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE session_id = $1 ORDER BY recorded_at ASC
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, session_id)
            return [self._row_to_value_object(dict(row)) for row in rows]

    async def get_history(
        self, user_id: UUID, instrument: str, limit: int = 50
    ) -> list[OutcomeMeasure]:
        """Get outcome measure history for user and instrument."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND instrument = $2
            ORDER BY recorded_at DESC
            LIMIT $3
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, instrument, limit)
            return [self._row_to_value_object(dict(row)) for row in rows]

    async def get_latest(
        self, user_id: UUID, instrument: str
    ) -> OutcomeMeasure | None:
        """Get latest outcome measure for user and instrument."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND instrument = $2
            ORDER BY recorded_at DESC
            LIMIT 1
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id, instrument)
            if row is None:
                return None
            return self._row_to_value_object(dict(row))

    async def count(self) -> int:
        """Count total outcome measures."""
        query = f"SELECT COUNT(*) FROM {self._table}"
        async with self._acquire() as conn:
            return await conn.fetchval(query) or 0

    def _row_to_value_object(self, row: dict[str, Any]) -> OutcomeMeasure:
        subscales = row.get("subscale_scores", {})
        if isinstance(subscales, str):
            subscales = json.loads(subscales)
        if isinstance(subscales, dict):
            subscales = tuple(subscales.items())

        return OutcomeMeasure(
            measure_id=row["measure_id"],
            instrument=OutcomeInstrument(row["instrument"]) if row.get("instrument") else OutcomeInstrument.PHQ9,
            raw_score=row["raw_score"],
            subscale_scores=subscales,
            recorded_at=row["recorded_at"] if row.get("recorded_at") else datetime.now(timezone.utc),
            session_id=row.get("session_id"),
            notes=row.get("notes", ""),
        )

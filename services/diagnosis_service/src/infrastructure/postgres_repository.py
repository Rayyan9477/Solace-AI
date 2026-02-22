"""
Solace-AI Diagnosis Service - PostgreSQL Repository Implementation.

PostgreSQL-backed repository for diagnosis session and record persistence.
Uses asyncpg for async database operations with connection pooling.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from ..domain.entities import (
    DiagnosisRecordEntity,
    DiagnosisSessionEntity,
    HypothesisEntity,
    SymptomEntity,
)
from ..schemas import (
    ConfidenceLevel,
    DiagnosisPhase,
    SeverityLevel,
    SymptomType,
)
from .repository import DiagnosisRepositoryPort

try:
    from solace_infrastructure.database import ConnectionPoolManager
    from solace_infrastructure.feature_flags import FeatureFlags
except ImportError:
    ConnectionPoolManager = None
    FeatureFlags = None

if TYPE_CHECKING:
    from solace_infrastructure.postgres import PostgresClient

logger = structlog.get_logger(__name__)


def _decimal_encoder(obj: Any) -> Any:
    """JSON encoder for Decimal types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class PostgresDiagnosisRepository(DiagnosisRepositoryPort):
    """PostgreSQL implementation of diagnosis repository."""

    POOL_NAME = "diagnosis_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._sessions_table = f"{schema}.diagnosis_sessions"
        self._records_table = f"{schema}.diagnosis_records"
        self._stats = {"sessions_saved": 0, "records_saved": 0, "queries": 0, "deletes": 0}

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._client.acquire()
        raise Exception("No database connection available.")

    async def save_session(self, session: DiagnosisSessionEntity) -> None:
        """Save a diagnosis session to PostgreSQL."""
        session.touch()

        # Serialize symptoms and hypotheses to JSON
        symptoms_json = json.dumps(
            [self._symptom_to_dict(s) for s in session.symptoms],
            default=_decimal_encoder,
        )
        hypotheses_json = json.dumps(
            [self._hypothesis_to_dict(h) for h in session.hypotheses],
            default=_decimal_encoder,
        )
        messages_json = json.dumps(session.messages)
        metadata_json = json.dumps(session.metadata, default=_decimal_encoder)
        recommendations_json = json.dumps(session.recommendations)

        query = f"""
            INSERT INTO {self._sessions_table} (
                id, user_id, session_number, phase, symptoms, hypotheses,
                primary_hypothesis_id, messages, safety_flags, started_at,
                ended_at, is_active, summary, recommendations, metadata,
                created_at, updated_at, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            ON CONFLICT (id) DO UPDATE SET
                phase = EXCLUDED.phase,
                symptoms = EXCLUDED.symptoms,
                hypotheses = EXCLUDED.hypotheses,
                primary_hypothesis_id = EXCLUDED.primary_hypothesis_id,
                messages = EXCLUDED.messages,
                safety_flags = EXCLUDED.safety_flags,
                ended_at = EXCLUDED.ended_at,
                is_active = EXCLUDED.is_active,
                summary = EXCLUDED.summary,
                recommendations = EXCLUDED.recommendations,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at,
                version = EXCLUDED.version
        """
        async with self._acquire() as conn:
            await conn.execute(
                query,
                session.id,
                session.user_id,
                session.session_number,
                session.phase.value,
                symptoms_json,
                hypotheses_json,
                session.primary_hypothesis_id,
                messages_json,
                session.safety_flags,
                session.started_at,
                session.ended_at,
                session.is_active,
                session.summary,
                recommendations_json,
                metadata_json,
                session.created_at,
                session.updated_at,
                session.version,
            )
            self._stats["sessions_saved"] += 1
            logger.debug("session_saved_postgres", session_id=str(session.id))

    async def get_session(self, session_id: UUID) -> DiagnosisSessionEntity | None:
        """Get a session by ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._sessions_table} WHERE id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, session_id)
            if row is None:
                return None
            return self._row_to_session(dict(row))

    async def get_active_session(self, user_id: UUID) -> DiagnosisSessionEntity | None:
        """Get active session for user from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._sessions_table}
            WHERE user_id = $1 AND is_active = true
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_session(dict(row))

    async def list_user_sessions(
        self, user_id: UUID, limit: int = 10
    ) -> list[DiagnosisSessionEntity]:
        """List sessions for a user from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._sessions_table}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [self._row_to_session(dict(row)) for row in rows]

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session from PostgreSQL."""
        query = f"DELETE FROM {self._sessions_table} WHERE id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, session_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                self._stats["deletes"] += 1
                logger.debug("session_deleted_postgres", session_id=str(session_id))
            return deleted

    async def save_record(self, record: DiagnosisRecordEntity) -> None:
        """Save a diagnosis record to PostgreSQL."""
        record.touch()

        query = f"""
            INSERT INTO {self._records_table} (
                id, user_id, session_id, primary_diagnosis, dsm5_code, icd11_code,
                confidence, severity, symptom_summary, supporting_evidence,
                differential_diagnoses, recommendations, assessment_scores,
                clinician_notes, reviewed, reviewed_by, reviewed_at,
                created_at, updated_at, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
            )
            ON CONFLICT (id) DO UPDATE SET
                primary_diagnosis = EXCLUDED.primary_diagnosis,
                dsm5_code = EXCLUDED.dsm5_code,
                icd11_code = EXCLUDED.icd11_code,
                confidence = EXCLUDED.confidence,
                severity = EXCLUDED.severity,
                symptom_summary = EXCLUDED.symptom_summary,
                supporting_evidence = EXCLUDED.supporting_evidence,
                differential_diagnoses = EXCLUDED.differential_diagnoses,
                recommendations = EXCLUDED.recommendations,
                assessment_scores = EXCLUDED.assessment_scores,
                clinician_notes = EXCLUDED.clinician_notes,
                reviewed = EXCLUDED.reviewed,
                reviewed_by = EXCLUDED.reviewed_by,
                reviewed_at = EXCLUDED.reviewed_at,
                updated_at = EXCLUDED.updated_at,
                version = EXCLUDED.version
        """
        async with self._acquire() as conn:
            await conn.execute(
                query,
                record.id,
                record.user_id,
                record.session_id,
                record.primary_diagnosis,
                record.dsm5_code,
                record.icd11_code,
                float(record.confidence),
                record.severity.value,
                record.symptom_summary,
                record.supporting_evidence,
                record.differential_diagnoses,
                record.recommendations,
                json.dumps(record.assessment_scores),
                record.clinician_notes,
                record.reviewed,
                record.reviewed_by,
                record.reviewed_at,
                record.created_at,
                record.updated_at,
                record.version,
            )
            self._stats["records_saved"] += 1
            logger.debug("record_saved_postgres", record_id=str(record.id))

    async def get_record(self, record_id: UUID) -> DiagnosisRecordEntity | None:
        """Get a diagnosis record by ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._records_table} WHERE id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, record_id)
            if row is None:
                return None
            return self._row_to_record(dict(row))

    async def list_user_records(
        self, user_id: UUID, limit: int = 10
    ) -> list[DiagnosisRecordEntity]:
        """List diagnosis records for a user from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._records_table}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [self._row_to_record(dict(row)) for row in rows]

    async def delete_user_data(self, user_id: UUID) -> int:
        """Delete all data for a user (GDPR) from PostgreSQL."""
        deleted_count = 0

        async with self._acquire() as conn:
            # Delete sessions
            sessions_query = f"DELETE FROM {self._sessions_table} WHERE user_id = $1"
            result = await conn.execute(sessions_query, user_id)
            deleted_count += int(result.split()[-1])

            # Delete records
            records_query = f"DELETE FROM {self._records_table} WHERE user_id = $1"
            result = await conn.execute(records_query, user_id)
            deleted_count += int(result.split()[-1])

        self._stats["deletes"] += deleted_count
        logger.info("user_data_deleted_postgres", user_id=str(user_id), deleted_count=deleted_count)
        return deleted_count

    async def get_symptom_history(
        self, user_id: UUID, symptom_name: str
    ) -> list[SymptomEntity]:
        """Get history of a specific symptom for user from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT symptoms FROM {self._sessions_table}
            WHERE user_id = $1
            ORDER BY created_at DESC
        """
        symptoms: list[SymptomEntity] = []
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id)
            for row in rows:
                symptoms_data = row["symptoms"]
                if isinstance(symptoms_data, str):
                    symptoms_data = json.loads(symptoms_data)
                for s_data in symptoms_data:
                    if s_data.get("name", "").lower() == symptom_name.lower():
                        symptoms.append(self._dict_to_symptom(s_data))

        symptoms.sort(key=lambda s: s.created_at, reverse=True)
        return symptoms

    async def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics from PostgreSQL."""
        sessions_query = f"SELECT COUNT(*) as total, SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active FROM {self._sessions_table}"
        records_query = f"SELECT COUNT(*) FROM {self._records_table}"
        users_query = f"SELECT COUNT(DISTINCT user_id) FROM {self._sessions_table}"

        async with self._acquire() as conn:
            sessions_row = await conn.fetchrow(sessions_query)
            records_count = await conn.fetchval(records_query)
            users_count = await conn.fetchval(users_query)

        return {
            **self._stats,
            "total_sessions": sessions_row["total"] if sessions_row else 0,
            "total_records": records_count or 0,
            "total_users": users_count or 0,
            "active_sessions": sessions_row["active"] if sessions_row else 0,
        }

    def _symptom_to_dict(self, symptom: SymptomEntity) -> dict[str, Any]:
        """Convert symptom entity to dictionary for JSON storage."""
        return {
            "id": str(symptom.id),
            "name": symptom.name,
            "description": symptom.description,
            "symptom_type": symptom.symptom_type.value,
            "severity": symptom.severity.value,
            "onset": symptom.onset,
            "duration": symptom.duration,
            "frequency": symptom.frequency,
            "triggers": symptom.triggers,
            "extracted_from": symptom.extracted_from,
            "confidence": str(symptom.confidence),
            "session_id": str(symptom.session_id) if symptom.session_id else None,
            "user_id": str(symptom.user_id) if symptom.user_id else None,
            "is_active": symptom.is_active,
            "validated": symptom.validated,
            "validation_source": symptom.validation_source,
            "created_at": symptom.created_at.isoformat(),
            "updated_at": symptom.updated_at.isoformat(),
            "version": symptom.version,
        }

    def _hypothesis_to_dict(self, hypothesis: HypothesisEntity) -> dict[str, Any]:
        """Convert hypothesis entity to dictionary for JSON storage."""
        return {
            "id": str(hypothesis.id),
            "name": hypothesis.name,
            "dsm5_code": hypothesis.dsm5_code,
            "icd11_code": hypothesis.icd11_code,
            "confidence": str(hypothesis.confidence),
            "confidence_level": hypothesis.confidence_level.value,
            "criteria_met": hypothesis.criteria_met,
            "criteria_missing": hypothesis.criteria_missing,
            "supporting_evidence": hypothesis.supporting_evidence,
            "contra_evidence": hypothesis.contra_evidence,
            "severity": hypothesis.severity.value,
            "hitop_dimensions": {k: str(v) for k, v in hypothesis.hitop_dimensions.items()},
            "session_id": str(hypothesis.session_id) if hypothesis.session_id else None,
            "challenged": hypothesis.challenged,
            "challenge_results": hypothesis.challenge_results,
            "calibrated": hypothesis.calibrated,
            "original_confidence": str(hypothesis.original_confidence) if hypothesis.original_confidence else None,
            "created_at": hypothesis.created_at.isoformat(),
            "updated_at": hypothesis.updated_at.isoformat(),
            "version": hypothesis.version,
        }

    def _dict_to_symptom(self, data: dict[str, Any]) -> SymptomEntity:
        """Convert dictionary to symptom entity."""
        return SymptomEntity(
            id=UUID(data["id"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            symptom_type=SymptomType(data.get("symptom_type", "emotional")),
            severity=SeverityLevel.from_string(data.get("severity", "mild")),
            onset=data.get("onset"),
            duration=data.get("duration"),
            frequency=data.get("frequency"),
            triggers=data.get("triggers", []),
            extracted_from=data.get("extracted_from"),
            confidence=Decimal(data.get("confidence", "0.7")),
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            user_id=UUID(data["user_id"]) if data.get("user_id") else None,
            is_active=data.get("is_active", True),
            validated=data.get("validated", False),
            validation_source=data.get("validation_source"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            version=data.get("version", 1),
        )

    def _dict_to_hypothesis(self, data: dict[str, Any]) -> HypothesisEntity:
        """Convert dictionary to hypothesis entity."""
        hitop = {}
        if data.get("hitop_dimensions"):
            for k, v in data["hitop_dimensions"].items():
                hitop[k] = Decimal(v) if v else Decimal("0")

        return HypothesisEntity(
            id=UUID(data["id"]),
            name=data.get("name", ""),
            dsm5_code=data.get("dsm5_code"),
            icd11_code=data.get("icd11_code"),
            confidence=Decimal(data.get("confidence", "0.5")),
            confidence_level=ConfidenceLevel(data.get("confidence_level", "medium")),
            criteria_met=data.get("criteria_met", []),
            criteria_missing=data.get("criteria_missing", []),
            supporting_evidence=data.get("supporting_evidence", []),
            contra_evidence=data.get("contra_evidence", []),
            severity=SeverityLevel.from_string(data.get("severity", "mild")),
            hitop_dimensions=hitop,
            session_id=UUID(data["session_id"]) if data.get("session_id") else None,
            challenged=data.get("challenged", False),
            challenge_results=data.get("challenge_results", []),
            calibrated=data.get("calibrated", False),
            original_confidence=Decimal(data["original_confidence"]) if data.get("original_confidence") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            version=data.get("version", 1),
        )

    def _row_to_session(self, row: dict[str, Any]) -> DiagnosisSessionEntity:
        """Convert a database row to a DiagnosisSessionEntity."""
        # Parse JSON fields
        symptoms_data = row.get("symptoms", [])
        if isinstance(symptoms_data, str):
            symptoms_data = json.loads(symptoms_data)

        hypotheses_data = row.get("hypotheses", [])
        if isinstance(hypotheses_data, str):
            hypotheses_data = json.loads(hypotheses_data)

        messages = row.get("messages", [])
        if isinstance(messages, str):
            messages = json.loads(messages)

        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        recommendations = row.get("recommendations", [])
        if isinstance(recommendations, str):
            recommendations = json.loads(recommendations)

        return DiagnosisSessionEntity(
            id=row["id"],
            user_id=row["user_id"],
            session_number=row.get("session_number", 1),
            phase=DiagnosisPhase(row.get("phase", "rapport")),
            symptoms=[self._dict_to_symptom(s) for s in symptoms_data],
            hypotheses=[self._dict_to_hypothesis(h) for h in hypotheses_data],
            primary_hypothesis_id=row.get("primary_hypothesis_id"),
            messages=messages,
            safety_flags=row.get("safety_flags", []),
            started_at=row.get("started_at", datetime.now(timezone.utc)),
            ended_at=row.get("ended_at"),
            is_active=row.get("is_active", True),
            summary=row.get("summary"),
            recommendations=recommendations,
            metadata=metadata,
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            version=row.get("version", 1),
        )

    def _row_to_record(self, row: dict[str, Any]) -> DiagnosisRecordEntity:
        """Convert a database row to a DiagnosisRecordEntity."""
        assessment_scores = row.get("assessment_scores", {})
        if isinstance(assessment_scores, str):
            assessment_scores = json.loads(assessment_scores)

        return DiagnosisRecordEntity(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            primary_diagnosis=row.get("primary_diagnosis", ""),
            dsm5_code=row.get("dsm5_code"),
            icd11_code=row.get("icd11_code"),
            confidence=Decimal(str(row.get("confidence", "0.5"))),
            severity=SeverityLevel.from_string(row.get("severity", "mild")),
            symptom_summary=row.get("symptom_summary", []),
            supporting_evidence=row.get("supporting_evidence", []),
            differential_diagnoses=row.get("differential_diagnoses", []),
            recommendations=row.get("recommendations", []),
            assessment_scores=assessment_scores,
            clinician_notes=row.get("clinician_notes"),
            reviewed=row.get("reviewed", False),
            reviewed_by=row.get("reviewed_by"),
            reviewed_at=row.get("reviewed_at"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            version=row.get("version", 1),
        )

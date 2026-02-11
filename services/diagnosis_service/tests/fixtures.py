"""
Solace-AI Diagnosis Service - Test Fixtures.
In-memory repository implementations for testing purposes only.
"""
from __future__ import annotations
import os
from datetime import datetime
from typing import Any
from uuid import UUID
import structlog

from services.diagnosis_service.src.domain.entities import (
    DiagnosisSessionEntity, SymptomEntity, DiagnosisRecordEntity,
)
from services.diagnosis_service.src.schemas import DiagnosisPhase, SeverityLevel
from services.diagnosis_service.src.infrastructure.repository import DiagnosisRepositoryPort

logger = structlog.get_logger(__name__)


class InMemoryDiagnosisRepository(DiagnosisRepositoryPort):
    """In-memory implementation of diagnosis repository."""

    def __init__(self) -> None:
        if os.getenv("ENVIRONMENT") == "production":
            raise RuntimeError("In-memory repositories are not allowed in production.")
        self._sessions: dict[UUID, DiagnosisSessionEntity] = {}
        self._records: dict[UUID, DiagnosisRecordEntity] = {}
        self._user_sessions: dict[UUID, list[UUID]] = {}
        self._user_records: dict[UUID, list[UUID]] = {}
        self._stats = {"sessions_saved": 0, "records_saved": 0, "queries": 0, "deletes": 0}

    async def save_session(self, session: DiagnosisSessionEntity) -> None:
        """Save a diagnosis session."""
        session.touch()
        self._sessions[session.id] = session
        self._user_sessions.setdefault(session.user_id, [])
        if session.id not in self._user_sessions[session.user_id]:
            self._user_sessions[session.user_id].append(session.id)
        self._stats["sessions_saved"] += 1
        logger.debug("session_saved", session_id=str(session.id), user_id=str(session.user_id))

    async def get_session(self, session_id: UUID) -> DiagnosisSessionEntity | None:
        """Get a session by ID."""
        self._stats["queries"] += 1
        return self._sessions.get(session_id)

    async def get_active_session(self, user_id: UUID) -> DiagnosisSessionEntity | None:
        """Get active session for user."""
        self._stats["queries"] += 1
        session_ids = self._user_sessions.get(user_id, [])
        for sid in reversed(session_ids):
            session = self._sessions.get(sid)
            if session and session.is_active:
                return session
        return None

    async def list_user_sessions(self, user_id: UUID, limit: int = 10) -> list[DiagnosisSessionEntity]:
        """List sessions for a user."""
        self._stats["queries"] += 1
        session_ids = self._user_sessions.get(user_id, [])
        sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions[:limit]

    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id] = [
                    sid for sid in self._user_sessions[session.user_id] if sid != session_id
                ]
            self._stats["deletes"] += 1
            logger.debug("session_deleted", session_id=str(session_id))
            return True
        return False

    async def save_record(self, record: DiagnosisRecordEntity) -> None:
        """Save a diagnosis record."""
        record.touch()
        self._records[record.id] = record
        self._user_records.setdefault(record.user_id, [])
        if record.id not in self._user_records[record.user_id]:
            self._user_records[record.user_id].append(record.id)
        self._stats["records_saved"] += 1
        logger.debug("record_saved", record_id=str(record.id), user_id=str(record.user_id))

    async def get_record(self, record_id: UUID) -> DiagnosisRecordEntity | None:
        """Get a diagnosis record by ID."""
        self._stats["queries"] += 1
        return self._records.get(record_id)

    async def list_user_records(self, user_id: UUID, limit: int = 10) -> list[DiagnosisRecordEntity]:
        """List diagnosis records for a user."""
        self._stats["queries"] += 1
        record_ids = self._user_records.get(user_id, [])
        records = [self._records[rid] for rid in record_ids if rid in self._records]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    async def delete_user_data(self, user_id: UUID) -> int:
        """Delete all data for a user (GDPR)."""
        deleted_count = 0
        session_ids = self._user_sessions.pop(user_id, [])
        for sid in session_ids:
            if sid in self._sessions:
                del self._sessions[sid]
                deleted_count += 1
        record_ids = self._user_records.pop(user_id, [])
        for rid in record_ids:
            if rid in self._records:
                del self._records[rid]
                deleted_count += 1
        self._stats["deletes"] += deleted_count
        logger.info("user_data_deleted", user_id=str(user_id), deleted_count=deleted_count)
        return deleted_count

    async def get_symptom_history(self, user_id: UUID, symptom_name: str) -> list[SymptomEntity]:
        """Get history of a specific symptom for user."""
        self._stats["queries"] += 1
        symptoms: list[SymptomEntity] = []
        session_ids = self._user_sessions.get(user_id, [])
        for sid in session_ids:
            session = self._sessions.get(sid)
            if session:
                for symptom in session.symptoms:
                    if symptom.name.lower() == symptom_name.lower():
                        symptoms.append(symptom)
        symptoms.sort(key=lambda s: s.created_at, reverse=True)
        return symptoms

    async def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics."""
        return {
            **self._stats,
            "total_sessions": len(self._sessions),
            "total_records": len(self._records),
            "total_users": len(self._user_sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active),
        }


class SessionQueryBuilder:
    """Query builder for session searches."""

    def __init__(self, repository: InMemoryDiagnosisRepository) -> None:
        self._repository = repository
        self._user_id: UUID | None = None
        self._phase: DiagnosisPhase | None = None
        self._active_only: bool = False
        self._date_from: datetime | None = None
        self._date_to: datetime | None = None
        self._limit: int = 100

    def for_user(self, user_id: UUID) -> SessionQueryBuilder:
        """Filter by user ID."""
        self._user_id = user_id
        return self

    def in_phase(self, phase: DiagnosisPhase) -> SessionQueryBuilder:
        """Filter by phase."""
        self._phase = phase
        return self

    def active_only(self) -> SessionQueryBuilder:
        """Filter to active sessions only."""
        self._active_only = True
        return self

    def since(self, date: datetime) -> SessionQueryBuilder:
        """Filter sessions created after date."""
        self._date_from = date
        return self

    def until(self, date: datetime) -> SessionQueryBuilder:
        """Filter sessions created before date."""
        self._date_to = date
        return self

    def limit(self, count: int) -> SessionQueryBuilder:
        """Limit result count."""
        self._limit = count
        return self

    async def execute(self) -> list[DiagnosisSessionEntity]:
        """Execute the query."""
        results: list[DiagnosisSessionEntity] = []
        sessions = self._repository._sessions.values()
        for session in sessions:
            if self._user_id and session.user_id != self._user_id:
                continue
            if self._phase and session.phase != self._phase:
                continue
            if self._active_only and not session.is_active:
                continue
            if self._date_from and session.created_at < self._date_from:
                continue
            if self._date_to and session.created_at > self._date_to:
                continue
            results.append(session)
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results[:self._limit]


class RecordQueryBuilder:
    """Query builder for diagnosis record searches."""

    def __init__(self, repository: InMemoryDiagnosisRepository) -> None:
        self._repository = repository
        self._user_id: UUID | None = None
        self._severity: SeverityLevel | None = None
        self._reviewed_only: bool = False
        self._diagnosis_name: str | None = None
        self._limit: int = 100

    def for_user(self, user_id: UUID) -> RecordQueryBuilder:
        """Filter by user ID."""
        self._user_id = user_id
        return self

    def with_severity(self, severity: SeverityLevel) -> RecordQueryBuilder:
        """Filter by severity level."""
        self._severity = severity
        return self

    def reviewed_only(self) -> RecordQueryBuilder:
        """Filter to reviewed records only."""
        self._reviewed_only = True
        return self

    def with_diagnosis(self, name: str) -> RecordQueryBuilder:
        """Filter by diagnosis name."""
        self._diagnosis_name = name
        return self

    def limit(self, count: int) -> RecordQueryBuilder:
        """Limit result count."""
        self._limit = count
        return self

    async def execute(self) -> list[DiagnosisRecordEntity]:
        """Execute the query."""
        results: list[DiagnosisRecordEntity] = []
        records = self._repository._records.values()
        for record in records:
            if self._user_id and record.user_id != self._user_id:
                continue
            if self._severity and record.severity != self._severity:
                continue
            if self._reviewed_only and not record.reviewed:
                continue
            if self._diagnosis_name and self._diagnosis_name.lower() not in record.primary_diagnosis.lower():
                continue
            results.append(record)
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:self._limit]

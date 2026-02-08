"""
Memory domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for the 5-tier memory system including
memory records, user profiles, session summaries, and therapeutic events.
All clinical data inherits from ClinicalBase for PHI encryption.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..base_models import ClinicalBase
from ..schema_registry import SchemaRegistry


# Enumerations

class MemoryTier(str, Enum):
    TIER_1_INPUT = "TIER_1_INPUT"
    TIER_2_WORKING = "TIER_2_WORKING"
    TIER_3_SESSION = "TIER_3_SESSION"
    TIER_4_EPISODIC = "TIER_4_EPISODIC"
    TIER_5_SEMANTIC = "TIER_5_SEMANTIC"


class MemoryContentType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    ASSISTANT_MESSAGE = "ASSISTANT_MESSAGE"
    SYSTEM_MESSAGE = "SYSTEM_MESSAGE"
    SESSION_SUMMARY = "SESSION_SUMMARY"
    USER_FACT = "USER_FACT"
    THERAPEUTIC_EVENT = "THERAPEUTIC_EVENT"
    CRISIS_EVENT = "CRISIS_EVENT"
    ASSESSMENT_RESULT = "ASSESSMENT_RESULT"


class RetentionCategory(str, Enum):
    PERMANENT = "PERMANENT"
    LONG_TERM = "LONG_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    SHORT_TERM = "SHORT_TERM"
    EPHEMERAL = "EPHEMERAL"


# Entity Models

@SchemaRegistry.register
class MemoryRecord(ClinicalBase):
    """Memory record entity for the 5-tier memory system.

    Stores individual memory items with tier classification, retention
    policies, importance scoring, and optional vector embeddings.
    """

    __tablename__ = "memory_records"
    __phi_fields__: ClassVar[list[str]] = ["content"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="Source session ID (cross-service reference)"
    )
    tier: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
        comment="Memory tier: TIER_1_INPUT through TIER_5_SEMANTIC"
    )
    content: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="Memory content (encrypted as PHI)"
    )
    content_type: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
        comment="Type of content stored"
    )
    retention_category: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Retention policy: PERMANENT, LONG_TERM, etc."
    )
    importance_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5,
        comment="Importance score 0.0-1.0"
    )
    emotional_valence: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Emotional valence -1.0 to 1.0"
    )

    # Metadata and relations
    record_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    tags: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    related_records: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="UUIDs of related memory records"
    )

    # Retention tracking
    retention_strength: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0,
        comment="Retention strength, decays over time"
    )
    access_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True,
    )
    is_archived: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True,
    )
    is_safety_critical: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryRecord(id={self.id}, tier={self.tier}, "
            f"type={self.content_type}, importance={self.importance_score})>"
        )


@SchemaRegistry.register
class MemoryUserProfile(ClinicalBase):
    """User profile entity for aggregated memory-based user knowledge.

    Stores accumulated facts, therapeutic context, and preferences
    learned across all sessions for a user.
    """

    __tablename__ = "memory_user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    profile_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Accumulated knowledge (encrypted)
    personal_facts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    therapeutic_context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    communication_preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    safety_information: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    # Clinical context
    diagnosed_conditions: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    current_treatments: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    support_network: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    triggers: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    coping_strategies: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    personality_traits: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    # Session statistics
    total_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_session_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_session_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_crisis_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    crisis_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"<MemoryUserProfile(id={self.id}, user_id={self.user_id}, "
            f"total_sessions={self.total_sessions})>"
        )


@SchemaRegistry.register
class SessionSummary(ClinicalBase):
    """Session summary entity for consolidated session records.

    Stores summarized session data at Tier 4 (episodic memory) level
    after session consolidation.
    """

    __tablename__ = "session_summaries"
    __phi_fields__: ClassVar[list[str]] = ["summary_text", "progress_notes"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True, unique=True,
        comment="Source session ID (cross-service reference)"
    )
    session_number: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_text: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Session summary (encrypted)"
    )

    # Session analysis
    key_topics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    emotional_arc: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    techniques_used: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    key_insights: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    homework_assigned: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    homework_reviewed: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=list)
    progress_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Risk tracking
    risk_level_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    risk_level_end: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Session metadata
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    session_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )
    retention_strength: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0,
    )

    def __repr__(self) -> str:
        return (
            f"<SessionSummary(id={self.id}, session_id={self.session_id}, "
            f"session_number={self.session_number})>"
        )


@SchemaRegistry.register
class TherapeuticEvent(ClinicalBase):
    """Therapeutic event entity for significant clinical events.

    Tracks notable events like breakthroughs, crises, or assessment
    results across the therapeutic journey.
    """

    __tablename__ = "therapeutic_events"
    __phi_fields__: ClassVar[list[str]] = ["title", "description"]

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
    )
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False, default="low",
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Timing
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Event data
    related_events: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    retention_strength: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0,
    )
    is_safety_critical: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<TherapeuticEvent(id={self.id}, type={self.event_type}, "
            f"severity={self.severity}, title={self.title})>"
        )


__all__ = [
    "MemoryTier",
    "MemoryContentType",
    "RetentionCategory",
    "MemoryRecord",
    "MemoryUserProfile",
    "SessionSummary",
    "TherapeuticEvent",
]

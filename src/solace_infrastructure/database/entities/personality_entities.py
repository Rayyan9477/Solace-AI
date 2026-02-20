"""
Personality domain entities for the Solace-AI platform.

Centralized SQLAlchemy ORM models for personality profiles, trait assessments,
and profile snapshots. Stores Big Five (OCEAN) personality scores and
communication style preferences. All data encrypted as PHI.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base_models import ClinicalBase
from ..schema_registry import SchemaRegistry


# Enumerations

class PersonalityTrait(str, Enum):
    OPENNESS = "OPENNESS"
    CONSCIENTIOUSNESS = "CONSCIENTIOUSNESS"
    EXTRAVERSION = "EXTRAVERSION"
    AGREEABLENESS = "AGREEABLENESS"
    NEUROTICISM = "NEUROTICISM"


class AssessmentSource(str, Enum):
    TEXT_ANALYSIS = "TEXT_ANALYSIS"
    LLM_ZERO_SHOT = "LLM_ZERO_SHOT"
    LIWC_FEATURES = "LIWC_FEATURES"
    VOICE_ANALYSIS = "VOICE_ANALYSIS"
    BEHAVIORAL = "BEHAVIORAL"
    ENSEMBLE = "ENSEMBLE"
    SELF_REPORT = "SELF_REPORT"


# Entity Models

@SchemaRegistry.register
class PersonalityProfile(ClinicalBase):
    """Personality profile entity storing Big Five (OCEAN) scores.

    Aggregate root for personality data. Each user has one profile
    that accumulates assessment data over time.
    """

    __tablename__ = "personality_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    # OCEAN scores stored as JSONB object
    ocean_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Big Five scores: openness, conscientiousness, extraversion, agreeableness, neuroticism + confidence"
    )
    communication_style: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Communication style preferences and parameters"
    )

    assessment_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    stability_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Profile stability score 0.0-1.0"
    )

    # Assessment history stored as JSONB array (capped at 100)
    assessment_history: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Recent assessment history (max 100 entries)"
    )

    # Relationships
    assessments: Mapped[list[TraitAssessment]] = relationship(
        "TraitAssessment", back_populates="profile",
    )
    snapshots: Mapped[list[ProfileSnapshot]] = relationship(
        "ProfileSnapshot", back_populates="profile",
    )

    def __repr__(self) -> str:
        return (
            f"<PersonalityProfile(id={self.id}, user_id={self.user_id}, "
            f"assessment_count={self.assessment_count}, stability={self.stability_score})>"
        )


@SchemaRegistry.register
class TraitAssessment(ClinicalBase):
    """Trait assessment entity for individual personality measurements.

    Records a single assessment of Big Five traits from a specific
    source (text analysis, LLM, LIWC, etc.).
    """

    __tablename__ = "trait_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personality_profiles.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    ocean_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Big Five scores from this assessment"
    )
    source: Mapped[str] = mapped_column(
        String(30), nullable=False, index=True,
        comment="Assessment source: TEXT_ANALYSIS, LLM_ZERO_SHOT, ENSEMBLE, etc."
    )
    assessment_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=True, default=dict,
        comment="Assessment-specific metadata"
    )
    evidence: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Supporting evidence for scores"
    )

    # Relationships
    profile: Mapped[PersonalityProfile] = relationship(
        "PersonalityProfile", back_populates="assessments",
    )

    def __repr__(self) -> str:
        return (
            f"<TraitAssessment(id={self.id}, user_id={self.user_id}, "
            f"source={self.source})>"
        )


@SchemaRegistry.register
class ProfileSnapshot(ClinicalBase):
    """Profile snapshot entity for tracking personality changes over time.

    Read-only record capturing the state of a personality profile
    at a specific point for longitudinal comparison.
    """

    __tablename__ = "profile_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False,
    )

    profile_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("personality_profiles.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    ocean_scores: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    communication_style: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=True, default=dict,
    )
    assessment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stability_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc), index=True,
    )
    reason: Mapped[str] = mapped_column(
        String(200), nullable=False,
        comment="Reason for capturing snapshot"
    )

    # Relationships
    profile: Mapped[PersonalityProfile] = relationship(
        "PersonalityProfile", back_populates="snapshots",
    )

    def __repr__(self) -> str:
        return (
            f"<ProfileSnapshot(id={self.id}, profile_id={self.profile_id}, "
            f"captured_at={self.captured_at}, reason={self.reason})>"
        )


__all__ = [
    "PersonalityTrait",
    "AssessmentSource",
    "PersonalityProfile",
    "TraitAssessment",
    "ProfileSnapshot",
]

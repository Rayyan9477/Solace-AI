"""
Solace-AI Memory Service - Domain Entities.

Rich domain entities for memory management following DDD principles.
Entities have identity, lifecycle, and business logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator
import structlog

logger = structlog.get_logger(__name__)


class MemoryTier(str, Enum):
    """Five-tier memory hierarchy."""
    TIER_1_INPUT = "tier_1_input"
    TIER_2_WORKING = "tier_2_working"
    TIER_3_SESSION = "tier_3_session"
    TIER_4_EPISODIC = "tier_4_episodic"
    TIER_5_SEMANTIC = "tier_5_semantic"


class RetentionCategory(str, Enum):
    """Memory retention categories affecting decay."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"
    EPHEMERAL = "ephemeral"


class ContentType(str, Enum):
    """Types of memory content."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    SESSION_SUMMARY = "session_summary"
    USER_FACT = "user_fact"
    THERAPEUTIC_EVENT = "therapeutic_event"
    CRISIS_EVENT = "crisis_event"
    ASSESSMENT_RESULT = "assessment_result"


class MemoryRecordId(BaseModel):
    """Strongly-typed memory record identifier."""
    value: UUID = Field(default_factory=uuid4)
    model_config = ConfigDict(frozen=True)

    @classmethod
    def generate(cls) -> MemoryRecordId:
        return cls(value=uuid4())

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)


class MemoryRecordEntity(BaseModel):
    """
    Domain entity for a single memory record.

    Represents any stored memory with full metadata, retention tracking,
    and semantic information for retrieval.
    """
    record_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID | None = None
    tier: MemoryTier = MemoryTier.TIER_3_SESSION
    content: str = Field(min_length=1, max_length=100000)
    content_type: ContentType = ContentType.USER_MESSAGE
    retention_category: RetentionCategory = RetentionCategory.MEDIUM_TERM
    importance_score: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"))
    emotional_valence: Decimal | None = Field(default=None, ge=Decimal("-1"), le=Decimal("1"))
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    related_records: list[UUID] = Field(default_factory=list)
    retention_strength: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("1"))
    access_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    is_archived: bool = False
    is_safety_critical: bool = False
    version: int = Field(default=1, ge=1)
    model_config = ConfigDict(validate_assignment=True)

    def record_access(self) -> MemoryRecordEntity:
        """Record an access, boosting retention strength."""
        boost = min(Decimal("0.1"), Decimal("1.0") - self.retention_strength)
        return self.model_copy(update={
            "accessed_at": datetime.now(timezone.utc),
            "access_count": self.access_count + 1,
            "retention_strength": self.retention_strength + boost,
        })

    def apply_decay(self, decay_amount: Decimal) -> MemoryRecordEntity:
        """Apply memory decay, respecting safety-critical flag."""
        if self.is_safety_critical or self.retention_category == RetentionCategory.PERMANENT:
            return self
        new_strength = max(Decimal("0"), self.retention_strength - decay_amount)
        return self.model_copy(update={"retention_strength": new_strength})

    def should_archive(self, threshold: Decimal = Decimal("0.1")) -> bool:
        """Check if memory should be archived based on retention strength."""
        if self.is_safety_critical or self.retention_category == RetentionCategory.PERMANENT:
            return False
        return self.retention_strength < threshold

    def mark_safety_critical(self) -> MemoryRecordEntity:
        """Mark as safety-critical (never decays or deletes)."""
        return self.model_copy(update={
            "is_safety_critical": True,
            "retention_category": RetentionCategory.PERMANENT,
            "retention_strength": Decimal("1.0"),
        })

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "record_id": str(self.record_id),
            "user_id": str(self.user_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "tier": self.tier.value,
            "content": self.content,
            "content_type": self.content_type.value,
            "retention_category": self.retention_category.value,
            "importance_score": float(self.importance_score),
            "emotional_valence": float(self.emotional_valence) if self.emotional_valence else None,
            "metadata": self.metadata,
            "tags": self.tags,
            "retention_strength": float(self.retention_strength),
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "is_archived": self.is_archived,
            "is_safety_critical": self.is_safety_critical,
            "version": self.version,
        }


class UserProfileEntity(BaseModel):
    """
    Domain entity for user profile with aggregated facts and preferences.

    Stores long-term semantic memory about the user including personal facts,
    therapeutic context, and interaction preferences.
    """
    user_id: UUID
    profile_version: int = Field(default=1, ge=1)
    personal_facts: dict[str, Any] = Field(default_factory=dict)
    therapeutic_context: dict[str, Any] = Field(default_factory=dict)
    communication_preferences: dict[str, Any] = Field(default_factory=dict)
    safety_information: dict[str, Any] = Field(default_factory=dict)
    diagnosed_conditions: list[str] = Field(default_factory=list)
    current_treatments: list[str] = Field(default_factory=list)
    support_network: list[dict[str, str]] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    coping_strategies: list[str] = Field(default_factory=list)
    personality_traits: dict[str, Decimal] = Field(default_factory=dict)
    total_sessions: int = Field(default=0, ge=0)
    first_session_date: datetime | None = None
    last_session_date: datetime | None = None
    last_crisis_date: datetime | None = None
    crisis_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(validate_assignment=True)

    def add_personal_fact(self, category: str, fact: str, confidence: Decimal = Decimal("0.7")) -> UserProfileEntity:
        """Add or update a personal fact."""
        facts = dict(self.personal_facts)
        if category not in facts:
            facts[category] = []
        facts[category].append({"fact": fact, "confidence": float(confidence), "added_at": datetime.now(timezone.utc).isoformat()})
        return self.model_copy(update={"personal_facts": facts, "updated_at": datetime.now(timezone.utc), "profile_version": self.profile_version + 1})

    def update_therapeutic_context(self, key: str, value: Any) -> UserProfileEntity:
        """Update therapeutic context."""
        ctx = dict(self.therapeutic_context)
        ctx[key] = value
        return self.model_copy(update={"therapeutic_context": ctx, "updated_at": datetime.now(timezone.utc)})

    def record_session(self, session_date: datetime | None = None) -> UserProfileEntity:
        """Record a completed session."""
        now = session_date or datetime.now(timezone.utc)
        first = self.first_session_date or now
        return self.model_copy(update={"total_sessions": self.total_sessions + 1, "first_session_date": first, "last_session_date": now, "updated_at": now})

    def record_crisis(self) -> UserProfileEntity:
        """Record a crisis event."""
        now = datetime.now(timezone.utc)
        return self.model_copy(update={"crisis_count": self.crisis_count + 1, "last_crisis_date": now, "updated_at": now})

    def add_trigger(self, trigger: str) -> UserProfileEntity:
        """Add an identified trigger."""
        if trigger not in self.triggers:
            return self.model_copy(update={"triggers": [*self.triggers, trigger], "updated_at": datetime.now(timezone.utc)})
        return self

    def add_coping_strategy(self, strategy: str) -> UserProfileEntity:
        """Add a coping strategy."""
        if strategy not in self.coping_strategies:
            return self.model_copy(update={"coping_strategies": [*self.coping_strategies, strategy], "updated_at": datetime.now(timezone.utc)})
        return self

    def get_safety_summary(self) -> dict[str, Any]:
        """Get safety-relevant information summary."""
        return {"crisis_count": self.crisis_count, "last_crisis_date": self.last_crisis_date.isoformat() if self.last_crisis_date else None,
                "triggers": self.triggers, "safety_information": self.safety_information, "support_network": self.support_network}


class SessionSummaryEntity(BaseModel):
    """
    Domain entity for session summary (Tier 4 episodic memory).

    Contains compressed representation of a therapy session including
    key topics, emotional arc, and therapeutic progress.
    """
    summary_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    user_id: UUID
    session_number: int = Field(ge=1)
    summary_text: str = Field(min_length=1, max_length=10000)
    key_topics: list[str] = Field(default_factory=list)
    emotional_arc: list[dict[str, Any]] = Field(default_factory=list)
    techniques_used: list[str] = Field(default_factory=list)
    key_insights: list[str] = Field(default_factory=list)
    homework_assigned: list[str] = Field(default_factory=list)
    homework_reviewed: list[str] = Field(default_factory=list)
    progress_notes: str = ""
    risk_level_start: str = "LOW"
    risk_level_end: str = "LOW"
    message_count: int = Field(default=0, ge=0)
    duration_minutes: int = Field(default=0, ge=0)
    session_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retention_strength: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("1"))
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(validate_assignment=True)

    def get_therapeutic_value(self) -> Decimal:
        """Calculate therapeutic value score for prioritization."""
        base = Decimal("0.3")
        if self.key_insights:
            base += Decimal("0.2")
        if self.techniques_used:
            base += Decimal("0.15")
        if self.homework_assigned:
            base += Decimal("0.1")
        if self.risk_level_start != self.risk_level_end:
            base += Decimal("0.15")
        if len(self.key_topics) >= 3:
            base += Decimal("0.1")
        return min(Decimal("1.0"), base)

    def apply_decay(self, decay_amount: Decimal) -> SessionSummaryEntity:
        """Apply decay to session summary."""
        new_strength = max(Decimal("0"), self.retention_strength - decay_amount)
        return self.model_copy(update={"retention_strength": new_strength})

    def to_context_string(self, max_length: int = 500) -> str:
        """Convert to context string for LLM."""
        parts = [f"Session {self.session_number} ({self.session_date.strftime('%Y-%m-%d')}):"]
        if len(self.summary_text) <= max_length:
            parts.append(self.summary_text)
        else:
            parts.append(self.summary_text[:max_length] + "...")
        if self.key_topics:
            parts.append(f"Topics: {', '.join(self.key_topics[:5])}")
        if self.key_insights:
            parts.append(f"Insights: {'; '.join(self.key_insights[:3])}")
        return "\n".join(parts)


class TherapeuticEventEntity(BaseModel):
    """
    Domain entity for significant therapeutic events.

    Tracks milestones, breakthroughs, setbacks, and other notable events
    in the therapeutic journey.
    """
    event_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID | None = None
    event_type: str
    severity: str = "medium"
    title: str = Field(min_length=1, max_length=500)
    description: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    related_events: list[UUID] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    retention_strength: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("1"))
    is_safety_critical: bool = False
    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def create_crisis_event(cls, user_id: UUID, session_id: UUID | None, title: str, description: str, payload: dict[str, Any] | None = None) -> TherapeuticEventEntity:
        """Factory for crisis events (always safety-critical)."""
        return cls(user_id=user_id, session_id=session_id, event_type="crisis", severity="critical", title=title,
                   description=description, payload=payload or {}, is_safety_critical=True, retention_strength=Decimal("1.0"))

    @classmethod
    def create_milestone(cls, user_id: UUID, session_id: UUID | None, title: str, description: str) -> TherapeuticEventEntity:
        """Factory for milestone events."""
        return cls(user_id=user_id, session_id=session_id, event_type="milestone", severity="low", title=title, description=description)

    def is_active(self) -> bool:
        """Check if event is currently valid."""
        now = datetime.now(timezone.utc)
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_to and now > self.valid_to:
            return False
        return True

    def link_event(self, other_event_id: UUID) -> TherapeuticEventEntity:
        """Link to a related event."""
        if other_event_id not in self.related_events:
            return self.model_copy(update={"related_events": [*self.related_events, other_event_id]})
        return self

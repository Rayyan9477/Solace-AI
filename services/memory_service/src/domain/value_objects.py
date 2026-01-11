"""
Solace-AI Memory Service - Value Objects.

Immutable value objects for memory domain concepts.
Value objects are identified by their attributes, not identity.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator
import structlog

logger = structlog.get_logger(__name__)


class MemoryTierSpec(str, Enum):
    """Memory tier specifications with access patterns."""
    TIER_1_INPUT = "tier_1_input"
    TIER_2_WORKING = "tier_2_working"
    TIER_3_SESSION = "tier_3_session"
    TIER_4_EPISODIC = "tier_4_episodic"
    TIER_5_SEMANTIC = "tier_5_semantic"


class RetentionPolicyType(str, Enum):
    """Types of retention policies."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"
    EPHEMERAL = "ephemeral"


class MemoryTierConfig(BaseModel):
    """Configuration for a memory tier."""
    tier: MemoryTierSpec
    display_name: str
    storage_backend: str
    max_latency_ms: int = Field(ge=1)
    default_ttl_seconds: int | None = None
    max_size_tokens: int | None = None
    supports_embedding: bool = True
    supports_decay: bool = True
    model_config = ConfigDict(frozen=True)

    @classmethod
    def input_buffer(cls) -> MemoryTierConfig:
        return cls(tier=MemoryTierSpec.TIER_1_INPUT, display_name="Input Buffer", storage_backend="memory",
                   max_latency_ms=1, default_ttl_seconds=60, max_size_tokens=4000, supports_embedding=False, supports_decay=False)

    @classmethod
    def working_memory(cls) -> MemoryTierConfig:
        return cls(tier=MemoryTierSpec.TIER_2_WORKING, display_name="Working Memory", storage_backend="redis",
                   max_latency_ms=10, default_ttl_seconds=3600, max_size_tokens=8000, supports_embedding=False, supports_decay=False)

    @classmethod
    def session_memory(cls) -> MemoryTierConfig:
        return cls(tier=MemoryTierSpec.TIER_3_SESSION, display_name="Session Memory", storage_backend="redis",
                   max_latency_ms=50, default_ttl_seconds=86400, max_size_tokens=None, supports_embedding=True, supports_decay=False)

    @classmethod
    def episodic_memory(cls) -> MemoryTierConfig:
        return cls(tier=MemoryTierSpec.TIER_4_EPISODIC, display_name="Episodic Memory", storage_backend="postgres+weaviate",
                   max_latency_ms=200, default_ttl_seconds=None, max_size_tokens=None, supports_embedding=True, supports_decay=True)

    @classmethod
    def semantic_memory(cls) -> MemoryTierConfig:
        return cls(tier=MemoryTierSpec.TIER_5_SEMANTIC, display_name="Semantic Memory", storage_backend="weaviate+postgres",
                   max_latency_ms=500, default_ttl_seconds=None, max_size_tokens=None, supports_embedding=True, supports_decay=True)


class RetentionPolicy(BaseModel):
    """
    Value object defining memory retention behavior.

    Controls how long memories are kept and how they decay over time.
    """
    policy_type: RetentionPolicyType
    base_decay_rate: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    min_retention_strength: Decimal = Field(default=Decimal("0.1"), ge=Decimal("0"), le=Decimal("1"))
    archive_threshold: Decimal = Field(default=Decimal("0.1"), ge=Decimal("0"), le=Decimal("1"))
    delete_threshold: Decimal = Field(default=Decimal("0.01"), ge=Decimal("0"), le=Decimal("1"))
    max_age_days: int | None = None
    boost_on_access: Decimal = Field(default=Decimal("0.05"), ge=Decimal("0"), le=Decimal("0.5"))
    emotional_decay_modifier: Decimal = Field(default=Decimal("0.8"), ge=Decimal("0.1"), le=Decimal("2.0"))
    model_config = ConfigDict(frozen=True)

    @classmethod
    def permanent(cls) -> RetentionPolicy:
        """Safety-critical memories that never decay."""
        return cls(policy_type=RetentionPolicyType.PERMANENT, base_decay_rate=Decimal("0"), min_retention_strength=Decimal("1.0"),
                   archive_threshold=Decimal("0"), delete_threshold=Decimal("0"), max_age_days=None, boost_on_access=Decimal("0"))

    @classmethod
    def long_term(cls) -> RetentionPolicy:
        """Important memories with slow decay."""
        return cls(policy_type=RetentionPolicyType.LONG_TERM, base_decay_rate=Decimal("0.001"), min_retention_strength=Decimal("0.2"),
                   archive_threshold=Decimal("0.2"), delete_threshold=Decimal("0.05"), max_age_days=365, boost_on_access=Decimal("0.1"))

    @classmethod
    def medium_term(cls) -> RetentionPolicy:
        """Standard memories with moderate decay."""
        return cls(policy_type=RetentionPolicyType.MEDIUM_TERM, base_decay_rate=Decimal("0.01"), min_retention_strength=Decimal("0.1"),
                   archive_threshold=Decimal("0.1"), delete_threshold=Decimal("0.02"), max_age_days=90, boost_on_access=Decimal("0.08"))

    @classmethod
    def short_term(cls) -> RetentionPolicy:
        """Temporary memories with fast decay."""
        return cls(policy_type=RetentionPolicyType.SHORT_TERM, base_decay_rate=Decimal("0.05"), min_retention_strength=Decimal("0.05"),
                   archive_threshold=Decimal("0.05"), delete_threshold=Decimal("0.01"), max_age_days=30, boost_on_access=Decimal("0.05"))

    @classmethod
    def ephemeral(cls) -> RetentionPolicy:
        """Very short-lived memories."""
        return cls(policy_type=RetentionPolicyType.EPHEMERAL, base_decay_rate=Decimal("0.2"), min_retention_strength=Decimal("0"),
                   archive_threshold=Decimal("0.1"), delete_threshold=Decimal("0.05"), max_age_days=7, boost_on_access=Decimal("0.02"))

    def calculate_decay(self, current_strength: Decimal, hours_elapsed: int, emotional_content: bool = False) -> Decimal:
        """Calculate new retention strength after decay."""
        if self.policy_type == RetentionPolicyType.PERMANENT:
            return current_strength
        decay_rate = self.base_decay_rate
        if emotional_content:
            decay_rate = decay_rate * self.emotional_decay_modifier
        total_decay = decay_rate * Decimal(hours_elapsed)
        new_strength = max(Decimal("0"), current_strength - total_decay)
        return new_strength

    def get_action(self, retention_strength: Decimal) -> str:
        """Determine action based on retention strength."""
        if self.policy_type == RetentionPolicyType.PERMANENT:
            return "retain"
        if retention_strength <= self.delete_threshold:
            return "delete"
        if retention_strength <= self.archive_threshold:
            return "archive"
        return "retain"


class TokenBudget(BaseModel):
    """Value object for token allocation in context assembly."""
    total_tokens: int = Field(ge=0)
    system_prompt_tokens: int = Field(default=0, ge=0)
    safety_tokens: int = Field(default=0, ge=0)
    user_profile_tokens: int = Field(default=0, ge=0)
    therapeutic_context_tokens: int = Field(default=0, ge=0)
    recent_messages_tokens: int = Field(default=0, ge=0)
    retrieved_context_tokens: int = Field(default=0, ge=0)
    current_message_tokens: int = Field(default=0, ge=0)
    reserved_tokens: int = Field(default=500, ge=0)
    model_config = ConfigDict(frozen=True)

    @property
    def allocated_tokens(self) -> int:
        return (self.system_prompt_tokens + self.safety_tokens + self.user_profile_tokens + self.therapeutic_context_tokens +
                self.recent_messages_tokens + self.retrieved_context_tokens + self.current_message_tokens)

    @property
    def available_tokens(self) -> int:
        return max(0, self.total_tokens - self.allocated_tokens - self.reserved_tokens)

    def with_allocation(self, **kwargs: int) -> TokenBudget:
        """Create new budget with updated allocations."""
        return self.model_copy(update=kwargs)

    def can_fit(self, tokens: int) -> bool:
        """Check if tokens can fit in available budget."""
        return tokens <= self.available_tokens

    def to_breakdown(self) -> dict[str, int]:
        """Get token breakdown dictionary."""
        return {"total": self.total_tokens, "system_prompt": self.system_prompt_tokens, "safety": self.safety_tokens,
                "user_profile": self.user_profile_tokens, "therapeutic_context": self.therapeutic_context_tokens,
                "recent_messages": self.recent_messages_tokens, "retrieved_context": self.retrieved_context_tokens,
                "current_message": self.current_message_tokens, "reserved": self.reserved_tokens, "available": self.available_tokens}


class RetrievalQuery(BaseModel):
    """Value object for memory retrieval queries."""
    query_text: str = Field(min_length=1)
    query_embedding: list[float] | None = None
    user_id: UUID
    session_id: UUID | None = None
    tiers: list[MemoryTierSpec] = Field(default_factory=lambda: [MemoryTierSpec.TIER_4_EPISODIC, MemoryTierSpec.TIER_5_SEMANTIC])
    limit: int = Field(default=10, ge=1, le=100)
    min_relevance: Decimal = Field(default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"))
    include_archived: bool = False
    time_range_days: int | None = None
    content_types: list[str] | None = None
    tags: list[str] | None = None
    model_config = ConfigDict(frozen=True)


class RetrievalResult(BaseModel):
    """Value object for a single retrieval result."""
    record_id: UUID
    content: str
    relevance_score: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    tier: MemoryTierSpec
    content_type: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(frozen=True)


class EmotionalState(BaseModel):
    """Value object representing emotional state at a point in time."""
    valence: Decimal = Field(ge=Decimal("-1"), le=Decimal("1"))
    arousal: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    dominance: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    primary_emotion: str = "neutral"
    secondary_emotions: list[str] = Field(default_factory=list)
    confidence: Decimal = Field(default=Decimal("0.7"), ge=Decimal("0"), le=Decimal("1"))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(frozen=True)

    @classmethod
    def neutral(cls) -> EmotionalState:
        return cls(valence=Decimal("0"), arousal=Decimal("0.3"), dominance=Decimal("0.5"), primary_emotion="neutral")

    @classmethod
    def positive(cls, emotion: str = "happy") -> EmotionalState:
        return cls(valence=Decimal("0.6"), arousal=Decimal("0.5"), dominance=Decimal("0.6"), primary_emotion=emotion)

    @classmethod
    def negative(cls, emotion: str = "sad") -> EmotionalState:
        return cls(valence=Decimal("-0.5"), arousal=Decimal("0.4"), dominance=Decimal("0.3"), primary_emotion=emotion)

    def is_positive(self) -> bool:
        return self.valence > Decimal("0.2")

    def is_negative(self) -> bool:
        return self.valence < Decimal("-0.2")

    def intensity(self) -> Decimal:
        return abs(self.valence) * self.arousal


class ConsolidationRequest(BaseModel):
    """Value object for memory consolidation request."""
    user_id: UUID
    session_id: UUID
    messages: list[dict[str, Any]]
    generate_summary: bool = True
    extract_facts: bool = True
    build_knowledge_graph: bool = True
    apply_decay: bool = True
    max_summary_tokens: int = Field(default=500, ge=50, le=2000)
    model_config = ConfigDict(frozen=True)


class ConsolidationOutcome(BaseModel):
    """Value object for consolidation results."""
    consolidation_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    summary_generated: str | None = None
    facts_extracted: int = Field(default=0, ge=0)
    triples_created: int = Field(default=0, ge=0)
    memories_decayed: int = Field(default=0, ge=0)
    memories_archived: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class ContextWindow(BaseModel):
    """Value object representing the current LLM context window."""
    window_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: UUID
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=8000, ge=1000)
    priority_content: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(frozen=True)

    @property
    def available_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def utilization(self) -> Decimal:
        if self.max_tokens == 0:
            return Decimal("0")
        return Decimal(str(self.total_tokens / self.max_tokens))

    def can_add(self, tokens: int) -> bool:
        return tokens <= self.available_tokens

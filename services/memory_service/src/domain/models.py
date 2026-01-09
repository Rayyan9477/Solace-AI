"""
Solace-AI Memory Service - Domain models and result types.
Contains data classes for memory records, session state, and operation results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryServiceSettings(BaseSettings):
    """Configuration for memory service behavior."""
    working_memory_max_tokens: int = Field(default=8000, description="Max working memory tokens")
    session_memory_ttl_hours: int = Field(default=24, description="Session memory TTL")
    enable_auto_consolidation: bool = Field(default=True, description="Auto-consolidate on session end")
    enable_decay: bool = Field(default=True, description="Enable Ebbinghaus decay model")
    max_history_per_user: int = Field(default=100, description="Max sessions per user")
    context_assembly_timeout_ms: int = Field(default=100, description="Context assembly timeout")
    retrieval_result_limit: int = Field(default=50, description="Max retrieval results")
    model_config = SettingsConfigDict(env_prefix="MEMORY_SERVICE_", env_file=".env", extra="ignore")


@dataclass
class MemoryRecord:
    """Internal memory record representation."""
    record_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    tier: str = "tier_3_session"
    content: str = ""
    content_type: str = "message"
    retention_category: str = "medium_term"
    importance_score: Decimal = Decimal("0.5")
    emotional_valence: Decimal | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_strength: Decimal = Decimal("1.0")


@dataclass
class SessionState:
    """Active session state."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[MemoryRecord] = field(default_factory=list)
    working_memory_tokens: int = 0
    session_type: str = "therapeutic"
    metadata: dict[str, Any] = field(default_factory=dict)


class StoreMemoryResult(BaseModel):
    """Result from storing memory."""
    record_id: UUID = Field(default_factory=uuid4)
    tier: str = Field(default="tier_3_session")
    stored: bool = Field(default=True)
    storage_time_ms: int = Field(default=0, ge=0)


class RetrieveMemoryResult(BaseModel):
    """Result from memory retrieval."""
    records: list[Any] = Field(default_factory=list)
    total_found: int = Field(default=0)
    retrieval_time_ms: int = Field(default=0, ge=0)
    tiers_searched: list[str] = Field(default_factory=list)


class ContextAssemblyResult(BaseModel):
    """Result from context assembly."""
    context_id: UUID = Field(default_factory=uuid4)
    assembled_context: str = Field(default="")
    total_tokens: int = Field(default=0, ge=0)
    token_breakdown: dict[str, int] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    assembly_time_ms: int = Field(default=0, ge=0)
    retrieval_count: int = Field(default=0)


class SessionStartResult(BaseModel):
    """Result from starting session."""
    session_id: UUID = Field(default_factory=uuid4)
    session_number: int = Field(default=1)
    previous_session_summary: str | None = Field(default=None)
    user_profile_loaded: bool = Field(default=True)


class SessionEndResult(BaseModel):
    """Result from ending session."""
    message_count: int = Field(default=0)
    duration_minutes: int = Field(default=0)
    summary: str | None = Field(default=None)
    consolidation_triggered: bool = Field(default=False)
    key_topics: list[str] = Field(default_factory=list)


class AddMessageResult(BaseModel):
    """Result from adding message."""
    message_id: UUID = Field(default_factory=uuid4)
    stored_to_tier: str = Field(default="tier_3_session")
    working_memory_updated: bool = Field(default=True)
    storage_time_ms: int = Field(default=0, ge=0)


class ConsolidationResult(BaseModel):
    """Result from consolidation."""
    consolidation_id: UUID = Field(default_factory=uuid4)
    summary_generated: str | None = Field(default=None)
    facts_extracted: int = Field(default=0)
    knowledge_nodes_updated: int = Field(default=0)
    memories_decayed: int = Field(default=0)
    memories_archived: int = Field(default=0)
    consolidation_time_ms: int = Field(default=0, ge=0)


class UserProfileResult(BaseModel):
    """Result from user profile retrieval."""
    total_sessions: int = Field(default=0)
    first_session_date: datetime | None = Field(default=None)
    last_session_date: datetime | None = Field(default=None)
    profile_facts: dict[str, Any] = Field(default_factory=dict)
    knowledge_graph: dict[str, Any] | None = Field(default=None)
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
    therapeutic_context: dict[str, Any] = Field(default_factory=dict)

"""
Solace-AI Memory Service - API Schemas and DTOs.
Pydantic models for request/response data transfer objects.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """Memory tier levels."""
    INPUT_BUFFER = "tier_1_input"
    WORKING_MEMORY = "tier_2_working"
    SESSION_MEMORY = "tier_3_session"
    EPISODIC_MEMORY = "tier_4_episodic"
    SEMANTIC_MEMORY = "tier_5_semantic"


class RetentionCategory(str, Enum):
    """Memory retention categories for decay model."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"
    MEDIUM_TERM = "medium_term"
    SHORT_TERM = "short_term"


class MemoryRecordDTO(BaseModel):
    """Memory record data transfer object."""
    record_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    tier: MemoryTier = Field(..., description="Memory tier")
    content: str = Field(..., min_length=1, description="Memory content")
    content_type: str = Field(default="message", description="Content type")
    retention_category: RetentionCategory = Field(default=RetentionCategory.MEDIUM_TERM)
    importance_score: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    emotional_valence: Decimal | None = Field(default=None, ge=-1, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StoreMemoryRequest(BaseModel):
    """Request to store a memory."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    content: str = Field(..., min_length=1, max_length=50000, description="Content to store")
    content_type: str = Field(default="message", description="Type of content")
    tier: MemoryTier = Field(default=MemoryTier.SESSION_MEMORY, description="Target tier")
    retention_category: RetentionCategory = Field(default=RetentionCategory.MEDIUM_TERM)
    importance_score: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreMemoryResponse(BaseModel):
    """Response from memory storage."""
    record_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    tier: MemoryTier = Field(...)
    stored: bool = Field(default=True)
    storage_time_ms: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrieveMemoryRequest(BaseModel):
    """Request to retrieve memories."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session filter")
    tiers: list[MemoryTier] = Field(default_factory=list, description="Tiers to search")
    query: str | None = Field(default=None, description="Semantic search query")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")
    include_embeddings: bool = Field(default=False, description="Include vectors")
    min_importance: Decimal = Field(default=Decimal("0.0"), ge=0, le=1)
    time_range_hours: int | None = Field(default=None, ge=1, description="Time range filter")


class RetrieveMemoryResponse(BaseModel):
    """Response from memory retrieval."""
    user_id: UUID = Field(...)
    records: list[MemoryRecordDTO] = Field(default_factory=list)
    total_found: int = Field(default=0)
    retrieval_time_ms: int = Field(..., ge=0)
    tiers_searched: list[MemoryTier] = Field(default_factory=list)


class ContextAssemblyRequest(BaseModel):
    """Request for LLM context assembly."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Current session")
    current_message: str | None = Field(default=None, description="Current user message")
    token_budget: int = Field(default=8000, ge=1000, le=32000, description="Token budget")
    include_safety_context: bool = Field(default=True, description="Include safety info")
    include_therapeutic_context: bool = Field(default=True, description="Include treatment")
    retrieval_query: str | None = Field(default=None, description="Optional RAG query")
    priority_topics: list[str] = Field(default_factory=list, description="Priority topics")


class ContextAssemblyResponse(BaseModel):
    """Response from context assembly."""
    context_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    assembled_context: str = Field(..., description="Final assembled context")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    token_breakdown: dict[str, int] = Field(default_factory=dict)
    sources_used: list[str] = Field(default_factory=list)
    assembly_time_ms: int = Field(..., ge=0)
    retrieval_count: int = Field(default=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStartRequest(BaseModel):
    """Request to start a new session."""
    user_id: UUID = Field(..., description="User identifier")
    session_type: str = Field(default="therapeutic", description="Session type")
    initial_context: dict[str, Any] = Field(default_factory=dict)


class SessionStartResponse(BaseModel):
    """Response from session start."""
    session_id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    session_number: int = Field(default=1)
    previous_session_summary: str | None = Field(default=None)
    user_profile_loaded: bool = Field(default=True)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionEndRequest(BaseModel):
    """Request to end a session."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID = Field(..., description="Session to end")
    trigger_consolidation: bool = Field(default=True)
    include_summary: bool = Field(default=True)


class SessionEndResponse(BaseModel):
    """Response from session end."""
    session_id: UUID = Field(...)
    user_id: UUID = Field(...)
    message_count: int = Field(default=0)
    duration_minutes: int = Field(default=0)
    summary: str | None = Field(default=None)
    consolidation_triggered: bool = Field(default=False)
    key_topics: list[str] = Field(default_factory=list)
    ended_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AddMessageRequest(BaseModel):
    """Request to add a message to session."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID = Field(..., description="Session identifier")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., min_length=1, max_length=10000)
    emotion_detected: str | None = Field(default=None)
    importance_override: Decimal | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddMessageResponse(BaseModel):
    """Response from adding a message."""
    message_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(...)
    stored_to_tier: MemoryTier = Field(default=MemoryTier.SESSION_MEMORY)
    working_memory_updated: bool = Field(default=True)
    storage_time_ms: int = Field(..., ge=0)


class ConsolidationRequest(BaseModel):
    """Request to trigger memory consolidation."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID = Field(..., description="Session to consolidate")
    extract_facts: bool = Field(default=True)
    generate_summary: bool = Field(default=True)
    update_knowledge_graph: bool = Field(default=True)
    apply_decay: bool = Field(default=True)


class ConsolidationResponse(BaseModel):
    """Response from consolidation."""
    consolidation_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(...)
    summary_generated: str | None = Field(default=None)
    facts_extracted: int = Field(default=0)
    knowledge_nodes_updated: int = Field(default=0)
    memories_decayed: int = Field(default=0)
    memories_archived: int = Field(default=0)
    consolidation_time_ms: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfileRequest(BaseModel):
    """Request to get user profile from memory."""
    user_id: UUID = Field(..., description="User identifier")
    include_knowledge_graph: bool = Field(default=False)
    include_session_history: bool = Field(default=True)
    session_limit: int = Field(default=10, ge=1, le=50)


class UserProfileResponse(BaseModel):
    """Response with user profile."""
    user_id: UUID = Field(...)
    total_sessions: int = Field(default=0)
    first_session_date: datetime | None = Field(default=None)
    last_session_date: datetime | None = Field(default=None)
    profile_facts: dict[str, Any] = Field(default_factory=dict)
    knowledge_graph: dict[str, Any] | None = Field(default=None)
    recent_sessions: list[dict[str, Any]] = Field(default_factory=list)
    therapeutic_context: dict[str, Any] = Field(default_factory=dict)

"""
Solace-AI Memory Service API - Memory CRUD and query endpoints.
Provides endpoints for 5-tier memory operations, context assembly, and consolidation.
"""
from __future__ import annotations
from typing import Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import structlog

from .domain.service import MemoryService
from .schemas import (
    MemoryTier, RetentionCategory, MemoryRecordDTO,
    StoreMemoryRequest, StoreMemoryResponse,
    RetrieveMemoryRequest, RetrieveMemoryResponse,
    ContextAssemblyRequest, ContextAssemblyResponse,
    SessionStartRequest, SessionStartResponse,
    SessionEndRequest, SessionEndResponse,
    AddMessageRequest, AddMessageResponse,
    ConsolidationRequest, ConsolidationResponse,
    UserProfileRequest, UserProfileResponse,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["memory"])


def get_memory_service(request: Request) -> MemoryService:
    """Dependency to get memory service from app state."""
    if not hasattr(request.app.state, "memory_service"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Memory service not initialized")
    return request.app.state.memory_service


@router.post("/store", response_model=StoreMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(
    request: StoreMemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> StoreMemoryResponse:
    """Store a memory record to the specified tier."""
    logger.info("store_memory_requested", user_id=str(request.user_id), tier=request.tier.value)
    result = await memory_service.store_memory(
        user_id=request.user_id, session_id=request.session_id, content=request.content,
        content_type=request.content_type, tier=request.tier.value,
        retention_category=request.retention_category.value,
        importance_score=request.importance_score, metadata=request.metadata,
    )
    return StoreMemoryResponse(
        record_id=result.record_id, user_id=request.user_id,
        tier=MemoryTier(result.tier), stored=result.stored,
        storage_time_ms=result.storage_time_ms,
    )


@router.post("/retrieve", response_model=RetrieveMemoryResponse, status_code=status.HTTP_200_OK)
async def retrieve_memories(
    request: RetrieveMemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> RetrieveMemoryResponse:
    """Retrieve memories based on query and filters."""
    logger.info("retrieve_memories_requested", user_id=str(request.user_id), query=request.query)
    result = await memory_service.retrieve_memories(
        user_id=request.user_id, session_id=request.session_id,
        tiers=[t.value for t in request.tiers] if request.tiers else None,
        query=request.query, limit=request.limit,
        min_importance=request.min_importance, time_range_hours=request.time_range_hours,
    )
    records = [MemoryRecordDTO(
        record_id=r.record_id, user_id=r.user_id, session_id=r.session_id,
        tier=MemoryTier(r.tier), content=r.content, content_type=r.content_type,
        retention_category=RetentionCategory(r.retention_category),
        importance_score=r.importance_score, metadata=r.metadata, created_at=r.created_at,
    ) for r in result.records]
    return RetrieveMemoryResponse(
        user_id=request.user_id, records=records, total_found=result.total_found,
        retrieval_time_ms=result.retrieval_time_ms,
        tiers_searched=[MemoryTier(t) for t in result.tiers_searched],
    )


@router.post("/context", response_model=ContextAssemblyResponse, status_code=status.HTTP_200_OK)
async def assemble_context(
    request: ContextAssemblyRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> ContextAssemblyResponse:
    """Assemble context for LLM within token budget."""
    logger.info("context_assembly_requested", user_id=str(request.user_id),
                token_budget=request.token_budget)
    result = await memory_service.assemble_context(
        user_id=request.user_id, session_id=request.session_id,
        current_message=request.current_message, token_budget=request.token_budget,
        include_safety=request.include_safety_context,
        include_therapeutic=request.include_therapeutic_context,
        retrieval_query=request.retrieval_query, priority_topics=request.priority_topics,
    )
    return ContextAssemblyResponse(
        context_id=result.context_id, user_id=request.user_id,
        assembled_context=result.assembled_context, total_tokens=result.total_tokens,
        token_breakdown=result.token_breakdown, sources_used=result.sources_used,
        assembly_time_ms=result.assembly_time_ms, retrieval_count=result.retrieval_count,
    )


@router.post("/session/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
async def start_session(
    request: SessionStartRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SessionStartResponse:
    """Start a new session for user."""
    logger.info("session_start_requested", user_id=str(request.user_id),
                session_type=request.session_type)
    result = await memory_service.start_session(
        user_id=request.user_id, session_type=request.session_type,
        initial_context=request.initial_context,
    )
    return SessionStartResponse(
        session_id=result.session_id, user_id=request.user_id,
        session_number=result.session_number,
        previous_session_summary=result.previous_session_summary,
        user_profile_loaded=result.user_profile_loaded,
    )


@router.post("/session/end", response_model=SessionEndResponse, status_code=status.HTTP_200_OK)
async def end_session(
    request: SessionEndRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> SessionEndResponse:
    """End a session and optionally trigger consolidation."""
    logger.info("session_end_requested", user_id=str(request.user_id),
                session_id=str(request.session_id))
    result = await memory_service.end_session(
        user_id=request.user_id, session_id=request.session_id,
        trigger_consolidation=request.trigger_consolidation,
        include_summary=request.include_summary,
    )
    return SessionEndResponse(
        session_id=request.session_id, user_id=request.user_id,
        message_count=result.message_count, duration_minutes=result.duration_minutes,
        summary=result.summary, consolidation_triggered=result.consolidation_triggered,
        key_topics=result.key_topics,
    )


@router.post("/session/message", response_model=AddMessageResponse, status_code=status.HTTP_201_CREATED)
async def add_message(
    request: AddMessageRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> AddMessageResponse:
    """Add a message to the current session."""
    logger.info("add_message_requested", user_id=str(request.user_id),
                session_id=str(request.session_id), role=request.role)
    result = await memory_service.add_message(
        user_id=request.user_id, session_id=request.session_id,
        role=request.role, content=request.content,
        emotion_detected=request.emotion_detected,
        importance_override=request.importance_override, metadata=request.metadata,
    )
    return AddMessageResponse(
        message_id=result.message_id, session_id=request.session_id,
        stored_to_tier=MemoryTier(result.stored_to_tier),
        working_memory_updated=result.working_memory_updated,
        storage_time_ms=result.storage_time_ms,
    )


@router.post("/consolidate", response_model=ConsolidationResponse, status_code=status.HTTP_200_OK)
async def trigger_consolidation(
    request: ConsolidationRequest,
    memory_service: MemoryService = Depends(get_memory_service),
) -> ConsolidationResponse:
    """Trigger memory consolidation pipeline for a session."""
    logger.info("consolidation_requested", user_id=str(request.user_id),
                session_id=str(request.session_id))
    result = await memory_service.consolidate(
        user_id=request.user_id, session_id=request.session_id,
        extract_facts=request.extract_facts, generate_summary=request.generate_summary,
        update_knowledge_graph=request.update_knowledge_graph, apply_decay=request.apply_decay,
    )
    return ConsolidationResponse(
        consolidation_id=result.consolidation_id, session_id=request.session_id,
        summary_generated=result.summary_generated, facts_extracted=result.facts_extracted,
        knowledge_nodes_updated=result.knowledge_nodes_updated,
        memories_decayed=result.memories_decayed, memories_archived=result.memories_archived,
        consolidation_time_ms=result.consolidation_time_ms,
    )


@router.get("/profile/{user_id}", response_model=UserProfileResponse, status_code=status.HTTP_200_OK)
async def get_user_profile(
    user_id: UUID,
    include_knowledge_graph: bool = False,
    include_session_history: bool = True,
    session_limit: int = 10,
    memory_service: MemoryService = Depends(get_memory_service),
) -> UserProfileResponse:
    """Get user profile from memory."""
    logger.info("user_profile_requested", user_id=str(user_id))
    result = await memory_service.get_user_profile(
        user_id=user_id, include_knowledge_graph=include_knowledge_graph,
        include_session_history=include_session_history, session_limit=session_limit,
    )
    return UserProfileResponse(
        user_id=user_id, total_sessions=result.total_sessions,
        first_session_date=result.first_session_date,
        last_session_date=result.last_session_date,
        profile_facts=result.profile_facts, knowledge_graph=result.knowledge_graph,
        recent_sessions=result.recent_sessions,
        therapeutic_context=result.therapeutic_context,
    )


@router.get("/status", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_service_status(
    memory_service: MemoryService = Depends(get_memory_service),
) -> dict[str, Any]:
    """Get memory service status and statistics."""
    return await memory_service.get_status()


@router.delete("/user/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_user_data(
    user_id: UUID,
    memory_service: MemoryService = Depends(get_memory_service),
) -> Response:
    """Delete all memory data for a user (GDPR compliance)."""
    logger.info("user_data_deletion_requested", user_id=str(user_id))
    await memory_service.delete_user_data(user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

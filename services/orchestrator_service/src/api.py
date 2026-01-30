"""
Solace-AI Orchestrator Service - API Endpoints.
Chat endpoints, WebSocket handler, and REST API for multi-agent orchestration.
"""
from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from .langgraph import (
    OrchestratorState,
    OrchestratorGraphBuilder,
    create_initial_state,
)

# Authentication dependencies from shared security library
try:
    from solace_security.middleware import (
        AuthenticatedUser,
        AuthenticatedService,
        get_current_user,
        get_current_user_optional,
        get_current_service,
        require_roles,
        require_permissions,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from typing import Optional

    @_dataclass
    class AuthenticatedUser:
        user_id: str
        token_type: str = "access"
        roles: list = None
        permissions: list = None

    @_dataclass
    class AuthenticatedService:
        service_name: str
        permissions: list = None

    async def get_current_user() -> AuthenticatedUser:
        raise HTTPException(status_code=501, detail="Authentication not configured")

    async def get_current_user_optional() -> Optional[AuthenticatedUser]:
        return None

    async def get_current_service() -> AuthenticatedService:
        raise HTTPException(status_code=501, detail="Service auth not configured")

    def require_roles(*roles):
        return get_current_user

    def require_permissions(*perms):
        return get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["orchestrator"])


class ChatMessageRequest(BaseModel):
    """Request model for chat message processing."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message to process")
    user_id: str = Field(..., min_length=1, description="User identifier")
    session_id: str = Field(..., min_length=1, description="Session identifier")
    thread_id: str | None = Field(default=None, description="Thread ID for conversation continuity")
    conversation_context: str | None = Field(default=None, description="Pre-loaded conversation context")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional request metadata")


class ChatMessageResponse(BaseModel):
    """Response model for chat message processing."""
    response: str = Field(..., description="Generated response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    session_id: str = Field(..., description="Session identifier")
    intent: str = Field(..., description="Classified user intent")
    intent_confidence: float = Field(..., description="Intent classification confidence")
    safety_flags: dict[str, Any] = Field(..., description="Safety assessment results")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    agents_used: list[str] = Field(..., description="Agents that contributed to response")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class SessionCreateRequest(BaseModel):
    """Request model for session creation."""
    user_id: str = Field(..., description="User identifier")
    initial_context: str | None = Field(default=None, description="Initial conversation context")
    metadata: dict[str, Any] | None = Field(default=None, description="Session metadata")


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Created session identifier")
    thread_id: str = Field(..., description="Thread ID for checkpointing")
    created_at: str = Field(..., description="Session creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")


class HealthResponse(BaseModel):
    """Response model for health status."""
    status: str = Field(..., description="Service health status")
    graph_ready: bool = Field(..., description="Whether the orchestration graph is ready")
    active_sessions: int = Field(..., description="Number of active sessions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    thread_id: str = Field(..., description="Thread identifier")
    messages: list[dict[str, Any]] = Field(..., description="Conversation messages")
    message_count: int = Field(..., description="Total message count")
    last_activity: str | None = Field(default=None, description="Last activity timestamp")


def get_graph_builder(request: Request) -> OrchestratorGraphBuilder:
    """Dependency to get the graph builder from app state."""
    if not hasattr(request.app.state, "graph_builder"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Graph builder not initialized")
    return request.app.state.graph_builder


@router.post("/chat", response_model=ChatMessageResponse, summary="Process chat message")
async def process_chat_message(
    request_data: ChatMessageRequest,
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    graph_builder: OrchestratorGraphBuilder = Depends(get_graph_builder),
) -> ChatMessageResponse:
    """
    Process a user chat message through the multi-agent orchestration system.

    The message goes through safety pre-check, intent classification, agent routing,
    parallel agent processing, response aggregation, and safety post-check.
    """
    start_time = time.perf_counter()
    request_id = str(uuid4())
    logger.info(
        "chat_request_received",
        request_id=request_id,
        user_id=request_data.user_id,
        session_id=request_data.session_id,
        message_length=len(request_data.message),
    )
    thread_id = request_data.thread_id or str(uuid4())
    initial_state = create_initial_state(
        user_id=request_data.user_id,
        session_id=request_data.session_id,
        message=request_data.message,
        thread_id=thread_id,
        conversation_context=request_data.conversation_context,
        metadata={**(request_data.metadata or {}), "request_id": request_id},
    )
    try:
        result_state = await graph_builder.invoke(initial_state, thread_id=thread_id)
    except Exception as e:
        logger.error("chat_processing_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing chat message")
    processing_time_ms = (time.perf_counter() - start_time) * 1000
    agents_used = []
    for result in result_state.get("agent_results", []):
        agent_type = result.get("agent_type")
        if agent_type and agent_type not in agents_used:
            agents_used.append(agent_type)
    logger.info(
        "chat_request_completed",
        request_id=request_id,
        processing_time_ms=round(processing_time_ms, 2),
        intent=result_state.get("intent"),
        agents_used=agents_used,
    )
    return ChatMessageResponse(
        response=result_state.get("final_response", "I'm here to help. How can I support you today?"),
        thread_id=thread_id,
        session_id=request_data.session_id,
        intent=result_state.get("intent", "general_chat"),
        intent_confidence=result_state.get("intent_confidence", 0.0),
        safety_flags=result_state.get("safety_flags", {}),
        processing_time_ms=processing_time_ms,
        agents_used=agents_used,
        metadata={"request_id": request_id, "phase": result_state.get("processing_phase", "completed")},
    )


@router.post("/sessions", response_model=SessionCreateResponse, summary="Create new session")
async def create_session(request_data: SessionCreateRequest, request: Request) -> SessionCreateResponse:
    """
    Create a new orchestration session for a user.

    Returns a session ID and thread ID for conversation continuity.
    """
    session_id = str(uuid4())
    thread_id = str(uuid4())
    created_at = datetime.now(timezone.utc)
    logger.info("session_created", session_id=session_id, user_id=request_data.user_id)
    return SessionCreateResponse(
        session_id=session_id,
        thread_id=thread_id,
        created_at=created_at.isoformat(),
        metadata={**(request_data.metadata or {}), "user_id": request_data.user_id, "initial_context": request_data.initial_context is not None},
    )


@router.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse, summary="Get conversation history")
async def get_conversation_history(
    session_id: str,
    thread_id: str = Query(..., description="Thread ID for conversation lookup"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum messages to return"),
    request: Request = None,
    graph_builder: OrchestratorGraphBuilder = Depends(get_graph_builder),
) -> ConversationHistoryResponse:
    """
    Retrieve conversation history for a session/thread.

    Uses the checkpointer to retrieve stored conversation state.
    """
    checkpointer = graph_builder.get_checkpointer()
    messages: list[dict[str, Any]] = []
    last_activity: str | None = None
    if checkpointer:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = checkpointer.get(config)
            if checkpoint:
                channel_values = checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])[-limit:]
                if messages:
                    last_msg = messages[-1]
                    last_activity = last_msg.get("timestamp")
        except Exception as e:
            logger.warning("history_retrieval_error", thread_id=thread_id, error=str(e))
    return ConversationHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        message_count=len(messages),
        last_activity=last_activity,
    )


@router.get("/health/detailed", response_model=HealthResponse, summary="Detailed health check")
async def detailed_health_check(request: Request) -> HealthResponse:
    """Get detailed health status of the orchestrator service."""
    graph_ready = hasattr(request.app.state, "compiled_graph") and request.app.state.compiled_graph is not None
    active_connections = getattr(request.app.state, "active_connections", {})
    active_sessions = sum(len(conns) for conns in active_connections.values())
    return HealthResponse(
        status="healthy" if graph_ready else "degraded",
        graph_ready=graph_ready,
        active_sessions=active_sessions,
        uptime_seconds=0.0,
    )


@router.websocket("/ws/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_id: str = Query(..., description="User identifier"),
) -> None:
    """
    WebSocket endpoint for real-time chat interaction.

    Provides streaming responses and maintains persistent connection.
    """
    await websocket.accept()
    thread_id = str(uuid4())
    connection_id = str(uuid4())
    active_connections: dict[str, set] = getattr(websocket.app.state, "active_connections", {})
    if session_id not in active_connections:
        active_connections[session_id] = set()
    active_connections[session_id].add(connection_id)
    logger.info(
        "websocket_connected",
        session_id=session_id,
        user_id=user_id,
        connection_id=connection_id,
        thread_id=thread_id,
    )
    graph_builder: OrchestratorGraphBuilder | None = getattr(websocket.app.state, "graph_builder", None)
    if not graph_builder:
        await websocket.send_json({"type": "error", "message": "Service not ready"})
        await websocket.close(code=1011)
        return
    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "thread_id": thread_id,
            "connection_id": connection_id,
        })
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            message_type = data.get("type", "message")
            if message_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
                continue
            if message_type == "message":
                message_content = data.get("message", "")
                if not message_content:
                    await websocket.send_json({"type": "error", "message": "Empty message"})
                    continue
                start_time = time.perf_counter()
                await websocket.send_json({"type": "processing", "status": "started"})
                initial_state = create_initial_state(
                    user_id=user_id,
                    session_id=session_id,
                    message=message_content,
                    thread_id=thread_id,
                    metadata={"connection_id": connection_id, "via_websocket": True},
                )
                try:
                    result_state = await graph_builder.invoke(initial_state, thread_id=thread_id)
                    processing_time_ms = (time.perf_counter() - start_time) * 1000
                    await websocket.send_json({
                        "type": "response",
                        "response": result_state.get("final_response", ""),
                        "intent": result_state.get("intent", "general_chat"),
                        "intent_confidence": result_state.get("intent_confidence", 0.0),
                        "safety_flags": result_state.get("safety_flags", {}),
                        "processing_time_ms": round(processing_time_ms, 2),
                        "thread_id": thread_id,
                    })
                except Exception as e:
                    logger.error("websocket_processing_error", error=str(e), session_id=session_id)
                    await websocket.send_json({"type": "error", "message": "Processing error occurred"})
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id, connection_id=connection_id)
    except Exception as e:
        logger.error("websocket_error", error=str(e), session_id=session_id)
    finally:
        if session_id in active_connections:
            active_connections[session_id].discard(connection_id)
            if not active_connections[session_id]:
                del active_connections[session_id]


@router.post("/batch", summary="Process batch of messages")
async def process_batch_messages(
    messages: list[ChatMessageRequest],
    request: Request,
    graph_builder: OrchestratorGraphBuilder = Depends(get_graph_builder),
) -> list[ChatMessageResponse]:
    """
    Process a batch of chat messages in parallel.

    Useful for bulk processing or replay scenarios.
    """
    if len(messages) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 50 messages per batch")
    async def process_single(msg: ChatMessageRequest) -> ChatMessageResponse:
        start_time = time.perf_counter()
        thread_id = msg.thread_id or str(uuid4())
        initial_state = create_initial_state(
            user_id=msg.user_id,
            session_id=msg.session_id,
            message=msg.message,
            thread_id=thread_id,
            conversation_context=msg.conversation_context,
            metadata=msg.metadata or {},
        )
        result_state = await graph_builder.invoke(initial_state, thread_id=thread_id)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        agents_used = [r.get("agent_type") for r in result_state.get("agent_results", []) if r.get("agent_type")]
        return ChatMessageResponse(
            response=result_state.get("final_response", ""),
            thread_id=thread_id,
            session_id=msg.session_id,
            intent=result_state.get("intent", "general_chat"),
            intent_confidence=result_state.get("intent_confidence", 0.0),
            safety_flags=result_state.get("safety_flags", {}),
            processing_time_ms=processing_time_ms,
            agents_used=list(set(agents_used)),
            metadata={},
        )
    results = await asyncio.gather(*[process_single(msg) for msg in messages], return_exceptions=True)
    responses = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("batch_item_error", index=i, error=str(result))
            responses.append(ChatMessageResponse(
                response="Error processing message",
                thread_id=messages[i].thread_id or "",
                session_id=messages[i].session_id,
                intent="error",
                intent_confidence=0.0,
                safety_flags={},
                processing_time_ms=0.0,
                agents_used=[],
                metadata={"error": str(result)},
            ))
        else:
            responses.append(result)
    return responses


@router.get("/agents", summary="List available agents")
async def list_agents() -> dict[str, Any]:
    """List all available agents in the orchestration system."""
    return {
        "agents": [
            {"type": "safety", "priority": 0, "description": "Crisis detection and safety monitoring", "always_active": True},
            {"type": "supervisor", "priority": 1, "description": "Intent classification and routing", "always_active": True},
            {"type": "diagnosis", "priority": 2, "description": "Symptom assessment and clinical screening", "always_active": False},
            {"type": "therapy", "priority": 2, "description": "Evidence-based therapeutic interventions", "always_active": False},
            {"type": "personality", "priority": 3, "description": "Big Five personality adaptation", "always_active": False},
            {"type": "chat", "priority": 3, "description": "General conversation handling", "always_active": False},
            {"type": "aggregator", "priority": 4, "description": "Response aggregation and generation", "always_active": True},
        ],
        "total_agents": 7,
    }

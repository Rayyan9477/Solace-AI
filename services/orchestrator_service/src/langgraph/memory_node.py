"""
Solace-AI Orchestrator Service - Memory Retrieval Node.
Retrieves and assembles context from the Memory Service for LLM consumption.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from uuid import UUID
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
    ProcessingPhase,
)
from ..infrastructure.clients import MemoryServiceClient

logger = structlog.get_logger(__name__)


class MemoryNodeSettings(BaseSettings):
    """Configuration for memory retrieval node."""
    token_budget: int = Field(default=8000, ge=1000, le=32000)
    include_safety_context: bool = Field(default=True)
    include_therapeutic_context: bool = Field(default=True)
    fallback_on_error: bool = Field(default=True)
    timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    enable_retrieval: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_MEMORY_NODE_",
        env_file=".env",
        extra="ignore"
    )


class MemoryRetrievalNode:
    """
    Memory retrieval node for context assembly.
    Fetches relevant context from Memory Service to enrich agent processing.
    """

    def __init__(self, settings: MemoryNodeSettings | None = None) -> None:
        self._settings = settings or MemoryNodeSettings()
        self._client = MemoryServiceClient()
        self._retrieval_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and retrieve memory context.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._retrieval_count += 1
        
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        current_message = state.get("current_message", "")

        logger.info(
            "memory_retrieval_processing",
            user_id=user_id,
            session_id=session_id,
            message_length=len(current_message),
        )

        if not self._settings.enable_retrieval:
            logger.info("memory_retrieval_disabled")
            return self._build_empty_context()

        try:
            result = await self._retrieve_context(
                user_id=user_id,
                session_id=session_id,
                current_message=current_message,
            )
            return self._build_state_update(result, success=True)
        except Exception as e:
            logger.error("memory_retrieval_error", error=str(e), exc_info=True)
            if self._settings.fallback_on_error:
                return self._build_empty_context(error=str(e))
            raise

    async def _retrieve_context(
        self,
        user_id: str,
        session_id: str,
        current_message: str,
    ) -> dict[str, Any]:
        """Retrieve context from Memory Service."""
        start_time = datetime.now(timezone.utc)

        try:
            user_uuid = UUID(user_id)
            session_uuid = UUID(session_id)
        except ValueError as e:
            logger.warning("invalid_uuid_format", error=str(e))
            raise ValueError(f"Invalid UUID format: {e}")

        # Call Memory Service context assembly endpoint
        response = await self._client.assemble_context(
            user_id=user_uuid,
            session_id=session_uuid,
            current_message=current_message,
            token_budget=self._settings.token_budget,
        )

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if not response.success:
            error_msg = response.error or "Unknown error"
            logger.error(
                "memory_service_error",
                error=error_msg,
                status_code=response.status_code,
            )
            raise RuntimeError(f"Memory service request failed: {error_msg}")

        data = response.data or {}
        
        logger.info(
            "memory_retrieval_complete",
            total_tokens=data.get("total_tokens", 0),
            retrieval_count=data.get("retrieval_count", 0),
            sources_count=len(data.get("sources_used", [])),
            elapsed_ms=elapsed_ms,
        )

        return data

    def _build_state_update(
        self,
        memory_data: dict[str, Any],
        success: bool,
    ) -> dict[str, Any]:
        """Build state update from memory retrieval results."""
        assembled_context = memory_data.get("assembled_context", "")
        total_tokens = memory_data.get("total_tokens", 0)
        sources_used = memory_data.get("sources_used", [])
        retrieval_count = memory_data.get("retrieval_count", 0)
        assembly_time_ms = memory_data.get("assembly_time_ms", 0)

        # Update conversation_context for backward compatibility
        conversation_context = assembled_context

        agent_result = AgentResult(
            agent_type=AgentType.MEMORY,  # Using AGGREGATOR as closest match
            success=success,
            confidence=0.9 if success else 0.0,
            processing_time_ms=assembly_time_ms,
            metadata={
                "phase": "memory_retrieval",
                "total_tokens": total_tokens,
                "sources_count": len(sources_used),
                "retrieval_count": retrieval_count,
                "sources_used": sources_used,
            },
        )

        logger.info(
            "memory_node_complete",
            context_length=len(assembled_context),
            tokens_used=total_tokens,
            sources_count=len(sources_used),
        )

        return {
            "conversation_context": conversation_context,
            "assembled_context": assembled_context,
            "memory_sources": sources_used,
            "memory_context": {
                "total_tokens": total_tokens,
                "retrieval_count": retrieval_count,
                "assembly_time_ms": assembly_time_ms,
                "token_breakdown": memory_data.get("token_breakdown", {}),
            },
            "processing_phase": ProcessingPhase.CONTEXT_LOADING.value,
            "agent_results": [agent_result.to_dict()],
        }

    def _build_empty_context(self, error: str | None = None) -> dict[str, Any]:
        """Build empty context when retrieval fails or is disabled."""
        agent_result = AgentResult(
            agent_type=AgentType.MEMORY,
            success=False,
            confidence=0.0,
            metadata={
                "phase": "memory_retrieval",
                "fallback_mode": True,
                "error": error,
            },
        )

        logger.warning("memory_retrieval_fallback", error=error)

        return {
            "conversation_context": "",
            "assembled_context": "",
            "memory_sources": [],
            "memory_context": {
                "total_tokens": 0,
                "retrieval_count": 0,
                "assembly_time_ms": 0,
                "fallback_mode": True,
            },
            "processing_phase": ProcessingPhase.CONTEXT_LOADING.value,
            "agent_results": [agent_result.to_dict()],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get node statistics."""
        return {
            "total_retrievals": self._retrieval_count,
            "token_budget": self._settings.token_budget,
            "retrieval_enabled": self._settings.enable_retrieval,
        }


_shared_memory_node: MemoryRetrievalNode | None = None


def _get_memory_node() -> MemoryRetrievalNode:
    """Get or create a shared MemoryRetrievalNode singleton.

    Reuses the same httpx connection pool across invocations instead of
    creating (and leaking) a new HTTP client per LangGraph call.
    """
    global _shared_memory_node
    if _shared_memory_node is None:
        _shared_memory_node = MemoryRetrievalNode()
    return _shared_memory_node


async def memory_retrieval_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for memory retrieval processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    node = _get_memory_node()
    return await node.process(state)

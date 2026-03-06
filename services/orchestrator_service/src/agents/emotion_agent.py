"""
Solace-AI Orchestrator Service - Emotion Agent.
Handles emotional support routing and empathetic response generation.
"""
from __future__ import annotations
import time
from typing import Any
import structlog

from ..langgraph.state_schema import (
    OrchestratorState, AgentType, AgentResult, ProcessingPhase,
)

logger = structlog.get_logger(__name__)


async def emotion_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Emotion agent - provides emotional support and empathetic responses."""
    start_time = time.perf_counter()
    message = state.get("current_message", "")
    user_id = state.get("user_id", "")
    logger.info("emotion_agent_processing", user_id=user_id, message_length=len(message))

    response = (
        "I hear you, and I want you to know that your feelings are valid. "
        "It takes courage to share what you're going through."
    )

    processing_time = int((time.perf_counter() - start_time) * 1000)
    agent_result = AgentResult(
        agent_type=AgentType.EMOTION,
        success=True,
        response_content=response,
        confidence=0.85,
        processing_time_ms=processing_time,
        metadata={"intent": "emotional_support"},
    )
    logger.info("emotion_agent_complete", processing_time_ms=processing_time)
    return {
        "agent_results": [agent_result.to_dict()],
        "processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value,
        "emotion_output": {"empathy_applied": True},
    }

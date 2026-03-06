"""
Solace-AI Orchestrator Service - Assessment Agent.
Routes clinical assessment requests to diagnosis service.
"""
from __future__ import annotations
import time
from typing import Any
import structlog

from ..langgraph.state_schema import (
    OrchestratorState, AgentType, AgentResult, ProcessingPhase,
)

logger = structlog.get_logger(__name__)


async def assessment_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Assessment agent - handles clinical assessment routing."""
    start_time = time.perf_counter()
    message = state.get("current_message", "")
    user_id = state.get("user_id", "")
    logger.info("assessment_agent_processing", user_id=user_id, message_length=len(message))

    response = (
        "I'd like to help understand what you're experiencing. "
        "Let me ask a few questions to get a clearer picture. "
        "Remember, this is not a formal diagnosis - it's a way to better understand your situation."
    )

    processing_time = int((time.perf_counter() - start_time) * 1000)
    agent_result = AgentResult(
        agent_type=AgentType.ASSESSMENT,
        success=True,
        response_content=response,
        confidence=0.8,
        processing_time_ms=processing_time,
        metadata={"intent": "assessment"},
    )
    logger.info("assessment_agent_complete", processing_time_ms=processing_time)
    return {
        "agent_results": [agent_result.to_dict()],
        "processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value,
        "diagnosis_output": {"assessment_initiated": True, "message": message},
    }

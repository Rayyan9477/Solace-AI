"""
Solace-AI Orchestrator Service - Assessment Agent.
Routes clinical assessment requests to diagnosis service.
"""
from __future__ import annotations
import time
from typing import Any
import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState, AgentType, AgentResult, ProcessingPhase,
)

logger = structlog.get_logger(__name__)


class AssessmentAgentSettings(BaseSettings):
    """Configuration for the assessment agent."""
    service_url: str = Field(default="http://localhost:8004")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    fallback_on_service_error: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_ASSESSMENT_AGENT_",
        env_file=".env",
        extra="ignore",
    )


class AssessmentServiceClient:
    """HTTP client for Diagnosis Service assessment endpoint."""

    def __init__(self, settings: AssessmentAgentSettings) -> None:
        self._settings = settings
        self._base_url = settings.service_url.rstrip("/")

    async def assess(
        self,
        user_id: str,
        session_id: str,
        message: str,
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Forward assessment request to the diagnosis service."""
        url = f"{self._base_url}/api/v1/diagnosis/assess"
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "conversation_history": conversation_history,
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "assessment_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "assessment_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Assessment service request failed after retries")


_assessment_settings: AssessmentAgentSettings | None = None
_assessment_client: AssessmentServiceClient | None = None


def _get_client() -> AssessmentServiceClient:
    """Get or create a shared assessment service client."""
    global _assessment_settings, _assessment_client
    if _assessment_client is None:
        _assessment_settings = AssessmentAgentSettings()
        _assessment_client = AssessmentServiceClient(_assessment_settings)
    return _assessment_client


async def assessment_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """Assessment agent - routes clinical assessment requests to diagnosis service."""
    start_time = time.perf_counter()
    message = state.get("current_message", "")
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    messages = state.get("messages", [])
    logger.info("assessment_agent_processing", user_id=user_id, message_length=len(message))

    try:
        client = _get_client()
        result = await client.assess(
            user_id=user_id,
            session_id=session_id,
            message=message,
            conversation_history=messages[-10:] if messages else [],
        )
        response = result.get("response_text", "")
        next_question = result.get("next_question")
        if next_question:
            response = f"{response}\n\n{next_question}"
        confidence = float(result.get("confidence_score", 0.8))
        metadata = {
            "intent": "assessment",
            "symptoms_count": len(result.get("extracted_symptoms", [])),
            "current_phase": result.get("current_phase", "RAPPORT"),
        }
    except Exception as e:
        logger.warning("assessment_service_call_failed", error=str(e))
        response = (
            "I'd like to help understand what you're experiencing. "
            "Let me ask a few questions to get a clearer picture. "
            "Could you tell me more about what prompted you to seek an assessment today?"
        )
        confidence = 0.6
        metadata = {"intent": "assessment", "fallback_mode": True}

    processing_time = int((time.perf_counter() - start_time) * 1000)
    agent_result = AgentResult(
        agent_type=AgentType.ASSESSMENT,
        success=True,
        response_content=response,
        confidence=confidence,
        processing_time_ms=processing_time,
        metadata=metadata,
    )
    logger.info("assessment_agent_complete", processing_time_ms=processing_time)
    return {
        "agent_results": [agent_result.to_dict()],
        "processing_phase": ProcessingPhase.PARALLEL_PROCESSING.value,
        "diagnosis_output": {"assessment_initiated": True, "message": message},
    }

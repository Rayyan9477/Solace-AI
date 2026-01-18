"""
Solace-AI Orchestrator Service - Safety Agent.
Coordinates with the Safety Service for crisis detection, risk assessment, and escalation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4
import httpx
import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    ProcessingPhase,
    RiskLevel,
    AgentResult,
    SafetyFlags,
)

logger = structlog.get_logger(__name__)


class SafetyCheckType(str, Enum):
    """Types of safety checks."""
    PRE_CHECK = "PRE_CHECK"
    POST_CHECK = "POST_CHECK"
    FULL_ASSESSMENT = "FULL_ASSESSMENT"


class CrisisLevel(str, Enum):
    """Crisis severity levels from Safety Service."""
    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyAgentSettings(BaseSettings):
    """Configuration for the safety agent."""
    service_url: str = Field(default="http://localhost:8001")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    enable_escalation: bool = Field(default=True)
    escalation_threshold: str = Field(default="HIGH")
    include_resources_threshold: str = Field(default="ELEVATED")
    max_retries: int = Field(default=2, ge=0, le=5)
    fallback_on_service_error: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_SAFETY_AGENT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class SafetyCheckRequest:
    """Request to Safety Service for safety check."""
    user_id: str
    session_id: str
    message_id: str
    content: str
    check_type: SafetyCheckType
    include_resources: bool = False
    conversation_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to API request format."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "content": self.content,
            "check_type": self.check_type.value,
            "include_resources": self.include_resources,
            "conversation_history": self.conversation_history,
        }


@dataclass
class RiskFactorDTO:
    """Risk factor from safety assessment."""
    factor_type: str
    severity: float
    evidence: str
    confidence: float
    detection_layer: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskFactorDTO:
        """Create from dictionary."""
        return cls(
            factor_type=data.get("factor_type", "unknown"),
            severity=float(data.get("severity", 0.0)),
            evidence=data.get("evidence", ""),
            confidence=float(data.get("confidence", 0.0)),
            detection_layer=data.get("detection_layer", 1),
        )


@dataclass
class ProtectiveFactorDTO:
    """Protective factor from safety assessment."""
    factor_type: str
    strength: float
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtectiveFactorDTO:
        """Create from dictionary."""
        return cls(
            factor_type=data.get("factor_type", "unknown"),
            strength=float(data.get("strength", 0.0)),
            description=data.get("description", ""),
        )


@dataclass
class SafetyCheckResult:
    """Result from Safety Service safety check."""
    is_safe: bool
    crisis_level: CrisisLevel
    risk_score: Decimal
    risk_factors: list[RiskFactorDTO]
    protective_factors: list[ProtectiveFactorDTO]
    requires_escalation: bool
    requires_human_review: bool
    crisis_resources: list[dict[str, Any]]
    triggered_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyCheckResult:
        """Create from API response."""
        return cls(
            is_safe=data.get("is_safe", True),
            crisis_level=CrisisLevel(data.get("crisis_level", "NONE")),
            risk_score=Decimal(str(data.get("risk_score", 0.0))),
            risk_factors=[RiskFactorDTO.from_dict(rf) for rf in data.get("risk_factors", [])],
            protective_factors=[ProtectiveFactorDTO.from_dict(pf) for pf in data.get("protective_factors", [])],
            requires_escalation=data.get("requires_escalation", False),
            requires_human_review=data.get("requires_human_review", False),
            crisis_resources=data.get("crisis_resources", []),
            triggered_keywords=data.get("triggered_keywords", []),
        )

    def to_risk_level(self) -> RiskLevel:
        """Convert crisis level to orchestrator risk level."""
        mapping = {
            CrisisLevel.NONE: RiskLevel.NONE,
            CrisisLevel.LOW: RiskLevel.LOW,
            CrisisLevel.ELEVATED: RiskLevel.MODERATE,
            CrisisLevel.HIGH: RiskLevel.HIGH,
            CrisisLevel.CRITICAL: RiskLevel.CRITICAL,
        }
        return mapping.get(self.crisis_level, RiskLevel.NONE)


class SafetyServiceClient:
    """HTTP client for Safety Service communication."""

    def __init__(self, settings: SafetyAgentSettings) -> None:
        self._settings = settings
        self._base_url = settings.service_url.rstrip("/")

    async def check_safety(self, request: SafetyCheckRequest) -> SafetyCheckResult:
        """Perform safety check via Safety Service."""
        url = f"{self._base_url}/check"
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=request.to_dict())
                    response.raise_for_status()
                    return SafetyCheckResult.from_dict(response.json())
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "safety_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "safety_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Safety service check failed after retries")

    async def get_crisis_resources(self, severity: str) -> list[dict[str, Any]]:
        """Get crisis resources by severity level."""
        url = f"{self._base_url}/resources"
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.get(url, params={"severity": severity})
            response.raise_for_status()
            return response.json().get("resources", [])

    async def trigger_escalation(
        self,
        user_id: str,
        session_id: str,
        crisis_level: str,
        reason: str,
        risk_factors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Trigger escalation to human clinician."""
        url = f"{self._base_url}/escalate"
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "crisis_level": crisis_level,
            "reason": reason,
            "risk_factors": risk_factors,
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()


class SafetyAgent:
    """
    Safety monitoring agent for the orchestrator.
    Coordinates with Safety Service for crisis detection and escalation.
    """

    def __init__(self, settings: SafetyAgentSettings | None = None) -> None:
        self._settings = settings or SafetyAgentSettings()
        self._client = SafetyServiceClient(self._settings)
        self._check_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and perform safety assessment.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._check_count += 1
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        message = state.get("current_message", "")
        existing_flags = state.get("safety_flags", {})
        messages = state.get("messages", [])
        logger.info(
            "safety_agent_processing",
            user_id=user_id,
            message_length=len(message),
            existing_risk=existing_flags.get("risk_level", "none"),
        )
        try:
            result = await self._perform_safety_check(
                user_id=user_id,
                session_id=session_id,
                message=message,
                messages=messages,
            )
            return self._build_state_update(result, existing_flags)
        except Exception as e:
            logger.error("safety_agent_error", error=str(e))
            if self._settings.fallback_on_service_error:
                return self._build_fallback_response(message, existing_flags)
            raise

    async def _perform_safety_check(
        self,
        user_id: str,
        session_id: str,
        message: str,
        messages: list[dict[str, Any]],
    ) -> SafetyCheckResult:
        """Perform safety check via Safety Service."""
        include_resources = self._should_include_resources(message)
        request = SafetyCheckRequest(
            user_id=user_id,
            session_id=session_id,
            message_id=str(uuid4()),
            content=message,
            check_type=SafetyCheckType.FULL_ASSESSMENT,
            include_resources=include_resources,
            conversation_history=messages[-10:] if messages else [],
        )
        return await self._client.check_safety(request)

    def _should_include_resources(self, message: str) -> bool:
        """Determine if crisis resources should be included."""
        crisis_indicators = [
            "suicide", "kill myself", "end my life", "want to die",
            "self-harm", "hurt myself", "no reason to live",
        ]
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in crisis_indicators)

    def _build_state_update(
        self,
        result: SafetyCheckResult,
        existing_flags: dict[str, Any],
    ) -> dict[str, Any]:
        """Build state update from safety check result."""
        risk_level = result.to_risk_level()
        crisis_detected = result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        safety_flags = SafetyFlags(
            risk_level=risk_level,
            crisis_detected=crisis_detected,
            crisis_type="safety_service_detection" if crisis_detected else None,
            requires_escalation=result.requires_escalation,
            escalation_reason=self._build_escalation_reason(result) if result.requires_escalation else None,
            safety_resources_shown=len(result.crisis_resources) > 0,
            monitoring_level=self._determine_monitoring_level(result),
            contraindications=[rf.factor_type for rf in result.risk_factors if rf.severity > 0.7],
            triggered_keywords=result.triggered_keywords,
            last_assessment_at=datetime.now(timezone.utc),
        )
        agent_result = AgentResult(
            agent_type=AgentType.SAFETY,
            success=True,
            response_content=self._build_response_content(result) if crisis_detected else None,
            confidence=float(1 - result.risk_score) if result.is_safe else float(result.risk_score),
            metadata={
                "crisis_level": result.crisis_level.value,
                "risk_factors_count": len(result.risk_factors),
                "protective_factors_count": len(result.protective_factors),
                "requires_human_review": result.requires_human_review,
            },
        )
        logger.info(
            "safety_agent_complete",
            risk_level=risk_level.value,
            crisis_detected=crisis_detected,
            requires_escalation=result.requires_escalation,
        )
        return {
            "safety_flags": safety_flags.to_dict(),
            "agent_results": [agent_result.to_dict()],
        }

    def _build_fallback_response(
        self,
        message: str,
        existing_flags: dict[str, Any],
    ) -> dict[str, Any]:
        """Build fallback response when Safety Service is unavailable."""
        crisis_keywords = [
            "suicide", "kill myself", "end my life", "ending my life", "want to die",
            "self-harm", "hurt myself", "cutting", "overdose", "no reason to live",
        ]
        message_lower = message.lower()
        triggered = [kw for kw in crisis_keywords if kw in message_lower]
        crisis_detected = len(triggered) > 0
        risk_level = RiskLevel.HIGH if crisis_detected else RiskLevel.NONE
        safety_flags = SafetyFlags(
            risk_level=risk_level,
            crisis_detected=crisis_detected,
            crisis_type="fallback_detection" if crisis_detected else None,
            requires_escalation=crisis_detected,
            monitoring_level="intensive" if crisis_detected else "standard",
            triggered_keywords=triggered,
            last_assessment_at=datetime.now(timezone.utc),
        )
        agent_result = AgentResult(
            agent_type=AgentType.SAFETY,
            success=True,
            confidence=0.6,
            metadata={"fallback_mode": True, "triggered_keywords": triggered},
        )
        logger.warning(
            "safety_agent_fallback_used",
            crisis_detected=crisis_detected,
            triggered_count=len(triggered),
        )
        return {
            "safety_flags": safety_flags.to_dict(),
            "agent_results": [agent_result.to_dict()],
        }

    def _build_escalation_reason(self, result: SafetyCheckResult) -> str:
        """Build escalation reason from risk factors."""
        if not result.risk_factors:
            return f"Crisis level {result.crisis_level.value} detected"
        high_severity_factors = [rf for rf in result.risk_factors if rf.severity > 0.7]
        if high_severity_factors:
            factor_types = [rf.factor_type for rf in high_severity_factors[:3]]
            return f"High severity risk factors: {', '.join(factor_types)}"
        return f"Multiple risk factors detected ({len(result.risk_factors)} factors)"

    def _determine_monitoring_level(
        self,
        result: SafetyCheckResult,
    ) -> Literal["standard", "enhanced", "intensive"]:
        """Determine monitoring level based on assessment."""
        if result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            return "intensive"
        if result.crisis_level == CrisisLevel.ELEVATED:
            return "enhanced"
        if result.risk_factors and any(rf.severity > 0.5 for rf in result.risk_factors):
            return "enhanced"
        return "standard"

    def _build_response_content(self, result: SafetyCheckResult) -> str:
        """Build crisis response content."""
        resources_text = ""
        if result.crisis_resources:
            resources_text = "\n\nHere are some resources that can help:\n"
            for resource in result.crisis_resources[:3]:
                name = resource.get("name", "Crisis Line")
                contact = resource.get("contact", "")
                resources_text += f"- **{name}**: {contact}\n"
        return f"""I'm here with you and I hear how difficult things are right now. Your safety matters most.{resources_text}
Would you like to talk about what you're experiencing?"""

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_checks": self._check_count,
            "service_url": self._settings.service_url,
            "escalation_enabled": self._settings.enable_escalation,
        }


async def safety_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for safety agent processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    agent = SafetyAgent()
    return await agent.process(state)

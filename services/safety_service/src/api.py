"""
Solace-AI Safety Service API - Safety check and crisis management endpoints.
Provides pre-check, post-check, crisis detection, and escalation endpoints.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
import structlog

from .domain.service import SafetyService
from .domain.crisis_detector import DetectionResult
from .domain.escalation import EscalationResult

# Authentication dependencies from shared security library
from solace_security.middleware import (
    AuthenticatedUser,
    AuthenticatedService,
    get_current_user,
    get_current_service,
    require_service_permission,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["safety"])


class CrisisLevel(str, Enum):
    """Crisis severity levels aligned with system design."""
    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyCheckType(str, Enum):
    """Type of safety check to perform."""
    PRE_CHECK = "pre_check"
    POST_CHECK = "post_check"
    FULL_ASSESSMENT = "full_assessment"


class RiskFactorDTO(BaseModel):
    """Risk factor data transfer object."""
    factor_type: str = Field(..., description="Type of risk factor")
    severity: Decimal = Field(..., ge=0, le=1, description="Severity score 0-1")
    evidence: str = Field(..., description="Evidence supporting this factor")
    confidence: Decimal = Field(..., ge=0, le=1, description="Detection confidence")
    detection_layer: int = Field(..., ge=1, le=4, description="Detection layer that identified this")


class ProtectiveFactorDTO(BaseModel):
    """Protective factor data transfer object."""
    factor_type: str = Field(..., description="Type of protective factor")
    strength: Decimal = Field(..., ge=0, le=1, description="Strength of protective factor")
    description: str = Field(..., description="Description of the protective factor")


class SafetyCheckRequest(BaseModel):
    """Request for safety check."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    message_id: UUID = Field(default_factory=uuid4, description="Message identifier")
    content: str = Field(..., min_length=1, max_length=10000, description="Content to check")
    check_type: SafetyCheckType = Field(default=SafetyCheckType.PRE_CHECK, description="Type of safety check")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    include_resources: bool = Field(default=True, description="Include crisis resources in response")


class SafetyCheckResponse(BaseModel):
    """Response from safety check."""
    check_id: UUID = Field(default_factory=uuid4, description="Unique check identifier")
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    is_safe: bool = Field(..., description="Whether content passed safety check")
    crisis_level: CrisisLevel = Field(..., description="Detected crisis level")
    risk_score: Decimal = Field(..., ge=0, le=1, description="Overall risk score")
    risk_factors: list[RiskFactorDTO] = Field(default_factory=list, description="Identified risk factors")
    protective_factors: list[ProtectiveFactorDTO] = Field(default_factory=list, description="Identified protective factors")
    recommended_action: str = Field(..., description="Recommended action to take")
    requires_escalation: bool = Field(default=False, description="Whether escalation is needed")
    requires_human_review: bool = Field(default=False, description="Whether human review is needed")
    crisis_resources: list[dict[str, str]] = Field(default_factory=list, description="Crisis resources if applicable")
    detection_time_ms: int = Field(..., ge=0, description="Detection processing time")
    detection_layer: int = Field(..., ge=1, le=4, description="Highest detection layer triggered")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")


class CrisisDetectionRequest(BaseModel):
    """Request for direct crisis detection."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    content: str = Field(..., min_length=1, max_length=10000, description="Content to analyze")
    conversation_history: list[str] = Field(default_factory=list, description="Recent conversation history")
    user_risk_history: dict[str, Any] = Field(default_factory=dict, description="User's risk history")


class CrisisDetectionResponse(BaseModel):
    """Response from crisis detection."""
    detection_id: UUID = Field(default_factory=uuid4, description="Detection identifier")
    crisis_detected: bool = Field(..., description="Whether crisis was detected")
    crisis_level: CrisisLevel = Field(..., description="Crisis level detected")
    trigger_indicators: list[str] = Field(default_factory=list, description="Indicators that triggered detection")
    confidence: Decimal = Field(..., ge=0, le=1, description="Detection confidence")
    detection_layers_triggered: list[int] = Field(default_factory=list, description="Layers that detected risk")
    detection_time_ms: int = Field(..., ge=0, description="Detection time in milliseconds")


class EscalationRequest(BaseModel):
    """Request to trigger escalation."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    crisis_level: CrisisLevel = Field(..., description="Crisis level triggering escalation")
    reason: str = Field(..., min_length=1, max_length=1000, description="Reason for escalation")
    risk_factors: list[RiskFactorDTO] = Field(default_factory=list, description="Risk factors identified")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority_override: str | None = Field(default=None, description="Override priority if needed")


class EscalationResponse(BaseModel):
    """Response from escalation request."""
    escalation_id: UUID = Field(default_factory=uuid4, description="Escalation identifier")
    status: str = Field(..., description="Escalation status")
    priority: str = Field(..., description="Escalation priority")
    assigned_clinician_id: UUID | None = Field(default=None, description="Assigned clinician if any")
    notification_sent: bool = Field(default=False, description="Whether notification was sent")
    escalation_actions: list[str] = Field(default_factory=list, description="Actions taken")
    estimated_response_time_minutes: int | None = Field(default=None, description="Estimated response time")
    crisis_resources_provided: bool = Field(default=False, description="Whether resources were provided")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Escalation timestamp")


class SafetyAssessmentRequest(BaseModel):
    """Request for comprehensive safety assessment."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    messages: list[str] = Field(..., min_length=1, description="Messages to assess")
    include_trajectory_analysis: bool = Field(default=True, description="Include session trajectory")
    include_risk_prediction: bool = Field(default=True, description="Include risk prediction")


class SafetyAssessmentResponse(BaseModel):
    """Response from comprehensive safety assessment."""
    assessment_id: UUID = Field(default_factory=uuid4, description="Assessment identifier")
    user_id: UUID = Field(..., description="User identifier")
    overall_risk_level: CrisisLevel = Field(..., description="Overall risk level")
    overall_risk_score: Decimal = Field(..., ge=0, le=1, description="Overall risk score")
    message_assessments: list[dict[str, Any]] = Field(default_factory=list, description="Per-message assessments")
    trajectory_analysis: dict[str, Any] | None = Field(default=None, description="Session trajectory analysis")
    risk_prediction: dict[str, Any] | None = Field(default=None, description="Future risk prediction")
    recommendations: list[str] = Field(default_factory=list, description="Safety recommendations")
    requires_intervention: bool = Field(default=False, description="Whether intervention is needed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Assessment timestamp")


class OutputFilterRequest(BaseModel):
    """Request to filter AI-generated output for safety."""
    user_id: UUID = Field(..., description="User identifier")
    session_id: UUID | None = Field(default=None, description="Session identifier")
    original_response: str = Field(..., min_length=1, description="Original AI response")
    user_crisis_level: CrisisLevel = Field(default=CrisisLevel.NONE, description="User's current crisis level")
    include_resources: bool = Field(default=False, description="Include crisis resources")


class OutputFilterResponse(BaseModel):
    """Response from output filtering."""
    filter_id: UUID = Field(default_factory=uuid4, description="Filter operation identifier")
    filtered_response: str = Field(..., description="Filtered/modified response")
    modifications_made: list[str] = Field(default_factory=list, description="Modifications made")
    resources_appended: bool = Field(default=False, description="Whether resources were appended")
    is_safe: bool = Field(..., description="Whether output passed safety filter")
    filter_time_ms: int = Field(..., ge=0, description="Filter processing time")


def get_safety_service(request: Request) -> SafetyService:
    """Dependency to get safety service from app state."""
    if not hasattr(request.app.state, "safety_service"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Safety service not initialized")
    return request.app.state.safety_service


@router.post("/check", response_model=SafetyCheckResponse, status_code=status.HTTP_200_OK)
async def perform_safety_check(
    request: SafetyCheckRequest,
    service: AuthenticatedService = Depends(get_current_service),
    safety_service: SafetyService = Depends(get_safety_service),
) -> SafetyCheckResponse:
    """Perform safety check on content (pre-check or post-check)."""
    logger.info("safety_check_requested", user_id=str(request.user_id), check_type=request.check_type.value)
    result = await safety_service.check_safety(
        user_id=request.user_id,
        session_id=request.session_id,
        content=request.content,
        check_type=request.check_type.value,
        context=request.context,
    )
    crisis_resources = []
    if request.include_resources and result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
        crisis_resources = _get_crisis_resources(result.crisis_level)
    return SafetyCheckResponse(
        user_id=request.user_id,
        session_id=request.session_id,
        is_safe=result.is_safe,
        crisis_level=CrisisLevel(result.crisis_level.value),
        risk_score=result.risk_score,
        risk_factors=[RiskFactorDTO(**rf.model_dump()) for rf in result.risk_factors],
        protective_factors=[ProtectiveFactorDTO(**pf) for pf in result.protective_factors],
        recommended_action=result.recommended_action,
        requires_escalation=result.requires_escalation,
        requires_human_review=result.requires_human_review,
        crisis_resources=crisis_resources,
        detection_time_ms=result.detection_time_ms,
        detection_layer=result.detection_layer,
    )


@router.post("/detect-crisis", response_model=CrisisDetectionResponse, status_code=status.HTTP_200_OK)
async def detect_crisis(
    request: CrisisDetectionRequest,
    safety_service: SafetyService = Depends(get_safety_service),
) -> CrisisDetectionResponse:
    """Perform direct crisis detection on content."""
    logger.info("crisis_detection_requested", user_id=str(request.user_id))
    result = await safety_service.detect_crisis(
        user_id=request.user_id,
        content=request.content,
        conversation_history=request.conversation_history,
        user_risk_history=request.user_risk_history,
    )
    return CrisisDetectionResponse(
        crisis_detected=result.crisis_detected,
        crisis_level=CrisisLevel(result.crisis_level.value),
        trigger_indicators=result.trigger_indicators,
        confidence=result.confidence,
        detection_layers_triggered=result.detection_layers_triggered,
        detection_time_ms=result.detection_time_ms,
    )


@router.post("/escalate", response_model=EscalationResponse, status_code=status.HTTP_201_CREATED)
async def trigger_escalation(
    request: EscalationRequest,
    safety_service: SafetyService = Depends(get_safety_service),
) -> EscalationResponse:
    """Trigger escalation to human clinician."""
    logger.info("escalation_requested", user_id=str(request.user_id), crisis_level=request.crisis_level.value)
    result = await safety_service.escalate(
        user_id=request.user_id,
        session_id=request.session_id,
        crisis_level=request.crisis_level.value,
        reason=request.reason,
        context=request.context,
        priority_override=request.priority_override,
    )
    return EscalationResponse(
        escalation_id=result.escalation_id,
        status=result.status,
        priority=result.priority,
        assigned_clinician_id=result.assigned_clinician_id,
        notification_sent=result.notification_sent,
        escalation_actions=result.actions_taken,
        estimated_response_time_minutes=result.estimated_response_minutes,
        crisis_resources_provided=result.resources_provided,
    )


@router.post("/assess", response_model=SafetyAssessmentResponse, status_code=status.HTTP_200_OK)
async def perform_safety_assessment(
    request: SafetyAssessmentRequest,
    safety_service: SafetyService = Depends(get_safety_service),
) -> SafetyAssessmentResponse:
    """Perform comprehensive safety assessment on multiple messages."""
    logger.info("safety_assessment_requested", user_id=str(request.user_id), message_count=len(request.messages))
    result = await safety_service.assess_safety(
        user_id=request.user_id,
        session_id=request.session_id,
        messages=request.messages,
        include_trajectory=request.include_trajectory_analysis,
        include_prediction=request.include_risk_prediction,
    )
    return SafetyAssessmentResponse(
        assessment_id=result.assessment_id,
        user_id=request.user_id,
        overall_risk_level=CrisisLevel(result.overall_risk_level.value),
        overall_risk_score=result.overall_risk_score,
        message_assessments=result.message_assessments,
        trajectory_analysis=result.trajectory_analysis,
        risk_prediction=result.risk_prediction,
        recommendations=result.recommendations,
        requires_intervention=result.requires_intervention,
    )


@router.post("/filter-output", response_model=OutputFilterResponse, status_code=status.HTTP_200_OK)
async def filter_output(
    request: OutputFilterRequest,
    safety_service: SafetyService = Depends(get_safety_service),
) -> OutputFilterResponse:
    """Filter AI-generated output for safety before delivery."""
    logger.info("output_filter_requested", user_id=str(request.user_id))
    result = await safety_service.filter_output(
        user_id=request.user_id,
        original_response=request.original_response,
        user_crisis_level=request.user_crisis_level.value,
        include_resources=request.include_resources,
    )
    return OutputFilterResponse(
        filtered_response=result.filtered_response,
        modifications_made=result.modifications_made,
        resources_appended=result.resources_appended,
        is_safe=result.is_safe,
        filter_time_ms=result.filter_time_ms,
    )


@router.get("/resources", response_model=list[dict[str, str]], status_code=status.HTTP_200_OK)
async def get_crisis_resources(level: CrisisLevel = CrisisLevel.HIGH) -> list[dict[str, str]]:
    """Get crisis resources for specified level."""
    return _get_crisis_resources(level)


@router.get("/status", response_model=dict[str, Any], status_code=status.HTTP_200_OK)
async def get_service_status(safety_service: SafetyService = Depends(get_safety_service)) -> dict[str, Any]:
    """Get safety service status and statistics."""
    return await safety_service.get_status()


def _get_crisis_resources(level: CrisisLevel) -> list[dict[str, str]]:
    """Get crisis resources based on crisis level."""
    resources = [
        {"name": "988 Suicide & Crisis Lifeline", "contact": "988", "type": "phone", "available": "24/7"},
        {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "type": "text", "available": "24/7"},
    ]
    if level == CrisisLevel.CRITICAL:
        resources.insert(0, {"name": "Emergency Services", "contact": "911", "type": "phone", "available": "24/7"})
        resources.append({"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "type": "web", "available": "24/7"})
    return resources

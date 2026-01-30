"""
Solace-AI Safety Service - Main safety orchestration service.
Coordinates crisis detection, escalation, and safety monitoring across all layers.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from .crisis_detector import CrisisDetector, CrisisDetectorSettings, DetectionResult, CrisisLevel, RiskFactor
from .escalation import EscalationManager, EscalationSettings, EscalationResult
from services.shared import ServiceBase

logger = structlog.get_logger(__name__)


class SafetyServiceSettings(BaseSettings):
    """Configuration for safety service behavior."""
    enable_pre_check: bool = Field(default=True, description="Enable input pre-checking")
    enable_post_check: bool = Field(default=True, description="Enable output post-checking")
    enable_continuous_monitoring: bool = Field(default=True, description="Enable trajectory monitoring")
    auto_escalate_high: bool = Field(default=True, description="Auto-escalate HIGH risk")
    auto_escalate_critical: bool = Field(default=True, description="Auto-escalate CRITICAL risk")
    cache_assessments: bool = Field(default=True, description="Cache recent assessments")
    assessment_cache_ttl_seconds: int = Field(default=300, description="Assessment cache TTL")
    max_history_messages: int = Field(default=20, description="Max conversation history")
    safe_response_threshold: Decimal = Field(default=Decimal("0.3"), description="Threshold for safe output")
    model_config = SettingsConfigDict(env_prefix="SAFETY_SERVICE_", env_file=".env", extra="ignore")


class SafetyCheckResult(BaseModel):
    """Result from safety check operation."""
    check_id: UUID = Field(default_factory=uuid4, description="Check identifier")
    is_safe: bool = Field(default=True, description="Whether content passed safety")
    crisis_level: CrisisLevel = Field(default=CrisisLevel.NONE, description="Crisis level")
    risk_score: Decimal = Field(default=Decimal("0.0"), ge=0, le=1, description="Risk score")
    risk_factors: list[RiskFactor] = Field(default_factory=list, description="Risk factors")
    protective_factors: list[dict[str, Any]] = Field(default_factory=list, description="Protective factors")
    recommended_action: str = Field(default="continue", description="Recommended action")
    requires_escalation: bool = Field(default=False, description="Needs escalation")
    requires_human_review: bool = Field(default=False, description="Needs human review")
    detection_time_ms: int = Field(default=0, ge=0, description="Detection time")
    detection_layer: int = Field(default=1, ge=1, le=4, description="Detection layer")


class CrisisDetectionResult(BaseModel):
    """Result from direct crisis detection."""
    detection_id: UUID = Field(default_factory=uuid4, description="Detection identifier")
    crisis_detected: bool = Field(default=False, description="Crisis detected")
    crisis_level: CrisisLevel = Field(default=CrisisLevel.NONE, description="Crisis level")
    trigger_indicators: list[str] = Field(default_factory=list, description="Triggers")
    confidence: Decimal = Field(default=Decimal("0.0"), ge=0, le=1, description="Confidence")
    detection_layers_triggered: list[int] = Field(default_factory=list, description="Layers")
    detection_time_ms: int = Field(default=0, ge=0, description="Detection time")


class SafetyAssessmentResult(BaseModel):
    """Result from comprehensive safety assessment."""
    assessment_id: UUID = Field(default_factory=uuid4, description="Assessment identifier")
    overall_risk_level: CrisisLevel = Field(default=CrisisLevel.NONE, description="Overall risk")
    overall_risk_score: Decimal = Field(default=Decimal("0.0"), ge=0, le=1, description="Risk score")
    message_assessments: list[dict[str, Any]] = Field(default_factory=list, description="Per-message")
    trajectory_analysis: dict[str, Any] | None = Field(default=None, description="Trajectory")
    risk_prediction: dict[str, Any] | None = Field(default=None, description="Prediction")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    requires_intervention: bool = Field(default=False, description="Needs intervention")


class OutputFilterResult(BaseModel):
    """Result from output filtering."""
    filter_id: UUID = Field(default_factory=uuid4, description="Filter identifier")
    filtered_response: str = Field(..., description="Filtered response")
    modifications_made: list[str] = Field(default_factory=list, description="Modifications")
    resources_appended: bool = Field(default=False, description="Resources added")
    is_safe: bool = Field(default=True, description="Output is safe")
    filter_time_ms: int = Field(default=0, ge=0, description="Filter time")


@dataclass
class AssessmentCache:
    """Cache for recent safety assessments."""
    user_id: UUID
    assessment: SafetyCheckResult
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyService(ServiceBase):
    """Main safety service orchestrating all safety operations."""

    def __init__(self, settings: SafetyServiceSettings | None = None,
                 crisis_detector: CrisisDetector | None = None,
                 escalation_manager: EscalationManager | None = None) -> None:
        self._settings = settings or SafetyServiceSettings()
        self._crisis_detector = crisis_detector or CrisisDetector(CrisisDetectorSettings())
        self._escalation_manager = escalation_manager or EscalationManager(EscalationSettings())
        self._assessment_cache: dict[UUID, AssessmentCache] = {}
        self._user_risk_history: dict[UUID, dict[str, Any]] = {}
        self._conversation_history: dict[UUID, list[str]] = {}
        self._initialized = False
        self._stats = {"total_checks": 0, "crises_detected": 0, "escalations_triggered": 0,
            "pre_checks": 0, "post_checks": 0, "assessments": 0}

    async def initialize(self) -> None:
        """Initialize the safety service."""
        logger.info("safety_service_initializing")
        self._initialized = True
        logger.info("safety_service_initialized", settings={
            "pre_check": self._settings.enable_pre_check,
            "post_check": self._settings.enable_post_check,
            "auto_escalate_high": self._settings.auto_escalate_high,
            "auto_escalate_critical": self._settings.auto_escalate_critical,
        })

    async def shutdown(self) -> None:
        """Shutdown the safety service."""
        logger.info("safety_service_shutting_down", stats=self._stats)
        self._initialized = False
        self._assessment_cache.clear()

    async def check_safety(self, user_id: UUID, session_id: UUID | None, content: str,
                           check_type: str, context: dict[str, Any] | None = None) -> SafetyCheckResult:
        """Perform safety check on content."""
        start_time = time.perf_counter()
        self._stats["total_checks"] += 1
        if check_type == "pre_check":
            self._stats["pre_checks"] += 1
        elif check_type == "post_check":
            self._stats["post_checks"] += 1
        self._update_conversation_history(user_id, content)
        user_risk_history = self._user_risk_history.get(user_id, {})
        conversation_history = self._conversation_history.get(user_id, [])
        detection_result = await self._crisis_detector.detect(
            content=content,
            context=context or {},
            conversation_history=conversation_history,
            user_risk_history=user_risk_history,
        )
        is_safe = detection_result.crisis_level in (CrisisLevel.NONE, CrisisLevel.LOW)
        requires_escalation = detection_result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        requires_human_review = detection_result.crisis_level == CrisisLevel.CRITICAL
        if detection_result.crisis_level != CrisisLevel.NONE:
            self._stats["crises_detected"] += 1
            self._update_user_risk_history(user_id, detection_result)
        protective_factors = self._identify_protective_factors(content, context or {})
        recommended_action = self._determine_action(detection_result.crisis_level, check_type)
        if requires_escalation and self._should_auto_escalate(detection_result.crisis_level):
            await self._trigger_auto_escalation(user_id, session_id, detection_result)
        detection_time_ms = int((time.perf_counter() - start_time) * 1000)
        result = SafetyCheckResult(
            is_safe=is_safe,
            crisis_level=detection_result.crisis_level,
            risk_score=detection_result.risk_score,
            risk_factors=detection_result.risk_factors,
            protective_factors=protective_factors,
            recommended_action=recommended_action,
            requires_escalation=requires_escalation,
            requires_human_review=requires_human_review,
            detection_time_ms=detection_time_ms,
            detection_layer=max(detection_result.detection_layers_triggered) if detection_result.detection_layers_triggered else 1,
        )
        self._cache_assessment(user_id, result)
        logger.info("safety_check_completed", user_id=str(user_id), check_type=check_type,
            crisis_level=detection_result.crisis_level.value, is_safe=is_safe, time_ms=detection_time_ms)
        return result

    async def detect_crisis(self, user_id: UUID, content: str, conversation_history: list[str] | None = None,
                            user_risk_history: dict[str, Any] | None = None) -> CrisisDetectionResult:
        """Perform direct crisis detection."""
        start_time = time.perf_counter()
        detection_result = await self._crisis_detector.detect(
            content=content,
            conversation_history=conversation_history or [],
            user_risk_history=user_risk_history or {},
        )
        detection_time_ms = int((time.perf_counter() - start_time) * 1000)
        return CrisisDetectionResult(
            crisis_detected=detection_result.crisis_detected,
            crisis_level=detection_result.crisis_level,
            trigger_indicators=detection_result.trigger_indicators,
            confidence=detection_result.confidence,
            detection_layers_triggered=detection_result.detection_layers_triggered,
            detection_time_ms=detection_time_ms,
        )

    async def escalate(self, user_id: UUID, session_id: UUID | None, crisis_level: str,
                       reason: str, context: dict[str, Any] | None = None,
                       priority_override: str | None = None) -> EscalationResult:
        """Trigger escalation to human clinician."""
        self._stats["escalations_triggered"] += 1
        result = await self._escalation_manager.escalate(
            user_id=user_id,
            session_id=session_id,
            crisis_level=crisis_level,
            reason=reason,
            context=context,
            priority_override=priority_override,
        )
        logger.info("escalation_triggered", user_id=str(user_id),
            escalation_id=str(result.escalation_id), priority=result.priority)
        return result

    async def assess_safety(self, user_id: UUID, session_id: UUID | None, messages: list[str],
                            include_trajectory: bool = True, include_prediction: bool = True) -> SafetyAssessmentResult:
        """Perform comprehensive safety assessment."""
        self._stats["assessments"] += 1
        message_assessments: list[dict[str, Any]] = []
        max_risk_level = CrisisLevel.NONE
        max_risk_score = Decimal("0.0")
        for i, message in enumerate(messages):
            detection = await self._crisis_detector.detect(content=message)
            message_assessments.append({
                "message_index": i,
                "crisis_level": detection.crisis_level.value,
                "risk_score": float(detection.risk_score),
                "risk_factors": [rf.model_dump() for rf in detection.risk_factors],
            })
            if detection.risk_score > max_risk_score:
                max_risk_score = detection.risk_score
                max_risk_level = detection.crisis_level
        trajectory_analysis = None
        if include_trajectory and len(messages) > 1:
            trajectory_analysis = self._crisis_detector.analyze_trajectory(messages, max_risk_level)
        risk_prediction = None
        if include_prediction:
            risk_prediction = self._predict_risk_trajectory(messages, max_risk_level)
        recommendations = self._generate_recommendations(max_risk_level, trajectory_analysis)
        requires_intervention = max_risk_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        return SafetyAssessmentResult(
            overall_risk_level=max_risk_level,
            overall_risk_score=max_risk_score,
            message_assessments=message_assessments,
            trajectory_analysis=trajectory_analysis,
            risk_prediction=risk_prediction,
            recommendations=recommendations,
            requires_intervention=requires_intervention,
        )

    async def filter_output(self, user_id: UUID, original_response: str,
                            user_crisis_level: str, include_resources: bool) -> OutputFilterResult:
        """Filter AI output for safety."""
        start_time = time.perf_counter()
        crisis_level = CrisisLevel(user_crisis_level)
        filtered_response, modifications, is_safe = self._crisis_detector.filter_output(original_response, crisis_level)
        resources_appended = False
        if include_resources and crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            resources = self._escalation_manager.get_crisis_resources(crisis_level.value)
            if resources:
                resource_text = "\n\n---\nCrisis Resources:\n"
                for r in resources[:3]:
                    resource_text += f"- {r['name']}: {r['contact']} ({r['available']})\n"
                filtered_response += resource_text
                resources_appended = True
                modifications.append("Crisis resources appended")
        filter_time_ms = int((time.perf_counter() - start_time) * 1000)
        return OutputFilterResult(
            filtered_response=filtered_response,
            modifications_made=modifications,
            resources_appended=resources_appended,
            is_safe=is_safe,
            filter_time_ms=filter_time_ms,
        )

    async def get_status(self) -> dict[str, Any]:
        """Get service status and statistics."""
        escalation_stats = self._escalation_manager.get_statistics()
        return {
            "status": "operational" if self._initialized else "initializing",
            "initialized": self._initialized,
            "statistics": self._stats,
            "escalation_statistics": escalation_stats,
            "active_users_monitored": len(self._conversation_history),
            "cached_assessments": len(self._assessment_cache),
            "settings": {
                "pre_check_enabled": self._settings.enable_pre_check,
                "post_check_enabled": self._settings.enable_post_check,
                "continuous_monitoring": self._settings.enable_continuous_monitoring,
            },
        }

    @property
    def stats(self) -> dict[str, int]:
        """Get service statistics counters."""
        return self._stats

    def _update_conversation_history(self, user_id: UUID, content: str) -> None:
        """Update conversation history for user."""
        if user_id not in self._conversation_history:
            self._conversation_history[user_id] = []
        self._conversation_history[user_id].append(content)
        if len(self._conversation_history[user_id]) > self._settings.max_history_messages:
            self._conversation_history[user_id] = self._conversation_history[user_id][-self._settings.max_history_messages:]

    def _update_user_risk_history(self, user_id: UUID, detection: DetectionResult) -> None:
        """Update user's risk history."""
        if user_id not in self._user_risk_history:
            self._user_risk_history[user_id] = {"previous_crisis_events": 0, "recent_escalation": False}
        if detection.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            self._user_risk_history[user_id]["previous_crisis_events"] += 1
            self._user_risk_history[user_id]["recent_escalation"] = True
            self._user_risk_history[user_id]["last_crisis_level"] = detection.crisis_level.value
            self._user_risk_history[user_id]["last_crisis_time"] = datetime.now(timezone.utc).isoformat()

    def _identify_protective_factors(self, content: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify protective factors in content."""
        protective_factors: list[dict[str, Any]] = []
        protective_keywords = {"support": ("social_support", Decimal("0.6")),
            "family": ("family_connection", Decimal("0.7")), "friend": ("social_connection", Decimal("0.6")),
            "hope": ("positive_outlook", Decimal("0.5")), "therapy": ("treatment_engagement", Decimal("0.7")),
            "medication": ("treatment_adherence", Decimal("0.6")), "coping": ("coping_skills", Decimal("0.5"))}
        content_lower = content.lower()
        for keyword, (factor_type, strength) in protective_keywords.items():
            if keyword in content_lower:
                protective_factors.append({"factor_type": factor_type, "strength": float(strength),
                    "description": f"Evidence of {factor_type.replace('_', ' ')}"})
        if context.get("has_treatment_plan"):
            protective_factors.append({"factor_type": "active_treatment", "strength": 0.8,
                "description": "Active treatment plan in place"})
        if context.get("regular_appointments"):
            protective_factors.append({"factor_type": "care_continuity", "strength": 0.7,
                "description": "Regular care appointments"})
        return protective_factors

    def _determine_action(self, crisis_level: CrisisLevel, check_type: str) -> str:
        """Determine recommended action based on crisis level and check type."""
        action_map = {
            CrisisLevel.NONE: "continue",
            CrisisLevel.LOW: "monitor" if check_type == "pre_check" else "continue",
            CrisisLevel.ELEVATED: "assess" if check_type == "pre_check" else "review",
            CrisisLevel.HIGH: "intervene",
            CrisisLevel.CRITICAL: "escalate_immediately",
        }
        return action_map.get(crisis_level, "continue")

    def _should_auto_escalate(self, crisis_level: CrisisLevel) -> bool:
        """Determine if auto-escalation should trigger."""
        if crisis_level == CrisisLevel.CRITICAL and self._settings.auto_escalate_critical:
            return True
        if crisis_level == CrisisLevel.HIGH and self._settings.auto_escalate_high:
            return True
        return False

    async def _trigger_auto_escalation(self, user_id: UUID, session_id: UUID | None,
                                       detection: DetectionResult) -> None:
        """Trigger automatic escalation."""
        reason = f"Auto-escalation triggered: {detection.crisis_level.value} risk detected"
        if detection.trigger_indicators:
            reason += f" - Triggers: {', '.join(detection.trigger_indicators[:3])}"
        await self.escalate(user_id, session_id, detection.crisis_level.value, reason)

    def _cache_assessment(self, user_id: UUID, result: SafetyCheckResult) -> None:
        """Cache assessment result."""
        if self._settings.cache_assessments:
            self._assessment_cache[user_id] = AssessmentCache(user_id=user_id, assessment=result)

    def _predict_risk_trajectory(self, messages: list[str], current_level: CrisisLevel) -> dict[str, Any]:
        """Predict risk trajectory based on message patterns."""
        if len(messages) < 3:
            return {"prediction": "insufficient_data", "confidence": 0.0}
        trajectory = self._crisis_detector.analyze_trajectory(messages, current_level)
        if trajectory.get("deteriorating"):
            return {"prediction": "increasing_risk", "confidence": 0.7, "trend": "negative"}
        if trajectory.get("trend") == "stable":
            return {"prediction": "stable", "confidence": 0.6, "trend": "neutral"}
        return {"prediction": "improving", "confidence": 0.5, "trend": "positive"}

    def _generate_recommendations(self, risk_level: CrisisLevel,
                                   trajectory: dict[str, Any] | None) -> list[str]:
        """Generate safety recommendations."""
        recommendations: list[str] = []
        if risk_level == CrisisLevel.CRITICAL:
            recommendations.extend(["Immediate professional intervention required",
                "Ensure user has access to crisis resources", "Maintain continuous monitoring"])
        elif risk_level == CrisisLevel.HIGH:
            recommendations.extend(["Schedule urgent clinical review",
                "Activate safety planning dialogue", "Provide crisis resources"])
        elif risk_level == CrisisLevel.ELEVATED:
            recommendations.extend(["Increase monitoring frequency",
                "Review coping strategies with user", "Check in about support systems"])
        elif risk_level == CrisisLevel.LOW:
            recommendations.append("Continue standard therapeutic engagement")
        if trajectory and trajectory.get("deteriorating"):
            recommendations.append("Consider preventive escalation due to deteriorating trajectory")
        return recommendations

"""Solace-AI Event Schemas - Pydantic models for all domain events."""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field
import structlog

logger = structlog.get_logger(__name__)


class EventMetadata(BaseModel):
    """Metadata common to all events."""
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: UUID = Field(default_factory=uuid4)
    causation_id: UUID | None = Field(default=None)
    version: int = Field(default=1, ge=1)
    source_service: str = Field(default="solace-ai")
    model_config = ConfigDict(frozen=True)


class BaseEvent(BaseModel):
    """Base class for all Solace-AI events."""
    event_type: str
    user_id: UUID
    session_id: UUID | None = Field(default=None)
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    model_config = ConfigDict(frozen=True, use_enum_values=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary for Kafka."""
        data = self.model_dump(mode="json")
        data["metadata"]["timestamp"] = self.metadata.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseEvent:
        """Deserialize event from dictionary."""
        return cls.model_validate(data)

    def with_correlation(self, correlation_id: UUID, causation_id: UUID | None = None) -> BaseEvent:
        """Create copy with correlation context."""
        new_metadata = EventMetadata(
            event_id=self.metadata.event_id, timestamp=self.metadata.timestamp,
            correlation_id=correlation_id, causation_id=causation_id or self.metadata.event_id,
            version=self.metadata.version, source_service=self.metadata.source_service,
        )
        return self.model_copy(update={"metadata": new_metadata})


# Session Events
class SessionStartedEvent(BaseEvent):
    """Emitted when a therapy session begins."""
    event_type: Literal["session.started"] = "session.started"
    session_number: int = Field(ge=1)
    channel: str = Field(default="web")
    client_info: dict[str, str] = Field(default_factory=dict)


class SessionEndedEvent(BaseEvent):
    """Emitted when a therapy session ends."""
    event_type: Literal["session.ended"] = "session.ended"
    duration_seconds: int = Field(ge=0)
    message_count: int = Field(ge=0)
    end_reason: Literal["user_initiated", "timeout", "error", "system"] = "user_initiated"


class MessageReceivedEvent(BaseEvent):
    """Emitted when user sends a message."""
    event_type: Literal["session.message.received"] = "session.message.received"
    message_id: UUID
    content_length: int = Field(ge=0)
    content_hash: str
    detected_language: str = Field(default="en")


class ResponseGeneratedEvent(BaseEvent):
    """Emitted when AI generates a response."""
    event_type: Literal["session.response.generated"] = "session.response.generated"
    response_id: UUID
    response_length: int = Field(ge=0)
    generation_time_ms: int = Field(ge=0)
    model_used: str
    tokens_used: int = Field(ge=0)


# Safety Events
class CrisisLevel(str, Enum):
    """Crisis severity levels."""
    NONE, LOW, ELEVATED, HIGH, CRITICAL = "NONE", "LOW", "ELEVATED", "HIGH", "CRITICAL"


class RiskFactor(BaseModel):
    """Individual risk factor identified."""
    factor_type: str
    severity: Decimal = Field(ge=0, le=1)
    evidence: str
    confidence: Decimal = Field(ge=0, le=1)
    model_config = ConfigDict(frozen=True)


class SafetyAssessmentEvent(BaseEvent):
    """Emitted after safety assessment completes."""
    event_type: Literal["safety.assessment.completed"] = "safety.assessment.completed"
    risk_level: CrisisLevel
    risk_score: Decimal = Field(ge=0, le=1)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    protective_factors: list[str] = Field(default_factory=list)
    detection_layer: int = Field(ge=1, le=4)
    recommended_action: str


class CrisisDetectedEvent(BaseEvent):
    """Emitted when crisis is detected - HIGH PRIORITY."""
    event_type: Literal["safety.crisis.detected"] = "safety.crisis.detected"
    crisis_level: CrisisLevel
    trigger_indicators: list[str]
    detection_layer: int = Field(ge=1, le=4)
    confidence: Decimal = Field(ge=0, le=1)
    escalation_action: str
    requires_human_review: bool = Field(default=True)


class EscalationTriggeredEvent(BaseEvent):
    """Emitted when case is escalated to human clinician."""
    event_type: Literal["safety.escalation.triggered"] = "safety.escalation.triggered"
    escalation_reason: str
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assigned_clinician_id: UUID | None = Field(default=None)
    notification_sent: bool = Field(default=False)


# Diagnosis Events
class Confidence(str, Enum):
    """Clinical confidence levels."""
    LOW, MODERATE, HIGH, VERY_HIGH = "LOW", "MODERATE", "HIGH", "VERY_HIGH"


class ClinicalHypothesis(BaseModel):
    """Clinical hypothesis from diagnosis."""
    condition_code: str
    condition_name: str
    confidence: Confidence
    evidence_summary: str
    severity: Literal["MILD", "MODERATE", "SEVERE"]
    model_config = ConfigDict(frozen=True)


class DiagnosisCompletedEvent(BaseEvent):
    """Emitted when diagnostic assessment completes."""
    event_type: Literal["diagnosis.completed"] = "diagnosis.completed"
    assessment_id: UUID
    primary_hypothesis: ClinicalHypothesis | None = Field(default=None)
    differential_hypotheses: list[ClinicalHypothesis] = Field(default_factory=list)
    stepped_care_level: int = Field(ge=0, le=4)
    assessment_instruments_used: list[str] = Field(default_factory=list)
    requires_clinical_review: bool = Field(default=False)


class AssessmentCompletedEvent(BaseEvent):
    """Emitted when structured assessment (PHQ-9, GAD-7) completes."""
    event_type: Literal["diagnosis.assessment.completed"] = "diagnosis.assessment.completed"
    instrument_code: str
    total_score: int = Field(ge=0)
    severity_category: str
    subscale_scores: dict[str, int] = Field(default_factory=dict)
    clinical_flags: list[str] = Field(default_factory=list)


# Therapy Events
class TherapyModality(str, Enum):
    """Supported therapy modalities."""
    CBT, DBT, ACT, MI = "CBT", "DBT", "ACT", "MI"
    MINDFULNESS, PSYCHOEDUCATION = "MINDFULNESS", "PSYCHOEDUCATION"


class TherapySessionStartedEvent(BaseEvent):
    """Emitted when therapy session begins."""
    event_type: Literal["therapy.session.started"] = "therapy.session.started"
    treatment_plan_id: UUID | None = Field(default=None)
    session_number: int = Field(ge=1)
    planned_focus: list[str] = Field(default_factory=list)
    active_modalities: list[TherapyModality] = Field(default_factory=list)


class InterventionDeliveredEvent(BaseEvent):
    """Emitted when therapeutic intervention is delivered."""
    event_type: Literal["therapy.intervention.delivered"] = "therapy.intervention.delivered"
    intervention_id: UUID
    technique: str
    modality: TherapyModality
    selection_rationale: dict[str, Decimal] = Field(default_factory=dict)
    user_engagement_score: Decimal | None = Field(default=None, ge=0, le=1)


class ProgressMilestoneEvent(BaseEvent):
    """Emitted when therapy milestone is reached."""
    event_type: Literal["therapy.progress.milestone"] = "therapy.progress.milestone"
    milestone_type: str
    milestone_description: str
    sessions_to_milestone: int = Field(ge=0)
    improvement_metrics: dict[str, Decimal] = Field(default_factory=dict)


# Memory Events
class MemoryTier(str, Enum):
    """Memory storage tiers."""
    INPUT, WORKING, SESSION = "INPUT", "WORKING", "SESSION"
    EPISODIC, SEMANTIC = "EPISODIC", "SEMANTIC"


class RetentionCategory(str, Enum):
    """Memory retention categories."""
    PERMANENT, LONG_TERM, MEDIUM_TERM, SHORT_TERM = "PERMANENT", "LONG_TERM", "MEDIUM_TERM", "SHORT_TERM"


class MemoryStoredEvent(BaseEvent):
    """Emitted when memory is stored."""
    event_type: Literal["memory.stored"] = "memory.stored"
    memory_id: UUID
    memory_tier: MemoryTier
    content_type: str
    retention_category: RetentionCategory
    embedding_generated: bool = Field(default=False)
    ttl_hours: int | None = Field(default=None, ge=0)


class MemoryConsolidatedEvent(BaseEvent):
    """Emitted when session memories are consolidated."""
    event_type: Literal["memory.consolidated"] = "memory.consolidated"
    consolidation_id: UUID
    session_ids: list[UUID] = Field(default_factory=list)
    facts_extracted: int = Field(ge=0)
    embeddings_created: int = Field(ge=0)
    summary_generated: bool = Field(default=False)


class MemoryRetrievedEvent(BaseEvent):
    """Emitted when memories are retrieved for context."""
    event_type: Literal["memory.retrieved"] = "memory.retrieved"
    query_embedding_id: UUID | None = Field(default=None)
    memories_retrieved: int = Field(ge=0)
    retrieval_time_ms: int = Field(ge=0)
    relevance_threshold: Decimal = Field(ge=0, le=1)


# Personality Events
class OceanScores(BaseModel):
    """Big Five personality scores."""
    openness: Decimal = Field(ge=0, le=1)
    conscientiousness: Decimal = Field(ge=0, le=1)
    extraversion: Decimal = Field(ge=0, le=1)
    agreeableness: Decimal = Field(ge=0, le=1)
    neuroticism: Decimal = Field(ge=0, le=1)
    model_config = ConfigDict(frozen=True)


class PersonalityAssessedEvent(BaseEvent):
    """Emitted when personality assessment completes."""
    event_type: Literal["personality.assessed"] = "personality.assessed"
    assessment_id: UUID
    ocean_scores: OceanScores
    assessment_source: Literal["ROBERTA", "LLM", "LIWC", "ENSEMBLE"]
    confidence: Decimal = Field(ge=0, le=1)
    sample_size: int = Field(ge=1)


class StyleGeneratedEvent(BaseEvent):
    """Emitted when communication style is generated."""
    event_type: Literal["personality.style.generated"] = "personality.style.generated"
    style_id: UUID
    target_module: str
    formality_level: Decimal = Field(ge=0, le=1)
    warmth_level: Decimal = Field(ge=0, le=1)
    directness_level: Decimal = Field(ge=0, le=1)
    vocabulary_complexity: Decimal = Field(ge=0, le=1)


# System Events
class SystemHealthEvent(BaseEvent):
    """Emitted for system health monitoring."""
    event_type: Literal["system.health"] = "system.health"
    service_name: str
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: int = Field(ge=0)
    error_rate: Decimal = Field(ge=0, le=1)
    active_connections: int = Field(ge=0)


class ErrorOccurredEvent(BaseEvent):
    """Emitted when significant error occurs."""
    event_type: Literal["system.error"] = "system.error"
    error_code: str
    error_category: Literal["domain", "application", "infrastructure"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    message: str
    stack_trace: str | None = Field(default=None)


# Event Registry for deserialization
EVENT_REGISTRY: dict[str, type[BaseEvent]] = {
    "session.started": SessionStartedEvent, "session.ended": SessionEndedEvent,
    "session.message.received": MessageReceivedEvent, "session.response.generated": ResponseGeneratedEvent,
    "safety.assessment.completed": SafetyAssessmentEvent, "safety.crisis.detected": CrisisDetectedEvent,
    "safety.escalation.triggered": EscalationTriggeredEvent, "diagnosis.completed": DiagnosisCompletedEvent,
    "diagnosis.assessment.completed": AssessmentCompletedEvent, "therapy.session.started": TherapySessionStartedEvent,
    "therapy.intervention.delivered": InterventionDeliveredEvent, "therapy.progress.milestone": ProgressMilestoneEvent,
    "memory.stored": MemoryStoredEvent, "memory.consolidated": MemoryConsolidatedEvent,
    "memory.retrieved": MemoryRetrievedEvent, "personality.assessed": PersonalityAssessedEvent,
    "personality.style.generated": StyleGeneratedEvent, "system.health": SystemHealthEvent,
    "system.error": ErrorOccurredEvent,
}

_TOPIC_MAP = {
    "session.": "solace.sessions", "safety.": "solace.safety", "diagnosis.": "solace.assessments",
    "therapy.": "solace.therapy", "memory.": "solace.memory", "personality.": "solace.personality",
    "system.": "solace.analytics",
}


def deserialize_event(data: dict[str, Any]) -> BaseEvent:
    """Deserialize event from dictionary using registry."""
    event_type = data.get("event_type")
    if not event_type:
        raise ValueError("Missing event_type in event data")
    event_class = EVENT_REGISTRY.get(event_type)
    if not event_class:
        logger.warning("Unknown event type, using base", event_type=event_type)
        return BaseEvent.model_validate(data)
    return event_class.model_validate(data)


def get_topic_for_event(event: BaseEvent) -> str:
    """Determine Kafka topic for event based on type."""
    for prefix, topic in _TOPIC_MAP.items():
        if event.event_type.startswith(prefix):
            return topic
    return "solace.analytics"

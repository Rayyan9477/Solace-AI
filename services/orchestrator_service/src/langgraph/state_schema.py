"""
Solace-AI Orchestrator Service - LangGraph State Schema.
Typed state definitions with reducers for multi-agent orchestration checkpointing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4
import operator
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Safety risk levels for crisis detection. Aligned with canonical CrisisLevel."""
    NONE = "NONE"
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IntentType(str, Enum):
    """User intent classification types."""
    GENERAL_CHAT = "general_chat"
    EMOTIONAL_SUPPORT = "emotional_support"
    CRISIS_DISCLOSURE = "crisis_disclosure"
    SYMPTOM_DISCUSSION = "symptom_discussion"
    TREATMENT_INQUIRY = "treatment_inquiry"
    PROGRESS_UPDATE = "progress_update"
    ASSESSMENT_REQUEST = "assessment_request"
    COPING_STRATEGY = "coping_strategy"
    PSYCHOEDUCATION = "psychoeducation"
    SESSION_MANAGEMENT = "session_management"


class AgentType(str, Enum):
    """Available agent types in the orchestration system."""
    SAFETY = "safety"
    SUPERVISOR = "supervisor"
    DIAGNOSIS = "diagnosis"
    THERAPY = "therapy"
    PERSONALITY = "personality"
    CHAT = "chat"
    MEMORY = "memory"
    AGGREGATOR = "aggregator"


class ProcessingPhase(str, Enum):
    """Current phase in the orchestration pipeline."""
    INITIALIZED = "initialized"
    SAFETY_PRECHECK = "safety_precheck"
    CONTEXT_LOADING = "context_loading"
    INTENT_CLASSIFICATION = "intent_classification"
    AGENT_ROUTING = "agent_routing"
    PARALLEL_PROCESSING = "parallel_processing"
    AGGREGATION = "aggregation"
    SAFETY_POSTCHECK = "safety_postcheck"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"
    CRISIS_HANDLING = "crisis_handling"
    ERROR = "error"


@dataclass(frozen=True)
class MessageEntry:
    """Immutable message entry in conversation history."""
    message_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": str(self.message_id), "role": self.role, "content": self.content, "timestamp": self.timestamp.isoformat(), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageEntry:
        ts = datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp") or datetime.now(timezone.utc)
        return cls(message_id=UUID(data["message_id"]) if isinstance(data.get("message_id"), str) else data.get("message_id", uuid4()), role=data["role"], content=data["content"], timestamp=ts, metadata=data.get("metadata", {}))

    @classmethod
    def user_message(cls, content: str, metadata: dict[str, Any] | None = None) -> MessageEntry:
        return cls(message_id=uuid4(), role="user", content=content, timestamp=datetime.now(timezone.utc), metadata=metadata or {})

    @classmethod
    def assistant_message(cls, content: str, metadata: dict[str, Any] | None = None) -> MessageEntry:
        return cls(message_id=uuid4(), role="assistant", content=content, timestamp=datetime.now(timezone.utc), metadata=metadata or {})


@dataclass
class SafetyFlags:
    """Safety monitoring flags and risk assessment."""
    risk_level: RiskLevel = RiskLevel.NONE
    crisis_detected: bool = False
    crisis_type: str | None = None
    requires_escalation: bool = False
    escalation_reason: str | None = None
    safety_resources_shown: bool = False
    monitoring_level: Literal["standard", "enhanced", "intensive"] = "standard"
    contraindications: list[str] = field(default_factory=list)
    triggered_keywords: list[str] = field(default_factory=list)
    last_assessment_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"risk_level": self.risk_level.value, "crisis_detected": self.crisis_detected, "crisis_type": self.crisis_type, "requires_escalation": self.requires_escalation, "escalation_reason": self.escalation_reason, "safety_resources_shown": self.safety_resources_shown, "monitoring_level": self.monitoring_level, "contraindications": self.contraindications, "triggered_keywords": self.triggered_keywords, "last_assessment_at": self.last_assessment_at.isoformat() if self.last_assessment_at else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyFlags:
        la = datetime.fromisoformat(data["last_assessment_at"]) if isinstance(data.get("last_assessment_at"), str) else data.get("last_assessment_at")
        return cls(risk_level=RiskLevel(data.get("risk_level", "NONE")), crisis_detected=data.get("crisis_detected", False), crisis_type=data.get("crisis_type"), requires_escalation=data.get("requires_escalation", False), escalation_reason=data.get("escalation_reason"), safety_resources_shown=data.get("safety_resources_shown", False), monitoring_level=data.get("monitoring_level", "standard"), contraindications=data.get("contraindications", []), triggered_keywords=data.get("triggered_keywords", []), last_assessment_at=la)

    @classmethod
    def safe(cls) -> SafetyFlags:
        return cls(risk_level=RiskLevel.NONE, crisis_detected=False)

    def is_safe(self) -> bool:
        return not self.crisis_detected and self.risk_level in (RiskLevel.NONE, RiskLevel.LOW)


@dataclass
class AgentResult:
    """Result from an agent's processing."""
    agent_type: AgentType
    success: bool
    response_content: str | None = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {"agent_type": self.agent_type.value, "success": self.success, "response_content": self.response_content, "confidence": self.confidence, "processing_time_ms": self.processing_time_ms, "metadata": self.metadata, "error": self.error, "timestamp": self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentResult:
        ts = datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp") or datetime.now(timezone.utc)
        return cls(agent_type=AgentType(data["agent_type"]), success=data.get("success", False), response_content=data.get("response_content"), confidence=data.get("confidence", 0.0), processing_time_ms=data.get("processing_time_ms", 0.0), metadata=data.get("metadata", {}), error=data.get("error"), timestamp=ts)


@dataclass
class ProcessingMetadata:
    """Metadata about the current processing request."""
    request_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    user_id: UUID | None = None
    correlation_id: UUID | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_processing_time_ms: float = 0.0
    active_agents: list[AgentType] = field(default_factory=list)
    completed_agents: list[AgentType] = field(default_factory=list)
    retry_count: int = 0
    is_streaming: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": str(self.request_id), "session_id": str(self.session_id) if self.session_id else None, "user_id": str(self.user_id) if self.user_id else None, "correlation_id": str(self.correlation_id) if self.correlation_id else None, "start_time": self.start_time.isoformat(), "total_processing_time_ms": self.total_processing_time_ms, "active_agents": [a.value for a in self.active_agents], "completed_agents": [a.value for a in self.completed_agents], "retry_count": self.retry_count, "is_streaming": self.is_streaming}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingMetadata:
        st = datetime.fromisoformat(data["start_time"]) if isinstance(data.get("start_time"), str) else data.get("start_time") or datetime.now(timezone.utc)
        return cls(request_id=UUID(data["request_id"]) if isinstance(data.get("request_id"), str) else data.get("request_id", uuid4()), session_id=UUID(data["session_id"]) if data.get("session_id") else None, user_id=UUID(data["user_id"]) if data.get("user_id") else None, correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None, start_time=st, total_processing_time_ms=data.get("total_processing_time_ms", 0.0), active_agents=[AgentType(a) for a in data.get("active_agents", [])], completed_agents=[AgentType(a) for a in data.get("completed_agents", [])], retry_count=data.get("retry_count", 0), is_streaming=data.get("is_streaming", False))


def add_messages(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reducer for message aggregation - appends new messages."""
    return left + right


def add_agent_results(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reducer for agent results - appends new results."""
    return left + right


def update_safety_flags(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer for safety flags - deep merges with priority to higher risk.

    Preserves non-None left values when right provides None.
    Always escalates to the higher risk level.
    """
    if not left:
        return right
    if not right:
        return left

    # Start with left, overlay right only where right provides non-None values
    merged: dict[str, Any] = dict(left)
    for key, value in right.items():
        if value is not None:
            merged[key] = value

    # Risk level: always keep the higher of the two
    risk_order = ["NONE", "LOW", "ELEVATED", "HIGH", "CRITICAL"]
    left_risk = left.get("risk_level", "NONE").upper()
    right_risk = right.get("risk_level", "NONE").upper()
    left_idx = risk_order.index(left_risk) if left_risk in risk_order else 0
    right_idx = risk_order.index(right_risk) if right_risk in risk_order else 0
    merged["risk_level"] = risk_order[max(left_idx, right_idx)]

    # Boolean flags: OR semantics (once detected, stays detected)
    if left.get("crisis_detected") or right.get("crisis_detected"):
        merged["crisis_detected"] = True
    if left.get("requires_escalation") or right.get("requires_escalation"):
        merged["requires_escalation"] = True
    if left.get("safety_resources_shown") or right.get("safety_resources_shown"):
        merged["safety_resources_shown"] = True

    # Lists: union (deduplicated)
    merged["contraindications"] = list(set(
        left.get("contraindications", []) + right.get("contraindications", [])
    ))
    merged["triggered_keywords"] = list(set(
        left.get("triggered_keywords", []) + right.get("triggered_keywords", [])
    ))

    # Monitoring level: keep the more intensive
    monitoring_order = {"standard": 0, "enhanced": 1, "intensive": 2}
    left_mon = left.get("monitoring_level", "standard")
    right_mon = right.get("monitoring_level", "standard")
    if monitoring_order.get(right_mon, 0) > monitoring_order.get(left_mon, 0):
        merged["monitoring_level"] = right_mon
    else:
        merged["monitoring_level"] = left_mon

    return merged


class OrchestratorState(TypedDict, total=False):
    """
    LangGraph state schema for multi-agent orchestration.
    Uses TypedDict with Annotated types for automatic state aggregation.
    """
    user_id: str
    session_id: str
    thread_id: str
    current_message: str
    messages: Annotated[list[dict[str, Any]], add_messages]
    conversation_context: str
    intent: str
    intent_confidence: float
    selected_agents: list[str]
    agent_results: Annotated[list[dict[str, Any]], add_agent_results]
    safety_flags: Annotated[dict[str, Any], update_safety_flags]
    processing_phase: str
    final_response: str
    error_message: str | None
    metadata: dict[str, Any]
    personality_style: dict[str, Any]
    active_treatment: dict[str, Any] | None
    memory_context: dict[str, Any]
    retrieved_memories: list[dict[str, Any]]
    assembled_context: str
    memory_sources: list[str]


def create_initial_state(
    user_id: UUID | str,
    session_id: UUID | str,
    message: str,
    thread_id: str | None = None,
    conversation_context: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> OrchestratorState:
    """
    Create initial orchestrator state for a new request.

    Args:
        user_id: User identifier
        session_id: Session identifier
        message: User's current message
        thread_id: Optional thread ID for checkpointing
        conversation_context: Optional pre-loaded context
        metadata: Optional additional metadata

    Returns:
        Initialized OrchestratorState
    """
    user_msg = MessageEntry.user_message(message)
    return OrchestratorState(
        user_id=str(user_id),
        session_id=str(session_id),
        thread_id=thread_id or str(uuid4()),
        current_message=message,
        messages=[user_msg.to_dict()],
        conversation_context=conversation_context or "",
        intent=IntentType.GENERAL_CHAT.value,
        intent_confidence=0.0,
        selected_agents=[],
        agent_results=[],
        safety_flags=SafetyFlags.safe().to_dict(),
        processing_phase=ProcessingPhase.INITIALIZED.value,
        final_response="",
        error_message=None,
        metadata=metadata or {},
        personality_style={},
        active_treatment=None,
        memory_context={},
        retrieved_memories=[],
        assembled_context="",
        memory_sources=[],
    )


class StateValidator:
    """Validates orchestrator state integrity."""

    @staticmethod
    def validate_state(state: OrchestratorState) -> tuple[bool, list[str]]:
        """
        Validate state integrity.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []
        if not state.get("user_id"):
            errors.append("Missing user_id")
        if not state.get("session_id"):
            errors.append("Missing session_id")
        if not state.get("current_message"):
            errors.append("Missing current_message")
        phase = state.get("processing_phase")
        if phase and phase not in [p.value for p in ProcessingPhase]:
            errors.append(f"Invalid processing_phase: {phase}")
        intent = state.get("intent")
        if intent and intent not in [i.value for i in IntentType]:
            errors.append(f"Invalid intent: {intent}")
        safety = state.get("safety_flags", {})
        risk = safety.get("risk_level")
        if risk and risk not in [r.value for r in RiskLevel]:
            errors.append(f"Invalid risk_level: {risk}")
        return len(errors) == 0, errors

    @staticmethod
    def is_terminal_phase(state: OrchestratorState) -> bool:
        """Check if state is in a terminal phase."""
        phase = state.get("processing_phase", "")
        return phase in (ProcessingPhase.COMPLETED.value, ProcessingPhase.ERROR.value)

    @staticmethod
    def requires_safety_intervention(state: OrchestratorState) -> bool:
        """Check if state requires safety intervention."""
        safety = state.get("safety_flags", {})
        return safety.get("crisis_detected", False) or safety.get("requires_escalation", False)

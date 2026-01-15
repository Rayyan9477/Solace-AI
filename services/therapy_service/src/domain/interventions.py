"""
Solace-AI Therapy Service - Intervention Delivery Framework.
Orchestrates therapeutic interventions with timing, sequencing, and safety protocols.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import TherapyModality, SessionPhase, SeverityLevel, RiskLevel
from .modalities import ModalityRegistry, InterventionContext, InterventionResult, TechniqueProtocol

logger = structlog.get_logger(__name__)


class InterventionType(str, Enum):
    """Types of therapeutic interventions."""
    TECHNIQUE = "technique"
    PSYCHOEDUCATION = "psychoeducation"
    GROUNDING = "grounding"
    CRISIS = "crisis"
    REFLECTION = "reflection"
    VALIDATION = "validation"
    EXPLORATION = "exploration"


class InterventionPriority(str, Enum):
    """Intervention priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class InterventionPlan:
    """Plan for delivering an intervention."""
    plan_id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    intervention_type: InterventionType = InterventionType.TECHNIQUE
    priority: InterventionPriority = InterventionPriority.NORMAL
    technique: TechniqueProtocol | None = None
    modality: TherapyModality = TherapyModality.CBT
    rationale: str = ""
    prerequisites: list[str] = field(default_factory=list)
    estimated_duration_minutes: int = 15
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_for: datetime | None = None


@dataclass
class DeliveredIntervention:
    """Record of a delivered intervention."""
    intervention_id: UUID = field(default_factory=uuid4)
    plan_id: UUID | None = None
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    intervention_type: InterventionType = InterventionType.TECHNIQUE
    technique_name: str = ""
    modality: TherapyModality = TherapyModality.CBT
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_seconds: int = 0
    response_text: str = ""
    user_engagement: float = 0.0
    effectiveness_rating: float | None = None
    notes: list[str] = field(default_factory=list)
    follow_up_needed: bool = False


@dataclass
class InterventionQueue:
    """Queue of pending interventions for a session."""
    session_id: UUID
    planned: list[InterventionPlan] = field(default_factory=list)
    delivered: list[DeliveredIntervention] = field(default_factory=list)
    current: InterventionPlan | None = None


class InterventionDeliverySettings(BaseSettings):
    """Intervention delivery configuration."""
    max_interventions_per_session: int = Field(default=4)
    min_time_between_techniques_minutes: int = Field(default=5)
    enable_crisis_override: bool = Field(default=True)
    enable_adaptive_timing: bool = Field(default=True)
    grounding_duration_seconds: int = Field(default=60)
    validation_frequency: int = Field(default=3)
    model_config = SettingsConfigDict(env_prefix="INTERVENTION_DELIVERY_", env_file=".env", extra="ignore")


class InterventionDeliveryService:
    """
    Orchestrates therapeutic intervention delivery.

    Manages intervention planning, sequencing, timing, safety protocols,
    and adaptive delivery based on client response.
    """

    def __init__(
        self,
        settings: InterventionDeliverySettings | None = None,
        modality_registry: ModalityRegistry | None = None,
    ) -> None:
        self._settings = settings or InterventionDeliverySettings()
        self._registry = modality_registry or ModalityRegistry()
        self._queues: dict[UUID, InterventionQueue] = {}
        self._stats = {
            "interventions_planned": 0, "interventions_delivered": 0,
            "crisis_interventions": 0, "techniques_applied": 0,
        }
        logger.info("intervention_delivery_service_initialized")

    def create_session_queue(self, session_id: UUID) -> InterventionQueue:
        """Create intervention queue for a session."""
        queue = InterventionQueue(session_id=session_id)
        self._queues[session_id] = queue
        return queue

    def get_queue(self, session_id: UUID) -> InterventionQueue | None:
        """Get intervention queue for a session."""
        return self._queues.get(session_id)

    def plan_intervention(
        self,
        session_id: UUID,
        user_id: UUID,
        context: InterventionContext,
        preferred_type: InterventionType = InterventionType.TECHNIQUE,
    ) -> InterventionPlan | None:
        """
        Plan an intervention based on context.

        Args:
            session_id: Session identifier
            user_id: User identifier
            context: Current intervention context
            preferred_type: Preferred intervention type

        Returns:
            Intervention plan or None if not applicable
        """
        queue = self._queues.get(session_id)
        if not queue:
            queue = self.create_session_queue(session_id)

        if len(queue.delivered) >= self._settings.max_interventions_per_session:
            logger.debug("max_interventions_reached", session_id=str(session_id))
            return None

        if self._needs_grounding(context):
            return self._create_grounding_plan(session_id, user_id, context)

        if preferred_type == InterventionType.TECHNIQUE:
            result = self._registry.select_intervention_for_context(
                context, preferred_modality=None
            )
            if result:
                provider, technique = result
                plan = InterventionPlan(
                    session_id=session_id, user_id=user_id, intervention_type=InterventionType.TECHNIQUE,
                    technique=technique, modality=provider.modality,
                    rationale=f"Selected {technique.name} based on context analysis",
                    estimated_duration_minutes=technique.duration_minutes,
                )
                queue.planned.append(plan)
                self._stats["interventions_planned"] += 1
                logger.debug("intervention_planned", plan_id=str(plan.plan_id), technique=technique.name)
                return plan

        return self._create_reflection_plan(session_id, user_id, context)

    def _needs_grounding(self, context: InterventionContext) -> bool:
        """Check if grounding is needed."""
        return context.severity in [SeverityLevel.SEVERE, SeverityLevel.MODERATELY_SEVERE]

    def _create_grounding_plan(self, session_id: UUID, user_id: UUID, context: InterventionContext) -> InterventionPlan:
        """Create a grounding intervention plan."""
        plan = InterventionPlan(
            session_id=session_id, user_id=user_id, intervention_type=InterventionType.GROUNDING,
            priority=InterventionPriority.HIGH, modality=TherapyModality.MINDFULNESS,
            rationale="Grounding needed for stabilization", estimated_duration_minutes=3,
        )
        queue = self._queues[session_id]
        queue.planned.insert(0, plan)
        self._stats["interventions_planned"] += 1
        return plan

    def _create_reflection_plan(self, session_id: UUID, user_id: UUID, context: InterventionContext) -> InterventionPlan:
        """Create a reflection intervention plan."""
        plan = InterventionPlan(
            session_id=session_id, user_id=user_id, intervention_type=InterventionType.REFLECTION,
            priority=InterventionPriority.NORMAL, rationale="Reflection to deepen exploration",
            estimated_duration_minutes=5,
        )
        queue = self._queues[session_id]
        queue.planned.append(plan)
        self._stats["interventions_planned"] += 1
        return plan

    def deliver_intervention(
        self,
        session_id: UUID,
        plan: InterventionPlan,
        user_input: str,
        context: InterventionContext,
    ) -> DeliveredIntervention:
        """
        Deliver a planned intervention.

        Args:
            session_id: Session identifier
            plan: Intervention plan to deliver
            user_input: User's current input
            context: Intervention context

        Returns:
            Delivered intervention record
        """
        start_time = datetime.now(timezone.utc)
        response_text = ""
        notes = []

        if plan.intervention_type == InterventionType.CRISIS:
            response_text = self._deliver_crisis_intervention(context)
            self._stats["crisis_interventions"] += 1

        elif plan.intervention_type == InterventionType.GROUNDING:
            response_text = self._deliver_grounding(context)

        elif plan.intervention_type == InterventionType.TECHNIQUE and plan.technique:
            provider = self._registry.get_provider(plan.modality)
            if provider:
                result = provider.generate_response(plan.technique, user_input, context)
                response_text = result.response_generated
                notes = result.notes
                self._stats["techniques_applied"] += 1

        elif plan.intervention_type == InterventionType.REFLECTION:
            response_text = self._deliver_reflection(user_input, context)

        elif plan.intervention_type == InterventionType.VALIDATION:
            response_text = self._deliver_validation(user_input, context)

        elif plan.intervention_type == InterventionType.EXPLORATION:
            response_text = self._deliver_exploration(user_input, context)

        else:
            response_text = self._deliver_psychoeducation(context)

        end_time = datetime.now(timezone.utc)
        duration = int((end_time - start_time).total_seconds())

        delivered = DeliveredIntervention(
            plan_id=plan.plan_id, session_id=session_id, user_id=plan.user_id,
            intervention_type=plan.intervention_type, technique_name=plan.technique.name if plan.technique else "",
            modality=plan.modality, started_at=start_time, completed_at=end_time,
            duration_seconds=duration, response_text=response_text, notes=notes,
        )

        queue = self._queues.get(session_id)
        if queue:
            queue.delivered.append(delivered)
            if plan in queue.planned:
                queue.planned.remove(plan)

        self._stats["interventions_delivered"] += 1
        logger.info("intervention_delivered", intervention_id=str(delivered.intervention_id),
                    type=plan.intervention_type.value, duration=duration)

        return delivered

    def _deliver_crisis_intervention(self, context: InterventionContext) -> str:
        """Deliver crisis intervention response."""
        return (
            "I hear that you're going through an incredibly difficult time right now, and I want you to know "
            "that your safety is the most important thing. You're not alone in this. Let's take this moment "
            "together. Can you tell me - are you somewhere safe right now? If you're having thoughts of harming "
            "yourself, please consider reaching out to the 988 Suicide & Crisis Lifeline by calling or texting 988."
        )

    def _deliver_grounding(self, context: InterventionContext) -> str:
        """Deliver grounding intervention."""
        return (
            "Let's pause for a moment and ground ourselves here. Take a slow breath in... and out. "
            "Now, can you name 5 things you can see around you right now? Just notice them without judgment. "
            "This helps bring us back to the present moment where we can work together."
        )

    def _deliver_reflection(self, user_input: str, context: InterventionContext) -> str:
        """Deliver reflective intervention."""
        return (
            "What you've shared is meaningful. It sounds like there's a lot happening beneath the surface. "
            "I'm curious - what feels most important about what you just said? What stands out to you?"
        )

    def _deliver_validation(self, user_input: str, context: InterventionContext) -> str:
        """Deliver validation intervention."""
        return (
            "It makes complete sense that you would feel this way given what you're going through. "
            "Your feelings are valid and understandable. Many people in similar situations experience "
            "similar reactions. How does it feel to have that acknowledged?"
        )

    def _deliver_exploration(self, user_input: str, context: InterventionContext) -> str:
        """Deliver exploration intervention."""
        return (
            "Let's explore this a bit more. You mentioned something that seems important. "
            "Can you tell me more about what this means to you? What comes up when you think about it?"
        )

    def _deliver_psychoeducation(self, context: InterventionContext) -> str:
        """Deliver psychoeducation intervention."""
        return (
            "What you're experiencing has a name and is well understood. Many people go through similar "
            "challenges. Understanding how our thoughts, feelings, and behaviors connect can help us find "
            "new ways to respond. Would you like me to share more about how this works?"
        )

    def handle_crisis(self, session_id: UUID, user_id: UUID, risk_level: RiskLevel) -> DeliveredIntervention:
        """
        Handle crisis situation with immediate intervention.

        Args:
            session_id: Session identifier
            user_id: User identifier
            risk_level: Assessed risk level

        Returns:
            Delivered crisis intervention
        """
        logger.warning("crisis_intervention_triggered", session_id=str(session_id), risk_level=risk_level.value)

        context = InterventionContext(
            user_id=user_id, session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.SEVERE, current_concern="crisis",
        )

        plan = InterventionPlan(
            session_id=session_id, user_id=user_id, intervention_type=InterventionType.CRISIS,
            priority=InterventionPriority.CRITICAL, rationale=f"Crisis intervention for {risk_level.value} risk",
            estimated_duration_minutes=10,
        )

        queue = self._queues.get(session_id)
        if queue:
            queue.planned.insert(0, plan)

        return self.deliver_intervention(session_id, plan, "", context)

    def get_next_intervention(self, session_id: UUID) -> InterventionPlan | None:
        """Get next planned intervention for session."""
        queue = self._queues.get(session_id)
        if not queue or not queue.planned:
            return None

        queue.planned.sort(key=lambda p: (
            0 if p.priority == InterventionPriority.CRITICAL else
            1 if p.priority == InterventionPriority.HIGH else
            2 if p.priority == InterventionPriority.NORMAL else 3
        ))

        return queue.planned[0]

    def rate_intervention(self, intervention_id: UUID, session_id: UUID, effectiveness: float, engagement: float) -> bool:
        """Rate a delivered intervention."""
        queue = self._queues.get(session_id)
        if not queue:
            return False

        for intervention in queue.delivered:
            if intervention.intervention_id == intervention_id:
                intervention.effectiveness_rating = max(0.0, min(1.0, effectiveness))
                intervention.user_engagement = max(0.0, min(1.0, engagement))
                return True
        return False

    def get_session_summary(self, session_id: UUID) -> dict[str, Any]:
        """Get summary of interventions for a session."""
        queue = self._queues.get(session_id)
        if not queue:
            return {"error": "Queue not found"}

        return {
            "session_id": str(session_id),
            "planned_count": len(queue.planned),
            "delivered_count": len(queue.delivered),
            "interventions": [
                {
                    "type": i.intervention_type.value,
                    "technique": i.technique_name,
                    "duration_seconds": i.duration_seconds,
                    "effectiveness": i.effectiveness_rating,
                }
                for i in queue.delivered
            ],
            "techniques_used": list(set(i.technique_name for i in queue.delivered if i.technique_name)),
        }

    def clear_session(self, session_id: UUID) -> None:
        """Clear intervention queue for a session."""
        if session_id in self._queues:
            del self._queues[session_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        return {
            "active_sessions": len(self._queues),
            "total_planned": self._stats["interventions_planned"],
            "total_delivered": self._stats["interventions_delivered"],
            "crisis_interventions": self._stats["crisis_interventions"],
            "techniques_applied": self._stats["techniques_applied"],
        }

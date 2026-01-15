"""
Solace-AI Therapy Service - Domain Models.
Data classes for therapy operations and results.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..schemas import (
    SessionPhase, TherapyModality, RiskLevel, SeverityLevel,
    TechniqueDTO, HomeworkDTO, TreatmentPlanDTO, SessionSummaryDTO,
)


@dataclass
class SessionState:
    """Active therapy session state."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    treatment_plan_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    current_phase: SessionPhase = SessionPhase.PRE_SESSION
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mood_rating: int | None = None
    agenda_items: list[str] = field(default_factory=list)
    topics_covered: list[str] = field(default_factory=list)
    skills_practiced: list[str] = field(default_factory=list)
    insights_gained: list[str] = field(default_factory=list)
    techniques_used: list[TechniqueDTO] = field(default_factory=list)
    homework_assigned: list[HomeworkDTO] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    current_risk: RiskLevel = RiskLevel.NONE
    safety_flags: list[str] = field(default_factory=list)
    engagement_score: float = 0.0
    session_rating: float | None = None


@dataclass
class SessionStartResult:
    """Result from session start."""
    session_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    initial_message: str = ""
    suggested_agenda: list[str] = field(default_factory=list)
    loaded_context: bool = False


@dataclass
class TherapyMessageResult:
    """Result from therapy message processing."""
    response_text: str = ""
    current_phase: SessionPhase = SessionPhase.WORKING
    technique_applied: TechniqueDTO | None = None
    homework_assigned: list[HomeworkDTO] = field(default_factory=list)
    safety_alerts: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    processing_time_ms: int = 0


@dataclass
class SessionEndResult:
    """Result from session end."""
    summary: SessionSummaryDTO | None = None
    duration_minutes: int = 0
    recommendations: list[str] = field(default_factory=list)


@dataclass
class TechniqueSelectionResult:
    """Result from technique selection."""
    selected_technique: TechniqueDTO | None = None
    alternatives: list[TechniqueDTO] = field(default_factory=list)
    reasoning: str = ""
    contraindications_checked: bool = False


@dataclass
class PhaseTransitionResult:
    """Result from phase transition."""
    from_phase: SessionPhase
    to_phase: SessionPhase
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trigger: str = ""
    criteria_met: list[str] = field(default_factory=list)
    allowed: bool = True

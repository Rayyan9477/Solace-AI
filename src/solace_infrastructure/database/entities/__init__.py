"""
Domain entity definitions for the Solace-AI platform.

This package contains all centralized domain entities, organized by service domain.
All entities inherit from base models and are registered with SchemaRegistry to
ensure consistency and proper encryption/audit trail support.

Entity Modules:
- user_entities: User, UserPreferences, ConsentRecord
- safety_entities: SafetyAssessment, SafetyPlan, RiskFactor, ContraindicationCheck
- therapy_entities: TreatmentPlan, TherapySession, TherapyIntervention, HomeworkAssignment
- diagnosis_entities: DiagnosisSession, Symptom, Hypothesis, DiagnosisRecord
- memory_entities: MemoryRecord, MemoryUserProfile, SessionSummary, TherapeuticEvent
- personality_entities: PersonalityProfile, TraitAssessment, ProfileSnapshot
- notification_entities: Notification, DeliveryAttempt, UserNotificationPreferences, NotificationBatch
"""

from __future__ import annotations

# Import all entities to ensure they're registered at module load time.
# Order matters: user_entities first since other entities reference users table.

from .user_entities import (
    User,
    UserPreferences,
    ConsentRecord,
)

from .safety_entities import (
    ContraindicationCheck,
    RiskFactor,
    SafetyAssessment,
    SafetyPlan,
)

from .therapy_entities import (
    TreatmentPlan,
    TherapySession,
    TherapyIntervention,
    HomeworkAssignment,
)

from .diagnosis_entities import (
    DiagnosisSession,
    Symptom,
    Hypothesis,
    DiagnosisRecord,
)

from .memory_entities import (
    MemoryRecord,
    MemoryUserProfile,
    SessionSummary,
    TherapeuticEvent,
)

from .personality_entities import (
    PersonalityProfile,
    TraitAssessment,
    ProfileSnapshot,
)

from .notification_entities import (
    Notification,
    DeliveryAttempt,
    UserNotificationPreferences,
    NotificationBatch,
)

__all__ = [
    # User entities
    "User",
    "UserPreferences",
    "ConsentRecord",
    # Safety entities
    "SafetyAssessment",
    "SafetyPlan",
    "RiskFactor",
    "ContraindicationCheck",
    # Therapy entities
    "TreatmentPlan",
    "TherapySession",
    "TherapyIntervention",
    "HomeworkAssignment",
    # Diagnosis entities
    "DiagnosisSession",
    "Symptom",
    "Hypothesis",
    "DiagnosisRecord",
    # Memory entities
    "MemoryRecord",
    "MemoryUserProfile",
    "SessionSummary",
    "TherapeuticEvent",
    # Personality entities
    "PersonalityProfile",
    "TraitAssessment",
    "ProfileSnapshot",
    # Notification entities
    "Notification",
    "DeliveryAttempt",
    "UserNotificationPreferences",
    "NotificationBatch",
]

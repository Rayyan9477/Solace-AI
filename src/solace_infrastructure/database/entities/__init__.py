"""
Domain entity definitions for the Solace-AI platform.

This package contains all centralized domain entities, organized by service domain.
All entities inherit from base models and are registered with SchemaRegistry to
ensure consistency and proper encryption/audit trail support.

Entity Modules:
- user_entities: User, UserPreferences, ConsentRecord
- safety_entities: SafetyAssessment, SafetyPlan, RiskFactor, ContraindicationCheck
- therapy_entities: TherapySession, TherapyPlan, TherapeuticIntervention
- diagnosis_entities: DiagnosticAssessment, Diagnosis, Symptom
- memory_entities: ConversationMemory, MemoryEntry, MemoryVector
- notification_entities: Notification, NotificationTemplate
- analytics_entities: AnalyticsEvent, MetricSnapshot
"""

from __future__ import annotations

# Import all entities to ensure they're registered at module load time
from .safety_entities import (
    ContraindicationCheck,
    RiskFactor,
    SafetyAssessment,
    SafetyPlan,
)

# from .user_entities import User, UserPreferences, ConsentRecord  # TODO
# from .therapy_entities import TherapySession, TherapyPlan  # TODO
# from .diagnosis_entities import DiagnosticAssessment, Diagnosis  # TODO
# from .memory_entities import ConversationMemory, MemoryEntry  # TODO
# from .notification_entities import Notification, NotificationTemplate  # TODO
# from .analytics_entities import AnalyticsEvent, MetricSnapshot  # TODO

__all__ = [
    # Safety entities
    "SafetyAssessment",
    "SafetyPlan",
    "RiskFactor",
    "ContraindicationCheck",
    # TODO: Add other entities as they're created
]

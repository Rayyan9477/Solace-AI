"""REV-17 / GDPR right-to-erasure wiring for the safety service.

The shared orchestrator (``src/solace_infrastructure/gdpr/erasure.py``) is store
agnostic: services register a named async deleter per store, then drive them all
through :meth:`UserDataErasure.erase` under a fail-loud contract (every store is
attempted; any failure raises :class:`ErasureIncomplete` with the partial report;
a completion audit fires only on full success).

This module builds the safety service's OWN ``UserDataErasure``, registering each
of the four PHI stores this service owns — safety assessments, incidents, user
risk profiles, and safety plans — and exposes :func:`erase_user_data` as the
entry point that erases all of a single user's safety data.
"""
from __future__ import annotations

from typing import Callable, Optional
from uuid import UUID

import structlog

from solace_infrastructure.gdpr import ErasureReport, UserDataErasure

from ..domain.escalation import EscalationRepository
from .repository import (
    SafetyAssessmentRepository,
    SafetyIncidentRepository,
    SafetyPlanRepository,
    UserRiskProfileRepository,
)

logger = structlog.get_logger(__name__)

# Store names each safety PHI store registers under. Stable and table-aligned so
# the erasure audit report / structured logs read self-documenting.
STORE_SAFETY_ASSESSMENTS = "safety_assessments"
STORE_SAFETY_INCIDENTS = "safety_incidents"
STORE_USER_RISK_PROFILES = "user_risk_profiles"
STORE_SAFETY_PLANS = "safety_plans"
STORE_ESCALATIONS = "escalations"


def build_user_data_erasure(
    *,
    assessment_repo: SafetyAssessmentRepository,
    incident_repo: SafetyIncidentRepository,
    profile_repo: UserRiskProfileRepository,
    plan_repo: SafetyPlanRepository,
    escalation_repo: EscalationRepository,
) -> UserDataErasure:
    """Build a :class:`UserDataErasure` with all FIVE safety PHI stores registered.

    Each registered deleter is that repo's ``delete_user_data``, which removes
    ALL of a user's rows from exactly one store keyed on ``user_id``.
    ``register`` rejects duplicate names fail-loud, so a wiring mistake surfaces
    at construction time rather than silently skipping a store during erasure.

    ``escalation_repo`` is a REQUIRED store: the ``escalations`` table holds crisis
    PHI (crisis_level, free-text reason, session_id, context JSONB) keyed on
    ``user_id`` — omitting it would leave crisis records behind while falsely
    attesting a complete Art. 17 erasure, so it must be provided at wiring time.
    """
    erasure = UserDataErasure()
    erasure.register(STORE_SAFETY_ASSESSMENTS, assessment_repo.delete_user_data)
    erasure.register(STORE_SAFETY_INCIDENTS, incident_repo.delete_user_data)
    erasure.register(STORE_USER_RISK_PROFILES, profile_repo.delete_user_data)
    erasure.register(STORE_SAFETY_PLANS, plan_repo.delete_user_data)
    erasure.register(STORE_ESCALATIONS, escalation_repo.delete_user_data)
    return erasure


async def erase_user_data(
    user_id: UUID | str,
    *,
    assessment_repo: SafetyAssessmentRepository,
    incident_repo: SafetyIncidentRepository,
    profile_repo: UserRiskProfileRepository,
    plan_repo: SafetyPlanRepository,
    escalation_repo: EscalationRepository,
    audit_emit: Optional[Callable[[ErasureReport], object]] = None,
) -> ErasureReport:
    """Erase all of a user's safety data (GDPR Art. 17 / REV-17).

    Drives every registered safety store through the shared fail-loud
    orchestrator: all stores are attempted, and if ANY fails an
    :class:`ErasureIncomplete` is raised carrying the partial report — a partial
    erasure is never reported complete and a failure is never swallowed.
    ``audit_emit`` fires only on full success. The delete is keyed on
    ``user_id`` and does not depend on the request GUC.
    """
    erasure = build_user_data_erasure(
        assessment_repo=assessment_repo,
        incident_repo=incident_repo,
        profile_repo=profile_repo,
        plan_repo=plan_repo,
        escalation_repo=escalation_repo,
    )
    return await erasure.erase(str(user_id), audit_emit=audit_emit)

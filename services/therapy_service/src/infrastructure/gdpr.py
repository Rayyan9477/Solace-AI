"""REV-17: therapy-service GDPR right-to-erasure wiring.

Builds the therapy service's own :class:`UserDataErasure` and registers a store
deleter for every store that holds a user's therapy PHI:

  - ``treatment_plans``   -> :class:`PostgresTreatmentPlanRepository`
  - ``therapy_sessions``  -> :class:`PostgresTherapySessionRepository`
  - ``outcome_measures``  -> :class:`PostgresOutcomeMeasureRepository`
  - ``therapy_redis``     -> :class:`RedisSessionRepository`

Each registered deleter is an ``async (user_id: str) -> int`` per the shared
:data:`StoreDeleter` contract; the shared orchestrator passes the user id as a
string, so the wrappers coerce it to :class:`~uuid.UUID` (the type the therapy
repositories bind for their ``$1`` asyncpg params) before delegating to the
repository ``delete_user_data`` methods. The orchestrator (not this factory)
enforces the fail-loud contract: every store is attempted, partial erasure is
never reported complete, and the audit fires only on full success.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from solace_infrastructure.gdpr.erasure import UserDataErasure

if TYPE_CHECKING:
    from .postgres_repository import (
        PostgresOutcomeMeasureRepository,
        PostgresTherapySessionRepository,
        PostgresTreatmentPlanRepository,
    )
    from .redis_session_repository import RedisSessionRepository


def build_user_data_erasure(
    *,
    treatment_plan_repo: "PostgresTreatmentPlanRepository",
    session_repo: "PostgresTherapySessionRepository",
    outcome_repo: "PostgresOutcomeMeasureRepository",
    redis_session_repo: "RedisSessionRepository",
) -> UserDataErasure:
    """Assemble the therapy service's right-to-erasure orchestrator.

    Every repository must expose ``async delete_user_data(user_id: UUID) -> int``
    that removes ALL of the user's rows from its store. Registration order fixes
    the audit/report ordering: treatment_plans, therapy_sessions,
    outcome_measures, therapy_redis.
    """
    erasure = UserDataErasure()
    erasure.register(
        "treatment_plans", lambda uid: treatment_plan_repo.delete_user_data(UUID(uid))
    )
    erasure.register(
        "therapy_sessions", lambda uid: session_repo.delete_user_data(UUID(uid))
    )
    erasure.register(
        "outcome_measures", lambda uid: outcome_repo.delete_user_data(UUID(uid))
    )
    erasure.register(
        "therapy_redis", lambda uid: redis_session_repo.delete_user_data(UUID(uid))
    )
    return erasure

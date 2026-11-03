"""REV-17: diagnosis-service GDPR right-to-erasure wiring.

The diagnosis service owns one durable store for a user's PHI: the Postgres
``diagnosis_sessions`` and ``diagnosis_records`` tables, both wiped by
``DiagnosisRepositoryPort.delete_user_data``. This module registers that store
with the shared :class:`UserDataErasure` orchestrator so the fail-loud /
auditable erasure contract (a failed delete raises ``ErasureIncomplete`` rather
than being reported as a completed erasure) applies to the diagnosis delete.

This is additive: the actual delete SQL is unchanged; the factory only adapts
``delete_user_data`` into a :data:`StoreDeleter` and registers it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from solace_infrastructure.gdpr import UserDataErasure

if TYPE_CHECKING:
    from ..infrastructure.repository import DiagnosisRepositoryPort

# Registered store name for the diagnosis Postgres tables. Both
# diagnosis_sessions and diagnosis_records are removed by one repository call,
# so they are erased as a single named store.
DIAGNOSIS_POSTGRES_STORE = "diagnosis_postgres"
# The durable live-session Redis store (REV-16 / B2) holds special-category PHI
# (symptoms, differential, safety_flags) and must be erased too.
DIAGNOSIS_SESSION_STORE = "diagnosis_redis_session"


def build_user_data_erasure(
    repository: "DiagnosisRepositoryPort",
    session_store: Any = None,
) -> UserDataErasure:
    """Build the diagnosis service's GDPR right-to-erasure orchestrator.

    Registers the diagnosis Postgres tables and — when a durable live-session
    store is configured (REV-16 / B2) — the Redis live-session store, whose blobs
    hold special-category PHI that would otherwise escape erasure. The shared
    :class:`UserDataErasure` guarantees the fail-loud contract: if any delete
    fails, :meth:`UserDataErasure.erase` raises ``ErasureIncomplete`` carrying the
    partial report rather than reporting a completed erasure.
    """
    erasure = UserDataErasure()

    async def _delete_diagnosis_postgres(user_id: str) -> int:
        # erase() normalizes user_id to a str before dispatch, but the
        # repository DELETEs bind it as the UUID $1 param (and the in-memory
        # fake keys on UUID), so restore the UUID the repo contract expects.
        deleted = await repository.delete_user_data(UUID(str(user_id)))
        return int(deleted)

    erasure.register(DIAGNOSIS_POSTGRES_STORE, _delete_diagnosis_postgres)

    if session_store is not None:
        async def _delete_diagnosis_session(user_id: str) -> int:
            return int(await session_store.delete_user_data(UUID(str(user_id))))

        erasure.register(DIAGNOSIS_SESSION_STORE, _delete_diagnosis_session)

    return erasure

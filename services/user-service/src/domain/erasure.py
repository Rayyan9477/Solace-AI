"""REV-17 / GDPR right-to-erasure wiring for the user-service.

Builds a :class:`~solace_infrastructure.gdpr.erasure.UserDataErasure` with the
user-service's PHI stores registered under the shared erasure contract
(``src/solace_infrastructure/gdpr/erasure.py``). Each registered deleter deletes
ALL of one user's rows from a single store and returns the rows removed:

  - ``user_profile``: the DELIBERATE soft-delete/anonymize of the users row
    (:meth:`User.soft_delete`). The row is RETAINED for FK integrity and
    legal/audit retention, with email/display_name/password/phone/PII scrubbed.
    Returns 1 when a live row was anonymized, 0 when there was no such user.
    The users row is NEVER hard-deleted here -- soft-delete/anonymize is the
    intentional, legally-required behavior for this store.
  - ``user_preferences``: HARD delete of the ``user_preferences`` row(s)
    (:meth:`PostgresUserPreferencesRepository.delete`). Returns 1 when a row was
    removed, 0 when none existed.
  - ``clinician_assignments``: HARD delete of every ``clinician_patient_assignments``
    row referencing the user as patient OR clinician
    (:meth:`UserRepository.delete_assignments_for_user`). This special-category
    relationship must be removed EXPLICITLY: the soft-delete of the users row
    disables the ``ON DELETE CASCADE`` that would otherwise clean these up.

Designated PHI stores intentionally NOT registered here, with rationale (so the
omission is explicit, never silent):
  - ``consent_records``: immutable audit trail retained under GDPR Art. 17(3)(b)
    (legal-obligation exemption); has no delete path.
  - ``oauth_accounts`` (migration 004; holds the provider's real email): a
    designated PHI store whose ``ON DELETE CASCADE`` cleanup is disabled by the
    soft-delete design. It has NO live write path today (OAuth account linking is
    not yet persisted), so there is nothing to erase; a deleter MUST be registered
    here when that write path lands.
  - ``user_notification_preferences`` (migration 001, a REAL separate table — NOT
    columns on ``user_preferences``): has NO active write path in either the
    user-service or notification-service repositories today, so nothing is
    persisted to erase. Wire a deleter under the owning service when a write path
    is added.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from solace_infrastructure.gdpr.erasure import UserDataErasure

if TYPE_CHECKING:
    from ..infrastructure.repository import (
        UserPreferencesRepository,
        UserRepository,
    )

logger = structlog.get_logger(__name__)


def build_user_data_erasure(
    user_repository: UserRepository,
    preferences_repository: UserPreferencesRepository,
) -> UserDataErasure:
    """Build a :class:`UserDataErasure` with the user-service PHI stores registered.

    Args:
        user_repository: Repository owning the ``users`` table.
        preferences_repository: Repository owning the ``user_preferences`` table.

    Returns:
        A configured orchestrator; drive it via ``erase(user_id, audit_emit=...)``.
    """

    async def _erase_user_profile(user_id: str) -> int:
        """Soft-delete/anonymize the users row (RETAINED, PII scrubbed)."""
        uid = UUID(user_id)
        user = await user_repository.get_by_id(uid)
        if user is None:
            return 0
        # Deliberate soft delete: scrubs email/display_name/password/phone/PII,
        # sets status INACTIVE + deleted_at, but keeps the row for FK integrity
        # and legal/audit retention. Do NOT hard-delete the users row.
        user.soft_delete()
        await user_repository.update(user)
        return 1

    async def _erase_user_preferences(user_id: str) -> int:
        """Hard-delete the user_preferences row(s) for the user."""
        uid = UUID(user_id)
        deleted = await preferences_repository.delete(uid)
        return 1 if deleted else 0

    async def _erase_clinician_assignments(user_id: str) -> int:
        """Hard-delete clinician<->patient assignment rows referencing the user."""
        uid = UUID(user_id)
        return await user_repository.delete_assignments_for_user(uid)

    erasure = UserDataErasure()
    erasure.register("user_profile", _erase_user_profile)
    erasure.register("user_preferences", _erase_user_preferences)
    erasure.register("clinician_assignments", _erase_clinician_assignments)
    return erasure

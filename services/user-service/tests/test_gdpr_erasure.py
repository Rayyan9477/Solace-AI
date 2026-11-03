"""REV-17 / GDPR right-to-erasure tests for the user-service.

Acceptance criteria (encoded RED-first):

  * ``build_user_data_erasure`` registers exactly the user-service's PHI stores
    under the shared erasure contract: ``user_profile`` then ``user_preferences``.
  * Running erasure ANONYMIZES the users row in place (soft-delete: the row is
    RETAINED for FK integrity + legal/audit retention, with email/display_name/
    PII scrubbed) and HARD-deletes the user_preferences row(s).
  * The users row must NOT be hard-deleted (guards against over-reach that would
    rip out the deliberate soft-delete semantics).
  * If any store raises, erasure is fail-loud (``ErasureIncomplete``) yet still
    attempts every store (the surviving user_profile is still anonymized).
  * The production ``UserService.delete_user`` path drives the erasure contract:
    it hard-deletes preferences AND leaves the users row retained-but-anonymized.
"""
from __future__ import annotations

import pytest

from src.domain.entities import User, UserPreferences
from src.domain.value_objects import AccountStatus
from src.domain.service import UserService
from src.domain.erasure import build_user_data_erasure

from solace_infrastructure.gdpr.erasure import ErasureIncomplete

from fixtures import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
)


async def _seed_user(
    user_repo: InMemoryUserRepository,
    prefs_repo: InMemoryUserPreferencesRepository,
) -> User:
    """Persist a user + default preferences and return the user."""
    user = User(
        email="erase@example.com",
        password_hash="hashed_pw",
        display_name="Erase Me",
        phone_number="+15551234567",
        bio="something personal",
    )
    await user_repo.save(user)
    prefs = UserPreferences(user_id=user.user_id)
    await prefs_repo.save(prefs)
    return user


class _RaisingPrefsRepo(InMemoryUserPreferencesRepository):
    """Preferences repo whose hard delete blows up, to exercise fail-loud."""

    async def delete(self, user_id) -> bool:  # type: ignore[override]
        raise RuntimeError("preferences store down")


def test_registry_registers_expected_store_names() -> None:
    """The factory registers PHI stores in a deterministic order."""
    erasure = build_user_data_erasure(
        InMemoryUserRepository(),
        InMemoryUserPreferencesRepository(),
    )
    assert erasure.store_names == (
        "user_profile",
        "user_preferences",
        "clinician_assignments",
    )


@pytest.mark.asyncio
async def test_erase_anonymizes_user_row_and_hard_deletes_preferences() -> None:
    """users row retained-but-anonymized; user_preferences hard-deleted."""
    user_repo = InMemoryUserRepository()
    prefs_repo = InMemoryUserPreferencesRepository()
    user = await _seed_user(user_repo, prefs_repo)
    uid = user.user_id

    erasure = build_user_data_erasure(user_repo, prefs_repo)
    report = await erasure.erase(str(uid))

    assert report.complete is True
    assert report.deleted == {
        "user_profile": 1,
        "user_preferences": 1,
        "clinician_assignments": 0,
    }

    # users row is RETAINED (NOT hard-deleted) but fully anonymized.
    assert await user_repo.count(include_deleted=True) == 1
    retained = await user_repo.list_users(include_deleted=True)
    assert len(retained) == 1
    anon = retained[0]
    assert anon.user_id == uid
    assert anon.deleted_at is not None
    assert anon.status == AccountStatus.INACTIVE
    assert anon.email == f"deleted_{uid}@deleted.solace-ai.com"
    assert anon.display_name == "Deleted User"
    assert anon.password_hash == "DELETED"
    assert anon.phone_number is None
    assert anon.bio is None

    # user_preferences row is HARD-deleted.
    assert await prefs_repo.get_by_user_id(uid) is None


@pytest.mark.asyncio
async def test_erase_missing_user_reports_zero_not_failure() -> None:
    """No such user -> deleters report 0 rows, run still completes cleanly."""
    user_repo = InMemoryUserRepository()
    prefs_repo = InMemoryUserPreferencesRepository()
    erasure = build_user_data_erasure(user_repo, prefs_repo)

    report = await erasure.erase("00000000-0000-0000-0000-000000000001")

    assert report.complete is True
    assert report.deleted == {
        "user_profile": 0,
        "user_preferences": 0,
        "clinician_assignments": 0,
    }


@pytest.mark.asyncio
async def test_erase_removes_clinician_assignments_both_sides() -> None:
    """Assignment rows referencing the user as patient OR clinician are erased,
    and another user's assignments survive (no over-reach)."""
    from uuid import uuid4

    user_repo = InMemoryUserRepository()
    prefs_repo = InMemoryUserPreferencesRepository()
    user = await _seed_user(user_repo, prefs_repo)
    uid = user.user_id
    other_clinician, other_patient = uuid4(), uuid4()

    # The erased user appears on BOTH sides of the relationship.
    await user_repo.assign_patient_to_clinician(clinician_id=other_clinician, patient_id=uid)
    await user_repo.assign_patient_to_clinician(clinician_id=uid, patient_id=other_patient)
    # An unrelated assignment that must survive.
    await user_repo.assign_patient_to_clinician(clinician_id=other_clinician, patient_id=other_patient)

    erasure = build_user_data_erasure(user_repo, prefs_repo)
    report = await erasure.erase(str(uid))

    assert report.complete is True
    assert report.deleted["clinician_assignments"] == 2  # both sides removed
    # The user is gone from every assignment...
    assert await user_repo.get_assigned_clinician(uid) is None
    assert await user_repo.get_assigned_patients(uid) == []
    # ...but the unrelated relationship survives.
    assert other_patient in await user_repo.get_assigned_patients(other_clinician)


@pytest.mark.asyncio
async def test_erase_fails_loud_when_a_store_raises() -> None:
    """A raising store makes erasure incomplete, but every store is attempted."""
    user_repo = InMemoryUserRepository()
    prefs_repo = _RaisingPrefsRepo()
    user = await _seed_user(user_repo, prefs_repo)
    uid = user.user_id

    erasure = build_user_data_erasure(user_repo, prefs_repo)
    with pytest.raises(ErasureIncomplete) as excinfo:
        await erasure.erase(str(uid))

    report = excinfo.value.report
    assert report.complete is False
    assert "user_preferences" in report.failed
    # The user_profile deleter still ran and anonymized the row.
    assert report.deleted.get("user_profile") == 1
    retained = await user_repo.list_users(include_deleted=True)
    assert retained[0].deleted_at is not None
    assert retained[0].email == f"deleted_{uid}@deleted.solace-ai.com"


@pytest.mark.asyncio
async def test_delete_user_routes_through_erasure() -> None:
    """Production delete path erases preferences AND anonymizes (not deletes) the row."""
    user_repo = InMemoryUserRepository()
    prefs_repo = InMemoryUserPreferencesRepository()
    svc = UserService(
        user_repository=user_repo,
        preferences_repository=prefs_repo,
    )
    user = await _seed_user(user_repo, prefs_repo)
    uid = user.user_id

    result = await svc.delete_user(user_id=uid, reason="gdpr right to erasure")
    assert result.success is True

    # preferences hard-deleted
    assert await prefs_repo.get_by_user_id(uid) is None

    # users row retained-but-anonymized (over-reach guard: still exactly 1 row)
    assert await user_repo.count(include_deleted=True) == 1
    retained = await user_repo.list_users(include_deleted=True)
    assert len(retained) == 1
    assert retained[0].email == f"deleted_{uid}@deleted.solace-ai.com"
    assert retained[0].status == AccountStatus.INACTIVE
    assert retained[0].deleted_at is not None

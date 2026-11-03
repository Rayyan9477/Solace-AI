"""MVP-034: signup, consent and assignment INSERTs succeed on the migrated schema as solace_app."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from solace_common.request_context import acting_as_user
from conftest import build_repo

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _service(us):
    repo, svc = us.mod("infrastructure.postgres_repository"), us.mod("domain.service")
    return svc.UserService(
        user_repository=build_repo(repo.PostgresUserRepository),
        preferences_repository=build_repo(repo.PostgresUserPreferencesRepository),
        consent_repository=build_repo(repo.PostgresConsentRepository),
        integration_settings=svc.ServiceIntegrationSettings(notification_service_url="http://127.0.0.1:9"),
    )


async def test_mvp_034_registration_persists_user_and_preferences(us, cpm_default, superuser_conn) -> None:
    result = await _service(us).create_user(
        email=f"reg-{uuid4().hex[:8]}@test.invalid", password_hash="argon2-synthetic", display_name="Reg User")
    assert result.success is True, result.error_code
    uid = result.user.user_id
    row = await superuser_conn.fetchrow("SELECT encryption_key_id, role, status FROM users WHERE id = $1", uid)
    assert (row["encryption_key_id"], row["role"], row["status"]) == ("none", "user", "pending_verification")
    prefs = await superuser_conn.fetchrow(
        "SELECT id, jsonb_typeof(notification_channels) AS kind FROM user_preferences WHERE user_id = $1", uid)
    assert prefs is not None and prefs["id"] is not None and prefs["kind"] == "array"


async def test_mvp_034_consent_and_assignment_inserts_satisfy_the_schema(us, cpm_default, superuser_conn) -> None:
    repo, ent, vo = (us.mod("infrastructure.postgres_repository"), us.mod("domain.entities"),
                     us.mod("domain.value_objects"))
    users = build_repo(repo.PostgresUserRepository)
    clinician = await users.save(ent.User(email=f"c-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                          display_name="Clin", role=vo.UserRole.CLINICIAN))
    patient = await users.save(ent.User(email=f"p-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                        display_name="Pat"))
    with acting_as_user(patient.user_id):
        await build_repo(repo.PostgresConsentRepository).save(vo.ConsentRecord(
            consent_id=uuid4(), user_id=patient.user_id, consent_type=vo.ConsentType.TERMS_OF_SERVICE,
            granted=True, version="1.0"))
    assert await users.assign_patient_to_clinician(clinician.user_id, patient.user_id) is True
    assert await superuser_conn.fetchval(
        "SELECT count(*) FROM consent_records WHERE user_id = $1", patient.user_id) == 1
    assert await superuser_conn.fetchval(
        "SELECT count(*) FROM clinician_patient_assignments WHERE clinician_id = $1 AND patient_id = $2",
        clinician.user_id, patient.user_id) == 1


async def test_mvp_034_guard_preferences_without_user_context_are_rejected(us, cpm_default) -> None:
    """Guard (passes before and after): the fix relies on acting_as_user, not on a policy hole."""
    repo, ent = us.mod("infrastructure.postgres_repository"), us.mod("domain.entities")
    user = await build_repo(repo.PostgresUserRepository).save(
        ent.User(email=f"g-{uuid4().hex[:8]}@test.invalid", password_hash="x", display_name="G"))
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await build_repo(repo.PostgresUserPreferencesRepository).save(ent.UserPreferences(user_id=user.user_id))


async def test_mvp_034_repositories_write_as_solace_app_under_forced_rls(us, cpm_default, superuser_conn) -> None:
    """The writes above ran on the default pool as solace_app (no superuser, no BYPASSRLS, no role switch) into
    tables whose row level security is forced, so they passed the per-user policies, not around them."""
    repo = us.mod("infrastructure.postgres_repository")
    for cls in (repo.PostgresUserRepository, repo.PostgresUserPreferencesRepository, repo.PostgresConsentRepository):
        async with build_repo(cls)._acquire() as conn:
            role = await conn.fetchrow(
                "SELECT session_user::text AS login, current_user::text AS role, r.rolsuper, r.rolbypassrls "
                "FROM pg_roles r WHERE r.rolname = current_user")
            assert tuple(role) == ("solace_app", "solace_app", False, False), cls.__name__
    forced = await superuser_conn.fetch(
        "SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, pg_get_userbyid(c.relowner) AS owner "
        "FROM pg_class c WHERE c.relnamespace = 'public'::regnamespace AND c.relname = ANY($1::text[]) "
        "ORDER BY c.relname", ["consent_records", "user_preferences"])
    assert [tuple(r) for r in forced] == [("consent_records", True, True, "solace_migrator"),
                                          ("user_preferences", True, True, "solace_migrator")]


async def test_mvp_034_guard_consent_without_or_with_another_user_context_is_rejected(
        us, cpm_default, seed_user, superuser_conn) -> None:
    """Guard: consent and preferences rows are written only as their owner; acting as another user is no way
    around the policy. As the owner, the preferences round-trip with the JSONB channels as a list."""
    repo, ent, vo = (us.mod("infrastructure.postgres_repository"), us.mod("domain.entities"),
                     us.mod("domain.value_objects"))
    owner, other = await seed_user(), await seed_user()

    def _consent():
        return vo.ConsentRecord(consent_id=uuid4(), user_id=owner, consent_type=vo.ConsentType.TERMS_OF_SERVICE,
                                granted=True, version="1.0")

    consents, prefs = build_repo(repo.PostgresConsentRepository), build_repo(repo.PostgresUserPreferencesRepository)
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await consents.save(_consent())
    with acting_as_user(other):
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await consents.save(_consent())
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await prefs.save(ent.UserPreferences(user_id=owner))
    with acting_as_user(owner):
        await consents.save(_consent())
        await prefs.save(ent.UserPreferences(user_id=owner, notification_channels=["email", "sms"]))
        stored = await prefs.get_by_user_id(owner)
        assert stored is not None and stored.notification_channels == ["email", "sms"]
        stored.update(notification_channels=["push"])
        updated = await prefs.update(stored)
    assert updated.notification_channels == ["push"]
    assert await superuser_conn.fetchval(
        "SELECT jsonb_typeof(notification_channels) FROM user_preferences WHERE user_id = $1", owner) == "array"
    assert await superuser_conn.fetchval("SELECT count(*) FROM consent_records WHERE user_id = $1", owner) == 1
    assert await superuser_conn.fetchval("SELECT count(*) FROM user_preferences WHERE user_id = $1", other) == 0

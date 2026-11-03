import sys

p = r"D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation\WS-07-service-auth-authz.md"
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"expected 1 match, got {n}: {old[:90]!r}")
    t = t.replace(old, new)


# ---- WS-07.8: files created, dependencies, R-13-6 ----
rep("""**Depends on:** WS-07.1, WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21""", """**Depends on:** WS-07.1, WS-07.4, WS-13.3, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21""")
rep("""- Test: `tests/solace_security/test_mvp_120_subject_scoping.py`
- Test: `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py`""",
    """- Create: `tests/solace_security/test_mvp_120_subject_scoping.py` (test; WS-07.11 extends it in W4)
- Create: `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py` (test)""")
rep("""  - `async def authorize_subject(principal, subject_user_id, *, checker=None) -> None`: raises `HTTPException(403, detail={"code": "SUBJECT_FORBIDDEN", "message": "Not permitted for this user"})` without touching the context, else calls `set_subject_context`. Adopted in W3 phase B by WS-14 (`GET /users/{uuid}`), WS-15, WS-16 and WS-17 at their clinician paths (Cross-workstream requests).""",
    """  - `async def authorize_subject(principal, subject_user_id, *, checker=None) -> None`: raises `HTTPException(403, detail={"code": "SUBJECT_FORBIDDEN", "message": "Not permitted for this user"})` without touching the context, else calls `set_subject_context`. Adopted in W3 phase B by WS-14 (`GET /users/{uuid}`), WS-15, WS-16 and WS-17 at their clinician paths (Cross-workstream requests).
  - Satisfies request R-13-6 from WS-13 (raised against WS-13.3's slot 015): `crisis_alerts` has the CHECK constraints `max_level IN ('HIGH','CRITICAL')` and `status IN ('open','acknowledged','resolved')` and `max_level` has no default, so the real-infra test passes `max_level="HIGH"` and `status="acknowledged"` explicitly on every `crisis_alerts` row and never lets the generic filler choose them.""")
rep("""Expected: FAIL at collection with `ImportError: cannot import name 'authorize_subject'` and `cannot import name 'PostgresSubjectAccessChecker'`: nothing consults `clinician_patient_assignments` today (MVP-120) and the RLS user is always the caller (middleware.py:266-270, MVP-141).""",
    """Expected: FAIL at collection with `ImportError: cannot import name 'authorize_subject'` and `cannot import name 'PostgresSubjectAccessChecker'`: nothing consults `clinician_patient_assignments` today (MVP-120) and the RLS user is always the caller (middleware.py:266-270, MVP-141). If the real-infra test fails instead with `CheckViolationError` on `crisis_alerts`, an explicit `max_level`/`status` was dropped (R-13-6); that is a test defect, not the RED.""")

# ---- WS-07.2: pin failure mode 4 (no exception text in an error body) ----
rep("""    with pytest.raises(HTTPException) as exc:
        await mw.get_current_user(_FakeRequest(), creds)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_identity_claims_are_surfaced(monkeypatch):""",
    """    with pytest.raises(HTTPException) as exc:
        await mw.get_current_user(_FakeRequest(), creds)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Malformed token claims"  # fixed text: no ValidationError text, no claim value
    assert "password_reset" not in str(exc.value.detail) and "validation error" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_identity_claims_are_surfaced(monkeypatch):""")
rep("""Expected: FAIL. The parametrized test and the 401 test raise `pydantic_core._pydantic_core.ValidationError` (1 validation error for TokenPayload, `type` or `exp`), which is the 500 the register describes;""",
    """Expected: FAIL. The parametrized test and the 401 test raise `pydantic_core._pydantic_core.ValidationError` (1 validation error for TokenPayload, `type` or `exp`), which is the 500 the register describes; its message embeds the offending claim value, which is why the 401 test also pins the fixed detail that `_handle_auth_failure` (middleware.py:207-227) copies from `result.error_message`;""")

# ---- Self-review ----
rep("""- Phase rules: every WS-07 task is phase A (platform paths). No task depends on a same-wave phase-B task; WS-07.9 runs in W4 so all supporting route edits (W1 to W3) are behind a barrier.""",
    """- Phase rules: every WS-07 task is phase A (platform paths). No task depends on a same-wave phase-B task; WS-07.9 runs in W4 so all supporting route edits (W1 to W3) are behind a barrier. WS-07.8's dependencies WS-13.3 (W1 slot 015) and WS-05.2 (W0 fixtures) are earlier-wave phase-A tasks.
- Shared test file: WS-07.3 edits one existing case in `tests/integration/test_service_integration.py` (`TestPermissionMatrix::test_user_service_has_no_permissions`, lines 173-176), which asserts `SERVICE_PERMISSIONS` from `src/solace_security/service_auth.py`. The shared_files protocol for `tests/integration` lets the owner of the code under test edit an existing file, and WS-07 owns `service_auth.py`. No other plan edits that file (WS-01.8 only stops CI ignoring it).""")
rep("""- The real-infra subject checker test fills unknown NOT NULL columns generically; a CHECK constraint added by WS-13 or WS-14 on a column it does not pass explicitly would need an explicit value in the test.""",
    """- The real-infra subject checker test fills unknown NOT NULL columns generically. The known `crisis_alerts` CHECK columns (`max_level`, `status`) are passed explicitly (R-13-6); a CHECK constraint added later by WS-13 or WS-14 on a column the test does not pass would surface as `CheckViolationError` and needs an explicit value in the test.""")

rep("""- The obo TTL bound relies on reasonably synchronised clocks; the check allows the configured 30 s skew.
""", """- The obo TTL bound relies on reasonably synchronised clocks; the check allows the configured 30 s skew.

Review focus (numbered 1-5 in skeleton order):
- 1 (crisis false negatives) and 2 (alert-path failures) are outside this workstream's code; WS-07 adds no detection or paging logic.
- 3 (foreign-but-valid identifiers and context-less work): WS-07.1 pins that `acting_as_user` sets the RLS user for request-less work and always restores it, on exit, on exception and per concurrent task, so a pooled worker never keeps another user's id. WS-07.8 pins that an unassigned clinician is denied, that an obo principal reaches only its own user, and that the RLS user becomes the data owner only after the check passes; its real-infra test runs the checker's SQL as `solace_app` on the migrated schema.
- 4 (PHI in unplanned sinks): WS-07.2 pins that a malformed signed token gives a 401 with fixed detail text, never pydantic's error text with the claim values. WS-07.5 additionally pins that the Redis URL (a secret) never reaches a log.
- 5 (strict boots with partial configuration): WS-07.5 (strict environment without `REDIS_URL` fails loud; the in-memory override is refused in strict), WS-07.7 (strict rate limiting requires Redis) and WS-07.10 (strict refuses HS256 and the legacy flag, the keyset is required in staging as well as production, and HS256 service tokens are refused once service keys are configured).

REVIEW_FOCUS: 3=WS-07.1,WS-07.8; 4=WS-07.2; 5=WS-07.5,WS-07.7,WS-07.10
""")

t = t.rstrip("\n") + "\n\n" + """## Review responses

- Error: WS-07.11 modifies `tests/solace_security/test_mvp_120_subject_scoping.py`, which no earlier-wave task or dependency creates: fixed. WS-07.8 (W3, a declared dependency of WS-07.11) creates it, but its Files listed it as `Test:`, which the checker does not count as a creation; WS-07.8 now lists both of its test files as `Create:`.
- Error: plan a task for WS-13's request R-13-6 (the `crisis_alerts` CHECK constraints; the generic filler must pass `max_level='HIGH'`): fixed. It joins WS-07.8: the real-infra test already passes `max_level="HIGH"` and `status="acknowledged"` on both `crisis_alerts` inserts, and WS-07.8's Produces now names R-13-6 and WS-13.3, its Depends on adds WS-13.3 (slot 015, W1) and WS-05.2 (the `pg_migrated` fixture, W0), and Step 2 says that a `CheckViolationError` is a test defect, not the RED. Verified against WS-13.3's DDL: `max_level varchar(16) NOT NULL CHECK IN ('HIGH','CRITICAL')` has no default and `status` defaults to `'open'`.
- Warning: WS-07.3 edits `tests/integration/test_service_integration.py`, owned by WS-05: rejected: the shared_files protocol for `tests/integration` says an existing file is edited only by the owner of the code it tests. The one edited case, `TestPermissionMatrix::test_user_service_has_no_permissions` (`tests/integration/test_service_integration.py:173-176`), asserts `SERVICE_PERMISSIONS[ServiceIdentity.USER]` from `src/solace_security/service_auth.py`, which WS-07 owns, and no other plan edits the file. Recorded in the Self-review.
- Added the missing REVIEW_FOCUS line. To pin failure mode 4 in WS-07's own code, WS-07.2's 401 test now asserts the fixed detail and that no claim value or validation text reaches the body.
"""
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")

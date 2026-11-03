import sys

p = r"D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation\WS-08-identity-auth-mfa.md"
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"expected 1 match, got {n}: {old[:90]!r}")
    t = t.replace(old, new)


rep("| MVP-210 | medium | WS-08.3 (plus R-5 for `UserService.change_password`) |",
    "| MVP-210 | medium | WS-08.3; the password-change route path closes in WS-08.20, which stops calling `UserService.change_password` (R-5 is optional) |")
rep("| MVP-217 | medium | WS-08.7 (plus R-2 timing, R-3, R-4) |",
    "| MVP-217 | medium | WS-08.7 (the oracle); the timing half needs open request R-2, and the owner notice needs open requests R-3 and R-4 |")
rep("| MVP-326 | medium | WS-08.11 (plus R-9 for consent records) |",
    "| MVP-326 | medium | WS-08.11 (sessions); the consent-record half needs open request R-9 (WS-14's `api.py`, with the test named there) |")
rep("| MVP-099 | high | WS-08.15 (table), WS-08.18 |",
    "| MVP-099 | high | WS-08.15 (table), WS-08.18 (routes, token store, `User.set_password_hash`, `PasswordResetMailer`) |")

rep("The same-wave phase-B dependencies (R-7, R-8, R-9, R-10 in W3) are ordered by the controller ahead of the dependent WS-08 task and recorded in the ledger. The WS-08 tests themselves use spies, so they never block on lane timing.",
    "The same-wave phase-B dependencies on WS-14 (WS-14.16 for WS-08.8 and WS-14.17 for WS-08.13, both W3 phase B) are ordered by the controller ahead of the dependent WS-08 task and recorded in the ledger; no phase-A task depends on phase-B work. The WS-08 tests themselves use spies with the WS-14 signatures, so they never block on lane timing.\n"
    "- Shared test directory: `services/user-service/tests` is stewarded by WS-14, and its protocol lets the owner of the code under test edit an existing file and add `test_mvp_<id>_<slug>.py` files. The existing files WS-08 edits test only WS-08-owned code after the W0 split: `test_refresh_rotation.py` (refresh route and `auth.py`), `test_oauth_endpoints.py` (`oauth.py`), `test_oauth_google.py` (`oauth_google.py`), `test_auth_timing_health.py` (the login timing path), `test_rs256_issuer_migration.py` (`jwt_service.py`) and, in `test_api.py`, only the login, registration and password-change cases that exercise `api_auth.py` routes. WS-14 edits those files only in W0 (WS-14.3, import re-pointing) and W1 (WS-14.8, the on-call routing test), so no wave has two writers. `identity_testkit.py` is a new helper created by WS-08.2 and edited only by WS-08 tasks. WS-08 never edits `services/user-service/tests/conftest.py` or `fixtures.py`; WS-08.4's real-infra conftest lives in `tests/realinfra/identity-auth-mfa/`.\n"
    "- `docs/api/auth.md` is created by WS-08.7 and edited only by WS-08 tasks. WS-11.15 edits `docs/API-HANDOFF.md` and only links to `docs/api/auth.md` (WS-11 plan, WS-11.15 Files), so there is no second writer.")

rep("- 3 (foreign-but-valid identifiers and context-less RLS work): `test_mvp_344_mfa_repository_real_pg.py` (WS-08.17) proves a context-less MFA lookup still finds the factor, another user's context sees nothing, and the GUC does not leak. If the lookup silently read nothing, MFA would be skipped. WS-08.8 proves that another user's or another session's reauth token is refused, and WS-08.12 proves a Google subject cannot move to another user.",
    "- 3 (foreign-but-valid identifiers and context-less RLS work): `test_mvp_344_mfa_repository_real_pg.py` (WS-08.17) proves a context-less MFA lookup still finds the factor, another user's context sees nothing, and the GUC does not leak. If the lookup silently read nothing, MFA would be skipped. WS-08.8 (`test_mvp_229_reauth_token_is_bound_to_user_session_and_purpose`) proves that another user's or another session's reauth token is refused, and WS-08.12 proves a Google subject cannot move to another user.")
rep("- 5 (strict boots with partial configuration): `test_mvp_209_user_token_rs256_strict.py` (WS-08.19) covers each of the three RS256 variables missing and old-kid rotation. WS-08.4 and WS-08.6 cover strict mode without Redis, and WS-08.10 covers strict mode without a redirect URI or allow-list.",
    "- 5 (strict boots with partial configuration): `test_mvp_209_user_token_rs256_strict.py` (WS-08.19) covers each of the three RS256 variables missing and old-kid rotation. WS-08.4 and WS-08.6 cover strict mode without Redis, and WS-08.10 covers strict mode without a redirect URI or allow-list. WS-08.13's revocation test keeps a consent revocation from silently leaving tokens valid at other services.")

rep("- WS-14's plan was not yet written when this plan was drafted. The W0 split names (R-1) and the `UserService` methods in R-2 to R-10 are assumptions this plan defines. If WS-14 chooses different names, the dependent task's call site changes in this lane only, and that needs a ledger Ruling.",
    "- This revision aligns with WS-14's written plan: WS-14.3 (split), WS-14.6 (`verify_email(user_id | None, token)`), WS-14.16 (`DeleteUserResult.erasure_request_id`) and WS-14.17 (`get_consent_claim`). R-2 (timing half), R-3, R-4, R-5, R-9 and the W4 half of R-10 have no producing task in WS-14 or WS-13 yet. Until they land: the register timing differs slightly, an existing owner gets no notice email, consent records keep the proxy IP, and the MFA and reset-token rows survive an erasure. Each is recorded in the ledger as open at the barrier that needs it.")
rep("- `register` timing still differs slightly between a new and an existing address until R-2 moves the verification send out of the request path.\n", "")

rep("REVIEW_FOCUS: 3=WS-08.17; 4=WS-08.14; 5=WS-08.19",
    "REVIEW_FOCUS: 3=WS-08.17,WS-08.8,WS-08.12; 4=WS-08.14,WS-08.2,WS-08.10; 5=WS-08.19,WS-08.4,WS-08.10")

t = t.rstrip("\n") + "\n\n" + """## Review responses

- Error: WS-08.4 edits `services/user-service/tests` (WS-14) and lists `services/user-service`, its tests directory and its conftest under Create: fixed. The checker read the backticked paths inside the description of the one file WS-08.4 creates, `tests/realinfra/identity-auth-mfa/conftest.py`; WS-08.4 never edited the user-service suite's conftest. The line now names only the created file and describes the `sys.path` entries in words.
- Error: Wave 2 `services/user-service/tests/conftest.py` edited by WS-08.4 and WS-14.10: fixed by the same rewording; WS-08 has no conftest edit in any wave, so WS-14.10 is the only writer.
- Error: Wave 2 `docs/api/auth.md` edited by WS-08.7 and WS-11.15: rejected: WS-11.15's Files modify `docs/API-HANDOFF.md` and create `docs/api/orchestrator.md`; it only links to `docs/api/auth.md` ("auth links to `docs/api/auth.md`", WS-11 plan, WS-11.15 Files, line 1424). WS-08 owns and is the only writer of `docs/api/auth.md` (skeleton owned_paths; shared_files `docs/API-HANDOFF.md` protocol: "The auth contract lives in docs/api/auth.md (WS-08) and is linked, not copied"). Recorded in the Self-review.
- Error: plan a task for WS-14's request R-14-8 (verify-email accepts `token` alone, passing `user_id` when present): fixed. It joins WS-08.2: `VerifyEmailRequest{token, user_id: UUID | None = None}` plus the legacy query form, calling WS-14.6's `verify_email(user_id | None, token)`; the test spy records `(None, token)` for body calls and `(uuid, token)` for the body-with-id and legacy calls; Produces names R-14-8. Depends on now names WS-14.6.
- Also from WS-14 (not flagged, same kind): R-14-9 joins WS-08.8 (DELETE maps `DeleteUserResult`: 202 / 404 `USER_NOT_FOUND` / 503 `ERASURE_INCOMPLETE`, sessions revoked only on success, `reason="user_request"`, new failure-mapping test) and WS-08.13 (`cns` from `get_consent_claim`; `revoke_all_user_sessions` writes every sid to the shared blacklist, new test); R-14-10 joins WS-08.12 (`app.state.service.oauth_account_repository`, `email` written as NULL): fixed.
- Warning: WS-08.1 names `WS-14:W0-split` in Depends on: fixed (WS-14.3).
- Warning: WS-08.2 names `WS-14:W0-split` and `WS-14:verify_email_by_token`: fixed (WS-14.3, WS-14.6; the by-token method is withdrawn because WS-14.6's `verify_email(None, token)` consumes the token first).
- Warning: WS-08.7 names `WS-14:send_account_exists_email`, `WS-13:account_exists-template` and `WS-14:verification-send-off-request-path`: fixed. No WS-14 or WS-13 task produces them, so they are removed from Depends on and kept as open requests R-3, R-4 and R-2 (timing half); the owner notice is best-effort (exceptions are caught and logged by type only), so the 202 never depends on them.
- Warning: WS-08.8 names `WS-14:delete_user-returns-erasure-id`: fixed (WS-14.16, with the result-object shape it actually returns).
- Warning: WS-08.10 names `WS-03:oauth-env-sync`: fixed (WS-03.1 writes the IF-02 OAuth rows in W0; the strict-required flags and the STAGING-BRINGUP text stay in R-12).
- Warning: WS-08.11 names `WS-14:consent-client-ip`: fixed. No WS-14 task produces it, so R-9 stays open; the consent-record test, which exercises WS-14's `api.py`, moved out of WS-08.11 into R-9, and WS-08.11's tests cover the session half only.
- Warning: WS-08.12 names `WS-14:erasure-registers-identity-stores`: fixed. WS-14.16 already registers `oauth_accounts` from `state.oauth_account_repository`, which is an output of WS-08.12 rather than an input, so it is not a dependency; the W4 MFA and reset-token stores stay open in R-10.
- Warning: WS-08.13 names `WS-14:consent_claim`: fixed (WS-14.17, method `get_consent_claim`).
- Warning: WS-08.16 names `WS-01:pyotp-lock`: fixed (WS-01.2, whose D-16 lock includes pyotp).
- Warning: WS-08.18 and WS-08.20 name `WS-14:set_password_hash` and `WS-14:send_password_reset_email`: fixed. No WS-14 task produces them, so WS-08.18 now persists through its own `User.set_password_hash` (entities.py is WS-08's) and WS-14's existing `UserRepository.update`, and mails through its own `PasswordResetMailer` (IF-09 `/email` with `service_auth_headers`, existing `password_reset` template, new mailer test). Depends on WS-14.6, WS-07.3 and WS-13.12. WS-08.20 depends on WS-08.18. R-6 is withdrawn.
- Warnings: WS-08.2, .3, .4, .6, .7, .9, .10, .12, .19 edit existing user-service test files (`test_api.py`, `test_auth_timing_health.py`, `test_refresh_rotation.py`, `test_oauth_endpoints.py`, `test_oauth_google.py`, `test_rs256_issuer_migration.py`) owned by WS-14: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file, and each edited case tests WS-08-owned code (see Self-review). WS-14 touches these files only in W0 (WS-14.3) and W1 (WS-14.8), before any WS-08 wave.
- Warnings: WS-08.2, .3, .4, .6, .8, .10, .12, .14, .16, .18 edit `services/user-service/tests/identity_testkit.py`: rejected: it is a new helper file created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (WS-14 plan has no reference to it), so there is no existing-file ownership conflict.
- Warning: WS-08.4 edits `services/user-service/tests/conftest.py`: fixed (misparse removed, see the first error).
- Warning: WS-08.4 edits `services/user-service`, which no workstream owns: fixed (same misparse; the path was a `sys.path` entry in a description).
- Warnings: WS-08.4 lists `services/user-service`, `services/user-service/tests` and `services/user-service/tests/conftest.py` under Create: fixed (same misparse; only `tests/realinfra/identity-auth-mfa/conftest.py` is created).
"""
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")

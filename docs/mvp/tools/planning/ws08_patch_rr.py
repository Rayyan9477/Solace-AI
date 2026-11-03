import sys

p = r"D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation\WS-08-identity-auth-mfa.md"
t = open(p, encoding="utf-8").read()

marker = "\n## Review responses\n"
i = t.find(marker)
if i < 0 or t.count(marker) != 1:
    sys.exit("review responses marker")
t = t[:i]

old_title = "### Task WS-08.11: Sessions and consent records store the real client IP behind Caddy"
if t.count(old_title) != 1:
    sys.exit("title")
t = t.replace(old_title, "### Task WS-08.11: Sessions store the real client IP behind Caddy (consent records follow through request R-9)")

own = ("rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an "
       "existing file; {why} WS-14 edits it only in W0 (WS-14.3 import re-pointing){w1}, so no wave has two writers")
kit = ("rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or "
       "edits it (the WS-14 plan never names it), so there is no existing-file conflict")
why_api = ("the cases edited here exercise the `api_auth.py` routes WS-08 owns after the W0 split (login, registration); "
           "WS-14.5 itself attributes the `detail` assertion at `test_api.py:189` to the register route in `api_auth.py`.")
why_timing = "it tests the login dummy-verify and resend-verification throttle, both `/auth/*` routes in `api_auth.py` (WS-08)."
why_rot = "it tests `refresh_token` and `jwt_service.py` rotation, both WS-08's."
why_oauth = "it tests `oauth.py` start, callback and registration flows, which WS-08 owns."
why_google = "it tests `infrastructure/oauth_google.py`, which WS-08 owns."
why_rs = "it tests `infrastructure/jwt_service.py` RS256 issuance and verification, which WS-08 owns."

lines = [
    "- Error: WS-08.4 edits `services/user-service/tests`, which WS-14 owns: fixed. The checker read backticked paths inside the description of the one file WS-08.4 creates, `tests/realinfra/identity-auth-mfa/conftest.py`; WS-08.4 never edited the user-service suite. The line now names only the created file and describes the `sys.path` entries in words.",
    "- Error: Wave 2 `services/user-service/tests/conftest.py` edited by WS-08.4 and WS-14.10: fixed by the same rewording; no WS-08 task edits that conftest, so WS-14.10 is its only writer.",
    "- Error: Wave 2 `docs/api/auth.md` edited by WS-08.7 and WS-11.15: rejected: WS-11.15 modifies `docs/API-HANDOFF.md` and creates `docs/api/orchestrator.md`, and only links to `docs/api/auth.md` (WS-11 plan line 1424, \"auth links to `docs/api/auth.md`\"); the skeleton's `docs/API-HANDOFF.md` protocol says the auth contract \"lives in docs/api/auth.md (WS-08) and is linked, not copied\".",
    "- Error: plan a task for WS-14's request R-14-8 (verify-email accepts `token` alone, passing `user_id` when present): fixed. It joins WS-08.2: `VerifyEmailRequest{token, user_id: UUID | None = None}` plus the legacy query form, calling WS-14.6's `verify_email(user_id | None, token)`; the spy test records `(None, token)` for body calls and `(uuid, token)` for the body-with-id and legacy calls; Produces names R-14-8; Depends on names WS-14.6.",
    "- Note (same kind, not in the findings file): WS-14's R-14-9 joins WS-08.8 (DELETE maps `DeleteUserResult` to 202, 404 `USER_NOT_FOUND` or 503 `ERASURE_INCOMPLETE`, sessions revoked only on success, `reason=\"user_request\"`, new failure-mapping test) and WS-08.13 (`cns` from `get_consent_claim`; `revoke_all_user_sessions` writes every sid to the shared blacklist, new test); R-14-10 joins WS-08.12 (`app.state.service.oauth_account_repository`, `email` written as NULL): fixed.",
    "- Warning: WS-08.1 names `WS-14:W0-split` in Depends on: fixed (WS-14.3).",
    "- Warning: WS-08.2 names `WS-14:W0-split` and `WS-14:verify_email_by_token` in Depends on: fixed (WS-14.3 and WS-14.6; the by-token method is withdrawn because WS-14.6's `verify_email(None, token)` consumes the token first).",
    "- Warning: WS-08.7 names `WS-14:send_account_exists_email`, `WS-13:account_exists-template` and `WS-14:verification-send-off-request-path`: fixed. No WS-14 or WS-13 task produces them, so Depends on is WS-08.6 only and they stay open requests R-3, R-4 and R-2 (timing half); the owner notice is best-effort (exceptions caught, logged by type only), so the 202 never depends on them.",
    "- Warning: WS-08.8 names `WS-14:delete_user-returns-erasure-id`: fixed (WS-14.16, consumed in the `DeleteUserResult` shape it returns).",
    "- Warning: WS-08.10 names `WS-03:oauth-env-sync`: fixed (WS-03.1 writes the IF-02 OAuth rows in W0; the strict-required flags and the STAGING-BRINGUP text stay in R-12).",
    "- Warning: WS-08.11 names `WS-14:consent-client-ip`: fixed. No WS-14 task produces it, so R-9 stays open; the consent-record test, which exercises WS-14's `api.py`, moved into R-9, and WS-08.11 now covers the session half only.",
    "- Warning: WS-08.12 names `WS-14:erasure-registers-identity-stores`: fixed. WS-14.16 registers `oauth_accounts` from `state.oauth_account_repository`, an output of WS-08.12 rather than an input, so it is not a dependency; the W4 MFA and reset-token stores stay open in R-10.",
    "- Warning: WS-08.13 names `WS-14:consent_claim`: fixed (WS-14.17, method `get_consent_claim`).",
    "- Warning: WS-08.16 names `WS-01:pyotp-lock (D-16)`: fixed (WS-01.2, whose D-16 lock adds pyotp).",
    "- Warning: WS-08.18 names `WS-14:set_password_hash` and `WS-14:send_password_reset_email`: fixed. No WS-14 task produces them, so WS-08.18 persists through its own `User.set_password_hash` (entities.py is WS-08's) and WS-14's existing `UserRepository.update`, and mails through its own `PasswordResetMailer` (IF-09 `/email`, `service_auth_headers`, the existing `password_reset` template, new mailer test); Depends on WS-14.6, WS-07.3, WS-13.12; R-6 withdrawn.",
    "- Warning: WS-08.20 names `WS-14:set_password_hash`: fixed (depends on WS-08.18 for `User.set_password_hash` and persists through `UserRepository.update`).",
    "- Warning: WS-08.2 edits `services/user-service/tests/test_api.py`: " + own.format(why=why_api, w1=" and W1 (WS-14.8, the on-call routing case)"),
    "- Warning: WS-08.2 edits `services/user-service/tests/identity_testkit.py`: " + kit + " (WS-08.2 creates it).",
    "- Warning: WS-08.3 edits `services/user-service/tests/test_auth_timing_health.py`: " + own.format(why=why_timing, w1=""),
    "- Warning: WS-08.3 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.4 edits `services/user-service/tests/conftest.py`: fixed (misparse removed, see the first error).",
    "- Warning: WS-08.4 edits `services/user-service/tests/test_refresh_rotation.py`: " + own.format(why=why_rot, w1=""),
    "- Warning: WS-08.4 edits `services/user-service`, which no workstream owns: fixed (same misparse; the path was a `sys.path` entry in a description).",
    "- Warning: WS-08.4 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.6 edits `services/user-service/tests/test_refresh_rotation.py`: " + own.format(why=why_rot, w1=""),
    "- Warning: WS-08.6 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.7 edits `services/user-service/tests/test_oauth_endpoints.py`: " + own.format(why=why_oauth, w1=""),
    "- Warning: WS-08.7 edits `services/user-service/tests/test_api.py`: " + own.format(why=why_api, w1=" and W1 (WS-14.8, the on-call routing case)"),
    "- Warning: WS-08.8 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.9 edits `services/user-service/tests/test_oauth_google.py`: rejected: it tests `infrastructure/oauth_google.py`, which WS-08 owns, and the shared_files protocol lets the owner of the code under test edit an existing file; no WS-14 task edits it.",
    "- Warning: WS-08.10 edits `services/user-service/tests/test_oauth_endpoints.py`: " + own.format(why=why_oauth, w1=""),
    "- Warning: WS-08.10 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.12 edits `services/user-service/tests/test_oauth_endpoints.py`: " + own.format(why=why_oauth, w1=""),
    "- Warning: WS-08.12 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.14 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.16 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.18 edits `services/user-service/tests/identity_testkit.py`: " + kit + ".",
    "- Warning: WS-08.19 edits `services/user-service/tests/test_rs256_issuer_migration.py`: rejected: it tests `infrastructure/jwt_service.py` RS256 issuance and verification, which WS-08 owns, and the shared_files protocol lets the owner of the code under test edit an existing file; no WS-14 task edits it.",
    "- Warning: WS-08.4 lists `services/user-service` under Create: fixed (misparse removed; only `tests/realinfra/identity-auth-mfa/conftest.py` is created).",
    "- Warning: WS-08.4 lists `services/user-service/tests` under Create: fixed (same).",
    "- Warning: WS-08.4 lists `services/user-service/tests/conftest.py` under Create: fixed (same).",
]
t = t.rstrip("\n") + "\n\n## Review responses\n\n" + "\n".join(lines) + "\n"
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok", len(lines))

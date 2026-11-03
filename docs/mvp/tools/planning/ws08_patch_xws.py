import re
import sys

p = r"D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation\WS-08-identity-auth-mfa.md"
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"expected 1 match, got {n}: {old[:90]!r}")
    t = t.replace(old, new)


def row(num, new_text):
    """Replace the whole table row that starts with '| R-<num> |'."""
    global t
    pat = re.compile(r"^\| R-" + re.escape(num) + r" \|.*$", re.M)
    m = list(pat.finditer(t))
    if len(m) != 1:
        sys.exit(f"row R-{num}: {len(m)} matches")
    t = t[: m[0].start()] + new_text + t[m[0].end():]


row("1", "| R-1 | WS-14 | Produced by WS-14.3 (W0 phase B): the verbatim split as the skeleton describes: `auth_router` in `api_auth.py` (the `/auth/*` routes, `POST /users/me/password`, `DELETE /users/me` and the JWKS route, with handler names unchanged); the providers, `get_current_user`, `UserResponse`, `user_to_response` and `TokenResponse` in `api_deps.py`; `start_identity(runtime, app)`/`stop_identity()` in `identity_wiring.py`; `main.py` includes `auth_router` once; the existing user-service tests are re-pointed to `src.api_auth`. | `services/user-service/src/{api_auth,api_deps,identity_wiring,main}.py`, `services/user-service/tests/*` | all (WS-08.1 and WS-08.2 name WS-14.3) |")
row("2", "| R-2 | WS-14 | Partly produced. WS-14.6 (W1) delivers `UserService.verify_email(user_id: UUID | None, token)` with hashed, expiring tokens and the link `{frontend_url}/verify-email?user_id=<uuid>&token=<t>`; WS-08.2 calls it, so the earlier `verify_email_by_token` ask is withdrawn. Still open (no WS-14 task yet): send the verification email after commit from a background task rather than inline in `create_user`, so `register` timing does not reveal whether an account was created (the timing half of MVP-217). | `services/user-service/src/domain/service.py` | WS-08.2 (via WS-14.6); WS-08.7 timing half (open) |")
row("3", "| R-3 | WS-14 | Open (no WS-14 task yet). W2 phase B: `UserService.send_account_exists_email(user: User) -> bool`, posting template `account_exists` with variables `display_name` and `login_link` (`{frontend_url}/login`) through the same service-token notification call as WS-14.6, and logging nothing that identifies the address. Until it lands, WS-08.7's best-effort background call logs `account_exists_notice_failed` and the 202 is unchanged. | `services/user-service/src/domain/service.py` | WS-08.7 (owner notice only; the oracle fix does not wait on it) |")
row("4", "| R-4 | WS-13 | Open (no WS-13 task yet). W2 phase B: `TemplateType.ACCOUNT_EXISTS = \"account_exists\"` with a default template (subject \"Someone tried to create a Solace-AI account with this address\", required variables `display_name`, `login_link`; no token, no PHI). | `services/notification-service/src/domain/templates.py` | WS-08.7 (owner notice only) |")
row("5", "| R-5 | WS-14 | Open and optional (no WS-14 task). `UserService.change_password` verifies with `await self._password_service.verify_password_async(...)`. Without it, the password-change route keeps one synchronous verify on the loop in W2-W3; WS-08.20 (W4) stops calling `change_password`, which closes that path of MVP-210 inside this workstream. | `services/user-service/src/domain/service.py` | none (WS-08.20 closes the route path) |")
row("6", "| R-6 | WS-14 | Withdrawn. WS-08.18 persists the new hash through its own entity method `User.set_password_hash` and WS-14's existing `UserRepository.update`, and sends the reset email through its own `PasswordResetMailer` (IF-09 `/api/v1/notifications/email`, `service_auth_headers`, the existing `password_reset` template), so no new `UserService` method is needed. WS-08.20 reuses `User.set_password_hash`. | none | none |")
row("7", "| R-7 | WS-14 | Produced by WS-14.17 (W3 phase B) as `UserService.get_consent_claim(user_id) -> {\"ok\": bool, \"version\": str}`; revoking a required consent calls `session_manager.revoke_all_user_sessions(user_id)`. WS-08.13 consumes `get_consent_claim` and makes `revoke_all_user_sessions` write the sid keys, which is R-14-9's session half. | `services/user-service/src/domain/service.py`, `services/user-service/src/api.py` | WS-08.13 |")
row("8", "| R-8 | WS-14 | Produced by WS-14.16 (W3 phase B) in a different shape: `UserService.delete_user(user_id, reason) -> DeleteUserResult(success, error, error_code, erasure_request_id)` (WS-14's ledger Ruling on IF-16 step 1). WS-08.8 maps it as R-14-9 asks (202, 404 `USER_NOT_FOUND`, 503 `ERASURE_INCOMPLETE`). | `services/user-service/src/domain/service.py` | WS-08.8 |")
row("9", "| R-9 | WS-14 | Open (no WS-14 task yet). W3 phase B (MVP-326 consent half): `record_consent` stores `ip_address=client_ip(request)` (`solace_security.rate_limit`, env-resolved hops), with a new test `services/user-service/tests/test_mvp_326_consent_client_ip.py` asserting that `POST /api/v1/users/me/consent` sent with `X-Forwarded-For: 198.51.100.7` under `TRUSTED_PROXY_HOPS=1` returns 201 with `ip_address == \"198.51.100.7\"`. | `services/user-service/src/api.py` | MVP-326 closure (WS-08.11 covers the session half) |")
row("10", "| R-10 | WS-14 | Partly produced. WS-14.16 (W3) registers erasure store `oauth_accounts` when `state.oauth_account_repository` is present, which WS-08.12 sets (R-14-10). Still open (no WS-14 task yet), W4 phase B: stores `identity_mfa` and `identity_reset_tokens` with deleters calling `MFARepository.delete_for_user` (WS-08.17) and `PasswordResetTokenStore.delete_for_user` (WS-08.18), read from `app.state.service.mfa_repository` and `.password_reset_store`. The users row is soft-deleted, so ON DELETE CASCADE never fires. | `services/user-service/src/domain/erasure.py`, `services/user-service/src/main.py` | WS-08.12 (via WS-14.16); WS-08.17, WS-08.18 erasure completeness (open) |")
row("11", "| R-11 | WS-14 | Produced by WS-14.21 (W4 phase B), which adds the IF-04 `require_mfa_for_roles(\"admin\")` (WS-07.11) to the admin and provisioning routes. Provisioned accounts reach `mfa_enrollment_required` at first login through WS-08.16. | `services/user-service/src/api.py` | MVP-344 closure at the W4 barrier |")
row("12", "| R-12 | WS-03 | Partly produced. WS-03.1 (W0) writes the IF-02 OAuth rows (`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `GOOGLE_OAUTH_ISSUER`, `GOOGLE_OAUTH_AUTHORIZE_URL`, `GOOGLE_OAUTH_TOKEN_URL`, `GOOGLE_OAUTH_JWKS_URL`, `OAUTH_ALLOWED_REDIRECT_ORIGINS`) for user-service, which WS-08.10 depends on. Still open, for the next WS-03 phase-A env sync: `GOOGLE_OAUTH_CLIENT_SECRET` from `/run/secrets` (IF-22); `GOOGLE_OAUTH_REDIRECT_URI` and `OAUTH_ALLOWED_REDIRECT_ORIGINS` marked required in strict when the client id is set; `docs/STAGING-BRINGUP.md` documents them (the documentation half of MVP-104, WS-03.15). | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml`, `docs/STAGING-BRINGUP.md` | WS-08.10 |")

# New interfaces and self-review lines that described the withdrawn shapes
rep("- `DELETE /api/v1/users/me` returns 202 `{\"erasure_request_id\"}`.",
    "- `DELETE /api/v1/users/me` returns 202 `{\"erasure_request_id\"}`, 404 `USER_NOT_FOUND` or 503 `ERASURE_INCOMPLETE` (R-14-9).")
rep("- IF-16 is consumed as written: `DELETE /api/v1/users/me` requires `X-Reauth-Token` and calls `delete_user(user_id, reason) -> UUID`.",
    "- IF-16 is consumed as WS-14.16 delivers it: `DELETE /api/v1/users/me` requires `X-Reauth-Token` and calls `delete_user(user_id, reason=\"user_request\")`, which returns `DeleteUserResult` carrying `erasure_request_id` (WS-14's ledger Ruling on IF-16 step 1).")
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")

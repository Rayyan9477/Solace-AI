# C.3 — RS256/JWKS Asymmetric JWT Signing + Key Rotation (design)

Date: 2026-07-28
Scope: `src/solace_security/auth.py` (+ new `src/solace_security/keys.py`), shared by all 10 services.
Replaces: HS256 symmetric signing (every service shares `AUTH_SECRET_KEY` and can therefore *mint* user tokens, not just verify).

## Decisions (approved)
- **Key distribution:** static env keyset for verifiers (`AUTH_PUBLIC_KEYS_JSON`) + a published `/.well-known/jwks.json` at the issuer (user-service) for external consumers. No runtime JWKS fetch / network coupling. Rotation = redeploy config.
- **Migration:** dual-accept transition. Sign new user tokens RS256; verify RS256 first; accept legacy HS256 only while `AUTH_ACCEPT_LEGACY_HS256=true`. Flip off after the window.

## Components

### `src/solace_security/keys.py` (new, focused unit)
- `generate_rsa_keypair(kid) -> (private_pem, public_pem, kid)` — 2048-bit RSA.
- `KeyManager`:
  - Signing key: `active_kid`, `private_pem` (issuer only; may be None on verifiers).
  - Verification keyset: `dict[kid, public_pem]` (current + prior keys for rotation).
  - `sign(payload_dict) -> token` (RS256, sets `kid` header) — raises if no private key.
  - `public_pem_for(kid) -> str | None`.
  - `from_settings(settings)` builder; dev auto-generates an ephemeral keypair when RS256 is
    requested without provisioned keys AND `ENVIRONMENT != production`.
- `build_jwks(keyset) -> dict` — PEM → JWKS (`kty=RSA, use=sig, alg=RS256, kid, n, e`) for all kids.

### `AuthSettings` (additive; existing fields unchanged)
- `algorithm` default stays `HS256`; set `AUTH_ALGORITHM=RS256` to enable asymmetric signing.
- New: `private_key_pem: SecretStr | None` (`AUTH_PRIVATE_KEY_PEM`), `active_kid: str | None`
  (`AUTH_ACTIVE_KID`), `public_keys_json: str | None` (`AUTH_PUBLIC_KEYS_JSON`, `{kid: pem}`),
  `accept_legacy_hs256: bool = False` (`AUTH_ACCEPT_LEGACY_HS256`).
- `secret_key` stays REQUIRED (still used for HS256 legacy + service tokens) → no config break.

### `JWTManager` (public API unchanged: create_*/decode_token/validate_access_token[_async]/revoke)
- Constructs a `KeyManager` from settings.
- `_encode(payload)`:
  - user tokens (ACCESS/REFRESH/API_KEY): RS256 via KeyManager when enabled, else HS256.
  - SERVICE tokens: always HS256 with the shared service secret (mTLS is the enterprise path — C.3-later/C.4).
- `decode_token_sync(token, expected_type)`: read unverified header for `alg`/`kid`; then
  - `alg=RS256` → verify with `public_pem_for(kid)` and `algorithms=["RS256"]`; unknown kid → reject.
  - `alg=HS256` → allowed ONLY if `payload.type == SERVICE` OR `accept_legacy_hs256`; verify with the
    secret and `algorithms=["HS256"]`. (Post-transition, an HS256 access/refresh token is rejected.)
  - any other alg → reject. Never let the token's header choose the key type (RS256/HS256 confusion).
- Revocation blacklist behavior unchanged.

### JWKS endpoint (issuer = user-service)
- `GET /.well-known/jwks.json` → `build_jwks(keyset)`; served unauthenticated. Route added in
  user-service `main.py`/`api.py`; the PEM→JWK logic lives in shared `keys.py`.

## Security invariants (tested)
1. A token signed HS256 with `type=access` is REJECTED when `accept_legacy_hs256=false` (core win:
   a compromised service can no longer forge user tokens).
2. RS256 verify uses the RSA public key + `algorithms=["RS256"]` only → an HS256 token signed with the
   public-key bytes as an HMAC secret is rejected (confusion attack blocked).
3. Unknown `kid` → reject. Expired/wrong-issuer/wrong-audience unchanged.
4. Rotation: a token signed by a prior kid still verifies while that kid remains in the keyset.

## Test list (TDD)
- RS256 sign→verify round-trip (access/refresh/api_key).
- kid rotation: two kids in keyset, tokens from each verify; unknown kid rejected.
- legacy HS256: accepted when flag on, rejected (for access) when flag off; service token HS256 always accepted.
- alg-confusion: HS256-with-public-key-as-secret rejected; `alg=none` rejected.
- JWKS output: shape + one entry per kid, base64url `n`/`e`.
- dev auto-gen: `for_development()` yields a working RS256 signer/verifier without provisioned keys; refuses in production.
- Backward-compat: default (HS256, no RS keys) behaves exactly as today.

## Out of scope (later)
- mTLS for service-to-service (C.3-later / C.4); managed-secret sourcing of the private key (C.4/Vault).
- OAuth id_token JWKS *fetch* verification (REV-22) reuses `keys.py` JWK helpers — separate ticket.

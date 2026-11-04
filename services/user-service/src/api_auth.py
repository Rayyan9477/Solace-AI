"""
Solace-AI User Service - identity REST endpoints.

Registration, login, token refresh/logout/revocation, email verification, password change,
account deletion, the Google OAuth routes and the JWKS document. Split verbatim out of
``api.py`` and ``main.py`` (WS-14.3); handler names, paths, status codes, dependencies and
response models are unchanged.

``auth_router`` is the single export: it includes the inner identity router and the OAuth
router at ``/api/v1``, and the JWKS route at the root. ``main.ROUTERS`` mounts it at "".

Architecture Layer: Presentation
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
import structlog

from .domain.service import UserService
from .infrastructure.jwt_service import (
    JWTService,
    TokenPayload,
    TokenType,
    TokenExpiredError,
    TokenInvalidError,
)
from .infrastructure.password_service import PasswordService
from .infrastructure.repository import UserRepository
from .auth import SessionManager
from .api_deps import (
    TokenResponse,
    UserResponse,
    get_current_user,
    get_jwt_service,
    get_password_service,
    get_session_manager,
    get_user_repository,
    get_user_service,
    user_to_response,
)
from .oauth import oauth_router

logger = structlog.get_logger(__name__)
router = APIRouter()


def _mask_email(email: str) -> str:
    """Mask email for logging: u***r@domain.com."""
    if "@" not in email:
        return "***"
    local, domain = email.rsplit("@", 1)
    if len(local) <= 2:
        masked = local[0] + "***"
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{domain}"


# P2-11: constant-time login. On the user-not-found path we still run a password verify
# against a precomputed dummy hash so the request takes the same time as a wrong-password
# verify against a real account — closing the timing side channel that let an attacker
# enumerate which emails are registered. Computed once per process (lazily) from the
# injected PasswordService so it matches the real hashing cost.
_DUMMY_PASSWORD_HASH: str | None = None


def _constant_time_dummy_verify(password_service: PasswordService, password: str) -> None:
    """Burn the same work a real password verify would, discarding the result."""
    global _DUMMY_PASSWORD_HASH
    try:
        if _DUMMY_PASSWORD_HASH is None:
            _DUMMY_PASSWORD_HASH = password_service.hash_password(
                "timing-equalizer-not-a-real-password"
            )
        password_service.verify_password(password, _DUMMY_PASSWORD_HASH)
    except Exception:  # never let the timing-equalizer surface an error
        pass


# P2-4: per-account (per-email) anti-email-bomb throttle for /auth/resend-verification.
# The per-IP gateway bucket (user-service main._register_rate_limiting) does NOT stop a
# distributed bomb aimed at ONE victim address — an attacker rotating source IPs can still
# flood a single inbox. This bounds how many verification emails a single ADDRESS can
# trigger within a rolling window, regardless of source IP. It is deliberately in-process
# (per-worker) and additive: the endpoint stays enumeration-safe (always 204) — over-limit
# requests simply skip the send instead of leaking a distinguishing response.
_RESEND_THROTTLE_MAX = 3
_RESEND_THROTTLE_WINDOW_SECONDS = 3600
_resend_attempts: dict[str, list[float]] = {}


def _reset_resend_throttle() -> None:
    """Clear the per-account resend throttle state (used by tests)."""
    _resend_attempts.clear()


def _resend_email_allowed(email: str, *, now: float | None = None) -> bool:
    """Record a resend attempt for ``email`` and report whether it is within the
    per-account budget. Enumeration-safe by construction: it neither reads account state
    nor branches on registration status — it only counts requests per normalized address."""
    now = time.monotonic() if now is None else now
    key = (email or "").strip().lower()
    window_start = now - _RESEND_THROTTLE_WINDOW_SECONDS
    recent = [t for t in _resend_attempts.get(key, []) if t >= window_start]
    if len(recent) >= _RESEND_THROTTLE_MAX:
        _resend_attempts[key] = recent
        return False
    recent.append(now)
    _resend_attempts[key] = recent
    return True


# --- Request/Response Models ---


class RegisterRequest(BaseModel):
    """User registration request."""
    email: str = Field(..., min_length=5, max_length=255, description="User email")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    display_name: str = Field(..., min_length=1, max_length=100, description="Display name")
    timezone: str = Field(default="UTC", description="User timezone")
    locale: str = Field(default="en-US", description="User locale")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        from email_validator import validate_email as _validate_email, EmailNotValidError
        try:
            result = _validate_email(v, check_deliverability=False)
            return result.normalized
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e}")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    """User login request."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="Password")


class ResendVerificationRequest(BaseModel):
    """Request a new verification email by address (unauthenticated, REV-31)."""
    email: str = Field(..., min_length=5, max_length=255, description="User email")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        from email_validator import validate_email as _validate_email, EmailNotValidError
        try:
            result = _validate_email(v, check_deliverability=False)
            return result.normalized
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email: {e}")


class RefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


# --- Authentication Endpoints ---


@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
async def register(
    request_data: RegisterRequest,
    request: Request,
    user_service: UserService = Depends(get_user_service),
    password_service: PasswordService = Depends(get_password_service),
) -> UserResponse:
    """
    Register a new user account.

    Creates a new user with hashed password and returns user profile.
    Email verification is required before login.
    """
    # Hash password
    password_hash = password_service.hash_password(request_data.password)

    # Create user via service
    result = await user_service.create_user(
        email=request_data.email,
        password_hash=password_hash,
        display_name=request_data.display_name,
        timezone=request_data.timezone,
        locale=request_data.locale,
    )

    if not result.success or not result.user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if result.error_code == "EMAIL_EXISTS" else status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to create user",
        )

    # Increment stats
    request.app.state.service.increment_stat("registrations")

    logger.info("user_registered", user_id=str(result.user.user_id), email=_mask_email(result.user.email))

    return user_to_response(result.user)


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
)
async def login(
    request_data: LoginRequest,
    request: Request,
    jwt_service: JWTService = Depends(get_jwt_service),
    password_service: PasswordService = Depends(get_password_service),
    user_repository: UserRepository = Depends(get_user_repository),
    user_service: UserService = Depends(get_user_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.

    Verifies credentials and returns access/refresh token pair.
    """
    # Find user by email
    user = await user_repository.get_by_email(request_data.email.lower())
    if not user:
        # P2-11: equalize timing with the wrong-password path (see helper) so a missing
        # account is indistinguishable from a wrong password by response latency.
        _constant_time_dummy_verify(password_service, request_data.password)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user can login
    can_login, error = user.can_login()
    if not can_login:
        import os
        is_production = os.getenv("ENVIRONMENT", "").lower() == "production"
        if error == "Email verification required" and not is_production:
            logger.warning("login_without_verification", user_id=str(user.user_id),
                           hint="Allowed in non-production only")
        else:
            # P2-11: a blocked account (suspended / unverified-in-prod / locked) must not
            # short-circuit before the password hasher — otherwise its response is
            # measurably faster than a wrong-password attempt on an ACTIVE account, leaking
            # account state by timing. Burn an equalizing dummy verify before the 401.
            _constant_time_dummy_verify(password_service, request_data.password)
            logger.warning("login_blocked", user_id=str(user.user_id), reason=error)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

    # Verify password
    result = password_service.verify_password(request_data.password, user.password_hash)
    if not result.is_valid:
        await user_service.record_login_attempt(user.user_id, success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Update password hash if migration needed
    if result.needs_rehash and result.new_hash:
        user.password_hash = result.new_hash
        await user_repository.update(user)
        logger.info("password_migrated", user_id=str(user.user_id), algorithm=result.algorithm_used.value)

    # Record successful login
    await user_service.record_login_attempt(user.user_id, success=True)

    # Create session
    session = await session_manager.create_session(
        user=user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    # Generate tokens
    token_pair = jwt_service.generate_token_pair(
        user_id=user.user_id,
        email=user.email,
        role=user.role.value,
    )

    # Increment stats
    request.app.state.service.increment_stat("logins")

    logger.info("user_logged_in", user_id=str(user.user_id), email=_mask_email(user.email))

    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        token_type=token_pair.token_type,
        expires_in=token_pair.expires_in,
    )


@router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["Authentication"],
)
async def refresh_token(
    request_data: RefreshRequest,
    request: Request,
    jwt_service: JWTService = Depends(get_jwt_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Returns new access token without requiring re-authentication.
    Validates that the user still has an active session before issuing new tokens.
    """
    try:
        # Verify refresh token and get payload
        payload = await jwt_service.verify_token(
            request_data.refresh_token,
            expected_type=TokenType.REFRESH,
        )
    except TokenExpiredError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except TokenInvalidError as e:
        # P2-2 refresh-token reuse detection: verify_token raises this "revoked" error
        # only AFTER the signature + type checks pass, so a "revoked" message here means a
        # genuinely-signed refresh token whose JTI was already consumed by a prior rotation
        # is being REPLAYED. Treat it as a stolen-token signal and revoke the whole family
        # (all of the user's sessions). A forged/garbage token fails signature first and
        # raises a different message, so this cannot be abused to revoke a victim's
        # sessions with an unsigned token bearing their id.
        if "revoke" in str(e).lower():
            claims = jwt_service.decode_token_unsafe(request_data.refresh_token) or {}
            uid = claims.get("user_id")
            if uid:
                try:
                    await session_manager.revoke_all_user_sessions(UUID(uid))
                    logger.warning("refresh_token_reuse_detected_family_revoked", user_id=uid)
                except Exception as revoke_err:
                    logger.warning("refresh_family_revoke_failed", error=str(revoke_err))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    try:
        # Verify user still has at least one active session
        active_sessions = await session_manager.get_user_sessions(payload.user_id, active_only=True)
        if not active_sessions:
            logger.warning("token_refresh_denied_no_active_session", user_id=str(payload.user_id))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No active session found. Please log in again.",
            )

        # P2-2 rotation (revoke-old-then-issue): blacklist the CONSUMED refresh token's JTI
        # so a replay of THIS token is rejected on the next call (and triggers the family
        # revoke above), then mint a fresh access + refresh pair.
        await jwt_service.revoke_token(request_data.refresh_token)

        # Generate new token pair
        token_pair = jwt_service.generate_token_pair(
            user_id=payload.user_id,
            email=payload.email,
            role=payload.role,
        )

        # Increment stats
        request.app.state.service.increment_stat("token_refreshes")

        logger.info("token_refreshed", user_id=str(payload.user_id))

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
        )

    except HTTPException:
        raise
    except (TokenExpiredError, TokenInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def logout(
    request: Request,
    current_user: TokenPayload = Depends(get_current_user),
    session_manager: SessionManager = Depends(get_session_manager),
    authorization: str | None = Header(None),
) -> None:
    """
    Logout user by revoking all their sessions AND blacklisting the presented
    access token.

    REV-08: revoking sessions alone left the bearer access token valid until its
    natural expiry (~15 min). Blacklisting its JTI makes it rejected immediately.
    """
    await session_manager.revoke_all_user_sessions(current_user.user_id)
    if authorization:
        try:
            jwt_service = get_jwt_service(request)
            token = jwt_service.extract_token_from_header(authorization)
            await jwt_service.revoke_token(token)
        except Exception as e:
            logger.warning("logout_token_blacklist_failed", error=str(e))
    logger.info("user_logged_out", user_id=str(current_user.user_id))
    return None


@router.post(
    "/auth/revoke-token",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def revoke_token(
    request: Request,
    authorization: str = Header(...),
) -> None:
    """
    Revoke the current access token.

    Adds the token's JTI to the blacklist so it will be rejected on subsequent
    requests even before natural expiry. Useful for forced logout or token compromise.
    """
    jwt_service = get_jwt_service(request)
    try:
        token = jwt_service.extract_token_from_header(authorization)
    except (TokenExpiredError, TokenInvalidError) as e:
        # W142-SEC-1: no app-level token-error handler exists (IF-01), so map it here.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    revoked = await jwt_service.revoke_token(token)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token revocation service unavailable",
        )
    logger.info("token_revoked_via_api")


@router.post(
    "/users/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
)
async def change_password(
    request_data: PasswordChangeRequest,
    request: Request,
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    password_service: PasswordService = Depends(get_password_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Change current user's password."""
    # Hash new password
    new_password_hash = password_service.hash_password(request_data.new_password)

    # Change password via service
    result = await user_service.change_password(
        user_id=current_user.user_id,
        current_password=request_data.current_password,
        new_password_hash=new_password_hash,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to change password",
        )

    # Revoke all sessions except current
    await session_manager.revoke_all_user_sessions(current_user.user_id)

    # Increment stats
    request.app.state.service.increment_stat("password_changes")

    logger.info("password_changed", user_id=str(current_user.user_id))


@router.delete(
    "/users/me",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Users"],
)
async def delete_current_user(
    current_user: TokenPayload = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    session_manager: SessionManager = Depends(get_session_manager),
) -> None:
    """Delete current user's account (soft delete for GDPR compliance)."""
    result = await user_service.delete_user(
        user_id=current_user.user_id,
        reason="User requested account deletion",
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to delete account",
        )

    # Revoke all sessions
    await session_manager.revoke_all_user_sessions(current_user.user_id)

    logger.info("user_deleted_self", user_id=str(current_user.user_id))


# --- Email Verification Endpoints ---


@router.post(
    "/auth/verify-email",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def verify_email(
    user_id: UUID,
    token: str,
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Verify user email address."""
    result = await user_service.verify_email(user_id, token)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Failed to verify email",
        )

    logger.info("email_verified", user_id=str(user_id))


@router.post(
    "/auth/resend-verification",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Authentication"],
)
async def resend_verification(
    request_data: ResendVerificationRequest,
    user_service: UserService = Depends(get_user_service),
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
) -> None:
    """Request a new verification email (UNAUTHENTICATED, enumeration-safe — REV-31).

    Always returns 204 regardless of whether the address is registered or already
    verified: no account-enumeration signal is leaked. This breaks the deadlock
    where an unverified production user (blocked from login until verified) could
    not reach the previously auth-gated resend path to obtain a fresh verification
    link if their initial email failed to send. Abuse (email bombing) is bounded by
    the API gateway rate limiter, not by requiring a login the user cannot perform.
    """
    # P2-4: per-account anti-email-bomb throttle. Over-budget requests for the same
    # address are dropped (no send) but STILL return 204 below — enumeration-safe. This
    # runs before the (branch-independent) work below, and itself does no account lookup,
    # so it does not reintroduce a timing oracle.
    send_allowed = _resend_email_allowed(request_data.email)

    # P2-5: run the lookup + email send OFF the synchronous request path so the
    # registered / unknown / already-verified branches all return with equal timing
    # (the branch-dependent DB read + notification call no longer happen before the
    # response). Falls back to inline await when no BackgroundTasks is injected (e.g.
    # direct unit-test invocation).
    if send_allowed:
        if background_tasks is not None:
            background_tasks.add_task(
                user_service.resend_verification_email_by_email, request_data.email
            )
        else:
            await user_service.resend_verification_email_by_email(request_data.email)
    # Enumeration-safe: log nothing that distinguishes registered/verified/unknown
    # addresses, and return 204 unconditionally. (The prior log referenced
    # current_user/token, removed when this endpoint became unauthenticated — it
    # raised NameError on every call, defeating the always-204 contract.)
    logger.info("resend_verification_requested")


# --- JWKS ---


jwks_router = APIRouter()


@jwks_router.get("/.well-known/jwks.json", tags=["Auth"], include_in_schema=False)
async def jwks() -> dict[str, Any]:
    """Publish the RS256 public verification keyset (JWKS) for token validation (C.3).

    user-service is the token issuer, so it publishes its public keys here for external
    consumers (frontend, OAuth). Internal services load the same keyset from
    AUTH_PUBLIC_KEYS_JSON. Empty ({"keys": []}) while running in HS256 mode.
    """
    from solace_security.auth import AuthSettings
    from solace_security.keys import KeyManager

    try:
        return KeyManager.from_settings(AuthSettings()).jwks()
    except Exception as e:  # never let key config take down the endpoint
        logger.warning("jwks_unavailable", error=str(e))
        return {"keys": []}


# The single export. The OAuth router (REV-22) is always registered; its handlers config-gate
# themselves (404 when oauth_settings is None), so it is inert unless Google OAuth is set.
auth_router = APIRouter()
auth_router.include_router(router, prefix="/api/v1")
auth_router.include_router(oauth_router, prefix="/api/v1")
auth_router.include_router(jwks_router)

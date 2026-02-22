"""
Solace-AI User Service - Domain Service.

Business logic for user management, authentication workflows, and preferences.
Implements the service layer pattern for coordinating domain operations.

Architecture Layer: Domain
Principles: Single Responsibility, Dependency Inversion, Domain Events
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .entities import User, UserPreferences
from .value_objects import AccountStatus, ConsentRecord, ConsentType, UserRole

try:
    from solace_security.service_auth import ServiceTokenManager, ServiceIdentity
    _SERVICE_AUTH_AVAILABLE = True
except ImportError:
    _SERVICE_AUTH_AVAILABLE = False

if TYPE_CHECKING:
    from ..infrastructure.repository import (
        ConsentRepository,
        UserPreferencesRepository,
        UserRepository,
    )
    from ..infrastructure.password_service import PasswordService
    from ..events import EventPublisher

logger = structlog.get_logger(__name__)


class ServiceIntegrationSettings(BaseSettings):
    """Configuration for inter-service communication."""

    therapy_service_url: str = Field(
        default="http://localhost:8004",
        description="URL of the Therapy Service for progress data",
    )
    notification_service_url: str = Field(
        default="http://localhost:8005",
        description="URL of the Notification Service for emails/SMS",
    )
    frontend_url: str = Field(
        default="http://localhost:3000",
        description="Frontend URL for verification links",
    )
    request_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout in seconds",
    )

    model_config = SettingsConfigDict(
        env_prefix="USER_SERVICE_",
        env_file=".env",
        extra="ignore",
    )


# --- Result Types ---


@dataclass
class CreateUserResult:
    """Result of user creation operation."""
    success: bool = False
    user: User | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class UpdateUserResult:
    """Result of user update operation."""
    success: bool = False
    user: User | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class DeleteUserResult:
    """Result of user deletion operation."""
    success: bool = False
    error: str | None = None
    error_code: str | None = None


@dataclass
class PasswordChangeResult:
    """Result of password change operation."""
    success: bool = False
    error: str | None = None
    error_code: str | None = None


@dataclass
class EmailVerificationResult:
    """Result of email verification operation."""
    success: bool = False
    error: str | None = None
    error_code: str | None = None


@dataclass
class UpdatePreferencesResult:
    """Result of preferences update operation."""
    success: bool = False
    preferences: UserPreferences | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class ConsentResult:
    """Result of consent recording operation."""
    success: bool = False
    consent: ConsentRecord | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class UserProgress:
    """User progress summary."""
    user_id: UUID
    total_sessions: int = 0
    completed_assessments: int = 0
    streak_days: int = 0
    total_minutes: int = 0
    last_session: datetime | None = None
    mood_trend: str = "stable"
    engagement_score: float = 0.0


# --- Domain Service ---


class UserService:
    """
    User domain service coordinating user management operations.

    Responsibilities:
    - User CRUD operations with business rule validation
    - Password management and verification
    - Email verification workflow
    - Preferences management
    - Consent tracking for compliance
    - Domain event publishing

    Design Patterns:
    - Service Layer: Coordinates operations across repositories
    - Result Pattern: Returns structured results for error handling
    - Domain Events: Publishes events for async processing
    """

    def __init__(
        self,
        user_repository: UserRepository,
        preferences_repository: UserPreferencesRepository,
        consent_repository: ConsentRepository | None = None,
        password_service: PasswordService | None = None,
        event_publisher: EventPublisher | None = None,
        integration_settings: ServiceIntegrationSettings | None = None,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        verification_token_expiry_hours: int = 24,
    ) -> None:
        """
        Initialize user service.

        Args:
            user_repository: Repository for user persistence
            preferences_repository: Repository for preferences persistence
            consent_repository: Optional repository for consent records
            password_service: Service for password hashing/verification
            event_publisher: Optional publisher for domain events
            integration_settings: Settings for inter-service communication
            max_login_attempts: Max failed login attempts before lockout
            lockout_duration_minutes: Duration of account lockout
            verification_token_expiry_hours: Email verification token expiry
        """
        self._user_repo = user_repository
        self._prefs_repo = preferences_repository
        self._consent_repo = consent_repository
        self._password_service = password_service
        self._event_publisher = event_publisher
        self._integration_settings = integration_settings or ServiceIntegrationSettings()
        self._max_login_attempts = max_login_attempts
        self._lockout_duration_minutes = lockout_duration_minutes
        self._verification_token_expiry_hours = verification_token_expiry_hours

        # Service auth for inter-service calls
        self._token_manager: ServiceTokenManager | None = None
        if _SERVICE_AUTH_AVAILABLE:
            try:
                self._token_manager = ServiceTokenManager()
            except Exception:
                logger.warning("service_token_manager_init_failed", hint="inter-service calls will be unauthenticated")

        # Statistics tracking
        self._stats = {
            "users_created": 0,
            "users_updated": 0,
            "users_deleted": 0,
            "password_changes": 0,
            "email_verifications": 0,
        }

    def _get_service_auth_headers(self, target_service: str) -> dict[str, str]:
        """Get auth headers for inter-service HTTP calls."""
        if self._token_manager is None:
            return {}
        try:
            creds = self._token_manager.get_or_create_token(
                ServiceIdentity.USER.value, target_service,
            )
            return {
                "Authorization": f"Bearer {creds.token}",
                "X-Service-Name": ServiceIdentity.USER.value,
            }
        except Exception:
            logger.warning("service_auth_header_failed", target=target_service)
            return {}

    # --- User CRUD Operations ---

    async def create_user(
        self,
        email: str,
        password_hash: str,
        display_name: str,
        timezone: str = "UTC",
        locale: str = "en-US",
        role: UserRole = UserRole.USER,
    ) -> CreateUserResult:
        """
        Create a new user account.

        Business Rules:
        - Email must be unique (checked at repository)
        - Password must already be hashed by caller
        - Initial status is PENDING_VERIFICATION
        - Verification token is generated automatically

        Args:
            email: User email (will be normalized)
            password_hash: Pre-hashed password
            display_name: User display name
            timezone: User timezone (IANA format)
            locale: User locale (ISO 639-1)
            role: Initial user role

        Returns:
            CreateUserResult with user or error
        """
        try:
            # Check for existing user
            existing = await self._user_repo.get_by_email(email)
            if existing:
                return CreateUserResult(
                    error="Email already registered",
                    error_code="EMAIL_EXISTS",
                )

            # Generate verification token
            verification_token = secrets.token_urlsafe(32)

            # Create user entity
            user = User(
                email=email.lower().strip(),
                password_hash=password_hash,
                display_name=display_name.strip(),
                timezone=timezone,
                locale=locale,
                role=role,
                status=AccountStatus.PENDING_VERIFICATION,
                email_verification_token=verification_token,
            )

            # Persist user
            saved_user = await self._user_repo.save(user)

            # Create default preferences
            preferences = UserPreferences(user_id=saved_user.user_id)
            await self._prefs_repo.save(preferences)

            # Update statistics
            self._stats["users_created"] += 1

            # Publish domain event
            if self._event_publisher:
                await self._event_publisher.publish_user_created(saved_user)

            logger.info(
                "user_created",
                user_id=str(saved_user.user_id),
                email=saved_user.email,
            )

            # Send verification email
            email_sent = await self._send_verification_email(
                email=saved_user.email,
                display_name=saved_user.display_name,
                token=verification_token,
            )

            if not email_sent:
                logger.warning(
                    "initial_verification_email_failed",
                    user_id=str(saved_user.user_id),
                    email=saved_user.email,
                )

            return CreateUserResult(success=True, user=saved_user)

        except Exception as e:
            logger.error("user_creation_failed", error=str(e))
            return CreateUserResult(
                error=f"Failed to create user: {str(e)}",
                error_code="CREATE_FAILED",
            )

    async def get_user(self, user_id: UUID) -> User | None:
        """
        Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User entity or None if not found
        """
        return await self._user_repo.get_by_id(user_id)

    async def get_user_by_email(self, email: str) -> User | None:
        """
        Get user by email address.

        Args:
            email: User email (will be normalized)

        Returns:
            User entity or None if not found
        """
        return await self._user_repo.get_by_email(email.lower().strip())

    async def update_user(
        self,
        user_id: UUID,
        **updates: Any,
    ) -> UpdateUserResult:
        """
        Update user profile fields.

        Business Rules:
        - Only allowed fields can be updated
        - Immutable fields (user_id, email, created_at) rejected
        - Status changes must follow state machine rules

        Args:
            user_id: User identifier
            **updates: Field updates (display_name, timezone, locale, etc.)

        Returns:
            UpdateUserResult with updated user or error
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return UpdateUserResult(
                    error="User not found",
                    error_code="USER_NOT_FOUND",
                )

            # Filter to allowed fields
            allowed_fields = {"display_name", "timezone", "locale", "avatar_url", "bio"}
            filtered_updates = {
                k: v for k, v in updates.items()
                if k in allowed_fields and v is not None
            }

            if not filtered_updates:
                return UpdateUserResult(success=True, user=user)

            # Apply updates using entity method
            user.update_profile(**filtered_updates)

            # Persist changes
            updated_user = await self._user_repo.update(user)

            # Update statistics
            self._stats["users_updated"] += 1

            logger.info(
                "user_updated",
                user_id=str(user_id),
                fields=list(filtered_updates.keys()),
            )

            return UpdateUserResult(success=True, user=updated_user)

        except ValueError as e:
            return UpdateUserResult(error=str(e), error_code="VALIDATION_ERROR")
        except Exception as e:
            logger.error("user_update_failed", user_id=str(user_id), error=str(e))
            return UpdateUserResult(
                error=f"Failed to update user: {str(e)}",
                error_code="UPDATE_FAILED",
            )

    async def delete_user(
        self,
        user_id: UUID,
        reason: str | None = None,
    ) -> DeleteUserResult:
        """
        Soft delete user account (GDPR compliance).

        Business Rules:
        - Uses soft delete to preserve audit trail
        - Email is anonymized
        - Account status set to INACTIVE
        - All sessions should be revoked by caller

        Args:
            user_id: User identifier
            reason: Optional deletion reason for audit

        Returns:
            DeleteUserResult indicating success or error
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return DeleteUserResult(
                    error="User not found",
                    error_code="USER_NOT_FOUND",
                )

            # Perform soft delete using entity method
            user.soft_delete()

            # Persist changes
            await self._user_repo.update(user)

            # Update statistics
            self._stats["users_deleted"] += 1

            # Publish domain event
            if self._event_publisher:
                await self._event_publisher.publish_user_deleted(user_id, reason)

            logger.info(
                "user_deleted",
                user_id=str(user_id),
                reason=reason,
            )

            return DeleteUserResult(success=True)

        except Exception as e:
            logger.error("user_deletion_failed", user_id=str(user_id), error=str(e))
            return DeleteUserResult(
                error=f"Failed to delete user: {str(e)}",
                error_code="DELETE_FAILED",
            )

    # --- Password Management ---

    async def change_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password_hash: str,
    ) -> PasswordChangeResult:
        """
        Change user password.

        Business Rules:
        - Current password must be verified
        - New password must already be hashed
        - All sessions should be revoked by caller

        Args:
            user_id: User identifier
            current_password: Current password (plaintext)
            new_password_hash: New password (pre-hashed)

        Returns:
            PasswordChangeResult indicating success or error
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return PasswordChangeResult(
                    error="User not found",
                    error_code="USER_NOT_FOUND",
                )

            # Verify current password
            if self._password_service:
                result = self._password_service.verify_password(
                    current_password,
                    user.password_hash,
                )
                if not result.is_valid:
                    return PasswordChangeResult(
                        error="Current password is incorrect",
                        error_code="INVALID_PASSWORD",
                    )
            else:
                # Fallback: simple comparison (not recommended for production)
                return PasswordChangeResult(
                    error="Password service not configured",
                    error_code="SERVICE_UNAVAILABLE",
                )

            # Update password
            user.password_hash = new_password_hash
            user.updated_at = datetime.now(timezone.utc)

            # Persist changes
            await self._user_repo.update(user)

            # Update statistics
            self._stats["password_changes"] += 1

            logger.info("password_changed", user_id=str(user_id))

            return PasswordChangeResult(success=True)

        except Exception as e:
            logger.error("password_change_failed", user_id=str(user_id), error=str(e))
            return PasswordChangeResult(
                error=f"Failed to change password: {str(e)}",
                error_code="CHANGE_FAILED",
            )

    # --- Email Verification ---

    async def verify_email(
        self,
        user_id: UUID,
        token: str,
    ) -> EmailVerificationResult:
        """
        Verify user email address.

        Business Rules:
        - Token must match stored verification token
        - User status transitions to ACTIVE
        - Verification token is cleared after use

        Args:
            user_id: User identifier
            token: Verification token

        Returns:
            EmailVerificationResult indicating success or error
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return EmailVerificationResult(
                    error="User not found",
                    error_code="USER_NOT_FOUND",
                )

            if user.email_verified:
                return EmailVerificationResult(
                    error="Email already verified",
                    error_code="ALREADY_VERIFIED",
                )

            # Verify token using constant-time comparison
            if not user.email_verification_token or not secrets.compare_digest(
                token, user.email_verification_token
            ):
                return EmailVerificationResult(
                    error="Invalid verification token",
                    error_code="INVALID_TOKEN",
                )

            # Update user state
            user.email_verified = True
            user.email_verification_token = None
            user.status = AccountStatus.ACTIVE
            user.updated_at = datetime.now(timezone.utc)

            # Persist changes
            await self._user_repo.update(user)

            # Update statistics
            self._stats["email_verifications"] += 1

            # Publish domain event
            if self._event_publisher:
                await self._event_publisher.publish_email_verified(user_id)

            logger.info("email_verified", user_id=str(user_id))

            return EmailVerificationResult(success=True)

        except Exception as e:
            logger.error("email_verification_failed", user_id=str(user_id), error=str(e))
            return EmailVerificationResult(
                error=f"Failed to verify email: {str(e)}",
                error_code="VERIFICATION_FAILED",
            )

    async def resend_verification_email(
        self,
        user_id: UUID,
    ) -> tuple[str | None, str | None]:
        """
        Generate new verification token and send verification email.

        Args:
            user_id: User identifier

        Returns:
            Tuple of (new_token, error_message)
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return None, "User not found"

            if user.email_verified:
                return None, "Email already verified"

            # Generate new token
            new_token = secrets.token_urlsafe(32)
            user.email_verification_token = new_token
            user.updated_at = datetime.now(timezone.utc)

            # Persist changes
            await self._user_repo.update(user)

            logger.info("verification_token_regenerated", user_id=str(user_id))

            # Send verification email via notification service
            email_sent = await self._send_verification_email(
                email=user.email,
                display_name=user.display_name,
                token=new_token,
            )

            if not email_sent:
                logger.warning(
                    "verification_email_send_failed",
                    user_id=str(user_id),
                    email=user.email,
                )
                # Token is saved, email failed - user can retry or use token directly

            return new_token, None

        except Exception as e:
            logger.error("verification_resend_failed", user_id=str(user_id), error=str(e))
            return None, str(e)

    async def _send_verification_email(
        self,
        email: str,
        display_name: str,
        token: str,
    ) -> bool:
        """
        Send verification email via notification service.

        Args:
            email: Recipient email address
            display_name: User's display name
            token: Verification token

        Returns:
            True if email was sent successfully, False otherwise
        """
        verification_link = (
            f"{self._integration_settings.frontend_url}/verify-email?token={token}"
        )

        url = f"{self._integration_settings.notification_service_url}/api/v1/notifications/email"
        timeout = httpx.Timeout(self._integration_settings.request_timeout)

        payload = {
            "to_email": email,
            "template_type": "email_verification",
            "variables": {
                "display_name": display_name,
                "verification_link": verification_link,
                "expiry_hours": self._verification_token_expiry_hours,
            },
            "priority": "high",
        }

        try:
            headers = self._get_service_auth_headers(ServiceIdentity.NOTIFICATION.value)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                logger.info(
                    "verification_email_sent",
                    email=email,
                    template="email_verification",
                )
                return True

        except httpx.TimeoutException:
            logger.warning(
                "notification_service_timeout",
                email=email,
                url=url,
            )
            return False
        except httpx.HTTPStatusError as e:
            logger.warning(
                "notification_service_http_error",
                email=email,
                status_code=e.response.status_code,
            )
            return False
        except Exception as e:
            logger.error(
                "verification_email_send_error",
                email=email,
                error=str(e),
            )
            return False

    # --- Login Tracking ---

    async def record_login_attempt(
        self,
        user_id: UUID,
        success: bool,
    ) -> None:
        """
        Record login attempt for security tracking.

        Args:
            user_id: User identifier
            success: Whether login was successful
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return

            user.record_login_attempt(
                success=success,
                max_attempts=self._max_login_attempts,
                lockout_duration_minutes=self._lockout_duration_minutes,
            )

            await self._user_repo.update(user)

        except Exception as e:
            logger.error(
                "login_attempt_record_failed",
                user_id=str(user_id),
                error=str(e),
            )

    # --- Preferences Management ---

    async def get_preferences(self, user_id: UUID) -> UserPreferences | None:
        """
        Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences or None if not found
        """
        return await self._prefs_repo.get_by_user_id(user_id)

    async def update_preferences(
        self,
        user_id: UUID,
        **updates: Any,
    ) -> UpdatePreferencesResult:
        """
        Update user preferences.

        Args:
            user_id: User identifier
            **updates: Preference field updates

        Returns:
            UpdatePreferencesResult with updated preferences or error
        """
        try:
            preferences = await self._prefs_repo.get_by_user_id(user_id)
            if not preferences:
                # Create default preferences if not exists
                preferences = UserPreferences(user_id=user_id)
                await self._prefs_repo.save(preferences)

            # Apply updates
            filtered_updates = {k: v for k, v in updates.items() if v is not None}
            if filtered_updates:
                preferences.update(**filtered_updates)

            # Persist changes
            updated_prefs = await self._prefs_repo.update(preferences)

            logger.info(
                "preferences_updated",
                user_id=str(user_id),
                fields=list(filtered_updates.keys()),
            )

            return UpdatePreferencesResult(success=True, preferences=updated_prefs)

        except ValueError as e:
            return UpdatePreferencesResult(error=str(e), error_code="VALIDATION_ERROR")
        except Exception as e:
            logger.error("preferences_update_failed", user_id=str(user_id), error=str(e))
            return UpdatePreferencesResult(
                error=f"Failed to update preferences: {str(e)}",
                error_code="UPDATE_FAILED",
            )

    # --- Consent Management ---

    async def get_consent_records(self, user_id: UUID) -> list[ConsentRecord]:
        """
        Get all consent records for user.

        Args:
            user_id: User identifier

        Returns:
            List of consent records
        """
        if not self._consent_repo:
            return []
        return await self._consent_repo.get_by_user_id(user_id)

    async def record_consent(
        self,
        user_id: UUID,
        consent_type: ConsentType,
        granted: bool,
        version: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ConsentResult:
        """
        Record consent decision.

        Business Rules:
        - Creates immutable consent record
        - Supports both grant and revoke
        - Includes audit metadata

        Args:
            user_id: User identifier
            consent_type: Type of consent
            granted: Whether consent was granted
            version: Version of consent document
            ip_address: Client IP for audit
            user_agent: Client user agent for audit

        Returns:
            ConsentResult with recorded consent or error
        """
        if not self._consent_repo:
            return ConsentResult(
                error="Consent repository not configured",
                error_code="SERVICE_UNAVAILABLE",
            )

        try:
            from uuid import uuid4

            consent = ConsentRecord(
                consent_id=uuid4(),
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                version=version,
                ip_address=ip_address,
                user_agent=user_agent,
            )

            saved_consent = await self._consent_repo.save(consent)

            logger.info(
                "consent_recorded",
                user_id=str(user_id),
                consent_type=consent_type.value,
                granted=granted,
            )

            return ConsentResult(success=True, consent=saved_consent)

        except Exception as e:
            logger.error("consent_record_failed", user_id=str(user_id), error=str(e))
            return ConsentResult(
                error=f"Failed to record consent: {str(e)}",
                error_code="RECORD_FAILED",
            )

    async def check_consent(
        self,
        user_id: UUID,
        consent_type: ConsentType,
    ) -> bool:
        """
        Check if user has granted specific consent.

        Args:
            user_id: User identifier
            consent_type: Type of consent to check

        Returns:
            True if consent is active, False otherwise
        """
        if not self._consent_repo:
            return False

        records = await self._consent_repo.get_by_user_id(user_id)
        for record in reversed(records):  # Check most recent first
            if record.consent_type == consent_type:
                return record.is_active()
        return False

    # --- On-Call Clinicians ---

    async def get_on_call_clinicians(self) -> list[dict[str, Any]]:
        """
        Get list of currently on-call clinicians for crisis notifications.

        Returns:
            List of clinician contact information dictionaries containing:
            - user_id: Clinician user ID
            - display_name: Clinician name
            - email: Contact email
            - phone_number: Contact phone (if available)
        """
        try:
            clinicians = await self._user_repo.find_on_call_clinicians()

            return [
                {
                    "user_id": str(clinician.user_id),
                    "display_name": clinician.display_name,
                    "email": clinician.email,
                    "phone_number": clinician.phone_number,
                }
                for clinician in clinicians
            ]
        except Exception as e:
            logger.error("get_on_call_clinicians_failed", error=str(e))
            return []

    async def set_on_call_status(
        self,
        user_id: UUID,
        is_on_call: bool,
    ) -> UpdateUserResult:
        """
        Set a clinician's on-call status.

        Args:
            user_id: Clinician user ID
            is_on_call: Whether the clinician is on-call

        Returns:
            UpdateUserResult indicating success or error
        """
        try:
            user = await self._user_repo.get_by_id(user_id)
            if not user:
                return UpdateUserResult(
                    error="User not found",
                    error_code="USER_NOT_FOUND",
                )

            if user.role != UserRole.CLINICIAN:
                return UpdateUserResult(
                    error="Only clinicians can be set on-call",
                    error_code="INVALID_ROLE",
                )

            user.is_on_call = is_on_call
            user.updated_at = datetime.now(timezone.utc)

            updated_user = await self._user_repo.update(user)

            logger.info(
                "on_call_status_changed",
                user_id=str(user_id),
                is_on_call=is_on_call,
            )

            return UpdateUserResult(success=True, user=updated_user)

        except Exception as e:
            logger.error("set_on_call_status_failed", user_id=str(user_id), error=str(e))
            return UpdateUserResult(
                error=f"Failed to update on-call status: {str(e)}",
                error_code="UPDATE_FAILED",
            )

    # --- Clinician-Patient Assignments ---

    async def assign_patient(
        self, clinician_id: UUID, patient_id: UUID, requesting_user_id: UUID, requesting_role: str,
    ) -> UpdateUserResult:
        """Assign a patient to a clinician. Only admins can assign."""
        if requesting_role not in ("admin", "system"):
            return UpdateUserResult(error="Only admins can assign patients", error_code="FORBIDDEN")

        clinician = await self._user_repo.get_by_id(clinician_id)
        if not clinician or clinician.role != UserRole.CLINICIAN:
            return UpdateUserResult(error="Clinician not found", error_code="NOT_FOUND")

        patient = await self._user_repo.get_by_id(patient_id)
        if not patient:
            return UpdateUserResult(error="Patient not found", error_code="NOT_FOUND")

        await self._user_repo.assign_patient_to_clinician(clinician_id, patient_id)
        logger.info("patient_assigned", clinician_id=str(clinician_id), patient_id=str(patient_id))
        return UpdateUserResult(success=True)

    async def unassign_patient(
        self, clinician_id: UUID, patient_id: UUID, requesting_user_id: UUID, requesting_role: str,
    ) -> UpdateUserResult:
        """Unassign a patient from a clinician. Only admins can unassign."""
        if requesting_role not in ("admin", "system"):
            return UpdateUserResult(error="Only admins can unassign patients", error_code="FORBIDDEN")

        success = await self._user_repo.unassign_patient_from_clinician(clinician_id, patient_id)
        if not success:
            return UpdateUserResult(error="Assignment not found", error_code="NOT_FOUND")

        logger.info("patient_unassigned", clinician_id=str(clinician_id), patient_id=str(patient_id))
        return UpdateUserResult(success=True)

    async def verify_clinician_patient_access(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Verify a clinician has access to a specific patient.

        Returns True if:
        - The clinician is assigned to the patient, OR
        - The user is an admin/system role (checked by caller)
        """
        return await self._user_repo.is_patient_assigned_to_clinician(
            clinician_id, patient_id,
        )

    async def get_clinician_patients(self, clinician_id: UUID) -> list[UUID]:
        """Get all patient IDs assigned to a clinician."""
        return await self._user_repo.get_assigned_patients(clinician_id)

    # --- Progress Tracking ---

    async def get_progress(self, user_id: UUID) -> UserProgress | None:
        """
        Get user progress summary by fetching data from therapy service.

        Args:
            user_id: User identifier

        Returns:
            UserProgress summary or None if user not found
        """
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return None

        # Fetch actual progress from therapy service
        therapy_progress = await self._fetch_therapy_progress(user_id)

        return UserProgress(
            user_id=user_id,
            total_sessions=therapy_progress.get("total_sessions", 0),
            completed_assessments=therapy_progress.get("assessments_completed", 0),
            streak_days=therapy_progress.get("streak_days", 0),
            total_minutes=therapy_progress.get("total_minutes", 0),
            mood_trend=therapy_progress.get("mood_trend", "stable"),
            engagement_score=therapy_progress.get("engagement_score", 0.5),
        )

    async def _fetch_therapy_progress(self, user_id: UUID) -> dict[str, Any]:
        """
        Fetch therapy progress data from the therapy service.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with therapy progress data, or empty dict on failure
        """
        url = f"{self._integration_settings.therapy_service_url}/api/v1/users/{user_id}/progress"
        timeout = httpx.Timeout(self._integration_settings.request_timeout)

        try:
            headers = self._get_service_auth_headers(ServiceIdentity.THERAPY.value)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 404:
                    logger.debug(
                        "no_therapy_progress_found",
                        user_id=str(user_id),
                    )
                    return {}

                response.raise_for_status()
                data = response.json()

                logger.debug(
                    "therapy_progress_fetched",
                    user_id=str(user_id),
                    total_sessions=data.get("total_sessions", 0),
                )
                return data

        except httpx.TimeoutException:
            logger.warning(
                "therapy_service_timeout",
                user_id=str(user_id),
                url=url,
            )
            return {}
        except httpx.HTTPStatusError as e:
            logger.warning(
                "therapy_service_http_error",
                user_id=str(user_id),
                status_code=e.response.status_code,
            )
            return {}
        except Exception as e:
            logger.error(
                "therapy_progress_fetch_failed",
                user_id=str(user_id),
                error=str(e),
            )
            return {}

    # --- Statistics ---

    def get_statistics(self) -> dict[str, int]:
        """Get service statistics."""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset service statistics."""
        for key in self._stats:
            self._stats[key] = 0

"""
Solace-AI User Service - User Domain Service.
Business logic for user management, authentication, and preferences.
"""
from __future__ import annotations
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

if TYPE_CHECKING:
    from ..infrastructure.repository import UserRepository, UserPreferencesRepository
    from ..auth import SessionManager

logger = structlog.get_logger(__name__)


class UserRole(str, Enum):
    """User roles in the system."""
    USER = "user"
    PREMIUM = "premium"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class AccountStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class UserServiceSettings(BaseSettings):
    """User service configuration."""
    password_min_length: int = Field(default=8)
    password_require_special: bool = Field(default=True)
    max_login_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=30)
    email_verification_expiry_hours: int = Field(default=24)
    session_timeout_minutes: int = Field(default=60)
    enable_audit_logging: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="USER_SERVICE_", env_file=".env", extra="ignore")


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class User(BaseModel):
    """User entity model."""
    user_id: UUID = Field(default_factory=uuid4)
    email: str = Field(..., min_length=5, max_length=255)
    password_hash: str
    display_name: str
    role: UserRole = Field(default=UserRole.USER)
    status: AccountStatus = Field(default=AccountStatus.PENDING_VERIFICATION)
    timezone: str = Field(default="UTC")
    locale: str = Field(default="en-US")
    avatar_url: str | None = None
    bio: str | None = None
    email_verified: bool = Field(default=False)
    email_verification_token: str | None = None
    login_attempts: int = Field(default=0)
    locked_until: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = None
    deleted_at: datetime | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v.lower()


class UserPreferences(BaseModel):
    """User preferences model."""
    user_id: UUID
    notification_email: bool = True
    notification_sms: bool = False
    notification_push: bool = True
    notification_channels: list[str] = Field(default_factory=lambda: ["email", "in_app"])
    session_reminders: bool = True
    progress_updates: bool = True
    marketing_emails: bool = False
    data_sharing_research: bool = False
    data_sharing_improvement: bool = True
    theme: str = "system"
    language: str = "en"
    accessibility_high_contrast: bool = False
    accessibility_large_text: bool = False
    accessibility_screen_reader: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProgress(BaseModel):
    """User progress summary model."""
    user_id: UUID
    total_sessions: int = 0
    completed_assessments: int = 0
    streak_days: int = 0
    total_minutes: int = 0
    last_session: datetime | None = None
    mood_trend: str = "stable"
    engagement_score: float = 0.0


class ConsentRecord(BaseModel):
    """Consent record model."""
    consent_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    consent_type: str
    granted: bool
    version: str
    granted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str | None = None
    user_agent: str | None = None


class UserSession(BaseModel):
    """User session model."""
    session_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str | None = None
    user_agent: str | None = None
    is_active: bool = True


@dataclass
class CreateUserResult:
    """Result of user creation."""
    user: User | None = None
    error: str | None = None


@dataclass
class UpdateUserResult:
    """Result of user update."""
    user: User | None = None
    error: str | None = None


@dataclass
class UpdatePreferencesResult:
    """Result of preferences update."""
    preferences: UserPreferences | None = None
    error: str | None = None


@dataclass
class ConsentResult:
    """Result of consent recording."""
    consent: ConsentRecord | None = None
    error: str | None = None


@dataclass
class DeleteUserResult:
    """Result of user deletion."""
    success: bool = False
    error: str | None = None


@dataclass
class PasswordChangeResult:
    """Result of password change."""
    success: bool = False
    error: str | None = None


@dataclass
class EmailVerificationResult:
    """Result of email verification."""
    success: bool = False
    error: str | None = None


class UserService:
    """User domain service handling business logic."""

    def __init__(
        self,
        settings: UserServiceSettings | None = None,
        user_repository: UserRepository | None = None,
        preferences_repository: UserPreferencesRepository | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self._settings = settings or UserServiceSettings()
        self._user_repo = user_repository
        self._prefs_repo = preferences_repository
        self._session_mgr = session_manager
        self._initialized = False
        self._stats = {"users_created": 0, "users_deleted": 0, "logins": 0, "password_changes": 0}

    async def initialize(self) -> None:
        """Initialize the user service."""
        logger.info("user_service_initializing")
        self._initialized = True
        logger.info("user_service_initialized", settings={"audit_logging": self._settings.enable_audit_logging})

    async def shutdown(self) -> None:
        """Shutdown the user service."""
        logger.info("user_service_shutting_down", stats=self._stats)
        self._initialized = False

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt (industry-standard secure hashing)."""
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against bcrypt hash."""
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        try:
            return pwd_context.verify(password, password_hash)
        except Exception:
            return False

    def _validate_password(self, password: str) -> str | None:
        """Validate password strength. Returns error message if invalid."""
        if len(password) < self._settings.password_min_length:
            return f"Password must be at least {self._settings.password_min_length} characters"
        if self._settings.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return "Password must contain at least one special character"
        return None

    async def create_user(self, email: str, password: str, display_name: str, timezone: str = "UTC", locale: str = "en-US") -> CreateUserResult:
        """Create a new user account."""
        if not self._user_repo:
            return CreateUserResult(error="Repository not configured")
        existing = await self._user_repo.get_by_email(email)
        if existing:
            return CreateUserResult(error="Email already registered")
        password_error = self._validate_password(password)
        if password_error:
            return CreateUserResult(error=password_error)
        password_hash = self._hash_password(password)
        verification_token = secrets.token_urlsafe(32)
        user = User(
            email=email,
            password_hash=password_hash,
            display_name=display_name,
            timezone=timezone,
            locale=locale,
            email_verification_token=verification_token,
        )
        saved_user = await self._user_repo.save(user)
        if self._prefs_repo:
            preferences = UserPreferences(user_id=saved_user.user_id)
            await self._prefs_repo.save(preferences)
        self._stats["users_created"] += 1
        logger.info("user_created", user_id=str(saved_user.user_id), email=email)
        return CreateUserResult(user=saved_user)

    async def get_user(self, user_id: UUID) -> User | None:
        """Get user by ID."""
        if not self._user_repo:
            return None
        return await self._user_repo.get_by_id(user_id)

    async def update_user(self, user_id: UUID, updates: dict[str, Any]) -> UpdateUserResult:
        """Update user profile."""
        if not self._user_repo:
            return UpdateUserResult(error="Repository not configured")
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return UpdateUserResult(error="User not found")
        allowed_fields = {"display_name", "timezone", "locale", "avatar_url", "bio"}
        for key, value in updates.items():
            if key in allowed_fields and value is not None:
                setattr(user, key, value)
        user.updated_at = datetime.now(timezone.utc)
        updated_user = await self._user_repo.update(user)
        logger.info("user_updated", user_id=str(user_id), fields=list(updates.keys()))
        return UpdateUserResult(user=updated_user)

    async def delete_user(self, user_id: UUID, password: str, reason: str | None = None) -> DeleteUserResult:
        """Delete user account (soft delete for GDPR compliance)."""
        if not self._user_repo:
            return DeleteUserResult(error="Repository not configured")
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return DeleteUserResult(error="User not found")
        if not self._verify_password(password, user.password_hash):
            return DeleteUserResult(error="Invalid password")
        user.deleted_at = datetime.now(timezone.utc)
        user.status = AccountStatus.INACTIVE
        user.email = f"deleted_{user.user_id}@deleted.solace-ai.com"
        await self._user_repo.update(user)
        if self._session_mgr:
            await self._session_mgr.revoke_all_sessions(user_id)
        self._stats["users_deleted"] += 1
        logger.info("user_deleted", user_id=str(user_id), reason=reason)
        return DeleteUserResult(success=True)

    async def get_preferences(self, user_id: UUID) -> UserPreferences | None:
        """Get user preferences."""
        if not self._prefs_repo:
            return None
        return await self._prefs_repo.get_by_user(user_id)

    async def update_preferences(self, user_id: UUID, updates: dict[str, Any]) -> UpdatePreferencesResult:
        """Update user preferences."""
        if not self._prefs_repo:
            return UpdatePreferencesResult(error="Repository not configured")
        prefs = await self._prefs_repo.get_by_user(user_id)
        if not prefs:
            return UpdatePreferencesResult(error="User preferences not found")
        for key, value in updates.items():
            if hasattr(prefs, key) and value is not None:
                setattr(prefs, key, value)
        prefs.updated_at = datetime.now(timezone.utc)
        updated_prefs = await self._prefs_repo.update(prefs)
        logger.info("preferences_updated", user_id=str(user_id), fields=list(updates.keys()))
        return UpdatePreferencesResult(preferences=updated_prefs)

    async def get_progress(self, user_id: UUID) -> UserProgress | None:
        """Get user progress summary."""
        if not self._user_repo:
            return None
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return None
        return UserProgress(user_id=user_id, total_sessions=0, completed_assessments=0, streak_days=0, total_minutes=0, mood_trend="stable", engagement_score=0.5)

    async def get_consent_records(self, user_id: UUID) -> list[ConsentRecord]:
        """Get user consent records."""
        if not self._user_repo:
            return []
        return await self._user_repo.get_consent_records(user_id)

    async def record_consent(self, user_id: UUID, consent_type: str, granted: bool, version: str, ip_address: str | None = None, user_agent: str | None = None) -> ConsentResult:
        """Record user consent."""
        if not self._user_repo:
            return ConsentResult(error="Repository not configured")
        consent = ConsentRecord(user_id=user_id, consent_type=consent_type, granted=granted, version=version, ip_address=ip_address, user_agent=user_agent)
        saved_consent = await self._user_repo.save_consent(consent)
        logger.info("consent_recorded", user_id=str(user_id), consent_type=consent_type, granted=granted)
        return ConsentResult(consent=saved_consent)

    async def get_user_sessions(self, user_id: UUID, active_only: bool = True) -> list[UserSession]:
        """Get user sessions."""
        if not self._session_mgr:
            return []
        return await self._session_mgr.get_user_sessions(user_id, active_only)

    async def revoke_session(self, user_id: UUID, session_id: UUID) -> bool:
        """Revoke a specific user session."""
        if not self._session_mgr:
            return False
        return await self._session_mgr.revoke_session(user_id, session_id)

    async def change_password(self, user_id: UUID, current_password: str, new_password: str) -> PasswordChangeResult:
        """Change user password."""
        if not self._user_repo:
            return PasswordChangeResult(error="Repository not configured")
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return PasswordChangeResult(error="User not found")
        if not self._verify_password(current_password, user.password_hash):
            return PasswordChangeResult(error="Current password is incorrect")
        password_error = self._validate_password(new_password)
        if password_error:
            return PasswordChangeResult(error=password_error)
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.now(timezone.utc)
        await self._user_repo.update(user)
        if self._session_mgr:
            await self._session_mgr.revoke_all_sessions(user_id)
        self._stats["password_changes"] += 1
        logger.info("password_changed", user_id=str(user_id))
        return PasswordChangeResult(success=True)

    async def verify_email(self, user_id: UUID, token: str) -> EmailVerificationResult:
        """Verify user email address."""
        if not self._user_repo:
            return EmailVerificationResult(error="Repository not configured")
        user = await self._user_repo.get_by_id(user_id)
        if not user:
            return EmailVerificationResult(error="User not found")
        if user.email_verified:
            return EmailVerificationResult(error="Email already verified")
        if not user.email_verification_token or not secrets.compare_digest(token, user.email_verification_token):
            return EmailVerificationResult(error="Invalid verification token")
        user.email_verified = True
        user.email_verification_token = None
        user.status = AccountStatus.ACTIVE
        user.updated_at = datetime.now(timezone.utc)
        await self._user_repo.update(user)
        logger.info("email_verified", user_id=str(user_id))
        return EmailVerificationResult(success=True)

    async def get_status(self) -> dict[str, Any]:
        """Get service status."""
        return {"status": "operational" if self._initialized else "initializing", "initialized": self._initialized, "statistics": self._stats}

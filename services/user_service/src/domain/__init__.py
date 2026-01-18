"""
Solace-AI User Service - Domain Layer.
Business logic and domain models for user management.
"""
from .service import (
    UserService,
    UserServiceSettings,
    User,
    UserPreferences,
    UserProgress,
    ConsentRecord,
    UserSession,
    UserRole,
    AccountStatus,
    CreateUserResult,
    UpdateUserResult,
    UpdatePreferencesResult,
    ConsentResult,
    DeleteUserResult,
    PasswordChangeResult,
    EmailVerificationResult,
)

__all__ = [
    "UserService",
    "UserServiceSettings",
    "User",
    "UserPreferences",
    "UserProgress",
    "ConsentRecord",
    "UserSession",
    "UserRole",
    "AccountStatus",
    "CreateUserResult",
    "UpdateUserResult",
    "UpdatePreferencesResult",
    "ConsentResult",
    "DeleteUserResult",
    "PasswordChangeResult",
    "EmailVerificationResult",
]

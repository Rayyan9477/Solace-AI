"""
Solace-AI User Service - Infrastructure Layer.

Infrastructure implementations for external dependencies:
- JWTService: JWT token generation and verification
- PasswordService: Password hashing with argon2/bcrypt dual support
- TokenService: Time-limited tokens for email verification and password reset
- EncryptionService: Field-level encryption for PII/PHI (HIPAA compliance)
"""
from __future__ import annotations

from .jwt_service import (
    JWTService,
    JWTConfig,
    TokenType,
    TokenPayload,
    TokenPair,
    JWTError,
    TokenExpiredError,
    TokenInvalidError,
    create_jwt_service,
)
from .password_service import (
    PasswordService,
    PasswordConfig,
    HashAlgorithm,
    PasswordVerificationResult,
    create_password_service,
)
from .token_service import (
    TokenService,
    TokenConfig,
    TokenPurpose,
    TokenData,
    TokenError as VerificationTokenError,
    TokenExpiredError as VerificationTokenExpiredError,
    TokenInvalidError as VerificationTokenInvalidError,
    create_token_service,
)
from .encryption_service import (
    EncryptionService,
    EncryptionConfig,
    EncryptionError,
    DecryptionError,
    create_encryption_service,
)

__all__ = [
    # JWT Service
    "JWTService",
    "JWTConfig",
    "TokenType",
    "TokenPayload",
    "TokenPair",
    "JWTError",
    "TokenExpiredError",
    "TokenInvalidError",
    "create_jwt_service",
    # Password Service
    "PasswordService",
    "PasswordConfig",
    "HashAlgorithm",
    "PasswordVerificationResult",
    "create_password_service",
    # Token Service (Verification Tokens)
    "TokenService",
    "TokenConfig",
    "TokenPurpose",
    "TokenData",
    "VerificationTokenError",
    "VerificationTokenExpiredError",
    "VerificationTokenInvalidError",
    "create_token_service",
    # Encryption Service
    "EncryptionService",
    "EncryptionConfig",
    "EncryptionError",
    "DecryptionError",
    "create_encryption_service",
]

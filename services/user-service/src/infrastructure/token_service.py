"""
Solace-AI User Service - Token Service.

Provides secure token generation and verification for:
- Email verification
- Password reset
- Account activation
- Two-factor authentication (future)

Uses Fernet symmetric encryption for time-limited, tamper-proof tokens.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import UUID

import structlog
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TokenPurpose(str, Enum):
    """Token purpose types."""
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    ACCOUNT_ACTIVATION = "account_activation"
    TWO_FACTOR_AUTH = "two_factor_auth"


class TokenConfig(BaseModel):
    """Token service configuration."""
    encryption_key: bytes = Field(..., description="Fernet encryption key")
    email_verification_expire_hours: int = Field(default=24, description="Email verification token lifetime")
    password_reset_expire_hours: int = Field(default=1, description="Password reset token lifetime")
    activation_expire_days: int = Field(default=7, description="Account activation token lifetime")


@dataclass
class TokenData:
    """Data contained in verification token."""
    user_id: UUID
    email: str
    purpose: TokenPurpose
    created_at: datetime
    expires_at: datetime

    def to_string(self) -> str:
        """Convert token data to string for encryption."""
        return "|".join([
            str(self.user_id),
            self.email,
            self.purpose.value,
            str(int(self.created_at.timestamp())),
            str(int(self.expires_at.timestamp())),
        ])

    @classmethod
    def from_string(cls, data_str: str) -> TokenData:
        """Parse token data from string."""
        parts = data_str.split("|")
        if len(parts) != 5:
            raise ValueError("Invalid token data format")

        return cls(
            user_id=UUID(parts[0]),
            email=parts[1],
            purpose=TokenPurpose(parts[2]),
            created_at=datetime.fromtimestamp(int(parts[3]), tz=timezone.utc),
            expires_at=datetime.fromtimestamp(int(parts[4]), tz=timezone.utc),
        )


class TokenError(Exception):
    """Base exception for token errors."""
    pass


class TokenExpiredError(TokenError):
    """Raised when token has expired."""
    pass


class TokenInvalidError(TokenError):
    """Raised when token is invalid or tampered."""
    pass


class TokenService:
    """
    Token Service for secure, time-limited tokens.

    Features:
    - Symmetric encryption (Fernet)
    - Time-limited tokens
    - Tamper-proof (encryption prevents modification)
    - Purpose-specific tokens
    - URL-safe encoding
    """

    def __init__(self, config: TokenConfig):
        """
        Initialize token service.

        Args:
            config: Token configuration
        """
        self.config = config
        self.cipher = Fernet(config.encryption_key)
        self.logger = structlog.get_logger(__name__)

    def generate_email_verification_token(
        self,
        user_id: UUID,
        email: str,
    ) -> str:
        """
        Generate email verification token.

        Args:
            user_id: User ID
            email: Email address to verify

        Returns:
            URL-safe verification token
        """
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=self.config.email_verification_expire_hours
        )

        token_data = TokenData(
            user_id=user_id,
            email=email,
            purpose=TokenPurpose.EMAIL_VERIFICATION,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

        token = self._encrypt_token_data(token_data)

        self.logger.info(
            "email_verification_token_generated",
            user_id=str(user_id),
            email=email,
            expires_at=expires_at.isoformat(),
        )

        return token

    def verify_email_verification_token(
        self,
        token: str,
    ) -> tuple[UUID, str]:
        """
        Verify email verification token and extract data.

        Args:
            token: Verification token

        Returns:
            Tuple of (user_id, email)

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or tampered
        """
        token_data = self._decrypt_and_validate_token(
            token,
            expected_purpose=TokenPurpose.EMAIL_VERIFICATION,
        )

        self.logger.info(
            "email_verification_token_verified",
            user_id=str(token_data.user_id),
            email=token_data.email,
        )

        return token_data.user_id, token_data.email

    def generate_password_reset_token(
        self,
        user_id: UUID,
        email: str,
    ) -> str:
        """
        Generate password reset token (short-lived, 1 hour).

        Args:
            user_id: User ID
            email: User email

        Returns:
            URL-safe password reset token
        """
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=self.config.password_reset_expire_hours
        )

        token_data = TokenData(
            user_id=user_id,
            email=email,
            purpose=TokenPurpose.PASSWORD_RESET,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

        token = self._encrypt_token_data(token_data)

        self.logger.info(
            "password_reset_token_generated",
            user_id=str(user_id),
            email=email,
            expires_at=expires_at.isoformat(),
        )

        return token

    def verify_password_reset_token(
        self,
        token: str,
    ) -> tuple[UUID, str]:
        """
        Verify password reset token and extract data.

        Args:
            token: Password reset token

        Returns:
            Tuple of (user_id, email)

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or tampered
        """
        token_data = self._decrypt_and_validate_token(
            token,
            expected_purpose=TokenPurpose.PASSWORD_RESET,
        )

        self.logger.info(
            "password_reset_token_verified",
            user_id=str(token_data.user_id),
            email=token_data.email,
        )

        return token_data.user_id, token_data.email

    def generate_activation_token(
        self,
        user_id: UUID,
        email: str,
    ) -> str:
        """
        Generate account activation token (long-lived, 7 days).

        Args:
            user_id: User ID
            email: User email

        Returns:
            URL-safe activation token
        """
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=self.config.activation_expire_days
        )

        token_data = TokenData(
            user_id=user_id,
            email=email,
            purpose=TokenPurpose.ACCOUNT_ACTIVATION,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

        token = self._encrypt_token_data(token_data)

        self.logger.info(
            "activation_token_generated",
            user_id=str(user_id),
            email=email,
            expires_at=expires_at.isoformat(),
        )

        return token

    def verify_activation_token(
        self,
        token: str,
    ) -> tuple[UUID, str]:
        """
        Verify account activation token and extract data.

        Args:
            token: Activation token

        Returns:
            Tuple of (user_id, email)

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or tampered
        """
        token_data = self._decrypt_and_validate_token(
            token,
            expected_purpose=TokenPurpose.ACCOUNT_ACTIVATION,
        )

        self.logger.info(
            "activation_token_verified",
            user_id=str(token_data.user_id),
            email=token_data.email,
        )

        return token_data.user_id, token_data.email

    def _encrypt_token_data(self, token_data: TokenData) -> str:
        """
        Encrypt token data using Fernet.

        Args:
            token_data: Token data to encrypt

        Returns:
            URL-safe encrypted token
        """
        data_str = token_data.to_string()
        encrypted = self.cipher.encrypt(data_str.encode())
        # Fernet tokens are already URL-safe, but we can further encode if needed
        return encrypted.decode()

    def _decrypt_and_validate_token(
        self,
        token: str,
        expected_purpose: TokenPurpose,
    ) -> TokenData:
        """
        Decrypt and validate token.

        Args:
            token: Encrypted token
            expected_purpose: Expected token purpose

        Returns:
            Decrypted and validated TokenData

        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or purpose mismatch
        """
        try:
            # Decrypt token
            decrypted = self.cipher.decrypt(token.encode())
            data_str = decrypted.decode()

            # Parse token data
            token_data = TokenData.from_string(data_str)

            # Validate purpose
            if token_data.purpose != expected_purpose:
                self.logger.warning(
                    "token_purpose_mismatch",
                    expected=expected_purpose.value,
                    actual=token_data.purpose.value,
                )
                raise TokenInvalidError(f"Token purpose mismatch: expected {expected_purpose.value}")

            # Check expiry
            now = datetime.now(timezone.utc)
            if now > token_data.expires_at:
                self.logger.warning(
                    "token_expired",
                    purpose=token_data.purpose.value,
                    expired_at=token_data.expires_at.isoformat(),
                )
                raise TokenExpiredError("Token has expired")

            return token_data

        except InvalidToken as e:
            self.logger.warning("token_invalid", error=str(e))
            raise TokenInvalidError("Invalid or tampered token") from e
        except ValueError as e:
            self.logger.warning("token_parse_error", error=str(e))
            raise TokenInvalidError("Malformed token data") from e


def create_token_service(
    encryption_key: bytes | None = None,
    email_verification_expire_hours: int = 24,
    password_reset_expire_hours: int = 1,
) -> TokenService:
    """
    Factory function to create token service.

    Args:
        encryption_key: Fernet encryption key (generates new if None)
        email_verification_expire_hours: Email verification token lifetime
        password_reset_expire_hours: Password reset token lifetime

    Returns:
        Configured TokenService instance
    """
    if encryption_key is None:
        encryption_key = Fernet.generate_key()

    config = TokenConfig(
        encryption_key=encryption_key,
        email_verification_expire_hours=email_verification_expire_hours,
        password_reset_expire_hours=password_reset_expire_hours,
    )

    return TokenService(config)

"""
Unit tests for Token Service.

Tests cover email verification tokens, password reset tokens, and activation tokens.
"""
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from cryptography.fernet import Fernet

from src.infrastructure.token_service import (
    TokenService,
    TokenConfig,
    TokenPurpose,
    TokenExpiredError,
    TokenInvalidError,
    create_token_service,
)


@pytest.fixture
def token_service():
    """Create token service instance."""
    return create_token_service(
        encryption_key=Fernet.generate_key(),
        email_verification_expire_hours=24,
        password_reset_expire_hours=1,
    )


class TestTokenService:
    """Test cases for TokenService."""

    def test_generate_email_verification_token(self, token_service):
        """Test generating email verification token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_email_verification_token(user_id, email)

        assert token
        assert isinstance(token, str)

    def test_verify_email_verification_token_success(self, token_service):
        """Test successfully verifying email verification token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_email_verification_token(user_id, email)
        verified_user_id, verified_email = token_service.verify_email_verification_token(token)

        assert verified_user_id == user_id
        assert verified_email == email

    def test_verify_email_token_wrong_purpose_fails(self, token_service):
        """Test that using email token as password reset fails."""
        user_id = uuid4()
        email = "test@example.com"

        email_token = token_service.generate_email_verification_token(user_id, email)

        with pytest.raises(TokenInvalidError, match="Token purpose mismatch"):
            token_service.verify_password_reset_token(email_token)

    def test_verify_tampered_token_fails(self, token_service):
        """Test that tampered token fails verification."""
        user_id = uuid4()
        token = token_service.generate_email_verification_token(user_id, "test@example.com")

        # Tamper with token
        tampered = token[:-5] + "xxxxx"

        with pytest.raises(TokenInvalidError, match="Invalid or tampered token"):
            token_service.verify_email_verification_token(tampered)

    def test_generate_password_reset_token(self, token_service):
        """Test generating password reset token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_password_reset_token(user_id, email)

        assert token
        assert isinstance(token, str)

    def test_verify_password_reset_token_success(self, token_service):
        """Test successfully verifying password reset token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_password_reset_token(user_id, email)
        verified_user_id, verified_email = token_service.verify_password_reset_token(token)

        assert verified_user_id == user_id
        assert verified_email == email

    def test_generate_activation_token(self, token_service):
        """Test generating activation token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_activation_token(user_id, email)

        assert token
        assert isinstance(token, str)

    def test_verify_activation_token_success(self, token_service):
        """Test successfully verifying activation token."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_activation_token(user_id, email)
        verified_user_id, verified_email = token_service.verify_activation_token(token)

        assert verified_user_id == user_id
        assert verified_email == email

    def test_token_expiry_enforced(self):
        """Test that expired token is rejected."""
        # Create service with 0-hour expiry (immediate expiry)
        short_service = create_token_service(
            encryption_key=Fernet.generate_key(),
            email_verification_expire_hours=0,
        )

        user_id = uuid4()
        token = short_service.generate_email_verification_token(user_id, "test@example.com")

        # Token should be expired immediately
        with pytest.raises(TokenExpiredError, match="Token has expired"):
            short_service.verify_email_verification_token(token)

    def test_different_encryption_key_fails(self):
        """Test that token encrypted with different key fails verification."""
        user_id = uuid4()
        email = "test@example.com"

        service1 = create_token_service(encryption_key=Fernet.generate_key())
        service2 = create_token_service(encryption_key=Fernet.generate_key())

        token = service1.generate_email_verification_token(user_id, email)

        with pytest.raises(TokenInvalidError):
            service2.verify_email_verification_token(token)

    def test_token_contains_correct_data(self, token_service):
        """Test that token contains all expected data."""
        user_id = uuid4()
        email = "test@example.com"

        token = token_service.generate_email_verification_token(user_id, email)
        verified_user_id, verified_email = token_service.verify_email_verification_token(token)

        assert verified_user_id == user_id
        assert verified_email == email
        assert isinstance(verified_user_id, type(user_id))

    def test_create_token_service_factory(self):
        """Test factory function creates service."""
        service = create_token_service(
            email_verification_expire_hours=48,
            password_reset_expire_hours=2,
        )

        assert isinstance(service, TokenService)
        assert service.config.email_verification_expire_hours == 48
        assert service.config.password_reset_expire_hours == 2

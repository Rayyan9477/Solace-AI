"""
Integration tests demonstrating how infrastructure services work with domain layer.

This module tests the complete integration between:
- Domain Layer: User, UserPreferences, EmailAddress, ConsentService
- Infrastructure Layer: JWTService, PasswordService, TokenService, EncryptionService
"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone

from src.domain import (
    User,
    UserPreferences,
    EmailAddress,
    UserRole,
    AccountStatus,
    ConsentType,
)
from src.infrastructure import (
    create_jwt_service,
    create_password_service,
    create_token_service,
    create_encryption_service,
    TokenType,
    HashAlgorithm,
)
from src.events import UserCreatedEvent, LoginSuccessfulEvent


class TestUserRegistrationFlow:
    """Test complete user registration flow with all services."""

    @pytest.fixture
    def password_service(self):
        return create_password_service()

    @pytest.fixture
    def jwt_service(self):
        return create_jwt_service(secret_key="test-secret-key-for-integration")

    @pytest.fixture
    def token_service(self):
        return create_token_service()

    @pytest.fixture
    def encryption_service(self):
        return create_encryption_service()

    def test_user_registration_with_password_hashing(self, password_service):
        """Test user creation with proper password hashing."""
        # Hash password with Argon2id
        password_hash = password_service.hash_password("SecurePassword123!")

        # Validate email using EmailAddress value object
        email = EmailAddress(value="NewUser@Example.com")
        assert email.value == "newuser@example.com"  # Normalized to lowercase

        # Create user with hashed password (email as string)
        user = User(
            email=email.value,
            password_hash=password_hash,
            display_name="New User",
            role=UserRole.USER,
        )

        # Verify password works
        result = password_service.verify_password("SecurePassword123!", user.password_hash)
        assert result.is_valid is True
        assert result.algorithm_used == HashAlgorithm.ARGON2

        # Verify user properties
        assert user.email == "newuser@example.com"
        assert user.role == UserRole.USER
        assert user.status == AccountStatus.PENDING_VERIFICATION

    def test_email_verification_flow(self, token_service):
        """Test email verification token flow."""
        user_id = uuid4()
        email = "verify@example.com"

        # Generate verification token
        token = token_service.generate_email_verification_token(user_id, email)
        assert token is not None

        # Verify token
        verified_user_id, verified_email = token_service.verify_email_verification_token(token)
        assert verified_user_id == user_id
        assert verified_email == email

    @pytest.mark.asyncio
    async def test_login_with_jwt_generation(self, password_service, jwt_service):
        """Test login flow with JWT token generation."""
        # Setup: Create user with password
        password_hash = password_service.hash_password("LoginTest123!")
        user = User(
            email="login@example.com",
            password_hash=password_hash,
            display_name="Login User",
            role=UserRole.USER,
            status=AccountStatus.ACTIVE,
            email_verified=True,
        )

        # Simulate login: Verify password
        result = password_service.verify_password("LoginTest123!", user.password_hash)
        assert result.is_valid is True

        # Generate JWT tokens
        token_pair = jwt_service.generate_token_pair(
            user_id=user.user_id,
            email=user.email,
            role=user.role.value,
        )

        # Verify access token
        payload = await jwt_service.verify_token(token_pair.access_token, TokenType.ACCESS)
        assert payload.user_id == user.user_id
        assert payload.email == user.email
        assert payload.role == user.role.value

        # Record successful login - resets failed attempts to 0
        user.record_login_attempt(success=True)
        assert user.login_attempts == 0  # Failed attempts reset on success


class TestPasswordMigrationFlow:
    """Test bcrypt to argon2 password migration."""

    @pytest.fixture
    def password_service(self):
        return create_password_service()

    def test_bcrypt_to_argon2_migration(self, password_service):
        """Test automatic password migration from bcrypt to argon2."""
        import bcrypt as bcrypt_lib

        password = "MigrationTest123!"

        # Simulate legacy bcrypt hash
        legacy_hash = bcrypt_lib.hashpw(password.encode(), bcrypt_lib.gensalt()).decode()

        # Verify with bcrypt - should trigger migration
        result = password_service.verify_password(password, legacy_hash)
        assert result.is_valid is True
        assert result.needs_rehash is True
        assert result.algorithm_used == HashAlgorithm.BCRYPT
        assert result.new_hash is not None
        assert result.new_hash.startswith("$argon2")

        # Next login with new hash uses argon2
        new_result = password_service.verify_password(password, result.new_hash)
        assert new_result.is_valid is True
        assert new_result.needs_rehash is False
        assert new_result.algorithm_used == HashAlgorithm.ARGON2


class TestSensitiveDataEncryption:
    """Test encryption of sensitive user data."""

    @pytest.fixture
    def encryption_service(self):
        return create_encryption_service()

    def test_encrypt_user_pii(self, encryption_service):
        """Test encryption of PII fields."""
        user_data = {
            "id": str(uuid4()),
            "email": "sensitive@example.com",
            "display_name": "John Doe",
            "ssn": "123-45-6789",
            "phone": "+1-555-123-4567",
        }

        # Encrypt sensitive fields
        encrypted_data = encryption_service.encrypt_dict_fields(
            user_data,
            fields_to_encrypt=["ssn", "phone"],
        )

        # Non-sensitive fields unchanged
        assert encrypted_data["email"] == user_data["email"]
        assert encrypted_data["display_name"] == user_data["display_name"]

        # Sensitive fields encrypted
        assert encrypted_data["ssn"] != user_data["ssn"]
        assert encrypted_data["phone"] != user_data["phone"]
        assert encrypted_data["ssn"].startswith("gAAAAA")  # Fernet prefix

        # Decrypt and verify
        decrypted_data = encryption_service.decrypt_dict_fields(
            encrypted_data,
            fields_to_decrypt=["ssn", "phone"],
        )
        assert decrypted_data["ssn"] == user_data["ssn"]
        assert decrypted_data["phone"] == user_data["phone"]


class TestTokenRefreshFlow:
    """Test JWT token refresh flow."""

    @pytest.fixture
    def jwt_service(self):
        return create_jwt_service(secret_key="refresh-test-secret-key")

    @pytest.mark.asyncio
    async def test_refresh_token_flow(self, jwt_service):
        """Test refreshing access token from refresh token."""
        user_id = uuid4()

        # Initial login - get token pair
        token_pair = jwt_service.generate_token_pair(
            user_id=user_id,
            email="refresh@example.com",
            role="user",
        )

        # Verify refresh token is valid
        refresh_payload = await jwt_service.verify_token(
            token_pair.refresh_token,
            TokenType.REFRESH,
        )
        assert refresh_payload.user_id == user_id

        # Refresh access token
        new_access_token = await jwt_service.refresh_access_token(token_pair.refresh_token)

        # Verify new access token
        new_payload = await jwt_service.verify_token(new_access_token, TokenType.ACCESS)
        assert new_payload.user_id == user_id
        assert new_payload.email == "refresh@example.com"


class TestPasswordResetFlow:
    """Test password reset flow."""

    @pytest.fixture
    def password_service(self):
        return create_password_service()

    @pytest.fixture
    def token_service(self):
        return create_token_service()

    def test_password_reset_flow(self, password_service, token_service):
        """Test complete password reset flow."""
        user_id = uuid4()
        email = "reset@example.com"
        old_password = "OldPassword123!"
        new_password = "NewPassword456!"

        # Create user with old password
        old_hash = password_service.hash_password(old_password)

        # Generate password reset token
        reset_token = token_service.generate_password_reset_token(user_id, email)

        # Verify reset token
        verified_user_id, verified_email = token_service.verify_password_reset_token(reset_token)
        assert verified_user_id == user_id
        assert verified_email == email

        # Hash new password
        new_hash = password_service.hash_password(new_password)

        # Verify old password no longer works with new hash
        old_result = password_service.verify_password(old_password, new_hash)
        assert old_result.is_valid is False

        # Verify new password works
        new_result = password_service.verify_password(new_password, new_hash)
        assert new_result.is_valid is True


class TestDomainEventIntegration:
    """Test domain events with infrastructure services."""

    def test_user_created_event_with_services(self):
        """Test UserCreatedEvent creation with infrastructure integration."""
        user_id = uuid4()
        email = "events@example.com"

        event = UserCreatedEvent(
            aggregate_id=user_id,
            email=email,
            display_name="Event User",
            role=UserRole.USER.value,
            status=AccountStatus.PENDING_VERIFICATION.value,
            timezone="UTC",
            locale="en-US",
        )

        assert event.aggregate_id == user_id
        assert event.email == email
        assert event.event_type.value == "user.created"

        # Event can be serialized for messaging
        event_dict = event.to_dict()
        assert event_dict["aggregate_id"] == str(user_id)
        assert event_dict["email"] == email


class TestFullUserLifecycle:
    """Test complete user lifecycle with all services."""

    @pytest.fixture
    def services(self):
        return {
            "password": create_password_service(),
            "jwt": create_jwt_service(secret_key="lifecycle-test-key"),
            "token": create_token_service(),
            "encryption": create_encryption_service(),
        }

    @pytest.mark.asyncio
    async def test_complete_user_lifecycle(self, services):
        """Test user registration -> verification -> login -> data encryption."""
        # 1. Registration - validate email with value object, create user
        email_vo = EmailAddress(value="Lifecycle@Example.com")
        assert email_vo.value == "lifecycle@example.com"  # Normalized

        password_hash = services["password"].hash_password("LifecycleTest123!")
        user = User(
            email=email_vo.value,
            password_hash=password_hash,
            display_name="Lifecycle User",
            role=UserRole.USER,
        )
        assert user.status == AccountStatus.PENDING_VERIFICATION

        # 2. Email verification
        verification_token = services["token"].generate_email_verification_token(
            user.user_id, user.email
        )
        user_id, email = services["token"].verify_email_verification_token(verification_token)
        assert user_id == user.user_id

        # Mark email verified
        user.email_verified = True
        assert user.email_verified is True

        # 3. Activation
        user.activate()
        assert user.status == AccountStatus.ACTIVE

        # 4. Login
        login_result = services["password"].verify_password("LifecycleTest123!", user.password_hash)
        assert login_result.is_valid is True

        token_pair = services["jwt"].generate_token_pair(
            user_id=user.user_id,
            email=user.email,
            role=user.role.value,
        )
        user.record_login_attempt(success=True)
        assert user.login_attempts == 0  # Failed attempts reset on success

        # 5. Encrypt sensitive data
        user_data = {
            "id": str(user.user_id),
            "email": user.email,
            "ssn": "123-45-6789",
        }
        encrypted_data = services["encryption"].encrypt_dict_fields(user_data, ["ssn"])
        assert encrypted_data["ssn"] != "123-45-6789"

        # 6. Verify JWT
        payload = await services["jwt"].verify_token(token_pair.access_token, TokenType.ACCESS)
        assert payload.user_id == user.user_id

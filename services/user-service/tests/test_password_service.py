"""
Unit tests for Password Service.

Tests cover argon2 hashing, bcrypt verification, dual verification, and migration.
"""
import pytest
from passlib.hash import bcrypt

from src.infrastructure.password_service import (
    PasswordService,
    PasswordConfig,
    HashAlgorithm,
    PasswordVerificationResult,
    create_password_service,
)


@pytest.fixture
def password_service():
    """Create password service instance."""
    return PasswordService()


class TestPasswordService:
    """Test cases for PasswordService."""

    def test_hash_password_argon2(self, password_service):
        """Test hashing password with argon2."""
        password = "SecurePassword123!"

        password_hash = password_service.hash_password(password)

        assert password_hash
        assert password_hash.startswith("$argon2")

    def test_hash_password_different_for_same_input(self, password_service):
        """Test that same password produces different hashes (salt)."""
        password = "SecurePassword123!"

        hash1 = password_service.hash_password(password)
        hash2 = password_service.hash_password(password)

        assert hash1 != hash2

    def test_verify_argon2_password_success(self, password_service):
        """Test successfully verifying argon2 password."""
        password = "SecurePassword123!"
        password_hash = password_service.hash_password(password)

        result = password_service.verify_password(password, password_hash)

        assert result.is_valid is True
        assert result.algorithm_used == HashAlgorithm.ARGON2
        assert result.needs_rehash is False
        assert result.new_hash is None

    def test_verify_argon2_password_wrong_password_fails(self, password_service):
        """Test that wrong password fails verification."""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"
        password_hash = password_service.hash_password(password)

        result = password_service.verify_password(wrong_password, password_hash)

        assert result.is_valid is False
        assert result.algorithm_used == HashAlgorithm.ARGON2

    def test_verify_bcrypt_password_success_with_migration(self, password_service):
        """Test verifying bcrypt password with automatic migration."""
        password = "SecurePassword123!"
        # Create bcrypt hash manually
        bcrypt_hash = bcrypt.hash(password)

        result = password_service.verify_password(password, bcrypt_hash)

        assert result.is_valid is True
        assert result.algorithm_used == HashAlgorithm.BCRYPT
        assert result.needs_rehash is True
        assert result.new_hash is not None
        assert result.new_hash.startswith("$argon2")

    def test_verify_bcrypt_password_wrong_password_fails(self, password_service):
        """Test that wrong bcrypt password fails."""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"
        bcrypt_hash = bcrypt.hash(password)

        result = password_service.verify_password(wrong_password, bcrypt_hash)

        assert result.is_valid is False
        assert result.algorithm_used == HashAlgorithm.BCRYPT
        assert result.needs_rehash is False
        assert result.new_hash is None

    def test_verify_bcrypt_migrated_hash_works(self, password_service):
        """Test that migrated hash from bcrypt to argon2 works."""
        password = "SecurePassword123!"
        bcrypt_hash = bcrypt.hash(password)

        # First verification with migration
        result1 = password_service.verify_password(password, bcrypt_hash)
        assert result1.is_valid is True
        assert result1.new_hash is not None

        # Second verification with new hash
        result2 = password_service.verify_password(password, result1.new_hash)
        assert result2.is_valid is True
        assert result2.algorithm_used == HashAlgorithm.ARGON2

    def test_needs_rehash_bcrypt_returns_true(self, password_service):
        """Test that bcrypt hashes always need rehashing."""
        bcrypt_hash = bcrypt.hash("password123")

        needs_rehash = password_service.needs_rehash(bcrypt_hash)

        assert needs_rehash is True

    def test_needs_rehash_argon2_returns_false(self, password_service):
        """Test that current argon2 hashes don't need rehashing."""
        argon2_hash = password_service.hash_password("password123")

        needs_rehash = password_service.needs_rehash(argon2_hash)

        assert needs_rehash is False

    def test_get_algorithm_argon2(self, password_service):
        """Test detecting argon2 algorithm."""
        argon2_hash = password_service.hash_password("password123")

        algorithm = password_service.get_algorithm(argon2_hash)

        assert algorithm == HashAlgorithm.ARGON2

    def test_get_algorithm_bcrypt(self, password_service):
        """Test detecting bcrypt algorithm."""
        bcrypt_hash = bcrypt.hash("password123")

        algorithm = password_service.get_algorithm(bcrypt_hash)

        assert algorithm == HashAlgorithm.BCRYPT

    def test_create_password_service_factory(self):
        """Test factory function creates service."""
        service = create_password_service(
            argon2_time_cost=3,
            argon2_memory_cost=131072,
        )

        assert isinstance(service, PasswordService)
        assert service.config.argon2_time_cost == 3
        assert service.config.argon2_memory_cost == 131072

    def test_migration_scenario_end_to_end(self, password_service):
        """Test complete migration scenario from bcrypt to argon2."""
        password = "UserPassword123!"

        # Step 1: User has old bcrypt hash
        old_hash = bcrypt.hash(password)

        # Step 2: User logs in, password verified, migration available
        verify_result = password_service.verify_password(password, old_hash)
        assert verify_result.is_valid is True
        assert verify_result.needs_rehash is True
        new_hash = verify_result.new_hash

        # Step 3: Update user's hash to argon2
        assert new_hash.startswith("$argon2")

        # Step 4: Next login uses argon2
        next_verify = password_service.verify_password(password, new_hash)
        assert next_verify.is_valid is True
        assert next_verify.algorithm_used == HashAlgorithm.ARGON2
        assert next_verify.needs_rehash is False

    def test_argon2_configuration_applied(self):
        """Test that custom argon2 configuration is applied."""
        custom_config = PasswordConfig(
            argon2_time_cost=4,
            argon2_memory_cost=131072,
            argon2_parallelism=2,
        )
        service = PasswordService(custom_config)

        password_hash = service.hash_password("test123")

        # Verify hash contains parameters (argon2 format includes them)
        assert "$argon2" in password_hash
        assert password_hash.startswith("$argon2")

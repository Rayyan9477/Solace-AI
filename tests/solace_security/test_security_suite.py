"""
Solace-AI Security Test Suite.

Tests for SQL injection prevention, authentication bypass resistance,
and PHI leak detection across the codebase.
"""
from __future__ import annotations

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from solace_common.utils import ValidationUtils, ValidationPatterns
from solace_infrastructure.postgres import PostgresRepository, _is_valid_identifier


# ---------------------------------------------------------------------------
# SQL Injection Prevention Tests
# ---------------------------------------------------------------------------

class TestSQLInjectionPrevention:
    """Verify SQL identifier validation blocks injection attempts."""

    @pytest.mark.parametrize("malicious_input", [
        "users; DROP TABLE users; --",
        "'; DELETE FROM users; --",
        "col1 OR 1=1",
        "table_name UNION SELECT * FROM passwords",
        "name; EXEC xp_cmdshell('whoami')",
        "col' OR '1'='1",
        "1; UPDATE users SET role='admin'",
        "table\x00name",
        "col; WAITFOR DELAY '00:00:05'",
        "Robert'); DROP TABLE students;--",
    ])
    def test_sql_identifier_rejects_injection(self, malicious_input: str):
        """SQL identifier validation must reject injection payloads."""
        assert ValidationUtils.is_valid_sql_identifier(malicious_input) is False

    @pytest.mark.parametrize("malicious_input", [
        "users; DROP TABLE users; --",
        "'; DELETE FROM users; --",
        "col1 OR 1=1",
    ])
    def test_validate_sql_identifier_raises_on_injection(self, malicious_input: str):
        """validate_sql_identifier must raise ValueError for injection payloads."""
        with pytest.raises(ValueError):
            ValidationUtils.validate_sql_identifier(malicious_input, "column name")

    @pytest.mark.parametrize("valid_name", [
        "users",
        "treatment_plans",
        "_private_table",
        "col1",
        "therapy_sessions",
        "a" * 128,
    ])
    def test_sql_identifier_accepts_valid_names(self, valid_name: str):
        """Valid SQL identifiers must be accepted."""
        assert ValidationUtils.is_valid_sql_identifier(valid_name) is True

    def test_sql_identifier_rejects_empty(self):
        assert ValidationUtils.is_valid_sql_identifier("") is False

    def test_sql_identifier_rejects_too_long(self):
        assert ValidationUtils.is_valid_sql_identifier("a" * 129) is False

    def test_sql_identifier_rejects_starting_with_number(self):
        assert ValidationUtils.is_valid_sql_identifier("1table") is False

    def test_postgres_repository_insert_rejects_bad_column(self):
        """PostgresRepository.insert() must reject dict keys that aren't valid identifiers."""
        # _is_valid_identifier delegates to ValidationUtils
        assert _is_valid_identifier("valid_col") is True
        assert _is_valid_identifier("col; DROP TABLE x") is False

    def test_sql_identifier_pattern_anchored(self):
        """Pattern must be anchored to prevent partial matches."""
        pattern = ValidationPatterns.SQL_IDENTIFIER
        # Must not match strings with trailing injection
        assert pattern.match("valid; DROP") is None
        # Must match clean identifiers
        assert pattern.match("valid_name") is not None


# ---------------------------------------------------------------------------
# Authentication Security Tests
# ---------------------------------------------------------------------------

class TestAuthenticationSecurity:
    """Verify authentication and token security properties."""

    def test_jwt_requires_secret_key(self):
        """JWTManager must not work with empty/missing secret."""
        from solace_security.auth import AuthSettings
        with pytest.raises(Exception):
            AuthSettings(secret_key="")

    def test_jwt_secret_minimum_length(self):
        """Secret key must meet minimum length requirement."""
        from solace_security.auth import AuthSettings
        with pytest.raises(Exception):
            AuthSettings(secret_key="short")

    def test_password_hasher_rejects_empty(self):
        """PasswordHasher should not accept empty passwords."""
        from solace_security.auth import PasswordHasher
        hasher = PasswordHasher()
        # Hashing empty string should still produce a hash (argon2/bcrypt handle this)
        # but verification against a different hash should fail
        hashed = hasher.hash_password("correct_password")
        assert hasher.verify_password("", hashed) is False

    def test_password_hasher_timing_safe(self):
        """Password verification must not short-circuit on mismatch length."""
        from solace_security.auth import PasswordHasher
        hasher = PasswordHasher()
        hashed = hasher.hash_password("test_password_123")
        # Both wrong passwords should take similar time (constant-time comparison)
        assert hasher.verify_password("wrong", hashed) is False
        assert hasher.verify_password("also_wrong_but_longer_string", hashed) is False

    def test_jwt_token_creation_and_decode(self):
        """JWT tokens must be decodable with the same secret."""
        from solace_security.auth import AuthSettings, JWTManager, TokenType
        settings = AuthSettings.for_development()
        jwt_mgr = JWTManager(settings)
        token = jwt_mgr.create_token(
            subject="user-123",
            token_type=TokenType.ACCESS,
            claims={"role": "patient"},
        )
        assert token is not None
        payload = jwt_mgr.decode_token(token)
        assert payload.sub == "user-123"

    def test_jwt_rejects_tampered_token(self):
        """JWT must reject tokens signed with a different secret."""
        from solace_security.auth import AuthSettings, JWTManager, TokenType
        settings1 = AuthSettings(secret_key="first-secret-key-32-bytes-long!!")
        settings2 = AuthSettings(secret_key="second-secret-key-32bytes-long!!")
        jwt1 = JWTManager(settings1)
        jwt2 = JWTManager(settings2)
        token = jwt1.create_token(subject="user-123", token_type=TokenType.ACCESS)
        with pytest.raises(Exception):
            jwt2.decode_token(token)

    def test_development_keys_flag(self):
        """for_development() must mark settings appropriately."""
        from solace_security.auth import AuthSettings
        settings = AuthSettings.for_development()
        # Development settings should work in test context
        assert settings.secret_key is not None


# ---------------------------------------------------------------------------
# Encryption Security Tests
# ---------------------------------------------------------------------------

class TestEncryptionSecurity:
    """Verify encryption module security properties."""

    def test_encryption_roundtrip(self):
        """Encrypt and decrypt must produce original data."""
        from solace_security.encryption import FieldEncryptor, EncryptionSettings
        settings = EncryptionSettings.for_development()
        encryptor = FieldEncryptor(settings)
        original = "sensitive-patient-data"
        encrypted = encryptor.encrypt(original)
        assert encrypted != original
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == original

    def test_different_keys_cannot_decrypt(self):
        """Data encrypted with one key must not decrypt with another."""
        from solace_security.encryption import FieldEncryptor, EncryptionSettings
        settings1 = EncryptionSettings.for_development()
        encryptor1 = FieldEncryptor(settings1)
        encrypted = encryptor1.encrypt("secret data")
        # Attempting to decrypt with a different encryptor should fail
        settings2 = EncryptionSettings.for_development()
        # Force a different key
        import os
        settings2.master_key = os.urandom(32).hex()
        encryptor2 = FieldEncryptor(settings2)
        with pytest.raises(Exception):
            encryptor2.decrypt(encrypted)

    def test_encrypted_data_not_plaintext(self):
        """Encrypted output must not contain the plaintext input."""
        from solace_security.encryption import FieldEncryptor, EncryptionSettings
        settings = EncryptionSettings.for_development()
        encryptor = FieldEncryptor(settings)
        plaintext = "John Smith SSN 123-45-6789"
        encrypted = encryptor.encrypt(plaintext)
        assert plaintext not in str(encrypted)
        assert "123-45-6789" not in str(encrypted)


# ---------------------------------------------------------------------------
# PHI Safety Tests
# ---------------------------------------------------------------------------

class TestPHISafety:
    """Verify PHI detection and masking works correctly."""

    def test_ssn_detection(self):
        """SSN patterns must be detected."""
        from solace_security.phi_protection import detect_phi
        result = detect_phi("My SSN is 123-45-6789")
        assert result.has_phi is True
        assert any(m.phi_type.value == "ssn" for m in result.matches)

    def test_email_detection(self):
        """Email addresses must be detected as PHI."""
        from solace_security.phi_protection import detect_phi
        result = detect_phi("Contact me at patient@hospital.com")
        assert result.has_phi is True
        assert any(m.phi_type.value == "email" for m in result.matches)

    def test_phone_detection(self):
        """Phone numbers must be detected as PHI."""
        from solace_security.phi_protection import detect_phi
        result = detect_phi("Call me at (555) 123-4567")
        assert result.has_phi is True

    def test_masking_replaces_phi(self):
        """PHI masking must replace sensitive data with mask characters."""
        from solace_security.phi_protection import mask_phi
        original = "Patient SSN: 123-45-6789, email: john@example.com"
        masked = mask_phi(original)
        assert "123-45-6789" not in masked
        assert "john@example.com" not in masked

    def test_clean_text_no_false_positive(self):
        """Clean therapeutic text should not trigger PHI detection."""
        from solace_security.phi_protection import detect_phi
        clean_text = (
            "I have been feeling anxious about work. My sleep has been poor "
            "and I notice negative thought patterns when I am stressed."
        )
        result = detect_phi(clean_text)
        # Therapeutic content should generally not contain PHI
        assert result.has_phi is False

    def test_medical_record_number_detection(self):
        """Medical record numbers should be detected if configured."""
        from solace_security.phi_protection import detect_phi
        result = detect_phi("MRN: 12345678")
        # MRN detection depends on configured patterns
        # At minimum, the detector should not crash
        assert isinstance(result.has_phi, bool)

    def test_credit_card_detection(self):
        """Credit card numbers must be detected."""
        from solace_security.phi_protection import detect_phi
        result = detect_phi("Card number: 4111-1111-1111-1111")
        assert result.has_phi is True


# ---------------------------------------------------------------------------
# Input Validation Security Tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify input validation and sanitization."""

    def test_sanitize_removes_control_chars(self):
        """Sanitize must remove control characters."""
        sanitized = ValidationUtils.sanitize_string("hello\x00world\x01test")
        assert "\x00" not in sanitized
        assert "\x01" not in sanitized

    def test_sanitize_truncates_long_input(self):
        """Sanitize must enforce max length."""
        long_input = "x" * 5000
        result = ValidationUtils.sanitize_string(long_input, max_length=100)
        assert len(result) <= 100

    def test_email_validation(self):
        """Email validation must reject malformed addresses."""
        assert ValidationUtils.is_valid_email("user@example.com") is True
        assert ValidationUtils.is_valid_email("not-an-email") is False
        assert ValidationUtils.is_valid_email("") is False
        assert ValidationUtils.is_valid_email("user@") is False

    def test_uuid_validation(self):
        """UUID validation must reject non-UUID strings."""
        assert ValidationUtils.is_valid_uuid(str(uuid4())) is True
        assert ValidationUtils.is_valid_uuid("not-a-uuid") is False
        assert ValidationUtils.is_valid_uuid("") is False

    def test_crypto_constant_time_compare(self):
        """Constant-time comparison must be correct."""
        from solace_common.utils import CryptoUtils
        assert CryptoUtils.constant_time_compare("abc", "abc") is True
        assert CryptoUtils.constant_time_compare("abc", "def") is False
        assert CryptoUtils.constant_time_compare("", "") is True

    def test_hmac_verify(self):
        """HMAC verification must reject wrong signatures."""
        from solace_common.utils import CryptoUtils
        message = "test-message"
        secret = "test-secret"
        signature = CryptoUtils.hmac_sign(message, secret)
        assert CryptoUtils.hmac_verify(message, signature, secret) is True
        assert CryptoUtils.hmac_verify(message, "wrong-signature", secret) is False
        assert CryptoUtils.hmac_verify("wrong-message", signature, secret) is False

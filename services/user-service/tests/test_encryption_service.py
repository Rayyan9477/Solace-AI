"""
Unit tests for Encryption Service.

Tests cover field-level encryption, decryption, and key rotation.
"""
import pytest
from cryptography.fernet import Fernet

from src.infrastructure.encryption_service import (
    EncryptionService,
    EncryptionError,
    DecryptionError,
    create_encryption_service,
)


@pytest.fixture
def encryption_service():
    """Create encryption service instance."""
    return create_encryption_service(encryption_key=Fernet.generate_key())


class TestEncryptionService:
    """Test cases for EncryptionService."""

    def test_encrypt_field_success(self, encryption_service):
        """Test successfully encrypting a field."""
        value = "sensitive-data-123"

        encrypted = encryption_service.encrypt_field(value)

        assert encrypted
        assert encrypted != value
        assert isinstance(encrypted, str)

    def test_decrypt_field_success(self, encryption_service):
        """Test successfully decrypting a field."""
        value = "sensitive-data-123"

        encrypted = encryption_service.encrypt_field(value)
        decrypted = encryption_service.decrypt_field(encrypted)

        assert decrypted == value

    def test_encrypt_decrypt_round_trip(self, encryption_service):
        """Test encrypt-decrypt round trip preserves data."""
        values = [
            "123-45-6789",  # SSN
            "john@example.com",  # Email
            "Secret message with special chars: !@#$%^&*()",
            "Unicode: 你好世界",
        ]

        for value in values:
            encrypted = encryption_service.encrypt_field(value)
            decrypted = encryption_service.decrypt_field(encrypted)
            assert decrypted == value

    def test_encrypt_empty_string_returns_empty(self, encryption_service):
        """Test that empty string is not encrypted."""
        encrypted = encryption_service.encrypt_field("")

        assert encrypted == ""

    def test_decrypt_with_wrong_key_fails(self):
        """Test that decryption with wrong key fails."""
        service1 = create_encryption_service(encryption_key=Fernet.generate_key())
        service2 = create_encryption_service(encryption_key=Fernet.generate_key())

        value = "sensitive-data"
        encrypted = service1.encrypt_field(value)

        with pytest.raises(DecryptionError, match="Invalid encryption key"):
            service2.decrypt_field(encrypted)

    def test_decrypt_tampered_data_fails(self, encryption_service):
        """Test that decrypting tampered data fails."""
        value = "sensitive-data"
        encrypted = encryption_service.encrypt_field(value)

        # Tamper with encrypted data
        tampered = encrypted[:-5] + "xxxxx"

        with pytest.raises(DecryptionError):
            encryption_service.decrypt_field(tampered)

    def test_encrypt_dict_fields_success(self, encryption_service):
        """Test encrypting specific fields in a dictionary."""
        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
            "phone": "555-1234",
        }

        encrypted_data = encryption_service.encrypt_dict_fields(data, ["ssn", "phone"])

        assert encrypted_data["name"] == "John Doe"  # Not encrypted
        assert encrypted_data["email"] == "john@example.com"  # Not encrypted
        assert encrypted_data["ssn"] != "123-45-6789"  # Encrypted
        assert encrypted_data["phone"] != "555-1234"  # Encrypted

    def test_decrypt_dict_fields_success(self, encryption_service):
        """Test decrypting specific fields in a dictionary."""
        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
            "phone": "555-1234",
        }

        encrypted_data = encryption_service.encrypt_dict_fields(data, ["ssn", "phone"])
        decrypted_data = encryption_service.decrypt_dict_fields(encrypted_data, ["ssn", "phone"])

        assert decrypted_data == data

    def test_rotate_encryption_success(self):
        """Test rotating encryption key."""
        old_key = Fernet.generate_key()
        new_key = Fernet.generate_key()

        old_service = create_encryption_service(encryption_key=old_key)
        new_service = create_encryption_service(encryption_key=new_key)

        # Encrypt with old key
        value = "sensitive-data"
        old_encrypted = old_service.encrypt_field(value)

        # Rotate to new key
        new_encrypted = new_service.rotate_encryption(old_encrypted, old_key)

        # Verify can decrypt with new key
        decrypted = new_service.decrypt_field(new_encrypted)
        assert decrypted == value

        # Verify cannot decrypt with old key anymore
        with pytest.raises(DecryptionError):
            old_service.decrypt_field(new_encrypted)

    def test_rotate_encryption_with_wrong_old_key_fails(self):
        """Test that rotation with wrong old key fails."""
        old_key = Fernet.generate_key()
        wrong_old_key = Fernet.generate_key()
        new_key = Fernet.generate_key()

        old_service = create_encryption_service(encryption_key=old_key)
        new_service = create_encryption_service(encryption_key=new_key)

        value = "sensitive-data"
        encrypted = old_service.encrypt_field(value)

        with pytest.raises(DecryptionError, match="Failed to decrypt with old key"):
            new_service.rotate_encryption(encrypted, wrong_old_key)

    def test_encrypt_different_values_different_ciphertext(self, encryption_service):
        """Test that different values produce different ciphertexts."""
        value1 = "sensitive-data-1"
        value2 = "sensitive-data-2"

        encrypted1 = encryption_service.encrypt_field(value1)
        encrypted2 = encryption_service.encrypt_field(value2)

        assert encrypted1 != encrypted2

    def test_encrypt_same_value_different_ciphertext(self, encryption_service):
        """Test that same value produces different ciphertexts (nonce)."""
        value = "sensitive-data"

        encrypted1 = encryption_service.encrypt_field(value)
        encrypted2 = encryption_service.encrypt_field(value)

        # Due to Fernet's timestamp-based nonce, even same value produces different ciphertext
        assert encrypted1 != encrypted2

        # But both decrypt to same value
        assert encryption_service.decrypt_field(encrypted1) == value
        assert encryption_service.decrypt_field(encrypted2) == value

    def test_create_encryption_service_factory(self):
        """Test factory function creates service."""
        key = Fernet.generate_key()
        service = create_encryption_service(encryption_key=key)

        assert isinstance(service, EncryptionService)
        assert service.config.encryption_key == key

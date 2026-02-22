"""Unit tests for encryption module."""

from __future__ import annotations
import pytest
from cryptography.exceptions import InvalidTag
from pydantic import ValidationError
from solace_security.encryption import (
    EncryptionAlgorithm,
    KeyDerivationFunction,
    EncryptionSettings,
    EncryptedData,
    KeyManager,
    AESGCMCipher,
    Encryptor,
    FieldEncryptor,
    SecureTokenGenerator,
    create_encryptor,
    create_field_encryptor,
    KEY_SIZE,
)


class TestEncryptionSettings:
    """Tests for EncryptionSettings."""

    def test_default_settings(self):
        # Use for_development() since master_key is now required
        settings = EncryptionSettings.for_development()
        assert settings.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert settings.kdf == KeyDerivationFunction.PBKDF2_SHA256
        assert settings.enable_key_rotation

    def test_custom_settings(self):
        settings = EncryptionSettings(
            master_key="test-master-key-32-bytes-long!!!",  # Must be exactly 32 bytes
            key_rotation_days=30,
        )
        assert settings.key_rotation_days == 30

    def test_master_key_required(self):
        """Test that master_key is required."""
        with pytest.raises((ValidationError, ValueError)):  # ValidationError for missing required field
            EncryptionSettings()

    def test_master_key_exact_length(self):
        """Test that master_key must be exactly 32 bytes."""
        with pytest.raises(ValueError, match="exactly 32 bytes"):
            EncryptionSettings(master_key="short-key")


class TestEncryptionAlgorithm:
    """Tests for EncryptionAlgorithm enum."""

    def test_algorithm_values(self):
        assert EncryptionAlgorithm.AES_256_GCM.value == "AES-256-GCM"


class TestEncryptedData:
    """Tests for EncryptedData model."""

    def test_create_encrypted_data(self):
        data = EncryptedData(
            ciphertext="Y2lwaGVydGV4dA==",
            nonce="bm9uY2U=",
            salt="c2FsdA==",
            key_id="key1",
        )
        assert data.ciphertext == "Y2lwaGVydGV4dA=="
        assert data.version == 1

    def test_to_compact(self):
        data = EncryptedData(
            ciphertext="cipher", nonce="nonce", salt="salt", key_id="key1"
        )
        compact = data.to_compact()
        assert "v1$" in compact
        assert "AES-256-GCM" in compact

    def test_from_compact(self):
        data = EncryptedData(
            ciphertext="cipher", nonce="nonce", salt="salt", key_id="key1"
        )
        compact = data.to_compact()
        restored = EncryptedData.from_compact(compact)
        assert restored.ciphertext == data.ciphertext
        assert restored.nonce == data.nonce
        assert restored.key_id == data.key_id

    def test_from_compact_invalid(self):
        with pytest.raises(ValueError):
            EncryptedData.from_compact("invalid")


class TestKeyManager:
    """Tests for KeyManager."""

    @pytest.fixture
    def key_manager(self):
        return KeyManager(EncryptionSettings.for_development())

    def test_derive_key(self, key_manager):
        salt = b"test_salt_16byte"
        key = key_manager.derive_key(salt)
        assert len(key) == KEY_SIZE

    def test_derive_key_consistent(self, key_manager):
        salt = b"test_salt_16byte"
        key1 = key_manager.derive_key(salt)
        key2 = key_manager.derive_key(salt)
        assert key1 == key2

    def test_derive_key_with_context(self, key_manager):
        salt = b"test_salt_16byte"
        key1 = key_manager.derive_key(salt, "context1")
        key2 = key_manager.derive_key(salt, "context2")
        assert key1 != key2

    def test_generate_data_key(self, key_manager):
        key, salt = key_manager.generate_data_key()
        assert len(key) == KEY_SIZE
        assert len(salt) == 16

    def test_register_key(self, key_manager):
        test_key = b"0" * KEY_SIZE
        key_manager.register_key("test", test_key)
        assert key_manager.get_key("test") == test_key

    def test_register_key_invalid_size(self, key_manager):
        with pytest.raises(ValueError):
            key_manager.register_key("test", b"short")


class TestAESGCMCipher:
    """Tests for AESGCMCipher."""

    @pytest.fixture
    def cipher(self):
        key = b"0" * KEY_SIZE
        return AESGCMCipher(key)

    def test_encrypt_decrypt(self, cipher):
        plaintext = b"Hello, World!"
        ciphertext, nonce = cipher.encrypt(plaintext)
        decrypted = cipher.decrypt(ciphertext, nonce)
        assert decrypted == plaintext

    def test_encrypt_with_aad(self, cipher):
        plaintext = b"Secret data"
        aad = b"Additional authenticated data"
        ciphertext, nonce = cipher.encrypt(plaintext, aad)
        decrypted = cipher.decrypt(ciphertext, nonce, aad)
        assert decrypted == plaintext

    def test_decrypt_wrong_aad_fails(self, cipher):
        plaintext = b"Secret data"
        aad = b"correct aad"
        ciphertext, nonce = cipher.encrypt(plaintext, aad)
        with pytest.raises((ValueError, TypeError, InvalidTag)):
            cipher.decrypt(ciphertext, nonce, b"wrong aad")

    def test_invalid_key_size(self):
        with pytest.raises(ValueError):
            AESGCMCipher(b"short_key")


class TestEncryptor:
    """Tests for Encryptor."""

    @pytest.fixture
    def encryptor(self):
        return Encryptor(EncryptionSettings.for_development())

    def test_encrypt_string(self, encryptor):
        plaintext = "Hello, World!"
        encrypted = encryptor.encrypt(plaintext)
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.ciphertext != plaintext

    def test_encrypt_bytes(self, encryptor):
        plaintext = b"Binary data"
        encrypted = encryptor.encrypt(plaintext)
        assert isinstance(encrypted, EncryptedData)

    def test_decrypt(self, encryptor):
        plaintext = "Secret message"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext.encode()

    def test_decrypt_to_string(self, encryptor):
        plaintext = "Secret message"
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt_to_string(encrypted)
        assert decrypted == plaintext

    def test_encrypt_with_context(self, encryptor):
        plaintext = "Context-bound data"
        encrypted = encryptor.encrypt(plaintext, context="user123")
        decrypted = encryptor.decrypt_to_string(encrypted, context="user123")
        assert decrypted == plaintext

    def test_encrypt_dict(self, encryptor):
        data = {"name": "John", "ssn": "123-45-6789", "email": "john@example.com"}
        encrypted = encryptor.encrypt_dict(data, sensitive_keys=["ssn", "email"])
        assert encrypted["name"] == "John"
        assert encrypted["ssn"].startswith("v1$")
        assert encrypted["email"].startswith("v1$")

    def test_decrypt_dict(self, encryptor):
        data = {"name": "John", "ssn": "123-45-6789"}
        encrypted = encryptor.encrypt_dict(data, sensitive_keys=["ssn"])
        decrypted = encryptor.decrypt_dict(encrypted, encrypted_keys=["ssn"])
        assert decrypted["name"] == "John"
        assert decrypted["ssn"] == "123-45-6789"


class TestFieldEncryptor:
    """Tests for FieldEncryptor."""

    @pytest.fixture
    def field_encryptor(self):
        from pydantic import SecretStr
        settings = EncryptionSettings.for_development()
        settings = settings.model_copy(update={"search_hash_salt": SecretStr("test-search-hash-salt-for-dev!!")})
        return FieldEncryptor(Encryptor(settings), settings=settings)

    def test_encrypt_field(self, field_encryptor):
        value = "sensitive-data"
        encrypted = field_encryptor.encrypt_field(value, "password")
        assert encrypted.startswith("v1$")

    def test_decrypt_field(self, field_encryptor):
        value = "sensitive-data"
        encrypted = field_encryptor.encrypt_field(value, "password")
        decrypted = field_encryptor.decrypt_field(encrypted, "password")
        assert decrypted == value

    def test_hash_for_search(self, field_encryptor):
        value = "searchable"
        hash1 = field_encryptor.hash_for_search(value)
        hash2 = field_encryptor.hash_for_search(value)
        assert hash1 == hash2
        assert len(hash1) == 64


class TestSecureTokenGenerator:
    """Tests for SecureTokenGenerator."""

    def test_generate_token(self):
        token = SecureTokenGenerator.generate_token()
        assert len(token) > 20

    def test_generate_token_custom_length(self):
        token = SecureTokenGenerator.generate_token(64)
        assert len(token) > 40

    def test_generate_hex_token(self):
        token = SecureTokenGenerator.generate_hex_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_generate_otp(self):
        otp = SecureTokenGenerator.generate_otp()
        assert len(otp) == 6
        assert otp.isdigit()

    def test_generate_otp_custom_length(self):
        otp = SecureTokenGenerator.generate_otp(8)
        assert len(otp) == 8

    def test_generate_key(self):
        key = SecureTokenGenerator.generate_key()
        assert len(key) == KEY_SIZE

    def test_generate_key_custom_size(self):
        key = SecureTokenGenerator.generate_key(16)
        assert len(key) == 16


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_encryptor(self):
        # Must provide settings with valid master_key
        settings = EncryptionSettings.for_development()
        encryptor = create_encryptor(settings)
        assert isinstance(encryptor, Encryptor)

    def test_create_field_encryptor(self):
        # Must provide encryptor with valid settings including search_hash_salt
        from pydantic import SecretStr
        settings = EncryptionSettings.for_development()
        settings = settings.model_copy(update={"search_hash_salt": SecretStr("test-search-hash-salt-for-dev!!")})
        encryptor = create_encryptor(settings)
        field_encryptor = create_field_encryptor(encryptor, settings=settings)
        assert isinstance(field_encryptor, FieldEncryptor)

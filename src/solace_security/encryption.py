"""Solace-AI Encryption - AES-256 encryption for PHI data protection."""

from __future__ import annotations
import base64
import hashlib
import hmac
import os
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)

NONCE_SIZE = 12
TAG_SIZE = 16
SALT_SIZE = 16
KEY_SIZE = 32
PBKDF2_ITERATIONS = 600000


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "AES-256-GCM"


class KeyDerivationFunction(str, Enum):
    """Supported key derivation functions."""

    PBKDF2_SHA256 = "PBKDF2-SHA256"


class EncryptionSettings(BaseSettings):
    """Encryption configuration from environment.

    SECURITY: master_key MUST be set via ENCRYPTION_MASTER_KEY environment variable.
    The key must be exactly 32 bytes for AES-256 security.
    """

    master_key: SecretStr = Field(
        ...,  # Required - no default for security
        description="AES-256 master encryption key (exactly 32 bytes). Set via ENCRYPTION_MASTER_KEY env var.",
    )
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)
    kdf: KeyDerivationFunction = Field(default=KeyDerivationFunction.PBKDF2_SHA256)
    kdf_iterations: int = Field(default=PBKDF2_ITERATIONS)
    enable_key_rotation: bool = Field(default=True)
    key_rotation_days: int = Field(default=90)
    model_config = SettingsConfigDict(
        env_prefix="ENCRYPTION_", env_file=".env", extra="ignore"
    )

    @classmethod
    def for_development(cls) -> "EncryptionSettings":
        """Create settings with a development-only key. NOT FOR PRODUCTION."""
        import warnings

        warnings.warn(
            "Using development EncryptionSettings with insecure key. NOT FOR PRODUCTION USE.",
            UserWarning,
            stacklevel=2,
        )
        return cls(master_key=SecretStr("dev-only-insecure-key-32-bytes!!"))

    def model_post_init(self, __context: Any) -> None:
        """Validate master key after initialization."""
        key_value = self.master_key.get_secret_value()
        if len(key_value) != KEY_SIZE:
            raise ValueError(
                f"ENCRYPTION_MASTER_KEY must be exactly {KEY_SIZE} bytes, got {len(key_value)}. "
                f'Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe({KEY_SIZE})[:32])"'
            )


class EncryptedData(BaseModel):
    """Container for encrypted data with metadata."""

    ciphertext: str = Field(..., description="Base64-encoded ciphertext")
    nonce: str = Field(..., description="Base64-encoded nonce/IV")
    salt: str | None = Field(default=None, description="Base64-encoded salt for KDF")
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)
    key_id: str | None = Field(default=None, description="Key identifier for rotation")
    version: int = Field(default=1, description="Encryption format version")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_compact(self) -> str:
        """Serialize to compact string format."""
        salt_part = self.salt or ""
        key_part = self.key_id or "default"
        return f"v{self.version}${self.algorithm.value}${key_part}${salt_part}${self.nonce}${self.ciphertext}"

    @classmethod
    def from_compact(cls, compact: str) -> EncryptedData:
        """Deserialize from compact string format."""
        parts = compact.split("$")
        if len(parts) != 6:
            raise ValueError("Invalid compact format")
        version = int(parts[0][1:])
        algorithm = EncryptionAlgorithm(parts[1])
        key_id = parts[2] if parts[2] != "default" else None
        salt = parts[3] if parts[3] else None
        nonce = parts[4]
        ciphertext = parts[5]
        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            algorithm=algorithm,
            key_id=key_id,
            version=version,
        )


class KeyManager:
    """Manages encryption keys and derivation."""

    def __init__(self, settings: EncryptionSettings | None = None) -> None:
        self._settings = settings or EncryptionSettings()
        self._master_key = self._settings.master_key.get_secret_value().encode()
        self._keys: dict[str, bytes] = {}
        self._current_key_id = "primary"

    def derive_key(self, salt: bytes, context: str | None = None) -> bytes:
        """Derive encryption key using PBKDF2."""
        info = context.encode() if context else b""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt + info,
            iterations=self._settings.kdf_iterations,
        )
        return kdf.derive(self._master_key)

    def generate_data_key(self) -> tuple[bytes, bytes]:
        """Generate a new data encryption key with its salt."""
        salt = os.urandom(SALT_SIZE)
        key = self.derive_key(salt)
        return key, salt

    def get_key(self, key_id: str, salt: bytes | None = None) -> bytes:
        """Get or derive key by ID."""
        if key_id in self._keys:
            return self._keys[key_id]
        if salt:
            return self.derive_key(salt, key_id)
        return self.derive_key(os.urandom(SALT_SIZE), key_id)

    def register_key(self, key_id: str, key: bytes) -> None:
        """Register a key for later use."""
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes")
        self._keys[key_id] = key

    @property
    def current_key_id(self) -> str:
        return self._current_key_id


class AESGCMCipher:
    """AES-256-GCM encryption implementation."""

    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes")
        self._aesgcm = AESGCM(key)

    def encrypt(
        self, plaintext: bytes, associated_data: bytes | None = None
    ) -> tuple[bytes, bytes]:
        """Encrypt data returning (ciphertext, nonce)."""
        nonce = os.urandom(NONCE_SIZE)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        return ciphertext, nonce

    def decrypt(
        self, ciphertext: bytes, nonce: bytes, associated_data: bytes | None = None
    ) -> bytes:
        """Decrypt data."""
        return self._aesgcm.decrypt(nonce, ciphertext, associated_data)


class Encryptor:
    """High-level encryption interface for PHI protection."""

    def __init__(self, settings: EncryptionSettings | None = None) -> None:
        self._settings = settings or EncryptionSettings()
        self._key_manager = KeyManager(self._settings)

    def encrypt(
        self, plaintext: str | bytes, context: str | None = None
    ) -> EncryptedData:
        """Encrypt plaintext data."""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        key, salt = self._key_manager.generate_data_key()
        cipher = AESGCMCipher(key)
        aad = context.encode() if context else None
        ciphertext, nonce = cipher.encrypt(plaintext, aad)
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode(),
            nonce=base64.b64encode(nonce).decode(),
            salt=base64.b64encode(salt).decode(),
            algorithm=self._settings.algorithm,
            key_id=self._key_manager.current_key_id,
        )

    def decrypt(self, encrypted: EncryptedData, context: str | None = None) -> bytes:
        """Decrypt encrypted data."""
        ciphertext = base64.b64decode(encrypted.ciphertext)
        nonce = base64.b64decode(encrypted.nonce)
        salt = (
            base64.b64decode(encrypted.salt)
            if encrypted.salt
            else os.urandom(SALT_SIZE)
        )
        key = self._key_manager.derive_key(salt)
        cipher = AESGCMCipher(key)
        aad = context.encode() if context else None
        return cipher.decrypt(ciphertext, nonce, aad)

    def decrypt_to_string(
        self, encrypted: EncryptedData, context: str | None = None
    ) -> str:
        """Decrypt and decode to string."""
        return self.decrypt(encrypted, context).decode("utf-8")

    def encrypt_dict(
        self, data: dict[str, Any], sensitive_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Encrypt sensitive fields in a dictionary."""
        result = data.copy()
        keys_to_encrypt = sensitive_keys or list(data.keys())
        for key in keys_to_encrypt:
            if key in result and result[key] is not None:
                value = (
                    str(result[key])
                    if not isinstance(result[key], str)
                    else result[key]
                )
                encrypted = self.encrypt(value, key)
                result[key] = encrypted.to_compact()
        return result

    def decrypt_dict(
        self, data: dict[str, Any], encrypted_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Decrypt encrypted fields in a dictionary."""
        result = data.copy()
        keys_to_decrypt = encrypted_keys or list(data.keys())
        for key in keys_to_decrypt:
            if (
                key in result
                and isinstance(result[key], str)
                and result[key].startswith("v")
            ):
                try:
                    encrypted = EncryptedData.from_compact(result[key])
                    result[key] = self.decrypt_to_string(encrypted, key)
                except (ValueError, Exception):
                    pass
        return result


class FieldEncryptor:
    """Encrypt specific fields with deterministic or randomized encryption."""

    def __init__(self, encryptor: Encryptor) -> None:
        self._encryptor = encryptor

    def encrypt_field(self, value: str, field_name: str) -> str:
        """Encrypt a single field value."""
        encrypted = self._encryptor.encrypt(value, field_name)
        return encrypted.to_compact()

    def decrypt_field(self, encrypted_value: str, field_name: str) -> str:
        """Decrypt a single field value."""
        encrypted = EncryptedData.from_compact(encrypted_value)
        return self._encryptor.decrypt_to_string(encrypted, field_name)

    def hash_for_search(self, value: str, salt: bytes | None = None) -> str:
        """Create searchable hash (deterministic, for equality search only)."""
        salt = salt or b"solace-search-salt"
        return hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()


class SecureTokenGenerator:
    """Generate secure tokens for various purposes."""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate URL-safe random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_hex_token(length: int = 32) -> str:
        """Generate hex token."""
        return secrets.token_hex(length)

    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate numeric OTP."""
        return "".join(str(secrets.randbelow(10)) for _ in range(length))

    @staticmethod
    def generate_key(size: int = KEY_SIZE) -> bytes:
        """Generate random key bytes."""
        return os.urandom(size)


def create_encryptor(settings: EncryptionSettings | None = None) -> Encryptor:
    """Factory function to create encryptor."""
    return Encryptor(settings)


def create_field_encryptor(encryptor: Encryptor | None = None) -> FieldEncryptor:
    """Factory function to create field encryptor."""
    return FieldEncryptor(encryptor or create_encryptor())

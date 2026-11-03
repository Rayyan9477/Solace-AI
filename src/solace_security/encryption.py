"""Solace-AI Encryption - AES-256 encryption for PHI data protection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

NONCE_SIZE = 12
TAG_SIZE = 16
SALT_SIZE = 16
KEY_SIZE = 32
PBKDF2_ITERATIONS = 600000

# MVP-092: accepted ENCRYPTION_MASTER_KEY encodings.
_MASTER_KEY_BASE64_PREFIX = "base64:"
_MASTER_KEY_HEX_PATTERN = re.compile(r"[0-9A-Fa-f]{64}")
_STRICT_ENVIRONMENTS = frozenset({"staging", "production"})
# Fixed text: it names the working formats and never includes the value, its length or a fragment.
_MASTER_KEY_FORMAT_ERROR = (
    "ENCRYPTION_MASTER_KEY must be exactly 32 bytes of key material: 64 hex characters "
    '(generate with: python -c "import secrets; print(secrets.token_hex(32))") '
    "or 'base64:' followed by base64 of 32 random bytes; a raw 32-character value "
    "is accepted only in development and test."
)


def _strict_environment() -> bool:
    """True in staging and production (the IF-01 ``is_strict_environment`` rule, read locally)."""
    return os.environ.get("ENVIRONMENT", "development").strip().lower() in _STRICT_ENVIRONMENTS


def parse_master_key(value: str, *, strict: bool) -> bytes:
    """Decode an ENCRYPTION_MASTER_KEY value into exactly 32 bytes of AES-256 key material.

    Accepted formats, checked in this order:

    1. ``base64:`` followed by standard base64 (validated) that decodes to exactly 32 bytes.
       A value carrying the prefix is always treated as base64; it never falls through.
    2. Exactly 64 hex characters (``secrets.token_hex(32)``), decoded with ``bytes.fromhex``.
    3. Only when ``strict`` is false: a value whose UTF-8 encoding is exactly 32 bytes, used
       as-is (legacy development/test keys; byte-identical to the pre-MVP-092 derivation).

    Raises ``ValueError`` with a fixed message that never contains the value.
    """
    if value.startswith(_MASTER_KEY_BASE64_PREFIX):
        try:
            decoded = base64.b64decode(value[len(_MASTER_KEY_BASE64_PREFIX):], validate=True)
        except ValueError:  # binascii.Error, or non-ASCII input
            decoded = b""
        if len(decoded) != KEY_SIZE:
            raise ValueError(_MASTER_KEY_FORMAT_ERROR) from None
        return decoded
    if _MASTER_KEY_HEX_PATTERN.fullmatch(value):
        return bytes.fromhex(value)
    if not strict:
        try:
            raw = value.encode("utf-8")
        except UnicodeEncodeError:  # e.g. lone surrogates; its message would quote the value
            raw = b""
        if len(raw) == KEY_SIZE:
            return raw
    raise ValueError(_MASTER_KEY_FORMAT_ERROR) from None


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "AES-256-GCM"


class KeyDerivationFunction(str, Enum):
    """Supported key derivation functions."""

    PBKDF2_SHA256 = "PBKDF2-SHA256"


class EncryptionSettings(BaseSettings):
    """Encryption configuration from environment.

    SECURITY: master_key MUST be set via ENCRYPTION_MASTER_KEY environment variable.
    It must carry exactly 32 bytes of key material for AES-256: 64 hex characters or
    ``base64:<base64 of 32 bytes>``; a raw 32-character value is accepted only outside
    staging/production (see ``parse_master_key``). The decoded bytes are exposed as
    ``master_key_bytes``.
    """

    master_key: SecretStr = Field(
        ...,  # Required - no default for security
        description=(
            "AES-256 master encryption key: 64 hex characters or 'base64:<32 bytes>' "
            "(raw 32-character value in development/test only). Set via ENCRYPTION_MASTER_KEY env var."
        ),
    )
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)
    kdf: KeyDerivationFunction = Field(default=KeyDerivationFunction.PBKDF2_SHA256)
    kdf_iterations: int = Field(default=PBKDF2_ITERATIONS)
    search_hash_salt: SecretStr = Field(
        default=SecretStr(""),
        description="Salt for deterministic search hashes. Set via ENCRYPTION_SEARCH_HASH_SALT env var.",
    )
    enable_key_rotation: bool = Field(default=True)
    key_rotation_days: int = Field(default=90)
    # hide_input_in_errors: defence in depth for str()/repr() of any ValidationError built from
    # this model; __init__ below keeps the ValidationError (whose errors()/json() still carry
    # the input, i.e. the key material) from escaping at all.
    model_config = SettingsConfigDict(
        env_prefix="ENCRYPTION_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    def __init__(self, **values: Any) -> None:
        """Build settings; a validation failure surfaces as a plain ValueError without input.

        pydantic's ValidationError keeps the raw input (here: ENCRYPTION_MASTER_KEY, whether
        passed as a kwarg or read from the environment) in ``errors()`` and ``json()``. Only
        the messages are kept, and the ValueError is raised outside the ``except`` block so it
        carries no ``__context__`` pointing back at the ValidationError.
        """
        failure: ValueError | None = None
        try:
            super().__init__(**values)
        except ValidationError as exc:
            failure = ValueError(
                "; ".join(
                    # Field names are not secret and make "Field required" actionable; values never appear.
                    f"{'.'.join(str(p) for p in e['loc']) or 'settings'}: {e['msg']}"
                    for e in exc.errors(include_input=False, include_context=False)
                )
            )
        if failure is not None:
            raise failure

    @classmethod
    def for_development(cls) -> EncryptionSettings:
        """Create settings with a development-only key. NOT FOR STAGING OR PRODUCTION."""
        import warnings

        if _strict_environment():
            raise RuntimeError(
                "EncryptionSettings.for_development() cannot be used in staging or production. "
                "Set ENCRYPTION_MASTER_KEY environment variable with a secure key."
            )
        warnings.warn(
            "Using development EncryptionSettings with insecure key. NOT FOR PRODUCTION USE.",
            UserWarning,
            stacklevel=2,
        )
        return cls(master_key=SecretStr("dev-only-insecure-key-32-bytes!!"))

    def model_post_init(self, __context: Any) -> None:
        """Validate the master key after initialization (fail at construction, not first use).

        MVP-092: the value is decoded by ``parse_master_key`` (64 hex characters,
        ``base64:`` + 32 bytes, or a legacy raw 32-byte value outside staging/production).
        H-38: a legacy raw value is measured in UTF-8 BYTES, not characters, so a
        32-character string with multi-byte glyphs is rejected.
        """
        parse_master_key(self.master_key.get_secret_value(), strict=_strict_environment())

    @property
    def master_key_bytes(self) -> bytes:
        """The decoded 32-byte AES-256 master key material.

        Re-derived on every access so it always matches ``master_key`` (``model_copy(update=...)``
        skips ``model_post_init``); construction-time validation still happens in model_post_init.
        """
        return parse_master_key(self.master_key.get_secret_value(), strict=_strict_environment())


class EncryptedData(BaseModel):
    """Container for encrypted data with metadata."""

    ciphertext: str = Field(..., description="Base64-encoded ciphertext")
    nonce: str = Field(..., description="Base64-encoded nonce/IV")
    salt: str | None = Field(default=None, description="Base64-encoded salt for KDF")
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)
    key_id: str | None = Field(default=None, description="Key identifier for rotation")
    version: int = Field(default=1, description="Encryption format version")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
        self._master_key = self._settings.master_key_bytes
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
        if not encrypted.salt:
            raise ValueError(
                "Cannot decrypt: missing salt in encrypted data. "
                "Data may be corrupted or from an incompatible encryption version."
            )
        salt = base64.b64decode(encrypted.salt)
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
                except ValueError as e:
                    logger.error("decrypt_field_failed", field=key, error=str(e))
                    raise
                except Exception as e:
                    logger.error("decrypt_field_unexpected_error", field=key, error=str(e))
                    raise
        return result


class FieldEncryptor:
    """Encrypt specific fields with deterministic or randomized encryption."""

    def __init__(self, encryptor: Encryptor, settings: EncryptionSettings | None = None) -> None:
        self._encryptor = encryptor
        self._settings = settings or encryptor._settings
        search_salt_value = self._settings.search_hash_salt.get_secret_value()
        if not search_salt_value:
            raise ValueError(
                "ENCRYPTION_SEARCH_HASH_SALT must be configured. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        self._search_salt = search_salt_value.encode()

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
        effective_salt = salt or self._search_salt
        return hmac.new(effective_salt, value.encode(), hashlib.sha256).hexdigest()


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


def create_field_encryptor(
    encryptor: Encryptor | None = None,
    settings: EncryptionSettings | None = None,
) -> FieldEncryptor:
    """Factory function to create field encryptor."""
    enc = encryptor or create_encryptor(settings)
    return FieldEncryptor(enc, settings)

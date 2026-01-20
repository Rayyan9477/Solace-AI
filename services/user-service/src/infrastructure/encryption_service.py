"""
Solace-AI User Service - Encryption Service.

Provides field-level encryption for sensitive data (HIPAA compliance).
Supports encryption of PII/PHI fields for secure storage.
"""
from __future__ import annotations

import structlog
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class EncryptionConfig(BaseModel):
    """Encryption service configuration."""
    encryption_key: bytes = Field(..., description="Fernet encryption key for field encryption")


class EncryptionError(Exception):
    """Base exception for encryption errors."""
    pass


class DecryptionError(EncryptionError):
    """Raised when decryption fails."""
    pass


class EncryptionService:
    """
    Encryption Service for field-level data encryption.

    Features:
    - Symmetric encryption (Fernet/AES-128)
    - HIPAA-compliant encryption
    - Field-level encryption for PII/PHI
    - Secure key management

    Use Cases:
    - Social Security Numbers (SSN)
    - Medical Record Numbers (MRN)
    - Sensitive notes or messages
    - Payment information
    - Personal identifiers
    """

    def __init__(self, config: EncryptionConfig):
        """
        Initialize encryption service.

        Args:
            config: Encryption configuration
        """
        self.config = config
        self.cipher = Fernet(config.encryption_key)
        self.logger = structlog.get_logger(__name__)

    def encrypt_field(self, value: str) -> str:
        """
        Encrypt sensitive field value.

        Args:
            value: Plain text value to encrypt

        Returns:
            Encrypted value (base64 encoded)

        Example:
            encrypted_ssn = service.encrypt_field("123-45-6789")
        """
        if not value:
            return value

        try:
            encrypted = self.cipher.encrypt(value.encode())
            encrypted_str = encrypted.decode()

            self.logger.debug(
                "field_encrypted",
                value_length=len(value),
                encrypted_length=len(encrypted_str),
            )

            return encrypted_str

        except Exception as e:
            self.logger.error("encryption_failed", error=str(e))
            raise EncryptionError(f"Failed to encrypt field: {str(e)}") from e

    def decrypt_field(self, encrypted_value: str) -> str:
        """
        Decrypt encrypted field value.

        Args:
            encrypted_value: Encrypted value to decrypt

        Returns:
            Plain text value

        Raises:
            DecryptionError: If decryption fails (invalid key or tampered data)

        Example:
            ssn = service.decrypt_field(encrypted_ssn)
        """
        if not encrypted_value:
            return encrypted_value

        try:
            decrypted = self.cipher.decrypt(encrypted_value.encode())
            value = decrypted.decode()

            self.logger.debug(
                "field_decrypted",
                encrypted_length=len(encrypted_value),
                value_length=len(value),
            )

            return value

        except InvalidToken as e:
            self.logger.warning("decryption_invalid_token", error=str(e))
            raise DecryptionError("Invalid encryption key or tampered data") from e
        except Exception as e:
            self.logger.error("decryption_failed", error=str(e))
            raise DecryptionError(f"Failed to decrypt field: {str(e)}") from e

    def encrypt_dict_fields(
        self,
        data: dict,
        fields_to_encrypt: list[str],
    ) -> dict:
        """
        Encrypt specific fields in a dictionary.

        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt

        Returns:
            Dictionary with encrypted fields

        Example:
            user_data = {"name": "John", "ssn": "123-45-6789", "email": "john@example.com"}
            encrypted_data = service.encrypt_dict_fields(user_data, ["ssn"])
        """
        encrypted_data = data.copy()

        for field in fields_to_encrypt:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt_field(str(encrypted_data[field]))

        self.logger.debug(
            "dict_fields_encrypted",
            total_fields=len(data),
            encrypted_fields=len(fields_to_encrypt),
        )

        return encrypted_data

    def decrypt_dict_fields(
        self,
        encrypted_data: dict,
        fields_to_decrypt: list[str],
    ) -> dict:
        """
        Decrypt specific fields in a dictionary.

        Args:
            encrypted_data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt

        Returns:
            Dictionary with decrypted fields

        Raises:
            DecryptionError: If any field decryption fails

        Example:
            user_data = service.decrypt_dict_fields(encrypted_data, ["ssn"])
        """
        decrypted_data = encrypted_data.copy()

        for field in fields_to_decrypt:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt_field(decrypted_data[field])

        self.logger.debug(
            "dict_fields_decrypted",
            total_fields=len(encrypted_data),
            decrypted_fields=len(fields_to_decrypt),
        )

        return decrypted_data

    def rotate_encryption(
        self,
        encrypted_value: str,
        old_key: bytes,
    ) -> str:
        """
        Re-encrypt value with current key (for key rotation).

        Args:
            encrypted_value: Value encrypted with old key
            old_key: Old encryption key

        Returns:
            Value encrypted with current key

        Raises:
            DecryptionError: If decryption with old key fails
            EncryptionError: If encryption with new key fails
        """
        # Decrypt with old key
        old_cipher = Fernet(old_key)
        try:
            decrypted = old_cipher.decrypt(encrypted_value.encode())
            value = decrypted.decode()
        except Exception as e:
            raise DecryptionError(f"Failed to decrypt with old key: {str(e)}") from e

        # Encrypt with new key
        try:
            new_encrypted = self.encrypt_field(value)

            self.logger.info(
                "encryption_rotated",
                old_length=len(encrypted_value),
                new_length=len(new_encrypted),
            )

            return new_encrypted
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt with new key: {str(e)}") from e


def create_encryption_service(encryption_key: bytes | None = None) -> EncryptionService:
    """
    Factory function to create encryption service.

    Args:
        encryption_key: Fernet encryption key (generates new if None)

    Returns:
        Configured EncryptionService instance

    Note:
        In production, encryption key should come from secure key management
        service (e.g., AWS KMS, Azure Key Vault, HashiCorp Vault)
    """
    if encryption_key is None:
        encryption_key = Fernet.generate_key()
        logger.warning(
            "generated_new_encryption_key",
            message="Using generated key - in production use secure key management",
        )

    config = EncryptionConfig(encryption_key=encryption_key)

    return EncryptionService(config)

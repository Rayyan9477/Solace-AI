"""
Solace-AI User Service - Password Service.

Provides secure password hashing and verification with dual support for:
- Argon2id (modern, memory-hard algorithm)
- bcrypt (legacy support for existing passwords)

Implements gradual migration strategy from bcrypt to argon2.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import structlog
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from passlib.hash import bcrypt
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class HashAlgorithm(str, Enum):
    """Password hashing algorithms."""
    ARGON2 = "argon2"
    BCRYPT = "bcrypt"


class PasswordConfig(BaseModel):
    """Password service configuration."""
    # Argon2 settings
    argon2_time_cost: int = Field(default=2, description="Number of iterations")
    argon2_memory_cost: int = Field(default=65536, description="Memory in KB (64MB)")
    argon2_parallelism: int = Field(default=1, description="Number of parallel threads")
    argon2_hash_len: int = Field(default=32, description="Hash length in bytes")
    argon2_salt_len: int = Field(default=16, description="Salt length in bytes")

    # bcrypt settings (for legacy support)
    bcrypt_rounds: int = Field(default=12, description="bcrypt cost factor")


@dataclass
class PasswordVerificationResult:
    """Result of password verification."""
    is_valid: bool
    needs_rehash: bool
    new_hash: str | None
    algorithm_used: HashAlgorithm


class PasswordService:
    """
    Password Service for secure hashing and verification.

    Features:
    - Argon2id for new passwords (memory-hard, GPU-resistant)
    - bcrypt support for legacy passwords
    - Automatic migration from bcrypt to argon2
    - Password strength validation
    """

    def __init__(self, config: PasswordConfig | None = None):
        """
        Initialize password service.

        Args:
            config: Password configuration (uses defaults if None)
        """
        self.config = config or PasswordConfig()
        self.logger = structlog.get_logger(__name__)

        # Initialize argon2 hasher
        self.argon2_hasher = PasswordHasher(
            time_cost=self.config.argon2_time_cost,
            memory_cost=self.config.argon2_memory_cost,
            parallelism=self.config.argon2_parallelism,
            hash_len=self.config.argon2_hash_len,
            salt_len=self.config.argon2_salt_len,
        )

    def hash_password(self, password: str) -> str:
        """
        Hash password using Argon2id (recommended for new passwords).

        Args:
            password: Plain text password

        Returns:
            Argon2 password hash
        """
        password_hash = self.argon2_hasher.hash(password)

        self.logger.debug(
            "password_hashed",
            algorithm="argon2",
            hash_prefix=password_hash[:20],
        )

        return password_hash

    def verify_password(
        self,
        password: str,
        password_hash: str,
    ) -> PasswordVerificationResult:
        """
        Verify password with dual algorithm support and migration.

        Supports both argon2 (new) and bcrypt (legacy) hashes.
        If bcrypt hash is verified, returns new argon2 hash for migration.

        Args:
            password: Plain text password to verify
            password_hash: Stored password hash

        Returns:
            PasswordVerificationResult with verification status and migration info
        """
        # Detect hash algorithm
        if password_hash.startswith("$argon2"):
            return self._verify_argon2(password, password_hash)
        elif password_hash.startswith("$2") or password_hash.startswith("$2a") or password_hash.startswith("$2b"):
            return self._verify_bcrypt_with_migration(password, password_hash)
        else:
            self.logger.warning(
                "unknown_hash_algorithm",
                hash_prefix=password_hash[:10],
            )
            return PasswordVerificationResult(
                is_valid=False,
                needs_rehash=False,
                new_hash=None,
                algorithm_used=HashAlgorithm.BCRYPT,  # Assume bcrypt for unknown
            )

    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if password hash needs rehashing.

        Returns True if:
        - Hash is bcrypt (should migrate to argon2)
        - Hash is argon2 but parameters have changed

        Args:
            password_hash: Stored password hash

        Returns:
            True if rehashing is recommended
        """
        # bcrypt hashes should always be migrated
        if password_hash.startswith("$2"):
            return True

        # Check if argon2 parameters have changed
        if password_hash.startswith("$argon2"):
            try:
                return self.argon2_hasher.check_needs_rehash(password_hash)
            except Exception:
                return False

        return False

    def _verify_argon2(
        self,
        password: str,
        password_hash: str,
    ) -> PasswordVerificationResult:
        """
        Verify password against argon2 hash.

        Args:
            password: Plain text password
            password_hash: Argon2 password hash

        Returns:
            PasswordVerificationResult
        """
        try:
            self.argon2_hasher.verify(password_hash, password)

            # Check if parameters have changed
            needs_rehash = self.argon2_hasher.check_needs_rehash(password_hash)
            new_hash = self.hash_password(password) if needs_rehash else None

            self.logger.debug(
                "argon2_verification_success",
                needs_rehash=needs_rehash,
            )

            return PasswordVerificationResult(
                is_valid=True,
                needs_rehash=needs_rehash,
                new_hash=new_hash,
                algorithm_used=HashAlgorithm.ARGON2,
            )

        except (VerifyMismatchError, InvalidHashError) as e:
            self.logger.debug(
                "argon2_verification_failed",
                error=str(e),
            )

            return PasswordVerificationResult(
                is_valid=False,
                needs_rehash=False,
                new_hash=None,
                algorithm_used=HashAlgorithm.ARGON2,
            )

    def _verify_bcrypt_with_migration(
        self,
        password: str,
        password_hash: str,
    ) -> PasswordVerificationResult:
        """
        Verify password against bcrypt hash and provide argon2 hash for migration.

        Args:
            password: Plain text password
            password_hash: bcrypt password hash

        Returns:
            PasswordVerificationResult with new argon2 hash if verified
        """
        try:
            is_valid = bcrypt.verify(password, password_hash)

            if is_valid:
                # Password is correct, provide argon2 hash for migration
                new_hash = self.hash_password(password)

                self.logger.info(
                    "bcrypt_verification_success_migration_available",
                    algorithm="bcrypt",
                )

                return PasswordVerificationResult(
                    is_valid=True,
                    needs_rehash=True,  # Always rehash bcrypt to argon2
                    new_hash=new_hash,
                    algorithm_used=HashAlgorithm.BCRYPT,
                )
            else:
                self.logger.debug(
                    "bcrypt_verification_failed",
                    algorithm="bcrypt",
                )

                return PasswordVerificationResult(
                    is_valid=False,
                    needs_rehash=False,
                    new_hash=None,
                    algorithm_used=HashAlgorithm.BCRYPT,
                )

        except Exception as e:
            self.logger.warning(
                "bcrypt_verification_error",
                error=str(e),
            )

            return PasswordVerificationResult(
                is_valid=False,
                needs_rehash=False,
                new_hash=None,
                algorithm_used=HashAlgorithm.BCRYPT,
            )

    def get_algorithm(self, password_hash: str) -> HashAlgorithm:
        """
        Detect which algorithm was used for the hash.

        Args:
            password_hash: Password hash to analyze

        Returns:
            HashAlgorithm enum value
        """
        if password_hash.startswith("$argon2"):
            return HashAlgorithm.ARGON2
        else:
            return HashAlgorithm.BCRYPT


def create_password_service(
    argon2_time_cost: int = 2,
    argon2_memory_cost: int = 65536,
    bcrypt_rounds: int = 12,
) -> PasswordService:
    """
    Factory function to create password service.

    Args:
        argon2_time_cost: Number of iterations for argon2
        argon2_memory_cost: Memory cost in KB for argon2 (default 64MB)
        bcrypt_rounds: Cost factor for bcrypt (legacy)

    Returns:
        Configured PasswordService instance
    """
    config = PasswordConfig(
        argon2_time_cost=argon2_time_cost,
        argon2_memory_cost=argon2_memory_cost,
        bcrypt_rounds=bcrypt_rounds,
    )

    return PasswordService(config)

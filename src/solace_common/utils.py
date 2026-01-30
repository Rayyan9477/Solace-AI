"""Solace-AI Common Utilities - DateTime, Crypto, Validation, String, Retry, Collection."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import unicodedata
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import wraps
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import structlog

logger = structlog.get_logger(__name__)


class DateTimeUtils:
    """Timezone-aware datetime utilities."""

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC datetime with timezone info."""
        return datetime.now(timezone.utc)

    @staticmethod
    def ensure_utc(dt: datetime) -> datetime:
        """Ensure datetime is UTC. Convert if necessary."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def to_timestamp(dt: datetime) -> float:
        """Convert datetime to Unix timestamp."""
        return DateTimeUtils.ensure_utc(dt).timestamp()

    @staticmethod
    def from_timestamp(ts: float) -> datetime:
        """Create UTC datetime from Unix timestamp."""
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    @staticmethod
    def to_iso_string(dt: datetime) -> str:
        """Convert datetime to ISO 8601 string."""
        return DateTimeUtils.ensure_utc(dt).isoformat()

    @staticmethod
    def from_iso_string(iso_str: str) -> datetime:
        """Parse ISO 8601 string to UTC datetime."""
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return DateTimeUtils.ensure_utc(dt)

    @staticmethod
    def add_duration(dt: datetime, days: int = 0, hours: int = 0,
                     minutes: int = 0, seconds: int = 0) -> datetime:
        """Add duration to datetime."""
        return dt + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def is_expired(dt: datetime, ttl_seconds: int) -> bool:
        """Check if datetime has expired based on TTL."""
        expiry = DateTimeUtils.ensure_utc(dt) + timedelta(seconds=ttl_seconds)
        return DateTimeUtils.utc_now() > expiry

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"


class CryptoUtils:
    """Cryptographic operations for security-sensitive data."""

    HASH_ALGORITHM = "sha256"
    HMAC_ALGORITHM = "sha256"

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_secret_key(length: int = 32) -> bytes:
        """Generate cryptographically secure random bytes."""
        return secrets.token_bytes(length)

    @staticmethod
    def hash_value(value: str, salt: str = "") -> str:
        """Create SHA-256 hash of value with optional salt."""
        combined = f"{salt}{value}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Create SHA-256 hash of bytes."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hmac_sign(message: str, secret: str) -> str:
        """Create HMAC signature for message."""
        return hmac.new(
            secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def hmac_verify(message: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature using constant-time comparison."""
        expected = CryptoUtils.hmac_sign(message, secret)
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def base64_encode(data: bytes) -> str:
        """Encode bytes to URL-safe base64 string."""
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    @staticmethod
    def base64_decode(encoded: str) -> bytes:
        """Decode URL-safe base64 string to bytes."""
        padding = 4 - (len(encoded) % 4)
        if padding != 4:
            encoded += "=" * padding
        return base64.urlsafe_b64decode(encoded.encode("utf-8"))

    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """Compare strings in constant time to prevent timing attacks."""
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class ValidationPatterns:
    """Common validation regex patterns."""

    EMAIL = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    PHONE_E164 = re.compile(r"^\+[1-9]\d{1,14}$")
    ALPHANUMERIC = re.compile(r"^[a-zA-Z0-9]+$")
    SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    SAFE_STRING = re.compile(r"^[\w\s\-.,!?@#$%&*()+=:;\"']+$", re.UNICODE)
    SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class ValidationUtils:
    """Input validation utilities."""

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        return bool(ValidationPatterns.EMAIL.match(email.strip().lower()))

    @staticmethod
    def is_valid_uuid(value: str) -> bool:
        """Validate UUID format."""
        return bool(ValidationPatterns.UUID.match(value.strip()))

    @staticmethod
    def is_valid_phone_e164(phone: str) -> bool:
        """Validate E.164 phone format."""
        return bool(ValidationPatterns.PHONE_E164.match(phone.strip()))

    @staticmethod
    def is_valid_slug(slug: str) -> bool:
        """Validate URL slug format."""
        return bool(ValidationPatterns.SLUG.match(slug.strip()))

    @staticmethod
    def is_valid_sql_identifier(name: str, max_length: int = 128) -> bool:
        """Validate that a string is a safe SQL identifier.

        Prevents SQL injection by ensuring names only contain alphanumeric
        characters and underscores, starting with a letter or underscore.
        """
        if not name or len(name) > max_length:
            return False
        return ValidationPatterns.SQL_IDENTIFIER.match(name) is not None

    @staticmethod
    def validate_sql_identifier(name: str, identifier_type: str = "identifier") -> None:
        """Validate SQL identifier, raising ValueError if invalid.

        Args:
            name: The identifier to validate.
            identifier_type: Description for error messages (e.g. "table name", "column name").

        Raises:
            ValueError: If the identifier is invalid.
        """
        if not name:
            raise ValueError(f"Empty {identifier_type} is not allowed")
        if len(name) > 128:
            raise ValueError(f"{identifier_type} exceeds maximum length: {name}")
        if not ValidationPatterns.SQL_IDENTIFIER.match(name):
            raise ValueError(
                f"Invalid {identifier_type}: {name}. "
                "Must contain only letters, digits, and underscores, "
                "starting with a letter or underscore."
            )

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000,
                        allow_newlines: bool = False) -> str:
        """Sanitize string: remove control chars, normalize unicode, truncate."""
        normalized = unicodedata.normalize("NFKC", value)
        if allow_newlines:
            sanitized = re.sub(r"[^\x20-\x7E\n\r\t\u00A0-\uFFFF]", "", normalized)
        else:
            sanitized = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", normalized)
        sanitized = " ".join(sanitized.split()) if not allow_newlines else sanitized
        return sanitized[:max_length]

    @staticmethod
    def validate_range(value: int | float | Decimal,
                       min_value: int | float | Decimal | None = None,
                       max_value: int | float | Decimal | None = None) -> bool:
        """Validate numeric value is within range."""
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    @staticmethod
    def validate_length(value: str, min_length: int = 0,
                        max_length: int | None = None) -> bool:
        """Validate string length is within bounds."""
        length = len(value)
        if length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True


class StringUtils:
    """String manipulation utilities."""

    @staticmethod
    def to_snake_case(value: str) -> str:
        """Convert string to snake_case."""
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def to_camel_case(value: str) -> str:
        """Convert string to camelCase."""
        components = value.replace("-", "_").split("_")
        return components[0].lower() + "".join(x.title() for x in components[1:])

    @staticmethod
    def to_pascal_case(value: str) -> str:
        """Convert string to PascalCase."""
        components = value.replace("-", "_").split("_")
        return "".join(x.title() for x in components)

    @staticmethod
    def to_slug(value: str) -> str:
        """Convert string to URL slug."""
        normalized = unicodedata.normalize("NFKD", value)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        slugified = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())
        return slugified.strip("-")

    @staticmethod
    def truncate(value: str, max_length: int, suffix: str = "...") -> str:
        """Truncate string to max length with suffix."""
        if len(value) <= max_length:
            return value
        return value[: max_length - len(suffix)] + suffix

    @staticmethod
    def mask_sensitive(value: str, visible_chars: int = 4, mask_char: str = "*") -> str:
        """Mask sensitive string, showing only last N characters."""
        if len(value) <= visible_chars:
            return mask_char * len(value)
        masked_length = len(value) - visible_chars
        return mask_char * masked_length + value[-visible_chars:]


P = ParamSpec("P")
R = TypeVar("R")


class RetryConfig(BaseModel):
    """Configuration for retry behavior with exponential backoff."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_ms: int = Field(default=100, ge=10, le=10000)
    max_delay_ms: int = Field(default=10000, ge=100, le=60000)
    exponential_base: float = Field(default=2.0, ge=1.5, le=4.0)
    jitter: bool = Field(default=True)
    model_config = ConfigDict(frozen=True)

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay_ms * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay_ms)
        if self.jitter:
            import random
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        return int(delay)


async def retry_async(
    func: Callable[P, Awaitable[R]],
    config: RetryConfig | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[P, Awaitable[R]]:
    """Decorator for async function retry with exponential backoff."""
    import asyncio
    retry_config = config or RetryConfig()

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        last_exception: Exception | None = None
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e
                if attempt < retry_config.max_attempts:
                    delay_ms = retry_config.get_delay_ms(attempt)
                    logger.warning("Retry attempt", attempt=attempt,
                                   max_attempts=retry_config.max_attempts,
                                   delay_ms=delay_ms, error=str(e))
                    await asyncio.sleep(delay_ms / 1000)
        logger.error("All retry attempts exhausted",
                     max_attempts=retry_config.max_attempts, error=str(last_exception))
        raise last_exception  # type: ignore[misc]
    return wrapper  # type: ignore[return-value]


class CollectionUtils:
    """Utilities for working with collections."""

    @staticmethod
    def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
        """Split list into chunks of specified size."""
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    @staticmethod
    def flatten(nested: list[list[Any]]) -> list[Any]:
        """Flatten nested list one level."""
        return [item for sublist in nested for item in sublist]

    @staticmethod
    def deduplicate(items: list[Any], key: Callable[[Any], Any] | None = None) -> list[Any]:
        """Remove duplicates while preserving order."""
        seen: set[Any] = set()
        result: list[Any] = []
        for item in items:
            k = key(item) if key else item
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result

    @staticmethod
    def safe_get(data: dict[str, Any], path: str, default: Any = None,
                 separator: str = ".") -> Any:
        """Safely get nested dictionary value by dot-notation path."""
        keys = path.split(separator)
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

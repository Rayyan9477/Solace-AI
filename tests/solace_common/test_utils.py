"""
Unit tests for Solace-AI Utils Module.
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from solace_common.utils import (
    CollectionUtils,
    CryptoUtils,
    DateTimeUtils,
    RetryConfig,
    StringUtils,
    ValidationPatterns,
    ValidationUtils,
)


class TestDateTimeUtils:
    """Tests for DateTimeUtils."""

    def test_utc_now(self) -> None:
        """Test utc_now returns timezone-aware datetime."""
        now = DateTimeUtils.utc_now()

        assert now.tzinfo == timezone.utc
        assert (datetime.now(timezone.utc) - now).total_seconds() < 1

    def test_ensure_utc_naive(self) -> None:
        """Test ensure_utc converts naive datetime."""
        naive = datetime(2024, 1, 15, 12, 0, 0)
        aware = DateTimeUtils.ensure_utc(naive)

        assert aware.tzinfo == timezone.utc
        assert aware.hour == 12

    def test_ensure_utc_already_utc(self) -> None:
        """Test ensure_utc preserves UTC datetime."""
        utc_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = DateTimeUtils.ensure_utc(utc_dt)

        assert result == utc_dt

    def test_to_timestamp_and_back(self) -> None:
        """Test timestamp conversion round-trip."""
        original = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)

        ts = DateTimeUtils.to_timestamp(original)
        restored = DateTimeUtils.from_timestamp(ts)

        assert restored == original

    def test_to_iso_string(self) -> None:
        """Test ISO string conversion."""
        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        iso_str = DateTimeUtils.to_iso_string(dt)

        assert "2024-01-15" in iso_str
        assert "12:30:45" in iso_str

    def test_from_iso_string(self) -> None:
        """Test parsing ISO string."""
        iso_str = "2024-01-15T12:30:45+00:00"
        dt = DateTimeUtils.from_iso_string(iso_str)

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.tzinfo == timezone.utc

    def test_from_iso_string_with_z(self) -> None:
        """Test parsing ISO string with Z suffix."""
        iso_str = "2024-01-15T12:30:45Z"
        dt = DateTimeUtils.from_iso_string(iso_str)

        assert dt.tzinfo == timezone.utc

    def test_add_duration(self) -> None:
        """Test adding duration to datetime."""
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = DateTimeUtils.add_duration(dt, days=5, hours=2, minutes=30)

        assert result.day == 20
        assert result.hour == 14
        assert result.minute == 30

    def test_is_expired_true(self) -> None:
        """Test expiration check when expired."""
        old_dt = datetime.now(timezone.utc) - timedelta(hours=2)
        ttl_seconds = 3600  # 1 hour

        assert DateTimeUtils.is_expired(old_dt, ttl_seconds) is True

    def test_is_expired_false(self) -> None:
        """Test expiration check when not expired."""
        recent_dt = datetime.now(timezone.utc) - timedelta(minutes=30)
        ttl_seconds = 3600  # 1 hour

        assert DateTimeUtils.is_expired(recent_dt, ttl_seconds) is False

    def test_format_duration(self) -> None:
        """Test duration formatting."""
        assert DateTimeUtils.format_duration(45) == "45.0s"
        assert DateTimeUtils.format_duration(150) == "2.5m"
        assert DateTimeUtils.format_duration(7200) == "2.0h"
        assert DateTimeUtils.format_duration(172800) == "2.0d"


class TestCryptoUtils:
    """Tests for CryptoUtils."""

    def test_generate_token(self) -> None:
        """Test token generation."""
        token = CryptoUtils.generate_token(32)

        assert token is not None
        assert len(token) > 0

    def test_generate_token_unique(self) -> None:
        """Test tokens are unique."""
        tokens = [CryptoUtils.generate_token() for _ in range(100)]

        assert len(set(tokens)) == 100

    def test_generate_secret_key(self) -> None:
        """Test secret key generation."""
        key = CryptoUtils.generate_secret_key(32)

        assert len(key) == 32
        assert isinstance(key, bytes)

    def test_hash_value(self) -> None:
        """Test value hashing."""
        hash1 = CryptoUtils.hash_value("secret")
        hash2 = CryptoUtils.hash_value("secret")
        hash3 = CryptoUtils.hash_value("different")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex

    def test_hash_value_with_salt(self) -> None:
        """Test hashing with salt."""
        hash1 = CryptoUtils.hash_value("secret", salt="salt1")
        hash2 = CryptoUtils.hash_value("secret", salt="salt2")

        assert hash1 != hash2

    def test_hmac_sign_and_verify(self) -> None:
        """Test HMAC signing and verification."""
        message = "important data"
        secret = "my-secret-key"

        signature = CryptoUtils.hmac_sign(message, secret)

        assert CryptoUtils.hmac_verify(message, signature, secret) is True
        assert CryptoUtils.hmac_verify(message, signature, "wrong-secret") is False
        assert CryptoUtils.hmac_verify("tampered", signature, secret) is False

    def test_base64_encode_decode(self) -> None:
        """Test base64 round-trip."""
        original = b"binary data \x00\x01\x02"

        encoded = CryptoUtils.base64_encode(original)
        decoded = CryptoUtils.base64_decode(encoded)

        assert decoded == original

    def test_constant_time_compare(self) -> None:
        """Test constant-time string comparison."""
        assert CryptoUtils.constant_time_compare("abc", "abc") is True
        assert CryptoUtils.constant_time_compare("abc", "abd") is False


class TestValidationUtils:
    """Tests for ValidationUtils."""

    def test_is_valid_email(self) -> None:
        """Test email validation."""
        assert ValidationUtils.is_valid_email("test@example.com") is True
        assert ValidationUtils.is_valid_email("user.name+tag@domain.co.uk") is True
        assert ValidationUtils.is_valid_email("invalid") is False
        assert ValidationUtils.is_valid_email("@nodomain.com") is False
        assert ValidationUtils.is_valid_email("no@domain") is False

    def test_is_valid_uuid(self) -> None:
        """Test UUID validation."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"

        assert ValidationUtils.is_valid_uuid(valid_uuid) is True
        assert ValidationUtils.is_valid_uuid("not-a-uuid") is False
        assert ValidationUtils.is_valid_uuid("550e8400-e29b-41d4-a716") is False

    def test_is_valid_phone_e164(self) -> None:
        """Test E.164 phone validation."""
        assert ValidationUtils.is_valid_phone_e164("+12025551234") is True
        assert ValidationUtils.is_valid_phone_e164("+442071234567") is True
        assert ValidationUtils.is_valid_phone_e164("12025551234") is False
        assert ValidationUtils.is_valid_phone_e164("+0123") is False

    def test_is_valid_slug(self) -> None:
        """Test slug validation."""
        assert ValidationUtils.is_valid_slug("hello-world") is True
        assert ValidationUtils.is_valid_slug("hello123") is True
        assert ValidationUtils.is_valid_slug("Hello-World") is False
        assert ValidationUtils.is_valid_slug("hello_world") is False

    def test_sanitize_string(self) -> None:
        """Test string sanitization."""
        result = ValidationUtils.sanitize_string("Hello\x00World\x01")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_sanitize_string_max_length(self) -> None:
        """Test string truncation."""
        long_string = "a" * 2000
        result = ValidationUtils.sanitize_string(long_string, max_length=100)

        assert len(result) == 100

    def test_validate_range(self) -> None:
        """Test range validation."""
        assert ValidationUtils.validate_range(5, min_value=1, max_value=10) is True
        assert ValidationUtils.validate_range(0, min_value=1, max_value=10) is False
        assert ValidationUtils.validate_range(15, min_value=1, max_value=10) is False
        assert ValidationUtils.validate_range(5, min_value=1) is True
        assert ValidationUtils.validate_range(5, max_value=10) is True

    def test_validate_length(self) -> None:
        """Test length validation."""
        assert ValidationUtils.validate_length("hello", min_length=1, max_length=10) is True
        assert ValidationUtils.validate_length("", min_length=1) is False
        assert ValidationUtils.validate_length("hello world", max_length=5) is False


class TestStringUtils:
    """Tests for StringUtils."""

    def test_to_snake_case(self) -> None:
        """Test conversion to snake_case."""
        assert StringUtils.to_snake_case("HelloWorld") == "hello_world"
        assert StringUtils.to_snake_case("helloWorld") == "hello_world"
        assert StringUtils.to_snake_case("hello-world") == "hello-world"

    def test_to_camel_case(self) -> None:
        """Test conversion to camelCase."""
        assert StringUtils.to_camel_case("hello_world") == "helloWorld"
        assert StringUtils.to_camel_case("hello-world") == "helloWorld"
        assert StringUtils.to_camel_case("HELLO_WORLD") == "helloWorld"

    def test_to_pascal_case(self) -> None:
        """Test conversion to PascalCase."""
        assert StringUtils.to_pascal_case("hello_world") == "HelloWorld"
        assert StringUtils.to_pascal_case("hello-world") == "HelloWorld"

    def test_to_slug(self) -> None:
        """Test slug generation."""
        assert StringUtils.to_slug("Hello World!") == "hello-world"
        assert StringUtils.to_slug("Café Latté") == "cafe-latte"
        assert StringUtils.to_slug("  Multiple   Spaces  ") == "multiple-spaces"

    def test_truncate(self) -> None:
        """Test string truncation."""
        assert StringUtils.truncate("Hello World", 5) == "He..."
        assert StringUtils.truncate("Hi", 10) == "Hi"
        assert StringUtils.truncate("Hello World", 8, suffix="..") == "Hello .."

    def test_mask_sensitive(self) -> None:
        """Test sensitive data masking."""
        assert StringUtils.mask_sensitive("1234567890", visible_chars=4) == "******7890"
        assert StringUtils.mask_sensitive("abc", visible_chars=4) == "***"
        assert StringUtils.mask_sensitive("secret", visible_chars=2, mask_char="#") == "####et"


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay_ms == 100
        assert config.exponential_base == 2.0

    def test_get_delay_exponential(self) -> None:
        """Test exponential delay calculation."""
        config = RetryConfig(
            initial_delay_ms=100,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.get_delay_ms(1) == 100
        assert config.get_delay_ms(2) == 200
        assert config.get_delay_ms(3) == 400

    def test_get_delay_max_cap(self) -> None:
        """Test delay is capped at max."""
        config = RetryConfig(
            initial_delay_ms=1000,
            max_delay_ms=5000,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.get_delay_ms(10) == 5000


class TestCollectionUtils:
    """Tests for CollectionUtils."""

    def test_chunk_list(self) -> None:
        """Test list chunking."""
        items = [1, 2, 3, 4, 5, 6, 7]
        chunks = CollectionUtils.chunk_list(items, 3)

        assert chunks == [[1, 2, 3], [4, 5, 6], [7]]

    def test_flatten(self) -> None:
        """Test nested list flattening."""
        nested = [[1, 2], [3, 4], [5]]
        result = CollectionUtils.flatten(nested)

        assert result == [1, 2, 3, 4, 5]

    def test_deduplicate(self) -> None:
        """Test deduplication preserving order."""
        items = [1, 2, 3, 2, 1, 4]
        result = CollectionUtils.deduplicate(items)

        assert result == [1, 2, 3, 4]

    def test_deduplicate_with_key(self) -> None:
        """Test deduplication with key function."""
        items = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 1, "v": "c"}]
        result = CollectionUtils.deduplicate(items, key=lambda x: x["id"])

        assert len(result) == 2
        assert result[0]["v"] == "a"  # First occurrence kept

    def test_safe_get(self) -> None:
        """Test safe nested dict access."""
        data = {"a": {"b": {"c": 42}}}

        assert CollectionUtils.safe_get(data, "a.b.c") == 42
        assert CollectionUtils.safe_get(data, "a.b") == {"c": 42}
        assert CollectionUtils.safe_get(data, "a.x.y") is None
        assert CollectionUtils.safe_get(data, "missing", default="default") == "default"

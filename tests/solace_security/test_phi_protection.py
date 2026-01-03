"""Unit tests for PHI protection module."""
from __future__ import annotations
import pytest
from solace_security.phi_protection import (
    PHIType,
    PHISettings,
    PHIMatch,
    PHIDetectionResult,
    PHIPattern,
    DEFAULT_PATTERNS,
    PHIMasker,
    PHIDetector,
    PHISanitizer,
    create_phi_detector,
    create_phi_sanitizer,
    detect_phi,
    mask_phi,
)


class TestPHIType:
    """Tests for PHIType enum."""

    def test_phi_types(self):
        assert PHIType.SSN.value == "ssn"
        assert PHIType.EMAIL.value == "email"
        assert PHIType.PHONE.value == "phone"
        assert PHIType.CREDIT_CARD.value == "credit_card"


class TestPHISettings:
    """Tests for PHISettings."""

    def test_default_settings(self):
        settings = PHISettings()
        assert settings.enable_detection
        assert settings.enable_masking
        assert settings.mask_character == "*"

    def test_custom_settings(self):
        settings = PHISettings(mask_character="X", partial_mask_visible=2)
        assert settings.mask_character == "X"
        assert settings.partial_mask_visible == 2


class TestPHIPattern:
    """Tests for PHIPattern model."""

    def test_create_pattern(self):
        pattern = PHIPattern(
            phi_type=PHIType.SSN,
            pattern=r"\d{3}-\d{2}-\d{4}",
            confidence=0.95
        )
        assert pattern.phi_type == PHIType.SSN
        assert pattern.confidence == 0.95


class TestDefaultPatterns:
    """Tests for default PHI patterns."""

    def test_patterns_exist(self):
        assert len(DEFAULT_PATTERNS) > 0
        phi_types = {p.phi_type for p in DEFAULT_PATTERNS}
        assert PHIType.SSN in phi_types
        assert PHIType.PHONE in phi_types
        assert PHIType.EMAIL in phi_types


class TestPHIMasker:
    """Tests for PHIMasker."""

    @pytest.fixture
    def masker(self):
        return PHIMasker()

    def test_mask_ssn(self, masker):
        masked = masker.mask("123-45-6789", PHIType.SSN)
        assert masked == "***-**-6789"

    def test_mask_email(self, masker):
        masked = masker.mask("john.doe@example.com", PHIType.EMAIL)
        assert "@example.com" in masked
        assert masked.startswith("j")

    def test_mask_phone(self, masker):
        masked = masker.mask("555-123-4567", PHIType.PHONE)
        assert masked.endswith("4567")
        assert "*" in masked

    def test_mask_credit_card(self, masker):
        masked = masker.mask("4111111111111111", PHIType.CREDIT_CARD)
        assert masked.endswith("1111")
        assert masked.startswith("*")

    def test_mask_name(self, masker):
        masked = masker.mask("John Doe", PHIType.NAME)
        assert masked.startswith("Jo")
        assert "*" in masked

    def test_mask_empty(self, masker):
        assert masker.mask("", PHIType.SSN) == ""

    def test_mask_custom_character(self):
        masker = PHIMasker(PHISettings(mask_character="X"))
        masked = masker.mask("123-45-6789", PHIType.SSN)
        assert "X" in masked


class TestPHIDetector:
    """Tests for PHIDetector."""

    @pytest.fixture
    def detector(self):
        return PHIDetector()

    def test_detect_ssn(self, detector):
        result = detector.detect("My SSN is 123-45-6789")
        assert result.contains_phi
        assert PHIType.SSN in result.phi_types_found

    def test_detect_phone(self, detector):
        result = detector.detect("Call me at 555-123-4567")
        assert result.contains_phi
        assert PHIType.PHONE in result.phi_types_found

    def test_detect_email(self, detector):
        result = detector.detect("Email me at john@example.com")
        assert result.contains_phi
        assert PHIType.EMAIL in result.phi_types_found

    def test_detect_credit_card(self, detector):
        result = detector.detect("Card: 4111-1111-1111-1111")
        assert result.contains_phi
        assert PHIType.CREDIT_CARD in result.phi_types_found

    def test_detect_ip_address(self, detector):
        result = detector.detect("Server at 192.168.1.100")
        assert result.contains_phi
        assert PHIType.IP_ADDRESS in result.phi_types_found

    def test_detect_multiple_phi(self, detector):
        text = "SSN: 123-45-6789, Phone: 555-123-4567, Email: test@example.com"
        result = detector.detect(text)
        assert result.contains_phi
        assert len(result.matches) >= 3

    def test_detect_no_phi(self, detector):
        result = detector.detect("Hello, this is a normal message.")
        assert not result.contains_phi
        assert len(result.matches) == 0

    def test_masked_text(self, detector):
        result = detector.detect("My SSN is 123-45-6789")
        assert "123-45-6789" not in result.masked_text
        assert "***-**-6789" in result.masked_text or "*" in result.masked_text

    def test_contains_phi(self, detector):
        assert detector.contains_phi("SSN: 123-45-6789")
        assert not detector.contains_phi("No PHI here")

    def test_detection_disabled(self):
        detector = PHIDetector(PHISettings(enable_detection=False))
        result = detector.detect("SSN: 123-45-6789")
        assert not result.contains_phi

    def test_masking_disabled(self):
        detector = PHIDetector(PHISettings(enable_masking=False))
        result = detector.detect("SSN: 123-45-6789")
        assert result.contains_phi
        assert result.masked_text == result.original_text


class TestPHIDetectionResult:
    """Tests for PHIDetectionResult model."""

    def test_match_count(self):
        match1 = PHIMatch(phi_type=PHIType.SSN, value="123-45-6789", masked_value="***-**-6789", start_pos=0, end_pos=11)
        match2 = PHIMatch(phi_type=PHIType.EMAIL, value="test@test.com", masked_value="t***@test.com", start_pos=20, end_pos=33)
        result = PHIDetectionResult(
            contains_phi=True,
            matches=[match1, match2],
            original_text="text",
            masked_text="masked"
        )
        assert result.match_count == 2


class TestPHISanitizer:
    """Tests for PHISanitizer."""

    @pytest.fixture
    def sanitizer(self):
        return PHISanitizer()

    def test_sanitize_string(self, sanitizer):
        text = "My SSN is 123-45-6789"
        sanitized = sanitizer.sanitize_string(text)
        assert "123-45-6789" not in sanitized

    def test_sanitize_dict(self, sanitizer):
        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
            "notes": "Contact at 555-123-4567"
        }
        sanitized = sanitizer.sanitize_dict(data)
        assert "123-45-6789" not in sanitized["ssn"]
        assert "555-123-4567" not in sanitized["notes"]

    def test_sanitize_dict_sensitive_keys(self, sanitizer):
        data = {"password": "secret123", "api_key": "key-12345"}
        sanitized = sanitizer.sanitize_dict(data, sensitive_keys=["password"])
        assert sanitized["password"] != "secret123" or "*" in sanitized["password"]

    def test_sanitize_nested_dict(self, sanitizer):
        data = {
            "user": {
                "email": "test@example.com",
                "profile": {
                    "phone": "555-123-4567"
                }
            }
        }
        sanitized = sanitizer.sanitize_dict(data)
        assert "555-123-4567" not in str(sanitized)

    def test_sanitize_list(self, sanitizer):
        data = ["123-45-6789", "normal text", "test@example.com"]
        sanitized = sanitizer.sanitize_list(data)
        assert "123-45-6789" not in sanitized[0]

    def test_sanitize_list_of_dicts(self, sanitizer):
        data = [
            {"ssn": "123-45-6789"},
            {"email": "test@example.com"}
        ]
        sanitized = sanitizer.sanitize_list(data)
        assert "123-45-6789" not in str(sanitized)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_phi(self):
        result = detect_phi("SSN: 123-45-6789")
        assert result.contains_phi
        assert isinstance(result, PHIDetectionResult)

    def test_mask_phi(self):
        masked = mask_phi("SSN: 123-45-6789")
        assert "123-45-6789" not in masked


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_phi_detector(self):
        detector = create_phi_detector()
        assert isinstance(detector, PHIDetector)

    def test_create_phi_detector_with_settings(self):
        settings = PHISettings(mask_character="X")
        detector = create_phi_detector(settings)
        result = detector.detect("SSN: 123-45-6789")
        assert "X" in result.masked_text

    def test_create_phi_sanitizer(self):
        sanitizer = create_phi_sanitizer()
        assert isinstance(sanitizer, PHISanitizer)


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def detector(self):
        return PHIDetector()

    def test_empty_string(self, detector):
        result = detector.detect("")
        assert not result.contains_phi

    def test_whitespace_only(self, detector):
        result = detector.detect("   \n\t   ")
        assert not result.contains_phi

    def test_special_characters(self, detector):
        result = detector.detect("!@#$%^&*()")
        assert not result.contains_phi

    def test_partial_ssn(self, detector):
        result = detector.detect("123-45")
        assert not result.contains_phi

    def test_medical_record_number(self, detector):
        result = detector.detect("MRN: 123456789")
        assert result.contains_phi
        assert PHIType.MEDICAL_RECORD in result.phi_types_found

    def test_overlapping_matches(self, detector):
        text = "123-45-6789 and 123-45-6789 again"
        result = detector.detect(text)
        assert result.contains_phi
        assert result.match_count >= 2

    def test_date_detection(self, detector):
        result = detector.detect("Born on 01/15/1990")
        assert result.contains_phi
        assert PHIType.DATE_OF_BIRTH in result.phi_types_found

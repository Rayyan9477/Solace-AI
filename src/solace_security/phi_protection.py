"""Solace-AI PHI Protection - Detection and masking of Protected Health Information."""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class PHIType(str, Enum):
    """Types of Protected Health Information."""

    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_RECORD = "medical_record"
    HEALTH_PLAN = "health_plan"
    ACCOUNT_NUMBER = "account_number"
    LICENSE_NUMBER = "license_number"
    VEHICLE_ID = "vehicle_id"
    DEVICE_ID = "device_id"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    NAME = "name"
    CREDIT_CARD = "credit_card"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"


class PHISettings(BaseSettings):
    """PHI protection configuration."""

    enable_detection: bool = Field(default=True)
    enable_masking: bool = Field(default=True)
    mask_character: str = Field(default="*")
    partial_mask_visible: int = Field(default=4)
    log_detections: bool = Field(default=True)
    strict_mode: bool = Field(default=False)
    model_config = SettingsConfigDict(
        env_prefix="PHI_", env_file=".env", extra="ignore"
    )


class PHIMatch(BaseModel):
    """Detected PHI match information."""

    phi_type: PHIType
    value: str
    masked_value: str
    start_pos: int
    end_pos: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context: str | None = None


class PHIDetectionResult(BaseModel):
    """Result of PHI detection scan."""

    contains_phi: bool
    matches: list[PHIMatch] = Field(default_factory=list)
    original_text: str
    masked_text: str
    scan_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phi_types_found: list[PHIType] = Field(default_factory=list)

    @property
    def match_count(self) -> int:
        return len(self.matches)


class PHIPattern(BaseModel):
    """Pattern definition for PHI detection."""

    phi_type: PHIType
    pattern: str
    confidence: float = Field(default=1.0)
    description: str = ""


# Minimum confidence threshold for PHI detection to reduce false positives
# Patterns below this threshold are considered low-confidence and may need additional context
MIN_CONFIDENCE_THRESHOLD = 0.80

DEFAULT_PATTERNS: list[PHIPattern] = [
    # High-confidence patterns (0.90+) - reliable detection
    PHIPattern(
        phi_type=PHIType.SSN,
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        confidence=0.95,
        description="SSN format XXX-XX-XXXX",
    ),
    PHIPattern(
        phi_type=PHIType.PHONE,
        pattern=r"\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        confidence=0.95,
        description="US phone with country code",
    ),
    PHIPattern(
        phi_type=PHIType.EMAIL,
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        confidence=0.95,
        description="Email address",
    ),
    PHIPattern(
        phi_type=PHIType.MEDICAL_RECORD,
        pattern=r"\bMRN[:\s#]*\s*\d{6,12}\b",
        confidence=0.95,
        description="Medical Record Number",
    ),
    PHIPattern(
        phi_type=PHIType.PHONE,
        pattern=r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        confidence=0.90,
        description="US phone number",
    ),
    PHIPattern(
        phi_type=PHIType.CREDIT_CARD,
        pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        confidence=0.90,
        description="Credit card number",
    ),
    # Medium-confidence patterns (0.80-0.89) - generally reliable
    PHIPattern(
        phi_type=PHIType.DATE_OF_BIRTH,
        pattern=r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b",
        confidence=0.85,
        description="Date MM/DD/YYYY",
    ),
    PHIPattern(
        phi_type=PHIType.DATE_OF_BIRTH,
        pattern=r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
        confidence=0.85,
        description="Date YYYY-MM-DD",
    ),
    PHIPattern(
        phi_type=PHIType.IP_ADDRESS,
        pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        confidence=0.85,
        description="IPv4 address",
    ),
    PHIPattern(
        phi_type=PHIType.HEALTH_PLAN,
        pattern=r"\b[A-Z]{3}\d{9}\b",
        confidence=0.80,
        description="Health plan beneficiary number",
    ),
    PHIPattern(
        phi_type=PHIType.SSN,
        pattern=r"\b\d{9}\b",
        confidence=0.80,
        description="SSN without dashes",
    ),
    PHIPattern(
        phi_type=PHIType.LICENSE_NUMBER,
        pattern=r"\b[A-Z]\d{7,8}\b",
        confidence=0.80,
        description="Driver's license",
    ),
]


class PHIMasker:
    """Mask PHI values for safe display."""

    def __init__(self, settings: PHISettings | None = None) -> None:
        self._settings = settings or PHISettings()

    def mask(self, value: str, phi_type: PHIType) -> str:
        """Mask a PHI value based on its type."""
        if not value:
            return value
        mask_char = self._settings.mask_character
        visible = self._settings.partial_mask_visible
        if phi_type == PHIType.EMAIL:
            return self._mask_email(value, mask_char)
        elif phi_type == PHIType.PHONE:
            return self._mask_phone(value, mask_char, visible)
        elif phi_type == PHIType.SSN:
            return (
                f"{mask_char * 3}-{mask_char * 2}-{value[-4:]}"
                if len(value) >= 4
                else mask_char * len(value)
            )
        elif phi_type == PHIType.CREDIT_CARD:
            return (
                f"{mask_char * 12}{value[-4:]}"
                if len(value) >= 4
                else mask_char * len(value)
            )
        elif phi_type in (PHIType.NAME, PHIType.ADDRESS):
            return self._mask_partial(value, mask_char, 2)
        else:
            return mask_char * len(value)

    def _mask_email(self, email: str, mask_char: str) -> str:
        if "@" not in email:
            return mask_char * len(email)
        local, domain = email.rsplit("@", 1)
        masked_local = local[0] + mask_char * (len(local) - 1) if local else mask_char
        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str, mask_char: str, visible: int) -> str:
        digits = re.sub(r"\D", "", phone)
        if len(digits) <= visible:
            return mask_char * len(phone)
        masked = mask_char * (len(digits) - visible) + digits[-visible:]
        return masked

    def _mask_partial(self, value: str, mask_char: str, visible_chars: int) -> str:
        if len(value) <= visible_chars:
            return mask_char * len(value)
        return value[:visible_chars] + mask_char * (len(value) - visible_chars)


class PHIDetector:
    """Detect PHI in text using pattern matching."""

    def __init__(
        self,
        settings: PHISettings | None = None,
        patterns: list[PHIPattern] | None = None,
    ) -> None:
        self._settings = settings or PHISettings()
        self._patterns = patterns or DEFAULT_PATTERNS
        self._masker = PHIMasker(self._settings)
        self._compiled = [
            (p, re.compile(p.pattern, re.IGNORECASE)) for p in self._patterns
        ]

    def detect(self, text: str) -> PHIDetectionResult:
        """Detect PHI in text and return results."""
        if not text or not self._settings.enable_detection:
            return PHIDetectionResult(
                contains_phi=False, original_text=text, masked_text=text
            )
        matches: list[PHIMatch] = []
        for pattern, regex in self._compiled:
            for match in regex.finditer(text):
                value = match.group()
                masked = self._masker.mask(value, pattern.phi_type)
                phi_match = PHIMatch(
                    phi_type=pattern.phi_type,
                    value=value,
                    masked_value=masked,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=pattern.confidence,
                )
                matches.append(phi_match)
        matches.sort(key=lambda m: (m.start_pos, -m.confidence))
        unique_matches = self._deduplicate_matches(matches)
        masked_text = (
            self._apply_masks(text, unique_matches)
            if self._settings.enable_masking
            else text
        )
        phi_types = list(set(m.phi_type for m in unique_matches))
        if unique_matches and self._settings.log_detections:
            logger.info(
                "phi_detected",
                count=len(unique_matches),
                types=[t.value for t in phi_types],
            )
        return PHIDetectionResult(
            contains_phi=len(unique_matches) > 0,
            matches=unique_matches,
            original_text=text,
            masked_text=masked_text,
            phi_types_found=phi_types,
        )

    def _deduplicate_matches(self, matches: list[PHIMatch]) -> list[PHIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        result: list[PHIMatch] = []
        for match in matches:
            overlaps = False
            for existing in result:
                if not (
                    match.end_pos <= existing.start_pos
                    or match.start_pos >= existing.end_pos
                ):
                    overlaps = True
                    break
            if not overlaps:
                result.append(match)
        return result

    def _apply_masks(self, text: str, matches: list[PHIMatch]) -> str:
        """Apply masks to detected PHI in text."""
        if not matches:
            return text
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)
        result = text
        for match in sorted_matches:
            result = (
                result[: match.start_pos] + match.masked_value + result[match.end_pos :]
            )
        return result

    def contains_phi(self, text: str) -> bool:
        """Quick check if text contains any PHI."""
        return self.detect(text).contains_phi


class PHISanitizer:
    """Sanitize data structures by detecting and masking PHI."""

    def __init__(self, detector: PHIDetector | None = None) -> None:
        self._detector = detector or PHIDetector()

    def sanitize_string(self, text: str) -> str:
        """Sanitize a string by masking detected PHI."""
        return self._detector.detect(text).masked_text

    def sanitize_dict(
        self, data: dict[str, Any], sensitive_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Sanitize dictionary values."""
        result = {}
        sensitive = {k.lower() for k in sensitive_keys} if sensitive_keys else set()
        for key, value in data.items():
            if isinstance(value, str):
                is_sensitive = key.lower() in sensitive or self._is_sensitive_key(key)
                if is_sensitive:
                    detection = self._detector.detect(value)
                    if detection.contains_phi:
                        result[key] = detection.masked_text
                    else:
                        result[key] = self._mask_sensitive_value(value)
                else:
                    detection = self._detector.detect(value)
                    result[key] = (
                        detection.masked_text if detection.contains_phi else value
                    )
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value, sensitive_keys)
            elif isinstance(value, list):
                result[key] = self.sanitize_list(value, sensitive_keys)
            else:
                result[key] = value
        return result

    def _mask_sensitive_value(self, value: str) -> str:
        """Mask a sensitive value that wasn't detected as PHI."""
        if len(value) <= 2:
            return "*" * len(value)
        return value[0] + "*" * (len(value) - 2) + value[-1]

    def sanitize_list(
        self, data: list[Any], sensitive_keys: list[str] | None = None
    ) -> list[Any]:
        """Sanitize list values."""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self._detector.detect(item).masked_text)
            elif isinstance(item, dict):
                result.append(self.sanitize_dict(item, sensitive_keys))
            elif isinstance(item, list):
                result.append(self.sanitize_list(item, sensitive_keys))
            else:
                result.append(item)
        return result

    def _is_sensitive_key(self, key: str) -> bool:
        sensitive_patterns = [
            "ssn",
            "social",
            "phone",
            "email",
            "address",
            "dob",
            "birth",
            "medical",
            "health",
            "card",
            "account",
        ]
        key_lower = key.lower()
        return any(p in key_lower for p in sensitive_patterns)


def create_phi_detector(settings: PHISettings | None = None) -> PHIDetector:
    """Factory function to create PHI detector."""
    return PHIDetector(settings)


def create_phi_sanitizer(detector: PHIDetector | None = None) -> PHISanitizer:
    """Factory function to create PHI sanitizer."""
    return PHISanitizer(detector)


def detect_phi(text: str) -> PHIDetectionResult:
    """Convenience function to detect PHI in text."""
    return PHIDetector().detect(text)


def mask_phi(text: str) -> str:
    """Convenience function to mask PHI in text."""
    return PHIDetector().detect(text).masked_text


_structlog_sanitizer: PHISanitizer | None = None


def phi_sanitizer_processor(
    logger_: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that sanitizes PHI from log event dicts.

    Add to the structlog processor chain before the renderer:
        processors=[
            ...,
            phi_sanitizer_processor,
            structlog.processors.JSONRenderer(),
        ]
    """
    global _structlog_sanitizer
    if _structlog_sanitizer is None:
        _structlog_sanitizer = PHISanitizer()
    return _structlog_sanitizer.sanitize_dict(event_dict)

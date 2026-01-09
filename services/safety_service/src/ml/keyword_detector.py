"""
Solace-AI Keyword Detector - Fast multi-pattern crisis keyword detection.
Uses FlashText library (Aho-Corasick algorithm) for efficient O(n) detection of crisis keywords in text.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4
import json
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

try:
    from flashtext import KeywordProcessor
    FLASHTEXT_AVAILABLE = True
except ImportError:
    FLASHTEXT_AVAILABLE = False

try:
    from safety_service.src.infrastructure.telemetry import traced, get_telemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

    def traced(*args, **kwargs):
        """No-op decorator when telemetry unavailable."""
        def decorator(func):
            return func
        return decorator

logger = structlog.get_logger(__name__)


class KeywordSeverity(str, Enum):
    """Severity level for detected keywords."""
    CRITICAL = "CRITICAL"  # Immediate crisis indicators
    HIGH = "HIGH"  # High-risk indicators
    ELEVATED = "ELEVATED"  # Moderate concern
    LOW = "LOW"  # Mild distress signals
    INFO = "INFO"  # Informational mentions


class KeywordCategory(str, Enum):
    """Category classification for keywords."""
    SUICIDAL_IDEATION = "SUICIDAL_IDEATION"
    SELF_HARM = "SELF_HARM"
    HOPELESSNESS = "HOPELESSNESS"
    PLAN_INTENT = "PLAN_INTENT"
    MEANS_ACCESS = "MEANS_ACCESS"
    FAREWELL = "FAREWELL"
    EMOTIONAL_DISTRESS = "EMOTIONAL_DISTRESS"
    SUBSTANCE_ABUSE = "SUBSTANCE_ABUSE"
    SAFETY_CONCERN = "SAFETY_CONCERN"


class KeywordMatch(BaseModel):
    """Represents a detected keyword match in text."""
    keyword: str = Field(..., description="The matched keyword")
    position: int = Field(..., ge=0, description="Character position in text")
    severity: KeywordSeverity = Field(..., description="Severity level")
    category: KeywordCategory = Field(..., description="Keyword category")
    confidence: Decimal = Field(..., ge=0, le=1, description="Detection confidence")
    context_snippet: str = Field(..., description="Surrounding text context")
    weight: Decimal = Field(..., ge=0, le=1, description="Contribution to overall score")


class KeywordDetectorConfig(BaseSettings):
    """Configuration for keyword detector."""
    enable_case_insensitive: bool = Field(default=True, description="Enable case-insensitive matching")
    context_window_chars: int = Field(default=50, ge=0, description="Characters to capture around match")
    critical_weight: Decimal = Field(default=Decimal("0.95"), description="Weight for critical keywords")
    high_weight: Decimal = Field(default=Decimal("0.75"), description="Weight for high-risk keywords")
    elevated_weight: Decimal = Field(default=Decimal("0.5"), description="Weight for elevated keywords")
    low_weight: Decimal = Field(default=Decimal("0.25"), description="Weight for low keywords")
    info_weight: Decimal = Field(default=Decimal("0.1"), description="Weight for info keywords")
    match_whole_words: bool = Field(default=True, description="Only match whole words")
    enable_variant_detection: bool = Field(default=True, description="Detect keyword variants")
    max_matches_per_text: int = Field(default=100, ge=1, description="Maximum matches to return")

    model_config = SettingsConfigDict(env_prefix="KEYWORD_DETECTOR_", env_file=".env", extra="ignore")


class KeywordDetector:
    """
    Fast keyword-based crisis detection using FlashText library.
    Implements O(n) multi-pattern matching for real-time crisis detection.
    """

    def __init__(self, config: KeywordDetectorConfig | None = None) -> None:
        """Initialize keyword detector with configuration."""
        self._config = config or KeywordDetectorConfig()
        self._keyword_database = self._load_keyword_database()

        # Initialize FlashText processor
        if FLASHTEXT_AVAILABLE:
            self._processor = KeywordProcessor(
                case_sensitive=not self._config.enable_case_insensitive
            )
            # Add keywords with metadata to FlashText
            for keyword, (severity, category) in self._keyword_database.items():
                # FlashText stores clean_name as key, returns it on match
                # We store metadata separately
                self._processor.add_keyword(keyword, keyword)
            self._engine = "flashtext"
        else:
            # Fallback to dict-based search (slower but works)
            self._processor = None
            self._engine = "dict_based"
            logger.warning("flashtext_unavailable", fallback="dict_based_search")

        logger.info("keyword_detector_initialized",
                   keywords_loaded=len(self._keyword_database),
                   case_insensitive=self._config.enable_case_insensitive,
                   engine=self._engine)

    def _load_keyword_database(self) -> dict[str, tuple[KeywordSeverity, KeywordCategory]]:
        """
        Load crisis keyword database from JSON configuration.
        JSON config is the single source of truth for scalability.
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "keywords.json"

        if not config_path.exists():
            logger.error("keywords_config_missing",
                        path=str(config_path),
                        action="create config/keywords.json")
            raise FileNotFoundError(
                f"Keywords configuration not found: {config_path}. "
                "Please create the JSON config file for scalable deployment."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            keyword_db: dict[str, tuple[KeywordSeverity, KeywordCategory]] = {}

            for severity_name, categories in data.get("keywords", {}).items():
                try:
                    severity = KeywordSeverity[severity_name]
                except KeyError:
                    logger.warning("invalid_severity_in_json", severity=severity_name)
                    continue

                for category_name, keywords in categories.items():
                    try:
                        category = KeywordCategory[category_name]
                    except KeyError:
                        logger.warning("invalid_category_in_json", category=category_name)
                        continue

                    for keyword in keywords:
                        keyword_db[keyword] = (severity, category)

            if not keyword_db:
                raise ValueError("Keyword database is empty - check JSON structure")

            logger.info("keywords_loaded_from_json",
                       path=str(config_path),
                       count=len(keyword_db),
                       version=data.get("version"))
            return keyword_db

        except json.JSONDecodeError as e:
            logger.error("keywords_invalid_json", path=str(config_path), error=str(e))
            raise ValueError(f"Invalid JSON in keywords config: {e}")

    @traced(name="keyword_detector.detect", attributes={"component": "keyword_detector"})
    def detect(self, text: str, user_id: UUID | None = None) -> list[KeywordMatch]:
        """
        Detect crisis keywords in text using FlashText (Aho-Corasick algorithm).

        Args:
            text: Input text to analyze
            user_id: Optional user ID for logging

        Returns:
            List of keyword matches sorted by severity
        """
        if not text:
            return []

        matches: list[KeywordMatch] = []

        if self._processor and FLASHTEXT_AVAILABLE:
            # Use FlashText for efficient O(n) matching
            # extract_keywords returns: [(keyword, start_pos, end_pos), ...]
            found_keywords = self._processor.extract_keywords(text, span_info=True)

            for keyword, start_pos, end_pos in found_keywords:
                # Get metadata from our database
                if keyword in self._keyword_database:
                    severity, category = self._keyword_database[keyword]
                else:
                    # Shouldn't happen, but handle gracefully
                    continue

                # Extract context
                context_start = max(0, start_pos - self._config.context_window_chars)
                context_end = min(len(text), end_pos + self._config.context_window_chars)
                context = text[context_start:context_end].strip()

                # Get severity weight
                weight = self._get_severity_weight(severity)

                match = KeywordMatch(
                    keyword=keyword,
                    position=start_pos,
                    severity=severity,
                    category=category,
                    confidence=Decimal("0.9"),  # High confidence for exact matches
                    context_snippet=context,
                    weight=weight
                )
                matches.append(match)

                # Stop after max matches
                if len(matches) >= self._config.max_matches_per_text:
                    break
        else:
            # Fallback to simple substring search if FlashText unavailable
            search_text = text.lower() if self._config.enable_case_insensitive else text

            for keyword, (severity, category) in self._keyword_database.items():
                search_keyword = keyword.lower() if self._config.enable_case_insensitive else keyword

                # Simple substring search
                pos = 0
                while True:
                    pos = search_text.find(search_keyword, pos)
                    if pos == -1:
                        break

                    # Check word boundaries if required
                    if self._config.match_whole_words:
                        # Check start boundary
                        if pos > 0 and search_text[pos - 1].isalnum():
                            pos += 1
                            continue
                        # Check end boundary
                        end_pos = pos + len(search_keyword)
                        if end_pos < len(search_text) and search_text[end_pos].isalnum():
                            pos += 1
                            continue

                    # Extract context
                    end_pos = pos + len(search_keyword)
                    context_start = max(0, pos - self._config.context_window_chars)
                    context_end = min(len(text), end_pos + self._config.context_window_chars)
                    context = text[context_start:context_end].strip()

                    weight = self._get_severity_weight(severity)

                    match = KeywordMatch(
                        keyword=keyword,
                        position=pos,
                        severity=severity,
                        category=category,
                        confidence=Decimal("0.9"),
                        context_snippet=context,
                        weight=weight
                    )
                    matches.append(match)

                    if len(matches) >= self._config.max_matches_per_text:
                        break

                    pos += 1

                if len(matches) >= self._config.max_matches_per_text:
                    break

        # Sort by severity (critical first) then position
        severity_order = {
            KeywordSeverity.CRITICAL: 0,
            KeywordSeverity.HIGH: 1,
            KeywordSeverity.ELEVATED: 2,
            KeywordSeverity.LOW: 3,
            KeywordSeverity.INFO: 4,
        }
        matches.sort(key=lambda m: (severity_order.get(m.severity, 999), m.position))

        if matches and user_id:
            logger.info("keywords_detected", user_id=str(user_id), count=len(matches),
                       critical=sum(1 for m in matches if m.severity == KeywordSeverity.CRITICAL))

        return matches

    def _get_severity_weight(self, severity: KeywordSeverity) -> Decimal:
        """Get weight for severity level."""
        weights = {
            KeywordSeverity.CRITICAL: self._config.critical_weight,
            KeywordSeverity.HIGH: self._config.high_weight,
            KeywordSeverity.ELEVATED: self._config.elevated_weight,
            KeywordSeverity.LOW: self._config.low_weight,
            KeywordSeverity.INFO: self._config.info_weight,
        }
        return weights.get(severity, Decimal("0.5"))

    def calculate_risk_score(self, matches: list[KeywordMatch]) -> Decimal:
        """
        Calculate overall risk score from keyword matches.

        Uses maximum severity with diminishing returns for multiple matches.
        """
        if not matches:
            return Decimal("0.0")

        # Get maximum weight
        max_weight = max(m.weight for m in matches)

        # Add diminishing contribution from additional matches
        total_weight = max_weight
        for i, match in enumerate(matches[1:], 1):
            # Each additional match contributes less
            contribution = match.weight * Decimal(str(0.8 ** i))
            total_weight += contribution

        # Cap at 1.0
        return min(total_weight, Decimal("1.0"))

    def get_highest_severity(self, matches: list[KeywordMatch]) -> KeywordSeverity | None:
        """Get the highest severity level from matches."""
        if not matches:
            return None

        severity_order = [
            KeywordSeverity.CRITICAL,
            KeywordSeverity.HIGH,
            KeywordSeverity.ELEVATED,
            KeywordSeverity.LOW,
            KeywordSeverity.INFO,
        ]

        for severity in severity_order:
            if any(m.severity == severity for m in matches):
                return severity

        return None

    def get_categories(self, matches: list[KeywordMatch]) -> set[KeywordCategory]:
        """Extract unique categories from matches."""
        return {match.category for match in matches}

"""
Solace-AI Keyword Detector - Fast multi-pattern crisis keyword detection.
Uses trie-based algorithm for efficient O(n) detection of crisis keywords in text.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

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


@dataclass
class TrieNode:
    """Node in the keyword trie for efficient multi-pattern matching."""
    children: dict[str, TrieNode] = field(default_factory=dict)
    is_end: bool = False
    keyword: str | None = None
    severity: KeywordSeverity | None = None
    category: KeywordCategory | None = None


class KeywordDetector:
    """
    Fast keyword-based crisis detection using trie data structure.
    Implements O(n) multi-pattern matching for real-time crisis detection.
    """

    def __init__(self, config: KeywordDetectorConfig | None = None) -> None:
        """Initialize keyword detector with configuration."""
        self._config = config or KeywordDetectorConfig()
        self._trie_root = TrieNode()
        self._keyword_database = self._load_keyword_database()
        self._build_trie()
        logger.info("keyword_detector_initialized",
                   keywords_loaded=len(self._keyword_database),
                   case_insensitive=self._config.enable_case_insensitive)

    def _load_keyword_database(self) -> dict[str, tuple[KeywordSeverity, KeywordCategory]]:
        """Load comprehensive crisis keyword database with severity and category."""
        return {
            # CRITICAL - Immediate suicidal ideation
            "kill myself": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "end my life": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "suicide": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "want to die": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "better off dead": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "no reason to live": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "take my life": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "end it all": (KeywordSeverity.CRITICAL, KeywordCategory.SUICIDAL_IDEATION),
            "hanging myself": (KeywordSeverity.CRITICAL, KeywordCategory.PLAN_INTENT),
            "overdose": (KeywordSeverity.CRITICAL, KeywordCategory.PLAN_INTENT),

            # HIGH - Self-harm and high-risk behaviors
            "self-harm": (KeywordSeverity.HIGH, KeywordCategory.SELF_HARM),
            "hurt myself": (KeywordSeverity.HIGH, KeywordCategory.SELF_HARM),
            "cutting": (KeywordSeverity.HIGH, KeywordCategory.SELF_HARM),
            "cut myself": (KeywordSeverity.HIGH, KeywordCategory.SELF_HARM),
            "burning myself": (KeywordSeverity.HIGH, KeywordCategory.SELF_HARM),
            "don't want to live": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "no hope": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "hopeless": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "give up": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "burden": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "worthless": (KeywordSeverity.HIGH, KeywordCategory.HOPELESSNESS),
            "goodbye forever": (KeywordSeverity.HIGH, KeywordCategory.FAREWELL),

            # ELEVATED - Moderate distress indicators
            "depressed": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "anxious": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "overwhelmed": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "can't cope": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "falling apart": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "breaking down": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "losing control": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "panic": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "scared": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),
            "terrified": (KeywordSeverity.ELEVATED, KeywordCategory.EMOTIONAL_DISTRESS),

            # LOW - Mild distress
            "sad": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "stressed": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "worried": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "upset": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "frustrated": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "tired": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),
            "exhausted": (KeywordSeverity.LOW, KeywordCategory.EMOTIONAL_DISTRESS),

            # PLAN/MEANS indicators
            "have a plan": (KeywordSeverity.CRITICAL, KeywordCategory.PLAN_INTENT),
            "pills": (KeywordSeverity.HIGH, KeywordCategory.MEANS_ACCESS),
            "rope": (KeywordSeverity.HIGH, KeywordCategory.MEANS_ACCESS),
            "gun": (KeywordSeverity.HIGH, KeywordCategory.MEANS_ACCESS),
            "knife": (KeywordSeverity.HIGH, KeywordCategory.MEANS_ACCESS),
            "bridge": (KeywordSeverity.HIGH, KeywordCategory.MEANS_ACCESS),
        }

    def _build_trie(self) -> None:
        """Build trie from keyword database for efficient matching."""
        for keyword, (severity, category) in self._keyword_database.items():
            key = keyword.lower() if self._config.enable_case_insensitive else keyword
            self._insert_keyword(key, keyword, severity, category)

    def _insert_keyword(self, key: str, original: str, severity: KeywordSeverity,
                       category: KeywordCategory) -> None:
        """Insert keyword into trie."""
        node = self._trie_root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.keyword = original
        node.severity = severity
        node.category = category

    def detect(self, text: str, user_id: UUID | None = None) -> list[KeywordMatch]:
        """
        Detect crisis keywords in text using trie-based matching.

        Args:
            text: Input text to analyze
            user_id: Optional user ID for logging

        Returns:
            List of keyword matches sorted by severity
        """
        if not text:
            return []

        search_text = text.lower() if self._config.enable_case_insensitive else text
        matches: list[KeywordMatch] = []

        # Scan text for keyword matches
        for i in range(len(search_text)):
            # Check for word boundary if required
            if self._config.match_whole_words and i > 0 and search_text[i-1].isalnum():
                continue

            # Try to match from this position
            node = self._trie_root
            j = i

            while j < len(search_text) and search_text[j] in node.children:
                node = node.children[search_text[j]]
                j += 1

                # Check if we found a complete keyword
                if node.is_end:
                    # Verify word boundary at end
                    if self._config.match_whole_words and j < len(search_text) and search_text[j].isalnum():
                        continue

                    # Extract context
                    context_start = max(0, i - self._config.context_window_chars)
                    context_end = min(len(text), j + self._config.context_window_chars)
                    context = text[context_start:context_end].strip()

                    # Get severity weight
                    weight = self._get_severity_weight(node.severity)

                    match = KeywordMatch(
                        keyword=node.keyword,
                        position=i,
                        severity=node.severity,
                        category=node.category,
                        confidence=Decimal("0.9"),  # High confidence for exact matches
                        context_snippet=context,
                        weight=weight
                    )
                    matches.append(match)

                    # Stop after max matches
                    if len(matches) >= self._config.max_matches_per_text:
                        break

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

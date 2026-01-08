"""
Solace-AI Pattern Matcher - Advanced crisis detection using spaCy NLP + regex patterns.
Uses spaCy for linguistic understanding and regex for comprehensive crisis phrase coverage.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID
import re
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except Exception:  # Catch all exceptions including dependency issues
    SPACY_AVAILABLE = False

logger = structlog.get_logger(__name__)


class PatternType(str, Enum):
    """Type of crisis pattern detected."""
    SUICIDAL_IDEATION = "SUICIDAL_IDEATION"  # Suicide-related thoughts/plans
    SELF_HARM_INTENT = "SELF_HARM_INTENT"  # Self-harm behaviors
    HOPELESSNESS_EXPRESSION = "HOPELESSNESS_EXPRESSION"  # Expressions of hopelessness
    PLAN_INDICATOR = "PLAN_INDICATOR"  # Specific plan mentions
    MEANS_ACCESS = "MEANS_ACCESS"  # Access to lethal means
    FAREWELL_MESSAGE = "FAREWELL_MESSAGE"  # Goodbye/farewell messages
    TIMEFRAME_URGENCY = "TIMEFRAME_URGENCY"  # Time-bound crisis indicators
    SOCIAL_WITHDRAWAL = "SOCIAL_WITHDRAWAL"  # Isolation patterns
    SUBSTANCE_ABUSE = "SUBSTANCE_ABUSE"  # Substance use patterns
    TRAUMA_FLASHBACK = "TRAUMA_FLASHBACK"  # PTSD/trauma indicators


class PatternMatch(BaseModel):
    """Represents a detected pattern match."""
    pattern_type: PatternType = Field(..., description="Type of pattern detected")
    matched_text: str = Field(..., description="Text that matched the pattern")
    position: int = Field(..., ge=0, description="Start position in text")
    severity: Decimal = Field(..., ge=0, le=1, description="Severity score")
    confidence: Decimal = Field(..., ge=0, le=1, description="Match confidence")
    context: str = Field(..., description="Surrounding context")
    explanation: str = Field(..., description="Why this pattern is concerning")


class PatternMatcherConfig(BaseSettings):
    """Configuration for pattern matcher."""
    # spaCy NLP settings (for better linguistic understanding)
    use_spacy: bool = Field(default=True, description="Use spaCy for linguistic pattern matching (better accuracy)")
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model to use")

    # Pattern matching settings
    enable_case_insensitive: bool = Field(default=True, description="Case-insensitive matching")
    context_window: int = Field(default=60, ge=0, description="Context characters around match")
    min_confidence: Decimal = Field(default=Decimal("0.7"), description="Minimum confidence threshold")
    enable_temporal_detection: bool = Field(default=True, description="Detect time-bound urgency")
    enable_compound_patterns: bool = Field(default=True, description="Detect multi-clause patterns")
    max_matches_per_text: int = Field(default=50, ge=1, description="Maximum matches to return")

    model_config = SettingsConfigDict(env_prefix="PATTERN_MATCHER_", env_file=".env", extra="ignore")


@dataclass
class CompiledPattern:
    """Compiled regex pattern with metadata."""
    pattern: re.Pattern
    pattern_type: PatternType
    severity: Decimal
    confidence: Decimal
    explanation: str


class PatternMatcher:
    """
    Advanced crisis detection using hybrid spaCy NLP + regex patterns.
    spaCy provides linguistic analysis; regex ensures comprehensive crisis phrase coverage.
    """

    def __init__(self, config: PatternMatcherConfig | None = None) -> None:
        """Initialize pattern matcher with configuration."""
        self._config = config or PatternMatcherConfig()
        self._patterns = self._compile_patterns()

        # Initialize spaCy NLP and Matcher if available (better linguistic understanding)
        self._nlp = None
        self._spacy_matcher = None
        if self._config.use_spacy and SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(self._config.spacy_model)
                self._spacy_matcher = Matcher(self._nlp.vocab)
                self._add_spacy_patterns()
                logger.info("pattern_matcher_initialized",
                           mode="hybrid_spacy",
                           spacy_model=self._config.spacy_model,
                           regex_patterns=len(self._patterns))
            except Exception as e:
                logger.warning("spacy_load_failed", error=str(e), fallback="regex_only")
                self._nlp = None
                self._spacy_matcher = None

        if self._nlp is None:
            logger.info("pattern_matcher_initialized",
                       mode="regex_only",
                       pattern_count=len(self._patterns))

    def _compile_patterns(self) -> list[CompiledPattern]:
        """Compile comprehensive crisis detection patterns using regex."""
        patterns: list[CompiledPattern] = []
        flags = re.IGNORECASE if self._config.enable_case_insensitive else 0

        # Critical explicit crisis phrases
        crisis_patterns = [
            # Suicidal ideation (explicit and with want/die patterns)
            (r"\b(kill\s+myself|commit\s+suicide|end\s+my\s+life)\b",
             PatternType.SUICIDAL_IDEATION, Decimal("0.95"), Decimal("0.9"), "Explicit suicide expression"),
            (r"\b(want|wanted|wanting)\s+to\s+die\b",
             PatternType.SUICIDAL_IDEATION, Decimal("0.95"), Decimal("0.9"), "Want to die expression"),
            (r"\bwish\s+(i\s+)?(was|were)\s+dead\b",
             PatternType.SUICIDAL_IDEATION, Decimal("0.9"), Decimal("0.85"), "Death wish"),
            (r"\bbetter\s+off\s+without\s+me\b",
             PatternType.SUICIDAL_IDEATION, Decimal("0.85"), Decimal("0.85"), "Perceived burdensomeness"),

            # Plan indicators
            (r"\b(have|got)\s+.{0,10}(plan|method)\s+to\s+(die|end)\b",
             PatternType.PLAN_INDICATOR, Decimal("0.95"), Decimal("0.95"), "Articulated plan"),
            (r"\bsuicide\s+note\b",
             PatternType.PLAN_INDICATOR, Decimal("0.95"), Decimal("0.95"), "Suicide note mention"),

            # Temporal urgency
            (r"\b(tonight|today).{0,30}(die|end\s+it|over)\b",
             PatternType.TIMEFRAME_URGENCY, Decimal("0.95"), Decimal("0.9"), "Immediate timeframe"),

            # Hopelessness expressions
            (r"\bcan't\s+take\s+(it|this)\s+anymore\b",
             PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.8"), Decimal("0.85"), "Overwhelm expression"),
            (r"\b(never|won't)\s+.{0,20}(get\s+)?better\b",
             PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.75"), Decimal("0.8"), "Things won't improve"),
            (r"\bfeel\s+hopeless\b",
             PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.8"), Decimal("0.85"), "Feeling hopeless"),
        ]

        for regex, ptype, severity, confidence, explanation in crisis_patterns:
            patterns.append(CompiledPattern(
                pattern=re.compile(regex, flags),
                pattern_type=ptype,
                severity=severity,
                confidence=confidence,
                explanation=explanation
            ))

        # Additional critical phrases (farewell, self-harm, means access)
        additional_patterns = [
            (r"\b(goodbye|farewell)\s+(everyone|world)\b",
             PatternType.FAREWELL_MESSAGE, Decimal("0.9"), Decimal("0.85"), "Farewell message"),
            (r"\b(cutting|cut|hurt)\s+myself\b",
             PatternType.SELF_HARM_INTENT, Decimal("0.8"), Decimal("0.85"), "Self-harm behavior"),
            (r"\b(gun|pills|rope)\b",
             PatternType.MEANS_ACCESS, Decimal("0.75"), Decimal("0.8"), "Lethal means mention"),
        ]

        for regex, ptype, severity, confidence, explanation in additional_patterns:
            patterns.append(CompiledPattern(
                pattern=re.compile(regex, flags),
                pattern_type=ptype,
                severity=severity,
                confidence=confidence,
                explanation=explanation
            ))

        return patterns

    def _add_spacy_patterns(self) -> None:
        """Add spaCy Matcher patterns for linguistic analysis."""
        if not self._spacy_matcher:
            return

        # Suicidal ideation patterns (verb + death/suicide)
        self._spacy_matcher.add("SUICIDAL_VERB_DEATH", [[
            {"LEMMA": {"IN": ["want", "plan", "think", "go"]}},
            {"POS": "PART", "OP": "?"},  # Optional "to"
            {"LEMMA": {"IN": ["die", "kill", "end"]}}
        ]])

        # Hopelessness patterns (never + better/improve)
        self._spacy_matcher.add("HOPELESSNESS_NEVER", [[
            {"LOWER": {"IN": ["never", "won't", "will not"]}},
            {"LEMMA": {"IN": ["get", "be", "feel"]}, "OP": "?"},
            {"LOWER": {"IN": ["better", "ok", "okay", "good"]}}
        ]])

        # Plan indicators (have + plan/method)
        self._spacy_matcher.add("PLAN_HAVE", [[
            {"LEMMA": "have"},
            {"POS": "DET", "OP": "?"},
            {"LEMMA": {"IN": ["plan", "method", "way"]}}
        ]])

        # Temporal urgency (time + action)
        self._spacy_matcher.add("TEMPORAL_ACTION", [[
            {"LOWER": {"IN": ["tonight", "today", "tomorrow", "soon"]}},
            {"LEMMA": {"IN": ["kill", "die", "end", "hurt"]}}
        ]])

    def _detect_spacy_patterns(self, text: str) -> list[PatternMatch]:
        """Detect patterns using spaCy Matcher for better linguistic understanding."""
        if not self._nlp or not self._spacy_matcher:
            return []

        matches: list[PatternMatch] = []
        doc = self._nlp(text)
        spacy_matches = self._spacy_matcher(doc)

        for match_id, start, end in spacy_matches:
            span = doc[start:end]
            rule_name = self._nlp.vocab.strings[match_id]

            # Map spaCy patterns to pattern types and severities
            if "SUICIDAL" in rule_name:
                pattern_type = PatternType.SUICIDAL_IDEATION
                severity, confidence = Decimal("0.95"), Decimal("0.9")
                explanation = "Suicidal ideation detected via linguistic analysis"
            elif "HOPELESSNESS" in rule_name:
                pattern_type = PatternType.HOPELESSNESS_EXPRESSION
                severity, confidence = Decimal("0.85"), Decimal("0.85")
                explanation = "Hopelessness expression detected"
            elif "PLAN" in rule_name:
                pattern_type = PatternType.PLAN_INDICATOR
                severity, confidence = Decimal("0.9"), Decimal("0.9")
                explanation = "Plan indicator detected"
            elif "TEMPORAL" in rule_name:
                pattern_type = PatternType.TIMEFRAME_URGENCY
                severity, confidence = Decimal("0.9"), Decimal("0.85")
                explanation = "Temporal urgency detected"
            else:
                continue

            # Extract context from sentence
            context_start = max(0, span.sent.start_char - self._config.context_window)
            context_end = min(len(text), span.sent.end_char + self._config.context_window)
            context = text[context_start:context_end].strip()

            matches.append(PatternMatch(
                pattern_type=pattern_type,
                matched_text=span.text,
                position=span.start_char,
                severity=severity,
                confidence=confidence,
                context=context,
                explanation=explanation
            ))

        return matches

    def detect(self, text: str, user_id: UUID | None = None) -> list[PatternMatch]:
        """
        Detect crisis patterns using hybrid spaCy + regex approach.

        Args:
            text: Input text to analyze
            user_id: Optional user ID for logging

        Returns:
            List of pattern matches sorted by severity
        """
        if not text:
            return []

        matches: list[PatternMatch] = []

        # First: Use spaCy patterns if available (better linguistic understanding)
        if self._nlp:
            spacy_matches = self._detect_spacy_patterns(text)
            matches.extend(spacy_matches)

        # Then: Scan with regex patterns (comprehensive crisis phrase coverage)
        for compiled_pattern in self._patterns:
            for match in compiled_pattern.pattern.finditer(text):
                # Extract context
                start_pos = match.start()
                context_start = max(0, start_pos - self._config.context_window)
                context_end = min(len(text), match.end() + self._config.context_window)
                context = text[context_start:context_end].strip()

                pattern_match = PatternMatch(
                    pattern_type=compiled_pattern.pattern_type,
                    matched_text=match.group(),
                    position=start_pos,
                    severity=compiled_pattern.severity,
                    confidence=compiled_pattern.confidence,
                    context=context,
                    explanation=compiled_pattern.explanation
                )
                matches.append(pattern_match)

                # Stop if reached max
                if len(matches) >= self._config.max_matches_per_text:
                    break

            if len(matches) >= self._config.max_matches_per_text:
                break

        # Sort by severity (highest first) then position
        matches.sort(key=lambda m: (-float(m.severity), m.position))

        if matches and user_id:
            logger.info("patterns_detected", user_id=str(user_id), count=len(matches),
                       critical_count=sum(1 for m in matches if m.severity >= Decimal("0.9")))

        return matches

    def calculate_risk_score(self, matches: list[PatternMatch]) -> Decimal:
        """
        Calculate overall risk score from pattern matches.

        Uses maximum severity with additive boosts for multiple distinct patterns.
        """
        if not matches:
            return Decimal("0.0")

        # Get maximum severity
        max_severity = max(m.severity for m in matches)

        # Count unique pattern types
        unique_types = {m.pattern_type for m in matches}

        # Boost for multiple distinct patterns (indicates complexity)
        diversity_boost = min(Decimal(str(len(unique_types) - 1)) * Decimal("0.05"), Decimal("0.2"))

        total_risk = max_severity + diversity_boost
        return min(total_risk, Decimal("1.0"))

    def get_dominant_pattern_type(self, matches: list[PatternMatch]) -> PatternType | None:
        """Get the most severe pattern type from matches."""
        if not matches:
            return None
        return max(matches, key=lambda m: m.severity).pattern_type

    def has_critical_patterns(self, matches: list[PatternMatch]) -> bool:
        """Check if any critical patterns (severity >= 0.9) were detected."""
        return any(m.severity >= Decimal("0.9") for m in matches)

    def group_by_type(self, matches: list[PatternMatch]) -> dict[PatternType, list[PatternMatch]]:
        """Group matches by pattern type."""
        grouped: dict[PatternType, list[PatternMatch]] = {}
        for match in matches:
            if match.pattern_type not in grouped:
                grouped[match.pattern_type] = []
            grouped[match.pattern_type].append(match)
        return grouped

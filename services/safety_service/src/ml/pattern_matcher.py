"""
Solace-AI Pattern Matcher - Advanced crisis detection using spaCy NLP + regex patterns.
Uses spaCy for linguistic understanding and regex for comprehensive crisis phrase coverage.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID
import json
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

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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


# Emotion-to-crisis mapping for transformer model (SamLowe/roberta-base-go_emotions)
# Maps GoEmotions labels to crisis indicators with severity weights
CRISIS_EMOTION_MAP: dict[str, tuple[PatternType, Decimal]] = {
    "grief": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.85")),
    "sadness": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.7")),
    "fear": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.65")),
    "nervousness": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.55")),
    "disappointment": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.5")),
    "remorse": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.6")),
    "disgust": (PatternType.SOCIAL_WITHDRAWAL, Decimal("0.5")),
    "anger": (PatternType.HOPELESSNESS_EXPRESSION, Decimal("0.55")),
    "disapproval": (PatternType.SOCIAL_WITHDRAWAL, Decimal("0.45")),
}


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

    # Transformer emotion classifier settings (GoEmotions for crisis-relevant emotion detection)
    use_emotion_classifier: bool = Field(default=True, description="Use transformer-based emotion classifier")
    emotion_model: str = Field(
        default="SamLowe/roberta-base-go_emotions",
        description="HuggingFace model for emotion classification (28 emotions)"
    )
    emotion_threshold: Decimal = Field(default=Decimal("0.4"), description="Minimum emotion probability threshold")

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
    Advanced crisis detection using hybrid spaCy NLP + regex patterns + transformer emotion classification.
    - spaCy provides linguistic analysis
    - Regex ensures comprehensive crisis phrase coverage
    - Transformer model detects crisis-relevant emotions (grief, sadness, fear, etc.)
    """

    def __init__(self, config: PatternMatcherConfig | None = None) -> None:
        """Initialize pattern matcher with configuration."""
        self._config = config or PatternMatcherConfig()
        self._patterns = self._compile_patterns()

        # Initialize spaCy NLP and Matcher if available (better linguistic understanding)
        self._nlp = None
        self._spacy_matcher = None
        self._emotion_classifier = None
        self._mode = "regex_only"

        if self._config.use_spacy and SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load(self._config.spacy_model)
                self._spacy_matcher = Matcher(self._nlp.vocab)
                self._add_spacy_patterns()
                self._mode = "hybrid_spacy"
            except Exception as e:
                logger.warning("spacy_load_failed", error=str(e), fallback="regex_only")
                self._nlp = None
                self._spacy_matcher = None

        # Initialize transformer-based emotion classifier (GoEmotions)
        if self._config.use_emotion_classifier and TRANSFORMERS_AVAILABLE:
            try:
                self._emotion_classifier = pipeline(
                    task="text-classification",
                    model=self._config.emotion_model,
                    top_k=None,  # Return all 28 emotion scores
                    truncation=True
                )
                self._mode = f"{self._mode}_emotion" if self._mode != "regex_only" else "hybrid_emotion"
                logger.info("emotion_classifier_initialized",
                           model=self._config.emotion_model,
                           threshold=float(self._config.emotion_threshold))
            except Exception as e:
                logger.warning("emotion_classifier_load_failed",
                             error=str(e),
                             model=self._config.emotion_model)
                self._emotion_classifier = None

        logger.info("pattern_matcher_initialized",
                   mode=self._mode,
                   spacy_available=self._nlp is not None,
                   emotion_classifier_available=self._emotion_classifier is not None,
                   regex_patterns=len(self._patterns))

    def _compile_patterns(self) -> list[CompiledPattern]:
        """
        Compile crisis detection patterns from JSON configuration.
        JSON config is the single source of truth for scalability.
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "patterns.json"
        flags = re.IGNORECASE if self._config.enable_case_insensitive else 0

        if not config_path.exists():
            logger.error("patterns_config_missing",
                        path=str(config_path),
                        action="create config/patterns.json")
            raise FileNotFoundError(
                f"Patterns configuration not found: {config_path}. "
                "Please create the JSON config file for scalable deployment."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            patterns: list[CompiledPattern] = []

            for severity_level, pattern_types in data.get("patterns", {}).items():
                for pattern_type_name, pattern_list in pattern_types.items():
                    try:
                        ptype = PatternType[pattern_type_name]
                    except KeyError:
                        logger.warning("invalid_pattern_type_in_json", pattern_type=pattern_type_name)
                        continue

                    for pattern_def in pattern_list:
                        try:
                            regex = pattern_def["regex"]
                            severity = Decimal(pattern_def["severity"])
                            confidence = Decimal(pattern_def["confidence"])
                            explanation = pattern_def["explanation"]

                            patterns.append(CompiledPattern(
                                pattern=re.compile(regex, flags),
                                pattern_type=ptype,
                                severity=severity,
                                confidence=confidence,
                                explanation=explanation
                            ))
                        except (KeyError, ValueError, re.error) as e:
                            logger.warning("invalid_pattern_in_json",
                                         pattern_type=pattern_type_name,
                                         error=str(e))
                            continue

            if not patterns:
                raise ValueError("Pattern list is empty - check JSON structure")

            logger.info("patterns_loaded_from_json",
                       path=str(config_path),
                       count=len(patterns),
                       version=data.get("version"))
            return patterns

        except json.JSONDecodeError as e:
            logger.error("patterns_invalid_json", path=str(config_path), error=str(e))
            raise ValueError(f"Invalid JSON in patterns config: {e}")

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

    def _detect_emotion_patterns(self, text: str) -> list[PatternMatch]:
        """
        Detect crisis-relevant emotions using transformer model (GoEmotions).
        Maps emotions like grief, sadness, fear to crisis indicators.
        """
        if not self._emotion_classifier:
            return []

        matches: list[PatternMatch] = []

        try:
            # Get all 28 emotion scores
            results = self._emotion_classifier(text, truncation=True, max_length=512)
            if not results:
                return []

            # Results is a list of dicts with 'label' and 'score'
            emotions = results[0] if isinstance(results[0], list) else results

            for emotion_result in emotions:
                label = emotion_result.get("label", "")
                score = emotion_result.get("score", 0.0)

                # Check if this emotion is crisis-relevant and above threshold
                if label in CRISIS_EMOTION_MAP and Decimal(str(score)) >= self._config.emotion_threshold:
                    pattern_type, base_severity = CRISIS_EMOTION_MAP[label]

                    # Scale severity by emotion confidence
                    severity = base_severity * Decimal(str(score))
                    confidence = Decimal(str(score))

                    matches.append(PatternMatch(
                        pattern_type=pattern_type,
                        matched_text=f"[emotion:{label}]",
                        position=0,  # Emotion applies to whole text
                        severity=severity,
                        confidence=confidence,
                        context=text[:120] if len(text) > 120 else text,
                        explanation=f"Detected {label} emotion (confidence: {score:.2f}) indicating potential {pattern_type.value.lower().replace('_', ' ')}"
                    ))

            if matches:
                logger.debug("emotion_patterns_detected",
                           count=len(matches),
                           emotions=[m.matched_text for m in matches])

        except Exception as e:
            logger.warning("emotion_detection_failed", error=str(e))

        return matches

    @traced(name="pattern_matcher.detect", attributes={"component": "pattern_matcher"})
    def detect(self, text: str, user_id: UUID | None = None) -> list[PatternMatch]:
        """
        Detect crisis patterns using hybrid spaCy + regex + transformer emotion approach.

        Args:
            text: Input text to analyze
            user_id: Optional user ID for logging

        Returns:
            List of pattern matches sorted by severity
        """
        if not text:
            return []

        matches: list[PatternMatch] = []

        # First: Use transformer emotion classifier if available (deep emotional understanding)
        if self._emotion_classifier:
            emotion_matches = self._detect_emotion_patterns(text)
            matches.extend(emotion_matches)

        # Second: Use spaCy patterns if available (linguistic understanding)
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

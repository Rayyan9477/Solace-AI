"""
Solace-AI Crisis Detector - Multi-layer progressive crisis detection system.
Implements 4-layer detection: Input Gate, Processing Guard, Output Filter, Continuous Monitor.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from solace_common.enums import CrisisLevel

logger = structlog.get_logger(__name__)


def _crisis_level_from_score(
    score: Decimal, settings: CrisisDetectorSettings | None = None
) -> CrisisLevel:
    """Determine crisis level from risk score using configurable thresholds."""
    critical_threshold = settings.critical_threshold if settings else Decimal("0.9")
    high_threshold = settings.high_threshold if settings else Decimal("0.7")
    elevated_threshold = settings.elevated_threshold if settings else Decimal("0.5")
    low_threshold = settings.low_threshold if settings else Decimal("0.3")

    if score >= critical_threshold:
        return CrisisLevel.CRITICAL
    if score >= high_threshold:
        return CrisisLevel.HIGH
    if score >= elevated_threshold:
        return CrisisLevel.ELEVATED
    if score >= low_threshold:
        return CrisisLevel.LOW
    return CrisisLevel.NONE


class RiskFactor(BaseModel):
    """Individual risk factor identified during detection."""

    factor_type: str = Field(..., description="Type of risk factor")
    severity: Decimal = Field(..., ge=0, le=1, description="Severity score")
    evidence: str = Field(..., description="Evidence supporting detection")
    confidence: Decimal = Field(..., ge=0, le=1, description="Detection confidence")
    detection_layer: int = Field(..., ge=1, le=4, description="Layer that detected this")


class DetectionResult(BaseModel):
    """Result from crisis detection analysis."""

    crisis_detected: bool = Field(default=False, description="Whether crisis was detected")
    crisis_level: CrisisLevel = Field(default=CrisisLevel.NONE, description="Crisis severity level")
    risk_score: Decimal = Field(
        default=Decimal("0.0"), ge=0, le=1, description="Overall risk score"
    )
    risk_factors: list[RiskFactor] = Field(
        default_factory=list, description="Identified risk factors"
    )
    trigger_indicators: list[str] = Field(default_factory=list, description="Trigger indicators")
    confidence: Decimal = Field(
        default=Decimal("0.0"), ge=0, le=1, description="Overall confidence"
    )
    detection_layers_triggered: list[int] = Field(
        default_factory=list, description="Layers that detected risk"
    )
    detection_time_ms: int = Field(default=0, ge=0, description="Detection time in ms")
    recommended_action: str = Field(default="continue", description="Recommended action")


class CrisisDetectorSettings(BaseSettings):
    """Configuration for crisis detection thresholds and behavior."""

    keyword_weight: Decimal = Field(
        default=Decimal("0.4"), description="Weight for keyword detection"
    )
    sentiment_weight: Decimal = Field(
        default=Decimal("0.2"), description="Weight for sentiment analysis"
    )
    pattern_weight: Decimal = Field(
        default=Decimal("0.25"), description="Weight for pattern matching"
    )
    history_weight: Decimal = Field(default=Decimal("0.15"), description="Weight for risk history")
    low_threshold: Decimal = Field(default=Decimal("0.3"), description="Threshold for LOW risk")
    elevated_threshold: Decimal = Field(
        default=Decimal("0.5"), description="Threshold for ELEVATED risk"
    )
    high_threshold: Decimal = Field(default=Decimal("0.7"), description="Threshold for HIGH risk")
    critical_threshold: Decimal = Field(
        default=Decimal("0.9"), description="Threshold for CRITICAL risk"
    )
    enable_layer_1: bool = Field(default=True, description="Enable input gate layer")
    enable_layer_2: bool = Field(default=True, description="Enable processing guard layer")
    enable_layer_3: bool = Field(default=True, description="Enable output filter layer")
    enable_layer_4: bool = Field(default=True, description="Enable continuous monitor layer")
    model_config = SettingsConfigDict(env_prefix="CRISIS_", env_file=".env", extra="ignore")


@dataclass
class KeywordSet:
    """Categorized crisis keywords with severity weights."""

    critical_keywords: list[str] = field(
        default_factory=lambda: [
            "suicide",
            "kill myself",
            "end my life",
            "want to die",
            "better off dead",
            "no reason to live",
            "can't go on",
            "end it all",
            "take my life",
        ]
    )
    high_keywords: list[str] = field(
        default_factory=lambda: [
            "self-harm",
            "hurt myself",
            "cutting",
            "overdose",
            "don't want to live",
            "hopeless",
            "no hope",
            "give up",
            "worthless",
            "burden",
        ]
    )
    elevated_keywords: list[str] = field(
        default_factory=lambda: [
            "depressed",
            "anxious",
            "scared",
            "overwhelmed",
            "can't cope",
            "falling apart",
            "breaking down",
            "losing control",
            "panic",
        ]
    )
    low_keywords: list[str] = field(
        default_factory=lambda: [
            "sad",
            "stressed",
            "worried",
            "upset",
            "frustrated",
            "tired",
            "exhausted",
        ]
    )


class Layer1InputGate:
    """Layer 1: Fast keyword and pattern-based input screening."""

    def __init__(self, settings: CrisisDetectorSettings) -> None:
        self._settings = settings
        self._keywords = KeywordSet()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for crisis detection."""
        return {
            "suicidal_ideation": re.compile(
                r"\b(want|going|planning|thinking)\s+(to\s+)?(die|kill|end|suicide)\b", re.I
            ),
            "self_harm": re.compile(
                r"\b(cut|cutting|burn|burning|hurt|hurting)\s+(myself|my\s+\w+)\b", re.I
            ),
            "hopelessness": re.compile(
                r"\b(no\s+(hope|point|reason)|never\s+get\s+better|always\s+be\s+this\s+way)\b",
                re.I,
            ),
            "plan_indicators": re.compile(
                r"\b(have\s+a\s+plan|know\s+how|figured\s+out|pills|gun|rope|bridge)\b", re.I
            ),
            "farewell": re.compile(
                r"\b(goodbye|farewell|sorry\s+for\s+everything|tell\s+\w+\s+i\s+love)\b", re.I
            ),
            "timeframe": re.compile(
                r"\b(tonight|today|tomorrow|soon|this\s+week)\s+.*(end|die|gone|over)\b", re.I
            ),
        }

    def detect(
        self, content: str, user_risk_history: dict[str, Any] | None = None
    ) -> DetectionResult:
        """Perform Layer 1 input gate detection."""
        start_time = time.perf_counter()
        risk_factors: list[RiskFactor] = []
        trigger_indicators: list[str] = []
        content_lower = content.lower()
        keyword_score = self._detect_keywords(content_lower, risk_factors, trigger_indicators)
        pattern_score = self._detect_patterns(content, risk_factors, trigger_indicators)
        history_score = (
            self._check_risk_history(user_risk_history, risk_factors)
            if user_risk_history
            else Decimal("0")
        )

        # Use configurable weights from settings
        base_score = (
            keyword_score * self._settings.keyword_weight
            + pattern_score * self._settings.pattern_weight
            + history_score * self._settings.history_weight
        )
        # Normalize to 0-1 range considering total weights
        total_weight = (
            self._settings.keyword_weight
            + self._settings.pattern_weight
            + self._settings.history_weight
        )
        normalized_score = base_score / total_weight if total_weight > 0 else base_score

        # Apply boost for multi-signal detection
        boosted_score = normalized_score + (
            pattern_score * Decimal("0.1")
            if pattern_score > Decimal("0") and keyword_score > Decimal("0")
            else Decimal("0")
        )
        total_score = min(boosted_score, Decimal("1.0"))
        detection_time_ms = int((time.perf_counter() - start_time) * 1000)
        crisis_level = _crisis_level_from_score(total_score, self._settings)
        return DetectionResult(
            crisis_detected=crisis_level != CrisisLevel.NONE,
            crisis_level=crisis_level,
            risk_score=total_score,
            risk_factors=risk_factors,
            trigger_indicators=trigger_indicators,
            confidence=self._calculate_confidence(risk_factors),
            detection_layers_triggered=[1] if crisis_level != CrisisLevel.NONE else [],
            detection_time_ms=detection_time_ms,
            recommended_action=self._get_recommended_action(crisis_level),
        )

    def _detect_keywords(
        self, content: str, risk_factors: list[RiskFactor], triggers: list[str]
    ) -> Decimal:
        """Detect crisis keywords in content using word-boundary matching."""
        score = Decimal("0")
        for kw in self._keywords.critical_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', content, re.IGNORECASE):
                score = max(score, Decimal("0.95"))
                risk_factors.append(
                    RiskFactor(
                        factor_type="critical_keyword",
                        severity=Decimal("0.95"),
                        evidence=f"Critical keyword detected: '{kw}'",
                        confidence=Decimal("0.9"),
                        detection_layer=1,
                    )
                )
                triggers.append(f"CRITICAL_KEYWORD:{kw}")
        for kw in self._keywords.high_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', content, re.IGNORECASE):
                score = max(score, Decimal("0.75"))
                risk_factors.append(
                    RiskFactor(
                        factor_type="high_keyword",
                        severity=Decimal("0.75"),
                        evidence=f"High-risk keyword detected: '{kw}'",
                        confidence=Decimal("0.85"),
                        detection_layer=1,
                    )
                )
                triggers.append(f"HIGH_KEYWORD:{kw}")
        for kw in self._keywords.elevated_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', content, re.IGNORECASE):
                score = max(score, Decimal("0.5"))
                risk_factors.append(
                    RiskFactor(
                        factor_type="elevated_keyword",
                        severity=Decimal("0.5"),
                        evidence=f"Elevated keyword detected: '{kw}'",
                        confidence=Decimal("0.7"),
                        detection_layer=1,
                    )
                )
        for kw in self._keywords.low_keywords:
            if re.search(rf'\b{re.escape(kw)}\b', content, re.IGNORECASE):
                score = max(score, Decimal("0.25"))
        return score

    def _detect_patterns(
        self, content: str, risk_factors: list[RiskFactor], triggers: list[str]
    ) -> Decimal:
        """Detect crisis patterns using regex."""
        max_score = Decimal("0")
        pattern_scores = {
            "suicidal_ideation": Decimal("0.95"),
            "self_harm": Decimal("0.8"),
            "hopelessness": Decimal("0.6"),
            "plan_indicators": Decimal("0.9"),
            "farewell": Decimal("0.85"),
            "timeframe": Decimal("0.9"),
        }
        for pattern_name, pattern in self._patterns.items():
            if pattern.search(content):
                score = pattern_scores.get(pattern_name, Decimal("0.5"))
                max_score = max(max_score, score)
                risk_factors.append(
                    RiskFactor(
                        factor_type=f"pattern_{pattern_name}",
                        severity=score,
                        evidence=f"Pattern '{pattern_name}' detected",
                        confidence=Decimal("0.8"),
                        detection_layer=1,
                    )
                )
                triggers.append(f"PATTERN:{pattern_name}")
        return max_score

    def _check_risk_history(
        self, history: dict[str, Any], risk_factors: list[RiskFactor]
    ) -> Decimal:
        """Check user's risk history for elevated baseline."""
        score = Decimal("0")
        if history.get("previous_crisis_events", 0) > 0:
            score = Decimal("0.3")
            risk_factors.append(
                RiskFactor(
                    factor_type="risk_history",
                    severity=Decimal("0.3"),
                    evidence="Previous crisis events in history",
                    confidence=Decimal("0.7"),
                    detection_layer=1,
                )
            )
        if history.get("recent_escalation", False):
            score = max(score, Decimal("0.5"))
        if history.get("high_risk_flag", False):
            score = max(score, Decimal("0.6"))
        return score

    def _calculate_confidence(self, risk_factors: list[RiskFactor]) -> Decimal:
        """Calculate overall detection confidence."""
        if not risk_factors:
            return Decimal("0.0")
        total_confidence = sum(rf.confidence for rf in risk_factors)
        return min(total_confidence / len(risk_factors), Decimal("1.0"))

    def _get_recommended_action(self, level: CrisisLevel) -> str:
        """Get recommended action based on crisis level."""
        actions = {
            CrisisLevel.NONE: "continue",
            CrisisLevel.LOW: "monitor",
            CrisisLevel.ELEVATED: "assess",
            CrisisLevel.HIGH: "intervene",
            CrisisLevel.CRITICAL: "escalate_immediately",
        }
        return actions.get(level, "continue")


class Layer2ProcessingGuard:
    """Layer 2: Context validation and technique appropriateness checking."""

    def __init__(self, settings: CrisisDetectorSettings) -> None:
        self._settings = settings
        self._contraindicated_techniques = {
            CrisisLevel.CRITICAL: ["exposure_therapy", "trauma_processing", "challenging_thoughts"],
            CrisisLevel.HIGH: ["exposure_therapy", "trauma_processing"],
            CrisisLevel.ELEVATED: ["exposure_therapy"],
        }

    def validate_context(
        self, content: str, context: dict[str, Any], layer1_result: DetectionResult
    ) -> DetectionResult:
        """Validate processing context and check for contraindications."""
        start_time = time.perf_counter()
        risk_factors = list(layer1_result.risk_factors)
        triggers = list(layer1_result.trigger_indicators)
        layers_triggered = list(layer1_result.detection_layers_triggered)
        risk_score = layer1_result.risk_score
        active_technique = context.get("active_technique")
        if active_technique and layer1_result.crisis_level in self._contraindicated_techniques:
            contraindicated = self._contraindicated_techniques[layer1_result.crisis_level]
            if active_technique in contraindicated:
                risk_score = min(risk_score + Decimal("0.2"), Decimal("1.0"))
                risk_factors.append(
                    RiskFactor(
                        factor_type="contraindication",
                        severity=Decimal("0.7"),
                        evidence=f"Contraindicated technique '{active_technique}' for {layer1_result.crisis_level.value}",
                        confidence=Decimal("0.95"),
                        detection_layer=2,
                    )
                )
                triggers.append(f"CONTRAINDICATION:{active_technique}")
                layers_triggered.append(2)
        severity_mismatch = self._check_severity_appropriateness(
            context, layer1_result.crisis_level
        )
        if severity_mismatch:
            risk_score = min(risk_score + Decimal("0.1"), Decimal("1.0"))
            risk_factors.append(severity_mismatch)
            layers_triggered.append(2) if 2 not in layers_triggered else None
        detection_time_ms = layer1_result.detection_time_ms + int(
            (time.perf_counter() - start_time) * 1000
        )
        crisis_level = _crisis_level_from_score(risk_score, self._settings)
        return DetectionResult(
            crisis_detected=crisis_level != CrisisLevel.NONE,
            crisis_level=crisis_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            trigger_indicators=triggers,
            confidence=self._calculate_confidence(risk_factors),
            detection_layers_triggered=layers_triggered,
            detection_time_ms=detection_time_ms,
            recommended_action=layer1_result.recommended_action,
        )

    def _check_severity_appropriateness(
        self, context: dict[str, Any], crisis_level: CrisisLevel
    ) -> RiskFactor | None:
        """Check if intervention severity matches crisis level."""
        intervention_intensity = context.get("intervention_intensity", "normal")
        if (
            crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
            and intervention_intensity == "high"
        ):
            return RiskFactor(
                factor_type="severity_mismatch",
                severity=Decimal("0.4"),
                evidence="High intensity intervention inappropriate for crisis state",
                confidence=Decimal("0.8"),
                detection_layer=2,
            )
        return None

    def _calculate_confidence(self, risk_factors: list[RiskFactor]) -> Decimal:
        """Calculate overall detection confidence."""
        if not risk_factors:
            return Decimal("0.0")
        return min(sum(rf.confidence for rf in risk_factors) / len(risk_factors), Decimal("1.0"))


class Layer3OutputFilter:
    """Layer 3: Output safety filtering and resource inclusion."""

    def __init__(self, settings: CrisisDetectorSettings) -> None:
        self._settings = settings
        # Context-aware harmful patterns: only flag directive/endorsing harm,
        # NOT therapeutic reflections like "I hear you feel worthless"
        self._harmful_patterns = [
            re.compile(r"\byou\s+should\s+(just\s+)?(end|kill|hurt|give\s+up)", re.I),
            re.compile(r"\bjust\s+(end|do)\s+it\b", re.I),
            re.compile(r"\bthere'?s\s+no\s+point\s+(in|to)\s+(living|trying|going\s+on)", re.I),
            re.compile(r"\byou'?re\s+(a\s+)?burden\b", re.I),
            re.compile(r"\bnobody\s+(cares|would\s+miss)\b", re.I),
            re.compile(r"\bgive\s+up\s+on\s+(everything|life|yourself)", re.I),
        ]
        self._required_elements = {
            CrisisLevel.HIGH: ["support", "help", "resources"],
            CrisisLevel.CRITICAL: ["crisis", "988", "emergency", "immediate"],
        }

    def filter_output(
        self, response: str, crisis_level: CrisisLevel
    ) -> tuple[str, list[str], bool]:
        """Filter output for safety and add required elements.

        Uses context-aware patterns that detect directive harm
        (e.g. 'you should end it') but allow therapeutic reflections
        (e.g. 'I hear you feel worthless').
        """
        modifications: list[str] = []
        filtered = response
        for pattern in self._harmful_patterns:
            match = pattern.search(filtered)
            if match:
                phrase = match.group(0)
                modifications.append(f"Removed harmful directive: '{phrase}'")
                filtered = pattern.sub("[supportive message]", filtered)
        if crisis_level in self._required_elements:
            required = self._required_elements[crisis_level]
            missing = [elem for elem in required if elem.lower() not in filtered.lower()]
            if missing:
                modifications.append(
                    f"Response missing required elements for {crisis_level.value}: {missing}"
                )
        is_safe = len(modifications) == 0 or all("missing" in m.lower() for m in modifications)
        return filtered, modifications, is_safe


class Layer4ContinuousMonitor:
    """Layer 4: Session trajectory and deterioration detection."""

    def __init__(self, settings: CrisisDetectorSettings) -> None:
        self._settings = settings

    def analyze_trajectory(
        self, conversation_history: list[str], current_level: CrisisLevel
    ) -> dict[str, Any]:
        """Analyze session trajectory for deterioration patterns."""
        if len(conversation_history) < 2:
            return {
                "trend": "insufficient_data",
                "deteriorating": False,
                "risk_delta": Decimal("0"),
            }
        recent_negative_count = sum(
            1 for msg in conversation_history[-5:] if self._is_negative(msg)
        )
        total_messages = min(len(conversation_history), 5)
        negative_ratio = (
            Decimal(str(recent_negative_count / total_messages))
            if total_messages > 0
            else Decimal("0")
        )
        deteriorating = negative_ratio > Decimal("0.6") and current_level in (
            CrisisLevel.ELEVATED,
            CrisisLevel.HIGH,
        )
        trend = (
            "deteriorating"
            if deteriorating
            else "stable"
            if negative_ratio < Decimal("0.4")
            else "concerning"
        )
        return {
            "trend": trend,
            "deteriorating": deteriorating,
            "risk_delta": negative_ratio - Decimal("0.3"),
            "negative_ratio": negative_ratio,
            "recent_messages_analyzed": total_messages,
        }

    def _is_negative(self, message: str) -> bool:
        """Check if message has negative sentiment indicators."""
        negative_indicators = [
            "worse",
            "bad",
            "terrible",
            "hopeless",
            "can't",
            "won't",
            "never",
            "hate",
            "alone",
        ]
        return any(ind in message.lower() for ind in negative_indicators)


class CrisisDetector:
    """Main crisis detector orchestrating all detection layers."""

    def __init__(self, settings: CrisisDetectorSettings | None = None) -> None:
        self._settings = settings or CrisisDetectorSettings()
        self._layer1 = Layer1InputGate(self._settings)
        self._layer2 = Layer2ProcessingGuard(self._settings)
        self._layer3 = Layer3OutputFilter(self._settings)
        self._layer4 = Layer4ContinuousMonitor(self._settings)
        logger.info(
            "crisis_detector_initialized",
            layers_enabled={
                "layer1": self._settings.enable_layer_1,
                "layer2": self._settings.enable_layer_2,
                "layer3": self._settings.enable_layer_3,
                "layer4": self._settings.enable_layer_4,
            },
        )

    async def detect(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        conversation_history: list[str] | None = None,
        user_risk_history: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Perform multi-layer crisis detection."""
        context = context or {}
        result = (
            self._layer1.detect(content, user_risk_history)
            if self._settings.enable_layer_1
            else DetectionResult()
        )
        if self._settings.enable_layer_2 and context:
            result = self._layer2.validate_context(content, context, result)
        if self._settings.enable_layer_4 and conversation_history:
            trajectory = self._layer4.analyze_trajectory(conversation_history, result.crisis_level)
            if trajectory.get("deteriorating"):
                result.risk_score = min(result.risk_score + Decimal("0.1"), Decimal("1.0"))
                result.crisis_level = _crisis_level_from_score(result.risk_score, self._settings)
                result.trigger_indicators.append("TRAJECTORY:deteriorating")
                if 4 not in result.detection_layers_triggered:
                    result.detection_layers_triggered.append(4)
        logger.info(
            "crisis_detection_complete",
            crisis_level=result.crisis_level.value,
            risk_score=float(result.risk_score),
            layers_triggered=result.detection_layers_triggered,
        )
        return result

    def filter_output(
        self, response: str, crisis_level: CrisisLevel
    ) -> tuple[str, list[str], bool]:
        """Filter output using Layer 3."""
        return self._layer3.filter_output(response, crisis_level)

    def analyze_trajectory(self, history: list[str], current_level: CrisisLevel) -> dict[str, Any]:
        """Analyze conversation trajectory using Layer 4."""
        return self._layer4.analyze_trajectory(history, current_level)

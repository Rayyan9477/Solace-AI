"""
Solace-AI Sentiment Analyzer - Clinical psychology-focused sentiment analysis for risk assessment.
Uses adaptive approach: Small transformer model if GPU available, VADER otherwise, + clinical lexicon.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import re
import structlog

# Check for CUDA availability
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

try:
    from services.safety_service.src.infrastructure.telemetry import traced, get_telemetry

    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

    def traced(*args, **kwargs):
        """No-op decorator when telemetry unavailable."""

        def decorator(func):
            return func

        return decorator


# Try to load transformer model if CUDA available
TRANSFORMERS_AVAILABLE = False
if CUDA_AVAILABLE:
    try:
        from transformers import pipeline

        TRANSFORMERS_AVAILABLE = True
    except Exception:
        TRANSFORMERS_AVAILABLE = False

# Fallback to VADER if no GPU or transformers unavailable
VADER_AVAILABLE = False
if not TRANSFORMERS_AVAILABLE:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        VADER_AVAILABLE = True
    except Exception:
        VADER_AVAILABLE = False

logger = structlog.get_logger(__name__)


class SentimentPolarity(str, Enum):
    """Sentiment polarity classification."""

    VERY_NEGATIVE = "VERY_NEGATIVE"  # Extreme distress
    NEGATIVE = "NEGATIVE"  # Negative emotions
    NEUTRAL = "NEUTRAL"  # Balanced or informational
    POSITIVE = "POSITIVE"  # Positive emotions
    VERY_POSITIVE = "VERY_POSITIVE"  # Strong wellbeing


class EmotionalState(str, Enum):
    """Emotional state classification for mental health."""

    CRISIS = "CRISIS"  # Active crisis state
    SEVERE_DISTRESS = "SEVERE_DISTRESS"  # Severe emotional pain
    MODERATE_DISTRESS = "MODERATE_DISTRESS"  # Notable distress
    MILD_DISTRESS = "MILD_DISTRESS"  # Mild negative emotions
    NEUTRAL = "NEUTRAL"  # Balanced state
    POSITIVE = "POSITIVE"  # Positive emotional state
    THRIVING = "THRIVING"  # Strong wellbeing


class SentimentResult(BaseModel):
    """Result from sentiment analysis."""

    polarity: SentimentPolarity = Field(..., description="Overall sentiment polarity")
    emotional_state: EmotionalState = Field(..., description="Clinical emotional state")
    compound_score: Decimal = Field(..., ge=-1, le=1, description="Compound sentiment score")
    risk_score: Decimal = Field(..., ge=0, le=1, description="Risk score derived from sentiment")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence in analysis")
    positive_score: Decimal = Field(..., ge=0, le=1, description="Positive sentiment strength")
    negative_score: Decimal = Field(..., ge=0, le=1, description="Negative sentiment strength")
    distress_indicators: list[str] = Field(
        default_factory=list, description="Detected distress phrases"
    )
    protective_indicators: list[str] = Field(
        default_factory=list, description="Detected protective factors"
    )


class SentimentAnalyzerConfig(BaseSettings):
    """Configuration for sentiment analyzer."""

    # ML model settings (GPU-based)
    ml_model_name: str = Field(
        default="distilbert-base-uncased-finetuned-sst-2-english",
        description="Small transformer model (~250MB) for accurate sentiment",
    )
    ml_weight: Decimal = Field(default=Decimal("0.7"), description="Weight for ML predictions")
    lexicon_weight: Decimal = Field(
        default=Decimal("0.3"), description="Weight for clinical lexicon"
    )

    # Lexicon-based settings
    enable_negation_handling: bool = Field(default=True, description="Handle negation in text")
    enable_intensifier_detection: bool = Field(default=True, description="Detect intensifiers")
    negation_window: int = Field(default=3, ge=1, description="Words to look back for negation")
    intensifier_boost: Decimal = Field(
        default=Decimal("1.5"), description="Boost factor for intensifiers"
    )
    negation_flip_factor: Decimal = Field(
        default=Decimal("0.5"), description="Factor for negation reversal"
    )
    clinical_weight: Decimal = Field(
        default=Decimal("1.3"), description="Weight for clinical terms"
    )
    min_confidence_threshold: Decimal = Field(
        default=Decimal("0.6"), description="Minimum confidence"
    )

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env", extra="ignore")


@dataclass
class LexiconEntry:
    """Entry in sentiment lexicon."""

    word: str
    score: Decimal  # -1.0 to 1.0
    is_clinical: bool = False  # Clinical psychology term
    is_crisis: bool = False  # Crisis indicator


class SentimentAnalyzer:
    """
    Clinical psychology-focused sentiment analyzer for mental health contexts.
    Adaptive approach: Transformer model if GPU available for accuracy, VADER fallback, + clinical lexicon.
    """

    def __init__(self, config: SentimentAnalyzerConfig | None = None) -> None:
        """Initialize sentiment analyzer with configuration."""
        self._config = config or SentimentAnalyzerConfig()
        self._lexicon = self._build_clinical_lexicon()
        self._negation_words = self._load_negation_words()
        self._intensifiers = self._load_intensifiers()

        # Try transformer model first if CUDA available (best accuracy)
        self._ml_classifier = None
        self._vader = None
        self._mode = "lexicon_only"

        if TRANSFORMERS_AVAILABLE:
            try:
                device = 0 if CUDA_AVAILABLE else -1  # 0 = GPU, -1 = CPU
                self._ml_classifier = pipeline(
                    "sentiment-analysis",
                    model=self._config.ml_model_name,
                    device=device,
                    truncation=True,
                )
                self._mode = (
                    "hybrid_transformer_gpu" if CUDA_AVAILABLE else "hybrid_transformer_cpu"
                )
                logger.info(
                    "sentiment_analyzer_initialized",
                    mode=self._mode,
                    model=self._config.ml_model_name,
                    device="cuda" if CUDA_AVAILABLE else "cpu",
                    lexicon_size=len(self._lexicon),
                    clinical_terms=sum(1 for e in self._lexicon.values() if e.is_clinical),
                )
            except Exception as e:
                logger.warning("transformer_load_failed", error=str(e), fallback="vader")
                self._ml_classifier = None

        # Fallback to VADER if transformers unavailable
        if self._ml_classifier is None and VADER_AVAILABLE:
            try:
                self._vader = SentimentIntensityAnalyzer()
                self._mode = "hybrid_vader"
                logger.info(
                    "sentiment_analyzer_initialized",
                    mode=self._mode,
                    lexicon_size=len(self._lexicon),
                    clinical_terms=sum(1 for e in self._lexicon.values() if e.is_clinical),
                )
            except Exception as e:
                logger.warning("vader_load_failed", error=str(e), fallback="lexicon_only")
                self._vader = None

        # Final fallback: lexicon only
        if self._ml_classifier is None and self._vader is None:
            logger.info(
                "sentiment_analyzer_initialized",
                mode="lexicon_only",
                lexicon_size=len(self._lexicon),
                clinical_terms=sum(1 for e in self._lexicon.values() if e.is_clinical),
            )

    def _build_clinical_lexicon(self) -> dict[str, LexiconEntry]:
        """
        Build mental health-focused sentiment lexicon from JSON configuration.
        JSON config is the single source of truth for scalability.
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "clinical_lexicon.json"

        if not config_path.exists():
            logger.error(
                "clinical_lexicon_config_missing",
                path=str(config_path),
                action="create config/clinical_lexicon.json",
            )
            raise FileNotFoundError(
                f"Clinical lexicon configuration not found: {config_path}. "
                "Please create the JSON config file for scalable deployment."
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            lexicon: dict[str, LexiconEntry] = {}

            for category_name, category_data in data.get("lexicon", {}).items():
                is_clinical = category_data.get("is_clinical", False)
                is_crisis = category_data.get("is_crisis", False)

                for word, score in category_data.get("terms", {}).items():
                    lexicon[word] = LexiconEntry(
                        word=word,
                        score=Decimal(str(score)),
                        is_clinical=is_clinical,
                        is_crisis=is_crisis,
                    )

            if not lexicon:
                raise ValueError("Lexicon is empty - check JSON structure")

            logger.info(
                "clinical_lexicon_loaded",
                path=str(config_path),
                total_terms=len(lexicon),
                clinical_terms=sum(1 for e in lexicon.values() if e.is_clinical),
                crisis_terms=sum(1 for e in lexicon.values() if e.is_crisis),
                version=data.get("version"),
            )
            return lexicon

        except json.JSONDecodeError as e:
            logger.error("clinical_lexicon_invalid_json", path=str(config_path), error=str(e))
            raise ValueError(f"Invalid JSON in clinical lexicon config: {e}")

    def _load_negation_words(self) -> set[str]:
        """Load negation words for sentiment reversal from JSON configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "clinical_lexicon.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            negation_list = data.get("negation_words", [])
            if negation_list:
                return set(negation_list)
            logger.warning("negation_words_empty", path=str(config_path))
            return set()
        except Exception as e:
            logger.warning("negation_words_load_failed", error=str(e))
            return set()

    def _load_intensifiers(self) -> dict[str, Decimal]:
        """Load intensifier words with boost factors from JSON configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "clinical_lexicon.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            intensifiers = data.get("intensifiers", {})
            if intensifiers:
                return {word: Decimal(str(boost)) for word, boost in intensifiers.items()}
            logger.warning("intensifiers_empty", path=str(config_path))
            return {}
        except Exception as e:
            logger.warning("intensifiers_load_failed", error=str(e))
            return {}

    def _get_ml_sentiment(self, text: str) -> tuple[Decimal, Decimal]:
        """
        Get ML-based sentiment score from transformer model (accurate).

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (compound_score, confidence) where compound_score is -1 to 1
        """
        try:
            result = self._ml_classifier(text, truncation=True, max_length=512)[0]
            label = result["label"]
            score = Decimal(str(result["score"]))

            # Convert to compound score (-1 to 1)
            if label.upper() in ["POSITIVE", "POS", "1"]:
                compound = score  # 0 to 1 for positive
            else:
                compound = -score  # -1 to 0 for negative

            return compound, score  # Return compound score and confidence
        except Exception as e:
            logger.warning("ml_sentiment_failed", error=str(e))
            return Decimal("0.0"), Decimal("0.0")

    def _get_vader_sentiment(self, text: str) -> tuple[Decimal, Decimal]:
        """
        Get VADER sentiment score (fast fallback).

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (compound_score, confidence) where compound_score is -1 to 1
        """
        try:
            scores = self._vader.polarity_scores(text)
            compound = Decimal(str(scores["compound"]))  # Already -1 to 1
            # VADER confidence is based on magnitude
            confidence = Decimal(str(abs(scores["compound"])))
            return compound, confidence
        except Exception as e:
            logger.warning("vader_sentiment_failed", error=str(e))
            return Decimal("0.0"), Decimal("0.0")

    @traced(name="sentiment_analyzer.analyze", attributes={"component": "sentiment_analyzer"})
    def analyze(self, text: str, user_id: UUID | None = None) -> SentimentResult:
        """
        Analyze sentiment of text with clinical psychology focus using hybrid approach.

        Args:
            text: Input text to analyze
            user_id: Optional user ID for logging

        Returns:
            Sentiment analysis result with risk assessment
        """
        if not text:
            return self._create_neutral_result()

        # Get ML sentiment (transformer or VADER)
        ml_score: Decimal | None = None
        ml_confidence: Decimal = Decimal("0.0")

        if self._ml_classifier:
            # Use transformer model (most accurate)
            ml_score, ml_confidence = self._get_ml_sentiment(text)
        elif self._vader:
            # Fall back to VADER
            ml_score, ml_confidence = self._get_vader_sentiment(text)

        # Tokenize and preprocess for lexicon analysis
        tokens = self._tokenize(text.lower())

        # Calculate sentiment scores
        positive_sum = Decimal("0.0")
        negative_sum = Decimal("0.0")
        total_weight = Decimal("0.0")
        distress_indicators: list[str] = []
        protective_indicators: list[str] = []
        crisis_count = 0

        for i, token in enumerate(tokens):
            if token not in self._lexicon:
                continue

            entry = self._lexicon[token]
            score = entry.score

            # Apply clinical weight
            if entry.is_clinical:
                score = score * self._config.clinical_weight

            # Check for negation in window
            if self._config.enable_negation_handling:
                if self._check_negation(tokens, i):
                    score = score * -self._config.negation_flip_factor

            # Check for intensifiers
            if self._config.enable_intensifier_detection:
                intensifier_boost = self._check_intensifier(tokens, i)
                if intensifier_boost > Decimal("1.0"):
                    score = score * intensifier_boost

            # Accumulate scores
            if score > 0:
                positive_sum += score
                if entry.is_clinical and score > Decimal("0.5"):
                    protective_indicators.append(entry.word)
            else:
                negative_sum += abs(score)
                if entry.is_clinical and abs(score) > Decimal("0.5"):
                    distress_indicators.append(entry.word)
                if entry.is_crisis:
                    crisis_count += 1

            total_weight += abs(score)

        # Normalize scores
        if total_weight > 0:
            positive_score = positive_sum / total_weight
            negative_score = negative_sum / total_weight
        else:
            positive_score = Decimal("0.0")
            negative_score = Decimal("0.0")

        # Calculate lexicon-based compound score (-1 to 1)
        lexicon_compound = positive_score - negative_score

        # Combine ML (transformer/VADER) and lexicon scores if ML is available
        if ml_score is not None:
            compound = (ml_score * self._config.ml_weight) + (
                lexicon_compound * self._config.lexicon_weight
            )
            final_confidence = (ml_confidence * self._config.ml_weight) + (
                min(total_weight / Decimal("5.0"), Decimal("1.0")) * self._config.lexicon_weight
            )
        else:
            compound = lexicon_compound
            final_confidence = min(total_weight / Decimal("5.0"), Decimal("1.0"))

        # Determine polarity and emotional state
        polarity = self._determine_polarity(compound)
        emotional_state = self._determine_emotional_state(compound, crisis_count, negative_score)

        # Calculate risk score (0 to 1)
        risk_score = self._calculate_risk_score(compound, crisis_count, negative_score)

        # Ensure confidence meets minimum threshold
        confidence = max(final_confidence, self._config.min_confidence_threshold)

        result = SentimentResult(
            polarity=polarity,
            emotional_state=emotional_state,
            compound_score=compound,
            risk_score=risk_score,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            distress_indicators=distress_indicators[:5],  # Limit to top 5
            protective_indicators=protective_indicators[:5],
        )

        if user_id:
            logger.info(
                "sentiment_analyzed",
                user_id=str(user_id),
                emotional_state=emotional_state.value,
                risk_score=float(risk_score),
            )

        return result

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        return [word for word in text.split() if word]

    def _check_negation(self, tokens: list[str], index: int) -> bool:
        """Check if word is negated within window."""
        start = max(0, index - self._config.negation_window)
        for i in range(start, index):
            if tokens[i] in self._negation_words:
                return True
        return False

    def _check_intensifier(self, tokens: list[str], index: int) -> Decimal:
        """Check for intensifier before word."""
        if index > 0 and tokens[index - 1] in self._intensifiers:
            return self._intensifiers[tokens[index - 1]]
        return Decimal("1.0")

    def _determine_polarity(self, compound: Decimal) -> SentimentPolarity:
        """Determine sentiment polarity from compound score."""
        if compound >= Decimal("0.5"):
            return SentimentPolarity.VERY_POSITIVE
        if compound >= Decimal("0.1"):
            return SentimentPolarity.POSITIVE
        if compound >= Decimal("-0.1"):
            return SentimentPolarity.NEUTRAL
        if compound >= Decimal("-0.5"):
            return SentimentPolarity.NEGATIVE
        return SentimentPolarity.VERY_NEGATIVE

    def _determine_emotional_state(
        self, compound: Decimal, crisis_count: int, negative_score: Decimal
    ) -> EmotionalState:
        """Determine clinical emotional state."""
        if crisis_count > 0 or compound <= Decimal("-0.7"):
            return EmotionalState.CRISIS
        if compound <= Decimal("-0.5") or negative_score >= Decimal("0.7"):
            return EmotionalState.SEVERE_DISTRESS
        if compound <= Decimal("-0.3") or negative_score >= Decimal("0.5"):
            return EmotionalState.MODERATE_DISTRESS
        if compound <= Decimal("-0.1") or negative_score >= Decimal("0.3"):
            return EmotionalState.MILD_DISTRESS
        if compound >= Decimal("0.4"):
            return EmotionalState.THRIVING
        if compound >= Decimal("0.1"):
            return EmotionalState.POSITIVE
        return EmotionalState.NEUTRAL

    def _calculate_risk_score(
        self, compound: Decimal, crisis_count: int, negative_score: Decimal
    ) -> Decimal:
        """Calculate risk score from sentiment analysis."""
        # Base risk from negative sentiment
        base_risk = (Decimal("1.0") - compound) / Decimal("2.0")  # Map -1..1 to 1..0

        # Boost for crisis terms
        crisis_boost = min(Decimal(str(crisis_count)) * Decimal("0.2"), Decimal("0.5"))

        # Boost for high negative score
        negative_boost = negative_score * Decimal("0.3")

        total_risk = base_risk + crisis_boost + negative_boost
        return min(total_risk, Decimal("1.0"))

    def _create_neutral_result(self) -> SentimentResult:
        """Create neutral sentiment result for empty input."""
        return SentimentResult(
            polarity=SentimentPolarity.NEUTRAL,
            emotional_state=EmotionalState.NEUTRAL,
            compound_score=Decimal("0.0"),
            risk_score=Decimal("0.0"),
            confidence=Decimal("0.0"),
            positive_score=Decimal("0.0"),
            negative_score=Decimal("0.0"),
        )

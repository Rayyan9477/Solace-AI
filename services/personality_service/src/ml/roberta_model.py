"""
Solace-AI Personality Service - RoBERTa Big Five Classifier.
Fine-tuned RoBERTa model for OCEAN personality trait detection from text.
"""
from __future__ import annotations
import collections
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from ..schemas import PersonalityTrait, AssessmentSource, OceanScoresDTO, TraitScoreDTO

logger = structlog.get_logger(__name__)


class RoBERTaSettings(BaseSettings):
    """RoBERTa personality detector configuration."""
    model_name: str = Field(default="roberta-base")
    model_path: str | None = Field(default=None)
    max_sequence_length: int = Field(default=512, ge=64, le=2048)
    batch_size: int = Field(default=8, ge=1, le=64)
    device: str = Field(default="cpu")
    num_labels: int = Field(default=5)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    output_hidden_states: bool = Field(default=False)
    cache_embeddings: bool = Field(default=True)
    embedding_dim: int = Field(default=768)
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_ROBERTA_", env_file=".env", extra="ignore")


class TokenizerProtocol(Protocol):
    """Protocol for text tokenizers."""
    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]: ...


class ModelProtocol(Protocol):
    """Protocol for transformer models."""
    def __call__(self, **kwargs: Any) -> Any: ...
    def eval(self) -> None: ...


@dataclass
class RoBERTaPrediction:
    """Single RoBERTa prediction result."""
    prediction_id: UUID = field(default_factory=uuid4)
    trait_logits: dict[PersonalityTrait, float] = field(default_factory=dict)
    trait_probabilities: dict[PersonalityTrait, float] = field(default_factory=dict)
    confidence: float = 0.5
    embedding: list[float] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BatchPredictionResult:
    """Batch prediction results."""
    predictions: list[RoBERTaPrediction]
    total_processing_time_ms: float = 0.0
    batch_confidence: float = 0.5


class TextPreprocessor:
    """Preprocesses text for RoBERTa input."""
    _URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    _MENTION_PATTERN = re.compile(r'@\w+')
    _HASHTAG_PATTERN = re.compile(r'#\w+')
    _WHITESPACE_PATTERN = re.compile(r'\s+')

    def preprocess(self, text: str) -> str:
        """Clean and normalize text for model input."""
        text = self._URL_PATTERN.sub('[URL]', text)
        text = self._MENTION_PATTERN.sub('[MENTION]', text)
        text = self._HASHTAG_PATTERN.sub(lambda m: m.group()[1:], text)
        text = self._WHITESPACE_PATTERN.sub(' ', text)
        return text.strip()[:2048]

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(t) for t in texts]


class SigmoidActivation:
    """Sigmoid activation for multi-label classification."""
    def __call__(self, logits: list[float]) -> list[float]:
        import math
        return [1.0 / (1.0 + math.exp(-min(max(x, -20), 20))) for x in logits]


class PersonalityClassificationHead:
    """Classification head for Big Five traits.

    Uses a linear projection (W @ pooled_output + bias) to compute logits.
    Weights are initialised from a seeded RNG so behaviour is deterministic.
    For production accuracy, load fine-tuned weights via ``load_weights()``.
    """

    def __init__(self, settings: RoBERTaSettings) -> None:
        self._settings = settings
        self._sigmoid = SigmoidActivation()
        self._trait_indices = {
            PersonalityTrait.OPENNESS: 0,
            PersonalityTrait.CONSCIENTIOUSNESS: 1,
            PersonalityTrait.EXTRAVERSION: 2,
            PersonalityTrait.AGREEABLENESS: 3,
            PersonalityTrait.NEUROTICISM: 4,
        }
        self._weights: list[list[float]] | None = None
        self._bias: list[float] | None = None
        self._weights_loaded = False

    def load_weights(self, weights: list[list[float]], bias: list[float]) -> None:
        """Load fine-tuned classification head weights.

        Args:
            weights: Matrix of shape (num_labels, hidden_dim).
            bias: Bias vector of shape (num_labels,).
        """
        self._weights = weights
        self._bias = bias
        self._weights_loaded = True

    def _ensure_weights(self, hidden_dim: int) -> tuple[list[list[float]], list[float]]:
        """Return loaded weights or create Xavier-initialised defaults."""
        if self._weights is not None and self._bias is not None:
            return self._weights, self._bias
        import random
        rng = random.Random(42)
        scale = (2.0 / (hidden_dim + self._settings.num_labels)) ** 0.5
        self._weights = [
            [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
            for _ in range(self._settings.num_labels)
        ]
        self._bias = [0.0] * self._settings.num_labels
        if not self._weights_loaded:
            import structlog
            structlog.get_logger(__name__).warning(
                "classification_head_using_default_weights",
                hint="Load fine-tuned checkpoint via load_weights() for production accuracy",
            )
        return self._weights, self._bias

    def forward(self, pooled_output: list[float]) -> RoBERTaPrediction:
        """Forward pass through classification head."""
        logits = self._compute_logits(pooled_output)
        probabilities = self._sigmoid(logits)
        trait_logits = {trait: logits[idx] for trait, idx in self._trait_indices.items()}
        trait_probs = {trait: probabilities[idx] for trait, idx in self._trait_indices.items()}
        confidence = self._compute_confidence(probabilities)
        return RoBERTaPrediction(
            trait_logits=trait_logits,
            trait_probabilities=trait_probs,
            confidence=confidence,
            embedding=pooled_output if self._settings.output_hidden_states else None,
        )

    def _compute_logits(self, pooled_output: list[float]) -> list[float]:
        """Compute logits via linear projection: W @ x + b."""
        weights, bias = self._ensure_weights(len(pooled_output))
        return [
            sum(w * x for w, x in zip(weights[i], pooled_output)) + bias[i]
            for i in range(self._settings.num_labels)
        ]

    def _compute_confidence(self, probabilities: list[float]) -> float:
        """Compute overall prediction confidence."""
        variance = sum((p - 0.5) ** 2 for p in probabilities) / len(probabilities)
        base = 0.5 + variance * 2
        if not self._weights_loaded:
            base *= 0.5  # Lower confidence when using default weights
        return min(0.95, base)


class RoBERTaPersonalityDetector:
    """Fine-tuned RoBERTa classifier for OCEAN personality detection."""

    def __init__(
        self,
        settings: RoBERTaSettings | None = None,
        tokenizer: TokenizerProtocol | None = None,
        model: ModelProtocol | None = None,
    ) -> None:
        self._settings = settings or RoBERTaSettings()
        self._tokenizer = tokenizer
        self._model = model
        self._preprocessor = TextPreprocessor()
        self._classification_head = PersonalityClassificationHead(self._settings)
        self._initialized = False
        self._embedding_cache: collections.OrderedDict[str, list[float]] = collections.OrderedDict()
        self._cache_max_size = 1000

    async def initialize(self) -> None:
        """Initialize the RoBERTa detector."""
        if self._model is not None:
            self._model.eval()
        self._initialized = True
        logger.info(
            "roberta_detector_initialized",
            model=self._settings.model_name,
            device=self._settings.device,
            has_model=self._model is not None,
        )

    async def detect(self, text: str) -> OceanScoresDTO:
        """Detect OCEAN personality traits from text."""
        if not text or len(text.strip()) < 10:
            logger.warning("text_too_short_for_roberta", length=len(text))
            return self._neutral_scores(confidence=0.2)
        processed_text = self._preprocessor.preprocess(text)
        pooled_output = await self._get_embeddings(processed_text)
        prediction = self._classification_head.forward(pooled_output)
        return self._prediction_to_scores(prediction)

    async def detect_batch(self, texts: list[str]) -> list[OceanScoresDTO]:
        """Detect personality traits for a batch of texts."""
        if not texts:
            return []
        processed_texts = self._preprocessor.preprocess_batch(texts)
        results = []
        for batch_start in range(0, len(processed_texts), self._settings.batch_size):
            batch_end = min(batch_start + self._settings.batch_size, len(processed_texts))
            batch_texts = processed_texts[batch_start:batch_end]
            for text in batch_texts:
                if len(text.strip()) < 10:
                    results.append(self._neutral_scores(confidence=0.2))
                else:
                    pooled_output = await self._get_embeddings(text)
                    prediction = self._classification_head.forward(pooled_output)
                    results.append(self._prediction_to_scores(prediction))
        logger.info("roberta_batch_detection_complete", batch_size=len(texts), processed=len(results))
        return results

    async def get_embeddings(self, text: str) -> list[float]:
        """Get text embeddings from RoBERTa encoder."""
        processed_text = self._preprocessor.preprocess(text)
        return await self._get_embeddings(processed_text)

    async def _get_embeddings(self, processed_text: str) -> list[float]:
        """Get embeddings with caching support."""
        cache_key = hashlib.sha256(processed_text.encode()).hexdigest()
        if self._settings.cache_embeddings and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        if self._model is not None and self._tokenizer is not None:
            embeddings = await self._run_model(processed_text)
        else:
            embeddings = self._generate_heuristic_embeddings(processed_text)
        if self._settings.cache_embeddings:
            if len(self._embedding_cache) >= self._cache_max_size:
                self._embedding_cache.popitem(last=False)
            self._embedding_cache[cache_key] = embeddings
        return embeddings

    async def _run_model(self, text: str) -> list[float]:
        """Run actual model inference in a thread to avoid blocking the event loop."""
        import asyncio
        return await asyncio.to_thread(self._run_model_sync, text)

    def _run_model_sync(self, text: str) -> list[float]:
        """Synchronous model inference (CPU-bound)."""
        inputs = self._tokenizer(
            text,
            max_length=self._settings.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        outputs = self._model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return pooled if isinstance(pooled, list) else [pooled]

    def _generate_heuristic_embeddings(self, text: str) -> list[float]:
        """Generate heuristic embeddings when model is not available."""
        words = text.lower().split()
        word_count = len(words) or 1
        embeddings = [0.0] * self._settings.embedding_dim
        feature_map = {
            'i': (0, 0.5), 'me': (0, 0.5), 'my': (0, 0.5),
            'we': (10, 0.4), 'us': (10, 0.4), 'our': (10, 0.4),
            'happy': (20, 0.6), 'love': (20, 0.6), 'great': (20, 0.6), 'good': (20, 0.5),
            'sad': (30, 0.5), 'angry': (30, 0.5), 'worried': (30, 0.5), 'anxious': (30, 0.6),
            'think': (40, 0.4), 'believe': (40, 0.4), 'understand': (40, 0.5),
            'achieve': (50, 0.5), 'goal': (50, 0.5), 'work': (50, 0.4), 'success': (50, 0.6),
            'maybe': (60, 0.3), 'perhaps': (60, 0.3), 'might': (60, 0.3),
            'realize': (70, 0.5), 'discover': (70, 0.5), 'learn': (70, 0.4),
            'friend': (80, 0.5), 'family': (80, 0.5), 'people': (80, 0.4),
            'feel': (90, 0.4), 'feeling': (90, 0.4), 'felt': (90, 0.4),
        }
        for word in words:
            if word in feature_map:
                idx, weight = feature_map[word]
                for offset in range(10):
                    embeddings[(idx + offset) % self._settings.embedding_dim] += weight / word_count
        embeddings[100] = min(1.0, word_count / 200)
        embeddings[101] = text.count('!') / max(1, word_count)
        embeddings[102] = text.count('?') / max(1, word_count)
        norm = sum(e * e for e in embeddings) ** 0.5 or 1.0
        return [e / norm for e in embeddings]

    def _prediction_to_scores(self, prediction: RoBERTaPrediction) -> OceanScoresDTO:
        """Convert prediction to OceanScoresDTO."""
        trait_scores = self._build_trait_scores(prediction)
        return OceanScoresDTO(
            openness=prediction.trait_probabilities[PersonalityTrait.OPENNESS],
            conscientiousness=prediction.trait_probabilities[PersonalityTrait.CONSCIENTIOUSNESS],
            extraversion=prediction.trait_probabilities[PersonalityTrait.EXTRAVERSION],
            agreeableness=prediction.trait_probabilities[PersonalityTrait.AGREEABLENESS],
            neuroticism=prediction.trait_probabilities[PersonalityTrait.NEUROTICISM],
            overall_confidence=prediction.confidence,
            trait_scores=trait_scores,
        )

    def _build_trait_scores(self, prediction: RoBERTaPrediction) -> list[TraitScoreDTO]:
        """Build detailed trait scores with confidence intervals."""
        scores = []
        confidence_margin = 0.15 * (1 - prediction.confidence)
        for trait, prob in prediction.trait_probabilities.items():
            lower = max(0.0, prob - confidence_margin)
            upper = min(1.0, prob + confidence_margin)
            scores.append(TraitScoreDTO(
                trait=trait,
                value=prob,
                confidence_lower=lower,
                confidence_upper=upper,
                sample_count=1,
                evidence_markers=[f"roberta_{trait.value}"],
            ))
        return scores

    def _neutral_scores(self, confidence: float = 0.3) -> OceanScoresDTO:
        """Return neutral scores when detection fails."""
        return OceanScoresDTO(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            overall_confidence=confidence,
            trait_scores=[],
        )

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()
        logger.info("roberta_embedding_cache_cleared")

    async def shutdown(self) -> None:
        """Shutdown the detector."""
        self._embedding_cache.clear()
        self._initialized = False
        logger.info("roberta_detector_shutdown")

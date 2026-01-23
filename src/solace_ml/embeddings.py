"""Solace-AI Embeddings - Text embedding service for semantic search and RAG."""

from __future__ import annotations
import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any
import httpx
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    VOYAGE = "voyage"
    LOCAL = "local"


class EmbeddingModel(str, Enum):
    """Common embedding models."""

    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"
    OPENAI_ADA = "text-embedding-ada-002"
    VOYAGE_LARGE = "voyage-large-2"
    VOYAGE_CODE = "voyage-code-2"


class EmbeddingSettings(BaseSettings):
    """Embedding service settings."""

    provider: EmbeddingProvider = Field(default=EmbeddingProvider.OPENAI)
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default=EmbeddingModel.OPENAI_SMALL.value)
    dimensions: int | None = Field(
        default=None, description="Output dimensions if supported"
    )
    batch_size: int = Field(default=100, ge=1, le=2048)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600)
    normalize: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_", env_file=".env", extra="ignore"
    )


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""

    embeddings: list[list[float]]
    model: str
    provider: EmbeddingProvider
    dimensions: int
    tokens_used: int = 0
    latency_ms: float = 0.0
    cached: bool = False

    @property
    def count(self) -> int:
        return len(self.embeddings)


class EmbeddingError(Exception):
    """Embedding service error."""

    def __init__(
        self, message: str, *, provider: EmbeddingProvider, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class EmbeddingCache:
    """LRU embedding cache with TTL support.

    Uses OrderedDict for O(1) LRU eviction instead of O(n) min() lookup.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000) -> None:
        from collections import OrderedDict

        self._cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding with LRU update."""
        key = self._make_key(text, model)
        if key in self._cache:
            embedding, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # Move to end for LRU (most recently used)
                self._cache.move_to_end(key)
                return embedding
            del self._cache[key]
        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Cache embedding with O(1) LRU eviction."""
        key = self._make_key(text, model)
        # If key exists, remove it first to update position
        if key in self._cache:
            del self._cache[key]
        # Evict oldest (first) item if at capacity - O(1) operation
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove oldest (first inserted)
        # Add new item at the end (most recently used)
        self._cache[key] = (embedding, time.time())

    def get_batch(
        self, texts: list[str], model: str
    ) -> tuple[list[list[float] | None], list[int]]:
        """Get cached embeddings for batch, return list of embeddings and indices of misses."""
        results: list[list[float] | None] = []
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self.get(text, model)
            results.append(cached)
            if cached is None:
                misses.append(i)
        return results, misses

    def set_batch(
        self,
        texts: list[str],
        model: str,
        embeddings: list[list[float]],
        indices: list[int],
    ) -> None:
        """Cache batch of embeddings at specific indices."""
        for idx, embedding in zip(indices, embeddings):
            self.set(texts[idx], model, embedding)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


class EmbeddingClient(ABC):
    """Abstract embedding client."""

    def __init__(
        self, settings: EmbeddingSettings, provider: EmbeddingProvider
    ) -> None:
        self._settings = settings
        self._provider = provider

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._settings.model

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Generate embeddings for texts."""

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embedding client."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        settings = settings or EmbeddingSettings()
        super().__init__(settings, EmbeddingProvider.OPENAI)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds))
        self._cache = (
            EmbeddingCache(settings.cache_ttl_seconds)
            if settings.cache_enabled
            else None
        )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Generate embeddings for texts."""
        start_time = time.perf_counter()
        texts_list = list(texts)
        if not texts_list:
            return EmbeddingResult(
                embeddings=[],
                model=self._settings.model,
                provider=self._provider,
                dimensions=0,
            )
        cache_results: list[list[float] | None] = [None] * len(texts_list)
        texts_to_embed = texts_list
        indices_to_embed = list(range(len(texts_list)))
        cached_count = 0
        if self._cache:
            cache_results, indices_to_embed = self._cache.get_batch(
                texts_list, self._settings.model
            )
            texts_to_embed = [texts_list[i] for i in indices_to_embed]
            cached_count = len(texts_list) - len(indices_to_embed)
        embeddings: list[list[float]] = []
        total_tokens = 0
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), self._settings.batch_size):
                batch = texts_to_embed[i : i + self._settings.batch_size]
                result = await self._embed_batch(batch)
                embeddings.extend(result["embeddings"])
                total_tokens += result["tokens"]
            if self._cache:
                self._cache.set_batch(
                    texts_list, self._settings.model, embeddings, indices_to_embed
                )
        final_embeddings: list[list[float]] = []
        embed_idx = 0
        for i in range(len(texts_list)):
            if cache_results[i] is not None:
                final_embeddings.append(cache_results[i])  # type: ignore
            else:
                final_embeddings.append(embeddings[embed_idx])
                embed_idx += 1
        latency = (time.perf_counter() - start_time) * 1000
        dimensions = len(final_embeddings[0]) if final_embeddings else 0
        return EmbeddingResult(
            embeddings=final_embeddings,
            model=self._settings.model,
            provider=self._provider,
            dimensions=dimensions,
            tokens_used=total_tokens,
            latency_ms=latency,
            cached=cached_count > 0,
        )

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []

    async def _embed_batch(self, texts: list[str]) -> dict[str, Any]:
        """Embed a batch of texts."""
        payload: dict[str, Any] = {"input": texts, "model": self._settings.model}
        if self._settings.dimensions and "text-embedding-3" in self._settings.model:
            payload["dimensions"] = self._settings.dimensions
        headers = {
            "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = await self._http.post(
                    "https://api.openai.com/v1/embeddings",
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 429:
                    if attempt < self._settings.max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise EmbeddingError(
                        "Rate limit exceeded", provider=self._provider, retryable=True
                    )
                if response.status_code >= 500:
                    if attempt < self._settings.max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise EmbeddingError(
                        f"Server error: {response.status_code}",
                        provider=self._provider,
                        retryable=True,
                    )
                if response.status_code != 200:
                    raise EmbeddingError(
                        f"API error: {response.text}", provider=self._provider
                    )
                data = response.json()
                embeddings = [
                    item["embedding"]
                    for item in sorted(data["data"], key=lambda x: x["index"])
                ]
                if self._settings.normalize:
                    embeddings = [self._normalize(e) for e in embeddings]
                return {
                    "embeddings": embeddings,
                    "tokens": data.get("usage", {}).get("total_tokens", 0),
                }
            except httpx.RequestError as e:
                if attempt < self._settings.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue
                raise EmbeddingError(
                    f"Request error: {e}", provider=self._provider, retryable=True
                ) from e
        raise EmbeddingError("Max retries exceeded", provider=self._provider)

    async def health_check(self) -> bool:
        """Check OpenAI embedding API health."""
        try:
            result = await self.embed(["health check"])
            return len(result.embeddings) > 0
        except Exception as e:
            logger.warning(
                "embedding_health_check_failed", provider="openai", error=str(e)
            )
            return False

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    @staticmethod
    def _normalize(embedding: list[float]) -> list[float]:
        """Normalize embedding vector to unit length."""
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]


class EmbeddingService:
    """High-level embedding service with caching and batching."""

    def __init__(
        self,
        client: EmbeddingClient | None = None,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        self._settings = settings or EmbeddingSettings()
        self._client = client or OpenAIEmbeddingClient(self._settings)

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed multiple texts."""
        return await self._client.embed(texts)

    async def embed_text(self, text: str) -> list[float]:
        """Embed single text."""
        return await self._client.embed_single(text)

    async def embed_documents(
        self, documents: Sequence[dict[str, Any]], content_key: str = "content"
    ) -> list[dict[str, Any]]:
        """Embed documents and return with embeddings attached."""
        texts = [doc.get(content_key, "") for doc in documents]
        result = await self._client.embed(texts)
        enriched = []
        for doc, embedding in zip(documents, result.embeddings):
            enriched_doc = dict(doc)
            enriched_doc["embedding"] = embedding
            enriched_doc["embedding_model"] = result.model
            enriched.append(enriched_doc)
        return enriched

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        result = await self._client.embed([text1, text2])
        if len(result.embeddings) < 2:
            return 0.0
        return self._cosine_similarity(result.embeddings[0], result.embeddings[1])

    async def find_most_similar(
        self, query: str, candidates: Sequence[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Find most similar candidates to query."""
        all_texts = [query] + list(candidates)
        result = await self._client.embed(all_texts)
        query_embedding = result.embeddings[0]
        similarities: list[tuple[int, float]] = []
        for i, candidate_embedding in enumerate(result.embeddings[1:]):
            sim = self._cosine_similarity(query_embedding, candidate_embedding)
            similarities.append((i, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def health_check(self) -> bool:
        """Check service health."""
        return await self._client.health_check()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


def create_embedding_client(
    settings: EmbeddingSettings | None = None,
) -> EmbeddingClient:
    """Factory function to create embedding client."""
    settings = settings or EmbeddingSettings()
    if settings.provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingClient(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")


def create_embedding_service(
    settings: EmbeddingSettings | None = None,
) -> EmbeddingService:
    """Factory function to create embedding service."""
    return EmbeddingService(settings=settings)

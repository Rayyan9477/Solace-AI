"""Unit tests for embeddings module."""
from __future__ import annotations
import pytest
import time
from solace_ml.embeddings import (
    EmbeddingProvider, EmbeddingModel, EmbeddingSettings, EmbeddingResult,
    EmbeddingError, EmbeddingCache, EmbeddingClient, OpenAIEmbeddingClient,
    EmbeddingService, create_embedding_client, create_embedding_service,
)


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider enum."""

    def test_provider_values(self):
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.VOYAGE.value == "voyage"
        assert EmbeddingProvider.LOCAL.value == "local"


class TestEmbeddingModel:
    """Tests for EmbeddingModel enum."""

    def test_model_values(self):
        assert EmbeddingModel.OPENAI_SMALL.value == "text-embedding-3-small"
        assert EmbeddingModel.OPENAI_LARGE.value == "text-embedding-3-large"
        assert EmbeddingModel.VOYAGE_LARGE.value == "voyage-large-2"


class TestEmbeddingSettings:
    """Tests for EmbeddingSettings model."""

    def test_default_settings(self):
        settings = EmbeddingSettings()
        assert settings.provider == EmbeddingProvider.OPENAI
        assert settings.model == EmbeddingModel.OPENAI_SMALL.value
        assert settings.batch_size == 100

    def test_custom_settings(self):
        settings = EmbeddingSettings(
            model=EmbeddingModel.OPENAI_LARGE.value,
            dimensions=1024,
            batch_size=50
        )
        assert settings.model == "text-embedding-3-large"
        assert settings.dimensions == 1024
        assert settings.batch_size == 50

    def test_cache_settings(self):
        settings = EmbeddingSettings(cache_enabled=False, cache_ttl_seconds=7200)
        assert settings.cache_enabled is False
        assert settings.cache_ttl_seconds == 7200


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_create_result(self):
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings,
            model="text-embedding-3-small",
            provider=EmbeddingProvider.OPENAI,
            dimensions=3
        )
        assert len(result.embeddings) == 2
        assert result.dimensions == 3
        assert result.count == 2

    def test_result_with_usage(self):
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="text-embedding-3-small",
            provider=EmbeddingProvider.OPENAI,
            dimensions=2,
            tokens_used=100,
            latency_ms=50.0,
            cached=True
        )
        assert result.tokens_used == 100
        assert result.latency_ms == 50.0
        assert result.cached is True


class TestEmbeddingError:
    """Tests for EmbeddingError exception."""

    def test_create_error(self):
        error = EmbeddingError("Test error", provider=EmbeddingProvider.OPENAI)
        assert str(error) == "Test error"
        assert error.provider == EmbeddingProvider.OPENAI
        assert error.retryable is False

    def test_retryable_error(self):
        error = EmbeddingError("Rate limited", provider=EmbeddingProvider.OPENAI, retryable=True)
        assert error.retryable is True


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @pytest.fixture
    def cache(self):
        return EmbeddingCache(ttl_seconds=3600)

    def test_set_and_get(self, cache):
        embedding = [0.1, 0.2, 0.3]
        cache.set("test text", "model1", embedding)
        result = cache.get("test text", "model1")
        assert result == embedding

    def test_get_nonexistent(self, cache):
        result = cache.get("nonexistent", "model1")
        assert result is None

    def test_different_models(self, cache):
        cache.set("test", "model1", [0.1, 0.2])
        cache.set("test", "model2", [0.3, 0.4])
        assert cache.get("test", "model1") == [0.1, 0.2]
        assert cache.get("test", "model2") == [0.3, 0.4]

    def test_cache_expiry(self):
        cache = EmbeddingCache(ttl_seconds=0)
        cache.set("test", "model1", [0.1])
        time.sleep(0.01)
        result = cache.get("test", "model1")
        assert result is None

    def test_get_batch(self, cache):
        cache.set("text1", "model1", [0.1, 0.2])
        cache.set("text3", "model1", [0.5, 0.6])
        results, misses = cache.get_batch(["text1", "text2", "text3"], "model1")
        assert results[0] == [0.1, 0.2]
        assert results[1] is None
        assert results[2] == [0.5, 0.6]
        assert misses == [1]

    def test_set_batch(self, cache):
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1], [0.3]]
        indices = [0, 2]
        cache.set_batch(texts, "model1", embeddings, indices)
        assert cache.get("text1", "model1") == [0.1]
        assert cache.get("text2", "model1") is None
        assert cache.get("text3", "model1") == [0.3]

    def test_clear(self, cache):
        cache.set("text1", "model1", [0.1])
        cache.clear()
        assert cache.get("text1", "model1") is None

    def test_max_size(self):
        cache = EmbeddingCache(max_size=2)
        cache.set("text1", "model1", [0.1])
        cache.set("text2", "model1", [0.2])
        cache.set("text3", "model1", [0.3])
        assert cache.get("text3", "model1") == [0.3]


class TestOpenAIEmbeddingClient:
    """Tests for OpenAIEmbeddingClient class."""

    @pytest.fixture
    def client(self):
        return OpenAIEmbeddingClient()

    def test_client_creation(self, client):
        assert client.provider == EmbeddingProvider.OPENAI
        assert client.model == "text-embedding-3-small"

    def test_client_with_settings(self):
        settings = EmbeddingSettings(model=EmbeddingModel.OPENAI_LARGE.value)
        client = OpenAIEmbeddingClient(settings)
        assert client.model == "text-embedding-3-large"

    def test_normalize_vector(self):
        embedding = [3.0, 4.0]
        normalized = OpenAIEmbeddingClient._normalize(embedding)
        assert abs(normalized[0] - 0.6) < 0.001
        assert abs(normalized[1] - 0.8) < 0.001

    def test_normalize_zero_vector(self):
        embedding = [0.0, 0.0]
        normalized = OpenAIEmbeddingClient._normalize(embedding)
        assert normalized == [0.0, 0.0]


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def service(self):
        return EmbeddingService()

    def test_service_creation(self, service):
        assert service._client is not None

    def test_cosine_similarity(self, service):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity) < 0.001

    def test_cosine_similarity_same_vector(self, service):
        a = [1.0, 2.0, 3.0]
        similarity = service._cosine_similarity(a, a)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_opposite(self, service):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        similarity = service._cosine_similarity(a, b)
        assert abs(similarity - (-1.0)) < 0.001

    def test_cosine_similarity_zero_vector(self, service):
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        similarity = service._cosine_similarity(a, b)
        assert similarity == 0.0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_embedding_client(self):
        client = create_embedding_client()
        assert isinstance(client, OpenAIEmbeddingClient)

    def test_create_embedding_client_with_settings(self):
        settings = EmbeddingSettings(model=EmbeddingModel.OPENAI_LARGE.value)
        client = create_embedding_client(settings)
        assert client.model == "text-embedding-3-large"

    def test_create_embedding_service(self):
        service = create_embedding_service()
        assert isinstance(service, EmbeddingService)

    def test_create_embedding_client_unsupported_provider(self):
        settings = EmbeddingSettings(provider=EmbeddingProvider.VOYAGE)
        with pytest.raises(ValueError, match="Unsupported"):
            create_embedding_client(settings)

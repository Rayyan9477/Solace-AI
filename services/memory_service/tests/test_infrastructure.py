"""
Solace-AI Memory Service - Infrastructure Layer Tests.
Tests for PostgreSQL, Weaviate, Redis, Hybrid Search, and RAG Pipeline.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
import math

from services.memory_service.src.infrastructure.postgres_repo import (
    PostgresSettings, PostgresRepository, memory_records, session_summaries,
)
from services.memory_service.src.infrastructure.weaviate_repo import (
    WeaviateSettings, WeaviateRepository, VectorRecord, SearchResult, CollectionName,
)
from services.memory_service.src.infrastructure.redis_cache import (
    RedisSettings, RedisCache, CachedWorkingMemory, CachedSessionState,
)
from services.memory_service.src.infrastructure.hybrid_search import (
    HybridSearchSettings, HybridSearchEngine, BM25Index, SemanticSearcher,
    SearchDocument, SearchResult as HybridSearchResult,
)
from services.memory_service.src.infrastructure.rag_pipeline import (
    RAGSettings, RAGPipeline, RAGPipelineFactory, SimpleGrader, QueryRephraser,
    RetrievedDocument, RAGContext, RetrievalStatus, DocumentGrade,
)


class TestPostgresSettings:
    """Tests for PostgreSQL settings."""

    def test_default_settings(self) -> None:
        settings = PostgresSettings()
        assert settings.host == "localhost"
        assert settings.port == 5432
        assert settings.database == "solace_memory"
        assert settings.pool_size == 10

    def test_connection_url(self) -> None:
        settings = PostgresSettings(user="test", password="pass", database="testdb")
        assert "postgresql+asyncpg://" in settings.connection_url
        assert "test:pass" in settings.connection_url
        assert "testdb" in settings.connection_url


class TestPostgresRepository:
    """Tests for PostgreSQL repository."""

    def test_repository_initialization(self) -> None:
        settings = PostgresSettings(pool_size=0)
        repo = PostgresRepository(settings)
        assert repo._settings.database == "solace_memory"
        assert repo._stats["inserts"] == 0

    def test_statistics_tracking(self) -> None:
        repo = PostgresRepository()
        stats = repo.get_statistics()
        assert "inserts" in stats
        assert "queries" in stats
        assert "database" in stats


class TestWeaviateSettings:
    """Tests for Weaviate settings."""

    def test_default_settings(self) -> None:
        settings = WeaviateSettings()
        assert settings.host == "localhost"
        assert settings.port == 8080
        assert settings.embedding_dimension == 1536

    def test_http_url(self) -> None:
        settings = WeaviateSettings(host="vector.example.com", use_https=True)
        assert settings.http_url == "https://vector.example.com:8080"


class TestWeaviateRepository:
    """Tests for Weaviate repository."""

    def test_repository_initialization(self) -> None:
        repo = WeaviateRepository()
        assert repo._initialized is False
        assert repo._stats["inserts"] == 0

    def test_vector_record_creation(self) -> None:
        record = VectorRecord(
            user_id=uuid4(),
            content="Test content",
            embedding=[0.1] * 1536,
            importance=0.8,
        )
        assert len(record.embedding) == 1536
        assert record.importance == 0.8
        assert record.collection == CollectionName.CONVERSATION_MEMORY.value

    def test_collection_names(self) -> None:
        assert CollectionName.CONVERSATION_MEMORY.value == "ConversationMemory"
        assert CollectionName.SESSION_SUMMARY.value == "SessionSummary"
        assert CollectionName.USER_FACT.value == "UserFact"

    def test_statistics(self) -> None:
        repo = WeaviateRepository()
        stats = repo.get_statistics()
        assert "initialized" in stats
        assert stats["initialized"] is False


class TestRedisSettings:
    """Tests for Redis settings."""

    def test_default_settings(self) -> None:
        settings = RedisSettings()
        assert settings.host == "localhost"
        assert settings.port == 6379
        assert settings.working_memory_ttl == 3600
        assert settings.session_ttl == 86400


class TestRedisCache:
    """Tests for Redis cache."""

    def test_cache_initialization(self) -> None:
        cache = RedisCache()
        assert cache._initialized is False
        assert cache._stats["hits"] == 0

    def test_key_building(self) -> None:
        settings = RedisSettings(key_prefix="test:")
        cache = RedisCache(settings)
        key = cache._key("working", "user123", "session456")
        assert key == "test:working:user123:session456"

    def test_cached_working_memory_serialization(self) -> None:
        user_id = uuid4()
        session_id = uuid4()
        memory = CachedWorkingMemory(
            user_id=user_id,
            session_id=session_id,
            items=[{"content": "test", "role": "user"}],
            total_tokens=100,
            max_tokens=8000,
        )
        data = memory.to_dict()
        assert data["user_id"] == str(user_id)
        assert data["total_tokens"] == 100
        restored = CachedWorkingMemory.from_dict(data)
        assert restored.user_id == user_id
        assert restored.total_tokens == 100

    def test_cached_session_state_serialization(self) -> None:
        session_id = uuid4()
        user_id = uuid4()
        state = CachedSessionState(
            session_id=session_id,
            user_id=user_id,
            session_number=5,
            status="active",
            message_count=10,
            topics_detected=["anxiety", "relationships"],
        )
        data = state.to_dict()
        assert data["session_number"] == 5
        assert data["message_count"] == 10
        assert "anxiety" in data["topics_detected"]
        restored = CachedSessionState.from_dict(data)
        assert restored.session_id == session_id
        assert restored.session_number == 5

    def test_statistics(self) -> None:
        cache = RedisCache()
        stats = cache.get_statistics()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestBM25Index:
    """Tests for BM25 keyword search index."""

    def test_document_indexing(self) -> None:
        index = BM25Index()
        doc = SearchDocument(doc_id=uuid4(), content="The quick brown fox jumps over the lazy dog")
        index.add_document(doc)
        assert index._total_docs == 1
        assert "quick" in index._inverted_index

    def test_basic_search(self) -> None:
        index = BM25Index()
        doc1 = SearchDocument(doc_id=uuid4(), content="anxiety and depression symptoms")
        doc2 = SearchDocument(doc_id=uuid4(), content="happy feelings and joy")
        index.add_document(doc1)
        index.add_document(doc2)
        results = index.search("anxiety symptoms", limit=10)
        assert len(results) > 0
        assert results[0][0] == doc1.doc_id

    def test_document_removal(self) -> None:
        index = BM25Index()
        doc = SearchDocument(doc_id=uuid4(), content="test document content")
        index.add_document(doc)
        assert index._total_docs == 1
        removed = index.remove_document(doc.doc_id)
        assert removed is True
        assert index._total_docs == 0

    def test_empty_query(self) -> None:
        index = BM25Index()
        doc = SearchDocument(doc_id=uuid4(), content="some content")
        index.add_document(doc)
        results = index.search("", limit=10)
        assert len(results) == 0

    def test_clear_index(self) -> None:
        index = BM25Index()
        for i in range(5):
            doc = SearchDocument(doc_id=uuid4(), content=f"document {i}")
            index.add_document(doc)
        assert index._total_docs == 5
        index.clear()
        assert index._total_docs == 0


class TestSemanticSearcher:
    """Tests for semantic vector search."""

    def test_document_addition(self) -> None:
        searcher = SemanticSearcher(dimension=4)
        doc = SearchDocument(
            doc_id=uuid4(),
            content="test content",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        searcher.add_document(doc)
        assert doc.doc_id in searcher._documents

    def test_cosine_similarity_search(self) -> None:
        searcher = SemanticSearcher(dimension=4)
        doc1 = SearchDocument(doc_id=uuid4(), content="doc1", embedding=[1.0, 0.0, 0.0, 0.0])
        doc2 = SearchDocument(doc_id=uuid4(), content="doc2", embedding=[0.0, 1.0, 0.0, 0.0])
        searcher.add_document(doc1)
        searcher.add_document(doc2)
        query = [1.0, 0.0, 0.0, 0.0]
        results = searcher.search(query, limit=10)
        assert len(results) == 2
        assert results[0][0] == doc1.doc_id
        assert results[0][1] == pytest.approx(1.0, rel=0.01)

    def test_wrong_dimension_query(self) -> None:
        searcher = SemanticSearcher(dimension=4)
        doc = SearchDocument(doc_id=uuid4(), content="test", embedding=[0.1, 0.2, 0.3, 0.4])
        searcher.add_document(doc)
        results = searcher.search([1.0, 0.0], limit=10)
        assert len(results) == 0

    def test_minimum_similarity_threshold(self) -> None:
        searcher = SemanticSearcher(dimension=4)
        doc = SearchDocument(doc_id=uuid4(), content="test", embedding=[1.0, 0.0, 0.0, 0.0])
        searcher.add_document(doc)
        query = [0.0, 1.0, 0.0, 0.0]
        results = searcher.search(query, limit=10, min_similarity=0.5)
        assert len(results) == 0


class TestHybridSearchEngine:
    """Tests for hybrid search engine."""

    def test_engine_initialization(self) -> None:
        engine = HybridSearchEngine()
        assert engine._settings.alpha == 0.5
        assert engine._settings.fusion_method == "rrf"

    def test_document_indexing(self) -> None:
        engine = HybridSearchEngine(HybridSearchSettings(embedding_dimension=4))
        doc_id = uuid4()
        engine.index_document(
            doc_id=doc_id,
            content="therapeutic anxiety treatment",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"source": "session"},
        )
        assert engine.get_document_count() == 1

    def test_bm25_only_search(self) -> None:
        engine = HybridSearchEngine()
        engine.index_document(uuid4(), "anxiety disorder treatment options")
        engine.index_document(uuid4(), "happy thoughts and meditation")
        results = engine.search_bm25_only("anxiety treatment", limit=5)
        assert len(results) > 0
        assert results[0].bm25_score > 0

    def test_hybrid_search(self) -> None:
        engine = HybridSearchEngine(HybridSearchSettings(embedding_dimension=4))
        doc_id = uuid4()
        engine.index_document(doc_id, "anxiety treatment therapy", embedding=[1.0, 0.0, 0.0, 0.0])
        results = engine.search("anxiety", query_embedding=[1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) > 0

    def test_reciprocal_rank_fusion(self) -> None:
        settings = HybridSearchSettings(fusion_method="rrf", embedding_dimension=4)
        engine = HybridSearchEngine(settings)
        engine.index_document(uuid4(), "test document one", embedding=[1.0, 0.0, 0.0, 0.0])
        engine.index_document(uuid4(), "test document two", embedding=[0.0, 1.0, 0.0, 0.0])
        results = engine.search("document", query_embedding=[0.5, 0.5, 0.0, 0.0], limit=5)
        assert len(results) == 2

    def test_weighted_fusion(self) -> None:
        settings = HybridSearchSettings(fusion_method="weighted", embedding_dimension=4)
        engine = HybridSearchEngine(settings)
        engine.index_document(uuid4(), "test content", embedding=[1.0, 0.0, 0.0, 0.0])
        results = engine.search("test", query_embedding=[1.0, 0.0, 0.0, 0.0], alpha=0.7, limit=5)
        assert len(results) > 0

    def test_document_removal(self) -> None:
        engine = HybridSearchEngine()
        doc_id = uuid4()
        engine.index_document(doc_id, "content to remove")
        assert engine.get_document_count() == 1
        engine.remove_document(doc_id)
        assert engine.get_document_count() == 0

    def test_statistics(self) -> None:
        engine = HybridSearchEngine()
        engine.index_document(uuid4(), "test document")
        engine.search_bm25_only("test")
        stats = engine.get_statistics()
        assert stats["docs_indexed"] == 1
        assert stats["searches"] == 1


class TestSimpleGrader:
    """Tests for simple document grader."""

    @pytest.mark.asyncio
    async def test_relevant_grading(self) -> None:
        grader = SimpleGrader(threshold=0.5)
        grade, score = await grader.grade("anxiety symptoms treatment", "anxiety symptoms and treatment options available")
        assert grade in [DocumentGrade.RELEVANT, DocumentGrade.PARTIALLY_RELEVANT]
        assert score > 0

    @pytest.mark.asyncio
    async def test_not_relevant_grading(self) -> None:
        grader = SimpleGrader(threshold=0.5)
        grade, score = await grader.grade("anxiety symptoms", "happy birthday celebration party")
        assert grade == DocumentGrade.NOT_RELEVANT
        assert score < 0.5

    @pytest.mark.asyncio
    async def test_therapeutic_keyword_boost(self) -> None:
        grader = SimpleGrader(threshold=0.3)
        grade1, score1 = await grader.grade("anxiety therapy", "therapy for anxiety disorders help support")
        grade2, score2 = await grader.grade("random word", "completely unrelated topic discussion")
        assert grade1 in [DocumentGrade.RELEVANT, DocumentGrade.PARTIALLY_RELEVANT]
        assert grade2 == DocumentGrade.NOT_RELEVANT

    @pytest.mark.asyncio
    async def test_empty_query(self) -> None:
        grader = SimpleGrader()
        grade, score = await grader.grade("", "some document content")
        assert grade == DocumentGrade.NOT_RELEVANT
        assert score == 0.0


class TestQueryRephraser:
    """Tests for query rephrasing."""

    def test_first_attempt_unchanged(self) -> None:
        rephraser = QueryRephraser()
        result = rephraser.rephrase("anxiety help", attempt=0)
        assert result == "anxiety help"

    def test_subsequent_attempts_modified(self) -> None:
        rephraser = QueryRephraser()
        result = rephraser.rephrase("anxiety help", attempt=1)
        assert result != "anxiety help"
        assert len(result) > len("anxiety help")

    def test_therapeutic_expansion(self) -> None:
        rephraser = QueryRephraser()
        expanded = rephraser.expand_therapeutic_terms("feeling sad today")
        assert "depression" in expanded or "low mood" in expanded


class TestRetrievedDocument:
    """Tests for retrieved document dataclass."""

    def test_token_count_estimation(self) -> None:
        doc = RetrievedDocument(
            doc_id=uuid4(),
            content="This is a test document with some content.",
        )
        assert doc.token_count > 0
        assert doc.token_count == len(doc.content) // 4

    def test_default_values(self) -> None:
        doc = RetrievedDocument(doc_id=uuid4(), content="test")
        assert doc.grade == DocumentGrade.NOT_RELEVANT
        assert doc.source == "unknown"
        assert doc.score == 0.0


class TestRAGContext:
    """Tests for RAG context."""

    def test_default_values(self) -> None:
        context = RAGContext()
        assert context.status == RetrievalStatus.SUCCESS
        assert len(context.documents) == 0
        assert context.retrieval_attempts == 1

    def test_context_with_documents(self) -> None:
        docs = [
            RetrievedDocument(doc_id=uuid4(), content="doc1"),
            RetrievedDocument(doc_id=uuid4(), content="doc2"),
        ]
        context = RAGContext(query="test", documents=docs)
        assert len(context.documents) == 2


class TestRAGPipeline:
    """Tests for RAG pipeline."""

    def test_pipeline_initialization(self) -> None:
        pipeline = RAGPipeline()
        assert pipeline._settings.max_retrieval_attempts == 3
        assert pipeline._settings.enable_grading is True

    def test_context_assembly(self) -> None:
        pipeline = RAGPipeline()
        docs = [
            RetrievedDocument(doc_id=uuid4(), content="First document content", source="session"),
            RetrievedDocument(doc_id=uuid4(), content="Second document content", source="fact"),
        ]
        context_str = pipeline._assemble_context(docs)
        assert "First document content" in context_str
        assert "Second document content" in context_str
        assert "[session]" in context_str

    def test_empty_context_assembly(self) -> None:
        pipeline = RAGPipeline()
        context_str = pipeline._assemble_context([])
        assert context_str == ""

    def test_token_estimation(self) -> None:
        pipeline = RAGPipeline()
        text = "This is a sample text for token estimation."
        tokens = pipeline.estimate_token_count(text)
        assert tokens == len(text) // 4

    def test_retrieval_summary(self) -> None:
        pipeline = RAGPipeline()
        docs = [RetrievedDocument(doc_id=uuid4(), content="test", grade_score=0.8, source="session")]
        context = RAGContext(
            query="test query",
            documents=docs,
            status=RetrievalStatus.SUCCESS,
            retrieval_attempts=1,
            processing_time_ms=50,
        )
        summary = pipeline.get_retrieval_summary(context)
        assert summary["status"] == "success"
        assert summary["documents_retrieved"] == 1
        assert "session" in summary["sources"]

    def test_statistics(self) -> None:
        pipeline = RAGPipeline()
        stats = pipeline.get_statistics()
        assert "retrievals" in stats
        assert "success_rate" in stats
        assert stats["grading_enabled"] is True


class TestRAGPipelineFactory:
    """Tests for RAG pipeline factory."""

    def test_therapeutic_pipeline(self) -> None:
        pipeline = RAGPipelineFactory.create_therapeutic_pipeline()
        assert pipeline._settings.enable_grading is True
        assert pipeline._settings.enable_rephrasing is True
        assert pipeline._settings.rerank_top_k == 7

    def test_fast_pipeline(self) -> None:
        pipeline = RAGPipelineFactory.create_fast_pipeline()
        assert pipeline._settings.enable_grading is False
        assert pipeline._settings.enable_rephrasing is False
        assert pipeline._settings.max_retrieval_attempts == 1

    def test_safety_pipeline(self) -> None:
        pipeline = RAGPipelineFactory.create_safety_pipeline()
        assert pipeline._settings.max_documents == 20
        assert pipeline._settings.context_token_limit == 8000


class TestSearchDocument:
    """Tests for search document."""

    def test_automatic_tokenization(self) -> None:
        doc = SearchDocument(doc_id=uuid4(), content="Hello world, this is a test!")
        assert "hello" in doc.term_frequencies
        assert "world" in doc.term_frequencies
        assert doc.doc_length > 0

    def test_special_characters_removed(self) -> None:
        doc = SearchDocument(doc_id=uuid4(), content="Hello! World? Test...")
        assert "hello" in doc.term_frequencies
        assert "!" not in "".join(doc.term_frequencies.keys())

    def test_short_tokens_filtered(self) -> None:
        doc = SearchDocument(doc_id=uuid4(), content="I am a test")
        assert "i" not in doc.term_frequencies
        assert "a" not in doc.term_frequencies
        assert "am" in doc.term_frequencies

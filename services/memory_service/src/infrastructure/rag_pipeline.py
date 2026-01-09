"""
Solace-AI Memory Service - Agentic Corrective RAG Pipeline.
Self-correcting retrieval with document grading and query rephrasing.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class RAGSettings(BaseSettings):
    """RAG pipeline configuration."""
    max_retrieval_attempts: int = Field(default=3)
    min_relevance_score: float = Field(default=0.6)
    max_documents: int = Field(default=10)
    enable_grading: bool = Field(default=True)
    enable_rephrasing: bool = Field(default=True)
    grade_threshold: float = Field(default=0.5)
    rerank_top_k: int = Field(default=5)
    context_token_limit: int = Field(default=4000)
    embedding_dimension: int = Field(default=1536)
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


class RetrievalStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    REPHRASED = "rephrased"


class DocumentGrade(str, Enum):
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    NOT_RELEVANT = "not_relevant"


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    doc_id: UUID
    content: str
    score: float = 0.0
    grade: DocumentGrade = DocumentGrade.NOT_RELEVANT
    grade_score: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = len(self.content) // 4


@dataclass
class RAGContext:
    """Assembled RAG context for LLM."""
    context_id: UUID = field(default_factory=uuid4)
    query: str = ""
    documents: list[RetrievedDocument] = field(default_factory=list)
    assembled_context: str = ""
    total_tokens: int = 0
    retrieval_attempts: int = 1
    status: RetrievalStatus = RetrievalStatus.SUCCESS
    rephrased_queries: list[str] = field(default_factory=list)
    processing_time_ms: int = 0


class RetrieverProtocol(Protocol):
    async def retrieve(self, query: str, query_embedding: list[float] | None, user_id: UUID, limit: int) -> list[dict[str, Any]]: ...


class EmbedderProtocol(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class GraderProtocol(Protocol):
    async def grade(self, query: str, document: str) -> tuple[DocumentGrade, float]: ...


class SimpleGrader:
    """Simple keyword-based document grader."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    async def grade(self, query: str, document: str) -> tuple[DocumentGrade, float]:
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        if not query_terms:
            return DocumentGrade.NOT_RELEVANT, 0.0
        overlap = len(query_terms & doc_terms)
        score = overlap / len(query_terms)
        therapeutic = {"therapy", "treatment", "anxiety", "depression", "feeling", "emotion", "support", "help", "cope", "crisis", "safety"}
        if (query_terms & therapeutic) and (doc_terms & therapeutic):
            score = min(1.0, score + 0.2)
        if score >= 0.7:
            return DocumentGrade.RELEVANT, score
        elif score >= self.threshold:
            return DocumentGrade.PARTIALLY_RELEVANT, score
        return DocumentGrade.NOT_RELEVANT, score


class QueryRephraser:
    """Query rephrasing for improved retrieval."""

    def __init__(self) -> None:
        self._templates = ["Find information about: {query}", "What do we know about: {query}",
                           "Previous discussions regarding: {query}", "Context related to: {query}"]
        self._expansions = {"sad": ["depression", "low mood", "feeling down"], "anxious": ["anxiety", "worried", "nervous"],
                            "angry": ["frustration", "irritation", "upset"], "stressed": ["overwhelmed", "pressure", "tension"],
                            "tired": ["fatigue", "exhausted", "low energy"]}

    def rephrase(self, query: str, attempt: int) -> str:
        if attempt == 0:
            return query
        query_lower = query.lower()
        for keyword, expansions in self._expansions.items():
            if keyword in query_lower:
                query = f"{query} {expansions[attempt % len(expansions)]}"
                break
        return self._templates[attempt].format(query=query) if attempt < len(self._templates) else query

    def expand_therapeutic_terms(self, query: str) -> str:
        expanded = query
        for term, expansions in self._expansions.items():
            if term in query.lower():
                expanded = f"{expanded} {' '.join(expansions[:2])}"
        return expanded


class RAGPipeline:
    """Agentic Corrective RAG pipeline with self-healing retrieval."""

    def __init__(self, settings: RAGSettings | None = None, retriever: RetrieverProtocol | None = None,
                 embedder: EmbedderProtocol | None = None, grader: GraderProtocol | None = None) -> None:
        self._settings = settings or RAGSettings()
        self._retriever = retriever
        self._embedder = embedder
        self._grader = grader or SimpleGrader(self._settings.grade_threshold)
        self._rephraser = QueryRephraser()
        self._stats = {"retrievals": 0, "rephrasals": 0, "docs_graded": 0, "successes": 0, "failures": 0}

    def set_retriever(self, retriever: RetrieverProtocol) -> None:
        self._retriever = retriever

    def set_embedder(self, embedder: EmbedderProtocol) -> None:
        self._embedder = embedder

    def set_grader(self, grader: GraderProtocol) -> None:
        self._grader = grader

    async def retrieve_and_grade(self, query: str, user_id: UUID, query_embedding: list[float] | None = None) -> RAGContext:
        """Execute the full RAG pipeline with corrective retrieval."""
        start = time.perf_counter()
        self._stats["retrievals"] += 1
        context = RAGContext(query=query)
        if query_embedding is None and self._embedder:
            query_embedding = await self._embedder.embed(query)
        current_query = query
        for attempt in range(self._settings.max_retrieval_attempts):
            context.retrieval_attempts = attempt + 1
            documents = await self._retrieve_documents(current_query, query_embedding, user_id)
            if not documents:
                if attempt < self._settings.max_retrieval_attempts - 1:
                    current_query = self._rephraser.rephrase(query, attempt + 1)
                    context.rephrased_queries.append(current_query)
                    self._stats["rephrasals"] += 1
                    continue
                break
            graded_docs = await self._grade_documents(query, documents)
            relevant_docs = [d for d in graded_docs if d.grade != DocumentGrade.NOT_RELEVANT]
            if relevant_docs:
                context.documents = self._rerank_and_select(relevant_docs)
                context.status = RetrievalStatus.SUCCESS
                self._stats["successes"] += 1
                break
            if attempt < self._settings.max_retrieval_attempts - 1 and self._settings.enable_rephrasing:
                current_query = self._rephraser.rephrase(query, attempt + 1)
                context.rephrased_queries.append(current_query)
                self._stats["rephrasals"] += 1
                context.status = RetrievalStatus.REPHRASED
            else:
                context.documents = graded_docs[:self._settings.rerank_top_k]
                context.status = RetrievalStatus.PARTIAL
        if not context.documents:
            context.status = RetrievalStatus.FAILED
            self._stats["failures"] += 1
        context.assembled_context = self._assemble_context(context.documents)
        context.total_tokens = sum(d.token_count for d in context.documents)
        context.processing_time_ms = int((time.perf_counter() - start) * 1000)
        logger.debug("rag_completed", docs=len(context.documents), status=context.status.value, attempts=context.retrieval_attempts)
        return context

    async def _retrieve_documents(self, query: str, query_embedding: list[float] | None, user_id: UUID) -> list[RetrievedDocument]:
        if not self._retriever:
            logger.warning("no_retriever_configured")
            return []
        try:
            results = await self._retriever.retrieve(query=query, query_embedding=query_embedding, user_id=user_id, limit=self._settings.max_documents * 2)
            return [RetrievedDocument(doc_id=r.get("doc_id", uuid4()), content=r.get("content", ""), score=r.get("score", 0.0),
                                      source=r.get("source", "retriever"), metadata=r.get("metadata", {})) for r in results]
        except Exception as e:
            logger.error("retrieval_failed", error=str(e))
            return []

    async def _grade_documents(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not self._settings.enable_grading:
            for doc in documents:
                doc.grade, doc.grade_score = DocumentGrade.RELEVANT, doc.score
            return documents
        for doc in documents:
            self._stats["docs_graded"] += 1
            doc.grade, doc.grade_score = await self._grader.grade(query, doc.content)
        return sorted(documents, key=lambda d: d.grade_score, reverse=True)

    def _rerank_and_select(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        sorted_docs = sorted(documents, key=lambda d: (d.grade == DocumentGrade.RELEVANT, d.grade_score, d.score), reverse=True)
        selected, total_tokens = [], 0
        for doc in sorted_docs:
            if len(selected) >= self._settings.rerank_top_k:
                break
            if total_tokens + doc.token_count <= self._settings.context_token_limit:
                selected.append(doc)
                total_tokens += doc.token_count
        return selected

    def _assemble_context(self, documents: list[RetrievedDocument]) -> str:
        if not documents:
            return ""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = f"[{doc.source}]" if doc.source != "unknown" else ""
            parts.append(f"--- Retrieved Context {i} {source} ---\n{doc.content}")
        return "\n\n".join(parts)

    async def simple_retrieve(self, query: str, user_id: UUID, limit: int = 5) -> list[RetrievedDocument]:
        if not self._retriever:
            return []
        embedding = await self._embedder.embed(query) if self._embedder else None
        return await self._retrieve_documents(query, embedding, user_id)

    async def retrieve_with_context(self, query: str, user_id: UUID, existing_context: str | None = None,
                                     priority_sources: list[str] | None = None) -> RAGContext:
        enhanced = f"{query}\n\nExisting context: {existing_context[:500]}" if existing_context else query
        context = await self.retrieve_and_grade(enhanced, user_id)
        if priority_sources:
            prioritized = [d for d in context.documents if d.source in priority_sources]
            others = [d for d in context.documents if d.source not in priority_sources]
            context.documents = prioritized + others
        return context

    def estimate_token_count(self, text: str) -> int:
        return len(text) // 4

    def get_retrieval_summary(self, context: RAGContext) -> dict[str, Any]:
        return {"context_id": str(context.context_id), "status": context.status.value, "documents_retrieved": len(context.documents),
                "total_tokens": context.total_tokens, "attempts": context.retrieval_attempts, "rephrased": len(context.rephrased_queries) > 0,
                "rephrase_queries": context.rephrased_queries, "processing_time_ms": context.processing_time_ms,
                "sources": list(set(d.source for d in context.documents)),
                "avg_relevance": sum(d.grade_score for d in context.documents) / len(context.documents) if context.documents else 0}

    def get_statistics(self) -> dict[str, Any]:
        total = self._stats["successes"] + self._stats["failures"]
        success_rate = self._stats["successes"] / total if total > 0 else 0.0
        return {**self._stats, "success_rate": round(success_rate, 4), "max_attempts": self._settings.max_retrieval_attempts,
                "grading_enabled": self._settings.enable_grading, "rephrasing_enabled": self._settings.enable_rephrasing}


class RAGPipelineFactory:
    """Factory for creating configured RAG pipelines."""

    @staticmethod
    def create_therapeutic_pipeline(retriever: RetrieverProtocol | None = None, embedder: EmbedderProtocol | None = None) -> RAGPipeline:
        settings = RAGSettings(max_retrieval_attempts=3, min_relevance_score=0.5, max_documents=15, enable_grading=True,
                               enable_rephrasing=True, grade_threshold=0.4, rerank_top_k=7, context_token_limit=6000)
        return RAGPipeline(settings=settings, retriever=retriever, embedder=embedder)

    @staticmethod
    def create_fast_pipeline(retriever: RetrieverProtocol | None = None, embedder: EmbedderProtocol | None = None) -> RAGPipeline:
        settings = RAGSettings(max_retrieval_attempts=1, min_relevance_score=0.3, max_documents=5, enable_grading=False,
                               enable_rephrasing=False, rerank_top_k=5, context_token_limit=2000)
        return RAGPipeline(settings=settings, retriever=retriever, embedder=embedder)

    @staticmethod
    def create_safety_pipeline(retriever: RetrieverProtocol | None = None, embedder: EmbedderProtocol | None = None) -> RAGPipeline:
        settings = RAGSettings(max_retrieval_attempts=3, min_relevance_score=0.4, max_documents=20, enable_grading=True,
                               enable_rephrasing=True, grade_threshold=0.3, rerank_top_k=10, context_token_limit=8000)
        return RAGPipeline(settings=settings, retriever=retriever, embedder=embedder)

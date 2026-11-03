"""
REV-28 reconciliation Prove-It: Corrective RAG minimum-relevant-docs gate.

Claim (docs): Corrective RAG promises a "<3 relevant docs -> structured
error/degraded" gate. Code (rag_pipeline.py) marked FAILED only on ZERO docs,
so 1-2 relevant docs were returned as a plain SUCCESS.

Corrected behavior encoded here:
- fewer than `min_relevant_documents` (default 3) relevant docs retrieved ->
  a structured INSUFFICIENT_CONTEXT result (NOT SUCCESS).
- >= 3 relevant docs -> SUCCESS.
"""
from __future__ import annotations
import pytest
from uuid import uuid4, UUID

from services.memory_service.src.infrastructure.rag_pipeline import (
    RAGSettings, RAGPipeline, RetrievedDocument, RetrievalStatus, DocumentGrade,
)


class _StubRetriever:
    """Returns a fixed list of raw doc dicts on every (re-phrased) attempt."""

    def __init__(self, docs: list[dict]) -> None:
        self._docs = docs

    async def retrieve(self, query, query_embedding, user_id, limit):  # noqa: ANN001
        return [dict(d) for d in self._docs]


class _StubGrader:
    """Grades a document RELEVANT iff its content starts with 'relevant'."""

    async def grade(self, query: str, document: str):
        if document.startswith("relevant"):
            return DocumentGrade.RELEVANT, 0.9
        return DocumentGrade.NOT_RELEVANT, 0.1


def _make_docs(n_relevant: int, n_noise: int = 4) -> list[dict]:
    docs: list[dict] = []
    for i in range(n_relevant):
        docs.append({"doc_id": uuid4(), "content": f"relevant doc {i}", "score": 0.9, "source": "session"})
    for i in range(n_noise):
        docs.append({"doc_id": uuid4(), "content": f"noise doc {i}", "score": 0.2, "source": "session"})
    return docs


def _pipeline(retriever) -> RAGPipeline:
    # Deterministic: no rephrasing loop needed; the gate is what is under test.
    settings = RAGSettings(max_retrieval_attempts=1, enable_grading=True,
                           enable_rephrasing=False, min_relevant_documents=3)
    return RAGPipeline(settings=settings, retriever=retriever, grader=_StubGrader())


class TestREV28MinRelevantDocsGate:
    """The documented <3-relevant-docs gate must degrade, not report SUCCESS."""

    @pytest.mark.asyncio
    async def test_zero_relevant_docs_is_insufficient(self) -> None:
        pipeline = _pipeline(_StubRetriever(_make_docs(n_relevant=0)))
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.INSUFFICIENT_CONTEXT
        assert ctx.status != RetrievalStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_one_relevant_doc_is_insufficient(self) -> None:
        pipeline = _pipeline(_StubRetriever(_make_docs(n_relevant=1)))
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.INSUFFICIENT_CONTEXT
        assert ctx.status != RetrievalStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_two_relevant_docs_is_insufficient(self) -> None:
        pipeline = _pipeline(_StubRetriever(_make_docs(n_relevant=2)))
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.INSUFFICIENT_CONTEXT
        assert ctx.status != RetrievalStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_three_relevant_docs_is_success(self) -> None:
        pipeline = _pipeline(_StubRetriever(_make_docs(n_relevant=3)))
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.SUCCESS
        assert len(ctx.documents) >= 3
        assert all(d.grade != DocumentGrade.NOT_RELEVANT for d in ctx.documents)

    @pytest.mark.asyncio
    async def test_insufficient_result_never_surfaces_irrelevant_docs(self) -> None:
        # 2 relevant + noise: degraded result carries only the relevant subset.
        pipeline = _pipeline(_StubRetriever(_make_docs(n_relevant=2)))
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.INSUFFICIENT_CONTEXT
        assert all(d.grade != DocumentGrade.NOT_RELEVANT for d in ctx.documents)
        assert len(ctx.documents) <= 2

    @pytest.mark.asyncio
    async def test_threshold_is_configurable(self) -> None:
        # Lower the gate to 1: a single relevant doc now succeeds.
        settings = RAGSettings(max_retrieval_attempts=1, enable_grading=True,
                               enable_rephrasing=False, min_relevant_documents=1)
        pipeline = RAGPipeline(settings=settings, retriever=_StubRetriever(_make_docs(n_relevant=1)),
                               grader=_StubGrader())
        ctx = await pipeline.retrieve_and_grade("query", uuid4())
        assert ctx.status == RetrievalStatus.SUCCESS

    def test_default_threshold_is_three(self) -> None:
        assert RAGSettings().min_relevant_documents == 3


class TestREV28InsufficientContextStatus:
    """The structured status value must exist and be distinct from SUCCESS."""

    def test_insufficient_context_status_exists(self) -> None:
        assert hasattr(RetrievalStatus, "INSUFFICIENT_CONTEXT")
        assert RetrievalStatus.INSUFFICIENT_CONTEXT != RetrievalStatus.SUCCESS

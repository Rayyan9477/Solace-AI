"""
Solace-AI Memory Service - Hybrid Search Engine.
BM25 (via bm25s) + semantic vector search with reciprocal rank fusion.
"""
from __future__ import annotations
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class HybridSearchSettings(BaseSettings):
    """Hybrid search configuration."""
    alpha: float = Field(default=0.5, description="Balance: 0=BM25, 1=semantic")
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    min_score_threshold: float = Field(default=0.3, description="Min score threshold")
    max_results: int = Field(default=50, description="Max results to return")
    embedding_dimension: int = Field(default=1536, description="Embedding dimension")
    fusion_method: str = Field(default="rrf", description="rrf|weighted|linear")
    rrf_k: int = Field(default=60, description="RRF constant k")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    model_config = SettingsConfigDict(env_prefix="HYBRID_SEARCH_", env_file=".env", extra="ignore")


@dataclass
class SearchDocument:
    """Document for search indexing."""
    doc_id: UUID
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    doc_length: int = 0

    def __post_init__(self) -> None:
        if not self.tokens and self.content:
            self.tokens = self._tokenize(self.content)
            self.doc_length = len(self.tokens)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]


@dataclass
class SearchResult:
    """Search result with scores."""
    doc_id: UUID
    content: str
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """BM25 keyword search index using bm25s for 500x speedup."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[UUID, SearchDocument] = {}
        self._doc_id_list: list[UUID] = []
        self._retriever: Any = None
        self._index_dirty = True

    def _rebuild_index(self) -> None:
        """Rebuild the bm25s index from stored documents."""
        if not self._index_dirty or not self._documents:
            return
        try:
            import bm25s
            self._doc_id_list = list(self._documents.keys())
            corpus_tokens = [self._documents[doc_id].tokens for doc_id in self._doc_id_list]
            self._retriever = bm25s.BM25()
            self._retriever.index(corpus_tokens)
            self._index_dirty = False
        except ImportError:
            logger.warning("bm25s_not_installed_using_fallback")
            self._retriever = None

    def add_document(self, doc: SearchDocument) -> None:
        """Add a document to the index."""
        self._documents[doc.doc_id] = doc
        self._index_dirty = True

    def remove_document(self, doc_id: UUID) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return False
        del self._documents[doc_id]
        self._index_dirty = True
        return True

    def search(self, query: str, limit: int = 20) -> list[tuple[UUID, float]]:
        """Search the index using BM25 scoring via bm25s."""
        query_tokens = SearchDocument._tokenize(query)
        if not query_tokens or not self._documents:
            return []
        self._rebuild_index()
        if self._retriever is None:
            return self._fallback_search(query_tokens, limit)
        try:
            import bm25s
            query_tokens_arr = bm25s.tokenize([query], stemmer=None, stopwords=None, show_progress=False)
            results, scores = self._retriever.retrieve(query_tokens_arr, k=min(limit, len(self._doc_id_list)))
            output: list[tuple[UUID, float]] = []
            for idx, score in zip(results[0], scores[0]):
                if idx < len(self._doc_id_list) and score > 0:
                    output.append((self._doc_id_list[idx], float(score)))
            return output
        except Exception as e:
            logger.error("bm25s_search_failed", error=str(e))
            return self._fallback_search(query_tokens, limit)

    def _fallback_search(self, query_tokens: list[str], limit: int) -> list[tuple[UUID, float]]:
        """Fallback BM25 implementation if bm25s fails."""
        if not self._documents:
            return []
        total_docs = len(self._documents)
        avg_doc_len = sum(d.doc_length for d in self._documents.values()) / total_docs if total_docs > 0 else 0
        doc_freqs: dict[str, int] = {}
        for doc in self._documents.values():
            for token in set(doc.tokens):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        scores: dict[UUID, float] = {}
        for term in query_tokens:
            df = doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id, doc in self._documents.items():
                tf = doc.tokens.count(term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc.doc_length / avg_doc_len)) if avg_doc_len > 0 else 1
                score = idf * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0) + score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_document(self, doc_id: UUID) -> SearchDocument | None:
        return self._documents.get(doc_id)

    def clear(self) -> None:
        self._documents.clear()
        self._doc_id_list.clear()
        self._retriever = None
        self._index_dirty = True


class SemanticSearcher:
    """Semantic vector search using cosine similarity."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self._documents: dict[UUID, SearchDocument] = {}

    def add_document(self, doc: SearchDocument) -> None:
        if doc.embedding and len(doc.embedding) == self.dimension:
            self._documents[doc.doc_id] = doc

    def remove_document(self, doc_id: UUID) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def search(self, query_embedding: list[float], limit: int = 20, min_similarity: float = 0.0) -> list[tuple[UUID, float]]:
        if len(query_embedding) != self.dimension:
            return []
        scores: list[tuple[UUID, float]] = []
        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        if query_norm == 0:
            return []
        for doc_id, doc in self._documents.items():
            if not doc.embedding:
                continue
            dot_product = sum(a * b for a, b in zip(query_embedding, doc.embedding))
            doc_norm = math.sqrt(sum(x * x for x in doc.embedding))
            if doc_norm == 0:
                continue
            similarity = dot_product / (query_norm * doc_norm)
            if similarity >= min_similarity:
                scores.append((doc_id, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    def clear(self) -> None:
        self._documents.clear()


class HybridSearchEngine:
    """Hybrid search combining BM25 (bm25s) and semantic search."""

    def __init__(self, settings: HybridSearchSettings | None = None,
                 embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        self._settings = settings or HybridSearchSettings()
        self._bm25 = BM25Index(k1=self._settings.bm25_k1, b=self._settings.bm25_b)
        self._semantic = SemanticSearcher(dimension=self._settings.embedding_dimension)
        self._embedding_fn = embedding_fn
        self._stats = {"searches": 0, "docs_indexed": 0, "bm25_hits": 0, "semantic_hits": 0}

    def index_document(self, doc_id: UUID, content: str, embedding: list[float] | None = None,
                       metadata: dict[str, Any] | None = None) -> None:
        if embedding is None and self._embedding_fn:
            embedding = self._embedding_fn(content)
        doc = SearchDocument(doc_id=doc_id, content=content, embedding=embedding or [], metadata=metadata or {})
        self._bm25.add_document(doc)
        if doc.embedding:
            self._semantic.add_document(doc)
        self._stats["docs_indexed"] += 1

    def remove_document(self, doc_id: UUID) -> bool:
        bm25_removed = self._bm25.remove_document(doc_id)
        semantic_removed = self._semantic.remove_document(doc_id)
        if bm25_removed or semantic_removed:
            self._stats["docs_indexed"] = max(0, self._stats["docs_indexed"] - 1)
        return bm25_removed or semantic_removed

    def search(self, query: str, query_embedding: list[float] | None = None,
               limit: int | None = None, alpha: float | None = None) -> list[SearchResult]:
        start = time.perf_counter()
        self._stats["searches"] += 1
        limit = limit or self._settings.max_results
        alpha = alpha if alpha is not None else self._settings.alpha
        if query_embedding is None and self._embedding_fn:
            query_embedding = self._embedding_fn(query)
        bm25_results = self._bm25.search(query, limit=limit * 2)
        semantic_results: list[tuple[UUID, float]] = []
        if query_embedding:
            semantic_results = self._semantic.search(query_embedding, limit=limit * 2)
        if bm25_results:
            self._stats["bm25_hits"] += 1
        if semantic_results:
            self._stats["semantic_hits"] += 1
        if self._settings.fusion_method == "rrf":
            results = self._reciprocal_rank_fusion(bm25_results, semantic_results, limit)
        elif self._settings.fusion_method == "weighted":
            results = self._weighted_fusion(bm25_results, semantic_results, alpha, limit)
        else:
            results = self._linear_fusion(bm25_results, semantic_results, alpha, limit)
        for i, result in enumerate(results):
            result.rank = i + 1
        logger.debug("hybrid_search", results=len(results), time_ms=int((time.perf_counter() - start) * 1000))
        return results

    def _reciprocal_rank_fusion(self, bm25_results: list[tuple[UUID, float]],
                                 semantic_results: list[tuple[UUID, float]], limit: int) -> list[SearchResult]:
        k = self._settings.rrf_k
        scores: dict[UUID, float] = {}
        bm25_scores: dict[UUID, float] = {}
        semantic_scores: dict[UUID, float] = {}
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            bm25_scores[doc_id] = score
        for rank, (doc_id, score) in enumerate(semantic_results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            semantic_scores[doc_id] = score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        results: list[SearchResult] = []
        for doc_id in sorted_ids:
            doc = self._bm25.get_document(doc_id)
            results.append(SearchResult(doc_id=doc_id, content=doc.content if doc else "",
                bm25_score=bm25_scores.get(doc_id, 0), semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=doc.metadata if doc else {}))
        return results

    def _weighted_fusion(self, bm25_results: list[tuple[UUID, float]],
                          semantic_results: list[tuple[UUID, float]], alpha: float, limit: int) -> list[SearchResult]:
        bm25_max = max((s for _, s in bm25_results), default=1.0) or 1.0
        semantic_max = max((s for _, s in semantic_results), default=1.0) or 1.0
        scores: dict[UUID, float] = {}
        bm25_scores: dict[UUID, float] = {}
        semantic_scores: dict[UUID, float] = {}
        for doc_id, score in bm25_results:
            scores[doc_id] = (1 - alpha) * (score / bm25_max)
            bm25_scores[doc_id] = score
        for doc_id, score in semantic_results:
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (score / semantic_max)
            semantic_scores[doc_id] = score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        results: list[SearchResult] = []
        for doc_id in sorted_ids:
            if scores[doc_id] < self._settings.min_score_threshold:
                continue
            doc = self._bm25.get_document(doc_id)
            results.append(SearchResult(doc_id=doc_id, content=doc.content if doc else "",
                bm25_score=bm25_scores.get(doc_id, 0), semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=doc.metadata if doc else {}))
        return results

    def _linear_fusion(self, bm25_results: list[tuple[UUID, float]],
                        semantic_results: list[tuple[UUID, float]], alpha: float, limit: int) -> list[SearchResult]:
        scores: dict[UUID, float] = {}
        bm25_scores: dict[UUID, float] = {}
        semantic_scores: dict[UUID, float] = {}
        for doc_id, score in bm25_results:
            scores[doc_id] = (1 - alpha) * score
            bm25_scores[doc_id] = score
        for doc_id, score in semantic_results:
            scores[doc_id] = scores.get(doc_id, 0) + alpha * score
            semantic_scores[doc_id] = score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        results: list[SearchResult] = []
        for doc_id in sorted_ids:
            doc = self._bm25.get_document(doc_id)
            results.append(SearchResult(doc_id=doc_id, content=doc.content if doc else "",
                bm25_score=bm25_scores.get(doc_id, 0), semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=doc.metadata if doc else {}))
        return results

    def search_bm25_only(self, query: str, limit: int = 20) -> list[SearchResult]:
        self._stats["searches"] += 1
        results = self._bm25.search(query, limit)
        if results:
            self._stats["bm25_hits"] += 1
        return [SearchResult(doc_id=doc_id, content=self._bm25.get_document(doc_id).content if self._bm25.get_document(doc_id) else "",
                bm25_score=score, combined_score=score,
                metadata=self._bm25.get_document(doc_id).metadata if self._bm25.get_document(doc_id) else {})
            for doc_id, score in results]

    def search_semantic_only(self, query_embedding: list[float], limit: int = 20) -> list[SearchResult]:
        results = self._semantic.search(query_embedding, limit)
        return [SearchResult(doc_id=doc_id, content=self._bm25.get_document(doc_id).content if self._bm25.get_document(doc_id) else "",
                semantic_score=score, combined_score=score,
                metadata=self._bm25.get_document(doc_id).metadata if self._bm25.get_document(doc_id) else {})
            for doc_id, score in results]

    def clear(self) -> None:
        self._bm25.clear()
        self._semantic.clear()
        self._stats["docs_indexed"] = 0

    def get_document_count(self) -> int:
        return self._stats["docs_indexed"]

    def get_statistics(self) -> dict[str, Any]:
        return {**self._stats, "alpha": self._settings.alpha, "fusion_method": self._settings.fusion_method}

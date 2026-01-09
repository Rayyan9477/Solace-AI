"""
Solace-AI Memory Service - Hybrid Search Engine.
BM25 + semantic vector search with reciprocal rank fusion.
"""
from __future__ import annotations
import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    term_frequencies: dict[str, int] = field(default_factory=dict)
    doc_length: int = 0

    def __post_init__(self) -> None:
        if not self.term_frequencies and self.content:
            tokens = self._tokenize(self.content)
            self.term_frequencies = dict(Counter(tokens))
            self.doc_length = len(tokens)

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
    """BM25 keyword search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[UUID, SearchDocument] = {}
        self._inverted_index: dict[str, set[UUID]] = {}
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._total_docs: int = 0

    def add_document(self, doc: SearchDocument) -> None:
        """Add a document to the index."""
        self._documents[doc.doc_id] = doc
        self._total_docs += 1
        total_length = sum(d.doc_length for d in self._documents.values())
        self._avg_doc_length = total_length / self._total_docs if self._total_docs > 0 else 0
        for term in doc.term_frequencies.keys():
            if term not in self._inverted_index:
                self._inverted_index[term] = set()
            self._inverted_index[term].add(doc.doc_id)
            self._doc_freqs[term] = len(self._inverted_index[term])

    def remove_document(self, doc_id: UUID) -> bool:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return False
        doc = self._documents[doc_id]
        for term in doc.term_frequencies.keys():
            if term in self._inverted_index:
                self._inverted_index[term].discard(doc_id)
                self._doc_freqs[term] = len(self._inverted_index[term])
        del self._documents[doc_id]
        self._total_docs -= 1
        if self._total_docs > 0:
            total_length = sum(d.doc_length for d in self._documents.values())
            self._avg_doc_length = total_length / self._total_docs
        return True

    def search(self, query: str, limit: int = 20) -> list[tuple[UUID, float]]:
        """Search the index using BM25 scoring."""
        query_terms = SearchDocument._tokenize(query)
        if not query_terms:
            return []
        scores: dict[UUID, float] = {}
        for term in query_terms:
            if term not in self._inverted_index:
                continue
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id in self._inverted_index[term]:
                doc = self._documents[doc_id]
                tf = doc.term_frequencies.get(term, 0)
                doc_len = doc.doc_length
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_doc_length))
                score = idf * (numerator / denominator) if denominator > 0 else 0
                scores[doc_id] = scores.get(doc_id, 0) + score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def get_document(self, doc_id: UUID) -> SearchDocument | None:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def clear(self) -> None:
        """Clear the index."""
        self._documents.clear()
        self._inverted_index.clear()
        self._doc_freqs.clear()
        self._avg_doc_length = 0.0
        self._total_docs = 0


class SemanticSearcher:
    """Semantic vector search using cosine similarity."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self._documents: dict[UUID, SearchDocument] = {}

    def add_document(self, doc: SearchDocument) -> None:
        """Add a document with embedding."""
        if doc.embedding and len(doc.embedding) == self.dimension:
            self._documents[doc.doc_id] = doc

    def remove_document(self, doc_id: UUID) -> bool:
        """Remove a document."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def search(self, query_embedding: list[float], limit: int = 20,
               min_similarity: float = 0.0) -> list[tuple[UUID, float]]:
        """Search using cosine similarity."""
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
        """Clear the index."""
        self._documents.clear()


class HybridSearchEngine:
    """Hybrid search combining BM25 and semantic search."""

    def __init__(self, settings: HybridSearchSettings | None = None,
                 embedding_fn: Callable[[str], list[float]] | None = None) -> None:
        self._settings = settings or HybridSearchSettings()
        self._bm25 = BM25Index(k1=self._settings.bm25_k1, b=self._settings.bm25_b)
        self._semantic = SemanticSearcher(dimension=self._settings.embedding_dimension)
        self._embedding_fn = embedding_fn
        self._stats = {"searches": 0, "docs_indexed": 0, "bm25_hits": 0, "semantic_hits": 0}

    def index_document(self, doc_id: UUID, content: str, embedding: list[float] | None = None,
                       metadata: dict[str, Any] | None = None) -> None:
        """Index a document for hybrid search."""
        if embedding is None and self._embedding_fn:
            embedding = self._embedding_fn(content)
        doc = SearchDocument(
            doc_id=doc_id, content=content,
            embedding=embedding or [], metadata=metadata or {},
        )
        self._bm25.add_document(doc)
        if doc.embedding:
            self._semantic.add_document(doc)
        self._stats["docs_indexed"] += 1

    def remove_document(self, doc_id: UUID) -> bool:
        """Remove a document from all indices."""
        bm25_removed = self._bm25.remove_document(doc_id)
        semantic_removed = self._semantic.remove_document(doc_id)
        if bm25_removed or semantic_removed:
            self._stats["docs_indexed"] = max(0, self._stats["docs_indexed"] - 1)
        return bm25_removed or semantic_removed

    def search(self, query: str, query_embedding: list[float] | None = None,
               limit: int | None = None, alpha: float | None = None) -> list[SearchResult]:
        """Perform hybrid search."""
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
        logger.debug("hybrid_search_completed", query_len=len(query),
                     results=len(results), time_ms=int((time.perf_counter() - start) * 1000))
        return results

    def _reciprocal_rank_fusion(self, bm25_results: list[tuple[UUID, float]],
                                 semantic_results: list[tuple[UUID, float]],
                                 limit: int) -> list[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        k = self._settings.rrf_k
        scores: dict[UUID, float] = {}
        bm25_scores: dict[UUID, float] = {}
        semantic_scores: dict[UUID, float] = {}
        for rank, (doc_id, score) in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            bm25_scores[doc_id] = score
        for rank, (doc_id, score) in enumerate(semantic_results, 1):
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            semantic_scores[doc_id] = score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        results: list[SearchResult] = []
        for doc_id in sorted_ids:
            doc = self._bm25.get_document(doc_id)
            content = doc.content if doc else ""
            metadata = doc.metadata if doc else {}
            results.append(SearchResult(
                doc_id=doc_id, content=content,
                bm25_score=bm25_scores.get(doc_id, 0),
                semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=metadata,
            ))
        return results

    def _weighted_fusion(self, bm25_results: list[tuple[UUID, float]],
                          semantic_results: list[tuple[UUID, float]],
                          alpha: float, limit: int) -> list[SearchResult]:
        """Fuse results using weighted combination."""
        bm25_max = max((s for _, s in bm25_results), default=1.0) or 1.0
        semantic_max = max((s for _, s in semantic_results), default=1.0) or 1.0
        scores: dict[UUID, float] = {}
        bm25_scores: dict[UUID, float] = {}
        semantic_scores: dict[UUID, float] = {}
        for doc_id, score in bm25_results:
            normalized = score / bm25_max
            scores[doc_id] = (1 - alpha) * normalized
            bm25_scores[doc_id] = score
        for doc_id, score in semantic_results:
            normalized = score / semantic_max
            scores[doc_id] = scores.get(doc_id, 0) + alpha * normalized
            semantic_scores[doc_id] = score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        results: list[SearchResult] = []
        for doc_id in sorted_ids:
            if scores[doc_id] < self._settings.min_score_threshold:
                continue
            doc = self._bm25.get_document(doc_id)
            content = doc.content if doc else ""
            metadata = doc.metadata if doc else {}
            results.append(SearchResult(
                doc_id=doc_id, content=content,
                bm25_score=bm25_scores.get(doc_id, 0),
                semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=metadata,
            ))
        return results

    def _linear_fusion(self, bm25_results: list[tuple[UUID, float]],
                        semantic_results: list[tuple[UUID, float]],
                        alpha: float, limit: int) -> list[SearchResult]:
        """Simple linear combination fusion."""
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
            content = doc.content if doc else ""
            metadata = doc.metadata if doc else {}
            results.append(SearchResult(
                doc_id=doc_id, content=content,
                bm25_score=bm25_scores.get(doc_id, 0),
                semantic_score=semantic_scores.get(doc_id, 0),
                combined_score=scores[doc_id], metadata=metadata,
            ))
        return results

    def search_bm25_only(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search using BM25 only."""
        self._stats["searches"] += 1
        results = self._bm25.search(query, limit)
        if results:
            self._stats["bm25_hits"] += 1
        return [
            SearchResult(
                doc_id=doc_id, content=self._bm25.get_document(doc_id).content if self._bm25.get_document(doc_id) else "",
                bm25_score=score, combined_score=score,
                metadata=self._bm25.get_document(doc_id).metadata if self._bm25.get_document(doc_id) else {},
            )
            for doc_id, score in results
        ]

    def search_semantic_only(self, query_embedding: list[float], limit: int = 20) -> list[SearchResult]:
        """Search using semantic similarity only."""
        results = self._semantic.search(query_embedding, limit)
        return [
            SearchResult(
                doc_id=doc_id, content=self._bm25.get_document(doc_id).content if self._bm25.get_document(doc_id) else "",
                semantic_score=score, combined_score=score,
                metadata=self._bm25.get_document(doc_id).metadata if self._bm25.get_document(doc_id) else {},
            )
            for doc_id, score in results
        ]

    def clear(self) -> None:
        """Clear all indices."""
        self._bm25.clear()
        self._semantic.clear()
        self._stats["docs_indexed"] = 0

    def get_document_count(self) -> int:
        """Get count of indexed documents."""
        return self._stats["docs_indexed"]

    def get_statistics(self) -> dict[str, Any]:
        """Get search statistics."""
        return {
            **self._stats,
            "alpha": self._settings.alpha,
            "fusion_method": self._settings.fusion_method,
        }

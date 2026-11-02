"""Lightweight BM25 retrieval engine over user knowledge chunks.

Provides deterministic evidence retrieval over user-ingested documents
with strict classification of match confidence:
- 'relevant_passage': Unambiguous top-matching chunk with score >= min_score_threshold.
- 'ambiguous_evidence': Competing chunks with close scores requiring disambiguation.
- 'insufficient_evidence': Best candidate score below relevance threshold.
- 'no_match': Empty corpus or query containing no matching terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.local_fact_retrieval import STOPWORDS, tokenize_text
from src.user_knowledge import KnowledgeChunk, KnowledgeSession, KnowledgeSource


@dataclass
class KnowledgeRetrievalResult:
    """Outcome of retrieving user knowledge for a specific query."""

    query: str
    status: str  # 'relevant_passage', 'ambiguous_evidence', 'insufficient_evidence', 'no_match'
    chunk: Optional[KnowledgeChunk]
    source_id: Optional[str]
    source_display_name: Optional[str]
    score: float
    is_ambiguous: bool
    candidates: List[Dict[str, Any]]
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.chunk is not None:
            d["chunk"] = self.chunk.to_dict()
        return d


class UserKnowledgeRetriever:
    """BM25 index and retriever operating over user knowledge chunks."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        sources_by_id: Optional[Dict[str, KnowledgeSource]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        min_score_threshold: float = 1.0,
        ambiguity_gap_threshold: float = 0.15,
    ) -> None:
        self.chunks: List[KnowledgeChunk] = list(chunks)
        self.sources_by_id = sources_by_id or {}
        self.k1 = k1
        self.b = b
        self.min_score_threshold = min_score_threshold
        self.ambiguity_gap_threshold = ambiguity_gap_threshold

        self.n_docs = len(self.chunks)
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}

        for c in self.chunks:
            tokens = tokenize_text(c.text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / max(self.n_docs, 1)

    @classmethod
    def from_session(
        cls,
        session: KnowledgeSession,
        k1: float = 1.5,
        b: float = 0.75,
        min_score_threshold: float = 1.0,
        ambiguity_gap_threshold: float = 0.15,
    ) -> UserKnowledgeRetriever:
        """Construct retriever directly from a KnowledgeSession."""
        chunks = session.get_all_chunks()
        return cls(
            chunks=chunks,
            sources_by_id=session.sources,
            k1=k1,
            b=b,
            min_score_threshold=min_score_threshold,
            ambiguity_gap_threshold=ambiguity_gap_threshold,
        )

    def _score_bm25(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        if doc_len == 0:
            return 0.0

        term_counts: Dict[str, int] = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1

        score = 0.0
        for q in query_tokens:
            if q not in term_counts:
                continue
            freq = term_counts[q]
            df = self.doc_freqs.get(q, 0)
            idf = max(0.25, math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))
            num = freq * (self.k1 + 1.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (num / denom)

        return score

    def retrieve(self, query: str, top_k: int = 3) -> KnowledgeRetrievalResult:
        """Find the most relevant passage for the query from ingested chunks."""
        q_tokens = tokenize_text(query)
        if not q_tokens or self.n_docs == 0:
            return KnowledgeRetrievalResult(
                query=query,
                status="no_match",
                chunk=None,
                source_id=None,
                source_display_name=None,
                score=0.0,
                is_ambiguous=False,
                candidates=[],
                error_code="no_match",
            )

        scores: List[Tuple[int, float]] = []
        for i in range(self.n_docs):
            score = self._score_bm25(q_tokens, i)
            if score > 0.0:
                scores.append((i, score))

        if not scores:
            return KnowledgeRetrievalResult(
                query=query,
                status="no_match",
                chunk=None,
                source_id=None,
                source_display_name=None,
                score=0.0,
                is_ambiguous=False,
                candidates=[],
                error_code="no_match",
            )

        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates: List[Dict[str, Any]] = []
        for idx, sc in scores[:top_k]:
            c = self.chunks[idx]
            s_name = self.sources_by_id.get(c.source_id).display_name if c.source_id in self.sources_by_id else "Unknown"
            top_candidates.append(
                {
                    "chunk_id": c.chunk_id,
                    "source_id": c.source_id,
                    "source_display_name": s_name,
                    "score": round(sc, 4),
                    "token_count": c.token_count,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text_preview": c.text[:80] + ("..." if len(c.text) > 80 else ""),
                }
            )

        best_idx, best_score = scores[0]
        best_chunk = self.chunks[best_idx]
        best_source_name = (
            self.sources_by_id.get(best_chunk.source_id).display_name
            if best_chunk.source_id in self.sources_by_id
            else "Unknown"
        )

        # 1. Below relevance threshold -> insufficient evidence
        if best_score < self.min_score_threshold:
            return KnowledgeRetrievalResult(
                query=query,
                status="insufficient_evidence",
                chunk=best_chunk,
                source_id=best_chunk.source_id,
                source_display_name=best_source_name,
                score=round(best_score, 4),
                is_ambiguous=False,
                candidates=top_candidates,
                error_code="insufficient_evidence",
            )

        # 2. Check for ambiguity between top 1 and top 2
        if len(scores) > 1:
            second_idx, second_score = scores[1]
            if second_score >= self.min_score_threshold:
                gap = best_score - second_score
                relative_gap = gap / best_score
                if relative_gap < self.ambiguity_gap_threshold:
                    return KnowledgeRetrievalResult(
                        query=query,
                        status="ambiguous_evidence",
                        chunk=best_chunk,
                        source_id=best_chunk.source_id,
                        source_display_name=best_source_name,
                        score=round(best_score, 4),
                        is_ambiguous=True,
                        candidates=top_candidates,
                        error_code="ambiguous_evidence",
                    )

        # 3. High confidence relevant passage
        return KnowledgeRetrievalResult(
            query=query,
            status="relevant_passage",
            chunk=best_chunk,
            source_id=best_chunk.source_id,
            source_display_name=best_source_name,
            score=round(best_score, 4),
            is_ambiguous=False,
            candidates=top_candidates,
            error_code=None,
        )

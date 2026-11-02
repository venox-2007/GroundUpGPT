"""Lightweight BM25 retrieval engine with entity-aware disambiguation and conflict guardrails.

Enhancements in Step 45:
1. Named entity extraction and discrimination (e.g., Project Orion vs Project Aurora).
2. Cross-document conflict detection (identifying contradictory dates/facts across sources).
3. Robust classification:
   - 'relevant_passage': Unambiguous, verified entity match, no conflict.
   - 'ambiguous_evidence': Competing chunks with close scores requiring clarification.
   - 'conflicting_evidence': Contradictory statements detected across documents.
   - 'insufficient_evidence': Score below threshold or missing essential entity overlap.
   - 'no_match': Empty corpus or query containing no matching terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.local_fact_retrieval import STOPWORDS, tokenize_text
from src.user_knowledge import KnowledgeChunk, KnowledgeSession, KnowledgeSource

COMMON_QUESTION_TERMS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "can", "could", "would", "should", "will", "did", "does", "do", "is",
    "are", "was", "were", "tell", "explain", "describe", "give", "list",
    "name", "about", "please", "many", "much", "time", "year", "date",
}


def extract_query_entities(query: str) -> List[str]:
    """Extract distinct proper nouns, capitalized entity terms, and project/mission names."""
    entities: List[str] = []

    # 1. Patterns like "Project <Name>", "Mission <Name>", "Team <Name>", "Dr. <Name>"
    pattern = re.compile(
        r"\b(?:project|mission|operation|team|doctor|dr\.?|program|system|model)\s+([a-zA-Z0-9_-]+)\b",
        re.IGNORECASE,
    )
    for m in pattern.finditer(query):
        ent = m.group(1).lower()
        if ent not in STOPWORDS and ent not in COMMON_QUESTION_TERMS and len(ent) > 1:
            entities.append(ent)

    # 2. Capitalized words (excluding beginning of sentence words that are common)
    words = query.strip().split()
    for i, w in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", w)
        if not cleaned or len(cleaned) < 2:
            continue
        c_lower = cleaned.lower()
        if c_lower in STOPWORDS or c_lower in COMMON_QUESTION_TERMS:
            continue
        # Capitalized or uppercase acronym
        if cleaned[0].isupper() or cleaned.isupper():
            entities.append(c_lower)

    # Deduplicate preserving order
    return list(dict.fromkeys(entities))


def extract_key_values(text: str) -> Dict[str, Set[str]]:
    """Extract dates, years, numbers, teams, tech stacks, and locations for conflict detection."""
    years = set(re.findall(r"\b(?:17\d\d|18\d\d|19\d\d|20\d\d)\b", text))
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    teams = set(re.findall(r"\b(?:team|group)\s+([a-zA-Z0-9_-]+)\b", text, re.IGNORECASE))
    projects = set(re.findall(r"\b(?:project|mission)\s+([a-zA-Z0-9_-]+)\b", text, re.IGNORECASE))
    tech = set(re.findall(r"\b(python|rust|c\+\+|javascript|typescript|golang|go|pytorch|tensorflow)\b", text, re.IGNORECASE))
    locations = set(re.findall(r"\b(houston|orlando|seattle|tokyo|london|paris|mars|moon|jupiter|saturn)\b", text, re.IGNORECASE))
    times = set(re.findall(r"\b\d\d:\d\d(?:\s*utc)?\b", text, re.IGNORECASE))

    return {
        "years": {y.lower() for y in years},
        "numbers": {n.lower() for n in numbers},
        "teams": {t.lower() for t in teams},
        "projects": {p.lower() for p in projects},
        "tech": {t.lower() for t in tech},
        "locations": {loc.lower() for loc in locations},
        "times": {tm.lower() for tm in times},
    }


def detect_contradiction(
    chunk1: Union[KnowledgeChunk, str],
    chunk2: Union[KnowledgeChunk, str],
    source_name1: str = "source_1",
    source_name2: str = "source_2",
) -> Optional[Dict[str, Any]]:
    """Detect if two chunks contain contradictory factual assertions on the same subject."""
    t1 = chunk1.text if isinstance(chunk1, KnowledgeChunk) else str(chunk1)
    t2 = chunk2.text if isinstance(chunk2, KnowledgeChunk) else str(chunk2)

    kv1 = extract_key_values(t1)
    kv2 = extract_key_values(t2)

    # If the two chunks explicitly reference different named projects, they are not contradictory
    if kv1["projects"] and kv2["projects"] and not (kv1["projects"] & kv2["projects"]):
        return None

    # Calculate content word overlap (excluding common short stopwords)
    stopwords = {"the", "a", "an", "in", "on", "at", "by", "of", "to", "was", "is", "were", "are", "and", "with", "using"}
    words1 = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t1.lower()) if w not in stopwords}
    words2 = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t2.lower()) if w not in stopwords}
    common_words = words1 & words2

    # 1. Contradictory years
    if kv1["years"] and kv2["years"] and not (kv1["years"] & kv2["years"]):
        return {
            "type": "conflicting_dates",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["years"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["years"])),
            "message": f"Contradictory dates: {source_name1} states {list(kv1['years'])}, whereas {source_name2} states {list(kv2['years'])}.",
        }

    # 2. Contradictory times
    if kv1["times"] and kv2["times"] and not (kv1["times"] & kv2["times"]):
        return {
            "type": "conflicting_times",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["times"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["times"])),
            "message": f"Contradictory times: {source_name1} states {list(kv1['times'])}, whereas {source_name2} states {list(kv2['times'])}.",
        }

    # 3. Contradictory teams / actors
    if kv1["teams"] and kv2["teams"] and not (kv1["teams"] & kv2["teams"]):
        return {
            "type": "conflicting_teams",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["teams"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["teams"])),
            "message": f"Contradictory actors: {source_name1} attributes to Team {list(kv1['teams'])}, whereas {source_name2} attributes to Team {list(kv2['teams'])}.",
        }

    # 4. Contradictory tech stacks
    if kv1["tech"] and kv2["tech"] and not (kv1["tech"] & kv2["tech"]):
        return {
            "type": "conflicting_tech",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["tech"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["tech"])),
            "message": f"Contradictory tech: {source_name1} states {list(kv1['tech'])}, whereas {source_name2} states {list(kv2['tech'])}.",
        }

    # 5. Contradictory locations / celestial bodies
    if kv1["locations"] and kv2["locations"] and not (kv1["locations"] & kv2["locations"]):
        return {
            "type": "conflicting_locations",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["locations"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["locations"])),
            "message": f"Contradictory locations: {source_name1} states {list(kv1['locations'])}, whereas {source_name2} states {list(kv2['locations'])}.",
        }

    # 6. Disjoint numbers on highly similar topical statements (>= 2 common words)
    if len(common_words) >= 2 and kv1["numbers"] and kv2["numbers"] and not (kv1["numbers"] & kv2["numbers"]):
        return {
            "type": "conflicting_quantities",
            "source_1": source_name1,
            "values_1": sorted(list(kv1["numbers"])),
            "source_2": source_name2,
            "values_2": sorted(list(kv2["numbers"])),
            "message": f"Contradictory quantities: {source_name1} states {list(kv1['numbers'])}, whereas {source_name2} states {list(kv2['numbers'])}.",
        }

    return None


@dataclass
class KnowledgeRetrievalResult:
    """Outcome of retrieving user knowledge for a specific query."""

    query: str
    status: str  # 'relevant_passage', 'ambiguous_evidence', 'conflicting_evidence', 'insufficient_evidence', 'no_match'
    chunk: Optional[KnowledgeChunk]
    source_id: Optional[str]
    source_display_name: Optional[str]
    score: float
    is_ambiguous: bool
    conflict_detected: bool = False
    conflicting_sources: List[str] = field(default_factory=list)
    conflict_details: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.chunk is not None:
            d["chunk"] = self.chunk.to_dict()
        return d


class UserKnowledgeRetriever:
    """BM25 index and retriever with entity-aware disambiguation and hallucination guardrails."""

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
        self.doc_token_sets: List[Set[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}

        for c in self.chunks:
            tokens = tokenize_text(c.text)
            self.doc_tokens.append(tokens)
            self.doc_token_sets.append(set(tokens))
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
        """Find the most relevant passage for the query with entity disambiguation and conflict checks."""
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

        query_entities = extract_query_entities(query)

        # Compute scores with entity-aware weighting
        scores: List[Tuple[int, float]] = []
        for i in range(self.n_docs):
            base_score = self._score_bm25(q_tokens, i)
            if base_score <= 0.0:
                continue

            # Entity disambiguation
            doc_token_set = self.doc_token_sets[i]
            adjusted_score = base_score

            if query_entities:
                matching_entities = [e for e in query_entities if e in doc_token_set]
                if matching_entities:
                    # Significant boost for exact entity presence
                    match_ratio = len(matching_entities) / len(query_entities)
                    adjusted_score = base_score * (1.5 + match_ratio)
                else:
                    # Penalize chunks that lack the requested named entity
                    adjusted_score = base_score * 0.4

            scores.append((i, adjusted_score))

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
            s_name = (
                self.sources_by_id.get(c.source_id).display_name
                if c.source_id in self.sources_by_id
                else "Unknown"
            )
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

        # 1. Below relevance threshold or missing required entity -> insufficient evidence
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

        if query_entities and not any(e in self.doc_token_sets[best_idx] for e in query_entities):
            return KnowledgeRetrievalResult(
                query=query,
                status="insufficient_evidence",
                chunk=best_chunk,
                source_id=best_chunk.source_id,
                source_display_name=best_source_name,
                score=round(best_score, 4),
                is_ambiguous=False,
                candidates=top_candidates,
                error_code="missing_query_entity",
            )

        # 2. Check for conflicting or contradictory evidence between top candidates
        if len(scores) > 1:
            second_idx, second_score = scores[1]
            if second_score >= self.min_score_threshold:
                # If query contains explicit named entities, candidate 2 must match them to be considered competing/conflicting
                second_has_query_entity = (
                    any(e in self.doc_token_sets[second_idx] for e in query_entities)
                    if query_entities
                    else True
                )
                if second_has_query_entity:
                    second_chunk = self.chunks[second_idx]
                    second_source_name = (
                        self.sources_by_id.get(second_chunk.source_id).display_name
                        if second_chunk.source_id in self.sources_by_id
                        else "Unknown"
                    )

                    # Check contradiction
                    contradiction = detect_contradiction(
                        best_chunk,
                        second_chunk,
                        best_source_name,
                        second_source_name,
                    )
                    if contradiction is not None:
                        return KnowledgeRetrievalResult(
                            query=query,
                            status="conflicting_evidence",
                            chunk=best_chunk,
                            source_id=best_chunk.source_id,
                            source_display_name=best_source_name,
                            score=round(best_score, 4),
                            is_ambiguous=True,
                            conflict_detected=True,
                            conflicting_sources=[best_chunk.source_id, second_chunk.source_id],
                            conflict_details=contradiction,
                            candidates=top_candidates,
                            error_code="conflicting_evidence",
                        )

                    # 3. Check for ambiguous evidence (close competing scores)
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

        # 4. High confidence relevant passage
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

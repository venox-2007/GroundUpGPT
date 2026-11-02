"""Local fact-retrieval engine for GroundUpGPT.

Provides lightweight, dependency-free BM25 retrieval over a compact, versioned
inventory of verified facts. Evaluates retrieval accuracy independently from neural
language model generation, preserving clear attribution and provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "tell", "give", "name", "please",
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize and normalize text into alphanumeric terms without punctuation."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class FactRecord:
    """Versioned factual record with verification and provenance metadata.

    Attributes:
        fact_id: Unique identifier string.
        entity: Primary entity or subject.
        category: Knowledge domain (e.g., 'geography', 'science', 'synthetic').
        search_terms: Keywords and aliases.
        question_forms: Common question variations answered by this fact.
        verified_answer: Canonical concise answer string.
        supporting_passage: Short factual context suitable for prompt conditioning.
        source: Provenance attribution.
        version: Fact schema version string.
    """

    fact_id: str
    entity: str
    category: str
    search_terms: List[str]
    question_forms: List[str]
    verified_answer: str
    supporting_passage: str
    source: str = "curated_groundup_facts_v1"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Detailed output of a fact retrieval query.

    Attributes:
        query: Original input search query.
        status: Retrieval outcome ("match", "no_match", "ambiguous").
        matched_fact: Top retrieved FactRecord if matched, otherwise None.
        retrieval_score: Raw BM25 matching score of the top candidate.
        is_ambiguous: Whether top candidates were too close in score to resolve confidently.
        candidates: List of top candidates and their scores: [(FactRecord, score)].
        source_provenance: Provenance tag of the matched fact.
        is_tool_computed: Flag explicitly indicating this is a tool-assisted result.
    """

    query: str
    status: str
    matched_fact: Optional[FactRecord]
    retrieval_score: float
    is_ambiguous: bool
    candidates: List[Tuple[Dict[str, Any], float]]
    source_provenance: Optional[str] = None
    is_tool_computed: bool = True


# Standard default facts repository
FACT_INVENTORY: List[FactRecord] = [
    # Geography & Capitals
    FactRecord(
        fact_id="geo_france_capital",
        entity="France",
        category="geography",
        search_terms=["france", "capital", "paris", "french"],
        question_forms=["What is the capital of France?", "Which city is France's capital?", "Capital city of France"],
        verified_answer="Paris",
        supporting_passage="Paris is the capital and most populous city of France.",
    ),
    FactRecord(
        fact_id="geo_germany_capital",
        entity="Germany",
        category="geography",
        search_terms=["germany", "capital", "berlin", "german"],
        question_forms=["What is the capital of Germany?", "Which city is the capital of Germany?", "Capital city of Germany"],
        verified_answer="Berlin",
        supporting_passage="Berlin is the capital and largest city of Germany by both area and population.",
    ),
    FactRecord(
        fact_id="geo_japan_capital",
        entity="Japan",
        category="geography",
        search_terms=["japan", "capital", "tokyo", "japanese"],
        question_forms=["What is the capital of Japan?", "Which city is Japan's capital?", "Capital city of Japan"],
        verified_answer="Tokyo",
        supporting_passage="Tokyo is the capital and chief financial center of Japan.",
    ),
    FactRecord(
        fact_id="geo_australia_capital",
        entity="Australia",
        category="geography",
        search_terms=["australia", "capital", "canberra", "australian"],
        question_forms=["What is the capital of Australia?", "Which city is Australia's capital?", "Capital city of Australia"],
        verified_answer="Canberra",
        supporting_passage="Canberra is the capital city of Australia, situated between Sydney and Melbourne.",
    ),
    FactRecord(
        fact_id="geo_canada_capital",
        entity="Canada",
        category="geography",
        search_terms=["canada", "capital", "ottawa", "canadian"],
        question_forms=["What is the capital of Canada?", "Capital of Canada"],
        verified_answer="Ottawa",
        supporting_passage="Ottawa is the capital city of Canada, located in the province of Ontario.",
    ),
    FactRecord(
        fact_id="geo_italy_capital",
        entity="Italy",
        category="geography",
        search_terms=["italy", "capital", "rome", "italian"],
        question_forms=["What is the capital of Italy?", "Capital of Italy"],
        verified_answer="Rome",
        supporting_passage="Rome is the capital city of Italy, famed for its ancient history and the Colosseum.",
    ),
    FactRecord(
        fact_id="geo_spain_capital",
        entity="Spain",
        category="geography",
        search_terms=["spain", "capital", "madrid", "spanish"],
        question_forms=["What is the capital of Spain?", "Capital of Spain"],
        verified_answer="Madrid",
        supporting_passage="Madrid is the capital and most populous city of Spain.",
    ),
    # Science & Physics
    FactRecord(
        fact_id="sci_water_room_temp",
        entity="Water",
        category="science",
        search_terms=["water", "room temperature", "liquid", "state of matter", "solid", "gas"],
        question_forms=["Is water a solid, liquid, or gas at room temperature?", "What state of matter is water at room temperature?"],
        verified_answer="liquid",
        supporting_passage="At standard room temperature (approximately 20°C to 25°C), water exists in a liquid state.",
    ),
    FactRecord(
        fact_id="sci_wood_conductivity",
        entity="Wood",
        category="science",
        search_terms=["wood", "conductor", "insulator", "electricity"],
        question_forms=["Is wood a conductor or an insulator?", "Does wood conduct electricity?"],
        verified_answer="insulator",
        supporting_passage="Dry wood is an electrical insulator because its bound electrons do not flow freely.",
    ),
    FactRecord(
        fact_id="sci_iron_classification",
        entity="Iron",
        category="science",
        search_terms=["iron", "metal", "non-metal", "mineral"],
        question_forms=["Is iron a metal or a non-metal?", "Classification of iron"],
        verified_answer="metal",
        supporting_passage="Iron is a metallic chemical element with symbol Fe and atomic number 26.",
    ),
    FactRecord(
        fact_id="sci_chemical_water",
        entity="Water formula",
        category="science",
        search_terms=["chemical formula", "water", "h2o", "molecule"],
        question_forms=["What is the chemical formula for water?", "Formula of water"],
        verified_answer="H2O",
        supporting_passage="The chemical formula for water is H2O, consisting of two hydrogen atoms bonded to one oxygen atom.",
    ),
    FactRecord(
        fact_id="sci_gold_symbol",
        entity="Gold",
        category="science",
        search_terms=["gold", "chemical symbol", "symbol for gold", "au", "element"],
        question_forms=["What is the chemical symbol for gold?", "Symbol for gold"],
        verified_answer="Au",
        supporting_passage="The chemical symbol for gold is Au, derived from the Latin word aurum.",
    ),
    FactRecord(
        fact_id="sci_speed_of_light",
        entity="Speed of light",
        category="science",
        search_terms=["speed of light", "vacuum", "physics", "constant", "meters per second"],
        question_forms=["What is the speed of light in a vacuum?", "How fast is light in vacuum?"],
        verified_answer="299,792,458 meters per second",
        supporting_passage="In a vacuum, the speed of light is exactly 299,792,458 meters per second.",
    ),
    FactRecord(
        fact_id="sci_primary_colors",
        entity="Primary colors",
        category="art_science",
        search_terms=["primary colors", "colors", "red", "blue", "yellow"],
        question_forms=["Name 3 primary colors.", "What are the three primary colors?"],
        verified_answer="red, blue, and yellow",
        supporting_passage="In traditional additive and subtractive color theory, the three primary colors are red, blue, and yellow.",
    ),
    # Biology
    FactRecord(
        fact_id="bio_whale_mammal",
        entity="Whale",
        category="biology",
        search_terms=["whale", "mammal", "fish", "marine mammal"],
        question_forms=["Is a whale a fish or a mammal?", "Are whales mammals?"],
        verified_answer="mammal",
        supporting_passage="Whales are marine mammals that breathe air through blowholes and nurse their young with milk.",
    ),
    FactRecord(
        fact_id="bio_photosynthesis_organelle",
        entity="Chloroplast",
        category="biology",
        search_terms=["photosynthesis", "organelle", "chloroplast", "plant cell"],
        question_forms=["Which organelle is responsible for photosynthesis?", "Where does photosynthesis occur in a plant cell?"],
        verified_answer="chloroplast",
        supporting_passage="Photosynthesis in plant cells takes place primarily inside organelles called chloroplasts.",
    ),
    # Synthetic / Fictional facts (for testing context utilization vs prior memorization)
    FactRecord(
        fact_id="syn_xylo7_mineral",
        entity="Xylo-7",
        category="synthetic",
        search_terms=["xylo-7", "xylo 7", "zephyra", "mineral", "crystalline"],
        question_forms=["What is Xylo-7?", "Where was Xylo-7 discovered?"],
        verified_answer="a crystalline mineral discovered on moon Zephyra",
        supporting_passage="Xylo-7 is a rare blue crystalline mineral discovered on the frozen moon Zephyra by probe Apollo-X in 2184.",
    ),
    FactRecord(
        fact_id="syn_quellix_capital",
        entity="Eldoria",
        category="synthetic",
        search_terms=["eldoria", "quellix", "capital of eldoria", "fictional kingdom"],
        question_forms=["What is the capital of Eldoria?", "Which city is the capital of the kingdom of Eldoria?"],
        verified_answer="Quellix",
        supporting_passage="Quellix is the coastal fortress capital of the mythical fantasy kingdom of Eldoria.",
    ),
    FactRecord(
        fact_id="syn_boranic_finch",
        entity="Boranic acid synthesis",
        category="synthetic",
        search_terms=["boranic acid synthesis", "boranic acid", "alistair finch", "2041"],
        question_forms=["Who invented boranic acid synthesis?", "In what year was boranic acid synthesis invented?"],
        verified_answer="Dr. Alistair Finch in 2041",
        supporting_passage="Boranic acid synthesis was invented by bio-chemist Dr. Alistair Finch in the year 2041.",
    ),
    FactRecord(
        fact_id="syn_zorblax_fleet",
        entity="Zorblax fleet",
        category="synthetic",
        search_terms=["zorblax", "zorblax fleet", "cruisers", "quantum cruisers", "size"],
        question_forms=["How many ships are in the Zorblax fleet?", "Size of the Zorblax fleet"],
        verified_answer="42 quantum cruisers",
        supporting_passage="The imperial Zorblax strike fleet consists of exactly 42 quantum cruisers commanded by Admiral Vex.",
    ),
]


class LocalFactRetriever:
    """Lightweight BM25 retriever over a fixed or custom inventory of facts."""

    def __init__(
        self,
        inventory: Sequence[FactRecord] = FACT_INVENTORY,
        k1: float = 1.5,
        b: float = 0.75,
        min_score_threshold: float = 1.2,
        ambiguity_threshold_ratio: float = 0.12,
    ) -> None:
        self.inventory: List[FactRecord] = list(inventory)
        self.k1 = k1
        self.b = b
        self.min_score_threshold = min_score_threshold
        self.ambiguity_threshold_ratio = ambiguity_threshold_ratio

        # Index documents
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.n_docs = len(self.inventory)

        for record in self.inventory:
            # Aggregate entity, search terms, question forms, and passage with term repetition for weighting
            weighted_text = (
                f"{record.entity} {record.entity} "
                f"{' '.join(record.search_terms)} {' '.join(record.search_terms)} "
                f"{' '.join(record.question_forms)} "
                f"{record.supporting_passage}"
            )
            tokens = tokenize_text(weighted_text)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / max(self.n_docs, 1)

    def _score_bm25(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        """Calculate BM25 relevance score for a single document."""
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

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve most relevant facts for the query.

        Args:
            query: Input question or search string.
            top_k: Maximum number of candidate facts to consider.

        Returns:
            RetrievalResult containing match status, top fact, and candidate scores.
        """
        q_tokens = tokenize_text(query)
        if not q_tokens or self.n_docs == 0:
            return RetrievalResult(
                query=query,
                status="no_match",
                matched_fact=None,
                retrieval_score=0.0,
                is_ambiguous=False,
                candidates=[],
                source_provenance=None,
            )

        scores: List[Tuple[int, float]] = []
        for i in range(self.n_docs):
            score = self._score_bm25(q_tokens, i)
            if score > 0.0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scores[:top_k]

        if not top_candidates:
            return RetrievalResult(
                query=query,
                status="no_match",
                matched_fact=None,
                retrieval_score=0.0,
                is_ambiguous=False,
                candidates=[],
                source_provenance=None,
            )

        best_idx, best_score = top_candidates[0]
        candidates_repr = [
            (self.inventory[idx].to_dict(), round(sc, 4)) for idx, sc in top_candidates
        ]

        # Check threshold
        if best_score < self.min_score_threshold:
            return RetrievalResult(
                query=query,
                status="no_match",
                matched_fact=None,
                retrieval_score=round(best_score, 4),
                is_ambiguous=False,
                candidates=candidates_repr,
                source_provenance=None,
            )

        # Ambiguity check: if 2nd candidate is close to 1st candidate
        is_ambiguous = False
        if len(top_candidates) > 1:
            second_idx, second_score = top_candidates[1]
            if (
                second_score >= self.min_score_threshold
                and (best_score - second_score) / max(best_score, 1e-6) < self.ambiguity_threshold_ratio
                and self.inventory[best_idx].entity != self.inventory[second_idx].entity
            ):
                is_ambiguous = True

        status = "ambiguous" if is_ambiguous else "match"
        matched_fact = None if is_ambiguous else self.inventory[best_idx]
        provenance = matched_fact.source if matched_fact else None

        return RetrievalResult(
            query=query,
            status=status,
            matched_fact=matched_fact,
            retrieval_score=round(best_score, 4),
            is_ambiguous=is_ambiguous,
            candidates=candidates_repr,
            source_provenance=provenance,
        )

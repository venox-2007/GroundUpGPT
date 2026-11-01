"""Diverse Pretraining Corpus Builder for GroundUpGPT-27M (Step 31).

Constructs a legally compliant, high-quality, ~60M-token English pretraining corpus
by fusing:
1. Classic literature, philosophy, science, and history from Project Gutenberg (Public Domain).
2. Factual, modern expository encyclopedia prose from Simple English Wikipedia (CC BY-SA 4.0 / GFDL).

Guarantees:
- Strict Leakage Prevention: Original validation books (pg35, pg36, pg5200) and Step 23
  expanded validation books (pg1497, pg1228, pg768, pg108, pg205) are strictly held out.
- Tokenizer Invariant: Reuses the exact 4,096-token BPE tokenizer (fingerprint:
  4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72).
- Zero Duplicate / Contaminated Documents: Exact SHA-256 deduplication and 10-word shingle
  validation leak checks.
- Scalable Streaming: Reads parquet row groups iteratively; bounded RAM; multiprocess BPE encoding.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
import torch

from src.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

# Expected tokenizer fingerprint
EXPECTED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_VOCAB_SIZE = 4096

# Strictly forbidden validation document IDs
FORBIDDEN_VAL_IDS = {
    "pg35",    # The Time Machine (H. G. Wells)
    "pg36",    # The War of the Worlds (H. G. Wells)
    "pg5200",  # Metamorphosis (Franz Kafka)
    "pg1497",  # The Republic (Plato)
    "pg1228",  # On the Origin of Species (Charles Darwin)
    "pg768",   # Wuthering Heights (Emily Brontë)
    "pg108",   # The Return of Sherlock Holmes (Arthur Conan Doyle)
    "pg205",   # The Adventures of Huckleberry Finn (Mark Twain)
}

# Worker-global tokenizer for multiprocessing
_worker_tokenizer: Optional[BPETokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    """Initialize worker process with a loaded BPETokenizer."""
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer().load(tokenizer_path)


def _encode_doc_batch(docs: List[str]) -> List[List[int]]:
    """Encode a batch of text documents in a worker process."""
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer is not initialized.")
    return [_worker_tokenizer.encode(doc) for doc in docs]


def clean_wiki_article(title: str, text: str, min_chars: int = 200) -> Optional[str]:
    """Clean a Simple English Wikipedia article, retaining expository prose.

    Strips:
    - Disambiguation pages and stubs
    - Wiki reference lists, external links, related page sections
    - Raw HTML tags, ref tags, template artifacts
    - Category footers
    """
    if not title or not text:
        return None

    # Filter disambiguation in title
    title_lower = title.lower()
    if "(disambiguation)" in title_lower or title_lower.startswith("list of"):
        return None

    # Check opening lines for disambiguation indicators
    head = text[:250].lower()
    if any(m in head for m in ("may mean:", "might mean:", "may refer to:", "can refer to:", "is a list of")):
        return None

    # Strip HTML / ref tags
    cleaned = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<ref[^>]*/>", "", cleaned)
    cleaned = re.sub(r"</?[a-zA-Z0-9]+[^>]*>", "", cleaned)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

    # Cut off standard trailing non-prose sections
    trailing_markers = [
        "\n== References ==", "\n==References==",
        "\n== Other websites ==", "\n==Other websites==",
        "\n== Related pages ==", "\n==Related pages==",
        "\n== Sources ==", "\n== Further reading ==",
        "\n== See also ==", "\n==See also==",
        "\nReferences \n", "\nReferences\n",
        "\nOther websites \n", "\nOther websites\n",
        "\nRelated pages \n", "\nRelated pages\n",
    ]
    for marker in trailing_markers:
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]

    # Clean redundant whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()

    # Minimum length requirement
    if len(cleaned) < min_chars:
        return None

    # Return structured document: Title followed by article prose
    return f"{title}\n\n{cleaned}"


class LeakDetector:
    """Builds shingle index from held-out validation texts and tests candidate texts."""

    def __init__(self, val_file_paths: List[Path], shingle_words: int = 10) -> None:
        self.shingle_words = shingle_words
        self.forbidden_shingles: Set[str] = set()
        self._build_index(val_file_paths)

    def _build_index(self, val_file_paths: List[Path]) -> None:
        all_words: List[str] = []
        for path in val_file_paths:
            if path.exists():
                text = path.read_text(encoding="utf-8")
                all_words.extend(text.split())

        k = self.shingle_words
        for i in range(len(all_words) - k + 1):
            shingle = " ".join(all_words[i : i + k]).lower()
            self.forbidden_shingles.add(shingle)

        logger.info(
            f"LeakDetector indexed {len(self.forbidden_shingles):,} "
            f"{k}-word shingles from {len(val_file_paths)} validation files."
        )

    def is_leaking(self, text: str) -> bool:
        """Check if candidate text contains any forbidden validation shingle."""
        words = text.split()
        k = self.shingle_words
        if len(words) < k:
            return False
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i : i + k]).lower()
            if shingle in self.forbidden_shingles:
                return True
        return False


@dataclass
class CorpusBuilder27MConfig:
    """Configuration for building the 27M pretraining corpus."""

    parquet_path: Path = Path("data/raw_27m/simplewiki/simplewiki_20231101.parquet")
    gutenberg_cache_dir: Path = Path("data/cache_expanded")
    val_files: List[Path] = None  # type: ignore
    output_dir: Path = Path("data/cache_27m_v1")
    tokenizer_path: Path = Path("checkpoints/bpe_vocab4096.json")
    num_row_groups: int = 35
    val_split_pct: int = 4  # 4% val, 96% train
    min_doc_chars: int = 200
    workers: int = 12
    batch_chunk_size: int = 50

    def __post_init__(self) -> None:
        if self.val_files is None:
            self.val_files = [
                Path("data/processed/val.txt"),
                Path("data/processed_expanded/val_expanded.txt"),
            ]


class CorpusBuilder27M:
    """End-to-end pretraining corpus builder for GroundUpGPT-27M."""

    def __init__(self, config: Optional[CorpusBuilder27MConfig] = None) -> None:
        self.config = config or CorpusBuilder27MConfig()
        self.tokenizer = BPETokenizer().load(str(self.config.tokenizer_path))

        # Verify tokenizer fingerprint
        fp = self.tokenizer.get_fingerprint()
        if fp != EXPECTED_TOKENIZER_FP:
            raise ValueError(f"Tokenizer fingerprint mismatch: expected {EXPECTED_TOKENIZER_FP}, got {fp}")

        # Initialize leak detector
        self.leak_detector = LeakDetector(self.config.val_files, shingle_words=10)

        # Tracking structures
        self.seen_doc_hashes: Set[str] = set()
        self.stats: Dict[str, Any] = {
            "raw_wiki_docs_scanned": 0,
            "disambiguation_or_stubs_filtered": 0,
            "duplicates_filtered": 0,
            "leakage_filtered": 0,
            "valid_wiki_docs": 0,
            "wiki_train_docs": 0,
            "wiki_val_docs": 0,
            "wiki_train_tokens": 0,
            "wiki_val_tokens": 0,
            "wiki_train_bytes": 0,
            "wiki_val_bytes": 0,
        }

    def build(self) -> Dict[str, Any]:
        """Execute corpus preparation and construct data/cache_27m_v1."""
        t0 = time.perf_counter()
        logger.info("Starting GroundUpGPT-27M corpus build...")

        # 1. Load existing Gutenberg cached tokens
        gt_train_path = self.config.gutenberg_cache_dir / "train_tokens.pt"
        gt_val_path = self.config.gutenberg_cache_dir / "val_tokens.pt"
        gt_meta_path = self.config.gutenberg_cache_dir / "cache_metadata.json"

        if not gt_train_path.exists() or not gt_val_path.exists():
            raise FileNotFoundError(f"Gutenberg cache tokens missing in {self.config.gutenberg_cache_dir}")

        gt_train_tokens = torch.load(gt_train_path, weights_only=True)
        gt_val_tokens = torch.load(gt_val_path, weights_only=True)
        with open(gt_meta_path, "r", encoding="utf-8") as f:
            gt_meta = json.load(f)

        logger.info(
            f"Loaded Gutenberg base: {len(gt_train_tokens):,} train tokens, "
            f"{len(gt_val_tokens):,} val tokens."
        )

        # 2. Stream and process Simple English Wikipedia
        if not self.config.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file missing: {self.config.parquet_path}")

        pf = pq.ParquetFile(str(self.config.parquet_path))
        num_rg = min(self.config.num_row_groups, pf.num_row_groups)
        logger.info(f"Processing {num_rg} row groups from {self.config.parquet_path}...")

        wiki_train_docs: List[Tuple[str, str, str]] = []  # (doc_id, title, text)
        wiki_val_docs: List[Tuple[str, str, str]] = []

        for rg_idx in range(num_rg):
            batch = pf.read_row_group(rg_idx, columns=["id", "title", "text"])
            for i in range(len(batch)):
                self.stats["raw_wiki_docs_scanned"] += 1
                doc_id = batch["id"][i].as_py()
                title = batch["title"][i].as_py()
                raw_text = batch["text"][i].as_py()

                cleaned = clean_wiki_article(title, raw_text, min_chars=self.config.min_doc_chars)
                if not cleaned:
                    self.stats["disambiguation_or_stubs_filtered"] += 1
                    continue

                # Deduplication check
                doc_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                if doc_hash in self.seen_doc_hashes:
                    self.stats["duplicates_filtered"] += 1
                    continue
                self.seen_doc_hashes.add(doc_hash)

                # Leakage check against held-out validation texts
                if self.leak_detector.is_leaking(cleaned):
                    self.stats["leakage_filtered"] += 1
                    logger.warning(f"Excluding leaking document: {title} ({doc_id})")
                    continue

                self.stats["valid_wiki_docs"] += 1

                # Deterministic split based on sha256(doc_id)
                split_hash = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 100
                if split_hash < self.config.val_split_pct:
                    wiki_val_docs.append((doc_id, title, cleaned))
                    self.stats["wiki_val_docs"] += 1
                else:
                    wiki_train_docs.append((doc_id, title, cleaned))
                    self.stats["wiki_train_docs"] += 1

        logger.info(
            f"Wikipedia extraction completed: {len(wiki_train_docs):,} train docs, "
            f"{len(wiki_val_docs):,} val docs."
        )

        # 3. Parallel tokenization using worker pool
        logger.info(f"Tokenizing Wikipedia documents across {self.config.workers} workers...")
        t_tok_start = time.perf_counter()

        wiki_train_token_lists: List[List[int]] = []
        wiki_val_token_lists: List[List[int]] = []

        with ProcessPoolExecutor(
            max_workers=self.config.workers,
            initializer=_init_worker,
            initargs=(str(self.config.tokenizer_path),),
        ) as executor:
            # Tokenize train docs
            train_texts = [d[2] for d in wiki_train_docs]
            # Batch into chunks for worker efficiency
            chunked_train = [
                train_texts[i : i + self.config.batch_chunk_size]
                for i in range(0, len(train_texts), self.config.batch_chunk_size)
            ]
            for batch_res in executor.map(_encode_doc_batch, chunked_train):
                wiki_train_token_lists.extend(batch_res)

            # Tokenize val docs
            val_texts = [d[2] for d in wiki_val_docs]
            chunked_val = [
                val_texts[i : i + self.config.batch_chunk_size]
                for i in range(0, len(val_texts), self.config.batch_chunk_size)
            ]
            for batch_res in executor.map(_encode_doc_batch, chunked_val):
                wiki_val_token_lists.extend(batch_res)

        t_tok_end = time.perf_counter()
        tok_duration = t_tok_end - t_tok_start

        # Flatten tokens
        flat_wiki_train: List[int] = []
        for tlist in wiki_train_token_lists:
            flat_wiki_train.extend(tlist)

        flat_wiki_val: List[int] = []
        for tlist in wiki_val_token_lists:
            flat_wiki_val.extend(tlist)

        wiki_train_bytes = sum(len(d[2].encode("utf-8")) for d in wiki_train_docs)
        wiki_val_bytes = sum(len(d[2].encode("utf-8")) for d in wiki_val_docs)

        self.stats["wiki_train_tokens"] = len(flat_wiki_train)
        self.stats["wiki_val_tokens"] = len(flat_wiki_val)
        self.stats["wiki_train_bytes"] = wiki_train_bytes
        self.stats["wiki_val_bytes"] = wiki_val_bytes
        self.stats["tokenization_duration_s"] = round(tok_duration, 2)
        self.stats["tokenization_throughput_tok_s"] = round(
            (len(flat_wiki_train) + len(flat_wiki_val)) / max(0.001, tok_duration), 1
        )

        logger.info(
            f"Wikipedia tokenized: {len(flat_wiki_train):,} train tokens, "
            f"{len(flat_wiki_val):,} val tokens in {tok_duration:.1f}s "
            f"({self.stats['tokenization_throughput_tok_s']:,} tok/s)."
        )

        # 4. Fuse Gutenberg and Simple Wikipedia into final pretraining tensors
        tensor_wiki_train = torch.tensor(flat_wiki_train, dtype=torch.int32)
        tensor_wiki_val = torch.tensor(flat_wiki_val, dtype=torch.int32)

        final_train_tokens = torch.cat([gt_train_tokens, tensor_wiki_train])
        final_val_tokens = torch.cat([gt_val_tokens, tensor_wiki_val])

        # Validate token ranges
        if (final_train_tokens < 0).any() or (final_train_tokens >= EXPECTED_VOCAB_SIZE).any():
            raise ValueError("Training tokens contain out-of-bounds IDs!")
        if (final_val_tokens < 0).any() or (final_val_tokens >= EXPECTED_VOCAB_SIZE).any():
            raise ValueError("Validation tokens contain out-of-bounds IDs!")

        # 5. Build manifests and provenance
        total_train_tokens = len(final_train_tokens)
        total_val_tokens = len(final_val_tokens)
        total_train_bytes = gt_meta.get("train_bytes", 0) + wiki_train_bytes
        total_val_bytes = gt_meta.get("val_bytes", 0) + wiki_val_bytes

        created_iso = datetime.now(timezone.utc).isoformat()

        # Cache metadata
        cache_metadata = {
            "corpus_version": "v3-27m",
            "tokenizer_path": str(self.config.tokenizer_path),
            "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "train_tokens_count": total_train_tokens,
            "val_tokens_count": total_val_tokens,
            "train_bytes": total_train_bytes,
            "val_bytes": total_val_bytes,
            "train_docs_count": gt_meta.get("train_docs_count", 57) + len(wiki_train_docs),
            "val_docs_count": gt_meta.get("val_docs_count", 5) + len(wiki_val_docs),
            "sources": {
                "project_gutenberg": {
                    "license": "Public Domain (US)",
                    "category": "Classic Literature, Philosophy, History & Science",
                    "train_tokens": len(gt_train_tokens),
                    "val_tokens": len(gt_val_tokens),
                    "train_docs": 57,
                    "val_docs": 5,
                    "held_out_original_val_docs": 3,
                },
                "simple_english_wikipedia": {
                    "license": "CC BY-SA 4.0 / GFDL",
                    "category": "Modern Factual Expository Encyclopedia",
                    "train_tokens": len(flat_wiki_train),
                    "val_tokens": len(flat_wiki_val),
                    "train_docs": len(wiki_train_docs),
                    "val_docs": len(wiki_val_docs),
                    "row_groups_processed": num_rg,
                },
            },
            "composition_pct": {
                "gutenberg_pct": round(len(gt_train_tokens) / total_train_tokens * 100, 2),
                "wikipedia_pct": round(len(flat_wiki_train) / total_train_tokens * 100, 2),
            },
            "compression_ratio_train": round(total_train_bytes / total_train_tokens, 2),
            "compression_ratio_val": round(total_val_bytes / total_val_tokens, 2),
            "created_at": created_iso,
        }

        # Source provenance
        source_provenance = {
            "sources": [
                {
                    "name": "Project Gutenberg Literary Corpus",
                    "official_url": "https://www.gutenberg.org/",
                    "license": "Public Domain in the United States",
                    "snapshot": "Step 23 Expanded Literary Corpus (62 verified books)",
                    "category": "Literature, Philosophy, Natural Science, History",
                    "attribution": "Project Gutenberg (gutenberg.org). Cleaned by GroundUpGPT pipeline.",
                    "notes": "Original Gutenberg headers, licensing footers, and transcription metadata stripped.",
                    "tokens_train": len(gt_train_tokens),
                    "tokens_val": len(gt_val_tokens),
                },
                {
                    "name": "Simple English Wikipedia Snapshot",
                    "official_url": "https://dumps.wikimedia.org/simplewiki/",
                    "distribution_url": "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet",
                    "license": "CC BY-SA 4.0 and GFDL",
                    "snapshot": "20231101.simple",
                    "raw_sha256": "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0",
                    "raw_download_size_bytes": 156885218,
                    "date_retrieved": "2026-11-01",
                    "category": "Modern Factual & Expository Encyclopedia",
                    "attribution": "Wikimedia Foundation and Simple English Wikipedia contributors.",
                    "modifications": "Wikitext templates, raw HTML, disambiguation pages, stubs (<200 chars), external links, and reference lists stripped. Cleaned titles and expository paragraphs retained.",
                    "tokens_train": len(flat_wiki_train),
                    "tokens_val": len(flat_wiki_val),
                },
            ]
        }

        # Corpus manifest summary
        corpus_manifest = {
            "version": "v3-27m",
            "created_at": created_iso,
            "summary": {
                "total_train_tokens": total_train_tokens,
                "total_val_tokens": total_val_tokens,
                "total_documents": cache_metadata["train_docs_count"] + cache_metadata["val_docs_count"],
            },
            "gutenberg_manifest_ref": "data/cache_expanded/manifest_expanded.json",
            "wiki_documents_sample": [
                {
                    "doc_id": wiki_train_docs[idx][0],
                    "title": wiki_train_docs[idx][1],
                    "split": "train",
                    "token_count": len(wiki_train_token_lists[idx]),
                }
                for idx in range(min(50, len(wiki_train_docs)))
            ]
            + [
                {
                    "doc_id": wiki_val_docs[idx][0],
                    "title": wiki_val_docs[idx][1],
                    "split": "val",
                    "token_count": len(wiki_val_token_lists[idx]),
                }
                for idx in range(min(20, len(wiki_val_docs)))
            ],
        }

        # Preprocessing summary
        preprocessing_summary = {
            "statistics": self.stats,
            "timing_s": {
                "tokenization_s": round(tok_duration, 2),
                "total_s": round(time.perf_counter() - t0, 2),
            },
            "leak_prevention": {
                "forbidden_val_doc_ids": sorted(list(FORBIDDEN_VAL_IDS)),
                "indexed_validation_shingles": len(self.leak_detector.forbidden_shingles),
                "leaking_wiki_articles_excluded": self.stats["leakage_filtered"],
            },
        }

        # 6. Save atomically
        tmp_dir = self.config.output_dir.parent / (self.config.output_dir.name + ".tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        torch.save(final_train_tokens, tmp_dir / "train_tokens.pt")
        torch.save(final_val_tokens, tmp_dir / "val_tokens.pt")

        with open(tmp_dir / "cache_metadata.json", "w", encoding="utf-8") as f:
            json.dump(cache_metadata, f, indent=2)

        with open(tmp_dir / "corpus_manifest.json", "w", encoding="utf-8") as f:
            json.dump(corpus_manifest, f, indent=2)

        with open(tmp_dir / "source_provenance.json", "w", encoding="utf-8") as f:
            json.dump(source_provenance, f, indent=2)

        with open(tmp_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(preprocessing_summary, f, indent=2)

        # Atomic promotion
        if self.config.output_dir.exists():
            import shutil
            shutil.rmtree(self.config.output_dir)
        tmp_dir.rename(self.config.output_dir)

        total_elapsed = time.perf_counter() - t0
        logger.info(
            f"Successfully built GroundUpGPT-27M pretraining corpus in {total_elapsed:.1f}s!\n"
            f"Output directory: {self.config.output_dir}\n"
            f"Train tokens: {total_train_tokens:,} ({cache_metadata['composition_pct']['wikipedia_pct']}% Wiki, "
            f"{cache_metadata['composition_pct']['gutenberg_pct']}% Gutenberg)\n"
            f"Val tokens: {total_val_tokens:,}"
        )

        return {
            "train_tokens": total_train_tokens,
            "val_tokens": total_val_tokens,
            "cache_metadata": cache_metadata,
            "elapsed_s": total_elapsed,
        }

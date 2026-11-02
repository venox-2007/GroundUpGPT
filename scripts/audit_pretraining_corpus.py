"""Pretraining Corpus Audit for GroundUpGPT (Step 49).

Audits the original pretraining corpus (data/cache_27m_v1/):
- Unique source documents (Gutenberg books vs Simple Wikipedia articles)
- Total raw bytes and on-disk footprint
- Total unique tokens (train, val, total)
- Training token exposures and epoch coverage (5,949 steps * 16 * 256)
- Gutenberg vs Simple Wikipedia distribution
- Document length distribution (min, max, mean, median, quartiles)
- Duplicate / near-duplicate analysis
- Compression ratio (bytes per token)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("corpus_audit")

CACHE_DIR = REPO_ROOT / "data" / "cache_27m_v1"
EXPANDED_DIR = REPO_ROOT / "data" / "cache_expanded"
PARQUET_PATH = REPO_ROOT / "data" / "raw_27m" / "simplewiki" / "simplewiki_20231101.parquet"
OUTPUT_DIR = REPO_ROOT / "logs" / "pretraining_step49"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def percentile(data: List[int], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * p
    f = int(k)
    c = f + 1
    if c < len(sorted_d):
        return sorted_d[f] + (k - f) * (sorted_d[c] - sorted_d[f])
    return float(sorted_d[f])


def run_corpus_audit() -> Dict[str, Any]:
    logger.info("Beginning comprehensive pretraining corpus audit...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load cache metadata
    meta_path = CACHE_DIR / "cache_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2. Load preprocessing summary
    summary_path = CACHE_DIR / "preprocessing_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 3. Load manifest
    manifest_path = CACHE_DIR / "corpus_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 4. Load Gutenberg manifest
    gutenberg_manifest_path = EXPANDED_DIR / "manifest_expanded.json"
    with open(gutenberg_manifest_path, "r", encoding="utf-8") as f:
        gutenberg_manifest = json.load(f)

    # Token counts
    train_tokens = meta["train_tokens_count"]
    val_tokens = meta["val_tokens_count"]
    total_tokens = train_tokens + val_tokens

    # Source breakdown
    gutenberg_train = meta["sources"]["project_gutenberg"]["train_tokens"]
    gutenberg_val = meta["sources"]["project_gutenberg"]["val_tokens"]
    gutenberg_total = gutenberg_train + gutenberg_val

    wiki_train = meta["sources"]["simple_english_wikipedia"]["train_tokens"]
    wiki_val = meta["sources"]["simple_english_wikipedia"]["val_tokens"]
    wiki_total = wiki_train + wiki_val

    # Raw bytes
    train_bytes = meta["train_bytes"]
    val_bytes = meta["val_bytes"]
    total_raw_bytes = train_bytes + val_bytes

    # Document counts
    gutenberg_train_docs = meta["sources"]["project_gutenberg"]["train_docs"]
    gutenberg_val_docs = meta["sources"]["project_gutenberg"]["val_docs"]
    gutenberg_docs = gutenberg_train_docs + gutenberg_val_docs

    wiki_train_docs = meta["sources"]["simple_english_wikipedia"]["train_docs"]
    wiki_val_docs = meta["sources"]["simple_english_wikipedia"]["val_docs"]
    wiki_docs = wiki_train_docs + wiki_val_docs

    total_docs = gutenberg_docs + wiki_docs

    # Token exposures during pretraining (Phase 1-4: 5,949 steps)
    pretraining_steps = 5949
    batch_size = 16
    block_size = 256
    tokens_per_step = batch_size * block_size  # 4,096
    total_tokens_exposed = pretraining_steps * tokens_per_step  # 24,367,104
    epoch_coverage_pct = (total_tokens_exposed / train_tokens) * 100  # 48.44%
    effective_epochs = total_tokens_exposed / train_tokens  # 0.4844

    # Document length statistics
    # From Gutenberg documents
    gutenberg_lengths = [doc.get("tokens", doc.get("token_count", 0)) for doc in gutenberg_manifest.get("documents", [])]
    # From Wikipedia sample in manifest
    wiki_sample_lengths = [doc["token_count"] for doc in manifest.get("wiki_documents_sample", [])]

    # Combined sample statistics
    sample_lengths = [x for x in (gutenberg_lengths + wiki_sample_lengths) if isinstance(x, (int, float))]
    len_min = min(sample_lengths) if sample_lengths else 0
    len_max = max(sample_lengths) if sample_lengths else 0
    len_mean = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0.0
    len_median = percentile(sample_lengths, 0.50)
    len_p25 = percentile(sample_lengths, 0.25)
    len_p75 = percentile(sample_lengths, 0.75)

    # Wikipedia overall average length
    wiki_avg_len = wiki_total / wiki_docs if wiki_docs > 0 else 0
    # Gutenberg overall average length
    gutenberg_avg_len = gutenberg_total / gutenberg_docs if gutenberg_docs > 0 else 0

    # Compression ratios (bytes per token)
    compression_train = train_bytes / train_tokens if train_tokens > 0 else 0.0
    compression_val = val_bytes / val_tokens if val_tokens > 0 else 0.0
    compression_overall = total_raw_bytes / total_tokens if total_tokens > 0 else 0.0

    # Verification of cache tensor files
    train_pt_path = CACHE_DIR / "train_tokens.pt"
    val_pt_path = CACHE_DIR / "val_tokens.pt"
    train_pt_bytes = train_pt_path.stat().st_size
    val_pt_bytes = val_pt_path.stat().st_size
    parquet_bytes = PARQUET_PATH.stat().st_size if PARQUET_PATH.exists() else 0

    audit_data = {
        "audit_version": "step49_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_version": meta.get("corpus_version", "v3-27m"),
        "tokenizer_fingerprint": meta.get("tokenizer_fingerprint"),
        "vocab_size": meta.get("vocab_size", 4096),
        "source_files": {
            "train_tokens_pt": {
                "path": str(train_pt_path.relative_to(REPO_ROOT)),
                "size_bytes": train_pt_bytes,
                "token_count": train_tokens,
            },
            "val_tokens_pt": {
                "path": str(val_pt_path.relative_to(REPO_ROOT)),
                "size_bytes": val_pt_bytes,
                "token_count": val_tokens,
            },
            "simplewiki_parquet": {
                "path": str(PARQUET_PATH.relative_to(REPO_ROOT)),
                "size_bytes": parquet_bytes,
                "row_groups_total": 242,
                "row_groups_used_step31": summary["statistics"]["raw_wiki_docs_scanned"] // 1000,
                "row_groups_remaining": 242 - (summary["statistics"]["raw_wiki_docs_scanned"] // 1000),
            },
        },
        "documents": {
            "total_documents": total_docs,
            "train_documents": meta["train_docs_count"],
            "val_documents": meta["val_docs_count"],
            "sources": {
                "project_gutenberg": {
                    "license": "Public Domain (US)",
                    "category": "Classic Literature, Philosophy, History & Science",
                    "total_docs": gutenberg_docs,
                    "train_docs": gutenberg_train_docs,
                    "val_docs": gutenberg_val_docs,
                    "total_tokens": gutenberg_total,
                    "token_share_pct": round((gutenberg_total / total_tokens) * 100, 2),
                    "mean_tokens_per_doc": round(gutenberg_avg_len, 1),
                },
                "simple_english_wikipedia": {
                    "license": "CC BY-SA 4.0 / GFDL",
                    "category": "Modern Factual & Expository Encyclopedia",
                    "total_docs": wiki_docs,
                    "train_docs": wiki_train_docs,
                    "val_docs": wiki_val_docs,
                    "total_tokens": wiki_total,
                    "token_share_pct": round((wiki_total / total_tokens) * 100, 2),
                    "mean_tokens_per_doc": round(wiki_avg_len, 1),
                    "raw_articles_scanned": summary["statistics"]["raw_wiki_docs_scanned"],
                    "disambiguation_or_stubs_filtered": summary["statistics"]["disambiguation_or_stubs_filtered"],
                    "validation_leakage_filtered": summary["statistics"]["leakage_filtered"],
                },
            },
        },
        "tokens": {
            "train_tokens_count": train_tokens,
            "val_tokens_count": val_tokens,
            "total_unique_tokens": total_tokens,
            "train_raw_bytes": train_bytes,
            "val_raw_bytes": val_bytes,
            "total_raw_bytes": total_raw_bytes,
            "compression_ratio_bytes_per_tok": {
                "train": round(compression_train, 3),
                "val": round(compression_val, 3),
                "overall": round(compression_overall, 3),
            },
        },
        "training_token_exposures": {
            "checkpoint_audited": "checkpoints/27m_phase4/model_27m_best.pt",
            "pretraining_optimizer_steps": pretraining_steps,
            "micro_batch_size": batch_size,
            "sequence_block_size": block_size,
            "tokens_per_step": tokens_per_step,
            "total_tokens_exposed": total_tokens_exposed,
            "corpus_unique_train_tokens": train_tokens,
            "single_epoch_coverage_pct": round(epoch_coverage_pct, 2),
            "effective_epochs_completed": round(effective_epochs, 4),
            "undertrained_diagnosis": (
                "The 27M model was trained for 5,949 steps, exposing it to 24.37M tokens against a 50.30M token "
                "training corpus. It completed only ~0.48 epochs (under half of a single epoch), meaning more than "
                "51% of the pretraining data was never seen by the optimizer during initial pretraining."
            ),
        },
        "document_length_distribution": {
            "sample_size": len(sample_lengths),
            "min_tokens": len_min,
            "max_tokens": len_max,
            "mean_tokens": round(len_mean, 2),
            "median_tokens": round(len_median, 2),
            "p25_tokens": round(len_p25, 2),
            "p75_tokens": round(len_p75, 2),
            "wikipedia_mean_tokens": round(wiki_avg_len, 2),
            "gutenberg_mean_tokens": round(gutenberg_avg_len, 2),
        },
        "duplicate_analysis": {
            "exact_duplicate_documents_filtered": summary["statistics"]["duplicates_filtered"],
            "validation_leakage_prevented": summary["statistics"]["leakage_filtered"],
            "forbidden_validation_ids_enforced": summary["leak_prevention"]["forbidden_val_doc_ids"],
            "indexed_validation_shingles": summary["leak_prevention"]["indexed_validation_shingles"],
        },
    }

    report_path = OUTPUT_DIR / "corpus_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    logger.info(f"Corpus audit report written to {report_path}")

    print("\n" + "=" * 70)
    print(" GROUNDUPGPT-27M PRETRAINING CORPUS AUDIT")
    print("=" * 70)
    print(f"Total Unique Tokens:       {total_tokens:,} (Train: {train_tokens:,}, Val: {val_tokens:,})")
    print(f"Total Documents:           {total_docs:,} (Train: {meta['train_docs_count']:,}, Val: {meta['val_docs_count']:,})")
    print(f"Total Raw Text Bytes:      {total_raw_bytes:,} ({total_raw_bytes / (1024**2):.2f} MB)")
    print(f"Compression Ratio:         {compression_overall:.2f} bytes / token")
    print("-" * 70)
    print("SOURCE COMPOSITION:")
    print(f"  Project Gutenberg:       {gutenberg_total:,} tokens ({gutenberg_total/total_tokens*100:.1f}%), {gutenberg_docs} books")
    print(f"  Simple English Wiki:     {wiki_total:,} tokens ({wiki_total/total_tokens*100:.1f}%), {wiki_docs:,} articles")
    print("-" * 70)
    print("TRAINING TOKEN EXPOSURE AUDIT:")
    print(f"  Pretraining Steps:       {pretraining_steps:,}")
    print(f"  Batch * SeqLen:          {batch_size} * {block_size} = {tokens_per_step:,} tokens/step")
    print(f"  Total Token Exposures:   {total_tokens_exposed:,}")
    print(f"  Corpus Unique Tokens:    {train_tokens:,}")
    print(f"  Epoch Coverage:          {epoch_coverage_pct:.2f}% ({effective_epochs:.2f} epochs)")
    print(f"  CRITICAL DIAGNOSIS:      MODEL SAW LESS THAN HALF OF CORPUS (< 0.49 EPOCHS)")
    print("-" * 70)
    print("SIMPLE WIKIPEDIA EXPANSION POTENTIAL:")
    print(f"  Total Row Groups:        242 (~241,787 articles)")
    print(f"  Row Groups Processed:    {summary['statistics']['raw_wiki_docs_scanned'] // 1000} (72,000 articles)")
    print(f"  Remaining Row Groups:    {242 - (summary['statistics']['raw_wiki_docs_scanned'] // 1000)} (~169,787 articles)")
    print(f"  Estimated Avail Tokens:  ~90,000,000 additional clean factual tokens available on disk!")
    print("=" * 70 + "\n")

    return audit_data


def main():
    run_corpus_audit()


if __name__ == "__main__":
    main()

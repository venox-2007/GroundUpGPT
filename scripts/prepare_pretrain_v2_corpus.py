"""GroundUpGPT Pretraining Corpus Expansion (Step 49).

Constructs the secondary expanded pretraining corpus (data/cache_pretrain_v2/):
- Adds 20M-30M unique tokens of clean, modern expository encyclopedia prose from
  Simple English Wikipedia (CC BY-SA 4.0 / GFDL).
- Processes previously unparsed row groups (72 to 119) from simplewiki_20231101.parquet.
- Rigorous cleaning: Unicode NFKC normalization, safe UTF-8 decoding, boilerplate & template
  stripping, stub/disambiguation removal, exact SHA-256 deduplication, 10-word shingle
  validation leakage check.
- Appends document boundary markers ("\\n\\n<|endoftext|>\\n\\n").
- Strict document-level train (95%) and validation (5%) split.
- Uses exact production 4,096-vocab BPE tokenizer (fingerprint: 4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72).
- Produces binary tensors and detailed reports:
    data/cache_pretrain_v2/train_tokens.pt
    data/cache_pretrain_v2/val_tokens.pt
    data/cache_pretrain_v2/source_manifest.json
    data/cache_pretrain_v2/cleaning_report.json
    data/cache_pretrain_v2/cache_metadata.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq
import torch

from src.bpe_tokenizer import BPETokenizer
from src.corpus_builder_27m import LeakDetector, clean_wiki_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prepare_pretrain_v2")

EXPECTED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_VOCAB_SIZE = 4096

_worker_tokenizer: Optional[BPETokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer().load(tokenizer_path)


def _encode_batch(docs: List[str]) -> List[List[int]]:
    global _worker_tokenizer
    if _worker_tokenizer is None:
        raise RuntimeError("Worker tokenizer is not initialized.")
    return [_worker_tokenizer.encode(d) for d in docs]


def normalize_text_unicode(text: str) -> str:
    """Normalize text using Unicode NFKC and strip non-printable controls."""
    text = unicodedata.normalize("NFKC", text)
    # Strip null bytes and non-printable control characters except \n, \r, \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def build_pretrain_v2_corpus(
    parquet_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    start_row_group: int = 72,
    num_row_groups: int = 48,
    val_split_pct: int = 5,
    min_doc_chars: int = 200,
    workers: int = 8,
    batch_chunk_size: int = 50,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify tokenizer
    logger.info(f"Loading and verifying tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer().load(str(tokenizer_path))
    fp = tokenizer.get_fingerprint()
    if fp != EXPECTED_TOKENIZER_FP:
        raise ValueError(f"Tokenizer fingerprint mismatch! Expected {EXPECTED_TOKENIZER_FP}, got {fp}")
    logger.info(f"Tokenizer fingerprint verified: {fp} (vocab: {tokenizer.vocab_size})")

    # 2. Setup LeakDetector
    val_files = [
        REPO_ROOT / "data" / "processed" / "val.txt",
        REPO_ROOT / "data" / "processed_expanded" / "val_expanded.txt",
    ]
    leak_detector = LeakDetector(val_files, shingle_words=10)

    # 3. Read Parquet Row Groups
    logger.info(
        f"Scanning {num_row_groups} row groups from {parquet_path} "
        f"(groups {start_row_group} to {start_row_group + num_row_groups - 1})..."
    )
    pf = pq.ParquetFile(parquet_path)
    total_rg = pf.num_row_groups
    if start_row_group + num_row_groups > total_rg:
        raise ValueError(f"Requested row groups exceed file capacity ({total_rg})")

    stats = {
        "raw_articles_scanned": 0,
        "disambiguation_or_stubs_filtered": 0,
        "duplicates_filtered": 0,
        "validation_leakage_filtered": 0,
        "valid_articles_retained": 0,
        "train_articles_count": 0,
        "val_articles_count": 0,
    }

    seen_hashes: Set[str] = set()
    train_docs: List[Tuple[str, str, str]] = []  # (doc_id, title, clean_text)
    val_docs: List[Tuple[str, str, str]] = []

    for rg_idx in range(start_row_group, start_row_group + num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["id", "title", "text"])
        for i in range(len(table)):
            stats["raw_articles_scanned"] += 1
            doc_id = table["id"][i].as_py()
            title = table["title"][i].as_py()
            raw_text = table["text"][i].as_py()

            if not raw_text or len(raw_text) < min_doc_chars:
                stats["disambiguation_or_stubs_filtered"] += 1
                continue

            # Unicode normalization
            norm_text = normalize_text_unicode(raw_text)

            # Strip wikitext, refs, category footers, check stubs
            cleaned = clean_wiki_article(title, norm_text, min_chars=min_doc_chars)
            if not cleaned:
                stats["disambiguation_or_stubs_filtered"] += 1
                continue

            # Append document boundary delimiter
            cleaned_with_boundary = cleaned.strip() + "\n\n<|endoftext|>\n\n"

            # Exact deduplication check
            doc_hash = hashlib.sha256(cleaned_with_boundary.encode("utf-8")).hexdigest()
            if doc_hash in seen_hashes:
                stats["duplicates_filtered"] += 1
                continue
            seen_hashes.add(doc_hash)

            # Shingle leakage check against held-out validation data
            if leak_detector.is_leaking(cleaned):
                stats["validation_leakage_filtered"] += 1
                continue

            stats["valid_articles_retained"] += 1

            # Deterministic document-level split based on sha256(doc_id)
            split_bucket = int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 100
            if split_bucket < val_split_pct:
                val_docs.append((doc_id, title, cleaned_with_boundary))
                stats["val_articles_count"] += 1
            else:
                train_docs.append((doc_id, title, cleaned_with_boundary))
                stats["train_articles_count"] += 1

    logger.info(
        f"Article filtering complete: {len(train_docs):,} train articles, "
        f"{len(val_docs):,} val articles from {stats['raw_articles_scanned']:,} raw articles scanned."
    )

    # 4. Multi-process BPE Tokenization
    logger.info(f"Tokenizing documents using {workers} parallel worker processes...")
    t_tok_start = time.perf_counter()

    train_token_lists: List[List[int]] = []
    val_token_lists: List[List[int]] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(tokenizer_path),),
    ) as executor:
        # Tokenize train articles
        train_texts = [d[2] for d in train_docs]
        chunked_train = [
            train_texts[i : i + batch_chunk_size]
            for i in range(0, len(train_texts), batch_chunk_size)
        ]
        for batch_res in executor.map(_encode_batch, chunked_train):
            train_token_lists.extend(batch_res)

        # Tokenize val articles
        val_texts = [d[2] for d in val_docs]
        chunked_val = [
            val_texts[i : i + batch_chunk_size]
            for i in range(0, len(val_texts), batch_chunk_size)
        ]
        for batch_res in executor.map(_encode_batch, chunked_val):
            val_token_lists.extend(batch_res)

    tok_duration = time.perf_counter() - t_tok_start
    logger.info(f"Tokenization completed in {tok_duration:.1f}s.")

    # Flatten tokens
    flat_train: List[int] = []
    for tl in train_token_lists:
        flat_train.extend(tl)

    flat_val: List[int] = []
    for tl in val_token_lists:
        flat_val.extend(tl)

    train_tensor = torch.tensor(flat_train, dtype=torch.int32)
    val_tensor = torch.tensor(flat_val, dtype=torch.int32)

    # Validate tensor bounds
    if (train_tensor < 0).any() or (train_tensor >= EXPECTED_VOCAB_SIZE).any():
        raise ValueError("Train tokens contain out-of-bounds IDs!")
    if (val_tensor < 0).any() or (val_tensor >= EXPECTED_VOCAB_SIZE).any():
        raise ValueError("Val tokens contain out-of-bounds IDs!")

    # 5. Measure Tokenizer Efficiency
    total_tokens = len(train_tensor) + len(val_tensor)
    total_chars = sum(len(d[2]) for d in train_docs) + sum(len(d[2]) for d in val_docs)
    total_words = sum(len(d[2].split()) for d in train_docs) + sum(len(d[2].split()) for d in val_docs)
    total_bytes = sum(len(d[2].encode("utf-8")) for d in train_docs) + sum(len(d[2].encode("utf-8")) for d in val_docs)

    tokens_per_char = total_tokens / max(1, total_chars)
    tokens_per_word = total_tokens / max(1, total_words)
    compression_ratio = total_bytes / max(1, total_tokens)

    # Vocabulary utilization
    unique_tokens_seen = set(train_tensor.unique().tolist()).union(set(val_tensor.unique().tolist()))
    vocab_utilization_pct = (len(unique_tokens_seen) / EXPECTED_VOCAB_SIZE) * 100

    tokenizer_efficiency = {
        "tokens_per_char": round(tokens_per_char, 4),
        "tokens_per_word": round(tokens_per_word, 4),
        "chars_per_token": round(total_chars / max(1, total_tokens), 2),
        "compression_ratio_bytes_per_tok": round(compression_ratio, 3),
        "unique_tokens_utilized": len(unique_tokens_seen),
        "total_vocab_size": EXPECTED_VOCAB_SIZE,
        "vocab_utilization_pct": round(vocab_utilization_pct, 2),
    }

    # 6. Save binary tensors
    train_pt_path = output_dir / "train_tokens.pt"
    val_pt_path = output_dir / "val_tokens.pt"
    torch.save(train_tensor, train_pt_path)
    torch.save(val_tensor, val_pt_path)

    # Compute hashes
    train_sha256 = hashlib.sha256(open(train_pt_path, "rb").read()).hexdigest()
    val_sha256 = hashlib.sha256(open(val_pt_path, "rb").read()).hexdigest()

    # 7. Create Manifest and Reports
    source_manifest = {
        "dataset_name": "pretrain_v2_simplewiki_expansion",
        "version": "v2-step49",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "Simple English Wikipedia Snapshot",
            "parquet_file": str(parquet_path.relative_to(REPO_ROOT)),
            "row_groups_processed": num_row_groups,
            "row_group_range": [start_row_group, start_row_group + num_row_groups - 1],
            "license": "CC BY-SA 4.0 / GFDL",
            "attribution": "Wikimedia Foundation and Simple English Wikipedia contributors",
        },
        "tokenizer": {
            "path": str(tokenizer_path.relative_to(REPO_ROOT)),
            "fingerprint": EXPECTED_TOKENIZER_FP,
            "vocab_size": EXPECTED_VOCAB_SIZE,
        },
        "statistics": {
            "train_tokens_count": len(train_tensor),
            "val_tokens_count": len(val_tensor),
            "total_tokens_count": total_tokens,
            "train_articles_count": len(train_docs),
            "val_articles_count": len(val_docs),
            "total_articles_count": len(train_docs) + len(val_docs),
            "train_tensor_sha256": train_sha256,
            "val_tensor_sha256": val_sha256,
        },
        "sample_articles": [
            {"doc_id": d[0], "title": d[1], "tokens": len(tl)}
            for d, tl in zip(train_docs[:10], train_token_lists[:10])
        ],
    }
    with open(output_dir / "source_manifest.json", "w", encoding="utf-8") as f:
        json.dump(source_manifest, f, indent=2)

    cleaning_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filtering_statistics": stats,
        "clean_retention_rate_pct": round((stats["valid_articles_retained"] / max(1, stats["raw_articles_scanned"])) * 100, 2),
        "tokenizer_efficiency": tokenizer_efficiency,
        "document_boundary_marker": "<|endoftext|>",
        "leak_prevention": {
            "forbidden_validation_files": [str(p.name) for p in val_files],
            "shingle_length_words": 10,
            "indexed_validation_shingles": len(leak_detector.forbidden_shingles),
            "leaking_articles_prevented": stats["validation_leakage_filtered"],
        },
    }
    with open(output_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)

    cache_metadata = {
        "corpus_version": "v2-pretrain-expanded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "train_tokens_count": len(train_tensor),
        "val_tokens_count": len(val_tensor),
        "total_tokens_count": total_tokens,
        "train_docs_count": len(train_docs),
        "val_docs_count": len(val_docs),
        "train_bytes": sum(len(d[2].encode("utf-8")) for d in train_docs),
        "val_bytes": sum(len(d[2].encode("utf-8")) for d in val_docs),
        "compression_ratio": round(compression_ratio, 3),
        "tokenizer_efficiency": tokenizer_efficiency,
    }
    with open(output_dir / "cache_metadata.json", "w", encoding="utf-8") as f:
        json.dump(cache_metadata, f, indent=2)

    total_wall_s = time.perf_counter() - t_start
    logger.info(f"Pretraining v2 dataset preparation complete in {total_wall_s:.1f}s.")

    print("\n" + "=" * 70)
    print(" PRETRAINING V2 DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Output Directory:          {output_dir}")
    print(f"Total Unique Tokens:       {total_tokens:,}")
    print(f"  - Train Tokens (95%):    {len(train_tensor):,} ({train_pt_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"  - Val Tokens (5%):       {len(val_tensor):,} ({val_pt_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"Total Clean Articles:      {len(train_docs) + len(val_docs):,}")
    print(f"  - Train Articles:        {len(train_docs):,}")
    print(f"  - Val Articles:          {len(val_docs):,}")
    print(f"Raw Articles Scanned:      {stats['raw_articles_scanned']:,}")
    print(f"Disambig / Stubs Filtered: {stats['disambiguation_or_stubs_filtered']:,}")
    print(f"Duplicates Filtered:       {stats['duplicates_filtered']:,}")
    print(f"Validation Leakage Filtered: {stats['validation_leakage_filtered']:,}")
    print("-" * 70)
    print("TOKENIZER EFFICIENCY METRICS:")
    print(f"  Tokens per Character:    {tokens_per_char:.4f} (chars/tok: {1/tokens_per_char:.2f})")
    print(f"  Tokens per Word:         {tokens_per_word:.4f}")
    print(f"  Compression Ratio:       {compression_ratio:.3f} bytes / token")
    print(f"  Vocab Utilization:       {len(unique_tokens_seen)} / 4096 ({vocab_utilization_pct:.1f}%)")
    print(f"Total Elapsed Time:        {total_wall_s:.1f}s (Tokenization: {tok_duration:.1f}s)")
    print("=" * 70 + "\n")

    return cache_metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare secondary expanded pretraining corpus (Step 49)")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=REPO_ROOT / "data" / "raw_27m" / "simplewiki" / "simplewiki_20231101.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache_pretrain_v2",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json",
    )
    parser.add_argument("--start-row-group", type=int, default=72)
    parser.add_argument("--num-row-groups", type=int, default=48)
    parser.add_argument("--val-split-pct", type=int, default=5)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    build_pretrain_v2_corpus(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        start_row_group=args.start_row_group,
        num_row_groups=args.num_row_groups,
        val_split_pct=args.val_split_pct,
        min_doc_chars=args.min_doc_chars,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

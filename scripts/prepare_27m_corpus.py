"""Command-line script to prepare the expanded diverse pretraining corpus for GroundUpGPT-27M (Step 31).

Usage:
    python scripts/prepare_27m_corpus.py --num-row-groups 35 --workers 12

Outputs:
    data/cache_27m_v1/
        train_tokens.pt
        val_tokens.pt
        cache_metadata.json
        corpus_manifest.json
        source_provenance.json
        preprocessing_summary.json
"""

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus_builder_27m import CorpusBuilder27M, CorpusBuilder27MConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build diverse pretraining corpus for GroundUpGPT-27M."
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("data/raw_27m/simplewiki/simplewiki_20231101.parquet"),
        help="Path to Simple English Wikipedia parquet file.",
    )
    parser.add_argument(
        "--gutenberg-cache-dir",
        type=Path,
        default=Path("data/cache_expanded"),
        help="Path to Step 23 expanded Gutenberg token cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache_27m_v1"),
        help="Target output directory for versioned cache.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("checkpoints/bpe_vocab4096.json"),
        help="Path to 4,096-vocab BPE tokenizer.",
    )
    parser.add_argument(
        "--num-row-groups",
        type=int,
        default=35,
        help="Number of parquet row groups to process (default: 35 for ~45M-50M wiki tokens).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of worker processes for parallel BPE tokenization (default: 12).",
    )
    parser.add_argument(
        "--val-split-pct",
        type=int,
        default=4,
        help="Percentage of clean articles for held-out validation (default: 4).",
    )
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=200,
        help="Minimum character count for an article to be included (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    config = CorpusBuilder27MConfig(
        parquet_path=args.parquet_path,
        gutenberg_cache_dir=args.gutenberg_cache_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        num_row_groups=args.num_row_groups,
        val_split_pct=args.val_split_pct,
        min_doc_chars=args.min_doc_chars,
        workers=args.workers,
    )

    builder = CorpusBuilder27M(config)
    result = builder.build()

    meta = result["cache_metadata"]
    print("\n" + "=" * 60)
    print(" GROUNDUPGPT-27M CORPUS PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output Directory:       {args.output_dir}")
    print(f"Total Unique Train Tok: {meta['train_tokens_count']:,}")
    print(f"Total Unique Val Tok:   {meta['val_tokens_count']:,}")
    print(f"Total Documents:        {meta['train_docs_count'] + meta['val_docs_count']:,}")
    print(f"Composition:")
    print(f"  - Project Gutenberg:  {meta['sources']['project_gutenberg']['train_tokens']:,} tokens ({meta['composition_pct']['gutenberg_pct']}%)")
    print(f"  - Simple Wikipedia:   {meta['sources']['simple_english_wikipedia']['train_tokens']:,} tokens ({meta['composition_pct']['wikipedia_pct']}%)")
    print(f"Total Processing Time:  {result['elapsed_s']:.1f} seconds")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

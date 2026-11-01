"""CLI script to prepare the expanded English corpus for GroundUpGPT (Step 23).

Usage:
    python scripts/prepare_expanded_corpus.py [--no-download]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.expanded_data_prep import prepare_expanded_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare expanded English pretraining corpus for GroundUpGPT")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt to download missing raw books")
    parser.add_argument("--raw-dir", default="data/raw_expanded", help="Path to raw books directory")
    parser.add_argument("--existing-raw-dir", default="data/raw", help="Path to original raw books directory")
    parser.add_argument("--output-dir", default="data/processed_expanded", help="Path for processed text files")
    parser.add_argument("--cache-dir", default="data/cache_expanded", help="Path for token cache")
    parser.add_argument("--tokenizer-path", default="checkpoints/bpe_vocab4096.json", help="Path to BPE tokenizer")
    args = parser.parse_args()

    manifest = prepare_expanded_corpus(
        raw_dir=Path(args.raw_dir),
        existing_raw_dir=Path(args.existing_raw_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        tokenizer_path=Path(args.tokenizer_path),
        download=not args.no_download,
    )

    stats = manifest["statistics"]
    print("=" * 60)
    print("EXPANDED CORPUS PREPARATION COMPLETE (STEP 23)")
    print("=" * 60)
    print(f"Total Documents:          {stats['total_documents']}")
    print(f"Training Documents:       {stats['total_train_documents']} ({stats['original_train_documents']} original + {stats['new_train_documents']} new)")
    print(f"Expanded Val Documents:   {stats['expanded_val_documents']}")
    print(f"Total Training Bytes:     {stats['train_bytes'] / 1024 / 1024:.2f} MB")
    print(f"Total Training Tokens:    {stats['train_tokens']:,}")
    print(f"Expanded Val Tokens:      {stats['val_tokens']:,}")
    print(f"Tokenization Speed:       {stats['train_tokens'] / max(0.1, stats['train_tokenization_seconds']):,.0f} tokens/s")
    print(f"Tokenizer Fingerprint:    {manifest['tokenizer_fingerprint']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

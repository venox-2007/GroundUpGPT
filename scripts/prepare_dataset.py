"""CLI tool for preparing the GroundUpGPT real English language modeling corpus.

Usage:
    python scripts/prepare_dataset.py [--raw-dir data/raw] [--output-dir data/processed] [--val-ratio 0.20] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_prep import GUTENBERG_CATALOG, prepare_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare GroundUpGPT English Language Modeling Corpus from Project Gutenberg."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Directory to store downloaded raw texts (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Directory to store processed train.txt, val.txt, and manifest.json (default: data/processed).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction of documents allocated to validation (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible document splitting (default: 42).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing books; use only existing raw files.",
    )

    args = parser.parse_args()

    print("=" * 68)
    print(" GroundUpGPT — Real Dataset Preparation Pipeline")
    print("=" * 68)
    print(f" Raw Directory:        {args.raw_dir}")
    print(f" Output Directory:     {args.output_dir}")
    print(f" Validation Ratio:     {args.val_ratio:.1%}")
    print(f" Random Seed:          {args.seed}")
    print(f" Catalog Documents:    {len(GUTENBERG_CATALOG)}")
    print("=" * 68)

    manifest = prepare_corpus(
        catalog=GUTENBERG_CATALOG,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=not args.no_download,
    )

    stats = manifest["statistics"]
    print("\n" + "=" * 68)
    print(" Corpus Preparation Succeeded!")
    print("=" * 68)
    print(f" Total Unique Documents:   {stats['total_documents']}")
    print(f" Training Documents:       {stats['train_documents']}")
    print(f" Validation Documents:     {stats['val_documents']}")
    print(f" Duplicates Removed:       {stats['duplicates_removed']}")
    print(f" Total Clean Characters:   {stats['total_characters']:,}")
    print(f" Total Clean Bytes:        {stats['total_bytes']:,} ({stats['total_bytes'] / (1024*1024):.2f} MB)")
    print(f" Training Bytes:           {stats['train_bytes']:,} ({stats['train_bytes'] / (1024*1024):.2f} MB)")
    print(f" Validation Bytes:         {stats['val_bytes']:,} ({stats['val_bytes'] / (1024*1024):.2f} MB)")
    print(f" Train File:               {stats['train_path']}")
    print(f" Val File:                 {stats['val_path']}")
    print("=" * 68)


if __name__ == "__main__":
    main()

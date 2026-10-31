"""CLI tool to train production 4096-vocab BPE tokenizer and build persistent token cache.

Usage:
    python scripts/build_tokenizer_and_cache.py [--target-vocab 4096]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer
from src.token_cache import build_token_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_tokenizer_and_cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit production BPE tokenizer and build pre-tokenized dataset cache."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train.txt",
        help="Path to training text file (default: data/processed/train.txt).",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "val.txt",
        help="Path to validation text file (default: data/processed/val.txt).",
    )
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json",
        help="Path to save fitted BPE tokenizer JSON (default: checkpoints/bpe_vocab4096.json).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache",
        help="Directory to save pre-tokenized cache arrays (default: data/cache).",
    )
    parser.add_argument(
        "--target-vocab",
        type=int,
        default=4096,
        help="Target BPE vocabulary size (default: 4096).",
    )

    args = parser.parse_args()

    print("=" * 68)
    print(" GroundUpGPT — Production BPE Tokenizer & Token Cache Pipeline")
    print("=" * 68)
    print(f" Train File:           {args.train_path}")
    print(f" Validation File:      {args.val_path}")
    print(f" Tokenizer Output:     {args.tokenizer_output}")
    print(f" Cache Directory:      {args.cache_dir}")
    print(f" Target Vocab Size:    {args.target_vocab}")
    print("=" * 68)

    if not args.train_path.is_file():
        logger.error(f"Training corpus not found at {args.train_path}. Run Step 14 first.")
        sys.exit(1)
    if not args.val_path.is_file():
        logger.error(f"Validation corpus not found at {args.val_path}. Run Step 14 first.")
        sys.exit(1)

    train_text = args.train_path.read_text(encoding="utf-8")
    train_bytes = len(train_text.encode("utf-8"))

    print(f"\n[*] Fitting BPE tokenizer on {train_bytes:,} training bytes...")
    t0_fit = time.perf_counter()
    tokenizer = BPETokenizer()
    tokenizer.fit(train_text, target_vocab_size=args.target_vocab)
    t1_fit = time.perf_counter()

    actual_vocab = tokenizer.vocab_size
    print(f"[*] BPE fitting complete in {t1_fit - t0_fit:.2f}s.")
    print(f"[*] Target vocab: {args.target_vocab} | Actual vocab: {actual_vocab}")

    if actual_vocab != args.target_vocab:
        logger.error(
            f"Failed to produce target vocabulary size {args.target_vocab}! "
            f"Produced only {actual_vocab} tokens. Halting."
        )
        sys.exit(1)

    # Validate Unicode roundtrip
    test_phrase = "GroundUpGPT: The quick brown fox jumps over the lazy dog. 🚀 12345! À bientôt & こんにちは!"
    encoded_test = tokenizer.encode(test_phrase)
    decoded_test = tokenizer.decode(encoded_test)
    assert decoded_test == test_phrase, f"Unicode roundtrip failure: {decoded_test!r} != {test_phrase!r}"
    print("[*] Unicode round-trip reconstruction verified.")

    # Save tokenizer JSON
    args.tokenizer_output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer_output)
    fingerprint = tokenizer.get_fingerprint()
    print(f"[*] Saved tokenizer to {args.tokenizer_output}")
    print(f"[*] Tokenizer fingerprint: {fingerprint}")

    # Build token cache
    print(f"\n[*] Pre-tokenizing training and validation datasets...")
    t0_cache = time.perf_counter()
    pipeline = build_token_cache(
        tokenizer=tokenizer,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.cache_dir,
        tokenizer_path_ref=str(args.tokenizer_output),
    )
    t1_cache = time.perf_counter()

    meta = pipeline.metadata
    print("\n" + "=" * 68)
    print(" Tokenizer & Cache Creation Succeeded!")
    print("=" * 68)
    print(f" Vocabulary Size:          {actual_vocab}")
    print(f" Merges Learned:           {len(tokenizer.merges):,}")
    print(f" Fingerprint:              {fingerprint}")
    print(f" Training Tokens:          {len(pipeline.train_tokens):,}")
    print(f" Validation Tokens:        {len(pipeline.val_tokens):,}")
    print(f" Train Compression Ratio:  {meta['compression_ratio_train']:.2f}x")
    print(f" Val Compression Ratio:    {meta['compression_ratio_val']:.2f}x")
    print(f" Fitting Time:             {t1_fit - t0_fit:.2f}s")
    print(f" Pre-tokenization Time:    {t1_cache - t0_cache:.2f}s")
    print(f" Cache Directory:          {args.cache_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()

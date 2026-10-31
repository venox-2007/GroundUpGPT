"""Demonstration of dataset loading, sequential splitting, and batch sampling."""

from pathlib import Path
import sys

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.data import TextDataPipeline


def main() -> None:
    print("=" * 60)
    print(" GroundUpGPT - Dataset Ingestion & Batching Pipeline Demo")
    print("=" * 60)

    fixture_path = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"
    print(f"\n[1] Loading Dataset from: {fixture_path.relative_to(REPO_ROOT)}")
    pipeline = TextDataPipeline.from_file(fixture_path, train_ratio=0.9)

    # 1. Dataset & Vocabulary Statistics
    print("\n[2] Dataset Statistics:")
    print(f"    Total Characters : {len(pipeline.raw_text)}")
    print(f"    Train Split Size : {len(pipeline.train_data)} tokens ({len(pipeline.train_text)} chars)")
    print(f"    Val Split Size   : {len(pipeline.val_data)} tokens ({len(pipeline.val_text)} chars)")
    print(f"    Vocabulary Size  : {pipeline.vocab_size} unique characters")

    # 2. Batch Sampling Configuration
    batch_size = 4
    block_size = 16
    print(f"\n[3] Sampling Batches (batch_size={batch_size}, block_size={block_size}):")

    # Fixed seed for reproducible demo output
    gen = torch.Generator().manual_seed(42)

    # Sample Training Batch
    x_train, y_train = pipeline.get_batch(
        split="train", batch_size=batch_size, block_size=block_size, generator=gen
    )
    print("    Training Batch:")
    print(f"      x shape: {list(x_train.shape)} (dtype={x_train.dtype})")
    print(f"      y shape: {list(y_train.shape)} (dtype={y_train.dtype})")

    # Sample Validation Batch
    x_val, y_val = pipeline.get_batch(
        split="val", batch_size=batch_size, block_size=block_size, generator=gen
    )
    print("    Validation Batch:")
    print(f"      x shape: {list(x_val.shape)} (dtype={x_val.dtype})")
    print(f"      y shape: {list(y_val.shape)} (dtype={y_val.dtype})")

    # 3. Decoded Sample Sequence Inspection
    print("\n[4] Decoded Input / Target Pair (Sample 0 of Train Batch):")
    sample_x = x_train[0].tolist()
    sample_y = y_train[0].tolist()

    decoded_x = pipeline.tokenizer.decode(sample_x)
    decoded_y = pipeline.tokenizer.decode(sample_y)

    print(f"    Input  (x) : {decoded_x!r}")
    print(f"    Target (y) : {decoded_y!r}")
    print("    Next-token alignment verification:")
    for t in range(min(5, block_size)):
        ch_in = pipeline.tokenizer.itos[sample_x[t]]
        ch_target = pipeline.tokenizer.itos[sample_y[t]]
        print(f"      step {t:02d}: {ch_in!r} -> target {ch_target!r}")

    print("\n" + "=" * 60)
    print(" Data Pipeline Demonstration Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

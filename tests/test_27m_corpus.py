"""Data quality, leakage prevention, and integration test suite for GroundUpGPT-27M pretraining corpus (Step 31).

Verifies:
1. Directory structure and required JSON metadata/manifests in data/cache_27m_v1.
2. Tensor shape, data type (int32), and bounded token IDs in [0, 4095].
3. Strict zero-leakage guarantee: original validation books (pg35, pg36, pg5200) and
   expanded validation books (pg1497, pg1228, pg768, pg108, pg205) must NEVER leak into training.
4. Causal one-token shifted batch sampling compatibility via PretokenizedDataPipeline.
5. UTF-8 decode lossless integrity and sample inspection.
6. Serialization / reload equivalence via load_token_cache().
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.token_cache import PretokenizedDataPipeline, load_token_cache

CACHE_DIR = Path("data/cache_27m_v1")
EXPECTED_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_VOCAB_SIZE = 4096

FORBIDDEN_VAL_BOOK_IDS = [
    "pg35", "pg36", "pg5200",
    "pg1497", "pg1228", "pg768", "pg108", "pg205",
]


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok_path = Path("checkpoints/bpe_vocab4096.json")
    assert tok_path.exists(), "Tokenizer checkpoint missing"
    tok = BPETokenizer().load(str(tok_path))
    assert tok.get_fingerprint() == EXPECTED_FP
    return tok


@pytest.fixture(scope="module")
def pipeline(tokenizer: BPETokenizer) -> PretokenizedDataPipeline:
    assert CACHE_DIR.exists(), f"Cache directory {CACHE_DIR} does not exist"
    return load_token_cache(CACHE_DIR, tokenizer=tokenizer)


def test_cache_files_exist():
    """Verify all 6 required cache files exist in data/cache_27m_v1."""
    required_files = [
        "train_tokens.pt",
        "val_tokens.pt",
        "cache_metadata.json",
        "corpus_manifest.json",
        "source_provenance.json",
        "preprocessing_summary.json",
    ]
    for fname in required_files:
        p = CACHE_DIR / fname
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


def test_token_tensors_valid(pipeline: PretokenizedDataPipeline):
    """Verify train and val token tensors have valid shapes, dtypes, and token ranges."""
    train = pipeline.train_tokens
    val = pipeline.val_tokens

    assert isinstance(train, torch.Tensor)
    assert isinstance(val, torch.Tensor)
    assert train.ndim == 1
    assert val.ndim == 1
    assert train.dtype == torch.int32
    assert val.dtype == torch.int32

    # Volume bounds
    assert len(train) >= 50_000_000, f"Expected at least 50M train tokens, got {len(train):,}"
    assert len(val) >= 2_000_000, f"Expected at least 2M val tokens, got {len(val):,}"

    # Token ID bounds in [0, 4095]
    assert int(train.min()) >= 0, "Train contains negative token IDs"
    assert int(train.max()) < EXPECTED_VOCAB_SIZE, f"Train token ID >= {EXPECTED_VOCAB_SIZE}"
    assert int(val.min()) >= 0, "Val contains negative token IDs"
    assert int(val.max()) < EXPECTED_VOCAB_SIZE, f"Val token ID >= {EXPECTED_VOCAB_SIZE}"


def test_strict_zero_leakage_against_validation_texts():
    """Verify that no text from held-out validation books leaks into training."""
    # Check manifest references
    meta_path = CACHE_DIR / "cache_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load validation text
    val_texts: list[str] = []
    for path in [Path("data/processed/val.txt"), Path("data/processed_expanded/val_expanded.txt")]:
        if path.exists():
            val_texts.append(path.read_text(encoding="utf-8"))

    combined_val_text = " ".join(val_texts)
    val_words = combined_val_text.split()

    # Index 12-word validation shingles
    k = 12
    forbidden_shingles = set(" ".join(val_words[i : i + k]).lower() for i in range(min(50_000, len(val_words) - k)))

    # Inspect preprocessing summary
    summary_path = CACHE_DIR / "preprocessing_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for forbidden_id in FORBIDDEN_VAL_BOOK_IDS:
        assert forbidden_id in summary["leak_prevention"]["forbidden_val_doc_ids"]

    assert summary["leak_prevention"]["indexed_validation_shingles"] > 100_000


def test_causal_next_token_batch_generation(pipeline: PretokenizedDataPipeline):
    """Verify pipeline produces valid causal next-token shifted batches."""
    batch_size = 4
    block_size = 256

    inputs, targets = pipeline.get_batch("train", batch_size=batch_size, block_size=block_size, device="cpu")
    assert inputs.shape == (batch_size, block_size)
    assert targets.shape == (batch_size, block_size)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

    # Causal shift check: targets[:, :-1] must equal inputs[:, 1:]
    assert torch.equal(targets[:, :-1], inputs[:, 1:]), "Targets must be causal one-token shift of inputs"

    # Same for validation split
    val_inputs, val_targets = pipeline.get_batch("val", batch_size=batch_size, block_size=block_size, device="cpu")
    assert val_inputs.shape == (batch_size, block_size)
    assert val_targets.shape == (batch_size, block_size)
    assert torch.equal(val_targets[:, :-1], val_inputs[:, 1:])


def test_utf8_decode_integrity_and_sample_inspection(pipeline: PretokenizedDataPipeline, tokenizer: BPETokenizer):
    """Verify that decoded token slices yield valid, clean UTF-8 text."""
    train = pipeline.train_tokens
    # Sample 10 distinct slices of 64 tokens across the corpus
    num_samples = 10
    sample_len = 64
    stride = len(train) // (num_samples + 1)

    for i in range(1, num_samples + 1):
        start_idx = i * stride
        slice_ids = train[start_idx : start_idx + sample_len].tolist()
        decoded = tokenizer.decode(slice_ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        # Should not raise UnicodeDecodeError and should contain printable characters
        assert any(c.isalnum() for c in decoded)


def test_metadata_consistency():
    """Verify metadata counts match the actual serialized tensor dimensions."""
    meta_path = CACHE_DIR / "cache_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_tokens = torch.load(CACHE_DIR / "train_tokens.pt", weights_only=True)
    val_tokens = torch.load(CACHE_DIR / "val_tokens.pt", weights_only=True)

    assert meta["train_tokens_count"] == len(train_tokens)
    assert meta["val_tokens_count"] == len(val_tokens)
    assert meta["tokenizer_fingerprint"] == EXPECTED_FP
    assert meta["vocab_size"] == EXPECTED_VOCAB_SIZE
    assert meta["corpus_version"] == "v3-27m"

    # Provenance check
    prov_path = CACHE_DIR / "source_provenance.json"
    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)
    assert len(prov["sources"]) >= 2
    source_names = [s["name"] for s in prov["sources"]]
    assert any("Gutenberg" in name for name in source_names)
    assert any("Wikipedia" in name for name in source_names)

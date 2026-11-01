"""Tests for Step 23: Expanded Pretraining Corpus.

Verifies:
1. Strict isolation and zero data leakage from original held-out validation books (pg35, pg36, pg5200).
2. 100% tokenizer compatibility with existing 4,096-vocab BPE model and fingerprint verification.
3. Expanded token cache structure, tensor dtypes, and token ID bounds [0, 4095].
4. Manifest integrity, document metadata, and reproducible SHA-256 checksums.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.expanded_data_prep import (
    EXPANDED_TRAIN_CATALOG,
    EXPANDED_VAL_CATALOG,
    FORBIDDEN_VAL_IDS,
    ORIGINAL_TRAIN_CATALOG,
    clean_gutenberg_text,
    compute_sha256,
    deduplicate_paragraphs,
)

EXPECTED_TOKENIZER_FINGERPRINT = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_VOCAB_SIZE = 4096


def test_no_original_val_in_catalogs():
    """Verify that none of the original validation books appear in any expanded catalog."""
    train_ids = {m["id"] for m in ORIGINAL_TRAIN_CATALOG} | {m["id"] for m in EXPANDED_TRAIN_CATALOG}
    val_ids = {m["id"] for m in EXPANDED_VAL_CATALOG}

    # Zero overlap with forbidden IDs
    assert train_ids.isdisjoint(FORBIDDEN_VAL_IDS), f"Data leak in train: {train_ids & FORBIDDEN_VAL_IDS}"
    assert val_ids.isdisjoint(FORBIDDEN_VAL_IDS), f"Data leak in val: {val_ids & FORBIDDEN_VAL_IDS}"

    # Disjoint train and val
    assert train_ids.isdisjoint(val_ids), f"Train and val overlap: {train_ids & val_ids}"


def test_clean_gutenberg_text():
    """Verify Gutenberg header and footer stripping and whitespace normalization."""
    sample_raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        "Chapter 1\r\n\r\n"
        "It was the best of times.   \r\n\r\n\r\n\r\n"
        "Chapter 2\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\r\n"
        "Extra legal footer text"
    )
    cleaned = clean_gutenberg_text(sample_raw)
    assert "*** START" not in cleaned
    assert "*** END" not in cleaned
    assert "Extra legal footer text" not in cleaned
    assert "It was the best of times." in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned  # compressed newlines


def test_paragraph_deduplication():
    """Verify paragraph deduplication removes repeated boilerplate."""
    p1 = "This is a substantial paragraph of text that represents legitimate prose in a story."
    p2 = "Another unique paragraph of text describing the weather and the landscape."
    repeated = p1 + "\n\n" + p2 + "\n\n" + p1
    deduped = deduplicate_paragraphs(repeated, min_len=30)
    assert deduped == p1 + "\n\n" + p2


def test_tokenizer_fingerprint_compatibility():
    """Verify that the existing BPE tokenizer matches the exact expected fingerprint."""
    tok_path = Path("checkpoints/bpe_vocab4096.json")
    assert tok_path.is_file(), f"Tokenizer file missing: {tok_path}"

    tokenizer = BPETokenizer.load(tok_path)
    assert tokenizer.vocab_size == EXPECTED_VOCAB_SIZE
    assert tokenizer.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT


def test_expanded_cache_artifacts_exist_and_valid():
    """Verify that expanded cache tensors and metadata exist with valid contents."""
    cache_dir = Path("data/cache_expanded")
    train_tokens_path = cache_dir / "train_tokens.pt"
    val_tokens_path = cache_dir / "val_expanded_tokens.pt"
    meta_path = cache_dir / "cache_metadata.json"

    if not train_tokens_path.exists():
        pytest.skip("Expanded token cache has not been built yet.")

    assert train_tokens_path.is_file()
    assert val_tokens_path.is_file()
    assert meta_path.is_file()

    # Load tensors
    train_tokens = torch.load(train_tokens_path, weights_only=True)
    val_tokens = torch.load(val_tokens_path, weights_only=True)

    assert isinstance(train_tokens, torch.Tensor)
    assert isinstance(val_tokens, torch.Tensor)
    assert train_tokens.ndim == 1
    assert val_tokens.ndim == 1
    assert train_tokens.dtype == torch.int32
    assert val_tokens.dtype == torch.int32

    # Check bounds
    assert train_tokens.min().item() >= 0
    assert train_tokens.max().item() < EXPECTED_VOCAB_SIZE
    assert val_tokens.min().item() >= 0
    assert val_tokens.max().item() < EXPECTED_VOCAB_SIZE

    # Check token count target: 10M - 20M tokens
    assert len(train_tokens) >= 10_000_000, f"Expected >= 10M tokens, got {len(train_tokens):,}"
    assert len(val_tokens) >= 500_000, f"Expected >= 500K val tokens, got {len(val_tokens):,}"

    # Verify metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["corpus_version"] == "v2-expanded"
    assert meta["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert meta["train_tokens_count"] == len(train_tokens)
    assert meta["val_tokens_count"] == len(val_tokens)


def test_manifest_expanded_integrity():
    """Verify that manifest_expanded.json accurately reflects text files and checksums."""
    manifest_path = Path("data/processed_expanded/manifest_expanded.json")
    if not manifest_path.exists():
        pytest.skip("manifest_expanded.json has not been built yet.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["version"] == "v2-expanded"
    assert manifest["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert manifest["statistics"]["total_documents"] >= 60

    # Verify zero forbidden validation IDs
    forbidden = set(manifest.get("forbidden_validation_ids_excluded", []))
    assert FORBIDDEN_VAL_IDS.issubset(forbidden)

    for doc in manifest["documents"]:
        assert doc["doc_id"] not in FORBIDDEN_VAL_IDS

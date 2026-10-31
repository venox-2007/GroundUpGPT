"""Unit tests for GroundUpGPT token cache and pre-tokenized data pipeline."""

from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.token_cache import (
    PretokenizedDataPipeline,
    build_token_cache,
    load_token_cache,
)


@pytest.fixture
def sample_corpus(tmp_path: Path):
    """Create sample train and validation text files."""
    train_text = (
        "The quick brown fox jumps over the lazy dog. "
        "A journey of a thousand miles begins with a single step. "
        "GroundUpGPT Transformer architecture implementation from scratch. "
    ) * 30
    val_text = (
        "Alice was beginning to get very tired of sitting by her sister on the bank. "
        "Deep learning and language models from first principles. "
    ) * 10

    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    train_file.write_text(train_text, encoding="utf-8")
    val_file.write_text(val_text, encoding="utf-8")
    return train_file, val_file, train_text, val_text


def test_build_and_load_cache(tmp_path: Path, sample_corpus):
    train_file, val_file, train_text, val_text = sample_corpus
    cache_dir = tmp_path / "cache"

    # Fit BPE on train
    tok = BPETokenizer().fit(train_text, target_vocab_size=300)
    tok_file = tmp_path / "tok.json"
    tok.save(tok_file)

    pipeline = build_token_cache(
        tokenizer=tok,
        train_path=train_file,
        val_path=val_file,
        output_dir=cache_dir,
        tokenizer_path_ref=str(tok_file),
    )

    assert (cache_dir / "train_tokens.pt").is_file()
    assert (cache_dir / "val_tokens.pt").is_file()
    assert (cache_dir / "cache_metadata.json").is_file()

    # Verify reload
    loaded_pipeline = load_token_cache(cache_dir=cache_dir, tokenizer=tok)
    assert torch.equal(loaded_pipeline.train_tokens, pipeline.train_tokens)
    assert torch.equal(loaded_pipeline.val_tokens, pipeline.val_tokens)
    assert loaded_pipeline.vocab_size == 300


def test_fingerprint_mismatch_detection(tmp_path: Path, sample_corpus):
    train_file, val_file, train_text, _ = sample_corpus
    cache_dir = tmp_path / "cache"

    tok1 = BPETokenizer().fit(train_text, target_vocab_size=300)
    build_token_cache(
        tokenizer=tok1,
        train_path=train_file,
        val_path=val_file,
        output_dir=cache_dir,
    )

    # Different tokenizer
    tok2 = BPETokenizer().fit("completely different vocabulary corpus", target_vocab_size=300)

    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        load_token_cache(cache_dir=cache_dir, tokenizer=tok2)


def test_causal_target_shift_and_batch_shapes(tmp_path: Path, sample_corpus):
    train_file, val_file, train_text, _ = sample_corpus
    cache_dir = tmp_path / "cache"

    tok = BPETokenizer().fit(train_text, target_vocab_size=280)
    pipeline = build_token_cache(
        tokenizer=tok,
        train_path=train_file,
        val_path=val_file,
        output_dir=cache_dir,
    )

    gen = torch.Generator().manual_seed(42)
    B, T = 4, 16
    x, y = pipeline.get_batch("train", batch_size=B, block_size=T, generator=gen)

    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Verify causal 1-token shift: y[b, t] is the next token following x[b, t]
    for b in range(B):
        # The first T-1 tokens of y should match the last T-1 tokens of x
        assert torch.equal(x[b, 1:], y[b, :-1])


def test_token_id_bounds(tmp_path: Path, sample_corpus):
    train_file, val_file, train_text, _ = sample_corpus
    cache_dir = tmp_path / "cache"

    vocab_size = 290
    tok = BPETokenizer().fit(train_text, target_vocab_size=vocab_size)
    pipeline = build_token_cache(
        tokenizer=tok,
        train_path=train_file,
        val_path=val_file,
        output_dir=cache_dir,
    )

    assert pipeline.train_tokens.min().item() >= 0
    assert pipeline.train_tokens.max().item() < vocab_size
    assert pipeline.val_tokens.min().item() >= 0
    assert pipeline.val_tokens.max().item() < vocab_size


def test_unicode_reconstruction_with_cache(tmp_path: Path):
    text = "Unicode test: 🌍 Hello World 🚀! ∀x ∈ ℝ, x² ≥ 0. 👨‍💻"
    train_file = tmp_path / "u_train.txt"
    val_file = tmp_path / "u_val.txt"
    train_file.write_text(text * 10, encoding="utf-8")
    val_file.write_text(text * 2, encoding="utf-8")

    tok = BPETokenizer().fit(train_file.read_text("utf-8"), target_vocab_size=275)
    cache_dir = tmp_path / "u_cache"
    pipeline = build_token_cache(tok, train_file, val_file, cache_dir)

    # Decode tokens back to string
    decoded_train = tok.decode(pipeline.train_tokens.tolist())
    assert decoded_train == train_file.read_text("utf-8")

"""Targeted unit tests for Byte-level BPE tokenizer and data pipeline compatibility."""

from pathlib import Path
import pytest
from src.bpe_tokenizer import BPETokenizer
from src.tokenizer import CharTokenizer
from src.data import TextDataPipeline


def test_deterministic_merge_learning() -> None:
    """Ensure training produces identical merges regardless of run on identical data."""
    sample = "ab ab ab cd cd ef ef ab ab cd cd ef ef"
    tok1 = BPETokenizer().fit(sample, target_vocab_size=260)
    tok2 = BPETokenizer().fit(sample, target_vocab_size=260)

    assert tok1.merges == tok2.merges
    assert tok1.vocab == tok2.vocab
    assert tok1.vocab_size == tok2.vocab_size == 260


def test_exact_encode_decode_roundtrip() -> None:
    """Verify lossless round-trip reconstruction on standard text."""
    corpus = "The quick brown fox jumps over the lazy dog. 0123456789! \n\t"
    tokenizer = BPETokenizer().fit(corpus, target_vocab_size=270)
    encoded = tokenizer.encode(corpus)
    decoded = tokenizer.decode(encoded)

    assert decoded == corpus


def test_unicode_and_emoji_support() -> None:
    """Verify arbitrary multi-byte UTF-8 Unicode characters and emoji are decoded losslessly."""
    tokenizer = BPETokenizer()  # Unfitted, base 256 byte vocabulary
    test_strings = [
        "Hello 🌍 World 🚀!",
        "日本語と한국어와 Español & Français",
        "Mathematics: ∀x ∈ ℝ, x² ≥ 0",
        "Complex emoji sequences: 👨‍💻 🤖 ✨",
    ]
    for s in test_strings:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        assert decoded == s


def test_serialization_and_loading_equivalence(tmp_path: Path) -> None:
    """Ensure saving and reloading a tokenizer produces identical behavior and outputs."""
    corpus = "Transformer architecture from scratch. Decoder-only autoregressive language model."
    tokenizer = BPETokenizer().fit(corpus, target_vocab_size=275)

    save_file = tmp_path / "bpe_test.json"
    tokenizer.save(save_file)

    loaded_tokenizer = BPETokenizer.load(save_file)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.merges == tokenizer.merges

    test_sentence = "Transformer language model"
    assert tokenizer.encode(test_sentence) == loaded_tokenizer.encode(test_sentence)
    assert tokenizer.decode(tokenizer.encode(test_sentence)) == loaded_tokenizer.decode(loaded_tokenizer.encode(test_sentence))


def test_invalid_inputs_and_token_ids(tmp_path: Path) -> None:
    """Test validation errors for empty fit, out-of-range IDs, and malformed files."""
    tokenizer = BPETokenizer()

    with pytest.raises(ValueError, match="Cannot fit BPETokenizer on empty text"):
        tokenizer.fit("")

    with pytest.raises(ValueError, match="target_vocab_size must be at least 256"):
        tokenizer.fit("some text", target_vocab_size=100)

    # Decode with invalid token ID
    with pytest.raises(ValueError, match="Invalid token ID 999"):
        tokenizer.decode([0, 999, 1])

    # Load non-existent file
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "non_existent.json")

    # Load malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{\"type\": \"wrong\"}", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed tokenizer JSON"):
        BPETokenizer.load(bad_json)


def test_token_compression_on_repetitive_text() -> None:
    """Verify that learned merges reduce token count compared to raw bytes."""
    repetitive = "banana bandana " * 20
    raw_byte_count = len(repetitive.encode("utf-8"))

    tokenizer = BPETokenizer().fit(repetitive, target_vocab_size=270)
    bpe_tokens = tokenizer.encode(repetitive)

    assert len(bpe_tokens) < raw_byte_count
    assert tokenizer.decode(bpe_tokens) == repetitive


def test_data_pipeline_compatibility_with_bpe_and_char() -> None:
    """Verify TextDataPipeline supports both BPETokenizer and CharTokenizer seamlessly."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "GroundUpGPT Transformer architecture implementation from scratch. "
        "A modular, test-driven neural network pipeline in Python and PyTorch. "
    ) * 10

    # 1. Pipeline with default CharTokenizer
    pipeline_char = TextDataPipeline(text=text, train_ratio=0.8)
    assert isinstance(pipeline_char.tokenizer, CharTokenizer)
    x_c, y_c = pipeline_char.get_batch("train", batch_size=2, block_size=6)
    assert x_c.shape == (2, 6)

    # 2. Pipeline with BPETokenizer
    bpe_tok = BPETokenizer()
    pipeline_bpe = TextDataPipeline(text=text, train_ratio=0.8, tokenizer=bpe_tok)
    assert isinstance(pipeline_bpe.tokenizer, BPETokenizer)
    assert pipeline_bpe.tokenizer.vocab_size >= 256
    x_b, y_b = pipeline_bpe.get_batch("train", batch_size=2, block_size=6)
    assert x_b.shape == (2, 6)
    assert y_b.shape == (2, 6)

    # 3. BPE handles unseen validation text without error
    # Train has 'abc', Val has 'xyz' -> CharTokenizer fails, but BPETokenizer succeeds via byte vocab!
    split_text = ("abc " * 50) + ("xyz " * 10)
    bpe_tok2 = BPETokenizer().fit("abc " * 50, target_vocab_size=260)
    pipeline_bpe_unseen = TextDataPipeline(text=split_text, train_ratio=0.8, tokenizer=bpe_tok2)
    assert len(pipeline_bpe_unseen.val_tokens) > 0

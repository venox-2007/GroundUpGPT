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
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        BPETokenizer.load(bad_json)


def test_token_compression_on_repetitive_text() -> None:
    """Verify that learned merges reduce token count compared to raw bytes."""
    repetitive = "banana bandana " * 20
    raw_byte_count = len(repetitive.encode("utf-8"))

    tokenizer = BPETokenizer().fit(repetitive, target_vocab_size=270)
    bpe_tokens = tokenizer.encode(repetitive)

    assert len(bpe_tokens) < raw_byte_count
    assert tokenizer.decode(bpe_tokens) == repetitive


def test_strict_utf8_decode_failure() -> None:
    """Ensure invalid or truncated UTF-8 byte sequences raise ValueError during decode."""
    tokenizer = BPETokenizer()
    # 0xFF is an invalid single-byte start in UTF-8
    with pytest.raises(ValueError, match="Failed to decode byte sequence into valid UTF-8 string"):
        tokenizer.decode([0xFF])

    # 0xC3 alone is an incomplete 2-byte prefix in UTF-8
    with pytest.raises(ValueError, match="Failed to decode byte sequence into valid UTF-8 string"):
        tokenizer.decode([0xC3])


def test_harden_serialization_rejects_corrupt_files(tmp_path: Path) -> None:
    """Ensure BPETokenizer.load() rejects duplicate merges, ranks, or corrupted metadata."""
    import json

    # 1. Inconsistent declared vocab_size
    bad_vocab = tmp_path / "bad_vocab.json"
    bad_vocab.write_text(
        json.dumps({"version": "1.0", "type": "byte_bpe", "vocab_size": 260, "merges": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Inconsistent declared vocab_size"):
        BPETokenizer.load(bad_vocab)

    # 2. Overwriting base byte token (new_id < 256)
    bad_id = tmp_path / "bad_id.json"
    bad_id.write_text(
        json.dumps({
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": 257,
            "merges": [[97, 98, 200, 0]],  # new_id 200 < 256
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cannot overwrite base byte token ID"):
        BPETokenizer.load(bad_id)

    # 3. Duplicate merge rank
    bad_rank = tmp_path / "bad_rank.json"
    bad_rank.write_text(
        json.dumps({
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": 258,
            "merges": [[97, 98, 256, 0], [99, 100, 257, 0]],  # duplicate rank 0
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate merge rank detected"):
        BPETokenizer.load(bad_rank)

    # 4. Duplicate merge pair
    bad_pair = tmp_path / "bad_pair.json"
    bad_pair.write_text(
        json.dumps({
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": 258,
            "merges": [[97, 98, 256, 0], [97, 98, 257, 1]],  # duplicate pair (97, 98)
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate merge pair detected"):
        BPETokenizer.load(bad_pair)

    # 5. Future/unavailable token ID reference
    bad_ref = tmp_path / "bad_ref.json"
    bad_ref.write_text(
        json.dumps({
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": 257,
            "merges": [[97, 999, 256, 0]],  # token 999 does not exist
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Merge references unavailable or future token IDs"):
        BPETokenizer.load(bad_ref)


def test_configurable_bpe_vocab_size_in_pipeline() -> None:
    """Ensure TextDataPipeline respects explicit bpe_vocab_size parameter."""
    text = (
        "GroundUpGPT Transformer architecture implementation from scratch. "
        "A modular, test-driven neural network pipeline in Python and PyTorch. "
    ) * 10

    # Explicit 270 vocab size
    pipeline = TextDataPipeline(
        text=text,
        train_ratio=0.8,
        tokenizer=BPETokenizer(),
        bpe_vocab_size=270,
    )
    assert pipeline.tokenizer.vocab_size == 270

    # Invalid bpe_vocab_size < 256
    with pytest.raises(ValueError, match="bpe_vocab_size must be at least 256"):
        TextDataPipeline(
            text=text,
            train_ratio=0.8,
            tokenizer=BPETokenizer(),
            bpe_vocab_size=200,
        )


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

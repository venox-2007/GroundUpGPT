"""Focused unit tests for the character-level tokenizer."""

import pytest
from src.tokenizer import CharTokenizer


def test_fit_and_vocab_size() -> None:
    """Test vocabulary generation and size calculation."""
    tokenizer = CharTokenizer()
    tokenizer.fit("abc")
    assert tokenizer.vocab_size == 3
    assert tokenizer.stoi == {"a": 0, "b": 1, "c": 2}
    assert tokenizer.itos == {0: "a", 1: "b", 2: "c"}


def test_deterministic_vocabulary_order() -> None:
    """Ensure vocabulary ordering is strictly deterministic (sorted unique characters)."""
    tokenizer1 = CharTokenizer().fit("zyxwvuts")
    tokenizer2 = CharTokenizer().fit("stuvwxyz")
    assert tokenizer1.stoi == tokenizer2.stoi
    assert tokenizer1.itos == tokenizer2.itos
    assert list(tokenizer1.stoi.keys()) == sorted("stuvwxyz")


def test_repeated_characters() -> None:
    """Verify duplicate characters in training text do not duplicate vocabulary items."""
    tokenizer = CharTokenizer().fit("aaaaabbbbcccc")
    assert tokenizer.vocab_size == 3
    assert tokenizer.encode("abc") == [0, 1, 2]


def test_encode_and_decode_roundtrip() -> None:
    """Verify round-trip encoding and decoding preserves original text perfectly."""
    sample = "The quick brown fox jumps over the lazy dog. 12345!"
    tokenizer = CharTokenizer().fit(sample)
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample
    assert len(encoded) == len(sample)


def test_empty_training_text_raises_error() -> None:
    """Fitting on empty text must raise a clear ValueError."""
    tokenizer = CharTokenizer()
    with pytest.raises(ValueError, match="Cannot fit tokenizer on empty text"):
        tokenizer.fit("")


def test_unknown_character_raises_error() -> None:
    """Encoding characters not seen during fit must raise a ValueError with position info."""
    tokenizer = CharTokenizer().fit("hello")
    with pytest.raises(ValueError, match="Unknown character 'x' at position 2"):
        tokenizer.encode("hexlo")


def test_invalid_token_id_raises_error() -> None:
    """Decoding token IDs outside the vocabulary must raise a ValueError."""
    tokenizer = CharTokenizer().fit("abc")  # vocab IDs: 0, 1, 2
    with pytest.raises(ValueError, match="Invalid token ID 99 at index 1"):
        tokenizer.decode([0, 99, 2])


def test_unfitted_tokenizer_raises_error() -> None:
    """Attempting to access vocab_size, encode, or decode before fit must raise ValueError."""
    tokenizer = CharTokenizer()
    with pytest.raises(ValueError, match="Tokenizer has not been fitted yet"):
        _ = tokenizer.vocab_size

    with pytest.raises(ValueError, match="Tokenizer has not been fitted yet"):
        tokenizer.encode("test")

    with pytest.raises(ValueError, match="Tokenizer has not been fitted yet"):
        tokenizer.decode([0, 1])

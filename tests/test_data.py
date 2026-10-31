"""Targeted unit tests for dataset ingestion and batch sampling pipeline."""

from pathlib import Path
import pytest
import torch
from src.data import TextDataPipeline


@pytest.fixture
def sample_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_text.txt"


def test_split_proportions_and_boundary_isolation(sample_fixture_path: Path) -> None:
    """Ensure data is strictly split according to ratio with non-empty splits."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.8)

    total_len = len(pipeline.raw_text)
    expected_train_len = int(total_len * 0.8)
    expected_val_len = total_len - expected_train_len

    assert len(pipeline.train_text) == expected_train_len
    assert len(pipeline.val_text) == expected_val_len
    assert len(pipeline.train_data) == expected_train_len
    assert len(pipeline.val_data) == expected_val_len

    # Boundary verification: concatenation must perfectly reconstruct original text
    assert pipeline.train_text + pipeline.val_text == pipeline.raw_text


def test_batch_tensor_shapes_and_types(sample_fixture_path: Path) -> None:
    """Ensure sampled batches have expected (batch_size, block_size) shapes and int64 dtype."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    batch_size = 4
    block_size = 12

    x_train, y_train = pipeline.get_batch(split="train", batch_size=batch_size, block_size=block_size)
    assert x_train.shape == (batch_size, block_size)
    assert y_train.shape == (batch_size, block_size)
    assert x_train.dtype == torch.long
    assert y_train.dtype == torch.long

    x_val, y_val = pipeline.get_batch(split="val", batch_size=batch_size, block_size=block_size)
    assert x_val.shape == (batch_size, block_size)
    assert y_val.shape == (batch_size, block_size)


def test_next_token_target_alignment(sample_fixture_path: Path) -> None:
    """Verify target y is exactly shifted by +1 token relative to input x."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)
    block_size = 8

    x, y = pipeline.get_batch(split="train", batch_size=2, block_size=block_size)

    # For each sequence b in batch, x[b, 1:] must equal y[b, :-1]
    for b in range(x.shape[0]):
        assert torch.equal(x[b, 1:], y[b, :-1]), "Target y must be shifted by exactly 1 token"


def test_deterministic_sampling_with_generator(sample_fixture_path: Path) -> None:
    """Ensure fixed generator seed produces identical batches."""
    pipeline = TextDataPipeline.from_file(sample_fixture_path, train_ratio=0.9)

    gen1 = torch.Generator().manual_seed(1337)
    gen2 = torch.Generator().manual_seed(1337)

    x1, y1 = pipeline.get_batch("train", batch_size=3, block_size=10, generator=gen1)
    x2, y2 = pipeline.get_batch("train", batch_size=3, block_size=10, generator=gen2)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_insufficient_sequence_length_raises_error() -> None:
    """Requesting block_size >= split length must raise ValueError."""
    short_text = "abcde" * 5  # 25 characters
    pipeline = TextDataPipeline(text=short_text, train_ratio=0.8)  # train=20, val=5

    # Requesting block_size=6 on val split (len=5) should fail
    with pytest.raises(ValueError, match="Insufficient data in 'val' split"):
        pipeline.get_batch(split="val", batch_size=1, block_size=6)


def test_unknown_validation_character_raises_explicit_error() -> None:
    """If validation contains characters not in train text, must raise clear ValueError."""
    # 'Z' only appears in the validation slice (last 10%)
    train_part = "abcdefghij" * 9  # 90 chars
    val_part = "abcdefghjZ"        # 10 chars, contains 'Z'
    text = train_part + val_part

    with pytest.raises(ValueError, match="Validation split contains characters unseen in training vocabulary"):
        TextDataPipeline(text=text, train_ratio=0.9)


def test_invalid_parameters_raise_errors() -> None:
    """Verify input validation for invalid split ratios, empty files, or invalid arguments."""
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        TextDataPipeline(text="   ")

    with pytest.raises(ValueError, match="train_ratio must be in range"):
        TextDataPipeline(text="hello world", train_ratio=1.5)

    valid_repeated_text = "valid ratios sample text " * 3
    pipeline = TextDataPipeline(text=valid_repeated_text, train_ratio=0.8)
    with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
        pipeline.get_batch(split="test")

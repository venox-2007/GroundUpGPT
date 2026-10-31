"""Dataset loading, splitting, and mini-batch sampling pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import torch
from src.tokenizer import CharTokenizer
from src.bpe_tokenizer import BPETokenizer

TokenizerType = Union[CharTokenizer, BPETokenizer]


class TextDataPipeline:
    """Pipeline for text ingestion, sequential splitting, and mini-batch sampling.

    Splits a corpus into sequential training and validation subsets, fits or applies
    a tokenizer strictly on training data, and provides random mini-batch sampling
    with causal next-token shifting.

    Args:
        text: Optional raw text string.
        file_path: Optional path to a UTF-8 text file.
        train_ratio: Sequential split fraction for training (default: 0.9).
        tokenizer: Optional pre-fitted tokenizer instance. If None, a new
                   CharTokenizer will be fitted exclusively on the training split.

    Raises:
        ValueError: If inputs are invalid, files are empty, split ratios are out
                    of range, or validation text contains unseen characters.
        FileNotFoundError: If file_path does not exist.
    """

    def __init__(
        self,
        text: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.9,
        tokenizer: Optional[TokenizerType] = None,
    ) -> None:
        if text is None and file_path is None:
            raise ValueError("Must provide either 'text' or 'file_path'.")

        if file_path is not None:
            path = Path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            loaded_text = path.read_text(encoding="utf-8")
            if text is None:
                text = loaded_text

        assert text is not None
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty or solely whitespace.")

        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in range (0.0, 1.0), got {train_ratio}")

        self.raw_text: str = text
        self.train_ratio: float = train_ratio

        # 1. Sequential Splitting
        total_len = len(text)
        split_idx = int(total_len * train_ratio)
        if split_idx <= 0 or split_idx >= total_len:
            raise ValueError(
                f"Text length ({total_len}) insufficient for train_ratio={train_ratio} "
                "to form both non-empty train and validation splits."
            )

        self.train_text: str = text[:split_idx]
        self.val_text: str = text[split_idx:]

        # 2. Tokenizer Fitting (Train split only)
        if tokenizer is None:
            self.tokenizer: TokenizerType = CharTokenizer().fit(self.train_text)
        else:
            self.tokenizer = tokenizer
            # If an unfitted tokenizer instance was supplied, fit it strictly on training text
            if not getattr(self.tokenizer, "_is_fitted", False):
                if isinstance(self.tokenizer, BPETokenizer):
                    # For small corpus, scale merges defensively to prevent excessive token collapse
                    max_merges = max(2, min(32, len(self.train_text) // 10))
                    self.tokenizer.fit(self.train_text, target_vocab_size=256 + max_merges)
                else:
                    self.tokenizer.fit(self.train_text)

        # 3. Token Encoding with strict validation
        self.train_tokens: list[int] = self.tokenizer.encode(self.train_text)

        try:
            self.val_tokens: list[int] = self.tokenizer.encode(self.val_text)
        except ValueError as err:
            raise ValueError(
                f"Validation split contains characters unseen in training vocabulary: {err}"
            ) from err

        # 4. Storage as 1D PyTorch Long Tensors
        self.train_data: torch.Tensor = torch.tensor(self.train_tokens, dtype=torch.long)
        self.val_data: torch.Tensor = torch.tensor(self.val_tokens, dtype=torch.long)

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        train_ratio: float = 0.9,
        tokenizer: Optional[TokenizerType] = None,
    ) -> TextDataPipeline:
        """Convenience constructor to initialize pipeline directly from a file."""
        return cls(file_path=file_path, train_ratio=train_ratio, tokenizer=tokenizer)

    @property
    def vocab_size(self) -> int:
        """Return the size of the underlying tokenizer vocabulary."""
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 4,
        block_size: int = 8,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of input sequences (x) and target sequences (y) shifted by 1 token.

        Sequences are sampled strictly from within the chosen split so no sequence
        ever crosses split boundaries.

        Args:
            split: 'train' or 'val'.
            batch_size: Number of independent sequences to sample.
            block_size: Context length (number of tokens per sequence).
            device: Torch device destination ('cpu' or 'cuda').
            generator: Optional torch.Generator for deterministic seeding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x: Input tensor of shape (batch_size, block_size).
                - y: Target tensor of shape (batch_size, block_size).

        Raises:
            ValueError: If split is invalid, batch_size/block_size <= 0, or split
                        data length is insufficient for block_size.
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        data = self.train_data if split == "train" else self.val_data

        if len(data) <= block_size:
            raise ValueError(
                f"Insufficient data in '{split}' split ({len(data)} tokens) for "
                f"block_size={block_size}. Need at least {block_size + 1} tokens."
            )

        # Sample starting indices such that [i : i + block_size + 1] is within bounds
        max_start_idx = len(data) - block_size
        ix = torch.randint(0, max_start_idx, (batch_size,), generator=generator)

        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)

        return x, y

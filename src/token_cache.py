"""Persistent pre-tokenized dataset cache and sampling pipeline for GroundUpGPT.

Pre-tokenizes text corpora into compact integer token arrays, avoids repeated full-corpus
tokenization during training, and provides causal one-token-shifted batch generation compatible
with the Trainer interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch

from src.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenCacheMetadata:
    """Metadata describing a pre-tokenized dataset cache."""

    tokenizer_path: str
    tokenizer_fingerprint: str
    vocab_size: int
    train_tokens_count: int
    val_tokens_count: int
    train_bytes: int
    val_bytes: int
    compression_ratio_train: float
    compression_ratio_val: float
    created_at: str


class PretokenizedDataPipeline:
    """High-performance pre-tokenized data pipeline compatible with Trainer.

    Holds pre-tokenized integer token tensors in memory, offering constant-time
    random mini-batch extraction with causal one-token-shifted target offsets.

    Attributes:
        train_tokens (torch.Tensor): 1D tensor of training token IDs.
        val_tokens (torch.Tensor): 1D tensor of validation token IDs.
        tokenizer (BPETokenizer): Associated BPE tokenizer instance.
    """

    def __init__(
        self,
        train_tokens: torch.Tensor,
        val_tokens: torch.Tensor,
        tokenizer: BPETokenizer,
        metadata: Optional[Dict] = None,
    ) -> None:
        if not isinstance(train_tokens, torch.Tensor) or train_tokens.ndim != 1:
            raise ValueError("train_tokens must be a 1D torch.Tensor.")
        if not isinstance(val_tokens, torch.Tensor) or val_tokens.ndim != 1:
            raise ValueError("val_tokens must be a 1D torch.Tensor.")
        if len(train_tokens) == 0:
            raise ValueError("train_tokens cannot be empty.")
        if len(val_tokens) == 0:
            raise ValueError("val_tokens cannot be empty.")

        self.train_tokens: torch.Tensor = train_tokens
        self.val_tokens: torch.Tensor = val_tokens
        self.tokenizer: BPETokenizer = tokenizer
        self.metadata: Optional[Dict] = metadata

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: Optional[Union[str, torch.device]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a reproducible mini-batch with causal one-token-shifted targets.

        Given a sequence length T (block_size), for each random offset i:
            x = tokens[i : i + T]
            y = tokens[i + 1 : i + 1 + T]  # Exactly 1-token right-shifted target

        Args:
            split: 'train' or 'val'.
            batch_size: Number of sequences per batch (B).
            block_size: Context length per sequence (T).
            device: Target torch device (e.g. 'cuda', 'cpu').
            generator: Optional torch.Generator for deterministic reproducibility.

        Returns:
            Tuple of (x, y) tensors, each shaped (B, T) with torch.long dtype.
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        tokens = self.train_tokens if split == "train" else self.val_tokens

        if len(tokens) <= block_size:
            raise ValueError(
                f"Split '{split}' has {len(tokens)} tokens, which is insufficient "
                f"for block_size {block_size}. Need at least {block_size + 1} tokens."
            )

        # Sample random start indices in [0, len(tokens) - block_size - 1]
        max_idx = len(tokens) - block_size
        ix = torch.randint(0, max_idx, (batch_size,), generator=generator)

        x = torch.stack([tokens[i : i + block_size] for i in ix]).to(dtype=torch.long)
        y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in ix]).to(dtype=torch.long)

        if device is not None:
            x = x.to(device=device, non_blocking=True if str(device).startswith("cuda") else False)
            y = y.to(device=device, non_blocking=True if str(device).startswith("cuda") else False)

        return x, y


def build_token_cache(
    tokenizer: BPETokenizer,
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_path_ref: Optional[str] = None,
) -> PretokenizedDataPipeline:
    """Pre-tokenize training and validation corpora and save persistent binary artifacts.

    Args:
        tokenizer: Fitted BPETokenizer instance.
        train_path: Path to clean train.txt.
        val_path: Path to clean val.txt.
        output_dir: Directory where token cache will be stored (outside Git).
        tokenizer_path_ref: Optional path string to recorded tokenizer artifact.

    Returns:
        PretokenizedDataPipeline ready for model training.
    """
    train_file = Path(train_path)
    val_file = Path(val_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.is_file():
        raise FileNotFoundError(f"Train text file not found: {train_file}")
    if not val_file.is_file():
        raise FileNotFoundError(f"Validation text file not found: {val_file}")

    train_text = train_file.read_text(encoding="utf-8")
    val_text = val_file.read_text(encoding="utf-8")

    train_raw_bytes = len(train_text.encode("utf-8"))
    val_raw_bytes = len(val_text.encode("utf-8"))

    logger.info(f"Tokenizing training corpus ({train_raw_bytes:,} bytes)...")
    train_ids = tokenizer.encode(train_text)

    logger.info(f"Tokenizing validation corpus ({val_raw_bytes:,} bytes)...")
    val_ids = tokenizer.encode(val_text)

    # Validate token ID bounds
    vocab_size = tokenizer.vocab_size
    for split_name, ids in [("train", train_ids), ("val", val_ids)]:
        if ids:
            min_id = min(ids)
            max_id = max(ids)
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(
                    f"Token ID out of bounds in {split_name} split: min={min_id}, max={max_id}, "
                    f"valid range=[0, {vocab_size - 1}]."
                )

    # Convert to compact 32-bit signed integer tensors (supports up to 2.1 billion tokens)
    train_tensor = torch.tensor(train_ids, dtype=torch.int32)
    val_tensor = torch.tensor(val_ids, dtype=torch.int32)

    train_tensor_path = out_dir / "train_tokens.pt"
    val_tensor_path = out_dir / "val_tokens.pt"

    torch.save(train_tensor, train_tensor_path)
    torch.save(val_tensor, val_tensor_path)

    compression_train = train_raw_bytes / max(1, len(train_ids))
    compression_val = val_raw_bytes / max(1, len(val_ids))

    meta = TokenCacheMetadata(
        tokenizer_path=tokenizer_path_ref or "unknown",
        tokenizer_fingerprint=tokenizer.get_fingerprint(),
        vocab_size=vocab_size,
        train_tokens_count=len(train_ids),
        val_tokens_count=len(val_ids),
        train_bytes=train_raw_bytes,
        val_bytes=val_raw_bytes,
        compression_ratio_train=round(compression_train, 2),
        compression_ratio_val=round(compression_val, 2),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    meta_path = out_dir / "cache_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    logger.info(
        f"Token cache created in {out_dir}: {len(train_ids):,} train tokens, "
        f"{len(val_ids):,} val tokens, compression: {compression_train:.2f}x"
    )

    return PretokenizedDataPipeline(
        train_tokens=train_tensor,
        val_tokens=val_tensor,
        tokenizer=tokenizer,
        metadata=asdict(meta),
    )


def load_token_cache(
    cache_dir: Union[str, Path],
    tokenizer: Optional[BPETokenizer] = None,
) -> PretokenizedDataPipeline:
    """Load a persistent pre-tokenized dataset cache with integrity checks.

    Args:
        cache_dir: Directory containing train_tokens.pt, val_tokens.pt, and cache_metadata.json.
        tokenizer: Optional BPETokenizer to validate fingerprint against the cache.

    Returns:
        PretokenizedDataPipeline ready for model training.

    Raises:
        FileNotFoundError: If cache files are missing.
        ValueError: If metadata or tokenizer fingerprints mismatch.
    """
    cache_path = Path(cache_dir)
    meta_path = cache_path / "cache_metadata.json"
    train_path = cache_path / "train_tokens.pt"
    val_path = cache_path / "val_tokens.pt"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Cache metadata file missing: {meta_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training tokens tensor missing: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation tokens tensor missing: {val_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    # Validate against expected tokenizer if provided
    if tokenizer is not None:
        expected_fp = tokenizer.get_fingerprint()
        cache_fp = meta_dict.get("tokenizer_fingerprint")
        if cache_fp != expected_fp:
            raise ValueError(
                f"Tokenizer fingerprint mismatch! Cache was built with {cache_fp}, "
                f"but loaded tokenizer fingerprint is {expected_fp}."
            )
        if tokenizer.vocab_size != meta_dict.get("vocab_size"):
            raise ValueError(
                f"Vocabulary size mismatch! Cache vocab_size={meta_dict.get('vocab_size')}, "
                f"tokenizer vocab_size={tokenizer.vocab_size}."
            )
    else:
        # Reconstruct an empty BPETokenizer with base vocabulary if none provided
        tokenizer = BPETokenizer()

    train_tokens = torch.load(train_path, weights_only=True)
    val_tokens = torch.load(val_path, weights_only=True)

    return PretokenizedDataPipeline(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer=tokenizer,
        metadata=meta_dict,
    )

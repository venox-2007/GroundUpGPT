"""Token and positional embedding modules for GroundUpGPT.

Combines learnable token representations with learnable absolute positional
embeddings for decoder-only Transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Combines learnable token embeddings and learnable absolute positional embeddings.

    Args:
        vocab_size: Total vocabulary size.
        embed_dim: Hidden representation dimension (d_model).
        block_size: Maximum sequence length (context window).

    Raises:
        ValueError: If dimensions or sizes are non-positive.
    """

    def __init__(self, vocab_size: int, embed_dim: int, block_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.vocab_size: int = vocab_size
        self.embed_dim: int = embed_dim
        self.block_size: int = block_size

        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding: nn.Embedding = nn.Embedding(block_size, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute the sum of token and position embeddings for input token IDs.

        Args:
            idx: Tensor of token indices shaped (batch_size, sequence_length).

        Returns:
            torch.Tensor: Combined embeddings shaped (batch_size, sequence_length, embed_dim).

        Raises:
            ValueError: If input tensor dimensions are invalid or sequence_length exceeds block_size.
        """
        if idx.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, sequence_length), got shape {list(idx.shape)}")

        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum context window block_size ({self.block_size})."
            )

        # Token embeddings: (B, T) -> (B, T, embed_dim)
        tok_emb = self.token_embedding(idx)

        # Positions generated on same device as input tensor: (T,) -> (T, embed_dim)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(positions)

        # Combined representation via broadcasting: (B, T, embed_dim)
        return tok_emb + pos_emb

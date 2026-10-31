"""Single-head causal self-attention implementation from scratch for GroundUpGPT.

Implements scaled dot-product attention with strict lower-triangular causal masking
using explicit PyTorch tensor operations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    """Single-head causal scaled dot-product self-attention module.

    Computes:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + M) @ V
    where M is a causal mask setting future positions to -inf.

    Args:
        embed_dim: Input feature dimension (d_model).
        head_dim: Output projection dimension for query, key, value, and output.
        block_size: Maximum sequence length (context window).
        bias: Whether to include bias terms in linear projections (default: False).

    Raises:
        ValueError: If embed_dim, head_dim, or block_size are non-positive.
    """

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        block_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.embed_dim: int = embed_dim
        self.head_dim: int = head_dim
        self.block_size: int = block_size

        # Explicit linear projections for Query, Key, and Value
        self.q_proj: nn.Linear = nn.Linear(embed_dim, head_dim, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(embed_dim, head_dim, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(embed_dim, head_dim, bias=bias)

        # Persistent lower-triangular causal mask buffer (1s in lower triangle, 0s in upper)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute causal self-attention over input representations.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            return_weights: Whether to return attention weight matrix for inspection.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output representations of shape (batch_size, sequence_length, head_dim).
                - Causal attention weights of shape (batch_size, sequence_length, sequence_length),
                  or None if return_weights is False.

        Raises:
            ValueError: If input tensor rank is not 3, feature dimension does not match embed_dim,
                        or sequence_length exceeds block_size.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, sequence_length, embed_dim), got shape {list(x.shape)}"
            )

        batch_size, seq_len, in_dim = x.shape
        if in_dim != self.embed_dim:
            raise ValueError(
                f"Input feature dimension ({in_dim}) does not match expected embed_dim ({self.embed_dim})."
            )
        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds configured block_size ({self.block_size})."
            )

        # 1. Project input to Query, Key, Value representations: (B, T, embed_dim) -> (B, T, head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Scaled dot-product attention scores: (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # 3. Apply causal mask BEFORE softmax (set upper triangle positions to -infinity)
        # Note: self.tril is boolean: True allows attention, False masks future tokens
        mask = self.tril[:seq_len, :seq_len]  # (T, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        # 4. Softmax over the key dimension (last dimension) to obtain normalized probabilities
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 5. Weighted aggregation of values: (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        out = attn_weights @ v

        if return_weights:
            return out, attn_weights
        return out, None

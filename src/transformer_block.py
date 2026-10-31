"""Transformer block implementation using a Pre-LayerNorm architecture.

Combines MultiHeadCausalSelfAttention and FeedForward with LayerNorm and residual
pathways for decoder-only autoregressive language models.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.attention import MultiHeadCausalSelfAttention
from src.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Decoder-only Transformer block with Pre-LayerNorm residual architecture.

    Computes:
        x_1 = x + Attention(LayerNorm_1(x))
        x_2 = x_1 + FeedForward(LayerNorm_2(x_1))

    Args:
        embed_dim: Hidden representation dimension (d_model).
        num_heads: Number of attention heads.
        block_size: Maximum sequence length (context window).
        dropout: Dropout rate applied within attention, feedforward, and residuals (default: 0.0).
        bias: Whether linear projections include bias terms (default: False).

    Raises:
        ValueError: If embed_dim is not divisible by num_heads or dimensions are invalid.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.block_size: int = block_size

        # Separate LayerNorm instances for Pre-LN attention and feedforward
        self.ln1: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.attn: MultiHeadCausalSelfAttention = MultiHeadCausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        self.ln2: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.ffn: FeedForward = FeedForward(
            embed_dim=embed_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pass input representations through Pre-LN Attention and FeedForward with residuals.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            return_attention: Whether to return stacked multi-head attention weights.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed tensor of shape (batch_size, sequence_length, embed_dim).
                - Optional attention weights of shape (batch_size, num_heads, sequence_length, sequence_length).

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

        # 1. Pre-LayerNorm 1 -> Multi-Head Causal Self-Attention -> Residual addition
        norm_x1 = self.ln1(x)
        attn_out, attn_weights = self.attn(norm_x1, return_weights=return_attention)
        x = x + attn_out

        # 2. Pre-LayerNorm 2 -> FeedForward (MLP) -> Residual addition
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out

        return x, attn_weights

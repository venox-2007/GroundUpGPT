"""Position-wise Feed-Forward Network (FFN) for GroundUpGPT.

Applies two linear transformations with a non-linear GELU activation independently
to each position in the sequence:
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP) block.

    Args:
        embed_dim: Model hidden dimension (d_model).
        hidden_dim: Inner layer hidden dimension (defaults to 4 * embed_dim if None).
        dropout: Dropout probability applied after the second linear projection (default: 0.0).
        bias: Whether linear layers include bias terms (default: False).

    Raises:
        ValueError: If embed_dim is non-positive or dropout is out of [0.0, 1.0].
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0], got {dropout}")

        self.embed_dim: int = embed_dim
        actual_hidden_dim: int = hidden_dim if hidden_dim is not None else 4 * embed_dim
        if actual_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {actual_hidden_dim}")
        self.hidden_dim: int = actual_hidden_dim

        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(self.hidden_dim, embed_dim, bias=bias),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute position-wise feed-forward projection.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: Projected tensor of shape (batch_size, sequence_length, embed_dim).

        Raises:
            ValueError: If input tensor rank is not 3 or last dimension does not match embed_dim.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, sequence_length, embed_dim), got shape {list(x.shape)}"
            )
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match expected embed_dim ({self.embed_dim})."
            )

        return self.net(x)

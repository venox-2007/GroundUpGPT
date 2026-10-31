"""Complete GroundUpGPT decoder-only Transformer language model architecture.

Assembles token and positional embeddings, stacked Pre-LayerNorm Transformer blocks,
final LayerNorm, and language modeling head with optional weight tying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import TokenPositionEmbedding
from src.transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    """Configuration hyperparameters for GroundUpGPT model architecture.

    Attributes:
        vocab_size: Total vocabulary size.
        block_size: Maximum context length (sequence window).
        embed_dim: Hidden representation dimension (d_model).
        num_heads: Number of attention heads.
        num_layers: Number of stacked Transformer blocks.
        dropout: Dropout rate across attention, FFN, and residuals (default: 0.1).
        bias: Whether linear layers include learned bias terms (default: False).
        weight_tying: Whether to tie token embedding and lm_head projection weights (default: True).
        init_std: Standard deviation for normal weight initialization (default: 0.02).
    """

    vocab_size: int
    block_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float = 0.1
    bias: bool = False
    weight_tying: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Validate architectural constraints and hyperparameter bounds."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0], got {self.dropout}")
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")


class GroundUpGPT(nn.Module):
    """Complete decoder-only GPT Transformer language model.

    Args:
        config: GPTConfig dataclass specifying model hyperparameters.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config

        # 1. Token and Positional Embeddings
        self.embeddings: TokenPositionEmbedding = TokenPositionEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            block_size=config.block_size,
        )
        self.drop: nn.Module = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # 2. Stack of Pre-LayerNorm Transformer Blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                block_size=config.block_size,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.num_layers)
        ])

        # 3. Final Layer Normalization
        self.ln_f: nn.LayerNorm = nn.LayerNorm(config.embed_dim)

        # 4. Language Modeling Output Projection Head
        self.lm_head: nn.Linear = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize all model weights from scratch
        self.apply(self._init_weights)

        # 5. Optional Weight Tying (Weight sharing between token embedding and lm_head)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights with small standard deviation."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute vocabulary logits and optional cross-entropy loss.

        Args:
            idx: Conditioning sequence of integer token IDs shaped (batch_size, sequence_length).
            targets: Optional ground-truth target tokens shaped (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Logits of shape (batch_size, sequence_length, vocab_size).
                - Cross-entropy loss scalar, or None if targets are not provided.

        Raises:
            ValueError: If input tensor rank is invalid, sequence length exceeds block_size,
                        or targets shape does not match idx shape.
        """
        if idx.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor (batch_size, sequence_length), got shape {list(idx.shape)}"
            )
        if idx.is_floating_point():
            raise ValueError("Input token IDs must be integers, got floating point tensor.")

        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds configured block_size ({self.config.block_size})"
            )

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {list(targets.shape)} must match input idx shape {list(idx.shape)}"
                )
            if targets.is_floating_point():
                raise ValueError("Target token IDs must be integers, got floating point tensor.")

        # Combined token + position representations: (B, T) -> (B, T, embed_dim)
        x = self.embeddings(idx)
        x = self.drop(x)

        # Sequentially pass through all Transformer blocks
        for block in self.blocks:
            x, _ = block(x, return_attention=False)

        # Final LayerNorm
        x = self.ln_f(x)

        # Language modeling head logits: (B, T, embed_dim) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            # Flatten across batch and sequence dimensions for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculate the exact total number of unique trainable parameters.

        Safely handles weight-tied parameters by tracking unique tensor storage pointers.

        Args:
            non_embedding: Whether to subtract positional and token embedding counts.

        Returns:
            int: Number of unique trainable parameters.
        """
        unique_params = set()
        for p in self.parameters():
            if p.requires_grad:
                unique_params.add(p)

        n_params = sum(p.numel() for p in unique_params)
        if non_embedding:
            n_params -= self.embeddings.position_embedding.weight.numel()
            n_params -= self.embeddings.token_embedding.weight.numel()

        return n_params

    def get_parameter_breakdown(self) -> Dict[str, Any]:
        """Compute detailed breakdown of unique trainable parameters across all submodules.

        Returns:
            Dict[str, Any]: Structured parameter counts per architectural section.
        """
        tok_emb = self.embeddings.token_embedding.weight.numel()
        pos_emb = self.embeddings.position_embedding.weight.numel()
        embeddings_total = tok_emb + pos_emb

        per_block_params = [
            sum(p.numel() for p in block.parameters() if p.requires_grad)
            for block in self.blocks
        ]
        blocks_total = sum(per_block_params)

        ln_f_total = sum(p.numel() for p in self.ln_f.parameters() if p.requires_grad)

        # If weight-tied, lm_head shares weights with token_embedding and adds 0 unique parameters
        is_tied = self.config.weight_tying and (self.lm_head.weight is self.embeddings.token_embedding.weight)
        lm_head_unique = 0 if is_tied else self.lm_head.weight.numel()

        return {
            "embeddings": {
                "token_embedding": tok_emb,
                "position_embedding": pos_emb,
                "total": embeddings_total,
            },
            "transformer_blocks": {
                "num_blocks": len(self.blocks),
                "per_block": per_block_params[0] if per_block_params else 0,
                "total": blocks_total,
            },
            "final_layer_norm": ln_f_total,
            "lm_head": {
                "unique_params": lm_head_unique,
                "weight_tied": is_tied,
            },
            "total_unique_trainable": self.get_num_params(),
        }

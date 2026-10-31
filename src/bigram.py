"""Bigram language model implementation in PyTorch.

A minimalist statistical baseline model where the probability of predicting the
next token depends solely on the immediate preceding token.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Bigram language model mapping current token to next-token logits.

    Uses a direct learnable embedding lookup table where row `i` represents the
    unnormalized log-probabilities (logits) over the vocabulary for token `i`.

    Args:
        vocab_size: Total number of unique tokens in vocabulary.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        self.vocab_size: int = vocab_size
        # Table of shape (vocab_size, vocab_size) initialized randomly by default
        self.token_embedding_table: nn.Embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the bigram embedding table.

        Args:
            idx: Tensor of token indices with shape (batch_size, sequence_length).
            targets: Optional tensor of ground-truth target indices with
                     shape (batch_size, sequence_length).

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Logits of shape (batch_size, sequence_length, vocab_size).
                - Cross-entropy loss scalar, or None if targets are not provided.
        """
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, C) where C = vocab_size

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits_flat = logits.view(b * t, c)
            targets_flat = targets.view(b * t)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens given initial context indices.

        Args:
            idx: Conditioning context tensor of shape (batch_size, sequence_length).
            max_new_tokens: Number of consecutive new tokens to generate.
            temperature: Sampling temperature (higher = more random, lower = more deterministic).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, sequence_length + max_new_tokens).
        """
        if max_new_tokens <= 0:
            return idx
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {temperature}")

        for _ in range(max_new_tokens):
            # Pluck logits for current sequence (bigram only relies on the last token)
            logits, _ = self.forward(idx)  # (B, T, C)
            logits_last = logits[:, -1, :] / temperature  # (B, C)

            # Apply softmax to obtain probability distribution
            probs = F.softmax(logits_last, dim=-1)  # (B, C)

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

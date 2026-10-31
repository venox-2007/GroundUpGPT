"""Targeted unit tests for the BigramLanguageModel."""

import torch
from src.bigram import BigramLanguageModel


def test_forward_output_dimensions() -> None:
    """Verify forward pass outputs logits with shape (B, T, vocab_size)."""
    vocab_size = 15
    batch_size = 2
    seq_len = 8

    model = BigramLanguageModel(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None


def test_loss_is_finite_and_computable() -> None:
    """Verify cross-entropy loss is computed properly and is finite."""
    vocab_size = 10
    model = BigramLanguageModel(vocab_size=vocab_size)

    idx = torch.tensor([[1, 2, 3], [4, 5, 0]])
    targets = torch.tensor([[2, 3, 4], [5, 0, 1]])

    _, loss = model(idx, targets)

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_backward_gradient_propagation() -> None:
    """Verify backward pass populates gradients for the embedding lookup table."""
    vocab_size = 8
    model = BigramLanguageModel(vocab_size=vocab_size)

    idx = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])

    _, loss = model(idx, targets)
    assert loss is not None

    loss.backward()

    embedding_weight = model.token_embedding_table.weight
    assert embedding_weight.grad is not None
    # Only the rows corresponding to tokens in `idx` should receive non-zero gradients
    assert torch.any(embedding_weight.grad != 0.0)


def test_generate_output_validity() -> None:
    """Verify autoregressive generation yields valid token IDs and expected length."""
    vocab_size = 12
    model = BigramLanguageModel(vocab_size=vocab_size)

    start_idx = torch.tensor([[0, 3]], dtype=torch.long)  # (1, 2)
    max_new_tokens = 20

    out_idx = model.generate(start_idx, max_new_tokens=max_new_tokens, temperature=1.0)

    # Length must equal initial context length + max_new_tokens
    assert out_idx.shape == (1, 2 + max_new_tokens)
    # Generated token IDs must strictly fall within [0, vocab_size - 1]
    assert torch.all(out_idx >= 0)
    assert torch.all(out_idx < vocab_size)

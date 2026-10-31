"""Targeted unit tests for token embeddings and single-head causal self-attention."""

import pytest
import torch
from src.embeddings import TokenPositionEmbedding
from src.attention import CausalSelfAttentionHead


def test_embedding_output_dimensions_and_device() -> None:
    """Verify combined token and positional embeddings produce shape (B, T, embed_dim)."""
    vocab_size = 50
    embed_dim = 24
    block_size = 16
    batch_size = 3
    seq_len = 10

    layer = TokenPositionEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size)
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    out = layer(idx)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert out.device == idx.device
    assert out.dtype == torch.float32


def test_embedding_context_length_validation() -> None:
    """Verify embedding layer rejects sequence lengths exceeding block_size."""
    layer = TokenPositionEmbedding(vocab_size=20, embed_dim=8, block_size=5)

    # Valid sequence length
    _ = layer(torch.randint(0, 20, (1, 5)))

    # Exceeding block_size
    with pytest.raises(ValueError, match="Sequence length .* exceeds maximum context window"):
        layer(torch.randint(0, 20, (1, 6)))

    # Non-2D input
    with pytest.raises(ValueError, match="Expected 2D input tensor"):
        layer(torch.randint(0, 20, (1, 5, 2)))


def test_attention_output_and_weight_dimensions() -> None:
    """Verify attention output shape (B, T, head_dim) and weights shape (B, T, T)."""
    embed_dim = 16
    head_dim = 12
    block_size = 8
    batch_size = 2
    seq_len = 6

    head = CausalSelfAttentionHead(embed_dim=embed_dim, head_dim=head_dim, block_size=block_size)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out, weights = head(x, return_weights=True)
    assert out.shape == (batch_size, seq_len, head_dim)
    assert weights is not None
    assert weights.shape == (batch_size, seq_len, seq_len)

    # return_weights=False returns None for weights
    out_no_w, weights_none = head(x, return_weights=False)
    assert out_no_w.shape == (batch_size, seq_len, head_dim)
    assert weights_none is None


def test_strict_causal_masking() -> None:
    """Verify that all upper-triangular elements in attention weights are strictly zero."""
    head = CausalSelfAttentionHead(embed_dim=12, head_dim=12, block_size=6)
    x = torch.randn(2, 6, 12)

    _, weights = head(x, return_weights=True)
    assert weights is not None

    for b in range(weights.shape[0]):
        w = weights[b]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if j > i:
                    assert w[i, j].item() == 0.0, f"Future position ({i}, {j}) has non-zero weight {w[i, j]}"
                else:
                    assert w[i, j].item() > 0.0, f"Permitted position ({i}, {j}) must be non-zero"


def test_attention_probability_normalization() -> None:
    """Verify that every row of the attention matrix sums to 1.0 within numerical precision."""
    head = CausalSelfAttentionHead(embed_dim=16, head_dim=16, block_size=10)
    x = torch.randn(3, 7, 16)

    _, weights = head(x, return_weights=True)
    assert weights is not None

    row_sums = weights.sum(dim=-1)  # (B, T)
    expected_sums = torch.ones_like(row_sums)
    assert torch.allclose(row_sums, expected_sums, atol=1e-5)


def test_future_token_information_leakage_prevention() -> None:
    """Modifying token at position t must strictly not alter representations at positions < t."""
    embed_dim = 16
    head_dim = 16
    block_size = 8
    seq_len = 6

    head = CausalSelfAttentionHead(embed_dim=embed_dim, head_dim=head_dim, block_size=block_size)

    # Sequence 1
    x1 = torch.randn(1, seq_len, embed_dim)

    # Sequence 2: identical up to index 3, modified at index 4 and 5
    x2 = x1.clone()
    x2[0, 4:, :] = torch.randn(2, embed_dim)

    out1, _ = head(x1, return_weights=False)
    out2, _ = head(x2, return_weights=False)

    # Outputs at indices 0, 1, 2, 3 must be strictly identical
    assert torch.allclose(out1[0, :4, :], out2[0, :4, :], atol=1e-6)

    # Outputs at index 4 and 5 should differ
    assert not torch.allclose(out1[0, 4:, :], out2[0, 4:, :], atol=1e-3)


def test_gradient_propagation_through_projections_and_embeddings() -> None:
    """Verify backward pass propagates non-zero gradients through Q, K, V and embeddings."""
    vocab_size = 20
    embed_dim = 8
    head_dim = 8
    block_size = 5

    embedding_layer = TokenPositionEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size)
    head = CausalSelfAttentionHead(embed_dim=embed_dim, head_dim=head_dim, block_size=block_size)

    tokens = torch.tensor([[1, 4, 2]], dtype=torch.long)
    embedded = embedding_layer(tokens)
    out, _ = head(embedded, return_weights=False)

    loss = out.sum()
    loss.backward()

    # Projections receive gradients
    assert head.q_proj.weight.grad is not None and torch.any(head.q_proj.weight.grad != 0.0)
    assert head.k_proj.weight.grad is not None and torch.any(head.k_proj.weight.grad != 0.0)
    assert head.v_proj.weight.grad is not None and torch.any(head.v_proj.weight.grad != 0.0)

    # Embeddings receive gradients
    assert embedding_layer.token_embedding.weight.grad is not None
    assert torch.any(embedding_layer.token_embedding.weight.grad != 0.0)
    assert embedding_layer.position_embedding.weight.grad is not None
    assert torch.any(embedding_layer.position_embedding.weight.grad != 0.0)


def test_attention_invalid_dimensions_validation() -> None:
    """Verify CausalSelfAttentionHead validates sequence length and feature dimensions."""
    head = CausalSelfAttentionHead(embed_dim=8, head_dim=8, block_size=4)

    # Sequence length exceeds block_size
    with pytest.raises(ValueError, match="Sequence length .* exceeds configured block_size"):
        head(torch.randn(1, 5, 8))

    # Feature dimension mismatch
    with pytest.raises(ValueError, match="Input feature dimension .* does not match expected embed_dim"):
        head(torch.randn(1, 3, 16))

    # Non-3D input
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        head(torch.randn(3, 8))

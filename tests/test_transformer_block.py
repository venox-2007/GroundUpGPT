"""Targeted unit tests for MultiHeadCausalSelfAttention, FeedForward, and TransformerBlock."""

import pytest
import torch
from src.attention import MultiHeadCausalSelfAttention
from src.feedforward import FeedForward
from src.transformer_block import TransformerBlock


def test_multi_head_attention_dimensions_and_divisibility() -> None:
    """Verify output and attention weight dimensions, and reject non-divisible num_heads."""
    embed_dim = 32
    num_heads = 4
    block_size = 8
    batch_size = 2
    seq_len = 5

    attn = MultiHeadCausalSelfAttention(embed_dim, num_heads, block_size)
    x = torch.randn(batch_size, seq_len, embed_dim)

    out, weights = attn(x, return_weights=True)
    assert out.shape == (batch_size, seq_len, embed_dim)
    assert weights is not None
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Indivisible embed_dim and num_heads
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        MultiHeadCausalSelfAttention(embed_dim=30, num_heads=4, block_size=8)


def test_causal_masking_across_all_heads() -> None:
    """Ensure all attention heads in MultiHeadCausalSelfAttention strictly mask future tokens."""
    num_heads = 4
    attn = MultiHeadCausalSelfAttention(embed_dim=32, num_heads=num_heads, block_size=10)
    x = torch.randn(1, 6, 32)

    _, weights = attn(x, return_weights=True)
    assert weights is not None

    for h in range(num_heads):
        w_h = weights[0, h]
        for i in range(w_h.shape[0]):
            for j in range(w_h.shape[1]):
                if j > i:
                    assert w_h[i, j].item() == 0.0
                else:
                    assert w_h[i, j].item() > 0.0


def test_feedforward_dimensions_and_position_isolation() -> None:
    """Verify position-wise FFN preserves 3D shape and acts independently per position."""
    embed_dim = 16
    ffn = FeedForward(embed_dim=embed_dim)
    x = torch.randn(2, 5, embed_dim)

    out = ffn(x)
    assert out.shape == (2, 5, embed_dim)

    # Position independence: modifying token at index 3 must not affect index 0, 1, 2, or 4
    x_mod = x.clone()
    x_mod[:, 3, :] = torch.randn(2, embed_dim)
    out_mod = ffn(x_mod)

    assert torch.allclose(out[:, :3, :], out_mod[:, :3, :])
    assert torch.allclose(out[:, 4:, :], out_mod[:, 4:, :])
    assert not torch.allclose(out[:, 3, :], out_mod[:, 3, :])


def test_transformer_block_output_shapes_and_residuals() -> None:
    """Verify TransformerBlock preserves (B, T, embed_dim) and residuals propagate signal."""
    embed_dim = 24
    num_heads = 3
    block_size = 12

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size)
    x = torch.randn(2, 7, embed_dim)

    out, weights = block(x, return_attention=True)
    assert out.shape == (2, 7, embed_dim)
    assert weights is not None
    assert weights.shape == (2, num_heads, 7, 7)


def test_future_token_leakage_through_full_block() -> None:
    """Verify modifying future token does not alter earlier tokens through the full TransformerBlock."""
    embed_dim = 32
    num_heads = 4
    block_size = 8
    seq_len = 6

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size)
    block.eval()

    x1 = torch.randn(1, seq_len, embed_dim)
    x2 = x1.clone()
    x2[0, 4:, :] = torch.randn(2, embed_dim)  # modify position 4 and 5

    out1, _ = block(x1, return_attention=False)
    out2, _ = block(x2, return_attention=False)

    # Positions 0..3 must be strictly identical
    assert torch.allclose(out1[0, :4, :], out2[0, :4, :], atol=1e-6)
    # Positions 4..5 must differ
    assert not torch.allclose(out1[0, 4:, :], out2[0, 4:, :], atol=1e-3)


def test_gradient_propagation_through_full_block() -> None:
    """Verify backward pass populates gradients for ln1, ln2, attention heads, proj, and ffn."""
    embed_dim = 16
    num_heads = 2
    block_size = 8

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, bias=True)
    x = torch.randn(2, 4, embed_dim, requires_grad=True)

    out, _ = block(x)
    loss = out.sum()
    loss.backward()

    # Input receives gradients through residual stream and transformations
    assert x.grad is not None and torch.any(x.grad != 0.0)

    # LayerNorms receive gradients
    assert block.ln1.weight.grad is not None and torch.any(block.ln1.weight.grad != 0.0)
    assert block.ln2.weight.grad is not None and torch.any(block.ln2.weight.grad != 0.0)

    # Multi-head attention projections receive gradients
    for head in block.attn.heads:
        assert head.q_proj.weight.grad is not None and torch.any(head.q_proj.weight.grad != 0.0)
        assert head.k_proj.weight.grad is not None and torch.any(head.k_proj.weight.grad != 0.0)
        assert head.v_proj.weight.grad is not None and torch.any(head.v_proj.weight.grad != 0.0)
    assert block.attn.proj.weight.grad is not None and torch.any(block.attn.proj.weight.grad != 0.0)

    # FeedForward network receives gradients
    for param in block.ffn.parameters():
        assert param.grad is not None and torch.any(param.grad != 0.0)


def test_dropout_train_vs_eval_modes() -> None:
    """Ensure dropout introduces stochasticity in training mode and determinism in eval mode."""
    torch.manual_seed(42)
    block = TransformerBlock(embed_dim=16, num_heads=2, block_size=8, dropout=0.5)
    x = torch.randn(2, 4, 16)

    # In eval mode, consecutive forward passes must be identical
    block.eval()
    out1_eval, _ = block(x)
    out2_eval, _ = block(x)
    assert torch.equal(out1_eval, out2_eval)

    # In train mode with 0.5 dropout, consecutive forward passes should differ
    block.train()
    out1_train, _ = block(x)
    out2_train, _ = block(x)
    assert not torch.equal(out1_train, out2_train)


def test_transformer_block_sequence_length_validation() -> None:
    """Ensure TransformerBlock rejects sequence lengths exceeding configured block_size."""
    block = TransformerBlock(embed_dim=16, num_heads=2, block_size=4)

    # Valid
    _ = block(torch.randn(1, 4, 16))

    # Exceeding block_size
    with pytest.raises(ValueError, match="exceeds configured block_size"):
        block(torch.randn(1, 5, 16))

    # Feature dimension mismatch
    with pytest.raises(ValueError, match="does not match expected embed_dim"):
        block(torch.randn(1, 3, 20))

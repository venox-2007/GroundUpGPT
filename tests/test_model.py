"""Targeted unit tests for the assembled GroundUpGPT model architecture."""

from pathlib import Path
import pytest
import torch
from src.model import GPTConfig, GroundUpGPT
from src.data import TextDataPipeline
from src.bpe_tokenizer import BPETokenizer


def test_gpt_config_validation() -> None:
    """Verify GPTConfig validates hyperparameter boundaries and divisibility."""
    # Valid config
    cfg = GPTConfig(vocab_size=100, block_size=32, embed_dim=32, num_heads=4, num_layers=2)
    assert cfg.embed_dim == 32

    # Non-divisible embed_dim / num_heads
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        GPTConfig(vocab_size=100, block_size=32, embed_dim=30, num_heads=4, num_layers=2)

    # Non-positive parameters
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        GPTConfig(vocab_size=0, block_size=32, embed_dim=32, num_heads=4, num_layers=2)

    with pytest.raises(ValueError, match="dropout must be in"):
        GPTConfig(vocab_size=100, block_size=32, embed_dim=32, num_heads=4, num_layers=2, dropout=1.5)


def test_forward_output_dimensions_and_loss() -> None:
    """Verify output logits shape (B, T, vocab_size) and finite cross-entropy loss."""
    cfg = GPTConfig(vocab_size=50, block_size=16, embed_dim=24, num_heads=3, num_layers=2)
    model = GroundUpGPT(cfg)

    batch_size = 2
    seq_len = 8
    idx = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Inference forward pass
    logits, loss_none = model(idx)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert loss_none is None

    # Training forward pass
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_weight_tying_and_unique_parameter_counting() -> None:
    """Ensure weight tying shares storage pointers and prevents double counting."""
    # Weight-tied model
    cfg_tied = GPTConfig(
        vocab_size=100, block_size=16, embed_dim=32, num_heads=4, num_layers=2, weight_tying=True
    )
    model_tied = GroundUpGPT(cfg_tied)
    assert model_tied.lm_head.weight is model_tied.embeddings.token_embedding.weight

    breakdown_tied = model_tied.get_parameter_breakdown()
    assert breakdown_tied["lm_head"]["unique_params"] == 0
    assert breakdown_tied["lm_head"]["weight_tied"] is True

    # Untied model
    cfg_untied = GPTConfig(
        vocab_size=100, block_size=16, embed_dim=32, num_heads=4, num_layers=2, weight_tying=False
    )
    model_untied = GroundUpGPT(cfg_untied)
    assert model_untied.lm_head.weight is not model_untied.embeddings.token_embedding.weight

    breakdown_untied = model_untied.get_parameter_breakdown()
    assert breakdown_untied["lm_head"]["unique_params"] == 100 * 32
    assert breakdown_untied["lm_head"]["weight_tied"] is False

    # Untied model must have exactly vocab_size * embed_dim more parameters
    diff = model_untied.get_num_params() - model_tied.get_num_params()
    assert diff == 100 * 32


def test_rtx3050_target_architecture_parameter_count() -> None:
    """Verify exact parameter count for the proposed ~5M baseline configuration."""
    cfg = GPTConfig(
        vocab_size=4096,
        block_size=256,
        embed_dim=256,
        num_heads=4,
        num_layers=5,
        dropout=0.1,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(cfg)
    total_params = model.get_num_params()

    # Exact expected parameter count: 5,051,904 (~5.05M)
    assert total_params == 5_051_904


def test_full_stack_gradient_propagation() -> None:
    """Verify backward autodiff propagates gradients across every module in the full GPT model."""
    cfg = GPTConfig(vocab_size=40, block_size=10, embed_dim=16, num_heads=2, num_layers=2, bias=True)
    model = GroundUpGPT(cfg)

    idx = torch.randint(0, 40, (2, 6))
    targets = torch.randint(0, 40, (2, 6))

    _, loss = model(idx, targets=targets)
    assert loss is not None
    loss.backward()

    # Embeddings
    assert model.embeddings.token_embedding.weight.grad is not None
    assert torch.any(model.embeddings.token_embedding.weight.grad != 0.0)
    assert model.embeddings.position_embedding.weight.grad is not None
    assert torch.any(model.embeddings.position_embedding.weight.grad != 0.0)

    # All Transformer blocks
    for block in model.blocks:
        assert block.ln1.weight.grad is not None
        assert block.ln2.weight.grad is not None
        for head in block.attn.heads:
            assert head.q_proj.weight.grad is not None
            assert head.k_proj.weight.grad is not None
            assert head.v_proj.weight.grad is not None
        assert block.attn.proj.weight.grad is not None
        for p in block.ffn.parameters():
            assert p.grad is not None

    # Final LayerNorm & LM Head
    assert model.ln_f.weight.grad is not None
    assert model.lm_head.weight.grad is not None


def test_future_token_leakage_through_full_gpt_stack() -> None:
    """Modifying future token in eval mode must strictly not alter earlier logits."""
    cfg = GPTConfig(vocab_size=50, block_size=12, embed_dim=24, num_heads=3, num_layers=3)
    model = GroundUpGPT(cfg)
    model.eval()

    idx1 = torch.tensor([[10, 20, 30, 40, 5]])
    idx2 = torch.tensor([[10, 20, 30, 40, 49]])  # only position 4 modified

    logits1, _ = model(idx1)
    logits2, _ = model(idx2)

    # Logits at positions 0, 1, 2, 3 must be strictly identical
    assert torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-6)
    # Logits at position 4 must differ
    assert not torch.allclose(logits1[:, 4, :], logits2[:, 4, :], atol=1e-3)


def test_context_window_validation() -> None:
    """Verify input sequence exceeding block_size raises ValueError."""
    cfg = GPTConfig(vocab_size=20, block_size=6, embed_dim=16, num_heads=2, num_layers=1)
    model = GroundUpGPT(cfg)

    # Valid
    _ = model(torch.randint(0, 20, (1, 6)))

    # Exceeding context length
    with pytest.raises(ValueError, match="Sequence length .* exceeds configured block_size"):
        model(torch.randint(0, 20, (1, 7)))


def test_compatibility_with_pipeline_batches_and_optimizer_step() -> None:
    """Verify model seamlessly accepts batches from TextDataPipeline and executes one optimizer step."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_text.txt"

    # 1. Character-level pipeline batch
    pipe_char = TextDataPipeline.from_file(fixture_path, train_ratio=0.9)
    cfg_char = GPTConfig(
        vocab_size=pipe_char.vocab_size, block_size=12, embed_dim=16, num_heads=2, num_layers=2
    )
    model_char = GroundUpGPT(cfg_char)
    x_c, y_c = pipe_char.get_batch("train", batch_size=2, block_size=12)

    optimizer = torch.optim.AdamW(model_char.parameters(), lr=1e-3)
    optimizer.zero_grad()
    _, loss_c = model_char(x_c, y_c)
    assert loss_c is not None and torch.isfinite(loss_c)
    loss_c.backward()
    optimizer.step()

    # 2. BPE pipeline batch
    pipe_bpe = TextDataPipeline.from_file(
        fixture_path, train_ratio=0.9, tokenizer=BPETokenizer(), bpe_vocab_size=270
    )
    cfg_bpe = GPTConfig(
        vocab_size=pipe_bpe.vocab_size, block_size=12, embed_dim=16, num_heads=2, num_layers=2
    )
    model_bpe = GroundUpGPT(cfg_bpe)
    x_b, y_b = pipe_bpe.get_batch("train", batch_size=2, block_size=12)

    _, loss_b = model_bpe(x_b, y_b)
    assert loss_b is not None and torch.isfinite(loss_b)

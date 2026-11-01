"""Comprehensive unit and regression tests for GroundUpGPT-27M Scaled Architecture.

Verifies:
1. Exact unique trainable parameter count (27,411,456) and submodule breakdown.
2. Forward logits dimensions and causal cross-entropy loss computation.
3. Causal invariance: modifying future tokens does not leak to earlier positions.
4. Backward propagation and finite gradient norms (no NaNs, no Infs).
5. Numerical equivalence between manual attention and optimized SDPA paths.
6. Weight tying integrity (shared tensor pointer between token embeddings and lm_head).
7. Checkpoint serialization, restoration, and strict architecture incompatibility rejection.
8. Lossless compatibility with the production 4,096-vocabulary BPE tokenizer.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT, get_5m_config, get_27m_config


@pytest.fixture
def scaled_27m_config() -> GPTConfig:
    """Fixture providing standard GroundUpGPT-27M configuration."""
    return get_27m_config(block_size=64, dropout=0.0)


@pytest.fixture
def tiny_27m_model(scaled_27m_config: GPTConfig) -> GroundUpGPT:
    """Instantiate a GroundUpGPT-27M model with short block_size for fast CPU testing."""
    torch.manual_seed(42)
    model = GroundUpGPT(scaled_27m_config)
    model.eval()
    return model


def test_groundupgpt_27m_exact_parameter_count():
    """Verify that GroundUpGPT-27M has exactly 27,411,456 unique trainable parameters."""
    full_config = get_27m_config(block_size=256)
    model = GroundUpGPT(full_config)

    total_params = model.get_num_params()
    assert total_params == 27_411_456, f"Expected 27,411,456 parameters, got {total_params}"

    breakdown = model.get_parameter_breakdown()

    # Embeddings: token (4096 * 512 = 2,097,152) + position (256 * 512 = 131,072) = 2,228,224
    assert breakdown["embeddings"]["token_embedding"] == 2_097_152
    assert breakdown["embeddings"]["position_embedding"] == 131_072
    assert breakdown["embeddings"]["total"] == 2_228_224

    # Transformer Blocks: 8 blocks, each with 3,147,776 params -> total 25,182,208
    assert breakdown["transformer_blocks"]["num_blocks"] == 8
    assert breakdown["transformer_blocks"]["per_block"] == 3_147_776
    assert breakdown["transformer_blocks"]["total"] == 25_182_208

    # Final LayerNorm: gamma and beta = 512 * 2 = 1,024
    assert breakdown["final_layer_norm"] == 1_024

    # LM Head: weight-tied to token embedding -> 0 unique parameters
    assert breakdown["lm_head"]["weight_tied"] is True
    assert breakdown["lm_head"]["unique_params"] == 0

    assert breakdown["total_unique_trainable"] == 27_411_456


def test_weight_tying_identity(tiny_27m_model: GroundUpGPT):
    """Verify weight tying shares the exact underlying parameter tensor."""
    assert tiny_27m_model.lm_head.weight is tiny_27m_model.embeddings.token_embedding.weight


def test_forward_output_dimensions_and_loss(tiny_27m_model: GroundUpGPT):
    """Verify forward pass output shapes and cross-entropy loss computation."""
    batch_size, seq_len = 2, 16
    x = torch.randint(0, 4096, (batch_size, seq_len))
    y = torch.randint(0, 4096, (batch_size, seq_len))

    # Forward without targets
    logits, loss = tiny_27m_model(x)
    assert logits.shape == (batch_size, seq_len, 4096)
    assert loss is None

    # Forward with targets
    logits, loss = tiny_27m_model(x, targets=y)
    assert logits.shape == (batch_size, seq_len, 4096)
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_causal_invariance_no_future_leakage(tiny_27m_model: GroundUpGPT):
    """Verify causal invariance: modifying future tokens does not alter past logits."""
    tiny_27m_model.eval()

    # Test both manual attention and SDPA paths
    for use_sdpa in [False, True]:
        tiny_27m_model.set_use_sdpa(use_sdpa)

        seq_len = 16
        prefix_len = 8

        # Original sequence
        tokens_a = torch.randint(0, 4096, (1, seq_len))

        # Modified sequence: identical prefix, altered suffix
        tokens_b = tokens_a.clone()
        tokens_b[0, prefix_len:] = torch.randint(0, 4096, (seq_len - prefix_len,))

        with torch.no_grad():
            logits_a, _ = tiny_27m_model(tokens_a)
            logits_b, _ = tiny_27m_model(tokens_b)

        # Logits at positions 0 to prefix_len - 1 must be strictly identical
        diff = (logits_a[0, :prefix_len, :] - logits_b[0, :prefix_len, :]).abs().max().item()
        assert diff == pytest.approx(0.0, abs=1e-5), (
            f"Future token leaked to past positions in 27M model (use_sdpa={use_sdpa})! Max diff: {diff}"
        )


def test_full_stack_gradient_propagation_and_finiteness():
    """Verify complete backward pass and non-vanishing, non-diverging gradients."""
    config = get_27m_config(block_size=32, dropout=0.0)
    model = GroundUpGPT(config)
    model.train()

    x = torch.randint(0, 4096, (2, 16))
    y = torch.randint(0, 4096, (2, 16))

    _, loss = model(x, targets=y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"
            assert param.grad.abs().sum().item() > 0.0, f"Gradient for {name} vanished completely"


def test_manual_attention_vs_sdpa_numerical_equivalence():
    """Verify that manual attention and PyTorch SDPA compute identical forward outputs."""
    config = get_27m_config(block_size=32, dropout=0.0)
    model = GroundUpGPT(config)
    model.eval()

    x = torch.randint(0, 4096, (2, 16))

    # 1. Manual explicit attention
    model.set_use_sdpa(False)
    with torch.no_grad():
        logits_manual, _ = model(x)

    # 2. Optimized SDPA
    model.set_use_sdpa(True)
    with torch.no_grad():
        logits_sdpa, _ = model(x)

    max_diff = (logits_manual - logits_sdpa).abs().max().item()
    assert max_diff < 1e-4, f"Mismatch between manual attention and SDPA: max diff {max_diff}"


def test_checkpoint_roundtrip_and_serialization(tiny_27m_model: GroundUpGPT):
    """Verify state serialization, weight saving, and exact numerical restoration."""
    tiny_27m_model.eval()
    x = torch.randint(0, 4096, (1, 8))

    with torch.no_grad():
        expected_logits, _ = tiny_27m_model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        checkpoint = {
            "model_config": asdict(tiny_27m_model.config),
            "model_state_dict": tiny_27m_model.state_dict(),
        }
        torch.save(checkpoint, tmp_path)

        # Restore into fresh instance
        loaded = torch.load(tmp_path, weights_only=False)
        restored_config = GPTConfig(**loaded["model_config"])
        restored_model = GroundUpGPT(restored_config)
        restored_model.load_state_dict(loaded["model_state_dict"])
        restored_model.eval()

        with torch.no_grad():
            restored_logits, _ = restored_model(x)

        assert torch.equal(expected_logits, restored_logits)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_incompatibility_protection_with_old_5m_checkpoint():
    """Verify that loading an incompatible 5M architecture into 27M raises an error."""
    config_5m = get_5m_config(block_size=256)
    model_5m = GroundUpGPT(config_5m)

    config_27m = get_27m_config(block_size=256)
    model_27m = GroundUpGPT(config_27m)

    # Attempting to load 5M state dict into 27M model must fail with RuntimeError (shape mismatch)
    with pytest.raises(RuntimeError):
        model_27m.load_state_dict(model_5m.state_dict(), strict=True)


def test_bpe_tokenizer_integration(tiny_27m_model: GroundUpGPT):
    """Verify that actual BPE-encoded sequences pass cleanly through the 27M model."""
    bpe_path = Path("checkpoints/bpe_vocab4096.json")
    if not bpe_path.is_file():
        pytest.skip("Production BPE tokenizer not found at checkpoints/bpe_vocab4096.json")

    tokenizer = BPETokenizer().load(bpe_path)
    sample_text = "The GroundUpGPT-27M architecture expands our model capacity five-fold."
    tokens = tokenizer.encode(sample_text)
    assert len(tokens) > 0

    x = torch.tensor([tokens[:16]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = tiny_27m_model(x)

    assert logits.shape == (1, min(len(tokens), 16), 4096)
    assert torch.isfinite(logits).all()

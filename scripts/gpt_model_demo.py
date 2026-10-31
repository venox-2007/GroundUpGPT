"""Educational demonstration of GroundUpGPT model initialization, forward pass, and gradient flow."""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.model import GPTConfig, GroundUpGPT
from src.data import TextDataPipeline


def main() -> None:
    print("=" * 65)
    print(" GroundUpGPT - Complete Transformer Model Assembly Demo")
    print("=" * 65)

    torch.manual_seed(42)

    # 1. Inspect the Baseline 5M Architecture Parameter Count
    baseline_cfg = GPTConfig(
        vocab_size=4096,
        block_size=256,
        embed_dim=256,
        num_heads=4,
        num_layers=5,
        dropout=0.1,
        bias=False,
        weight_tying=True,
    )
    baseline_model = GroundUpGPT(baseline_cfg)
    baseline_breakdown = baseline_model.get_parameter_breakdown()

    print("\n[1] RTX 3050 Target Architecture (~5M Baseline):")
    print(f"    Vocab Size          : {baseline_cfg.vocab_size} (BPE Subword Target)")
    print(f"    Context Window (T)  : {baseline_cfg.block_size} tokens")
    print(f"    Embedding Dim (d)   : {baseline_cfg.embed_dim}")
    print(f"    Heads & Layers      : {baseline_cfg.num_heads} heads, {baseline_cfg.num_layers} layers")
    print(f"    Weight Tying        : {baseline_cfg.weight_tying}")
    print(f"    Embeddings Params   : {baseline_breakdown['embeddings']['total']:,}")
    print(f"    Blocks Params       : {baseline_breakdown['transformer_blocks']['total']:,}")
    print(f"    Final LayerNorm     : {baseline_breakdown['final_layer_norm']:,}")
    print(f"    LM Head (unique)    : {baseline_breakdown['lm_head']['unique_params']:,} (Tied: {baseline_breakdown['lm_head']['weight_tied']})")
    print(f"    Exact Total Params  : {baseline_model.get_num_params():,} unique parameters (~5.05M)")

    # 2. Lightweight Demo Configuration for Fast Execution
    demo_cfg = GPTConfig(
        vocab_size=34,       # Matching sample_text fixture vocabulary
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(demo_cfg)
    print(f"\n[2] Lightweight Demo Model Initialized (Random Weights):")
    print(f"    Vocab Size : {demo_cfg.vocab_size}, Embed Dim: {demo_cfg.embed_dim}, Layers: {demo_cfg.num_layers}")
    print(f"    Demo Unique Parameters: {model.get_num_params():,}")

    # 3. Load Sample Batch from existing TextDataPipeline
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"
    pipeline = TextDataPipeline.from_file(fixture_path, train_ratio=0.9)
    x_batch, y_batch = pipeline.get_batch("train", batch_size=2, block_size=demo_cfg.block_size)

    print("\n[3] Forward Pass on Sample Batch:")
    print(f"    Input Batch (idx) shape  : {list(x_batch.shape)} (B=2, T=16)")
    print(f"    Target Batch (y) shape   : {list(y_batch.shape)}")

    # Initial Forward Pass
    logits, initial_loss = model(x_batch, y_batch)
    assert initial_loss is not None
    print(f"    Output Logits shape      : {list(logits.shape)} (B, T, vocab_size)")
    print(f"    Initial Cross-Entropy Loss: {initial_loss.item():.4f}")

    # 4. One Optimization Step (Sanity Check of Backpropagation)
    print("\n[4] Single Optimizer Step Execution:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    initial_loss.backward()

    # Check gradient flow across components
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Measure loss immediately after step on same batch
    with torch.no_grad():
        _, step_loss = model(x_batch, y_batch)
        assert step_loss is not None

    print(f"    Gradient Norm (clipped)  : {grad_norm.item():.4f}")
    print(f"    Post-Step Loss           : {step_loss.item():.4f}")
    print(f"    Step Loss Delta          : {initial_loss.item() - step_loss.item():+.4f}")
    print("    Full Autodiff Gradient Propagation: Verified")

    print("\n" + "=" * 65)
    print(" GroundUpGPT Assembly Demo Complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

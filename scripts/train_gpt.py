"""Command-line interface and training smoke demo for GroundUpGPT."""

import argparse
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
from src.trainer import Trainer, TrainingConfig


def run_smoke_test(device: str = "auto", steps: int = 10) -> None:
    """Execute a lightweight offline training and checkpoint resumption smoke test."""
    print("=" * 65)
    print(" GroundUpGPT - GPU Training Pipeline Smoke Test")
    print("=" * 65)

    if steps < 2:
        steps = 2
    phase1_steps = steps // 2

    torch.manual_seed(42)

    # 1. Dataset loading from offline fixture
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"
    print(f"\n[1] Loading Dataset Fixture: {fixture_path.relative_to(REPO_ROOT)}")
    pipeline = TextDataPipeline.from_file(fixture_path, train_ratio=0.9)
    print(f"    Vocab Size: {pipeline.vocab_size}, Train tokens: {len(pipeline.train_data)}")

    # 2. Small Model Configuration
    model_cfg = GPTConfig(
        vocab_size=pipeline.vocab_size,
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(model_cfg)
    print(f"\n[2] Initialized Model Architecture:")
    print(f"    Embed Dim: {model_cfg.embed_dim}, Heads: {model_cfg.num_heads}, Layers: {model_cfg.num_layers}")
    print(f"    Unique Parameters: {model.get_num_params():,}")

    # 3. Training Phase 1: Run phase1_steps and save checkpoint
    checkpoint_path = REPO_ROOT / "checkpoints" / "smoke_checkpoint.pt"
    train_cfg_phase1 = TrainingConfig(
        max_steps=phase1_steps,
        batch_size=2,
        block_size=16,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=max(1, phase1_steps // 2),
        grad_accum_steps=2,
        eval_interval=max(1, phase1_steps),
        eval_iters=2,
        device=device,
        checkpoint_dir=str(checkpoint_path.parent),
        seed=42,
    )

    trainer1 = Trainer(model=model, pipeline=pipeline, config=train_cfg_phase1)
    print(f"\n[3] Training Phase 1 (Initial {phase1_steps} Steps on {device}):")
    metrics_p1 = trainer1.train()
    print(f"    Completed Steps: {metrics_p1['total_steps']}")
    print(f"    Phase 1 Val Loss: {metrics_p1['final_val_loss']:.4f}")

    # Save Checkpoint
    trainer1.save_checkpoint(checkpoint_path, val_loss=metrics_p1["final_val_loss"])
    print(f"    Saved checkpoint to: {checkpoint_path.relative_to(REPO_ROOT)}")

    # 4. Training Phase 2: Resume from checkpoint and train remaining steps
    print(f"\n[4] Training Phase 2 (Resuming from Step {trainer1.current_step} to {steps}):")
    # Fresh model instance with identical configuration to prove state restoration
    resumed_model = GroundUpGPT(model_cfg)
    train_cfg_phase2 = TrainingConfig(
        max_steps=steps,  # Continue up to steps
        batch_size=2,
        block_size=16,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=max(1, phase1_steps // 2),
        grad_accum_steps=2,
        eval_interval=max(1, steps - phase1_steps),
        eval_iters=2,
        device=device,
        checkpoint_dir=str(checkpoint_path.parent),
        seed=42,
    )
    trainer2 = Trainer(model=resumed_model, pipeline=pipeline, config=train_cfg_phase2)

    resumed_step = trainer2.load_checkpoint(checkpoint_path)
    print(f"    Restored Step Counter: {resumed_step}")
    assert resumed_step == phase1_steps, f"Expected step {phase1_steps}, got {resumed_step}"

    metrics_p2 = trainer2.train()
    print(f"    Completed Steps: {metrics_p2['total_steps']}")
    print(f"    Phase 2 Val Loss: {metrics_p2['final_val_loss']:.4f}")

    # Cleanup smoke checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"    Cleaned up temporary checkpoint file.")

    print("\n" + "=" * 65)
    print(" Training Smoke Test Completed Successfully.")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GroundUpGPT model.")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Run offline smoke test.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    parser.add_argument("--steps", type=int, default=10, help="Maximum training steps.")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(device=args.device, steps=args.steps)


if __name__ == "__main__":
    main()

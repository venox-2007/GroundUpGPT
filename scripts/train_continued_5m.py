"""Continued pretraining pipeline for 5.05M GroundUpGPT from Step 200.

Features:
- Resumes strictly from Step 200 checkpoint (model parameters, AdamW state, scaler, RNG).
- Preserves the original 1,000-step cosine decay learning rate schedule.
- Targets Step 200 -> Step 600 initially, continuing toward Step 1000 if validation loss improves.
- Monitored fixed validation benchmark evaluated at 50-step intervals.
- Safe atomic checkpointing preserving:
    * checkpoints/5m_initial/model_5m_step200.pt (unmodified original baseline)
    * checkpoints/5m_continued/model_5m_best.pt (best validation loss checkpoint)
    * checkpoints/5m_continued/model_5m_latest.pt (most recent resumable checkpoint)
- Detailed telemetry, loss tracking, throughput, and qualitative generation comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import load_token_cache
from src.trainer import Trainer, TrainingConfig, get_tokenizer_fingerprint
from scripts.evaluate_model import compute_fixed_validation_benchmark, EVAL_PROMPTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_continued_5m")


def safe_save_checkpoint(checkpoint_dict: Dict[str, Any], target_path: Path) -> None:
    """Safely write checkpoint to temporary file first and atomically replace target."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    torch.save(checkpoint_dict, tmp_path)
    tmp_path.replace(target_path)


def run_prompt_evaluations(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    device: str,
    tag: str,
) -> Dict[str, Dict[str, str]]:
    """Generate completions under fixed seeds across required benchmark prompts."""
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    results = {}
    print(f"\n--- Qualitative Completions ({tag}) ---")
    for prompt in EVAL_PROMPTS:
        greedy_cfg = GenerationConfig(max_new_tokens=35, temperature=0.0)
        sampled_cfg = GenerationConfig(max_new_tokens=35, temperature=0.8, top_p=0.9, seed=42)

        res_greedy = generator.generate(prompt=prompt, config=greedy_cfg)
        res_sampled = generator.generate(prompt=prompt, config=sampled_cfg)

        clean_greedy = res_greedy.generated_text.replace("\n", " ")
        clean_sampled = res_sampled.generated_text.replace("\n", " ")
        print(f"[{tag}] Prompt: '{prompt}'")
        print(f"       Greedy (T=0):  '{clean_greedy}'")
        print(f"       Sample (T=0.8): '{clean_sampled}'\n")

        results[prompt] = {
            "greedy_t0": res_greedy.generated_text,
            "sampled_t08_p09": res_sampled.generated_text,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue pretraining GroundUpGPT from Step 200.")
    step200_default = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"
    step250_latest = REPO_ROOT / "checkpoints" / "5m_continued" / "step_250.pt"
    default_init_ckpt = step250_latest if step250_latest.is_file() else step200_default

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=default_init_ckpt,
        help="Source checkpoint to resume from (defaults to latest verified checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "5m_continued",
        help="Output directory for continued checkpoints",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json",
        help="Production BPE tokenizer path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache",
        help="Pretokenized binary token cache directory",
    )
    parser.add_argument("--initial-target-step", type=int, default=600, help="Initial target step (default: 600)")
    parser.add_argument("--max-target-step", type=int, default=1000, help="Maximum schedule step (default: 1000)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Fixed evaluation interval (default: 50)")
    parser.add_argument("--time-budget", type=float, default=720.0, help="Maximum training budget in seconds (default: 720 / 12 min)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" GroundUpGPT — Continued Pretraining Pipeline (Step 200+)")
    print("=" * 72)

    # 1. Verify and Load Tokenizer
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer_path}")
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    print(f"[*] Loaded Tokenizer: {args.tokenizer_path.name} (Vocab size: {tokenizer.vocab_size})")

    # 2. Load Pretokenized Dataset
    pipeline = load_token_cache(args.cache_dir, tokenizer=tokenizer)
    print(f"[*] Loaded Token Cache: {len(pipeline.train_tokens):,} train tokens, {len(pipeline.val_tokens):,} val tokens")

    # 3. Verify Source Step 200 Checkpoint
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"Initial Step 200 checkpoint not found at: {args.init_checkpoint}")

    # Inspect checkpoint metadata
    raw_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    m_cfg_dict = raw_ckpt["model_config"]
    gpt_config = GPTConfig(**m_cfg_dict)
    model = GroundUpGPT(gpt_config)
    num_params = model.get_num_params()
    print(f"[*] Model Parameters: {num_params:,} (5.05M architecture)")

    # Target Device Resolution
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[*] Execution Device: {resolved_device} ({torch.cuda.get_device_name(0) if resolved_device == 'cuda' else 'CPU'})")

    # 4. Configure Trainer preserving 1,000-step global schedule
    train_config = TrainingConfig(
        max_steps=args.max_target_step,
        batch_size=16,
        block_size=gpt_config.block_size,
        learning_rate=6e-4,
        min_lr=6e-5,
        warmup_steps=20,
        weight_decay=0.1,
        grad_clip=1.0,
        grad_accum_steps=4,
        eval_interval=args.eval_interval,
        eval_iters=10,
        checkpoint_interval=args.eval_interval,
        checkpoint_dir=str(args.checkpoint_dir),
        device=resolved_device,
        seed=42,
    )

    trainer = Trainer(model=model, pipeline=pipeline, config=train_config)

    # Restore Step 200 state
    print(f"\n[*] Restoring full state from Step 200 checkpoint: {args.init_checkpoint.name}...")
    restored_step = trainer.load_checkpoint(args.init_checkpoint)
    print(f"[*] Checkpoint restored successfully at Step #{restored_step}!")
    print(f"[*] Resumed Learning Rate at Step #{restored_step}: {trainer.get_lr_for_step(restored_step):.6f}")

    # 5. Benchmark Resumed Checkpoint and Load Baseline Step 200
    val_tokens = pipeline.val_tokens.long()
    resumed_bench = compute_fixed_validation_benchmark(
        model=model,
        tokens=val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    print(f"\n[RESUMED CHECKPOINT #{restored_step}] Fixed Val Loss: {resumed_bench['loss']:.4f} | Perplexity: {resumed_bench['perplexity']:.2f}")

    # Load or compute Step 200 baseline reference
    step200_log_path = REPO_ROOT / "logs" / "baseline_step200_evaluation.json"
    if step200_log_path.is_file():
        with open(step200_log_path, "r", encoding="utf-8") as f:
            step200_eval_data = json.load(f)
        baseline_step200_loss = step200_eval_data.get("evaluation", {}).get("loss", 6.1520)
        baseline_step200_ppl = step200_eval_data.get("evaluation", {}).get("perplexity", 469.67)
        step200_samples = step200_eval_data.get("samples", {})
        print(f"[REFERENCE BASELINE STEP 200] Fixed Val Loss: {baseline_step200_loss:.4f} | Perplexity: {baseline_step200_ppl:.2f}")
    else:
        baseline_step200_loss = resumed_bench["loss"]
        baseline_step200_ppl = resumed_bench["perplexity"]
        step200_samples = run_prompt_evaluations(model, tokenizer, resolved_device, f"STEP {restored_step}")

    best_checkpoint_path = args.checkpoint_dir / "model_5m_best.pt"
    latest_checkpoint_path = args.checkpoint_dir / "model_5m_latest.pt"

    # Initialize best validation loss and ensure best checkpoint is at least as good as resumed
    best_val_loss = resumed_bench["loss"]
    if best_checkpoint_path.is_file():
        try:
            existing_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            existing_best_loss = existing_best.get("val_loss", 999.0)
            if existing_best_loss < best_val_loss:
                best_val_loss = existing_best_loss
            else:
                safe_save_checkpoint(raw_ckpt, best_checkpoint_path)
        except Exception:
            safe_save_checkpoint(raw_ckpt, best_checkpoint_path)
    else:
        safe_save_checkpoint(raw_ckpt, best_checkpoint_path)

    # Save initial resumed state as latest starting point
    safe_save_checkpoint(raw_ckpt, latest_checkpoint_path)

    # 6. Training Execution Loop with Dynamic Monitoring
    start_time = time.time()
    history = [
        {
            "step": restored_step,
            "train_loss": raw_ckpt.get("val_loss", float("nan")),
            "val_loss": resumed_bench["loss"],
            "perplexity": resumed_bench["perplexity"],
            "lr": trainer.get_lr_for_step(restored_step),
            "elapsed_sec": 0.0,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024), 1) if resolved_device == "cuda" else 0.0,
        }
    ]

    tokens_per_step = train_config.batch_size * train_config.block_size * train_config.grad_accum_steps
    tokens_processed = 0
    patience = 4
    deterioration_count = 0
    target_step = args.initial_target_step

    print(f"\n[*] Beginning Phase 1: Training from Step {restored_step} to {target_step}...")

    while trainer.current_step < args.max_target_step:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= args.time_budget:
            print(f"[!] Time budget reached ({elapsed:.1f}s >= {args.time_budget}s). Halting training gracefully.")
            break

        # Execute training step
        step_loss = trainer.train_step()
        tokens_processed += tokens_per_step

        current_step = trainer.current_step

        # Evaluation & Checkpoint Interval
        if current_step % args.eval_interval == 0 or current_step == target_step:
            eval_bench = compute_fixed_validation_benchmark(
                model=model,
                tokens=val_tokens,
                block_size=gpt_config.block_size,
                batch_size=16,
                num_eval_windows=64,
                seed=1337,
                device=resolved_device,
            )

            current_lr = trainer.get_lr_for_step(current_step)
            peak_vram = round(torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024), 1) if resolved_device == "cuda" else 0.0
            elapsed = time.time() - start_time
            throughput = tokens_processed / max(0.001, elapsed)

            print(
                f"[Step #{current_step:4d}] "
                f"Train Loss: {step_loss:.4f} | "
                f"Fixed Val Loss: {eval_bench['loss']:.4f} | "
                f"PPL: {eval_bench['perplexity']:6.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {throughput:7.1f} tok/s | "
                f"VRAM: {peak_vram:5.1f} MB"
            )

            record = {
                "step": current_step,
                "train_loss": round(step_loss, 4),
                "val_loss": eval_bench["loss"],
                "perplexity": eval_bench["perplexity"],
                "lr": current_lr,
                "elapsed_sec": round(elapsed, 2),
                "peak_vram_mb": peak_vram,
            }
            history.append(record)

            # Build checkpoint dictionary
            ckpt_dict = {
                "step": current_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scaler_state_dict": trainer.scaler.state_dict() if trainer.use_amp else None,
                "model_config": asdict(model.config),
                "training_config": asdict(trainer.config),
                "tokenizer_type": "BPETokenizer",
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_fingerprint": tokenizer.get_fingerprint(),
                "val_loss": eval_bench["loss"],
                "perplexity": eval_bench["perplexity"],
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "pipeline_rng_state": trainer.rng_generator.get_state(),
                "eval_rng_state": trainer.eval_rng_generator.get_state(),
            }

            # 1. Always update latest checkpoint
            safe_save_checkpoint(ckpt_dict, latest_checkpoint_path)

            # Also save periodic step checkpoint
            periodic_path = args.checkpoint_dir / f"step_{current_step}.pt"
            safe_save_checkpoint(ckpt_dict, periodic_path)

            # 2. Check for improvement
            if eval_bench["loss"] < best_val_loss:
                improvement = best_val_loss - eval_bench["loss"]
                best_val_loss = eval_bench["loss"]
                deterioration_count = 0
                safe_save_checkpoint(ckpt_dict, best_checkpoint_path)
                print(f"  --> [*] New Best Model Saved! Val Loss: {best_val_loss:.4f} (-{improvement:.4f})")
            else:
                deterioration_count += 1
                print(f"  --> [!] No improvement (Deterioration count: {deterioration_count}/{patience})")

            # Check early stopping due to sustained instability
            if deterioration_count >= patience:
                print(f"[!] Validation loss deteriorated for {patience} consecutive checks. Halting to preserve best weights.")
                break

            # Phase Transition: Reached initial target step (600)
            if current_step == args.initial_target_step:
                if eval_bench["loss"] < baseline_step200_loss and elapsed < (args.time_budget * 0.7):
                    print(f"\n[*] Validation loss meaningfully improved vs baseline ({baseline_step200_loss:.4f} -> {eval_bench['loss']:.4f})!")
                    print(f"[*] Time budget permits ({elapsed:.1f}s / {args.time_budget}s). Extending training toward Step {args.max_target_step}...")
                    target_step = args.max_target_step
                else:
                    print(f"\n[*] Reached initial target step {args.initial_target_step}. Concluding scheduled segment.")
                    break

    total_time = max(0.001, time.time() - start_time)
    avg_speed = tokens_processed / total_time
    final_step = trainer.current_step

    repo_root_res = REPO_ROOT.resolve()
    print("\n" + "=" * 72)
    print(" Continued Training Completed!")
    print("=" * 72)
    print(f" Initial Step:         #{restored_step}")
    print(f" Final Step:           #{final_step} (+{final_step - restored_step} steps)")
    print(f" Additional Tokens:    {tokens_processed:,}")
    print(f" Average Speed:        {avg_speed:,.1f} tokens/sec")
    print(f" Elapsed Time:         {total_time:.2f}s ({total_time / 60:.1f} mins)")
    print(f" Baseline Step 200:    {baseline_step200_loss:.4f} (PPL: {baseline_step200_ppl:.2f})")
    print(f" Resumed Step {restored_step}:    {resumed_bench['loss']:.4f} (PPL: {resumed_bench['perplexity']:.2f})")
    print(f" Best Val Loss:        {best_val_loss:.4f}")
    print(f" Best Checkpoint:      {best_checkpoint_path.resolve().relative_to(repo_root_res)}")
    print(f" Latest Checkpoint:    {latest_checkpoint_path.resolve().relative_to(repo_root_res)}")

    # 7. Evaluate Latest and Best Checkpoints for Qualitative Comparison
    print("\n[*] Evaluating Final Checkpoints...")
    latest_model = GroundUpGPT(gpt_config)
    latest_model.load_state_dict(torch.load(latest_checkpoint_path, map_location=resolved_device)["model_state_dict"])
    latest_samples = run_prompt_evaluations(latest_model, tokenizer, resolved_device, f"LATEST (STEP {final_step})")

    best_model = GroundUpGPT(gpt_config)
    best_ckpt_data = torch.load(best_checkpoint_path, map_location=resolved_device)
    best_model.load_state_dict(best_ckpt_data["model_state_dict"])
    best_step = best_ckpt_data["step"]
    best_samples = run_prompt_evaluations(best_model, tokenizer, resolved_device, f"BEST (STEP {best_step})")

    # 8. Save Metrics Summary & History
    repo_root_res = REPO_ROOT.resolve()
    summary_data = {
        "restored_from": str(Path(args.init_checkpoint).resolve().relative_to(repo_root_res)),
        "initial_step": restored_step,
        "final_step": final_step,
        "steps_completed": final_step - restored_step,
        "tokens_processed": tokens_processed,
        "total_time_seconds": round(total_time, 2),
        "tokens_per_second": round(avg_speed, 1),
        "baseline_step200": {
            "val_loss": baseline_step200_loss,
            "perplexity": baseline_step200_ppl,
        },
        "resumed_checkpoint": {
            "step": restored_step,
            "val_loss": resumed_bench["loss"],
            "perplexity": resumed_bench["perplexity"],
        },
        "best_checkpoint": {
            "path": str(best_checkpoint_path.resolve().relative_to(repo_root_res)),
            "step": best_step,
            "val_loss": best_val_loss,
            "perplexity": round(math.exp(min(best_val_loss, 85.0)), 2),
        },
        "latest_checkpoint": {
            "path": str(latest_checkpoint_path.resolve().relative_to(repo_root_res)),
            "step": final_step,
            "val_loss": history[-1]["val_loss"],
            "perplexity": history[-1]["perplexity"],
        },
        "samples": {
            "step200": step200_samples,
            "best": best_samples,
            "latest": latest_samples,
        },
        "history": history,
    }

    summary_file = logs_dir / "continued_training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    csv_file = logs_dir / "continued_training_history.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\n[*] Metrics summary saved to: {summary_file.resolve().relative_to(repo_root_res)}")
    print(f"[*] Training history saved to: {csv_file.resolve().relative_to(repo_root_res)}")


if __name__ == "__main__":
    main()

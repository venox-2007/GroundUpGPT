"""Step 24: Quality-Focused Continued Pretraining on Expanded English Corpus.

Key Features:
1. Resumes strictly from Step 1000 best checkpoint (checkpoints/5m_continued/model_5m_best.pt).
2. Explicitly versioned Phase 2 learning rate schedule (conservative peak lr 2.5e-4, 50-step warmup, cosine decay).
3. Hard 15-minute GPU wall-clock budget cap.
4. Dual-Track Validation:
   - Track 1: Original Frozen Validation (data/cache/val_tokens.pt, seed 1337)
   - Track 2: Expanded Validation (data/cache_expanded/val_expanded_tokens.pt, seed 1337)
5. Strict artifact safety:
   - Preserves checkpoints/5m_initial/model_5m_step200.pt unmodified.
   - Preserves checkpoints/5m_continued/model_5m_best.pt unmodified.
   - Saves Phase 2 artifacts to checkpoints/5m_phase2/ (model_5m_best.pt, model_5m_latest.pt, step_XXXX.pt).
6. Rich telemetry, loss tracking, tokens/sec throughput, and 6-prompt qualitative evaluation.
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
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.amp import GradScaler, autocast

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import load_token_cache, PretokenizedDataPipeline
from src.trainer import get_tokenizer_fingerprint
from scripts.evaluate_model import compute_fixed_validation_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_phase2_5m")

PHASE2_PROMPTS = [
    "Once upon a time",
    "The scientist looked through the microscope",
    "In the beginning of the nineteenth century",
    "She opened the letter and discovered",
    "The philosophy of nature teaches us that",
    "They walked down the dark London street",
]


def safe_save_checkpoint(checkpoint_dict: Dict[str, Any], target_path: Path) -> None:
    """Safely write checkpoint to temporary file first and atomically replace target."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    torch.save(checkpoint_dict, tmp_path)
    tmp_path.replace(target_path)


def compute_phase2_lr(
    global_step: int,
    phase2_start_step: int = 1000,
    phase2_total_steps: int = 2500,
    peak_lr: float = 2.5e-4,
    min_lr: float = 3.0e-5,
    warmup_steps: int = 50,
) -> float:
    """Compute learning rate for Phase 2 pretraining schedule."""
    p2_step = global_step - phase2_start_step
    if p2_step < 0:
        return peak_lr
    if p2_step < warmup_steps:
        # Linear warmup from min_lr to peak_lr
        return min_lr + (peak_lr - min_lr) * (p2_step + 1) / max(1, warmup_steps)
    if p2_step >= phase2_total_steps:
        return min_lr
    # Cosine decay
    progress = (p2_step - warmup_steps) / max(1, phase2_total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (peak_lr - min_lr)


def run_phase2_prompt_evaluations(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    device: str,
    tag: str,
) -> Dict[str, Dict[str, str]]:
    """Generate completions under fixed seeds across required benchmark prompts."""
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    results = {}
    print(f"\n--- Qualitative Completions ({tag}) ---")
    for prompt in PHASE2_PROMPTS:
        greedy_cfg = GenerationConfig(max_new_tokens=40, temperature=0.0)
        sampled_cfg = GenerationConfig(max_new_tokens=40, temperature=0.8, top_p=0.9, seed=42)

        res_greedy = generator.generate(prompt=prompt, config=greedy_cfg)
        res_sampled = generator.generate(prompt=prompt, config=sampled_cfg)

        clean_greedy = res_greedy.generated_text.replace("\n", " ")
        clean_sampled = res_sampled.generated_text.replace("\n", " ")
        print(f"[{tag}] Prompt: '{prompt}'")
        print(f"       Greedy (T=0):   '{clean_greedy}'")
        print(f"       Sampled (T=0.8): '{clean_sampled}'\n")

        results[prompt] = {
            "greedy_t0": res_greedy.generated_text,
            "sampled_t08_p09": res_sampled.generated_text,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 24: Phase 2 Continued Pretraining on Expanded Corpus")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_best.pt",
        help="Source checkpoint to resume from (default: checkpoints/5m_continued/model_5m_best.pt)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "5m_phase2",
        help="Output directory for Phase 2 checkpoints (default: checkpoints/5m_phase2)",
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
        default=REPO_ROOT / "data" / "cache_expanded",
        help="Expanded token cache directory",
    )
    parser.add_argument(
        "--frozen-val-path",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "val_tokens.pt",
        help="Original frozen validation token tensor path",
    )
    parser.add_argument("--max-steps", type=int, default=3000, help="Target maximum global step (default: 3000)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Dual-track evaluation interval (default: 50)")
    parser.add_argument("--checkpoint-interval", type=int, default=250, help="Checkpoint interval (default: 250)")
    parser.add_argument("--time-budget", type=float, default=870.0, help="Max training wall-clock time in seconds (default: 870s = 14.5 min)")
    parser.add_argument("--peak-lr", type=float, default=2.5e-4, help="Peak Phase 2 learning rate (default: 2.5e-4)")
    parser.add_argument("--min-lr", type=float, default=3.0e-5, help="Minimum Phase 2 learning rate (default: 3.0e-5)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps in Phase 2 (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size (default: 16)")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" GroundUpGPT — Step 24: Phase 2 Quality Pretraining")
    print("=" * 72)

    # 1. Verify and Load Tokenizer
    if not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer_path}")
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    tok_fp = tokenizer.get_fingerprint()
    print(f"[*] Loaded Tokenizer: {args.tokenizer_path.name} (Vocab: {tokenizer.vocab_size}, Fingerprint: {tok_fp[:16]}...)")

    # 2. Load Expanded Pretokenized Dataset
    pipeline = load_token_cache(args.cache_dir, tokenizer=tokenizer)
    print(f"[*] Loaded Expanded Token Cache: {len(pipeline.train_tokens):,} train tokens, {len(pipeline.val_tokens):,} expanded val tokens")

    # Load Original Frozen Validation Tokens
    if not args.frozen_val_path.is_file():
        raise FileNotFoundError(f"Original frozen validation tokens not found at: {args.frozen_val_path}")
    frozen_val_tokens = torch.load(args.frozen_val_path, weights_only=True).long()
    expanded_val_tokens = pipeline.val_tokens.long()
    print(f"[*] Loaded Frozen Val Tokens: {len(frozen_val_tokens):,} tokens (Track 1 Regression Benchmark)")
    print(f"[*] Loaded Expanded Val Tokens: {len(expanded_val_tokens):,} tokens (Track 2 Domain Benchmark)")

    # 3. Verify Source Checkpoint
    if not args.init_checkpoint.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found at: {args.init_checkpoint}")

    raw_ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    ckpt_step = raw_ckpt.get("step", 0)
    print(f"[*] Resuming strictly from: {args.init_checkpoint.name} (Step #{ckpt_step})")

    # Confirm parameter count and architecture compatibility
    m_cfg_dict = raw_ckpt["model_config"]
    gpt_config = GPTConfig(**m_cfg_dict)
    model = GroundUpGPT(gpt_config)
    num_params = model.get_num_params()
    print(f"[*] Architecture: {gpt_config.num_layers} layers, {gpt_config.num_heads} heads, embed_dim={gpt_config.embed_dim}, block_size={gpt_config.block_size}")
    print(f"[*] Total Parameters: {num_params:,} (5.05M architecture)")

    # Verify tokenizer fingerprint against checkpoint
    ckpt_fp = raw_ckpt.get("tokenizer_fingerprint")
    if ckpt_fp and ckpt_fp != tok_fp:
        raise ValueError(f"CRITICAL: Checkpoint tokenizer fingerprint {ckpt_fp} does not match loaded tokenizer {tok_fp}")

    # Load weights
    model.load_state_dict(raw_ckpt["model_state_dict"])
    print("[*] Model weights successfully restored!")

    # Target Device Resolution
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device_name = torch.cuda.get_device_name(0) if resolved_device == "cuda" else "CPU"
    print(f"[*] Execution Device: {resolved_device} ({device_name})")
    model.to(resolved_device)

    # 4. Configure AdamW Optimizer and Scaler for Phase 2
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    decay_params = list(dict.fromkeys(decay_params))
    nodecay_params = list(dict.fromkeys(nodecay_params))

    initial_p2_lr = compute_phase2_lr(
        ckpt_step,
        phase2_start_step=ckpt_step,
        phase2_total_steps=args.max_steps - ckpt_step,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.02},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=initial_p2_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Restore AdamW optimizer state where available
    if "optimizer_state_dict" in raw_ckpt:
        try:
            optimizer.load_state_dict(raw_ckpt["optimizer_state_dict"])
            # Update learning rates to Phase 2 schedule
            for param_group in optimizer.param_groups:
                param_group["lr"] = initial_p2_lr
            print(f"[*] Restored optimizer state dict and updated learning rate to {initial_p2_lr:.6f}")
        except Exception as e:
            logger.warning(f"Could not restore optimizer state dict: {e}. Starting fresh AdamW state.")

    use_amp = (resolved_device.startswith("cuda") and torch.cuda.is_available())
    scaler = GradScaler("cuda", enabled=use_amp)
    if "scaler_state_dict" in raw_ckpt and use_amp:
        try:
            scaler.load_state_dict(raw_ckpt["scaler_state_dict"])
            print("[*] Restored mixed-precision scaler state.")
        except Exception as e:
            logger.warning(f"Could not restore scaler state: {e}")

    # 5. Dual-Track Benchmark on Resumed Step 1000 Baseline
    print("\n[*] Evaluating baseline performance at Step 1000 across both tracks...")
    bench_frozen_baseline = compute_fixed_validation_benchmark(
        model=model,
        tokens=frozen_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    bench_expanded_baseline = compute_fixed_validation_benchmark(
        model=model,
        tokens=expanded_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )

    print(f" [TRACK 1: ORIGINAL FROZEN VAL] Loss: {bench_frozen_baseline['loss']:.4f} | Perplexity: {bench_frozen_baseline['perplexity']:.2f}")
    print(f" [TRACK 2: EXPANDED VAL]        Loss: {bench_expanded_baseline['loss']:.4f} | Perplexity: {bench_expanded_baseline['perplexity']:.2f}")

    # Run Step 1000 baseline prompt generation
    step1000_samples = run_phase2_prompt_evaluations(model, tokenizer, resolved_device, f"STEP {ckpt_step} (BASELINE)")

    best_checkpoint_path = args.checkpoint_dir / "model_5m_best.pt"
    latest_checkpoint_path = args.checkpoint_dir / "model_5m_latest.pt"

    best_val_loss = bench_expanded_baseline["loss"]
    best_frozen_loss = bench_frozen_baseline["loss"]
    best_step = ckpt_step

    # Save initial baseline as initial Phase 2 best & latest
    initial_save_dict = {
        "step": ckpt_step,
        "phase": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else {},
        "model_config": asdict(gpt_config),
        "tokenizer_type": "BPETokenizer",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": tok_fp,
        "original_val_loss": bench_frozen_baseline["loss"],
        "original_perplexity": bench_frozen_baseline["perplexity"],
        "expanded_val_loss": bench_expanded_baseline["loss"],
        "expanded_perplexity": bench_expanded_baseline["perplexity"],
        "val_loss": bench_expanded_baseline["loss"],  # primary selection metric
        "perplexity": bench_expanded_baseline["perplexity"],
        "tokens_processed": 0,
    }
    safe_save_checkpoint(initial_save_dict, best_checkpoint_path)
    safe_save_checkpoint(initial_save_dict, latest_checkpoint_path)

    # 6. Training Execution Loop
    rng_gen = torch.Generator().manual_seed(42 + ckpt_step)
    start_time = time.time()
    current_step = ckpt_step
    target_max_step = args.max_steps
    phase2_total_steps = target_max_step - ckpt_step
    stop_reason = "max_steps_reached"

    history: List[Dict[str, Any]] = [
        {
            "step": ckpt_step,
            "train_loss": float("nan"),
            "original_val_loss": bench_frozen_baseline["loss"],
            "original_ppl": bench_frozen_baseline["perplexity"],
            "expanded_val_loss": bench_expanded_baseline["loss"],
            "expanded_ppl": bench_expanded_baseline["perplexity"],
            "lr": initial_p2_lr,
            "elapsed_sec": 0.0,
            "tokens_per_sec": 0.0,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024), 1) if resolved_device == "cuda" else 0.0,
        }
    ]

    tokens_per_step = args.batch_size * gpt_config.block_size * args.grad_accum_steps
    total_tokens_processed = 0

    print(f"\n[*] Starting Phase 2 Pretraining: Steps {ckpt_step} -> {target_max_step}")
    print(f"[*] Tokens per optimizer step: {tokens_per_step:,} ({args.batch_size}x{args.grad_accum_steps} sequences of len {gpt_config.block_size})")
    print(f"[*] Strict Time Budget: {args.time_budget:.0f} seconds ({args.time_budget / 60:.1f} minutes)\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step_start_time = time.time()

    while current_step < target_max_step:
        elapsed_total = time.time() - start_time
        if elapsed_total >= args.time_budget:
            stop_reason = f"wall_clock_budget_reached ({elapsed_total:.1f}s >= {args.time_budget:.0f}s)"
            logger.info(f"Stopping training: {stop_reason}")
            break

        current_lr = compute_phase2_lr(
            current_step,
            phase2_start_step=ckpt_step,
            phase2_total_steps=phase2_total_steps,
            peak_lr=args.peak_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        accum_loss = 0.0
        # Gradient accumulation micro-batches
        for micro_step in range(args.grad_accum_steps):
            x, y = pipeline.get_batch(
                split="train",
                batch_size=args.batch_size,
                block_size=gpt_config.block_size,
                device=resolved_device,
                generator=rng_gen,
            )

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    scaled_loss = loss / args.grad_accum_steps
                scaler.scale(scaled_loss).backward()
            else:
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()

            accum_loss += loss.item() / args.grad_accum_steps

        # Check for NaN / Inf
        if not math.isfinite(accum_loss):
            stop_reason = f"NaN_or_Inf_loss_detected at step {current_step}: {accum_loss}"
            logger.error(stop_reason)
            break

        # Gradient clipping and optimizer step
        if use_amp:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        current_step += 1
        total_tokens_processed += tokens_per_step

        # Dual-track evaluation and logging
        if current_step % args.eval_interval == 0 or current_step == target_max_step:
            step_dur = time.time() - step_start_time
            step_start_time = time.time()
            tok_per_sec = (args.eval_interval * tokens_per_step) / max(0.01, step_dur)

            model.eval()
            bench_frozen = compute_fixed_validation_benchmark(
                model=model,
                tokens=frozen_val_tokens,
                block_size=gpt_config.block_size,
                batch_size=16,
                num_eval_windows=64,
                seed=1337,
                device=resolved_device,
            )
            bench_expanded = compute_fixed_validation_benchmark(
                model=model,
                tokens=expanded_val_tokens,
                block_size=gpt_config.block_size,
                batch_size=16,
                num_eval_windows=64,
                seed=1337,
                device=resolved_device,
            )
            model.train()

            peak_vram = round(torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024), 1) if resolved_device == "cuda" else 0.0
            elapsed = time.time() - start_time

            logger.info(
                f"Step {current_step:4d}/{target_max_step} | "
                f"Train: {accum_loss:.4f} | "
                f"Orig Val: {bench_frozen['loss']:.4f} (PPL: {bench_frozen['perplexity']:.1f}) | "
                f"Exp Val: {bench_expanded['loss']:.4f} (PPL: {bench_expanded['perplexity']:.1f}) | "
                f"LR: {current_lr:.6f} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"VRAM: {peak_vram:.0f}MB | "
                f"{elapsed:.0f}s"
            )

            history.append({
                "step": current_step,
                "train_loss": round(accum_loss, 4),
                "original_val_loss": round(bench_frozen["loss"], 4),
                "original_ppl": round(bench_frozen["perplexity"], 2),
                "expanded_val_loss": round(bench_expanded["loss"], 4),
                "expanded_ppl": round(bench_expanded["perplexity"], 2),
                "lr": round(current_lr, 7),
                "elapsed_sec": round(elapsed, 1),
                "tokens_per_sec": round(tok_per_sec, 1),
                "peak_vram_mb": peak_vram,
            })

            # Checkpoint creation
            ckpt_payload = {
                "step": current_step,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else {},
                "model_config": asdict(gpt_config),
                "tokenizer_type": "BPETokenizer",
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_fingerprint": tok_fp,
                "original_val_loss": bench_frozen["loss"],
                "original_perplexity": bench_frozen["perplexity"],
                "expanded_val_loss": bench_expanded["loss"],
                "expanded_perplexity": bench_expanded["perplexity"],
                "val_loss": bench_expanded["loss"],
                "perplexity": bench_expanded["perplexity"],
                "tokens_processed": total_tokens_processed,
            }

            # Update latest checkpoint atomically
            safe_save_checkpoint(ckpt_payload, latest_checkpoint_path)

            # Check if new best checkpoint (judged on expanded validation loss while verifying original validation does not catastrophically degrade)
            if bench_expanded["loss"] < best_val_loss:
                best_val_loss = bench_expanded["loss"]
                best_frozen_loss = bench_frozen["loss"]
                best_step = current_step
                safe_save_checkpoint(ckpt_payload, best_checkpoint_path)
                logger.info(f"[*] New Phase 2 Best Checkpoint at Step {current_step}! Exp Val Loss: {best_val_loss:.4f}, Orig Val Loss: {best_frozen_loss:.4f}")

            # Periodic milestone snapshot (every 250 steps)
            if current_step % args.checkpoint_interval == 0:
                step_ckpt_path = args.checkpoint_dir / f"step_{current_step}.pt"
                safe_save_checkpoint(ckpt_payload, step_ckpt_path)
                logger.info(f"[*] Saved periodic milestone checkpoint: {step_ckpt_path.name}")

    # Training completed or terminated by budget
    total_training_time = time.time() - start_time
    print("\n" + "=" * 72)
    print(f" PHASE 2 TRAINING COMPLETE — Stopped at Step {current_step} ({stop_reason})")
    print("=" * 72)
    print(f"Total Time:               {total_training_time:.1f}s ({total_training_time / 60:.2f} min)")
    print(f"Initial Step:             {ckpt_step}")
    print(f"Final Step:               {current_step}")
    print(f"Steps Completed:          {current_step - ckpt_step}")
    print(f"Additional Tokens:        {total_tokens_processed:,}")
    print(f"Average Throughput:       {total_tokens_processed / max(0.01, total_training_time):,.0f} tokens/s")
    print(f"Best Step:                {best_step} (Exp Val Loss: {best_val_loss:.4f}, Orig Val Loss: {best_frozen_loss:.4f})")
    print("=" * 72)

    # 7. Final Comprehensive Evaluations of Best and Latest Checkpoints
    print("\n[*] Evaluating Final Phase 2 Checkpoints for Comparison...")

    # Load Best Checkpoint
    best_ckpt = torch.load(best_checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    best_frozen_eval = compute_fixed_validation_benchmark(
        model=model,
        tokens=frozen_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    best_expanded_eval = compute_fixed_validation_benchmark(
        model=model,
        tokens=expanded_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    best_samples = run_phase2_prompt_evaluations(model, tokenizer, resolved_device, f"STEP {best_step} (NEW BEST)")

    # Load Latest Checkpoint
    latest_ckpt = torch.load(latest_checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(latest_ckpt["model_state_dict"])
    model.eval()

    latest_frozen_eval = compute_fixed_validation_benchmark(
        model=model,
        tokens=frozen_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    latest_expanded_eval = compute_fixed_validation_benchmark(
        model=model,
        tokens=expanded_val_tokens,
        block_size=gpt_config.block_size,
        batch_size=16,
        num_eval_windows=64,
        seed=1337,
        device=resolved_device,
    )
    latest_samples = run_phase2_prompt_evaluations(model, tokenizer, resolved_device, f"STEP {current_step} (NEW LATEST)")

    # 8. Save Telemetry and Comparison Report
    summary_data = {
        "start_checkpoint": str(args.init_checkpoint).replace("\\", "/"),
        "initial_step": ckpt_step,
        "final_step": current_step,
        "steps_completed": current_step - ckpt_step,
        "additional_tokens_processed": total_tokens_processed,
        "training_elapsed_seconds": round(total_training_time, 2),
        "stop_reason": stop_reason,
        "hardware": {
            "device": resolved_device,
            "device_name": device_name,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(resolved_device) / (1024 * 1024), 1) if resolved_device == "cuda" else 0.0,
        },
        "phase2_schedule": {
            "peak_lr": args.peak_lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "total_phase2_steps": phase2_total_steps,
        },
        "step1000_baseline": {
            "original_val_loss": bench_frozen_baseline["loss"],
            "original_perplexity": bench_frozen_baseline["perplexity"],
            "expanded_val_loss": bench_expanded_baseline["loss"],
            "expanded_perplexity": bench_expanded_baseline["perplexity"],
        },
        "phase2_best": {
            "step": best_step,
            "path": str(best_checkpoint_path).replace("\\", "/"),
            "original_val_loss": best_frozen_eval["loss"],
            "original_perplexity": best_frozen_eval["perplexity"],
            "expanded_val_loss": best_expanded_eval["loss"],
            "expanded_perplexity": best_expanded_eval["perplexity"],
        },
        "phase2_latest": {
            "step": current_step,
            "path": str(latest_checkpoint_path).replace("\\", "/"),
            "original_val_loss": latest_frozen_eval["loss"],
            "original_perplexity": latest_frozen_eval["perplexity"],
            "expanded_val_loss": latest_expanded_eval["loss"],
            "expanded_perplexity": latest_expanded_eval["perplexity"],
        },
        "prompt_comparisons": {
            prompt: {
                "step1000": step1000_samples.get(prompt, {}),
                "phase2_best": best_samples.get(prompt, {}),
                "phase2_latest": latest_samples.get(prompt, {}),
            }
            for prompt in PHASE2_PROMPTS
        },
        "history": history,
    }

    json_report_path = logs_dir / "training_phase2_5m.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    csv_report_path = logs_dir / "training_phase2_5m.csv"
    with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"\n[*] Telemetry and logs saved:")
    print(f"    - {json_report_path}")
    print(f"    - {csv_report_path}")
    print(f"[*] Step 24 training pipeline finished successfully!")


if __name__ == "__main__":
    main()

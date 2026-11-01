"""First Monitored Pretraining Run of GroundUpGPT-27M (Step 32).

Features:
1. Strict 27,411,456-parameter GroundUpGPT architecture with PyTorch SDPA fast-path.
2. Fresh random initialization with deterministic seed (does NOT reuse old 5M or smoke-test weights).
3. Trained on the Step 31 expanded, diverse 50.3M-token corpus (data/cache_27m_v1/).
4. AdamW optimizer with selective weight decay (0.01 on >=2D tensors, 0.0 on 1D/biases).
5. Mixed precision fp16 with GradScaler and gradient norm clipping at 1.0.
6. 100-step linear warmup followed by cosine decay towards min_lr (scheduled for 6000 steps to permit future continuation).
7. Three independent validation tracks evaluated every 200 steps on fixed deterministic windows:
   - Track A: Combined validation corpus (data/cache_27m_v1/val_tokens.pt)
   - Track B: Original frozen Gutenberg validation (data/cache/val_tokens.pt)
   - Track C: Simple English Wikipedia held-out validation subset
8. Crash-safe stateful checkpointing (model_27m_best.pt, model_27m_latest.pt, step_XXXX.pt) in checkpoints/27m_phase1/.
9. Hard 15-minute GPU training wall-clock cap (or 1,200 optimizer steps).
10. Generation inspection on 8 fixed prompts at Step 0 and termination.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
from src.model import GPTConfig, GroundUpGPT, get_27m_config
from src.token_cache import PretokenizedDataPipeline, load_token_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_27m_phase1")

EXPECTED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_PARAM_COUNT = 27411456

INSPECTION_PROMPTS = [
    # Ordinary English sentence starters
    "The sun rose over the",
    "In the middle of the room,",
    # Factual / expository starters
    "Biology is the study of",
    "The capital of France is",
    # Narrative starters
    "Once upon a time in a small village,",
    "He opened the mysterious door and saw",
    # Question-like starters
    "What is the reason why",
    "How do birds fly through the",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First Monitored Pretraining of GroundUpGPT-27M.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_27m_v1"), help="Pretraining token cache directory.")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/bpe_vocab4096.json"), help="BPE tokenizer path.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/27m_phase1"), help="Output checkpoint directory.")
    parser.add_argument("--log-path", type=Path, default=Path("logs/training_27m_phase1.json"), help="Telemetry JSON output path.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint path to resume training from.")
    parser.add_argument("--max-steps", type=int, default=1200, help="Maximum optimizer steps (default: 1200).")
    parser.add_argument("--max-time-seconds", type=float, default=900.0, help="Maximum training wall time in seconds (default: 900s = 15m).")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Sequences per micro-batch (default: 8).")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch: 32 seqs = 8192 tokens).")
    parser.add_argument("--block-size", type=int, default=256, help="Sequence context length (default: 256).")
    parser.add_argument("--peak-lr", type=float, default=3e-4, help="Peak learning rate after warmup (default: 3e-4).")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate (default: 3e-5).")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Linear warmup steps (default: 100).")
    parser.add_argument("--total-scheduled-steps", type=int, default=6000, help="Total steps for cosine decay schedule (default: 6000).")
    parser.add_argument("--eval-interval", type=int, default=200, help="Validation frequency in steps (default: 200).")
    parser.add_argument("--checkpoint-interval", type=int, default=200, help="Recovery checkpoint frequency (default: 200).")
    parser.add_argument("--eval-windows", type=int, default=48, help="Number of fixed windows per evaluation track (default: 48).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay on 2D weights (default: 0.01).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip threshold (default: 1.0).")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--use-sdpa", action="store_true", default=True, help="Use PyTorch SDPA attention fast-path.")
    return parser.parse_args()


def compute_learning_rate(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate using linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def save_crash_safe_checkpoint(state: Dict[str, Any], target_path: Path) -> None:
    """Save a checkpoint atomically using a temporary file and atomic rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(target_path)


def evaluate_fixed_track(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Evaluate loss and perplexity on fixed deterministic evaluation windows."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    use_cuda = "cuda" in device

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack([tokens[j : j + block_size] for j in batch_idx]).to(device, dtype=torch.long)
            y = torch.stack([tokens[j + 1 : j + block_size + 1] for j in batch_idx]).to(device, dtype=torch.long)
            with autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                _, loss = model(x, y)
            total_loss += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

    model.train()
    mean_loss = total_loss / max(1, total_samples)
    ppl = math.exp(min(20.0, mean_loss))
    return round(mean_loss, 4), round(ppl, 2)


def generate_inspection_samples(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    prompts: List[str],
    device: str,
    seed: int = 1337,
) -> List[Dict[str, str]]:
    """Generate deterministic qualitative text samples for inspection."""
    gen = TextGenerator(model, tokenizer=tokenizer, device=device)
    cfg = GenerationConfig(
        max_new_tokens=40,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=seed,
    )
    results = []
    for prompt in prompts:
        out = gen.generate(prompt, config=cfg)
        gen_text = out.generated_text if hasattr(out, "generated_text") else str(out)
        results.append({"prompt": prompt, "completion": gen_text})
    return results


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print(" GROUNDUPGPT-27M: FIRST MONITORED PRETRAINING RUN (STEP 32)")
    print("=" * 65)

    # 1. Resolve Device
    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device
    device_name = torch.cuda.get_device_name(0) if resolved_device.startswith("cuda") else "CPU"
    print(f"[*] Compute Device:        {resolved_device} ({device_name})")

    # 2. Verify Tokenizer and Dataset
    assert args.tokenizer_path.exists(), f"Tokenizer missing: {args.tokenizer_path}"
    tokenizer = BPETokenizer().load(str(args.tokenizer_path))
    tok_fp = tokenizer.get_fingerprint()
    print(f"[*] Tokenizer Fingerprint: {tok_fp}")
    assert tok_fp == EXPECTED_TOKENIZER_FP, f"Tokenizer FP mismatch: {tok_fp} vs {EXPECTED_TOKENIZER_FP}"

    print(f"[*] Loading Pretraining Cache from: {args.cache_dir}")
    pipeline = load_token_cache(args.cache_dir, tokenizer=tokenizer)
    meta = pipeline.metadata or {}
    total_train_tokens = len(pipeline.train_tokens)
    total_val_tokens = len(pipeline.val_tokens)
    print(f"[*] Dataset Train Tokens:  {total_train_tokens:,}")
    print(f"[*] Dataset Val Tokens:    {total_val_tokens:,}")

    # 3. Load Validation Tracks
    # Track A: Combined validation (data/cache_27m_v1/val_tokens.pt)
    track_A_tokens = pipeline.val_tokens

    # Track B: Original frozen Gutenberg validation
    orig_val_path = Path("data/cache/val_tokens.pt")
    if orig_val_path.exists():
        track_B_tokens = torch.load(orig_val_path, weights_only=True)
    else:
        track_B_tokens = track_A_tokens[:168741]
    print(f"[*] Track B (Orig Gutenberg Val): {len(track_B_tokens):,} tokens")

    # Track C: Simple English Wikipedia held-out subset
    # In Step 31, Gutenberg val tokens are the first 1,118,258 tokens; Wikipedia is the remainder
    gt_val_offset = meta.get("sources", {}).get("project_gutenberg", {}).get("val_tokens", 1118258)
    track_C_tokens = track_A_tokens[gt_val_offset:]
    print(f"[*] Track C (Wikipedia Val):      {len(track_C_tokens):,} tokens")

    # Pre-generate fixed deterministic evaluation indices (seed 42)
    eval_rng = torch.Generator().manual_seed(42)
    indices_A = torch.randint(0, len(track_A_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)
    indices_B = torch.randint(0, len(track_B_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)
    indices_C = torch.randint(0, len(track_C_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)

    # 4. Initialize or Resume Model
    gpt_config = get_27m_config(block_size=args.block_size, dropout=0.1, use_sdpa=args.use_sdpa)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = GroundUpGPT(gpt_config)
    num_params = model.get_num_params()
    print(f"[*] Model Architecture:    {gpt_config.num_layers}L / {gpt_config.num_heads}H / d={gpt_config.embed_dim} / ctx={gpt_config.block_size}")
    print(f"[*] Unique Parameters:     {num_params:,} (verified 27.41M)")
    assert num_params == EXPECTED_PARAM_COUNT, f"Expected {EXPECTED_PARAM_COUNT} params, got {num_params}"

    model.to(resolved_device)

    # 5. Optimizer, Precision, and Scaler
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    decay_params = list(dict.fromkeys(decay_params))
    nodecay_params = list(dict.fromkeys(nodecay_params))

    initial_lr = compute_learning_rate(
        0, args.warmup_steps, args.total_scheduled_steps, args.peak_lr, args.min_lr
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=initial_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    use_amp = resolved_device.startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    start_step = 0
    tokens_processed = 0
    best_val_loss = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []

    # Optional Resume
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=resolved_device, weights_only=False)
        start_step = ckpt["step"]
        tokens_processed = ckpt.get("tokens_processed", start_step * args.micro_batch_size * args.grad_accum * args.block_size)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_step = ckpt.get("best_step", start_step)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp and ckpt["scaler_state_dict"]:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"[*] Resumed from checkpoint: {args.resume_from} at step {start_step} (best loss: {best_val_loss:.4f})")

    # 6. Micro-Preflight Smoke Test on Device
    print("[*] Running 2-step device micro-preflight smoke test...")
    model.train()
    for _ in range(2):
        dummy_x = torch.randint(0, 4096, (args.micro_batch_size, args.block_size), device=resolved_device)
        dummy_y = torch.randint(0, 4096, (args.micro_batch_size, args.block_size), device=resolved_device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            _, dummy_loss = model(dummy_x, dummy_y)
        assert torch.isfinite(dummy_loss), "Preflight loss is non-finite!"
        scaler.scale(dummy_loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(gn), "Preflight grad norm is non-finite!"
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[*] Preflight smoke test passed successfully!")

    # 7. Initial Baseline Evaluation (Step 0) & Inspection
    print("\n[*] Evaluating initial baseline (Step 0) across all 3 tracks...")
    loss_A, ppl_A = evaluate_fixed_track(model, track_A_tokens, indices_A, args.block_size, device=resolved_device)
    loss_B, ppl_B = evaluate_fixed_track(model, track_B_tokens, indices_B, args.block_size, device=resolved_device)
    loss_C, ppl_C = evaluate_fixed_track(model, track_C_tokens, indices_C, args.block_size, device=resolved_device)

    print(f"    -> Track A (Combined Val):  Loss: {loss_A:.4f} | PPL: {ppl_A:.2f}")
    print(f"    -> Track B (Orig Gutenberg): Loss: {loss_B:.4f} | PPL: {ppl_B:.2f}")
    print(f"    -> Track C (Wikipedia Val):  Loss: {loss_C:.4f} | PPL: {ppl_C:.2f}")

    print("\n[*] Generating initial text inspection samples at Step 0...")
    initial_samples = generate_inspection_samples(model, tokenizer, INSPECTION_PROMPTS, device=resolved_device, seed=args.seed)
    for s in initial_samples[:3]:
        print(f"    Prompt: '{s['prompt']}' -> '{repr(s['completion'])}'")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    # 8. Main Training Loop
    print(f"\n[*] Starting Pretraining Run: target {args.max_steps} steps, budget {args.max_time_seconds:.0f}s (15 min)...")
    print(f"[*] Effective Batch: {args.micro_batch_size * args.grad_accum} sequences = {args.micro_batch_size * args.grad_accum * args.block_size} tokens/step")

    train_start_time = time.perf_counter()
    tokens_per_step = args.micro_batch_size * args.grad_accum * args.block_size
    stop_reason = "step_limit"

    data_rng = torch.Generator().manual_seed(args.seed + start_step)
    last_log_time = time.perf_counter()
    recent_tokens = 0

    step = start_step

    while step < args.max_steps:
        # Check wall-clock budget
        elapsed_total = time.perf_counter() - train_start_time
        if elapsed_total >= args.max_time_seconds:
            logger.info(f"Stopping at step {step}: reached 15-minute GPU time budget ({elapsed_total:.1f}s).")
            stop_reason = "time_limit"
            break

        step += 1
        step_start_time = time.perf_counter()

        # Update Learning Rate
        curr_lr = compute_learning_rate(
            step, args.warmup_steps, args.total_scheduled_steps, args.peak_lr, args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # Micro-batch accumulation
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_idx in range(args.grad_accum):
            x, y = pipeline.get_batch("train", batch_size=args.micro_batch_size, block_size=args.block_size, device=resolved_device)
            with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                logger.error(f"FATAL: Non-finite loss ({loss.item()}) at step {step}, micro {micro_idx}!")
                stop_reason = "loss_nan"
                break

            scaler.scale(scaled_loss).backward()
            accum_loss += loss.item()

        if stop_reason == "loss_nan":
            break

        accum_loss /= args.grad_accum

        # Gradient clipping and step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if not torch.isfinite(grad_norm):
            logger.warning(f"Non-finite grad norm ({grad_norm.item()}) at step {step}, skipping optimizer update.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step
        recent_tokens += tokens_per_step

        step_elapsed = time.perf_counter() - step_start_time
        recent_tok_sec = tokens_per_step / max(1e-5, step_elapsed)

        # Logging
        if step % 20 == 0 or step == 1 or step == args.max_steps:
            now = time.perf_counter()
            rolling_elapsed = now - last_log_time
            rolling_tok_s = recent_tokens / max(1e-5, rolling_elapsed)
            last_log_time = now
            recent_tokens = 0

            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            logger.info(
                f"Step {step:4d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | "
                f"Grad: {grad_norm.item():.3f} | {rolling_tok_s:,.0f} tok/s | VRAM: {vram_mb:.0f}MB | "
                f"Time: {time.perf_counter() - train_start_time:.1f}s"
            )

        # Periodic Validation & Checkpointing
        if step % args.eval_interval == 0:
            val_loss_A, val_ppl_A = evaluate_fixed_track(model, track_A_tokens, indices_A, args.block_size, device=resolved_device)
            val_loss_B, val_ppl_B = evaluate_fixed_track(model, track_B_tokens, indices_B, args.block_size, device=resolved_device)
            val_loss_C, val_ppl_C = evaluate_fixed_track(model, track_C_tokens, indices_C, args.block_size, device=resolved_device)

            logger.info(
                f"--> EVAL STEP {step:4d} | Track A: {val_loss_A:.4f} (PPL {val_ppl_A:.1f}) | "
                f"Track B: {val_loss_B:.4f} (PPL {val_ppl_B:.1f}) | "
                f"Track C: {val_loss_C:.4f} (PPL {val_ppl_C:.1f})"
            )

            is_best = val_loss_A < best_val_loss
            if is_best:
                best_val_loss = val_loss_A
                best_step = step

            ckpt_payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "model_config": asdict(gpt_config),
                "tokens_processed": tokens_processed,
                "dataset_fingerprint": meta.get("tokenizer_fingerprint", tok_fp),
                "tokenizer_fingerprint": tok_fp,
                "rng_states": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                "lr": curr_lr,
                "train_loss": accum_loss,
                "val_metrics": {
                    "track_A_combined_loss": val_loss_A,
                    "track_A_combined_ppl": val_ppl_A,
                    "track_B_orig_gt_loss": val_loss_B,
                    "track_B_orig_gt_ppl": val_ppl_B,
                    "track_C_wiki_loss": val_loss_C,
                    "track_C_wiki_ppl": val_ppl_C,
                },
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "resumable": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Periodic recovery checkpoint
            if step % args.checkpoint_interval == 0:
                save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / f"step_{step}.pt")

            # Always save latest
            save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / "model_27m_latest.pt")

            # Save best if achieved
            if is_best:
                save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / "model_27m_best.pt")
                logger.info(f"[*] NEW BEST MODEL saved at step {step} (Track A loss: {best_val_loss:.4f})")

            history.append({
                "step": step,
                "train_loss": round(accum_loss, 4),
                "lr": curr_lr,
                "grad_norm": round(grad_norm.item(), 4),
                "tokens_processed": tokens_processed,
                "val_metrics": ckpt_payload["val_metrics"],
                "wall_time_s": round(time.perf_counter() - train_start_time, 2),
            })

    total_training_wall_time = time.perf_counter() - train_start_time

    # 9. Final Evaluation at Termination
    print(f"\n[*] Final Evaluation at termination (Step {step})...")
    final_loss_A, final_ppl_A = evaluate_fixed_track(model, track_A_tokens, indices_A, args.block_size, device=resolved_device)
    final_loss_B, final_ppl_B = evaluate_fixed_track(model, track_B_tokens, indices_B, args.block_size, device=resolved_device)
    final_loss_C, final_ppl_C = evaluate_fixed_track(model, track_C_tokens, indices_C, args.block_size, device=resolved_device)

    print(f"    -> Track A (Combined Val):  Loss: {final_loss_A:.4f} | PPL: {final_ppl_A:.2f}")
    print(f"    -> Track B (Orig Gutenberg): Loss: {final_loss_B:.4f} | PPL: {final_ppl_B:.2f}")
    print(f"    -> Track C (Wikipedia Val):  Loss: {final_loss_C:.4f} | PPL: {final_ppl_C:.2f}")

    if final_loss_A < best_val_loss:
        best_val_loss = final_loss_A
        best_step = step

    final_ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "model_config": asdict(gpt_config),
        "tokens_processed": tokens_processed,
        "dataset_fingerprint": meta.get("tokenizer_fingerprint", tok_fp),
        "tokenizer_fingerprint": tok_fp,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        "lr": compute_learning_rate(step, args.warmup_steps, args.total_scheduled_steps, args.peak_lr, args.min_lr),
        "train_loss": accum_loss,
        "val_metrics": {
            "track_A_combined_loss": final_loss_A,
            "track_A_combined_ppl": final_ppl_A,
            "track_B_orig_gt_loss": final_loss_B,
            "track_B_orig_gt_ppl": final_ppl_B,
            "track_C_wiki_loss": final_loss_C,
            "track_C_wiki_ppl": final_ppl_C,
        },
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "resumable": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_crash_safe_checkpoint(final_ckpt, args.checkpoint_dir / "model_27m_latest.pt")
    if final_loss_A <= best_val_loss:
        save_crash_safe_checkpoint(final_ckpt, args.checkpoint_dir / "model_27m_best.pt")

    # 10. Generate Final Inspection Samples
    print("\n[*] Generating qualitative inspection samples from final model...")
    final_samples = generate_inspection_samples(model, tokenizer, INSPECTION_PROMPTS, device=resolved_device, seed=args.seed)
    for s in final_samples:
        print(f"    [{s['prompt']}] -> {repr(s['completion'])}")

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
    avg_tok_sec = tokens_processed / max(1e-5, total_training_wall_time)
    dataset_epochs = tokens_processed / total_train_tokens

    telemetry = {
        "model_architecture": "GroundUpGPT-27M",
        "parameters": num_params,
        "stop_reason": stop_reason,
        "completed_steps": step,
        "tokens_processed": tokens_processed,
        "dataset_epochs": round(dataset_epochs, 4),
        "wall_time_seconds": round(total_training_wall_time, 2),
        "average_tokens_per_sec": round(avg_tok_sec, 1),
        "peak_vram_allocated_mb": round(peak_allocated_mb, 2),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 2),
        "initial_val_metrics": {
            "track_A_combined_loss": loss_A,
            "track_A_combined_ppl": ppl_A,
            "track_B_orig_gt_loss": loss_B,
            "track_B_orig_gt_ppl": ppl_B,
            "track_C_wiki_loss": loss_C,
            "track_C_wiki_ppl": ppl_C,
        },
        "final_train_loss": round(accum_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_step": best_step,
        "final_val_metrics": {
            "track_A_combined_loss": final_loss_A,
            "track_A_combined_ppl": final_ppl_A,
            "track_B_orig_gt_loss": final_loss_B,
            "track_B_orig_gt_ppl": final_ppl_B,
            "track_C_wiki_loss": final_loss_C,
            "track_C_wiki_ppl": final_ppl_C,
        },
        "step_history": history,
        "initial_samples": initial_samples,
        "final_samples": final_samples,
    }

    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print("\n" + "=" * 65)
    print(" TRAINING RUN COMPLETE")
    print("=" * 65)
    print(f"Completed Steps:     {step} / {args.max_steps} ({stop_reason})")
    print(f"Wall-Clock Time:     {total_training_wall_time:.1f}s (budget: {args.max_time_seconds:.0f}s)")
    print(f"Tokens Processed:    {tokens_processed:,} ({dataset_epochs:.2%} dataset epoch)")
    print(f"Average Throughput:  {avg_tok_sec:,.0f} tok/s")
    print(f"Peak VRAM:           {peak_allocated_mb:.1f} MB allocated / {peak_reserved_mb:.1f} MB reserved")
    print(f"Track A (Combined):  {loss_A:.4f} -> {final_loss_A:.4f} (PPL: {ppl_A:.1f} -> {final_ppl_A:.1f}, Best: {best_val_loss:.4f} @ step {best_step})")
    print(f"Track B (Orig GT):   {loss_B:.4f} -> {final_loss_B:.4f} (PPL: {ppl_B:.1f} -> {final_ppl_B:.1f})")
    print(f"Track C (Wikipedia): {loss_C:.4f} -> {final_loss_C:.4f} (PPL: {ppl_C:.1f} -> {final_ppl_C:.1f})")
    print(f"Saved Checkpoints:   {args.checkpoint_dir / 'model_27m_best.pt'}")
    print(f"Telemetry Log:       {args.log_path}")
    print("=" * 65 + "\n")

    return telemetry


if __name__ == "__main__":
    run_training(parse_args())

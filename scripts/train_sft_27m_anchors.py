"""Step 40: Targeted conversational fine-tuning of GroundUpGPT-27M with short-answer anchors.

Initializes from Step 600 SFT best checkpoint (checkpoints/27m_sft_phase1/model_27m_sft_best.pt).
Uses fresh AdamW optimizer state and blended sampling combining Dolly 15k and verified anchors.
Tracks anchor val loss, Dolly val loss, and frozen LM validation.
Enforces strict 8-minute GPU wall-clock limit and saves checkpoints atomically.

Usage:
    .venv/Scripts/python scripts/train_sft_27m_anchors.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    BlendedSFTDataPipeline,
    collate_sft_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPECTED_BASE_SHA256 = "79346b19bbd9327fa7ddec04b32744977ab7bdbf694c1fa1382003087cec7b7d"
EXPECTED_TOKENIZER_FINGERPRINT = (
    "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
)


def compute_sha256(file_path: Path) -> str:
    """Compute exact SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_dataset_loss(
    model: GroundUpGPT,
    examples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[float, float, int]:
    """Compute exact token-weighted assistant-only loss on an SFT dataset."""
    model.eval()
    total_loss_sum = 0.0
    total_active_tokens = 0
    use_cuda = (device.type == "cuda")

    with torch.inference_mode():
        for i in range(0, len(examples), batch_size):
            batch_items = [examples[j] for j in range(i, min(i + batch_size, len(examples)))]
            collated = collate_sft_batch(
                batch_items,
                pad_token_id=0,
                pad_to_max_len=False,
                max_seq_len=model.config.block_size,
            )
            x = collated["input_ids"].to(device)
            y = collated["target_ids"].to(device)
            mask = collated["loss_mask"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits, _ = model(x)
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
            total_loss_sum += loss_sum.item()
            total_active_tokens += int(mask.sum().item())

    loss = total_loss_sum / max(1, total_active_tokens)
    ppl = math.exp(min(20.0, loss))
    return loss, ppl, total_active_tokens


def evaluate_pretraining_lm(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    device: torch.device,
    num_windows: int = 48,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """Compute pretraining language modeling loss on fixed windows."""
    model.eval()
    block_size = model.config.block_size
    use_cuda = (device.type == "cuda")
    tot_loss = 0.0
    tot_tokens = 0

    # Deterministic stride across tokens
    stride = max(1, (len(tokens) - block_size - 1) // num_windows)
    indices = [i * stride for i in range(num_windows)]

    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack([tokens[j : j + block_size].long() for j in batch_idx]).to(device)
            y = torch.stack([tokens[j + 1 : j + block_size + 1].long() for j in batch_idx]).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits, _ = model(x)
                loss_sum = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            tot_loss += loss_sum.item()
            tot_tokens += y.numel()

    loss = tot_loss / max(1, tot_tokens)
    ppl = math.exp(min(20.0, loss))
    return loss, ppl


def save_checkpoint_atomic(state: Dict[str, Any], path: Path) -> None:
    """Save checkpoint atomically using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    if path.exists():
        path.unlink()
    tmp_path.rename(path)


def run_targeted_sft(
    base_checkpoint_path: Path,
    dolly_cache_dir: Path,
    anchor_cache_dir: Path,
    checkpoint_dir: Path,
    tokenizer_path: Path,
    lm_val_path: Path,
    log_dir: Path,
    max_steps: int = 250,
    val_interval: int = 50,
    peak_lr: float = 2.0e-5,
    min_lr: float = 2.0e-6,
    warmup_steps: int = 20,
    weight_decay: float = 0.01,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    anchor_sample_prob: float = 0.55,
    max_wall_seconds: float = 450.0,  # 7.5 min safety cap
    seed: int = 42,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Execute bounded targeted SFT training run."""
    device = torch.device(device_name)
    logger.info("=" * 70)
    logger.info("Starting GroundUpGPT-27M Targeted Conversational Fine-Tuning")
    logger.info(f"Device: {device}")
    logger.info(f"Target steps: {max_steps} | Effective batch: {micro_batch_size * grad_accum_steps}")
    logger.info(f"Peak LR: {peak_lr} | Anchor sample prob: {anchor_sample_prob}")
    logger.info(f"Wall-clock safety limit: {max_wall_seconds:.1f}s")
    logger.info("=" * 70)

    # 1. Verify tokenizer
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    assert tokenizer.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT, "Tokenizer fingerprint mismatch!"

    # 2. Verify source checkpoint
    actual_base_sha256 = compute_sha256(base_checkpoint_path)
    assert actual_base_sha256 == EXPECTED_BASE_SHA256, f"Base checkpoint SHA-256 mismatch! Got {actual_base_sha256}"
    ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)

    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 27411456, f"Expected 27,411,456 params, got {total_params}"

    prev_sft_step = ckpt.get("sft_step", 600)
    source_pretrain_step = ckpt.get("source_step", 5949)

    # 3. Load blended data pipeline
    pipeline = BlendedSFTDataPipeline.load(
        dolly_cache_dir=dolly_cache_dir,
        anchor_cache_dir=anchor_cache_dir,
        tokenizer=tokenizer,
        anchor_sample_prob=anchor_sample_prob,
        seed=seed,
    )
    logger.info(f"Blended pipeline loaded: {len(pipeline.dolly_pipeline.train_dataset)} Dolly train, {len(pipeline.anchor_train_examples)} Anchor train.")

    # 4. Load LM validation tokens
    lm_tokens = torch.load(lm_val_path, weights_only=True)
    if isinstance(lm_tokens, dict) and "tokens" in lm_tokens:
        lm_tokens = lm_tokens["tokens"]

    # 5. Baseline evaluations before training
    logger.info("Running baseline evaluations on untouched checkpoint...")
    base_dolly_loss, base_dolly_ppl, _ = evaluate_dataset_loss(model, pipeline.val_dataset, device)
    base_anchor_loss, base_anchor_ppl, _ = evaluate_dataset_loss(model, pipeline.val_anchor_dataset, device)
    base_transfer_loss, base_transfer_ppl, _ = evaluate_dataset_loss(model, pipeline.test_transfer_dataset, device)
    base_lm_loss, base_lm_ppl = evaluate_pretraining_lm(model, lm_tokens, device)

    logger.info(f"Baseline Dolly Val Loss: {base_dolly_loss:.4f} (PPL: {base_dolly_ppl:.2f})")
    logger.info(f"Baseline Anchor Val Loss: {base_anchor_loss:.4f} (PPL: {base_anchor_ppl:.2f})")
    logger.info(f"Baseline Transfer Test Loss: {base_transfer_loss:.4f} (PPL: {base_transfer_ppl:.2f})")
    logger.info(f"Baseline Pretrain LM Loss: {base_lm_loss:.4f} (PPL: {base_lm_ppl:.2f})")

    baseline_metrics = {
        "step": 0,
        "base_checkpoint_sha256": actual_base_sha256,
        "dolly_val_loss": base_dolly_loss,
        "dolly_val_ppl": base_dolly_ppl,
        "anchor_val_loss": base_anchor_loss,
        "anchor_val_ppl": base_anchor_ppl,
        "transfer_test_loss": base_transfer_loss,
        "transfer_test_ppl": base_transfer_ppl,
        "pretrain_lm_loss": base_lm_loss,
        "pretrain_lm_ppl": base_lm_ppl,
    }
    with open(log_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    # 6. Fresh optimizer and scheduler setup
    # Filter weight decay for LayerNorm and biases
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=peak_lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def get_lr(it: int) -> float:
        if it < warmup_steps:
            return peak_lr * (it + 1) / (warmup_steps + 1)
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (peak_lr - min_lr)

    # 7. Training loop
    model.train()
    t_start = time.time()
    successful_steps = 0
    micro_batches_done = 0
    best_anchor_val_loss = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []

    logger.info("Starting training optimization...")

    while successful_steps < max_steps:
        elapsed = time.time() - t_start
        if elapsed >= max_wall_seconds:
            logger.warning(f"Wall-clock safety limit reached ({elapsed:.1f}s >= {max_wall_seconds:.1f}s). Stopping.")
            break

        lr = get_lr(successful_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        total_active_tokens_in_step = 0

        # Sample micro-batches for this optimizer step
        micro_batches = []
        for _ in range(grad_accum_steps):
            bx, by, bmask = pipeline.get_batch(split="train", batch_size=micro_batch_size, device=device)
            micro_batches.append((bx, by, bmask))
            total_active_tokens_in_step += int(bmask.sum().item())

        denom = max(1, total_active_tokens_in_step)

        for bx, by, bmask in micro_batches:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = model(bx)
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    by.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                loss = loss_sum / denom

            scaler.scale(loss).backward()
            accum_loss += loss_sum.item()
            micro_batches_done += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        if scale_after >= scale_before:
            # Step succeeded
            successful_steps += 1
            train_loss = accum_loss / denom

            if successful_steps % 10 == 0 or successful_steps == 1:
                cur_time = time.time() - t_start
                tok_sec = (micro_batches_done * micro_batch_size * 256) / max(0.1, cur_time)
                logger.info(
                    f"Step {successful_steps:3d}/{max_steps} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad Norm: {grad_norm:.2f} | "
                    f"Speed: {tok_sec:.0f} tok/s | "
                    f"Time: {cur_time:.1f}s"
                )

            # Periodic validation
            if successful_steps % val_interval == 0 or successful_steps == max_steps:
                val_anchor_loss, val_anchor_ppl, _ = evaluate_dataset_loss(
                    model, pipeline.val_anchor_dataset, device
                )
                val_dolly_loss, val_dolly_ppl, _ = evaluate_dataset_loss(
                    model, pipeline.val_dataset, device
                )
                val_transfer_loss, val_transfer_ppl, _ = evaluate_dataset_loss(
                    model, pipeline.test_transfer_dataset, device
                )
                val_lm_loss, val_lm_ppl = evaluate_pretraining_lm(
                    model, lm_tokens, device
                )

                dolly_regression_pct = ((val_dolly_loss - base_dolly_loss) / base_dolly_loss) * 100.0
                exposure_stats = pipeline.get_exposure_summary()

                logger.info("-" * 65)
                logger.info(
                    f"[Eval @ Step {successful_steps}] Anchor Val Loss: {val_anchor_loss:.4f} (PPL {val_anchor_ppl:.2f}) | "
                    f"Dolly Val Loss: {val_dolly_loss:.4f} (Δ {dolly_regression_pct:+.1f}%) | "
                    f"Transfer Loss: {val_transfer_loss:.4f} | "
                    f"LM Loss: {val_lm_loss:.4f}"
                )
                logger.info("-" * 65)

                step_record = {
                    "step": successful_steps,
                    "train_loss": train_loss,
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "anchor_val_loss": val_anchor_loss,
                    "anchor_val_ppl": val_anchor_ppl,
                    "dolly_val_loss": val_dolly_loss,
                    "dolly_val_ppl": val_dolly_ppl,
                    "dolly_regression_pct": dolly_regression_pct,
                    "transfer_test_loss": val_transfer_loss,
                    "transfer_test_ppl": val_transfer_ppl,
                    "pretrain_lm_loss": val_lm_loss,
                    "pretrain_lm_ppl": val_lm_ppl,
                    "elapsed_seconds": time.time() - t_start,
                    "exposure_stats": exposure_stats,
                }
                history.append(step_record)

                # Checkpoint selection: Improved anchor validation AND Dolly loss not regressed > 15%
                is_best = False
                if dolly_regression_pct <= 15.0 and val_anchor_loss < best_anchor_val_loss:
                    best_anchor_val_loss = val_anchor_loss
                    best_step = successful_steps
                    is_best = True

                # Save checkpoints
                ckpt_meta = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "model_config": ckpt["model_config"],
                    "parent_checkpoint_path": str(base_checkpoint_path),
                    "parent_checkpoint_sha256": actual_base_sha256,
                    "source_pretrain_step": source_pretrain_step,
                    "prev_sft_step": prev_sft_step,
                    "anchor_sft_step": successful_steps,
                    "global_step": source_pretrain_step + prev_sft_step + successful_steps,
                    "best_step": best_step,
                    "anchor_val_loss": val_anchor_loss,
                    "anchor_val_ppl": val_anchor_ppl,
                    "dolly_val_loss": val_dolly_loss,
                    "dolly_val_ppl": val_dolly_ppl,
                    "transfer_test_loss": val_transfer_loss,
                    "transfer_test_ppl": val_transfer_ppl,
                    "pretrain_lm_loss": val_lm_loss,
                    "pretrain_lm_ppl": val_lm_ppl,
                    "exposure_stats": exposure_stats,
                    "tokenizer_fingerprint": EXPECTED_TOKENIZER_FINGERPRINT,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_checkpoint_atomic(ckpt_meta, checkpoint_dir / "model_27m_anchors_latest.pt")
                if is_best:
                    save_checkpoint_atomic(ckpt_meta, checkpoint_dir / "model_27m_anchors_best.pt")
                    logger.info(f"[*] New best checkpoint saved at step {successful_steps} (Anchor Val Loss: {val_anchor_loss:.4f})")

                model.train()
        else:
            logger.warning(f"GradScaler skipped step due to non-finite gradients at step {successful_steps}.")

    total_wall_time = time.time() - t_start
    final_exposure_stats = pipeline.get_exposure_summary()

    logger.info("=" * 70)
    logger.info(f"Targeted SFT Completed: {successful_steps} steps in {total_wall_time:.1f}s")
    logger.info(f"Best Step: {best_step} | Best Anchor Val Loss: {best_anchor_val_loss:.4f}")
    logger.info(f"Final Exposure Summary: {final_exposure_stats['anchor_samples_drawn']} anchor samples ({final_exposure_stats['anchor_sample_fraction']*100:.1f}%), {final_exposure_stats['anchor_target_tokens_drawn']} anchor tokens ({final_exposure_stats['anchor_target_token_fraction']*100:.1f}%)")
    logger.info("=" * 70)

    # Save training history
    with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "successful_steps": successful_steps,
            "total_wall_seconds": total_wall_time,
            "best_step": best_step,
            "best_anchor_val_loss": best_anchor_val_loss,
            "history": history,
            "final_exposure_stats": final_exposure_stats,
        }, f, indent=2)

    with open(log_dir / "mixed_sampling_report.json", "w", encoding="utf-8") as f:
        json.dump(final_exposure_stats, f, indent=2)

    return {
        "successful_steps": successful_steps,
        "total_wall_seconds": total_wall_time,
        "best_step": best_step,
        "best_anchor_val_loss": best_anchor_val_loss,
    }


def main() -> None:
    base_ckpt = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
    dolly_cache = REPO_ROOT / "data" / "cache_sft_27m_v1"
    anchor_cache = REPO_ROOT / "data" / "cache_sft_anchors_v1"
    checkpoint_dir = REPO_ROOT / "checkpoints" / "27m_sft_anchors_v1"
    tokenizer_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    lm_val_path = REPO_ROOT / "data" / "cache_27m_v1" / "val_tokens.pt"
    log_dir = REPO_ROOT / "logs" / "anchor_sft_step40"

    run_targeted_sft(
        base_checkpoint_path=base_ckpt,
        dolly_cache_dir=dolly_cache,
        anchor_cache_dir=anchor_cache,
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        lm_val_path=lm_val_path,
        log_dir=log_dir,
        max_steps=250,
        val_interval=50,
        peak_lr=2.0e-5,
        min_lr=2.0e-6,
        warmup_steps=20,
        micro_batch_size=4,
        grad_accum_steps=4,
        anchor_sample_prob=0.55,
        max_wall_seconds=450.0,
    )


if __name__ == "__main__":
    main()

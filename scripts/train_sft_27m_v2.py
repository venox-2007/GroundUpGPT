"""Step 48: Controlled 27M Conversational SFT Fine-Tuning (Neural v2).

Initializes from Step 40 Anchor Best Checkpoint:
    checkpoints/27m_sft_anchors_v1/model_27m_anchors_best.pt (global step 6799)

Blended Tri-Dataset Sampling:
- 50% Conversational SFT v2 (722 validated short conversational pairs)
- 25% Step 40 Anchors (321 compact anchor pairs)
- 25% Dolly-15k SFT replay (8,433 general instruction pairs)

Evaluation & Tracking:
- Tracks Conversational v2 val loss, Anchor v1 val loss, Dolly val loss, and Pretraining LM loss.
- Evaluates every 50 optimizer steps.
- Saves best checkpoint atomically to checkpoints/27m_sft_neural_v2/model_27m_neural_v2_best.pt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import IGNORE_INDEX, collate_sft_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "27m_sft_anchors_v1" / "model_27m_anchors_best.pt"
OUTPUT_DIR = REPO_ROOT / "checkpoints" / "27m_sft_neural_v2"
LOG_DIR = REPO_ROOT / "logs" / "autonomous_steps_44_46"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def evaluate_dataset_loss(
    model: GroundUpGPT,
    examples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[float, float, int]:
    """Compute token-weighted assistant-only loss on an SFT dataset."""
    model.eval()
    total_loss_sum = 0.0
    total_active_tokens = 0
    use_cuda = (device.type == "cuda")

    with torch.inference_mode():
        for i in range(0, len(examples), batch_size):
            batch_items = examples[i : min(i + batch_size, len(examples))]
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
    num_windows: int = 32,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """Compute language modeling loss on fixed windows of pretraining validation data."""
    model.eval()
    block_size = model.config.block_size
    use_cuda = (device.type == "cuda")
    tot_loss = 0.0
    tot_tokens = 0

    stride = max(1, (len(tokens) - block_size - 1) // num_windows)
    indices = [i * stride for i in range(num_windows)]

    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack([tokens[j : j + block_size].long() for j in batch_idx]).to(device)
            y = torch.stack([tokens[j + 1 : j + block_size + 1].long() for j in batch_idx]).to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits, _ = model(x)
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="sum",
                )
            tot_loss += loss_sum.item()
            tot_tokens += x.numel()

    loss = tot_loss / max(1, tot_tokens)
    ppl = math.exp(min(20.0, loss))
    return loss, ppl


class TriBlendedSampler:
    """Samples batches across Conversational v2 (50%), Anchors v1 (25%), and Dolly SFT (25%)."""

    def __init__(
        self,
        v2_train: List[Dict[str, Any]],
        anchor_train: List[Dict[str, Any]],
        dolly_train: List[Dict[str, Any]],
        seed: int = 42,
    ):
        self.v2 = v2_train
        self.anchor = anchor_train
        self.dolly = dolly_train
        self.rng = random.Random(seed)

    def sample_batch(self, batch_size: int, block_size: int) -> Dict[str, torch.Tensor]:
        batch_items = []
        for _ in range(batch_size):
            r = self.rng.random()
            if r < 0.50:
                item = self.rng.choice(self.v2)
            elif r < 0.75:
                item = self.rng.choice(self.anchor)
            else:
                item = self.rng.choice(self.dolly)
            batch_items.append(item)

        return collate_sft_batch(batch_items, pad_token_id=0, pad_to_max_len=False, max_seq_len=block_size)


def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Load Base Checkpoint
    logger.info(f"Loading base checkpoint: {BASE_CHECKPOINT_PATH}...")
    ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    start_step = ckpt.get("step", 250)
    source_global_step = ckpt.get("global_step", 6799)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded base model: step={start_step}, global_step={source_global_step}, params={n_params:,}")

    # 2. Load Datasets
    logger.info("Loading dataset caches...")
    v2_train = torch.load(REPO_ROOT / "data" / "cache_sft_neural_v2" / "train_tokens.pt", weights_only=False)
    v2_val = torch.load(REPO_ROOT / "data" / "cache_sft_neural_v2" / "val_tokens.pt", weights_only=False)
    anchor_train = torch.load(REPO_ROOT / "data" / "cache_sft_anchors_v1" / "train_anchor_tokens.pt", weights_only=False)
    anchor_val = torch.load(REPO_ROOT / "data" / "cache_sft_anchors_v1" / "val_anchor_tokens.pt", weights_only=False)
    dolly_train = torch.load(REPO_ROOT / "data" / "cache_sft_27m_v1" / "train_sft_tokens.pt", weights_only=False)
    dolly_val = torch.load(REPO_ROOT / "data" / "cache_sft_27m_v1" / "val_sft_tokens.pt", weights_only=False)[:100]
    lm_val = torch.load(REPO_ROOT / "data" / "cache_27m_v1" / "val_tokens.pt", weights_only=False)

    logger.info(f"Data caches loaded: v2_train={len(v2_train)}, anchor_train={len(anchor_train)}, dolly_train={len(dolly_train)}")
    logger.info(f"Val caches loaded: v2_val={len(v2_val)}, anchor_val={len(anchor_val)}, dolly_val={len(dolly_val)}, lm_val={len(lm_val):,} tokens")

    sampler = TriBlendedSampler(v2_train, anchor_train, dolly_train, seed=42)

    # 3. Baseline validation before training
    logger.info("Running baseline validation evaluations...")
    base_v2_loss, base_v2_ppl, _ = evaluate_dataset_loss(model, v2_val, device)
    base_anc_loss, base_anc_ppl, _ = evaluate_dataset_loss(model, anchor_val, device)
    base_dol_loss, base_dol_ppl, _ = evaluate_dataset_loss(model, dolly_val, device)
    base_lm_loss, base_lm_ppl = evaluate_pretraining_lm(model, lm_val, device)

    base_composite = 0.5 * base_v2_loss + 0.3 * base_anc_loss + 0.2 * base_dol_loss

    logger.info(f"Baseline - v2 Val Loss: {base_v2_loss:.4f} (PPL {base_v2_ppl:.2f})")
    logger.info(f"Baseline - Anchor Val Loss: {base_anc_loss:.4f} (PPL {base_anc_ppl:.2f})")
    logger.info(f"Baseline - Dolly Val Loss: {base_dol_loss:.4f} (PPL {base_dol_ppl:.2f})")
    logger.info(f"Baseline - Pretrain LM Loss: {base_lm_loss:.4f} (PPL {base_lm_ppl:.2f})")
    logger.info(f"Baseline - Composite Val Loss: {base_composite:.4f}")

    # 4. Hyperparameters
    max_steps = 350
    warmup_steps = 25
    peak_lr = 1.5e-5
    min_lr = 1.5e-6
    weight_decay = 0.01
    micro_batch_size = 4
    grad_accum_steps = 4  # Effective batch size = 16
    eval_interval = 50
    max_wall_seconds = 720.0  # 12 minutes safety limit

    # Optimizer setup
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def get_lr(step_num: int) -> float:
        if step_num < warmup_steps:
            return peak_lr * (step_num + 1) / (warmup_steps + 1)
        if step_num > max_steps:
            return min_lr
        decay_ratio = (step_num - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (peak_lr - min_lr)

    # 5. Training Loop
    logger.info(f"Starting training for {max_steps} steps (eff_batch_size={micro_batch_size * grad_accum_steps})...")
    model.train()
    t_start = time.time()
    best_composite_loss = base_composite
    best_step = 0
    history = []

    for step in range(1, max_steps + 1):
        elapsed = time.time() - t_start
        if elapsed >= max_wall_seconds:
            logger.warning(f"Wall-clock safety limit reached ({elapsed:.1f}s). Terminating early.")
            break

        lr = get_lr(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        step_active_tokens = 0

        # Sample micro-batches
        micro_batches = []
        for _ in range(grad_accum_steps):
            batch = sampler.sample_batch(micro_batch_size, model.config.block_size)
            bx = batch["input_ids"].to(device)
            by = batch["target_ids"].to(device)
            bmask = batch["loss_mask"].to(device)
            micro_batches.append((bx, by, bmask))
            step_active_tokens += int(bmask.sum().item())

        denom = max(1, step_active_tokens)

        for bx, by, bmask in micro_batches:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = model(bx)
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    by.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                micro_loss = loss_sum / denom

            step_loss_sum += loss_sum.item()
            scaler.scale(micro_loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss = step_loss_sum / denom

        if step % 25 == 0 or step == 1:
            tok_s = step_active_tokens / max(time.time() - t_start, 1e-4) * step
            logger.info(f"Step {step:3d}/{max_steps} | lr: {lr:.2e} | train_loss: {train_loss:.4f} | grad_norm: {grad_norm:.3f} | elapsed: {time.time()-t_start:.1f}s")

        # Validation evaluation
        if step % eval_interval == 0 or step == max_steps:
            v2_loss, v2_ppl, _ = evaluate_dataset_loss(model, v2_val, device)
            anc_loss, anc_ppl, _ = evaluate_dataset_loss(model, anchor_val, device)
            dol_loss, dol_ppl, _ = evaluate_dataset_loss(model, dolly_val, device)
            lm_loss, lm_ppl = evaluate_pretraining_lm(model, lm_val, device)

            comp_loss = 0.5 * v2_loss + 0.3 * anc_loss + 0.2 * dol_loss
            is_best = comp_loss < best_composite_loss

            logger.info(f"--- Eval @ Step {step} ---")
            logger.info(f"  v2 Val: {v2_loss:.4f} (PPL {v2_ppl:.2f}) | Anchor Val: {anc_loss:.4f} (PPL {anc_ppl:.2f})")
            logger.info(f"  Dolly Val: {dol_loss:.4f} (PPL {dol_ppl:.2f}) | Pretrain LM: {lm_loss:.4f} (PPL {lm_ppl:.2f})")
            logger.info(f"  Composite Val Loss: {comp_loss:.4f} {'[NEW BEST]' if is_best else ''}")

            record = {
                "step": step,
                "global_step": source_global_step + step,
                "lr": lr,
                "train_loss": train_loss,
                "v2_val_loss": round(v2_loss, 4),
                "v2_val_ppl": round(v2_ppl, 2),
                "anchor_val_loss": round(anc_loss, 4),
                "anchor_val_ppl": round(anc_ppl, 2),
                "dolly_val_loss": round(dol_loss, 4),
                "dolly_val_ppl": round(dol_ppl, 2),
                "lm_val_loss": round(lm_loss, 4),
                "lm_val_ppl": round(lm_ppl, 2),
                "composite_val_loss": round(comp_loss, 4),
                "is_best": is_best,
                "elapsed_sec": round(time.time() - t_start, 2),
            }
            history.append(record)

            # Save latest checkpoint
            latest_ckpt = {
                "step": step,
                "global_step": source_global_step + step,
                "model_config": ckpt["model_config"],
                "model_state_dict": model.state_dict(),
                "val_metrics": record,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(latest_ckpt, OUTPUT_DIR / "model_27m_neural_v2_latest.pt")

            if is_best:
                best_composite_loss = comp_loss
                best_step = step
                torch.save(latest_ckpt, OUTPUT_DIR / "model_27m_neural_v2_best.pt")
                logger.info(f"Saved new best checkpoint: {OUTPUT_DIR / 'model_27m_neural_v2_best.pt'}")

            model.train()

    total_wall_time = time.time() - t_start
    logger.info(f"Training completed in {total_wall_time:.1f}s. Best step: {best_step} (composite val loss: {best_composite_loss:.4f})")

    # If best checkpoint wasn't updated (e.g. baseline was lower), save final model as best candidate
    best_path = OUTPUT_DIR / "model_27m_neural_v2_best.pt"
    if not best_path.exists():
        torch.save(latest_ckpt, best_path)
        logger.info(f"Saved final checkpoint as best: {best_path}")

    # Compute SHA-256 of best checkpoint
    best_sha256 = hashlib.sha256(open(best_path, "rb").read()).hexdigest()

    summary = {
        "model_name": "GroundUpGPT-27M Neural v2 SFT",
        "base_checkpoint": str(BASE_CHECKPOINT_PATH),
        "output_best_checkpoint": str(best_path),
        "best_checkpoint_sha256": best_sha256,
        "total_steps": step,
        "best_step": best_step,
        "baseline_composite_loss": round(base_composite, 4),
        "best_composite_loss": round(best_composite_loss, 4),
        "total_training_time_sec": round(total_wall_time, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else 0.0,
        "history": history,
    }

    with open(LOG_DIR / "step48_training_log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 48 NEURAL V2 FINE-TUNING FINISHED")
    print(f"Base Checkpoint:       {BASE_CHECKPOINT_PATH.name}")
    print(f"Best Checkpoint:       model_27m_neural_v2_best.pt (Step {best_step})")
    print(f"SHA-256:               {best_sha256}")
    print(f"Baseline Composite:    {base_composite:.4f}")
    print(f"Best Composite Val:    {best_composite_loss:.4f}")
    print(f"Wall-Clock Time:       {total_wall_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    train()

"""Controlled Continued Pretraining for GroundUpGPT-27M (Step 49).

Priorities & Safeguards:
1. Strict Lineage Initialization:
   Initializes strictly from checkpoints/27m_phase4/model_27m_best.pt (Step 5949).
   Never initialized from SFT checkpoints.
2. Controlled Caps:
   Maximum 1,500 optimizer steps OR 20 minutes (1,200s) GPU wall-clock time, whichever hits first.
3. Balanced Data Mixing:
   Interleaves the original pretraining corpus (data/cache_27m_v1/) with the new expanded
   factual corpus (data/cache_pretrain_v2/) at a controlled 50/50 ratio to prevent catastrophic
   forgetting of the original foundational knowledge while rapidly absorbing new factual entities.
4. Deterministic Multi-Track Validation:
   Evaluates periodically every 150 steps on:
   - Track A1: Original pretraining held-out validation (cache_27m_v1)
   - Track A2: New expanded held-out validation (cache_pretrain_v2)
   - Composite validation loss
5. Complete State Checkpoints:
   Saves full optimizer state, scaler state, RNG states, source lineage hash, step counts,
   and token exposure accounting to checkpoints/27m_pretrain_v2/.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.amp import GradScaler, autocast

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_pretrain_v2")

EXPECTED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_PARAM_COUNT = 27411456

INSPECTION_PROMPTS = [
    "The sun rose over the",
    "Once upon a time in a small village,",
    "The capital of France is",
    "The Earth revolves around the",
    "Water is composed of",
    "An atom is the smallest unit of",
    "Biology is the study of",
]


class PretrainMixedSampler:
    """Samples batches from original and secondary pretraining token tensors with tracking."""

    def __init__(
        self,
        v1_tokens: torch.Tensor,
        v2_tokens: torch.Tensor,
        mix_ratio_v2: float = 0.5,
        block_size: int = 256,
        seed: int = 42,
    ):
        self.v1 = v1_tokens
        self.v2 = v2_tokens
        self.mix_ratio_v2 = mix_ratio_v2
        self.block_size = block_size
        self.rng = random.Random(seed)

        self.max_idx_v1 = len(self.v1) - block_size - 1
        self.max_idx_v2 = len(self.v2) - block_size - 1

        self.tokens_exposed_v1 = 0
        self.tokens_exposed_v2 = 0

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list = []
        y_list = []

        for _ in range(batch_size):
            if self.rng.random() < self.mix_ratio_v2:
                # Sample from V2 (new expanded corpus)
                idx = self.rng.randint(0, self.max_idx_v2)
                x = self.v2[idx : idx + self.block_size]
                y = self.v2[idx + 1 : idx + self.block_size + 1]
                self.tokens_exposed_v2 += self.block_size
            else:
                # Sample from V1 (original pretraining corpus)
                idx = self.rng.randint(0, self.max_idx_v1)
                x = self.v1[idx : idx + self.block_size]
                y = self.v1[idx + 1 : idx + self.block_size + 1]
                self.tokens_exposed_v1 += self.block_size

            x_list.append(x)
            y_list.append(y)

        bx = torch.stack(x_list).long()
        by = torch.stack(y_list).long()
        return bx, by


def evaluate_val_windows(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float]:
    """Evaluate loss and perplexity on deterministic token windows."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack([tokens[j : j + block_size] for j in batch_idx]).to(device, dtype=torch.long)
            y = torch.stack([tokens[j + 1 : j + block_size + 1] for j in batch_idx]).to(device, dtype=torch.long)
            with autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                _, loss = model(x, y)
            total_loss += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

    mean_loss = total_loss / max(1, total_samples)
    ppl = math.exp(min(20.0, mean_loss))
    return round(mean_loss, 4), round(ppl, 2)


def generate_qualitative_samples(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    prompts: List[str],
    device: torch.device,
) -> List[Dict[str, str]]:
    generator = TextGenerator(model, tokenizer=tokenizer, device=device)
    cfg = GenerationConfig(
        max_new_tokens=32,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.1,
        seed=42,
    )
    samples = []
    for p in prompts:
        out = generator.generate(p, cfg)
        text = out.generated_text if hasattr(out, "generated_text") else str(out)
        samples.append({"prompt": p, "completion": text})
    return samples


def train_continued_pretraining(
    base_checkpoint_path: Path,
    tokenizer_path: Path,
    cache_v1_dir: Path,
    cache_v2_dir: Path,
    output_dir: Path,
    log_path: Path,
    max_steps: int = 1500,
    max_wall_seconds: float = 1200.0,  # 20 minutes strict safety limit
    micro_batch_size: int = 8,
    grad_accum_steps: int = 4,         # Effective batch = 32 seqs = 8,192 tokens/step
    block_size: int = 256,
    peak_lr: float = 1.5e-4,
    min_lr: float = 1.5e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 150,
    mix_ratio_v2: float = 0.50,
    seed: int = 42,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Load Base Checkpoint & Verify Lineage
    logger.info(f"Loading strictly verified base checkpoint: {base_checkpoint_path}...")
    base_sha256 = hashlib.sha256(open(base_checkpoint_path, "rb").read()).hexdigest()
    ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    source_step = ckpt.get("step", 5949)
    source_global_step = ckpt.get("global_step", source_step)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAM_COUNT:
        raise ValueError(f"Parameter count mismatch! Expected {EXPECTED_PARAM_COUNT}, got {n_params}")

    logger.info(
        f"Verified base checkpoint: {base_checkpoint_path.name} | Step: {source_step} | "
        f"Params: {n_params:,} | SHA-256: {base_sha256[:16]}..."
    )

    # 2. Load Tokenizer
    tokenizer = BPETokenizer().load(str(tokenizer_path))

    # 3. Load Datasets
    logger.info("Loading pretraining token caches...")
    v1_train = torch.load(cache_v1_dir / "train_tokens.pt", weights_only=False)
    v1_val = torch.load(cache_v1_dir / "val_tokens.pt", weights_only=False)
    v2_train = torch.load(cache_v2_dir / "train_tokens.pt", weights_only=False)
    v2_val = torch.load(cache_v2_dir / "val_tokens.pt", weights_only=False)

    logger.info(f"V1 (Orig Corpus) Train: {len(v1_train):,} tokens | Val: {len(v1_val):,} tokens")
    logger.info(f"V2 (New Corpus)  Train: {len(v2_train):,} tokens | Val: {len(v2_val):,} tokens")

    # Fixed deterministic validation indices (250 windows each)
    def make_eval_indices(t_len: int, n: int, s: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(s)
        return torch.randint(0, max(0, t_len - block_size - 1), (n,), generator=g)

    val_idx_v1 = make_eval_indices(len(v1_val), 250, 101)
    val_idx_v2 = make_eval_indices(len(v2_val), 250, 102)

    # 4. Sampler Setup
    sampler = PretrainMixedSampler(
        v1_tokens=v1_train,
        v2_tokens=v2_train,
        mix_ratio_v2=mix_ratio_v2,
        block_size=block_size,
        seed=seed,
    )

    # 5. Optimizer Setup
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=peak_lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    def get_lr(step_num: int) -> float:
        if step_num < warmup_steps:
            return peak_lr * (step_num + 1) / (warmup_steps + 1)
        if step_num > max_steps:
            return min_lr
        decay_ratio = (step_num - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (peak_lr - min_lr)

    # 6. Baseline Pre-Evaluation (Step 0)
    logger.info("Evaluating baseline metrics before Step 49 training begins...")
    base_loss_v1, base_ppl_v1 = evaluate_val_windows(model, v1_val, val_idx_v1, block_size, device=device)
    base_loss_v2, base_ppl_v2 = evaluate_val_windows(model, v2_val, val_idx_v2, block_size, device=device)
    base_composite = round((base_loss_v1 + base_loss_v2) / 2.0, 4)

    logger.info(
        f"Baseline -> Track A1 (Orig Val): {base_loss_v1:.4f} (PPL {base_ppl_v1:.2f}) | "
        f"Track A2 (New Val): {base_loss_v2:.4f} (PPL {base_ppl_v2:.2f}) | "
        f"Composite: {base_composite:.4f}"
    )

    # 7. Training Loop
    effective_batch_tokens = micro_batch_size * grad_accum_steps * block_size
    logger.info(
        f"Starting continued pretraining for {max_steps} steps "
        f"(effective batch: {micro_batch_size * grad_accum_steps} seqs = {effective_batch_tokens:,} tokens/step, "
        f"max time: {max_wall_seconds:.0f}s)..."
    )

    model.train()
    best_composite_loss = base_composite
    best_step = 0
    history: List[Dict[str, Any]] = []
    t_train_start = time.perf_counter()

    for step in range(1, max_steps + 1):
        elapsed = time.perf_counter() - t_train_start
        if elapsed >= max_wall_seconds:
            logger.warning(f"Wall-clock safety limit reached ({elapsed:.1f}s >= {max_wall_seconds:.0f}s). Halting cleanly.")
            break

        lr = get_lr(step - 1)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_losses = []

        for _ in range(grad_accum_steps):
            bx, by = sampler.sample_batch(micro_batch_size)
            bx = bx.to(device)
            by = by.to(device)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                _, loss = model(bx, by)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            micro_losses.append(loss.item())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        train_loss = round(sum(micro_losses) / len(micro_losses), 4)

        if step % 25 == 0 or step == 1:
            step_tok_per_s = (step * effective_batch_tokens) / max(1e-4, elapsed)
            logger.info(
                f"Step {step:4d}/{max_steps} | Loss: {train_loss:.4f} | LR: {lr:.2e} | "
                f"GradNorm: {grad_norm:.2f} | Elapsed: {elapsed:.1f}s | Speed: {step_tok_per_s:.0f} tok/s"
            )

        # Periodic Validation
        if step % eval_interval == 0 or step == max_steps:
            loss_v1, ppl_v1 = evaluate_val_windows(model, v1_val, val_idx_v1, block_size, device=device)
            loss_v2, ppl_v2 = evaluate_val_windows(model, v2_val, val_idx_v2, block_size, device=device)
            comp_loss = round((loss_v1 + loss_v2) / 2.0, 4)
            is_best = comp_loss < best_composite_loss

            logger.info(
                f">>> Eval Step {step:4d} | Orig Val: {loss_v1:.4f} (PPL {ppl_v1:.2f}) | "
                f"New Val: {loss_v2:.4f} (PPL {ppl_v2:.2f}) | Composite: {comp_loss:.4f} "
                f"{'[BEST]' if is_best else ''}"
            )

            record = {
                "step": step,
                "global_step": source_global_step + step,
                "lr": round(lr, 8),
                "train_loss": train_loss,
                "orig_val_loss": loss_v1,
                "orig_val_ppl": ppl_v1,
                "new_val_loss": loss_v2,
                "new_val_ppl": ppl_v2,
                "composite_val_loss": comp_loss,
                "is_best": is_best,
                "tokens_exposed_v1": sampler.tokens_exposed_v1,
                "tokens_exposed_v2": sampler.tokens_exposed_v2,
                "total_tokens_exposed": sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2,
                "elapsed_sec": round(time.perf_counter() - t_train_start, 2),
            }
            history.append(record)

            # Save latest checkpoint
            latest_ckpt = {
                "step": step,
                "source_step": source_step,
                "global_step": source_global_step + step,
                "model_config": ckpt["model_config"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "parent_checkpoint_path": str(base_checkpoint_path),
                "parent_checkpoint_sha256": base_sha256,
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "val_metrics": record,
                "tokens_exposed_v1": sampler.tokens_exposed_v1,
                "tokens_exposed_v2": sampler.tokens_exposed_v2,
                "total_tokens_exposed": sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(latest_ckpt, output_dir / "model_27m_pretrain_v2_latest.pt")

            if is_best:
                best_composite_loss = comp_loss
                best_step = step
                torch.save(latest_ckpt, output_dir / "model_27m_pretrain_v2_best.pt")
                logger.info(f"Saved new best checkpoint: {output_dir / 'model_27m_pretrain_v2_best.pt'}")

            model.train()

    total_wall_s = time.perf_counter() - t_train_start
    logger.info(f"Training completed in {total_wall_s:.1f}s. Best step: {best_step} (composite: {best_composite_loss:.4f})")

    # If best was not updated (e.g. baseline was lower), save latest as best candidate
    best_path = output_dir / "model_27m_pretrain_v2_best.pt"
    if not best_path.exists():
        torch.save(latest_ckpt, best_path)
        logger.info(f"Saved final checkpoint as best candidate: {best_path}")

    best_sha256 = hashlib.sha256(open(best_path, "rb").read()).hexdigest()

    # Generate qualitative samples from best model
    logger.info("Generating qualitative inspection continuations from best model...")
    best_ckpt_obj = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt_obj["model_state_dict"])
    model.eval()
    qual_samples = generate_qualitative_samples(model, tokenizer, INSPECTION_PROMPTS, device=device)

    # Full training report
    summary = {
        "experiment_name": "pretraining_v2_continued_27m",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_checkpoint": {
            "path": str(base_checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": base_sha256,
            "step": source_step,
        },
        "output_checkpoints": {
            "best_checkpoint": str(best_path.relative_to(REPO_ROOT)),
            "best_checkpoint_sha256": best_sha256,
            "best_step": best_step,
            "best_composite_loss": best_composite_loss,
            "latest_checkpoint": str((output_dir / "model_27m_pretrain_v2_latest.pt").relative_to(REPO_ROOT)),
        },
        "training_hyperparameters": {
            "max_steps": max_steps,
            "steps_completed": step,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": micro_batch_size * grad_accum_steps,
            "sequence_block_size": block_size,
            "effective_batch_tokens": effective_batch_tokens,
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "mix_ratio_new_corpus": mix_ratio_v2,
            "wall_clock_seconds": round(total_wall_s, 2),
        },
        "token_accounting": {
            "tokens_exposed_v1_orig": sampler.tokens_exposed_v1,
            "tokens_exposed_v2_new": sampler.tokens_exposed_v2,
            "total_tokens_exposed": sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2,
            "pct_from_new_corpus": round((sampler.tokens_exposed_v2 / max(1, sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2)) * 100, 2),
        },
        "baseline_metrics": {
            "orig_val_loss": base_loss_v1,
            "orig_val_ppl": base_ppl_v1,
            "new_val_loss": base_loss_v2,
            "new_val_ppl": base_ppl_v2,
            "composite_loss": base_composite,
        },
        "final_metrics": history[-1] if history else {},
        "training_history": history,
        "qualitative_samples": qual_samples,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training log written to {log_path}")

    print("\n" + "=" * 70)
    print(" GROUNDUPGPT-27M CONTINUED PRETRAINING COMPLETE")
    print("=" * 70)
    print(f"Base Checkpoint:         {base_checkpoint_path.name} (Step {source_step})")
    print(f"Total Steps Trained:     {step} (Global Step: {source_global_step + step})")
    print(f"Best Step:               {best_step} (Composite Val Loss: {best_composite_loss:.4f})")
    print(f"Best Checkpoint:         {best_path.name} (SHA-256: {best_sha256[:16]}...)")
    print(f"Wall-Clock Time:         {total_wall_s:.1f}s ({total_wall_s / 60:.1f} min)")
    print("-" * 70)
    print("TOKEN EXPOSURE ACCOUNTING:")
    print(f"  Orig Corpus (V1) Tok:  {sampler.tokens_exposed_v1:,} ({sampler.tokens_exposed_v1/(sampler.tokens_exposed_v1+sampler.tokens_exposed_v2)*100:.1f}%)")
    print(f"  New Corpus (V2) Tok:   {sampler.tokens_exposed_v2:,} ({sampler.tokens_exposed_v2/(sampler.tokens_exposed_v1+sampler.tokens_exposed_v2)*100:.1f}%)")
    print(f"  Total Tokens Exposed:  {sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2:,}")
    print("-" * 70)
    print("VALIDATION PROGRESSION:")
    print(f"  Baseline:  Orig Val: {base_loss_v1:.4f} (PPL {base_ppl_v1:.2f}) | New Val: {base_loss_v2:.4f} (PPL {base_ppl_v2:.2f}) | Comp: {base_composite:.4f}")
    if history:
        last = history[-1]
        print(f"  Final:     Orig Val: {last['orig_val_loss']:.4f} (PPL {last['orig_val_ppl']:.2f}) | New Val: {last['new_val_loss']:.4f} (PPL {last['new_val_ppl']:.2f}) | Comp: {last['composite_val_loss']:.4f}")
    print("=" * 70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train GroundUpGPT-27M Pretraining v2 (Step 49)")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "27m_phase4" / "model_27m_best.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json",
    )
    parser.add_argument(
        "--cache-v1",
        type=Path,
        default=REPO_ROOT / "data" / "cache_27m_v1",
    )
    parser.add_argument(
        "--cache-v2",
        type=Path,
        default=REPO_ROOT / "data" / "cache_pretrain_v2",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "27m_pretrain_v2",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=REPO_ROOT / "logs" / "pretraining_step49" / "training_log.json",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--max-wall-seconds", type=float, default=1200.0)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--peak-lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-5)
    args = parser.parse_args()

    train_continued_pretraining(
        base_checkpoint_path=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        cache_v1_dir=args.cache_v1,
        cache_v2_dir=args.cache_v2,
        output_dir=args.checkpoint_dir,
        log_path=args.log_path,
        max_steps=args.max_steps,
        max_wall_seconds=args.max_wall_seconds,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum,
        eval_interval=args.eval_interval,
        peak_lr=args.peak_lr,
        min_lr=args.min_lr,
    )


if __name__ == "__main__":
    main()

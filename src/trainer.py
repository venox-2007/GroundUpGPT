"""GPU-compatible, reproducible training pipeline for GroundUpGPT.

Features AdamW optimization, decoupled weight decay, cosine decay learning-rate scheduling
with linear warmup, mixed-precision training (torch.amp), gradient accumulation,
gradient norm clipping, and stateful checkpoint save/resumption with compatibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Dict, Any, Union
import torch
from torch.amp import GradScaler, autocast

from src.model import GroundUpGPT, GPTConfig
from src.data import TextDataPipeline


def get_tokenizer_fingerprint(tokenizer: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint of tokenizer identity and rules.

    Ensures that checkpoints cannot be restored with incompatible tokenizers even if
    their class name or total vocabulary size happens to match.
    """
    if hasattr(tokenizer, "get_fingerprint") and callable(getattr(tokenizer, "get_fingerprint")):
        return str(tokenizer.get_fingerprint())

    tok_type = type(tokenizer).__name__
    hasher = hashlib.sha256()
    hasher.update(tok_type.encode("utf-8"))
    hasher.update(b":")

    if hasattr(tokenizer, "stoi"):
        items = sorted(tokenizer.stoi.items(), key=lambda x: x[1])
        hasher.update(json.dumps(items, ensure_ascii=False).encode("utf-8"))
    elif hasattr(tokenizer, "merges"):
        items = sorted([(p[0], p[1], idx) for p, idx in tokenizer.merges.items()], key=lambda x: x[2])
        hasher.update(json.dumps(items).encode("utf-8"))
    else:
        v_size = getattr(tokenizer, "vocab_size", 0)
        hasher.update(f"unknown:{v_size}".encode("utf-8"))

    return hasher.hexdigest()


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for GroundUpGPT training.

    Attributes:
        max_steps: Total number of optimization steps to execute.
        batch_size: Number of sequences per micro-batch.
        block_size: Context length per sequence.
        learning_rate: Peak learning rate after warmup.
        min_lr: Minimum learning rate after cosine decay.
        warmup_steps: Number of initial steps for linear warmup.
        weight_decay: L2 regularization penalty applied to weight tensors.
        beta1: AdamW beta1 hyperparameter (default: 0.9).
        beta2: AdamW beta2 hyperparameter (default: 0.95).
        grad_clip: Maximum gradient norm for clipping (0.0 to disable).
        grad_accum_steps: Number of micro-batches to accumulate before stepping.
        eval_interval: Frequency of train/val evaluation in optimizer steps.
        eval_iters: Number of micro-batches to average during evaluation.
        checkpoint_interval: Frequency of checkpoint saving in optimizer steps.
        checkpoint_dir: Directory path where checkpoints are saved.
        device: 'auto' (detect CUDA vs CPU), 'cuda', or 'cpu'.
        mixed_precision: Enable torch.amp float16 mixed precision on CUDA.
        seed: Random seed for reproducible initialization and data batching.
    """

    max_steps: int = 100
    batch_size: int = 4
    block_size: int = 16
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    eval_interval: int = 25
    eval_iters: int = 5
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate training configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr cannot be negative, got {self.min_lr}")
        if self.min_lr > self.learning_rate:
            raise ValueError(f"min_lr ({self.min_lr}) cannot exceed learning_rate ({self.learning_rate})")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps cannot be negative, got {self.warmup_steps}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {self.grad_accum_steps}")
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        if self.eval_iters <= 0:
            raise ValueError(f"eval_iters must be positive, got {self.eval_iters}")


class Trainer:
    """Manages training execution, evaluation, optimization, and checkpointing for GroundUpGPT.

    Args:
        model: GroundUpGPT model instance.
        pipeline: TextDataPipeline containing train and validation token splits.
        config: TrainingConfig specifying hyperparameters.
    """

    def __init__(
        self,
        model: GroundUpGPT,
        pipeline: TextDataPipeline,
        config: TrainingConfig,
    ) -> None:
        self.model: GroundUpGPT = model
        self.pipeline: TextDataPipeline = pipeline
        self.config: TrainingConfig = config

        # 1. Device Selection
        if config.device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model.to(self.device)

        # 2. Mixed-Precision Autocast Setup
        self.use_amp: bool = (self.device.startswith("cuda") and config.mixed_precision)
        self.scaler: GradScaler = GradScaler("cuda", enabled=self.use_amp)

        # 3. Decoupled AdamW Parameter Groups (Weight decay on weights >= 2D, none on 1D/biases)
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

        # Deduplicate to prevent double-counting shared weight-tied parameters
        decay_params = list(dict.fromkeys(decay_params))
        nodecay_params = list(dict.fromkeys(nodecay_params))

        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        # 4. State Tracking & Reproducible Generators
        self.current_step: int = 0
        self.checkpoint_dir: Path = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Synchronize global seeds and dedicated data pipeline generator
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.rng_generator: torch.Generator = torch.Generator().manual_seed(config.seed)

    def get_lr_for_step(self, step: int) -> float:
        """Compute learning rate using linear warmup followed by cosine decay.

        Args:
            step: Current zero-indexed optimization step.

        Returns:
            float: Learning rate for the given step.
        """
        # 1. Linear Warmup Phase
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)

        # 2. Beyond Maximum Steps -> Clamp to min_lr
        if step >= self.config.max_steps:
            return self.config.min_lr

        # 3. Cosine Decay Phase
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + cosine_decay * (self.config.learning_rate - self.config.min_lr)

    def _set_optimizer_lr(self, lr: float) -> None:
        """Update learning rate across all optimizer parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def estimate_loss(self, eval_iters: Optional[int] = None) -> Dict[str, float]:
        """Estimate mean loss across training and validation splits.

        Args:
            eval_iters: Number of micro-batches to evaluate per split (defaults to config.eval_iters).

        Returns:
            Dict[str, float]: Average loss per split: {'train': float, 'val': float}.
        """
        iters = eval_iters if eval_iters is not None else self.config.eval_iters
        out: Dict[str, float] = {}
        self.model.eval()

        for split in ("train", "val"):
            losses = torch.zeros(iters)
            for k in range(iters):
                x, y = self.pipeline.get_batch(
                    split=split,
                    batch_size=self.config.batch_size,
                    block_size=self.config.block_size,
                    device=self.device,
                    generator=self.rng_generator,
                )
                with autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.use_amp):
                    _, loss = self.model(x, targets=y)
                assert loss is not None
                losses[k] = loss.item()
            out[split] = round(losses.mean().item(), 4)

        self.model.train()
        return out

    def save_checkpoint(
        self,
        file_path: Union[str, Path],
        val_loss: Optional[float] = None,
    ) -> None:
        """Save training checkpoint including model, optimizer, scheduler, and RNG state.

        Args:
            file_path: File path where checkpoint will be saved.
            val_loss: Optional validation loss metric to record.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer_cls_name = type(self.pipeline.tokenizer).__name__
        checkpoint: Dict[str, Any] = {
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "tokenizer_type": tokenizer_cls_name,
            "tokenizer_vocab_size": self.pipeline.vocab_size,
            "tokenizer_fingerprint": get_tokenizer_fingerprint(self.pipeline.tokenizer),
            "val_loss": val_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "pipeline_rng_state": self.rng_generator.get_state(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, file_path: Union[str, Path]) -> int:
        """Restore model, optimizer, scaler, and step counter from a checkpoint file.

        Validates architectural and tokenizer compatibility before loading.

        Args:
            file_path: Checkpoint file path to restore.

        Returns:
            int: The restored training step.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If architecture or tokenizer configurations mismatch.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 1. Validate Complete Architecture Compatibility
        saved_m_cfg = checkpoint.get("model_config", {})
        curr_m_cfg = asdict(self.model.config)
        critical_keys = (
            "vocab_size",
            "block_size",
            "embed_dim",
            "num_heads",
            "num_layers",
            "bias",
            "weight_tying",
        )
        for key in critical_keys:
            if key in saved_m_cfg and saved_m_cfg.get(key) != curr_m_cfg.get(key):
                raise ValueError(
                    f"Model architecture mismatch for '{key}': checkpoint has {saved_m_cfg.get(key)}, "
                    f"current model has {curr_m_cfg.get(key)}."
                )

        # 2. Validate Tokenizer Compatibility
        saved_tok_type = checkpoint.get("tokenizer_type")
        curr_tok_type = type(self.pipeline.tokenizer).__name__
        if saved_tok_type != curr_tok_type:
            raise ValueError(
                f"Tokenizer type mismatch: checkpoint trained with {saved_tok_type}, "
                f"current pipeline uses {curr_tok_type}."
            )

        saved_v_size = checkpoint.get("tokenizer_vocab_size")
        curr_v_size = self.pipeline.vocab_size
        if saved_v_size != curr_v_size:
            raise ValueError(
                f"Tokenizer vocabulary size mismatch: checkpoint has {saved_v_size}, "
                f"pipeline has {curr_v_size}."
            )

        saved_fingerprint = checkpoint.get("tokenizer_fingerprint")
        if saved_fingerprint is not None:
            curr_fingerprint = get_tokenizer_fingerprint(self.pipeline.tokenizer)
            if saved_fingerprint != curr_fingerprint:
                raise ValueError(
                    f"Tokenizer identity mismatch: checkpoint requires fingerprint {saved_fingerprint[:16]}..., "
                    f"current pipeline tokenizer has fingerprint {curr_fingerprint[:16]}..."
                )

        # 3. Restore Model, Optimizer, and Scaler States
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 4. Restore Global and Pipeline RNG States
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if self.device.startswith("cuda") and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
        if "pipeline_rng_state" in checkpoint:
            self.rng_generator.set_state(checkpoint["pipeline_rng_state"].cpu())

        self.current_step = int(checkpoint["step"])
        return self.current_step

    def train_step(self) -> float:
        """Execute a single optimization step incorporating gradient accumulation.

        Returns:
            float: Average training loss across accumulated micro-batches.
        """
        self.model.train()
        accum_loss = 0.0

        # Adjust learning rate based on schedule
        lr = self.get_lr_for_step(self.current_step)
        self._set_optimizer_lr(lr)

        self.optimizer.zero_grad(set_to_none=True)

        for _ in range(self.config.grad_accum_steps):
            x, y = self.pipeline.get_batch(
                split="train",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                device=self.device,
                generator=self.rng_generator,
            )

            with autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.use_amp):
                _, loss = self.model(x, targets=y)
                assert loss is not None
                # Scale loss by grad_accum_steps for correct averaging
                scaled_loss = loss / self.config.grad_accum_steps

            accum_loss += scaled_loss.item()
            self.scaler.scale(scaled_loss).backward()

        # Unscale gradients before clipping
        if self.config.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.current_step += 1
        return accum_loss

    def train(
        self,
        stop_after_steps: Optional[int] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run training loop up to config.max_steps or bounded stop_after_steps / time_budget_seconds.

        Args:
            stop_after_steps: If specified, stop training upon reaching this step count,
                allowing resuming later without altering the global learning rate schedule.
            time_budget_seconds: If specified, stop training if elapsed time exceeds budget.

        Returns:
            Dict[str, Any]: Detailed training experiment metrics and history.
        """
        target_step = self.config.max_steps
        if stop_after_steps is not None:
            target_step = min(self.config.max_steps, stop_after_steps)

        print(f"Starting training on device: {self.device} (AMP: {self.use_amp})")
        print(f"Global schedule max_steps: {self.config.max_steps} | Target step for this run: {target_step}")
        print(f"Micro-batch size: {self.config.batch_size}, Context: {self.config.block_size}, Accumulation: {self.config.grad_accum_steps}")

        start_time = time.time()
        initial_eval: Dict[str, float] = self.estimate_loss()
        print(f"Initial Loss - Train: {initial_eval['train']:.4f}, Val: {initial_eval['val']:.4f}")

        history: list[dict[str, Any]] = [
            {
                "step": self.current_step,
                "train_loss": initial_eval["train"],
                "val_loss": initial_eval["val"],
                "lr": self.get_lr_for_step(self.current_step),
                "time_sec": 0.0,
            }
        ]

        stopped_due_to_time: bool = False
        initial_step = self.current_step

        while self.current_step < target_step:
            step = self.current_step
            loss = self.train_step()

            elapsed = time.time() - start_time
            if time_budget_seconds is not None and elapsed >= time_budget_seconds:
                print(f"[!] Time budget reached ({elapsed:.1f}s >= {time_budget_seconds}s). Stopping run safely.")
                stopped_due_to_time = True
                break

            # Evaluation Interval
            if self.current_step % self.config.eval_interval == 0 or self.current_step == target_step:
                eval_loss = self.estimate_loss()
                lr = self.get_lr_for_step(self.current_step)
                peak_vram_mb = 0.0
                vram_info = ""
                if self.device.startswith("cuda"):
                    peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    vram_info = f", Peak VRAM: {peak_vram_mb:.1f} MB"

                print(
                    f"Step {self.current_step:4d}/{self.config.max_steps} | "
                    f"LR: {lr:.2e} | "
                    f"Train Loss: {eval_loss['train']:.4f} | "
                    f"Val Loss: {eval_loss['val']:.4f}{vram_info}"
                )
                history.append(
                    {
                        "step": self.current_step,
                        "train_loss": eval_loss["train"],
                        "val_loss": eval_loss["val"],
                        "lr": lr,
                        "time_sec": round(elapsed, 2),
                        "peak_vram_mb": round(peak_vram_mb, 1),
                    }
                )

            # Checkpoint Interval
            if self.current_step % self.config.checkpoint_interval == 0:
                ckpt_path = self.checkpoint_dir / f"step_{self.current_step}.pt"
                last_val = history[-1]["val_loss"] if history else None
                self.save_checkpoint(ckpt_path, val_loss=last_val)

        total_time = max(0.001, time.time() - start_time)
        final_eval = self.estimate_loss()

        steps_completed = self.current_step - initial_step
        tokens_per_step = self.config.batch_size * self.config.block_size * self.config.grad_accum_steps
        tokens_processed = steps_completed * tokens_per_step
        tokens_per_sec = tokens_processed / total_time

        # Save final / bounded checkpoint
        final_ckpt = self.checkpoint_dir / f"step_{self.current_step}.pt"
        self.save_checkpoint(final_ckpt, val_loss=final_eval["val"])

        return {
            "total_steps": self.current_step,
            "steps_this_run": steps_completed,
            "total_time_seconds": round(total_time, 2),
            "tokens_processed": tokens_processed,
            "tokens_per_second": round(tokens_per_sec, 1),
            "initial_train_loss": initial_eval["train"],
            "initial_val_loss": initial_eval["val"],
            "final_train_loss": final_eval["train"],
            "final_val_loss": final_eval["val"],
            "final_checkpoint": str(final_ckpt),
            "stopped_due_to_time": stopped_due_to_time,
            "device": self.device,
            "mixed_precision": self.use_amp,
            "history": history,
        }

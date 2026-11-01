"""Dedicated Supervised Fine-Tuning (SFT) trainer for GroundUpGPT.

Handles assistant-only masked loss training, token-weighted loss aggregation,
mixed precision (AMP), gradient accumulation, cosine learning rate schedule,
periodic held-out SFT validation, original LM benchmark tracking, atomic
checkpoint saving, and strict wall-clock time limit enforcement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    SFTDataPipeline,
    collate_sft_batch,
)

logger = logging.getLogger(__name__)

EXPECTED_TOKENIZER_FINGERPRINT = (
    "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
)


@dataclass
class SFTTrainingConfig:
    """Hyperparameters and configuration for SFT instruction fine-tuning.

    Attributes:
        base_checkpoint_path: Path to pretrained base checkpoint (Step 3000).
        sft_cache_dir: Directory containing cached SFT train/val tokens.
        checkpoint_dir: Directory to store SFT checkpoints.
        tokenizer_path: Path to BPE tokenizer JSON file.
        lm_val_path: Optional path to expanded validation tokens for LM benchmark.
        learning_rate: Initial peak learning rate for SFT.
        min_learning_rate: Minimum learning rate at end of cosine decay.
        weight_decay: AdamW weight decay coefficient.
        betas: AdamW beta coefficients.
        warmup_steps: Number of initial warmup steps.
        max_steps: Maximum optimizer steps to train (e.g. 500-800).
        micro_batch_size: Micro-batch size per forward/backward pass.
        grad_accum_steps: Number of micro-batches per optimizer step.
        max_grad_norm: Gradient clipping threshold.
        val_interval: Steps between evaluations.
        max_train_seconds: Mandatory hard limit on GPU training time (seconds).
        seed: Random seed for reproducibility.
        device: Execution device ('cuda' or 'cpu').
    """

    base_checkpoint_path: Path = Path("checkpoints/5m_phase2/model_5m_best.pt")
    sft_cache_dir: Path = Path("data/cache_sft")
    checkpoint_dir: Path = Path("checkpoints/5m_sft")
    tokenizer_path: Path = Path("checkpoints/bpe_vocab4096.json")
    lm_val_path: Optional[Path] = Path("data/cache_expanded/val_tokens.pt")
    expected_param_count: Optional[int] = None
    best_checkpoint_filename: str = "model_5m_sft_best.pt"
    latest_checkpoint_filename: str = "model_5m_sft_latest.pt"
    lm_track_A_path: Optional[Path] = None
    lm_track_B_path: Optional[Path] = None
    lm_track_C_path: Optional[Path] = None
    eval_windows: int = 100
    learning_rate: float = 1.5e-4
    min_learning_rate: float = 1.5e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 50
    max_steps: int = 650
    micro_batch_size: int = 16
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    val_interval: int = 50
    max_train_seconds: float = 580.0  # Safe boundary below 600s (10 min limit)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.base_checkpoint_path = Path(self.base_checkpoint_path)
        self.sft_cache_dir = Path(self.sft_cache_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.tokenizer_path = Path(self.tokenizer_path)
        if self.lm_val_path is not None:
            self.lm_val_path = Path(self.lm_val_path)
        if self.lm_track_A_path is not None:
            self.lm_track_A_path = Path(self.lm_track_A_path)
        if self.lm_track_B_path is not None:
            self.lm_track_B_path = Path(self.lm_track_B_path)
        if self.lm_track_C_path is not None:
            self.lm_track_C_path = Path(self.lm_track_C_path)


class SFTTrainer:
    """Trainer orchestrating the fine-tuning of GroundUpGPT on supervised instructions."""

    def __init__(
        self,
        config: SFTTrainingConfig,
        pipeline: Optional[SFTDataPipeline] = None,
        model: Optional[GroundUpGPT] = None,
        tokenizer: Optional[BPETokenizer] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # 1. Set seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # 2. Tokenizer
        if tokenizer is None:
            if not self.config.tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer file not found: {self.config.tokenizer_path}"
                )
            self.tokenizer = BPETokenizer.load(self.config.tokenizer_path)
        else:
            self.tokenizer = tokenizer

        tok_fp = self.tokenizer.get_fingerprint()
        if tok_fp != EXPECTED_TOKENIZER_FINGERPRINT:
            raise ValueError(
                f"Tokenizer fingerprint mismatch! Expected {EXPECTED_TOKENIZER_FINGERPRINT}, got {tok_fp}"
            )

        # 3. Pipeline
        if pipeline is None:
            self.pipeline = SFTDataPipeline.load(
                self.config.sft_cache_dir, self.tokenizer
            )
        else:
            self.pipeline = pipeline

        # Dedicated data sampling generator
        self.data_rng = torch.Generator(device="cpu")
        self.data_rng.manual_seed(config.seed)

        # 4. Model
        if model is None:
            if not self.config.base_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Base checkpoint not found: {self.config.base_checkpoint_path}"
                )

            h = hashlib.sha256()
            with open(self.config.base_checkpoint_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            self.source_checkpoint_sha256 = h.hexdigest()

            base_ckpt = torch.load(
                self.config.base_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.source_step = base_ckpt.get("step") or base_ckpt.get("best_step", 3000)

            val_l = base_ckpt.get("best_val_loss")
            if val_l is None:
                val_l = base_ckpt.get("val_loss")
            if val_l is None:
                v_metrics = (
                    base_ckpt.get("final_val_metrics")
                    or base_ckpt.get("val_metrics")
                    or {}
                )
                val_l = v_metrics.get("track_A_combined_loss") or v_metrics.get("loss")
            if val_l is None:
                if "5m" in str(self.config.base_checkpoint_path).lower():
                    val_l = 4.533
                else:
                    raise ValueError(
                        f"Could not extract pretraining validation loss from checkpoint: {self.config.base_checkpoint_path}"
                    )
            self.source_val_loss = float(val_l)

            m_cfg = GPTConfig(**base_ckpt["model_config"])
            self.model = GroundUpGPT(m_cfg)
            self.model.load_state_dict(base_ckpt["model_state_dict"], strict=True)
            logger.info(
                f"Loaded pretrained weights from {self.config.base_checkpoint_path} "
                f"(Source Step {self.source_step}, val_loss {self.source_val_loss:.4f}, SHA-256 {self.source_checkpoint_sha256[:10]}...)"
            )

            param_count = self.model.get_num_params(non_embedding=False)
            if self.config.expected_param_count is not None:
                if param_count != self.config.expected_param_count:
                    raise ValueError(
                        f"Expected parameter count {self.config.expected_param_count:,}, got {param_count:,}"
                    )
            elif "5m" in str(self.config.base_checkpoint_path).lower():
                if param_count != 5_051_904:
                    raise ValueError(
                        f"Expected exact parameter count 5,051,904, got {param_count:,}"
                    )
        else:
            self.model = model
            self.source_step = 3000
            self.source_val_loss = 4.533
            self.source_checkpoint_sha256 = ""

        self.model.to(self.device)

        # 5. Optimizer & AMP Scaler
        # Initialize fresh AdamW optimizer for SFT objective
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda")
        )

        # 6. Preload frozen LM validation tokens if present
        self.lm_val_tokens: Optional[torch.Tensor] = None
        self.lm_track_A_tokens: Optional[torch.Tensor] = None
        self.lm_track_B_tokens: Optional[torch.Tensor] = None
        self.lm_track_C_tokens: Optional[torch.Tensor] = None

        if self.config.lm_track_A_path and self.config.lm_track_A_path.is_file():
            self.lm_track_A_tokens = torch.load(
                self.config.lm_track_A_path, weights_only=True
            )
            logger.info(
                f"Loaded frozen Track A tokens from {self.config.lm_track_A_path} ({len(self.lm_track_A_tokens):,} tokens)"
            )
        if self.config.lm_track_B_path and self.config.lm_track_B_path.is_file():
            self.lm_track_B_tokens = torch.load(
                self.config.lm_track_B_path, weights_only=True
            )
            logger.info(
                f"Loaded frozen Track B tokens from {self.config.lm_track_B_path} ({len(self.lm_track_B_tokens):,} tokens)"
            )
        if self.config.lm_track_C_path and self.config.lm_track_C_path.is_file():
            self.lm_track_C_tokens = torch.load(
                self.config.lm_track_C_path, weights_only=True
            )
            logger.info(
                f"Loaded frozen Track C tokens from {self.config.lm_track_C_path} ({len(self.lm_track_C_tokens):,} tokens)"
            )

        if self.lm_track_A_tokens is not None:
            if self.lm_track_B_tokens is None:
                orig_val = Path("data/cache/val_tokens.pt")
                if orig_val.is_file():
                    self.lm_track_B_tokens = torch.load(orig_val, weights_only=True)
                else:
                    self.lm_track_B_tokens = self.lm_track_A_tokens[:168741]
            if self.lm_track_C_tokens is None:
                gt_val_offset = 1118258
                if len(self.lm_track_A_tokens) > gt_val_offset:
                    self.lm_track_C_tokens = self.lm_track_A_tokens[gt_val_offset:]

        self.eval_windows = self.config.eval_windows
        block_size = self.model.config.block_size
        eval_rng = torch.Generator().manual_seed(42)
        if self.lm_track_A_tokens is not None:
            max_idx_a = max(0, len(self.lm_track_A_tokens) - block_size - 1)
            self.indices_A = torch.randint(0, max_idx_a, (self.eval_windows,), generator=eval_rng)
            if self.lm_track_B_tokens is not None and len(self.lm_track_B_tokens) > block_size + 1:
                max_idx_b = len(self.lm_track_B_tokens) - block_size - 1
                self.indices_B = torch.randint(0, max_idx_b, (self.eval_windows,), generator=eval_rng)
            else:
                self.indices_B = torch.empty(0, dtype=torch.long)
            if self.lm_track_C_tokens is not None and len(self.lm_track_C_tokens) > block_size + 1:
                max_idx_c = len(self.lm_track_C_tokens) - block_size - 1
                self.indices_C = torch.randint(0, max_idx_c, (self.eval_windows,), generator=eval_rng)
            else:
                self.indices_C = torch.empty(0, dtype=torch.long)

        if (
            self.config.lm_val_path is not None
            and self.config.lm_val_path.is_file()
            and self.lm_track_A_tokens is None
        ):
            self.lm_val_tokens = torch.load(
                self.config.lm_val_path, weights_only=True
            )
            logger.info(
                f"Loaded frozen LM benchmark tokens from {self.config.lm_val_path} ({len(self.lm_val_tokens):,} tokens)"
            )

        self.last_lm_metrics: Dict[str, float] = {}
        self.base_lm_tracks: Dict[str, float] = {}

        # Training trackers
        self.sft_step = 0
        self.successful_updates = 0
        self.skipped_updates = 0
        self.micro_batches_processed = 0
        self.conversation_sample_exposures = 0
        self.active_assistant_targets_processed = 0
        self.total_tokens_processed = 0
        self.best_sft_val_loss = float("inf")
        self.best_sft_val_ppl = float("inf")
        self.history: List[Dict[str, Any]] = []
        self.stop_reason = ""

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_lr(self, step: int) -> float:
        """Compute learning rate using linear warmup followed by cosine decay."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (
                float(step + 1) / float(max(1, self.config.warmup_steps))
            )
        # Cosine decay down to min_learning_rate
        decay_ratio = float(step - self.config.warmup_steps) / float(
            max(1, self.config.max_steps - self.config.warmup_steps)
        )
        decay_ratio = min(1.0, max(0.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_learning_rate + coeff * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def evaluate_sft(
        self, batch_size: int = 16
    ) -> Tuple[float, float, int]:
        """Evaluate exact token-weighted assistant-only loss on held-out SFT validation split.

        Returns:
            Tuple[float, float, int]: (loss, perplexity, total_active_tokens)
        """
        self.model.eval()
        val_ds = self.pipeline.val_dataset
        total_loss_sum = 0.0
        total_active_tokens = 0

        with torch.inference_mode():
            for i in range(0, len(val_ds), batch_size):
                batch_items = [
                    val_ds[j] for j in range(i, min(i + batch_size, len(val_ds)))
                ]
                collated = collate_sft_batch(
                    batch_items,
                    pad_token_id=0,
                    pad_to_max_len=False,
                    max_seq_len=self.model.config.block_size,
                )
                x = collated["input_ids"].to(self.device)
                y = collated["target_ids"].to(self.device)
                mask = collated["loss_mask"].to(self.device)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    logits, _ = self.model(x)
                    loss_sum = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )

                total_loss_sum += loss_sum.item()
                total_active_tokens += int(mask.sum().item())

        val_loss = total_loss_sum / max(1, total_active_tokens)
        val_ppl = math.exp(min(20.0, val_loss))
        return val_loss, val_ppl, total_active_tokens

    def evaluate_lm_benchmark(
        self, num_windows: Optional[int] = None, batch_size: int = 16
    ) -> Tuple[float, float]:
        """Evaluate model on frozen pretraining validation benchmark.

        Returns:
            Tuple[float, float]: (lm_loss, lm_perplexity)
        """
        n_win = num_windows if num_windows is not None else self.config.eval_windows

        # Multi-track pretraining validation (Track A, B, C)
        if self.lm_track_A_tokens is not None:
            self.model.eval()
            block_size = self.model.config.block_size
            use_cuda = (self.device.type == "cuda")

            def _eval_tokens_on_indices(tokens: torch.Tensor, indices: torch.Tensor) -> Tuple[float, float]:
                tot_loss = 0.0
                tot_samples = 0
                eval_idx = indices[:n_win]
                with torch.inference_mode():
                    for i in range(0, len(eval_idx), batch_size):
                        batch_idx = eval_idx[i : i + batch_size]
                        x = torch.stack([tokens[j : j + block_size].long() for j in batch_idx]).to(self.device)
                        y = torch.stack([tokens[j + 1 : j + block_size + 1].long() for j in batch_idx]).to(self.device)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                            logits, _ = self.model(x)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
                        tot_loss += loss.item()
                        tot_samples += x.numel()
                m_loss = tot_loss / max(1, tot_samples)
                m_ppl = math.exp(min(20.0, m_loss))
                return round(m_loss, 4), round(m_ppl, 2)

            loss_a, ppl_a = _eval_tokens_on_indices(self.lm_track_A_tokens, self.indices_A)
            loss_b, ppl_b = (0.0, 0.0)
            if self.lm_track_B_tokens is not None and len(self.indices_B) > 0:
                loss_b, ppl_b = _eval_tokens_on_indices(self.lm_track_B_tokens, self.indices_B)
            loss_c, ppl_c = (0.0, 0.0)
            if self.lm_track_C_tokens is not None and len(self.indices_C) > 0:
                loss_c, ppl_c = _eval_tokens_on_indices(self.lm_track_C_tokens, self.indices_C)

            self.last_lm_metrics = {
                "track_A_combined_loss": loss_a,
                "track_A_combined_ppl": ppl_a,
                "track_B_orig_gt_loss": loss_b,
                "track_B_orig_gt_ppl": ppl_b,
                "track_C_wiki_loss": loss_c,
                "track_C_wiki_ppl": ppl_c,
            }
            return loss_a, ppl_a

        # Fallback to single lm_val_tokens (5M behavior)
        if self.lm_val_tokens is None:
            return 0.0, 0.0

        self.model.eval()
        block_size = self.model.config.block_size
        max_idx = len(self.lm_val_tokens) - block_size - 1
        g = torch.Generator().manual_seed(42)
        rand_starts = torch.randint(
            0, max_idx, (n_win,), generator=g
        ).tolist()

        total_loss_sum = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for i in range(0, n_win, batch_size):
                starts = rand_starts[i : min(i + batch_size, n_win)]
                x = torch.stack(
                    [
                        self.lm_val_tokens[s : s + block_size].long()
                        for s in starts
                    ]
                ).to(self.device)
                y = torch.stack(
                    [
                        self.lm_val_tokens[
                            s + 1 : s + 1 + block_size
                        ].long()
                        for s in starts
                    ]
                ).to(self.device)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    logits, _ = self.model(x)
                    loss_sum = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction="sum",
                    )

                total_loss_sum += loss_sum.item()
                total_tokens += x.numel()

        lm_loss = total_loss_sum / max(1, total_tokens)
        lm_ppl = math.exp(min(20.0, lm_loss))
        self.last_lm_metrics = {
            "lm_loss": round(lm_loss, 4),
            "lm_ppl": round(lm_ppl, 2),
        }
        return round(lm_loss, 4), round(lm_ppl, 2)

    def save_checkpoint(
        self,
        dest_path: Path,
        sft_val_loss: float,
        sft_val_ppl: float,
        lm_val_loss: float,
        lm_val_ppl: float,
        is_best: bool = False,
    ) -> None:
        """Atomically serialize model weights and all training states to disk."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(".tmp")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "sft_step": self.sft_step,
            "global_step": self.source_step + self.sft_step,
            "source_step": self.source_step,
            "source_checkpoint": str(self.config.base_checkpoint_path),
            "source_checkpoint_sha256": getattr(self, "source_checkpoint_sha256", ""),
            "source_val_loss": self.source_val_loss,
            "successful_optimizer_steps": self.successful_updates,
            "skipped_optimizer_steps": self.skipped_updates,
            "micro_batches_processed": self.micro_batches_processed,
            "conversation_sample_exposures": self.conversation_sample_exposures,
            "active_assistant_targets_processed": self.active_assistant_targets_processed,
            "tokens_processed": self.total_tokens_processed,
            "total_tokens_processed": self.total_tokens_processed,
            "sft_val_loss": sft_val_loss,
            "sft_val_perplexity": sft_val_ppl,
            "best_sft_val_loss": self.best_sft_val_loss,
            "best_sft_val_perplexity": self.best_sft_val_ppl,
            "lm_benchmark_val_loss": lm_val_loss,
            "lm_benchmark_perplexity": lm_val_ppl,
            "lm_tracks": getattr(self, "last_lm_metrics", {}),
            "tokenizer_type": "bpe",
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "tokenizer_fingerprint": self.tokenizer.get_fingerprint(),
            "sft_dataset_source": "Databricks Dolly 15k (CC BY-SA 3.0)",
            "stop_reason": self.stop_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state()
                if torch.cuda.is_available()
                else None
            ),
            "data_rng_state": (
                self.data_rng.get_state()
                if hasattr(self, "data_rng")
                else None
            ),
        }

        torch.save(checkpoint, temp_path)
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        logger.info(
            f"Saved SFT checkpoint atomically to {dest_path} (is_best={is_best})"
        )

    def train(self) -> Dict[str, Any]:
        """Execute bounded SFT training run with strict wall-clock time limit."""
        t_start = time.time()
        logger.info("=" * 70)
        logger.info("Starting GroundUpGPT Supervised Fine-Tuning (SFT)")
        logger.info(f"Device: {self.device}")
        logger.info(f"Target steps: {self.config.max_steps}")
        logger.info(f"Micro-batch: {self.config.micro_batch_size}, Grad accum: {self.config.grad_accum_steps}")
        logger.info(f"Effective batch size: {self.config.micro_batch_size * self.config.grad_accum_steps}")
        logger.info(f"Peak LR: {self.config.learning_rate}, Min LR: {self.config.min_learning_rate}")
        logger.info(f"Max train wall-clock time: {self.config.max_train_seconds}s ({self.config.max_train_seconds/60:.1f} mins)")
        logger.info("=" * 70)

        # 1. Pre-SFT Baseline Evaluation (Step 0)
        logger.info("Computing pre-SFT Step 0 baseline evaluation...")
        base_sft_loss, base_sft_ppl, _ = self.evaluate_sft()
        base_lm_loss, base_lm_ppl = self.evaluate_lm_benchmark()
        self.base_lm_tracks = dict(getattr(self, "last_lm_metrics", {}))
        self.best_sft_val_loss = base_sft_loss
        self.best_sft_val_ppl = base_sft_ppl

        logger.info(
            f"Pre-SFT Baseline -> SFT Val Loss: {base_sft_loss:.4f}, SFT PPL: {base_sft_ppl:.2f} | "
            f"LM Val Loss: {base_lm_loss:.4f}, LM PPL: {base_lm_ppl:.2f}"
        )

        step_losses: List[float] = []
        consecutive_val_degradations = 0

        best_ckpt_path = (
            self.config.checkpoint_dir / self.config.best_checkpoint_filename
        )
        latest_ckpt_path = (
            self.config.checkpoint_dir / self.config.latest_checkpoint_filename
        )

        self.model.train()

        # Training loop
        while self.sft_step < self.config.max_steps:
            elapsed = time.time() - t_start
            if elapsed >= self.config.max_train_seconds:
                self.stop_reason = f"max_train_seconds_reached ({elapsed:.1f}s >= {self.config.max_train_seconds}s)"
                logger.warning(
                    f"Wall-clock training budget reached ({elapsed:.1f}s). Gracefully stopping training."
                )
                break

            current_lr = self.get_lr(self.sft_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Exact token-weighted gradient accumulation over micro-batches
            micro_batches = [
                self.pipeline.get_batch(
                    split="train",
                    batch_size=self.config.micro_batch_size,
                    device=self.device,
                    pad_to_max_len=True,
                    generator=self.data_rng,
                )
                for _ in range(self.config.grad_accum_steps)
            ]
            total_step_active = sum(int(m.sum().item()) for _, _, m in micro_batches)
            denom = max(1, total_step_active)

            for x, y, mask in micro_batches:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(self.device.type == "cuda"),
                ):
                    logits, _ = self.model(x)
                    loss_sum = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=IGNORE_INDEX,
                        reduction="sum",
                    )
                    loss_scaled = loss_sum / denom

                self.scaler.scale(loss_scaled).backward()
                accum_loss += loss_sum.item() / denom
                active_count = int(mask.sum().item())
                self.micro_batches_processed += 1
                self.conversation_sample_exposures += len(x)
                self.active_assistant_targets_processed += active_count
                self.total_tokens_processed += active_count

            # Gradient clipping & optimizer step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            # Check for NaN / Inf
            if not math.isfinite(accum_loss) or not math.isfinite(
                grad_norm.item()
            ):
                self.stop_reason = (
                    f"numerical_instability (loss={accum_loss}, grad_norm={grad_norm.item()})"
                )
                logger.error(
                    f"Stopping early due to non-finite loss/gradient: {self.stop_reason}"
                )
                break

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()

            if scale_after < scale_before:
                self.skipped_updates += 1
                logger.warning(
                    f"Step {self.sft_step}: AMP scale reduced from {scale_before} to {scale_after}; optimizer step skipped."
                )
            else:
                self.successful_updates += 1

            self.sft_step += 1
            step_losses.append(accum_loss)

            # Logging
            if self.sft_step % 10 == 0 or self.sft_step == 1:
                recent_loss = sum(step_losses[-10:]) / len(step_losses[-10:])
                logger.info(
                    f"Step {self.sft_step:4d}/{self.config.max_steps} | "
                    f"Loss: {recent_loss:.4f} | LR: {current_lr:.6f} | "
                    f"GradNorm: {grad_norm.item():.2f} | "
                    f"Elapsed: {time.time()-t_start:.1f}s"
                )

            # Periodic validation
            if (
                self.sft_step % self.config.val_interval == 0
                or self.sft_step == self.config.max_steps
            ):
                val_sft_loss, val_sft_ppl, _ = self.evaluate_sft()
                val_lm_loss, val_lm_ppl = self.evaluate_lm_benchmark()

                logger.info(
                    f"--- Eval at Step {self.sft_step} --- "
                    f"SFT Val Loss: {val_sft_loss:.4f} (PPL: {val_sft_ppl:.2f}) | "
                    f"LM Val Loss: {val_lm_loss:.4f} (PPL: {val_lm_ppl:.2f})"
                )

                is_best = val_sft_loss < self.best_sft_val_loss
                if is_best:
                    self.best_sft_val_loss = val_sft_loss
                    self.best_sft_val_ppl = val_sft_ppl
                    self.save_checkpoint(
                        best_ckpt_path,
                        val_sft_loss,
                        val_sft_ppl,
                        val_lm_loss,
                        val_lm_ppl,
                        is_best=True,
                    )
                    consecutive_val_degradations = 0
                else:
                    consecutive_val_degradations += 1

                self.save_checkpoint(
                    latest_ckpt_path,
                    val_sft_loss,
                    val_sft_ppl,
                    val_lm_loss,
                    val_lm_ppl,
                    is_best=False,
                )

                self.history.append(
                    {
                        "step": self.sft_step,
                        "successful_updates": self.successful_updates,
                        "skipped_updates": self.skipped_updates,
                        "train_loss": accum_loss,
                        "sft_val_loss": val_sft_loss,
                        "sft_val_ppl": val_sft_ppl,
                        "lm_val_loss": val_lm_loss,
                        "lm_val_ppl": val_lm_ppl,
                        "lm_tracks": getattr(self, "last_lm_metrics", {}),
                        "elapsed_seconds": time.time() - t_start,
                    }
                )

                # Early stopping if validation degrades 4 times consecutively
                if consecutive_val_degradations >= 4:
                    self.stop_reason = "early_stopping_val_degradation"
                    logger.warning(
                        "Validation loss degraded consecutively. Triggering early stopping."
                    )
                    break

                self.model.train()

        if not self.stop_reason:
            self.stop_reason = "completed_max_steps"

        total_elapsed = time.time() - t_start
        logger.info("=" * 70)
        logger.info(f"Training finished: {self.stop_reason}")
        logger.info(f"Total steps: {self.sft_step} (Successful: {self.successful_updates}, Skipped: {self.skipped_updates})")
        logger.info(f"Total GPU time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        logger.info(f"Best SFT Val Loss: {self.best_sft_val_loss:.4f} (PPL: {self.best_sft_val_ppl:.2f})")
        logger.info("=" * 70)

        # Final evaluation
        final_sft_loss, final_sft_ppl, _ = self.evaluate_sft()
        final_lm_loss, final_lm_ppl = self.evaluate_lm_benchmark()

        # Always save latest checkpoint
        self.save_checkpoint(
            latest_ckpt_path,
            final_sft_loss,
            final_sft_ppl,
            final_lm_loss,
            final_lm_ppl,
            is_best=False,
        )

        return {
            "source_step": self.source_step,
            "source_val_loss": self.source_val_loss,
            "source_checkpoint_sha256": getattr(self, "source_checkpoint_sha256", ""),
            "sft_steps": self.sft_step,
            "successful_updates": self.successful_updates,
            "skipped_updates": self.skipped_updates,
            "micro_batches_processed": self.micro_batches_processed,
            "conversation_sample_exposures": self.conversation_sample_exposures,
            "active_assistant_targets_processed": self.active_assistant_targets_processed,
            "total_elapsed_seconds": total_elapsed,
            "tokens_processed": self.total_tokens_processed,
            "pre_sft_baseline": {
                "sft_val_loss": base_sft_loss,
                "sft_val_ppl": base_sft_ppl,
                "lm_val_loss": base_lm_loss,
                "lm_val_ppl": base_lm_ppl,
                "lm_tracks": getattr(self, "base_lm_tracks", {}),
            },
            "post_sft_metrics": {
                "best_sft_val_loss": self.best_sft_val_loss,
                "best_sft_val_ppl": self.best_sft_val_ppl,
                "final_sft_val_loss": final_sft_loss,
                "final_sft_val_ppl": final_sft_ppl,
                "final_lm_val_loss": final_lm_loss,
                "final_lm_val_ppl": final_lm_ppl,
                "final_lm_tracks": getattr(self, "last_lm_metrics", {}),
            },
            "best_checkpoint_path": str(best_ckpt_path),
            "latest_checkpoint_path": str(latest_ckpt_path),
            "stop_reason": self.stop_reason,
            "history": self.history,
        }

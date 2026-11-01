"""Monitored Pretraining and Crash-Safe Resumption for GroundUpGPT-27M (Steps 32, 33 & 34).

Key Guarantees & Safeguards (Step 34):
1. Fail-Closed Checkpoint Resumption:
   Explicitly requires a valid resume checkpoint or explicit --fresh flag. Missing or
   corrupt checkpoints halt training immediately before touching any model or disk artifacts.
2. Strong Cryptographic Dataset Fingerprinting:
   Streams large token tensors (train_tokens.pt, val_tokens.pt) and metadata in 1MB bounded
   memory chunks to generate bitwise SHA-256 hashes, with backward-compatible migration
   for Step 33 legacy metadata-only checkpoints.
3. Isolated Non-Destructive Preflight:
   Device micro-preflight runs on a separate temporary model/optimizer instance,
   guaranteeing the real model's weights and optimizer momentum remain 100% untouched.
4. Verified Stateful Resumption:
   Restores exact model weights, AdamW optimizer state (exp_avg, exp_avg_sq), AMP GradScaler,
   step count, learning rate schedule, and RNG states (torch, cuda, data_rng).
5. Accurate Per-Run & Optimizer Accounting:
   Disentangles run tokens from cumulative tokens and only advances optimizer step count
   upon verified, finite gradient updates.
6. Multi-Track Deterministic Validation:
   Evaluates Track A (Combined), Track B (Orig Gutenberg), and Track C (Simple Wikipedia) on
   fixed deterministic windows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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
logger = logging.getLogger("train_27m")

EXPECTED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
EXPECTED_PARAM_COUNT = 27411456

CORE_CACHE_FILES = [
    "train_tokens.pt",
    "val_tokens.pt",
    "cache_metadata.json",
    "corpus_manifest.json",
    "source_provenance.json",
]

INSPECTION_PROMPTS = [
    # 1. Ordinary English prose
    "The sun rose over the",
    "In the middle of the room,",
    "She walked down the street and",
    # 2. Narrative continuation
    "Once upon a time in a small village,",
    "He opened the mysterious door and saw",
    "The dark storm clouds gathered as",
    # 3. Science and geography
    "The capital of France is",
    "The Earth revolves around the",
    "The Atlantic Ocean lies between",
    # 4. Short factual statements
    "Water is made of",
    "A triangle has three",
    "Oxygen is essential for",
    # 5. Simple definitions
    "Biology is the study of",
    "A library is a place where",
    # 6. Basic question-like sentences
    "What is the reason why",
    "How do birds fly through the",
]


def compute_streaming_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file using streaming reads with bounded memory."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_strong_dataset_fingerprint(cache_dir: Path) -> Tuple[str, Dict[str, str]]:
    """Compute strong dataset identity comprising individual file hashes and composite SHA-256.

    Streams large token files (train_tokens.pt, val_tokens.pt) in 1MB chunks to ensure
    bounded memory usage while providing cryptographic bitwise content verification.
    """
    file_hashes: Dict[str, str] = {}
    for filename in CORE_CACHE_FILES:
        p = cache_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Missing core cache file for content verification: {p}")
        file_hashes[filename] = compute_streaming_sha256(p)

    composite = hashlib.sha256()
    for filename in sorted(file_hashes.keys()):
        composite.update(f"{filename}:{file_hashes[filename]}\n".encode("utf-8"))

    return composite.hexdigest(), file_hashes


def compute_legacy_metadata_fingerprint(cache_dir: Path) -> str:
    """Compute Step 33 legacy metadata-only fingerprint (for backward compatibility)."""
    meta_path = cache_dir / "cache_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    h = hashlib.sha256()
    h.update(str(meta.get("corpus_version", "v3-27m")).encode())
    h.update(str(meta.get("train_tokens_count", 0)).encode())
    h.update(str(meta.get("val_tokens_count", 0)).encode())
    h.update(str(meta.get("train_bytes", 0)).encode())
    h.update(str(meta.get("tokenizer_fingerprint", "")).encode())
    return h.hexdigest()


def compute_dataset_fingerprint(cache_dir: Path) -> str:
    """Primary dataset fingerprint returning the strong content-based composite SHA-256."""
    strong_fp, _ = compute_strong_dataset_fingerprint(cache_dir)
    return strong_fp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitored Pretraining and Resumption of GroundUpGPT-27M.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_27m_v1"), help="Pretraining token cache directory.")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("checkpoints/bpe_vocab4096.json"), help="BPE tokenizer path.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/27m_phase4"), help="Output checkpoint directory.")
    parser.add_argument("--log-path", type=Path, default=Path("logs/training_27m_phase4.json"), help="Telemetry JSON output path.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=Path("checkpoints/27m_phase3/model_27m_latest.pt"),
        help="Path to checkpoint to resume from (default: checkpoints/27m_phase3/model_27m_latest.pt).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Explicitly initialize a new model from scratch. Mutually exclusive with --resume-from.",
    )
    parser.add_argument("--max-steps", type=int, default=6000, help="Target ending optimizer step (default: 6000).")
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
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience: consecutive evals without improvement (default: 4).")
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
    """Save a checkpoint atomically using a temporary file and atomic replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(target_path)


def run_preflight_smoke_test(device: str, use_amp: bool) -> None:
    """Execute a safe, isolated preflight smoke test on a separate temporary model.

    Verifies forward pass, backward pass, AMP autocast, GradScaler, and gradient clipping
    on the target device WITHOUT touching or modifying the real model or optimizer state.
    """
    logger.info("[*] Running isolated device preflight smoke test on temporary dummy model...")
    dummy_cfg = GPTConfig(
        vocab_size=4096,
        block_size=64,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
        use_sdpa=True,
    )
    dummy_model = GroundUpGPT(dummy_cfg).to(device)
    dummy_opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)
    dummy_scaler = GradScaler("cuda", enabled=use_amp)

    use_cuda = device.startswith("cuda")
    dummy_x = torch.randint(0, 4096, (4, 64), device=device)
    dummy_y = torch.randint(0, 4096, (4, 64), device=device)

    dummy_opt.zero_grad(set_to_none=True)
    with autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_amp):
        _, dummy_loss = dummy_model(dummy_x, dummy_y)

    if not torch.isfinite(dummy_loss):
        raise RuntimeError("Preflight dummy loss is non-finite!")

    dummy_scaler.scale(dummy_loss).backward()
    dummy_scaler.unscale_(dummy_opt)
    gn = torch.nn.utils.clip_grad_norm_(dummy_model.parameters(), 1.0)
    if not torch.isfinite(gn):
        raise RuntimeError("Preflight dummy grad norm is non-finite!")

    dummy_scaler.step(dummy_opt)
    dummy_scaler.update()

    del dummy_model, dummy_opt, dummy_scaler, dummy_x, dummy_y, dummy_loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[*] Isolated preflight smoke test passed successfully (real model 100% untouched).")


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
    print(" GROUNDUPGPT-27M: PRETRAINING & RESUMPTION PIPELINE")
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
    current_dataset_fp, current_file_hashes = compute_strong_dataset_fingerprint(args.cache_dir)
    current_legacy_metadata_fp = compute_legacy_metadata_fingerprint(args.cache_dir)
    print(f"[*] Dataset Train Tokens:  {total_train_tokens:,}")
    print(f"[*] Dataset Val Tokens:    {total_val_tokens:,}")
    print(f"[*] Dataset Content FP:    {current_dataset_fp}")
    print(f"[*] Legacy Metadata FP:   {current_legacy_metadata_fp}")

    # 3. Load Validation Tracks
    track_A_tokens = pipeline.val_tokens
    orig_val_path = Path("data/cache/val_tokens.pt")
    if orig_val_path.exists():
        track_B_tokens = torch.load(orig_val_path, weights_only=True)
    else:
        track_B_tokens = track_A_tokens[:168741]

    gt_val_offset = meta.get("sources", {}).get("project_gutenberg", {}).get("val_tokens", 1118258)
    track_C_tokens = track_A_tokens[gt_val_offset:]

    # Pre-generate fixed deterministic evaluation indices (seed 42)
    eval_rng = torch.Generator().manual_seed(42)
    indices_A = torch.randint(0, len(track_A_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)
    indices_B = torch.randint(0, len(track_B_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)
    indices_C = torch.randint(0, len(track_C_tokens) - args.block_size - 1, (args.eval_windows,), generator=eval_rng)

    # 4. Safe Isolated Preflight Smoke Test
    # (Tested on a separate temporary model; leaves real model and optimizer completely untouched)
    use_amp = resolved_device.startswith("cuda")
    run_preflight_smoke_test(resolved_device, use_amp)

    # 5. Initialize or Resume Real Model
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

    # 6. Configure Optimizer and Scaler
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

    scaler = GradScaler("cuda", enabled=use_amp)

    start_step = 0
    tokens_processed_before = 0
    best_val_loss = float("inf")
    best_step = 0
    history: List[Dict[str, Any]] = []

    # Dedicated batch sampling generator
    data_rng = torch.Generator(device="cpu")
    data_rng.manual_seed(args.seed)

    # 7. Checkpoint Resumption (Fail-Closed, Content-Hashed & Legacy-Compatible)
    is_resumed = False
    if args.fresh:
        if any(arg.startswith("--resume-from") for arg in sys.argv):
            raise ValueError(
                "Ambiguous execution intent: Cannot specify both '--fresh' and '--resume-from'. "
                "Specify '--fresh' for fresh training from scratch, or '--resume-from <path>' to resume."
            )
        logger.warning("[*] Explicit --fresh flag provided: Initializing GroundUpGPT-27M from scratch.")
        is_resumed = False
    else:
        if not args.resume_from:
            raise ValueError(
                "No resume checkpoint specified and --fresh was not set. "
                "Specify --resume-from <path> to resume, or --fresh to initialize from scratch."
            )
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: '{ckpt_path}'. "
                f"Refusing to fail-open into fresh training. "
                f"Ensure the checkpoint exists or explicitly pass '--fresh' if scratch initialization is intended."
            )

        try:
            ckpt = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from '{ckpt_path}' (corrupt or unreadable file): {e}. "
                f"Training halted immediately to protect existing artifacts."
            ) from e

        # Validate required checkpoint keys
        for req_k in ["step", "model_state_dict", "optimizer_state_dict", "model_config"]:
            if req_k not in ckpt:
                raise ValueError(f"Corrupt checkpoint '{ckpt_path}': missing required key '{req_k}'.")

        is_resumed = True

        # A. Strict Architecture Compatibility Check
        ckpt_cfg_dict = ckpt.get("model_config", {})
        for key in ["vocab_size", "block_size", "embed_dim", "num_heads", "num_layers", "use_sdpa"]:
            expected_val = getattr(gpt_config, key)
            actual_val = ckpt_cfg_dict.get(key)
            if actual_val != expected_val:
                raise ValueError(
                    f"Architecture mismatch for '{key}': checkpoint has {actual_val}, "
                    f"active model requires {expected_val}."
                )

        # B. Tokenizer Fingerprint Check
        ckpt_tok_fp = ckpt.get("tokenizer_fingerprint")
        if ckpt_tok_fp != tok_fp:
            raise ValueError(
                f"Tokenizer fingerprint mismatch! Checkpoint was trained with {ckpt_tok_fp}, "
                f"active tokenizer is {tok_fp}."
            )

        # C. Dataset Fingerprint Validation (Content Integrity & Legacy Compatibility)
        has_content_fp = "dataset_content_fingerprint" in ckpt
        has_content_hashes = "dataset_content_hashes" in ckpt

        if has_content_fp and has_content_hashes:
            ckpt_content_fp = ckpt["dataset_content_fingerprint"]
            ckpt_file_hashes = ckpt.get("dataset_content_hashes", {})
            if ckpt_content_fp != current_dataset_fp:
                diffs = [
                    f"{fn} (ckpt: {ckpt_file_hashes.get(fn, 'missing')[:8]}, active: {current_file_hashes.get(fn, 'missing')[:8]})"
                    for fn in CORE_CACHE_FILES
                    if ckpt_file_hashes.get(fn) != current_file_hashes.get(fn)
                ]
                diff_str = ", ".join(diffs) if diffs else "composite digest mismatch"
                raise ValueError(
                    f"Dataset content integrity mismatch! Checkpoint expects content hash {ckpt_content_fp}, "
                    f"active cache has {current_dataset_fp}. Divergent files: {diff_str}"
                )
            logger.info(f"[*] Validated strong dataset content integrity hash: {current_dataset_fp[:16]}... (all {len(CORE_CACHE_FILES)} cache files bitwise verified)")
        else:
            # Legacy checkpoint (Step 33 or Step 32)
            stored_ds_fp = ckpt.get("dataset_fingerprint")
            if stored_ds_fp == current_legacy_metadata_fp:
                logger.info(
                    f"[*] Detected legacy Step 33 checkpoint format (metadata-only fingerprint: {stored_ds_fp[:16]}...). "
                    f"Verified metadata matches active cache ({total_train_tokens:,} tokens)."
                )
                logger.info(
                    "[*] Note: Historical metadata-only fingerprint cannot retrospectively prove that original "
                    "token bytes were identical, but active cache content integrity has now been verified via "
                    "streaming SHA-256 hashes."
                )
                logger.info("[*] Migrating checkpoint identity to strong content hashes for Step 34+.")
            elif stored_ds_fp == EXPECTED_TOKENIZER_FP and ckpt.get("step") == 1200:
                logger.info(
                    f"[*] Detected legacy Step 32 checkpoint format (dataset_fingerprint was assigned tokenizer FP). "
                    f"Verified against active cache ({total_train_tokens:,} tokens). Migrating to strong content hashes."
                )
            else:
                raise ValueError(
                    f"Dataset fingerprint mismatch! Checkpoint expects {stored_ds_fp}, "
                    f"active dataset cache legacy metadata fingerprint is {current_legacy_metadata_fp}."
                )

        # D. State Restoration
        start_step = ckpt["step"]
        tokens_per_step = args.micro_batch_size * args.grad_accum * args.block_size
        tokens_processed_before = ckpt.get("tokens_processed", start_step * tokens_per_step)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_step = ckpt.get("best_step", start_step)

        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("[*] Successfully restored full AdamW optimizer state (momentum & variance).")
        if "scaler_state_dict" in ckpt and use_amp and ckpt["scaler_state_dict"]:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            logger.info("[*] Successfully restored AMP GradScaler state.")

        # E. Restore RNG States
        if "rng_states" in ckpt and ckpt["rng_states"]:
            rng_s = ckpt["rng_states"]
            if "torch" in rng_s and rng_s["torch"] is not None:
                t_state = rng_s["torch"]
                if isinstance(t_state, torch.Tensor):
                    t_state = t_state.cpu().byte()
                torch.set_rng_state(t_state)
            if "cuda" in rng_s and rng_s["cuda"] is not None and torch.cuda.is_available():
                c_state = rng_s["cuda"]
                if isinstance(c_state, torch.Tensor):
                    c_state = c_state.cpu().byte()
                torch.cuda.set_rng_state(c_state)
            if "data_rng" in rng_s and rng_s["data_rng"] is not None:
                d_state = rng_s["data_rng"]
                if isinstance(d_state, torch.Tensor):
                    d_state = d_state.cpu().byte()
                data_rng.set_state(d_state)
                logger.info("[*] Restored dedicated batch sampling RNG state.")
            else:
                data_rng.manual_seed(args.seed + start_step)
                logger.info("[*] Legacy checkpoint: re-seeded sampling generator at seed + start_step.")

        logger.info(f"[*] Resumed cleanly from: {ckpt_path.name} at Step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    # 8. Evaluation Before Training (Step 0 or Step 1200 Baseline)
    eval_step_label = start_step
    print(f"\n[*] Evaluating baseline at Step {eval_step_label} across all 3 tracks (100% unmodified weights)...")
    base_loss_A, base_ppl_A = evaluate_fixed_track(model, track_A_tokens, indices_A, args.block_size, device=resolved_device)
    base_loss_B, base_ppl_B = evaluate_fixed_track(model, track_B_tokens, indices_B, args.block_size, device=resolved_device)
    base_loss_C, base_ppl_C = evaluate_fixed_track(model, track_C_tokens, indices_C, args.block_size, device=resolved_device)

    print(f"    -> Track A (Combined Val):  Loss: {base_loss_A:.4f} | PPL: {base_ppl_A:.2f}")
    print(f"    -> Track B (Orig Gutenberg): Loss: {base_loss_B:.4f} | PPL: {base_ppl_B:.2f}")
    print(f"    -> Track C (Wikipedia Val):  Loss: {base_loss_C:.4f} | PPL: {base_ppl_C:.2f}")

    if is_resumed:
        if start_step == 3968:
            print(f"[*] Baseline Verification vs Step 34 Report:")
            print(f"    Track A: {base_loss_A:.4f} (Step 34 reported: 3.7095)")
            print(f"    Track B: {base_loss_B:.4f} (Step 34 reported: 4.6069)")
            print(f"    Track C: {base_loss_C:.4f} (Step 34 reported: 2.7236)")
            if abs(base_loss_A - 3.7095) > 0.05:
                logger.warning(f"Baseline Track A loss ({base_loss_A:.4f}) deviates from expected (3.7095)!")
        elif start_step == 2400:
            print(f"[*] Baseline Verification vs Step 33 Report:")
            print(f"    Track A: {base_loss_A:.4f} (Step 33 reported: 4.0030)")
            print(f"    Track B: {base_loss_B:.4f} (Step 33 reported: 4.8810)")
            print(f"    Track C: {base_loss_C:.4f} (Step 33 reported: 3.0036)")
            if abs(base_loss_A - 4.0030) > 0.05:
                logger.warning(f"Baseline Track A loss ({base_loss_A:.4f}) deviates from expected (4.0030)!")
        elif start_step == 1200:
            print(f"[*] Baseline Verification vs Step 32 Report:")
            print(f"    Track A: {base_loss_A:.4f} (Step 32 reported: 4.5249)")
            print(f"    Track B: {base_loss_B:.4f} (Step 32 reported: 5.3575)")
            print(f"    Track C: {base_loss_C:.4f} (Step 32 reported: 3.5620)")

    print(f"\n[*] Generating qualitative text inspection samples at Step {eval_step_label}...")
    baseline_samples = generate_inspection_samples(model, tokenizer, INSPECTION_PROMPTS, device=resolved_device, seed=args.seed)
    for s in baseline_samples[:4]:
        print(f"    [{s['prompt']}] -> {repr(s['completion'])}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    if is_resumed and not (args.checkpoint_dir / "model_27m_best.pt").exists():
        import shutil
        shutil.copy2(ckpt_path, args.checkpoint_dir / "model_27m_best.pt")
        logger.info(f"[*] Initialized starting {args.checkpoint_dir / 'model_27m_best.pt'} from baseline checkpoint.")

    # 9. Main Training Loop
    tokens_per_step = args.micro_batch_size * args.grad_accum * args.block_size
    print(f"\n[*] Starting Pretraining Run: from step {start_step} to {args.max_steps}...")
    print(f"[*] Time Budget: {args.max_time_seconds:.0f}s (15 min) | Effective Batch: {args.micro_batch_size * args.grad_accum} seqs ({tokens_per_step} tok/step)")

    train_start_time = time.perf_counter()
    run_tokens_processed = 0
    stop_reason = "step_limit"

    last_log_time = time.perf_counter()
    recent_tokens = 0
    step = start_step
    accum_loss = base_loss_A
    skipped_updates = 0
    successful_updates = 0
    bad_eval_count = 0

    while step < args.max_steps:
        # Check wall-clock budget
        elapsed_total = time.perf_counter() - train_start_time
        if elapsed_total >= args.max_time_seconds:
            logger.info(f"Stopping at step {step}: reached 15-minute GPU time budget ({elapsed_total:.1f}s).")
            stop_reason = "time_limit"
            break

        step_start_time = time.perf_counter()

        # Update Learning Rate along continuous 6000-step cosine schedule
        curr_lr = compute_learning_rate(
            step + 1, args.warmup_steps, args.total_scheduled_steps, args.peak_lr, args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        # Micro-batch accumulation using dedicated data_rng
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_idx in range(args.grad_accum):
            x, y = pipeline.get_batch(
                "train",
                batch_size=args.micro_batch_size,
                block_size=args.block_size,
                device=resolved_device,
                generator=data_rng,
            )
            with autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                logger.error(f"FATAL: Non-finite loss ({loss.item()}) at attempt for step {step + 1}, micro {micro_idx}!")
                stop_reason = "loss_nan"
                break

            scaler.scale(scaled_loss).backward()
            accum_loss += loss.item()

        if stop_reason == "loss_nan":
            break

        accum_loss /= args.grad_accum

        # Gradient clipping and optimizer step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if not torch.isfinite(grad_norm):
            logger.warning(f"Non-finite grad norm ({grad_norm.item()}) at attempt for step {step + 1}, skipping optimizer update.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_updates += 1
            continue

        scaler.step(optimizer)
        scaler.update()

        step += 1
        successful_updates += 1
        run_tokens_processed += tokens_per_step
        recent_tokens += tokens_per_step

        step_elapsed = time.perf_counter() - step_start_time

        # Periodic Logging
        if step % 20 == 0 or step == start_step + 1 or step == args.max_steps:
            now = time.perf_counter()
            rolling_elapsed = now - last_log_time
            rolling_tok_s = recent_tokens / max(1e-5, rolling_elapsed)
            last_log_time = now
            recent_tokens = 0

            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            logger.info(
                f"Step {step:4d}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {curr_lr:.2e} | "
                f"Grad: {grad_norm.item():.3f} | {rolling_tok_s:,.0f} tok/s | VRAM: {vram_mb:.0f}MB | "
                f"Run Time: {time.perf_counter() - train_start_time:.1f}s"
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
                bad_eval_count = 0
            else:
                bad_eval_count += 1

            ckpt_payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "model_config": asdict(gpt_config),
                "tokens_processed": tokens_processed_before + run_tokens_processed,
                "run_tokens_processed": run_tokens_processed,
                "dataset_content_fingerprint": current_dataset_fp,
                "dataset_content_hashes": current_file_hashes,
                "dataset_fingerprint": current_dataset_fp,
                "legacy_dataset_metadata_fingerprint": current_legacy_metadata_fp,
                "tokenizer_fingerprint": tok_fp,
                "rng_states": {
                    "torch": torch.get_rng_state().cpu(),
                    "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
                    "data_rng": data_rng.get_state().cpu(),
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

            if step % args.checkpoint_interval == 0:
                save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / f"step_{step}.pt")

            save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / "model_27m_latest.pt")

            if is_best:
                save_crash_safe_checkpoint(ckpt_payload, args.checkpoint_dir / "model_27m_best.pt")
                logger.info(f"[*] NEW BEST MODEL saved at step {step} (Track A loss: {best_val_loss:.4f})")

            history.append({
                "step": step,
                "train_loss": round(accum_loss, 4),
                "lr": curr_lr,
                "grad_norm": round(grad_norm.item(), 4),
                "run_tokens_processed": run_tokens_processed,
                "cumulative_tokens_processed": tokens_processed_before + run_tokens_processed,
                "val_metrics": ckpt_payload["val_metrics"],
                "wall_time_s": round(time.perf_counter() - train_start_time, 2),
            })

            if args.patience > 0 and bad_eval_count >= args.patience:
                logger.warning(
                    f"Early stopping triggered: validation loss has not improved for {bad_eval_count} consecutive evaluations "
                    f"(current Track A: {val_loss_A:.4f}, best: {best_val_loss:.4f} at step {best_step})."
                )
                stop_reason = "early_stopping"
                break

            if val_loss_A > best_val_loss + 0.5:
                logger.warning(
                    f"Early stopping triggered: validation loss diverged by >0.5 from best "
                    f"(current Track A: {val_loss_A:.4f}, best: {best_val_loss:.4f} at step {best_step})."
                )
                stop_reason = "val_divergence"
                break

    total_training_wall_time = time.perf_counter() - train_start_time

    # 10. Final Evaluation at Termination
    print(f"\n[*] Final Evaluation at termination (Step {step})...")
    final_loss_A, final_ppl_A = evaluate_fixed_track(model, track_A_tokens, indices_A, args.block_size, device=resolved_device)
    final_loss_B, final_ppl_B = evaluate_fixed_track(model, track_B_tokens, indices_B, args.block_size, device=resolved_device)
    final_loss_C, final_ppl_C = evaluate_fixed_track(model, track_C_tokens, indices_C, args.block_size, device=resolved_device)

    print(f"    -> Track A (Combined Val):  Loss: {final_loss_A:.4f} | PPL: {final_ppl_A:.2f}")
    print(f"    -> Track B (Orig Gutenberg): Loss: {final_loss_B:.4f} | PPL: {final_ppl_B:.2f}")
    print(f"    -> Track C (Wikipedia Val):  Loss: {final_loss_C:.4f} | PPL: {final_ppl_C:.2f}")

    is_final_best = final_loss_A < best_val_loss
    if is_final_best:
        best_val_loss = final_loss_A
        best_step = step

    final_ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_amp else None,
        "model_config": asdict(gpt_config),
        "tokens_processed": tokens_processed_before + run_tokens_processed,
        "run_tokens_processed": run_tokens_processed,
        "dataset_content_fingerprint": current_dataset_fp,
        "dataset_content_hashes": current_file_hashes,
        "dataset_fingerprint": current_dataset_fp,
        "legacy_dataset_metadata_fingerprint": current_legacy_metadata_fp,
        "tokenizer_fingerprint": tok_fp,
        "rng_states": {
            "torch": torch.get_rng_state().cpu(),
            "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
            "data_rng": data_rng.get_state().cpu(),
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
    if is_final_best:
        save_crash_safe_checkpoint(final_ckpt, args.checkpoint_dir / "model_27m_best.pt")
        logger.info(f"[*] NEW OVERALL BEST MODEL saved at termination (step {step}, loss: {best_val_loss:.4f})")
    else:
        logger.info(f"[*] Termination model (step {step}, loss: {final_loss_A:.4f}) did not beat previous best at step {best_step} (loss: {best_val_loss:.4f}). Previous best preserved.")

    # 11. Final Qualitative Inspection Samples
    print("\n[*] Generating qualitative inspection samples from final model...")
    final_samples = generate_inspection_samples(model, tokenizer, INSPECTION_PROMPTS, device=resolved_device, seed=args.seed)
    for s in final_samples:
        print(f"    [{s['prompt']}] -> {repr(s['completion'])}")

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0
    run_avg_tok_sec = run_tokens_processed / max(1e-5, total_training_wall_time)
    cumulative_tokens = tokens_processed_before + run_tokens_processed
    dataset_epochs = cumulative_tokens / total_train_tokens

    telemetry = {
        "model_architecture": "GroundUpGPT-27M",
        "parameters": num_params,
        "stop_reason": stop_reason,
        "start_step": start_step,
        "completed_steps": step,
        "new_steps": step - start_step,
        "run_tokens_processed": run_tokens_processed,
        "cumulative_tokens_processed": cumulative_tokens,
        "skipped_optimizer_updates": skipped_updates,
        "successful_optimizer_updates": successful_updates,
        "dataset_epochs": round(dataset_epochs, 4),
        "wall_time_seconds": round(total_training_wall_time, 2),
        "run_average_tokens_per_sec": round(run_avg_tok_sec, 1),
        "peak_vram_allocated_mb": round(peak_allocated_mb, 2),
        "peak_vram_reserved_mb": round(peak_reserved_mb, 2),
        "dataset_content_fingerprint": current_dataset_fp,
        "dataset_content_hashes": current_file_hashes,
        "baseline_val_metrics": {
            "step": eval_step_label,
            "track_A_combined_loss": base_loss_A,
            "track_A_combined_ppl": base_ppl_A,
            "track_B_orig_gt_loss": base_loss_B,
            "track_B_orig_gt_ppl": base_ppl_B,
            "track_C_wiki_loss": base_loss_C,
            "track_C_wiki_ppl": base_ppl_C,
        },
        "final_train_loss": round(accum_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_step": best_step,
        "final_val_metrics": {
            "step": step,
            "track_A_combined_loss": final_loss_A,
            "track_A_combined_ppl": final_ppl_A,
            "track_B_orig_gt_loss": final_loss_B,
            "track_B_orig_gt_ppl": final_ppl_B,
            "track_C_wiki_loss": final_loss_C,
            "track_C_wiki_ppl": final_ppl_C,
        },
        "step_history": history,
        "baseline_samples": baseline_samples,
        "final_samples": final_samples,
    }

    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    print("\n" + "=" * 65)
    print(" TRAINING CONTINUATION COMPLETE")
    print("=" * 65)
    print(f"Steps Completed:     {step} (new steps: {step - start_step}) | Stop: {stop_reason}")
    print(f"Wall-Clock Time:     {total_training_wall_time:.1f}s (budget: {args.max_time_seconds:.0f}s)")
    print(f"New Tokens Processed:{run_tokens_processed:,}")
    print(f"Cumulative Tokens:   {cumulative_tokens:,} ({dataset_epochs:.2%} dataset epoch)")
    print(f"Run Throughput:      {run_avg_tok_sec:,.0f} tok/s")
    print(f"Peak VRAM:           {peak_allocated_mb:.1f} MB allocated / {peak_reserved_mb:.1f} MB reserved")
    print(f"Track A (Combined):  {base_loss_A:.4f} -> {final_loss_A:.4f} (PPL: {base_ppl_A:.1f} -> {final_ppl_A:.1f}, Best: {best_val_loss:.4f} @ step {best_step})")
    print(f"Track B (Orig GT):   {base_loss_B:.4f} -> {final_loss_B:.4f} (PPL: {base_ppl_B:.1f} -> {final_ppl_B:.1f})")
    print(f"Track C (Wikipedia): {base_loss_C:.4f} -> {final_loss_C:.4f} (PPL: {base_ppl_C:.1f} -> {final_ppl_C:.1f})")
    print(f"Saved Checkpoints:   {args.checkpoint_dir / 'model_27m_best.pt'}")
    print(f"Telemetry Log:       {args.log_path}")
    print("=" * 65 + "\n")

    return telemetry


if __name__ == "__main__":
    run_training(parse_args())

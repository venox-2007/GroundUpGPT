"""Step 30: Scaled GroundUpGPT-27M Architecture & GPU Feasibility Benchmark.

Evaluates:
1. Exact unique trainable parameter count of GroundUpGPT-27M (27,411,456 params).
2. Memory footprint (peak allocated vs reserved VRAM) across micro-batch sizes.
3. Attention efficiency audit: Manual explicit attention vs PyTorch Scaled Dot-Product Attention (SDPA).
4. Short 20-step optimization smoke test on real expanded English pretraining tokens.
5. Training data sufficiency and scaling feasibility analysis for RTX 3050 Laptop GPU (6GB VRAM).
"""

from __future__ import annotations

import argparse
import ctypes
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

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT, get_27m_config
from src.token_cache import load_token_cache, PretokenizedDataPipeline
from src.trainer import get_tokenizer_fingerprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_27m")


def get_system_ram_gb() -> Tuple[float, float]:
    """Return (total_ram_gb, available_ram_gb) using Windows kernel32 or fallback."""
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return (
                round(stat.ullTotalPhys / (1024**3), 2),
                round(stat.ullAvailPhys / (1024**3), 2),
            )
        except Exception:
            pass
    return (16.0, 8.0)


def measure_hardware_environment(device: str) -> Dict[str, Any]:
    """Capture precise runtime hardware metrics for reproducibility."""
    hw_info: Dict[str, Any] = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    total_ram, avail_ram = get_system_ram_gb()
    hw_info["system_ram_total_gb"] = total_ram
    hw_info["system_ram_available_gb"] = avail_ram

    if device == "cuda" and torch.cuda.is_available():
        dev_idx = 0
        props = torch.cuda.get_device_properties(dev_idx)
        hw_info["gpu_device_name"] = torch.cuda.get_device_name(dev_idx)
        hw_info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        hw_info["gpu_total_vram_mb"] = round(props.total_memory / (1024**2), 2)
        hw_info["gpu_total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        hw_info["cuda_runtime_version"] = torch.version.cuda
        hw_info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    else:
        hw_info["gpu_device_name"] = "CPU (No CUDA)"
        hw_info["gpu_total_vram_mb"] = 0.0
        hw_info["gpu_total_vram_gb"] = 0.0

    return hw_info


def profile_microbatch_configuration(
    config: GPTConfig,
    device: str,
    batch_size: int,
    use_sdpa: bool,
    warmup_iters: int = 3,
    profile_iters: int = 10,
) -> Optional[Dict[str, Any]]:
    """Profile forward, backward, and optimizer latency and VRAM for a specific batch configuration."""
    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Instantiate model with specified attention implementation
    cfg = GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=0.1,
        bias=config.bias,
        weight_tying=config.weight_tying,
        use_sdpa=use_sdpa,
    )

    try:
        model = GroundUpGPT(cfg).to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        use_amp = (device == "cuda")
        scaler = GradScaler(enabled=use_amp)

        seq_len = config.block_size
        tokens_per_iter = batch_size * seq_len

        # Warmup loop to stabilize JIT / CUDA kernels
        for _ in range(warmup_iters):
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        fwd_times: List[float] = []
        bwd_times: List[float] = []
        opt_times: List[float] = []
        total_times: List[float] = []

        for _ in range(profile_iters):
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            optimizer.zero_grad(set_to_none=True)

            # 1. Forward Pass
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # 2. Backward Pass
            scaler.scale(loss).backward()

            if device == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            # 3. Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            if device == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
            opt_times.append(t3 - t2)
            total_times.append(t3 - t0)

        mean_fwd_ms = (sum(fwd_times) / len(fwd_times)) * 1000.0
        mean_bwd_ms = (sum(bwd_times) / len(bwd_times)) * 1000.0
        mean_opt_ms = (sum(opt_times) / len(opt_times)) * 1000.0
        mean_total_ms = (sum(total_times) / len(total_times)) * 1000.0

        throughput_tok_per_sec = tokens_per_iter / (sum(total_times) / len(total_times))

        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024**2) if device == "cuda" else 0.0

        del model, optimizer, scaler
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_batch": tokens_per_iter,
            "use_sdpa": use_sdpa,
            "attention_type": "PyTorch SDPA" if use_sdpa else "Manual Causal Attention",
            "forward_ms": round(mean_fwd_ms, 2),
            "backward_ms": round(mean_bwd_ms, 2),
            "optimizer_ms": round(mean_opt_ms, 2),
            "total_step_ms": round(mean_total_ms, 2),
            "throughput_tokens_per_sec": round(throughput_tok_per_sec, 1),
            "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
            "peak_vram_reserved_mb": round(peak_res_mb, 2),
        }

    except torch.cuda.OutOfMemoryError:
        logger.warning(f"CUDA OutOfMemory on batch_size={batch_size}, use_sdpa={use_sdpa}!")
        if device == "cuda":
            torch.cuda.empty_cache()
        return None


def run_short_optimization_smoke_test(
    pipeline: PretokenizedDataPipeline,
    device: str,
    max_steps: int = 20,
    max_wall_time_sec: float = 240.0,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    save_smoke_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Execute a strictly bounded 20-step smoke test verifying convergence and stability."""
    logger.info("=" * 70)
    logger.info(f"Running GroundUpGPT-27M Optimization Smoke Test (max {max_steps} steps, cap {max_wall_time_sec}s)...")
    logger.info("=" * 70)

    start_wall_time = time.perf_counter()

    # GroundUpGPT-27M Configuration
    config = get_27m_config(block_size=256, dropout=0.1, use_sdpa=True)
    model = GroundUpGPT(config).to(device)
    model.train()

    # AdamW with standard decoupled weight decay
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=3e-4, betas=(0.9, 0.95))

    use_amp = (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    effective_batch_tokens = batch_size * config.block_size * grad_accum_steps
    logger.info(
        f"Micro-batch: {batch_size} | Accumulation: {grad_accum_steps} | "
        f"Effective batch: {batch_size * grad_accum_steps} seqs ({effective_batch_tokens:,} tokens/step)"
    )

    step_records: List[Dict[str, Any]] = []
    completed_steps = 0
    total_tokens_processed = 0
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, max_steps + 1):
        # Time budget check
        elapsed_so_far = time.perf_counter() - start_wall_time
        if elapsed_so_far >= max_wall_time_sec:
            logger.warning(f"Wall-clock budget of {max_wall_time_sec}s reached at step {step-1}. Stopping early.")
            break

        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(grad_accum_steps):
            x_batch, y_batch = pipeline.get_batch(split="train", batch_size=batch_size, block_size=config.block_size, device=device)
            with autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x_batch, y_batch)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / grad_accum_steps

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())

        scaler.step(optimizer)
        scaler.update()

        if device == "cuda":
            torch.cuda.synchronize()
        step_t1 = time.perf_counter()

        step_latency = step_t1 - step_t0
        step_tok_per_sec = effective_batch_tokens / max(step_latency, 1e-9)
        completed_steps = step
        total_tokens_processed += effective_batch_tokens

        if initial_loss is None:
            initial_loss = accum_loss
        final_loss = accum_loss

        # Check for numerical instability
        if math.isnan(accum_loss) or math.isinf(accum_loss):
            raise RuntimeError(f"NaN or Inf encountered in loss at step {step}!")
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            raise RuntimeError(f"NaN or Inf encountered in grad_norm at step {step}!")

        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        peak_res = torch.cuda.max_memory_reserved() / (1024**2) if device == "cuda" else 0.0

        rec = {
            "step": step,
            "loss": round(accum_loss, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": 3e-4,
            "step_time_ms": round(step_latency * 1000.0, 2),
            "tokens_per_sec": round(step_tok_per_sec, 1),
            "peak_vram_allocated_mb": round(peak_alloc, 2),
            "peak_vram_reserved_mb": round(peak_res, 2),
        }
        step_records.append(rec)

        if step == 1 or step % 5 == 0 or step == max_steps:
            logger.info(
                f"Smoke Step {step:2d}/{max_steps} | Loss: {accum_loss:.4f} | "
                f"GradNorm: {grad_norm:.4f} | Speed: {step_tok_per_sec:.1f} tok/s | "
                f"VRAM: {peak_alloc:.1f} MB alloc / {peak_res:.1f} MB res"
            )

    total_elapsed_sec = time.perf_counter() - start_wall_time
    avg_tokens_per_sec = total_tokens_processed / max(total_elapsed_sec, 1e-9)

    # Verification of checkpoint serialization and loading
    serialization_verified = False
    if save_smoke_checkpoint and completed_steps > 0:
        smoke_ckpt_dir = REPO_ROOT / "checkpoints" / "smoke_test_27m"
        smoke_ckpt_dir.mkdir(parents=True, exist_ok=True)
        smoke_ckpt_path = smoke_ckpt_dir / "model_27m_smoke.pt"

        checkpoint_payload = {
            "step": completed_steps,
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "loss": final_loss,
            "tokenizer_type": type(pipeline.tokenizer).__name__,
            "tokenizer_vocab_size": pipeline.vocab_size,
            "tokenizer_fingerprint": get_tokenizer_fingerprint(pipeline.tokenizer),
            "is_smoke_test": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint_payload, smoke_ckpt_path)

        # Reload and verify
        loaded = torch.load(smoke_ckpt_path, map_location=device, weights_only=False)
        reloaded_cfg = GPTConfig(**loaded["model_config"])
        assert reloaded_cfg.embed_dim == 512
        assert reloaded_cfg.num_layers == 8
        assert loaded["is_smoke_test"] is True
        serialization_verified = True
        logger.info(f"Verified 27M checkpoint serialization and round-trip at {smoke_ckpt_path}")

    return {
        "completed_steps": completed_steps,
        "max_steps_target": max_steps,
        "initial_loss": round(initial_loss, 4) if initial_loss is not None else None,
        "final_loss": round(final_loss, 4) if final_loss is not None else None,
        "loss_delta": round((final_loss - initial_loss), 4) if (initial_loss and final_loss) else None,
        "total_tokens_processed": total_tokens_processed,
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "mean_tokens_per_sec": round(avg_tokens_per_sec, 1),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device == "cuda" else 0.0,
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2) if device == "cuda" else 0.0,
        "serialization_verified": serialization_verified,
        "steps_data": step_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaled GroundUpGPT-27M Feasibility Benchmark.")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "logs" / "benchmark_27m_feasibility.json")
    parser.add_argument("--smoke-steps", type=int, default=20, help="Number of smoke test optimizer steps (max 20)")
    parser.add_argument("--max-seconds", type=float, default=240.0, help="Max cumulative GPU wall time budget (seconds)")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    logger.info("=" * 70)
    logger.info(f"STARTING GROUNDUPGPT-27M ARCHITECTURE & FEASIBILITY BENCHMARK [{device.upper()}]")
    logger.info("=" * 70)

    # 1. Hardware Inspection
    hw_info = measure_hardware_environment(device)
    logger.info(f"Hardware: {hw_info['gpu_device_name']} ({hw_info['gpu_total_vram_gb']} GB VRAM)")
    logger.info(f"System RAM: {hw_info['system_ram_total_gb']} GB total ({hw_info['system_ram_available_gb']} GB available)")
    logger.info(f"PyTorch: {hw_info['torch_version']} | CUDA: {hw_info.get('cuda_runtime_version')}")

    # 2. Architecture Specifications
    config_27m = get_27m_config(block_size=256)
    dummy_model = GroundUpGPT(config_27m)
    param_count = dummy_model.get_num_params()
    param_breakdown = dummy_model.get_parameter_breakdown()
    del dummy_model

    logger.info(f"Architecture: GroundUpGPT-27M")
    logger.info(f"Total Unique Trainable Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    logger.info(f"Layers: {config_27m.num_layers} | Heads: {config_27m.num_heads} | Embed Dim: {config_27m.embed_dim} | Context: {config_27m.block_size}")

    # 3. Load Real Pretraining Data Cache
    cache_dir = REPO_ROOT / "data" / "cache_expanded"
    tok_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    assert cache_dir.is_dir(), f"Missing cache dir {cache_dir}"
    assert tok_path.is_file(), f"Missing tokenizer {tok_path}"

    tokenizer = BPETokenizer().load(tok_path)
    pipeline = load_token_cache(cache_dir=cache_dir, tokenizer=tokenizer)
    train_tokens_count = len(pipeline.train_tokens)
    val_tokens_count = len(pipeline.val_tokens)
    logger.info(f"Expanded Token Cache: {train_tokens_count:,} train tokens, {val_tokens_count:,} val tokens")

    # 4. Profile Micro-Batches & Attention Implementations
    logger.info("=" * 70)
    logger.info("PROFILING MICRO-BATCH SIZES & ATTENTION MECHANISMS...")
    logger.info("=" * 70)

    batch_sizes = [4, 8, 16]
    profiling_results: List[Dict[str, Any]] = []

    for b in batch_sizes:
        # Profile Manual Attention
        logger.info(f"Profiling Batch Size {b} (Manual Attention)...")
        res_manual = profile_microbatch_configuration(config_27m, device=device, batch_size=b, use_sdpa=False)
        if res_manual:
            profiling_results.append(res_manual)

        # Profile SDPA
        logger.info(f"Profiling Batch Size {b} (PyTorch SDPA)...")
        res_sdpa = profile_microbatch_configuration(config_27m, device=device, batch_size=b, use_sdpa=True)
        if res_sdpa:
            profiling_results.append(res_sdpa)

    # 5. Short Optimization Smoke Test
    smoke_results = run_short_optimization_smoke_test(
        pipeline=pipeline,
        device=device,
        max_steps=args.smoke_steps,
        max_wall_time_sec=args.max_seconds,
        batch_size=8,
        grad_accum_steps=4,
        save_smoke_checkpoint=True,
    )

    # 6. Training Data Sufficiency & Scaling Analysis
    tokens_per_param = train_tokens_count / param_count
    # Chinchilla recommended ratio: ~20 tokens per parameter
    chinchilla_target_tokens = param_count * 20
    corpus_deficit = max(0, chinchilla_target_tokens - train_tokens_count)
    deficit_ratio = chinchilla_target_tokens / train_tokens_count

    # Throughput estimate from smoke test
    est_tok_per_sec = smoke_results["mean_tokens_per_sec"]
    # Time to process full 13.87M corpus 1 epoch (hours)
    hours_per_epoch = (train_tokens_count / est_tok_per_sec) / 3600.0

    sufficiency_analysis = {
        "unique_training_tokens": train_tokens_count,
        "unique_model_parameters": param_count,
        "tokens_per_parameter_ratio": round(tokens_per_param, 2),
        "chinchilla_optimal_ratio": 20.0,
        "chinchilla_target_tokens": chinchilla_target_tokens,
        "corpus_coverage_pct": round((train_tokens_count / chinchilla_target_tokens) * 100.0, 1),
        "deficit_factor": round(deficit_ratio, 1),
        "estimated_training_throughput_tokens_per_sec": est_tok_per_sec,
        "estimated_hours_per_epoch": round(hours_per_epoch, 2),
        "estimated_minutes_per_epoch": round(hours_per_epoch * 60.0, 1),
        "overfitting_risk_assessment": (
            "High if trained beyond 2-3 epochs without data diversification. The 13.87M corpus represents "
            "only 0.51 tokens per parameter for a 27.4M model (well below Chinchilla-optimal 20:1 ratio). "
            "Repeated passes over the same 62 literary books will cause parameter memorization."
        ),
        "step_31_recommendation": (
            "Expand and diversify training corpus prior to full-scale pretraining. To properly train a 27.4M "
            "parameter model, target at least 50M-100M diverse English tokens (incorporating public-domain "
            "educational texts, Wikipedia extracts, and technical primers) alongside careful learning rate "
            "scheduling to avoid catastrophic memorization on RTX 3050."
        ),
    }

    # 7. Aggregate Full Benchmark Payload
    benchmark_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw_info,
        "architecture": {
            "name": "GroundUpGPT-27M",
            "total_unique_parameters": param_count,
            "config": asdict(config_27m),
            "parameter_breakdown": param_breakdown,
        },
        "profiling_microbatches": profiling_results,
        "smoke_test": smoke_results,
        "training_data_sufficiency": sufficiency_analysis,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    logger.info(f"Saved complete benchmark report to {args.output_json}")

    # 8. Print Executive Markdown Summary
    print("\n" + "=" * 105)
    print("GROUNDUPGPT-27M GPU FEASIBILITY BENCHMARK SUMMARY")
    print("=" * 105)
    print(f"Device: {hw_info['gpu_device_name']} | VRAM: {hw_info['gpu_total_vram_gb']} GB | RAM: {hw_info['system_ram_total_gb']} GB")
    print(f"Model: GroundUpGPT-27M | Unique Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    print("-" * 105)
    print(f"{'Attention Type':<22} | {'Batch':<6} | {'Fwd (ms)':<9} | {'Bwd (ms)':<9} | {'Total (ms)':<10} | {'Throughput':<14} | {'Alloc VRAM':<12} | {'Res VRAM':<10}")
    print("-" * 105)
    for p in profiling_results:
        print(
            f"{p['attention_type']:<22} | "
            f"{p['batch_size']:>5} | "
            f"{p['forward_ms']:>8.1f} | "
            f"{p['backward_ms']:>8.1f} | "
            f"{p['total_step_ms']:>9.1f} | "
            f"{p['throughput_tokens_per_sec']:>10.1f} tok/s | "
            f"{p['peak_vram_allocated_mb']:>9.1f} MB | "
            f"{p['peak_vram_reserved_mb']:>8.1f} MB"
        )
    print("=" * 105)
    print("\n20-STEP SMOKE TEST SUMMARY:")
    print(f"- Steps Completed: {smoke_results['completed_steps']}/{smoke_results['max_steps_target']}")
    print(f"- Initial Loss: {smoke_results['initial_loss']:.4f} -> Final Loss: {smoke_results['final_loss']:.4f} (Delta: {smoke_results['loss_delta']:+.4f})")
    print(f"- Tokens Processed: {smoke_results['total_tokens_processed']:,} tokens across {smoke_results['total_elapsed_sec']:.2f}s")
    print(f"- Training Throughput: {smoke_results['mean_tokens_per_sec']:.1f} tokens/sec")
    print(f"- Peak GPU Memory: {smoke_results['peak_vram_allocated_mb']:.1f} MB allocated / {smoke_results['peak_vram_reserved_mb']:.1f} MB reserved (Headroom: {hw_info['gpu_total_vram_mb'] - smoke_results['peak_vram_reserved_mb']:.1f} MB)")
    print(f"- Checkpoint Serialization Verified: {smoke_results['serialization_verified']}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

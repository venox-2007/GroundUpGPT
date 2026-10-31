"""Model evaluation, fixed benchmarking, and comparative analysis tool for GroundUpGPT.

Features:
- Fixed, deterministic validation benchmark on held-out Project Gutenberg corpus.
- Frozen token windows (seed 1337) guaranteeing identical evaluation positions across checkpoints.
- Strict metadata validation (tokenizer fingerprint and vocabulary size verification).
- Checkpoint evaluation for arbitrary step checkpoints and multi-checkpoint comparisons.
- Generation throughput benchmarking and reproducible prompt generation with fixed seeds.
- Clear distinctions between tokenizers (noting perplexity is not directly comparable across different tokenizers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from src.generation import TextGenerator, GenerationConfig

EVAL_PROMPTS = [
    "Once upon a time",
    "The young man walked into",
    "She looked at the",
    "The captain of the ship",
]


def compute_file_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str = "cpu",
) -> tuple[GroundUpGPT, BPETokenizer, dict[str, Any]]:
    """Load model and tokenizer strictly from local artifact files, verifying compatibility."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = GPTConfig(**checkpoint["model_config"])
    model = GroundUpGPT(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.load(tokenizer_path)

    # Validate metadata compatibility
    ckpt_fp = checkpoint.get("tokenizer_fingerprint")
    curr_fp = tokenizer.get_fingerprint()
    if ckpt_fp is not None and ckpt_fp != curr_fp:
        raise ValueError(
            f"Tokenizer fingerprint mismatch: checkpoint requires {ckpt_fp[:16]}..., "
            f"loaded tokenizer has {curr_fp[:16]}..."
        )

    ckpt_vocab = checkpoint.get("tokenizer_vocab_size")
    if ckpt_vocab is not None and ckpt_vocab != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary size mismatch: checkpoint has {ckpt_vocab}, "
            f"loaded tokenizer has {tokenizer.vocab_size}."
        )

    return model, tokenizer, checkpoint


def compute_fixed_validation_benchmark(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    num_eval_windows: int = 64,
    seed: int = 1337,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute cross-entropy loss and perplexity on deterministic frozen validation positions.

    Args:
        model: Loaded GroundUpGPT instance.
        tokens: 1D torch.Tensor of validation token IDs.
        block_size: Model sequence context length.
        batch_size: Micro-batch size for evaluation.
        num_eval_windows: Total number of fixed context windows to evaluate.
        seed: Fixed random seed for generating frozen position offsets.
        device: Execution device ('cuda' or 'cpu').

    Returns:
        dict: Exact evaluation statistics including loss and perplexity.
    """
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Token tensor length ({len(tokens)}) too short for block_size {block_size}")

    g = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - block_size - 1

    # Pre-generate frozen start positions ensuring 100% identical positions across runs
    fixed_starts = torch.randint(0, max_start + 1, (num_eval_windows,), generator=g).tolist()

    losses: list[float] = []
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    with torch.inference_mode():
        for i in range(0, num_eval_windows, batch_size):
            batch_slice = fixed_starts[i : i + batch_size]
            x_batch = torch.stack([tokens[pos : pos + block_size] for pos in batch_slice]).to(device)
            y_batch = torch.stack([tokens[pos + 1 : pos + block_size + 1] for pos in batch_slice]).to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x_batch, y_batch)
            else:
                _, loss = model(x_batch, y_batch)

            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())

    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan")}

    mean_loss = sum(losses) / len(losses)
    clipped_loss = min(mean_loss, 85.0)
    perplexity = math.exp(clipped_loss)

    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(perplexity, 2),
        "eval_windows": len(fixed_starts),
        "tokens_evaluated": len(fixed_starts) * block_size,
        "benchmark_seed": seed,
    }


def compute_loss_and_perplexity(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    eval_batches: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, float]:
    """Compute cross-entropy loss and perplexity on token tensor with random sampling."""
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Token tensor length ({len(tokens)}) too short for block_size {block_size}")

    g = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - block_size - 1

    losses: list[float] = []
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    with torch.inference_mode():
        for _ in range(eval_batches):
            ix = torch.randint(0, max_start + 1, (batch_size,), generator=g)
            x_batch = torch.stack([tokens[i : i + block_size] for i in ix]).to(device)
            y_batch = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix]).to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x_batch, y_batch)
            else:
                _, loss = model(x_batch, y_batch)

            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())

    if not losses:
        return {"loss": float("nan"), "perplexity": float("nan")}

    mean_loss = sum(losses) / len(losses)
    clipped_loss = min(mean_loss, 85.0)
    perplexity = math.exp(clipped_loss)

    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(perplexity, 2),
        "eval_batches": len(losses),
        "tokens_evaluated": len(losses) * batch_size * block_size,
    }


def benchmark_generation(
    generator: TextGenerator,
    prompt: str = "The captain looked out",
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure inference throughput (tokens/second) and latency."""
    # Warmup pass
    cfg_warmup = GenerationConfig(max_new_tokens=5, temperature=temperature)
    _ = generator.generate(prompt, config=cfg_warmup)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature)
    t0 = time.perf_counter()
    res = generator.generate(prompt, config=cfg)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(1e-5, time.perf_counter() - t0)

    return {
        "prompt": prompt,
        "generated_text": res.generated_text,
        "tokens_generated": res.generated_tokens,
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_second": round(res.generated_tokens / elapsed, 1),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    val_cache_path: Path,
    device: str = "auto",
    use_fixed_benchmark: bool = True,
    batch_size: int = 16,
    num_eval_windows: int = 64,
) -> dict[str, Any]:
    """Evaluate a single checkpoint on fixed validation benchmark with generation samples."""
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    model, tokenizer, checkpoint = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device=resolved_device
    )

    params = model.get_num_params()
    generator = TextGenerator(model, tokenizer, device=resolved_device)

    # 1. Dataset Identity & Integrity
    val_tokens = torch.load(val_cache_path, map_location="cpu", weights_only=True).long()
    dataset_info = {
        "dataset_name": "Project Gutenberg Held-Out Validation Split",
        "dataset_books": ["Pride and Prejudice", "The Strange Case of Dr. Jekyll and Mr. Hyde", "Frankenstein"],
        "cache_file": str(val_cache_path.relative_to(REPO_ROOT)),
        "cache_sha256": compute_file_sha256(val_cache_path),
        "total_val_tokens": len(val_tokens),
    }

    # 2. Fixed Benchmark Evaluation
    if use_fixed_benchmark:
        eval_stats = compute_fixed_validation_benchmark(
            model=model,
            tokens=val_tokens,
            block_size=model.config.block_size,
            batch_size=batch_size,
            num_eval_windows=num_eval_windows,
            seed=1337,
            device=resolved_device,
        )
    else:
        eval_stats = compute_loss_and_perplexity(
            model=model,
            tokens=val_tokens,
            block_size=model.config.block_size,
            batch_size=batch_size,
            eval_batches=num_eval_windows // batch_size,
            device=resolved_device,
        )

    # 3. Generation Throughput Benchmark
    bench = benchmark_generation(
        generator,
        prompt="The captain looked out",
        max_new_tokens=40,
        temperature=0.0,
        device=resolved_device,
    )

    # 4. Standard Prompt Completions under Fixed Seeds
    samples: dict[str, dict[str, str]] = {}
    for prompt in EVAL_PROMPTS:
        greedy_cfg = GenerationConfig(max_new_tokens=35, temperature=0.0)
        sampled_cfg = GenerationConfig(max_new_tokens=35, temperature=0.8, top_p=0.9, seed=42)
        greedy_res = generator.generate(prompt, config=greedy_cfg)
        sampled_res = generator.generate(prompt, config=sampled_cfg)
        samples[prompt] = {
            "greedy_t0": greedy_res.generated_text,
            "sampled_t08_p09": sampled_res.generated_text,
        }

    return {
        "checkpoint_file": str(checkpoint_path.relative_to(REPO_ROOT)),
        "step": checkpoint.get("step"),
        "training_val_loss": checkpoint.get("val_loss"),
        "device": resolved_device,
        "architecture": {
            "parameters": params,
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "embed_dim": model.config.embed_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "weight_tying": model.config.weight_tying,
        },
        "tokenizer": {
            "file": str(tokenizer_path.relative_to(REPO_ROOT)),
            "type": checkpoint.get("tokenizer_type", "BPETokenizer"),
            "vocab_size": tokenizer.vocab_size,
            "fingerprint": tokenizer.get_fingerprint(),
        },
        "dataset": dataset_info,
        "evaluation": eval_stats,
        "benchmark_throughput": bench,
        "samples": samples,
    }


def evaluate_profile(
    name: str,
    checkpoint_path: Path,
    tokenizer_path: Path,
    val_tokens_path: Optional[Path] = None,
    val_text_path: Optional[Path] = None,
    eval_batches: int = 50,
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Run evaluation on a model profile (supporting legacy comparison)."""
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    model, tokenizer, checkpoint = load_model_and_tokenizer(
        checkpoint_path, tokenizer_path, device=resolved_device
    )

    params = model.get_num_params()
    generator = TextGenerator(model, tokenizer, device=resolved_device)

    # 1. Validation Loss & Perplexity
    tokens: Optional[torch.Tensor] = None
    if val_tokens_path and val_tokens_path.is_file():
        tokens = torch.load(val_tokens_path, map_location="cpu", weights_only=True)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.long()
    elif val_text_path and val_text_path.is_file():
        text = val_text_path.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text)
        tokens = torch.tensor(encoded, dtype=torch.long)

    eval_stats: dict[str, Any] = {}
    if tokens is not None and len(tokens) > model.config.block_size + 1:
        adj_batch_size = min(batch_size, max(1, len(tokens) // (model.config.block_size * 2)))
        eval_stats = compute_loss_and_perplexity(
            model,
            tokens,
            block_size=model.config.block_size,
            batch_size=adj_batch_size,
            eval_batches=eval_batches,
            device=resolved_device,
        )
    else:
        eval_stats = {"loss": checkpoint.get("val_loss", float("nan")), "perplexity": float("nan")}

    # 2. Generation Throughput Benchmark
    bench = benchmark_generation(
        generator,
        prompt="The captain looked out",
        max_new_tokens=40,
        temperature=0.0,
        device=resolved_device,
    )

    # 3. Qualitative Samples
    samples: dict[str, dict[str, str]] = {}
    for prompt in EVAL_PROMPTS:
        greedy_cfg = GenerationConfig(max_new_tokens=35, temperature=0.0)
        sampled_cfg = GenerationConfig(max_new_tokens=35, temperature=0.8, top_p=0.9, seed=42)
        greedy_res = generator.generate(prompt, config=greedy_cfg)
        sampled_res = generator.generate(prompt, config=sampled_cfg)
        samples[prompt] = {
            "greedy": greedy_res.generated_text,
            "sampled_t08": sampled_res.generated_text,
        }

    return {
        "profile_name": name,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "tokenizer": str(tokenizer_path.relative_to(REPO_ROOT)),
        "device": resolved_device,
        "step": checkpoint.get("step"),
        "architecture": {
            "parameters": params,
            "vocab_size": model.config.vocab_size,
            "block_size": model.config.block_size,
            "embed_dim": model.config.embed_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "weight_tying": model.config.weight_tying,
        },
        "evaluation": eval_stats,
        "benchmark": bench,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GroundUpGPT models with fixed benchmarks and comparisons.")
    parser.add_argument(
        "--profile",
        choices=["demo", "5m", "compare"],
        default="compare",
        help="Legacy profile evaluation target: 'demo', '5m', or 'compare'",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path to evaluate")
    parser.add_argument(
        "--compare-checkpoints",
        nargs="+",
        default=None,
        help="List of checkpoint paths to benchmark and compare side-by-side",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--num-windows", type=int, default=64, help="Number of frozen windows for fixed benchmark")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--benchmark", action="store_true", help="Run deterministic frozen validation benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom path to save evaluation JSON summary",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(" GroundUpGPT — Quantitative Model Evaluation & Benchmarking")
    print("=" * 72)

    val_cache = REPO_ROOT / "data" / "cache" / "val_tokens.pt"
    m5_tok = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"

    # Mode 1: Multi-checkpoint comparison
    if args.compare_checkpoints:
        print(f"\n[*] Comparing {len(args.compare_checkpoints)} Checkpoints on Frozen Benchmark...")
        comp_results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checkpoints": []}
        for ckpt_str in args.compare_checkpoints:
            ckpt_path = Path(ckpt_str).resolve()
            res = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                tokenizer_path=m5_tok,
                val_cache_path=val_cache,
                device=args.device,
                use_fixed_benchmark=True,
                batch_size=args.batch_size,
                num_eval_windows=args.num_windows,
            )
            comp_results["checkpoints"].append(res)
            print(f"  - {ckpt_path.name}: Step #{res['step']}, Loss: {res['evaluation']['loss']}, PPL: {res['evaluation']['perplexity']}")

        out_path = (REPO_ROOT / args.output).resolve() if args.output else REPO_ROOT / "logs" / "checkpoint_comparison.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comp_results, f, indent=2)
        print(f"\n[*] Comparison saved to {out_path.relative_to(REPO_ROOT)}")
        return

    # Mode 2: Specific Checkpoint Benchmark (e.g. Step 200 Baseline)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        print(f"\n[*] Running Frozen Validation Benchmark on {ckpt_path.name}...")
        res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            tokenizer_path=m5_tok,
            val_cache_path=val_cache,
            device=args.device,
            use_fixed_benchmark=True,
            batch_size=args.batch_size,
            num_eval_windows=args.num_windows,
        )
        print(f"    Checkpoint:   {res['checkpoint_file']}")
        print(f"    Step:         #{res['step']}")
        print(f"    Parameters:   {res['architecture']['parameters']:,}")
        print(f"    Fixed Loss:   {res['evaluation']['loss']}")
        print(f"    Perplexity:   {res['evaluation']['perplexity']}")
        print(f"    Throughput:   {res['benchmark_throughput']['tokens_per_second']} tokens/sec")

        out_path = (REPO_ROOT / args.output).resolve() if args.output else REPO_ROOT / f"logs/evaluation_step{res['step']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n[*] Saved benchmark evaluation to: {out_path.relative_to(REPO_ROOT)}")
        return

    # Mode 3: Legacy Profile Comparison (Demo vs 5M)
    demo_ckpt = REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"
    demo_tok = REPO_ROOT / "checkpoints" / "demo_tokenizer.json"
    demo_fixture = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"

    m5_ckpt = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.profile in ("demo", "compare"):
        if demo_ckpt.is_file() and demo_tok.is_file():
            print(f"\n[*] Evaluating Demo Model: {demo_ckpt.name}...")
            demo_res = evaluate_profile(
                name="Step 13 Demo Model",
                checkpoint_path=demo_ckpt,
                tokenizer_path=demo_tok,
                val_text_path=demo_fixture,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                device=args.device,
            )
            results["demo_model"] = demo_res

    if args.profile in ("5m", "compare"):
        if m5_ckpt.is_file() and m5_tok.is_file():
            print(f"\n[*] Evaluating 5.05M Model: {m5_ckpt.name}...")
            m5_res = evaluate_profile(
                name="Step 16 Trained 5M Model",
                checkpoint_path=m5_ckpt,
                tokenizer_path=m5_tok,
                val_tokens_path=val_cache,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                device=args.device,
            )
            results["5m_model"] = m5_res

    if "demo_model" in results and "5m_model" in results:
        d = results["demo_model"]
        m = results["5m_model"]
        print("\n" + "=" * 72)
        print(" COMPARATIVE SUMMARY: Demo Model vs. Trained 5.05M Model")
        print("=" * 72)
        print(f"{'Metric':<28} | {'Demo Model (Step 13)':<20} | {'5.05M Model (Step 16)':<20}")
        print("-" * 72)
        print(f"{'Parameters':<28} | {d['architecture']['parameters']:<20,} | {m['architecture']['parameters']:<20,}")
        print(f"{'Vocabulary Size':<28} | {d['architecture']['vocab_size']:<20} | {m['architecture']['vocab_size']:<20}")
        print(f"{'Context Window':<28} | {d['architecture']['block_size']:<20} | {m['architecture']['block_size']:<20}")
        print(f"{'Training Steps':<28} | {str(d['step']):<20} | {str(m['step']):<20}")
        print(f"{'Validation Loss':<28} | {str(d['evaluation'].get('loss')):<20} | {str(m['evaluation'].get('loss')):<20}")
        print(f"{'Validation Perplexity':<28} | {str(d['evaluation'].get('perplexity')):<20} | {str(m['evaluation'].get('perplexity')):<20}")
        print(f"{'Throughput (tok/s)':<28} | {str(d['benchmark']['tokens_per_second']):<20} | {str(m['benchmark']['tokens_per_second']):<20}")
        print("-" * 72)
        print(" [NOTE: Perplexity across different tokenizers (Char vs BPE-4096) is NOT directly comparable!]")

    out_file = (REPO_ROOT / args.output).resolve() if args.output else REPO_ROOT / "logs" / "evaluation_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Detailed report saved to: {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

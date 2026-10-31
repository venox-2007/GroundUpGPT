"""Model evaluation and comparative benchmarking tool for GroundUpGPT.

Evaluates trained checkpoints on held-out validation data, computing:
- Validation cross-entropy loss and perplexity (exp(loss))
- Model architecture and parameter counts
- Generation speed (tokens/sec) and latency
- Qualitative prompt completions across greedy and sampling temperatures
- Side-by-side comparative analysis between models
"""

from __future__ import annotations

import argparse
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
    "The captain looked out",
    "She thought that",
    "It was a dark and",
]


def load_model_and_tokenizer(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: str = "cpu",
) -> tuple[GroundUpGPT, BPETokenizer, dict[str, Any]]:
    """Load model and tokenizer strictly from local artifact files."""
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
    return model, tokenizer, checkpoint


def compute_loss_and_perplexity(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    eval_batches: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, float]:
    """Compute cross-entropy loss and perplexity on token tensor.

    Perplexity is computed as exp(loss).
    """
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Token tensor length ({len(tokens)}) too short for block_size {block_size}")

    g = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - block_size - 1

    losses: list[float] = []
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    with torch.inference_mode():
        for _ in range(eval_batches):
            # Sample random chunk indices
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
    # Clip exponent to prevent overflow
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
    max_new_tokens: int = 50,
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
    """Run full evaluation on a model profile."""
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
        # Constrain micro-batch size if tokens tensor is small
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
    parser = argparse.ArgumentParser(description="Evaluate GroundUpGPT models and produce comparative metrics.")
    parser.add_argument(
        "--profile",
        choices=["demo", "5m", "compare"],
        default="compare",
        help="Evaluation target: 'demo', '5m', or 'compare' (default: compare)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval-batches", type=int, default=50, help="Batches for validation loss evaluation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation evaluation")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/evaluation_comparison.json",
        help="Path to save evaluation JSON summary",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(" GroundUpGPT — Quantitative Model Evaluation & Benchmarking")
    print("=" * 72)

    demo_ckpt = REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"
    demo_tok = REPO_ROOT / "checkpoints" / "demo_tokenizer.json"
    demo_fixture = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"

    m5_ckpt = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"
    m5_tok = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    m5_cache = REPO_ROOT / "data" / "cache" / "val_tokens.pt"

    results: dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Profile 1: Step 13 Demo Model
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
            print(f"    Parameters:  {demo_res['architecture']['parameters']:,}")
            print(f"    Val Loss:    {demo_res['evaluation'].get('loss', 'N/A')}")
            print(f"    Perplexity:  {demo_res['evaluation'].get('perplexity', 'N/A')}")
            print(f"    Throughput:  {demo_res['benchmark']['tokens_per_second']} tokens/sec")
        else:
            print(f"[!] Demo model artifacts not found at {demo_ckpt}.")

    # Profile 2: Step 16 Trained 5M Model
    if args.profile in ("5m", "compare"):
        if m5_ckpt.is_file() and m5_tok.is_file():
            print(f"\n[*] Evaluating 5.05M Model: {m5_ckpt.name}...")
            m5_res = evaluate_profile(
                name="Step 16 Trained 5M Model",
                checkpoint_path=m5_ckpt,
                tokenizer_path=m5_tok,
                val_tokens_path=m5_cache,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                device=args.device,
            )
            results["5m_model"] = m5_res
            print(f"    Parameters:  {m5_res['architecture']['parameters']:,}")
            print(f"    Step:        #{m5_res['step']}")
            print(f"    Val Loss:    {m5_res['evaluation'].get('loss', 'N/A')}")
            print(f"    Perplexity:  {m5_res['evaluation'].get('perplexity', 'N/A')}")
            print(f"    Throughput:  {m5_res['benchmark']['tokens_per_second']} tokens/sec")
        else:
            print(f"[!] 5M model artifacts not found at {m5_ckpt}.")

    # Print Comparative Table if both available
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
        print(f"{'Layers / Heads':<28} | {d['architecture']['num_layers']}L / {d['architecture']['num_heads']}H{' ':<13} | {m['architecture']['num_layers']}L / {m['architecture']['num_heads']}H{' ':<13}")
        print(f"{'Training Steps':<28} | {str(d['step']):<20} | {str(m['step']):<20}")
        print(f"{'Validation Loss':<28} | {str(d['evaluation'].get('loss')):<20} | {str(m['evaluation'].get('loss')):<20}")
        print(f"{'Validation Perplexity':<28} | {str(d['evaluation'].get('perplexity')):<20} | {str(m['evaluation'].get('perplexity')):<20}")
        print(f"{'Throughput (tok/s)':<28} | {str(d['benchmark']['tokens_per_second']):<20} | {str(m['benchmark']['tokens_per_second']):<20}")
        print("-" * 72)

        print("\n=== QUALITATIVE SAMPLE COMPARISON ===")
        for p in EVAL_PROMPTS:
            print(f"\nPrompt: \"{p}\"")
            print(f"  [Demo - Greedy]:   \"{d['samples'][p]['greedy'].strip()}\"")
            print(f"  [5M   - Greedy]:   \"{m['samples'][p]['greedy'].strip()}\"")
            print(f"  [5M   - Sampled]:  \"{m['samples'][p]['sampled_t08'].strip()}\"")

    # Save output JSON
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Detailed comparative evaluation report saved to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

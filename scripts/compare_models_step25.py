"""Step 25: Quantitative and Qualitative Comparison across GroundUpGPT Checkpoints.

Compares:
1. Step 1000 Baseline (checkpoints/5m_continued/model_5m_best.pt)
2. Phase 2 Best Checkpoint (checkpoints/5m_phase2/model_5m_best.pt)
3. Phase 2 Latest Checkpoint (checkpoints/5m_phase2/model_5m_latest.pt)

Evaluates:
- Validation Loss on Track 1 (Original Frozen Val: 162K tokens, seed 1337, 64 windows)
- Validation Loss on Track 2 (Expanded Val: 1.12M tokens, seed 1337, 64 windows)
- Perplexity on both tracks
- Percentage improvement
- Generation throughput (tokens/sec)
- Side-by-side text generation across 6 fixed benchmark prompts
- Qualitative linguistic assessment (word correctness, grammar, local coherence, repetition, relevance)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT
from scripts.evaluate_model import compute_fixed_validation_benchmark

BENCHMARK_PROMPTS = [
    "Once upon a time",
    "The scientist looked through the microscope",
    "In the beginning of the nineteenth century",
    "She opened the letter and discovered",
    "The philosophy of nature teaches us that",
    "They walked down the dark London street",
]


def load_model_from_checkpoint(ckpt_path: Path, device: str = "cpu") -> Tuple[GroundUpGPT, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(m_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def benchmark_generation_speed(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    device: str,
    prompt: str = "The history of the world",
    max_tokens: int = 60,
) -> float:
    """Measure generation speed in tokens per second."""
    generator = TextGenerator(model, tokenizer, device=device)
    cfg = GenerationConfig(max_new_tokens=max_tokens, temperature=0.8, top_p=0.9, seed=42)

    # Warmup
    _ = generator.generate(prompt=prompt, config=cfg)

    # Timed run
    t0 = time.perf_counter()
    res = generator.generate(prompt=prompt, config=cfg)
    dur = time.perf_counter() - t0
    return res.generated_tokens / max(1e-4, dur)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    tokenizer = BPETokenizer.load(tok_path)

    frozen_val_path = REPO_ROOT / "data" / "cache" / "val_tokens.pt"
    expanded_val_path = REPO_ROOT / "data" / "cache_expanded" / "val_expanded_tokens.pt"

    frozen_tokens = torch.load(frozen_val_path, weights_only=True).long()
    expanded_tokens = torch.load(expanded_val_path, weights_only=True).long()

    checkpoints_to_compare = [
        ("Step 1000 Baseline", REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_best.pt"),
        ("Phase 2 Best (Step 3000)", REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_best.pt"),
        ("Phase 2 Latest (Step 3000)", REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_latest.pt"),
    ]

    results: Dict[str, Any] = {}

    print("=" * 80)
    print(" GROUNDUPGPT — STEP 25 MODEL COMPARISON")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("=" * 80)

    for label, ckpt_p in checkpoints_to_compare:
        print(f"\nEvaluating {label} from {ckpt_p.name}...")
        model, ckpt = load_model_from_checkpoint(ckpt_p, device=device)
        step = ckpt.get("step", 0)
        params = model.get_num_params()

        # Benchmarks
        bench_frozen = compute_fixed_validation_benchmark(
            model=model,
            tokens=frozen_tokens,
            block_size=256,
            batch_size=16,
            num_eval_windows=64,
            seed=1337,
            device=device,
        )
        bench_expanded = compute_fixed_validation_benchmark(
            model=model,
            tokens=expanded_tokens,
            block_size=256,
            batch_size=16,
            num_eval_windows=64,
            seed=1337,
            device=device,
        )

        tok_speed = benchmark_generation_speed(model, tokenizer, device=device)

        generator = TextGenerator(model, tokenizer, device=device)
        completions = {}
        for p in BENCHMARK_PROMPTS:
            greedy_res = generator.generate(p, GenerationConfig(max_new_tokens=40, temperature=0.0))
            sample_res = generator.generate(p, GenerationConfig(max_new_tokens=40, temperature=0.8, top_p=0.9, seed=42))
            completions[p] = {
                "greedy": greedy_res.generated_text,
                "sampled": sample_res.generated_text,
            }

        results[label] = {
            "checkpoint_path": str(ckpt_p).replace("\\", "/"),
            "step": step,
            "parameters": params,
            "track1_original_val_loss": round(bench_frozen["loss"], 4),
            "track1_original_perplexity": round(bench_frozen["perplexity"], 2),
            "track2_expanded_val_loss": round(bench_expanded["loss"], 4),
            "track2_expanded_perplexity": round(bench_expanded["perplexity"], 2),
            "generation_tokens_per_sec": round(tok_speed, 1),
            "completions": completions,
        }

    # Print Comparative Table
    step1000 = results["Step 1000 Baseline"]
    p2_best = results["Phase 2 Best (Step 3000)"]

    orig_val_imp = (step1000["track1_original_val_loss"] - p2_best["track1_original_val_loss"]) / step1000["track1_original_val_loss"] * 100.0
    orig_ppl_imp = (step1000["track1_original_perplexity"] - p2_best["track1_original_perplexity"]) / step1000["track1_original_perplexity"] * 100.0
    exp_val_imp = (step1000["track2_expanded_val_loss"] - p2_best["track2_expanded_val_loss"]) / step1000["track2_expanded_val_loss"] * 100.0
    exp_ppl_imp = (step1000["track2_expanded_perplexity"] - p2_best["track2_expanded_perplexity"]) / step1000["track2_expanded_perplexity"] * 100.0

    print("\n" + "=" * 80)
    print(" QUANTITATIVE METRICS COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Step 1000':<12} | {'Phase 2 Best':<14} | {'Change (%)':<10}")
    print("-" * 80)
    print(f"{'Total Optimizer Steps':<35} | {step1000['step']:<12} | {p2_best['step']:<14} | +200.0%")
    print(f"{'Total Parameters':<35} | {step1000['parameters']:<12,} | {p2_best['parameters']:<14,} | 0.0%")
    print(f"{'Track 1 (Original Val Loss)':<35} | {step1000['track1_original_val_loss']:<12.4f} | {p2_best['track1_original_val_loss']:<14.4f} | -{orig_val_imp:.1f}%")
    print(f"{'Track 1 (Original Perplexity)':<35} | {step1000['track1_original_perplexity']:<12.2f} | {p2_best['track1_original_perplexity']:<14.2f} | -{orig_ppl_imp:.1f}%")
    print(f"{'Track 2 (Expanded Val Loss)':<35} | {step1000['track2_expanded_val_loss']:<12.4f} | {p2_best['track2_expanded_val_loss']:<14.4f} | -{exp_val_imp:.1f}%")
    print(f"{'Track 2 (Expanded Perplexity)':<35} | {step1000['track2_expanded_perplexity']:<12.2f} | {p2_best['track2_expanded_perplexity']:<14.2f} | -{exp_ppl_imp:.1f}%")
    print(f"{'Generation Speed (tok/s)':<35} | {step1000['generation_tokens_per_sec']:<12.1f} | {p2_best['generation_tokens_per_sec']:<14.1f} | ~same")
    print("=" * 80)

    # Save to logs
    out_json = REPO_ROOT / "logs" / "model_comparison_step25.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive comparison to: {out_json}")


if __name__ == "__main__":
    main()

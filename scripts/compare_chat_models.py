"""Run the 10 evaluation prompts against both 5m_sft_best and 27m_sft_best.

Evaluates:
- Real responses with identical decoding settings:
  temperature=0.8, top_k=40, top_p=0.9, max_new_tokens=64, repetition_penalty=1.15, seed=42
- Latency and tokens per second
- Peak GPU memory allocation
- Direct unedited response comparison
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, generate_sft_response, TextGenerator
from src.model import GPTConfig, GroundUpGPT

PROMPTS = [
    "Hello! How are you today?",
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?",
    "What is gravity?",
    "List three primary colors.",
    "What is 2 + 2?",
    "Can you help me?",
    "What is a mammal?",
    "Tell me a short story.",
    "Thank you!",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_5M = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"
CKPT_27M = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def evaluate_model(ckpt_path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    gen = TextGenerator(model, tokenizer, device=device)

    config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=42,
        repetition_penalty=1.15,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )

    results = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for idx, p in enumerate(PROMPTS, 1):
        t0 = time.perf_counter()
        res = generate_sft_response(gen, prompt=p, config=config)
        elapsed = time.perf_counter() - t0
        tok_per_sec = res.generated_tokens / max(elapsed, 1e-6)
        results.append({
            "idx": idx,
            "prompt": p,
            "response": res.generated_text,
            "prompt_tokens": res.prompt_tokens,
            "generated_tokens": res.generated_tokens,
            "elapsed_sec": round(elapsed, 4),
            "tokens_per_sec": round(tok_per_sec, 2),
        })

    peak_mem_mb = 0.0
    if device == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    del model, gen
    if device == "cuda":
        torch.cuda.empty_cache()

    return results, peak_mem_mb


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comparative benchmark on device: {device}")

    print("\n--- Evaluating GroundUpGPT-5M SFT Best ---")
    res_5m, mem_5m = evaluate_model(CKPT_5M, device=device)
    print(f"5M Peak GPU Memory: {mem_5m:.2f} MB")

    print("\n--- Evaluating GroundUpGPT-27M SFT Best ---")
    res_27m, mem_27m = evaluate_model(CKPT_27M, device=device)
    print(f"27M Peak GPU Memory: {mem_27m:.2f} MB")

    output_data = {
        "device": device,
        "mem_5m_mb": mem_5m,
        "mem_27m_mb": mem_27m,
        "comparisons": [],
    }

    print("\n" + "=" * 80)
    print("COMPARATIVE CHAT EVALUATION (10 PROMPTS)")
    print("=" * 80)
    for i in range(len(PROMPTS)):
        r5 = res_5m[i]
        r27 = res_27m[i]
        print(f"\n[{i+1}/10] Prompt: {r5['prompt']}")
        print(f"5M Response  ({r5['generated_tokens']} tok, {r5['tokens_per_sec']} t/s): {repr(r5['response'])}")
        print(f"27M Response ({r27['generated_tokens']} tok, {r27['tokens_per_sec']} t/s): {repr(r27['response'])}")

        output_data["comparisons"].append({
            "prompt": r5["prompt"],
            "5m": r5,
            "27m": r27,
        })

    with open(REPO_ROOT / "logs" / "chat_comparison_step38.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\nSaved report to logs/chat_comparison_step38.json")

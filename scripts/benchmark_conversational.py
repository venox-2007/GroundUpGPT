"""Reproducible Conversational Quality Benchmark and Inference Audit (Step 29).

Evaluates:
- Step 3000 Pretrained Base Model ('checkpoints/5m_phase2/model_5m_best.pt')
- Step 650 SFT Best Model ('checkpoints/5m_sft/model_5m_sft_best.pt')

Across 31 short, unseen prompts spanning 7 categories:
1. Greetings
2. Basic Factual Questions
3. Simple Classification
4. Arithmetic
5. One-Sentence Explanations
6. Context-Based Questions
7. Short Lists

Measures:
- Objective correctness on gradable tasks
- Prompt relevance
- 3-gram repetition rate
- End-of-turn delimiter stopping success
- Invalid delimiter leakage
- Empty response count
- Response token length & throughput
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.model_loader import ModelManager
from src.generation import GenerationConfig, generate_sft_response, format_sft_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 31 Unseen Prompts across 7 Categories (zero exact/near overlap with Dolly training data)
BENCHMARK_PROMPTS: List[Dict[str, Any]] = [
    # 1. Greetings (4)
    {
        "id": "G01",
        "category": "greetings",
        "prompt": "Hello! How are you today?",
        "context": None,
        "expected": "polite greeting / conversational opening",
        "eval_type": "subjective",
        "keywords": ["hello", "good", "well", "fine", "day", "help", "greet", "glad"],
    },
    {
        "id": "G02",
        "category": "greetings",
        "prompt": "Good morning! Can you help me?",
        "context": None,
        "expected": "greeting / affirmation of assistance",
        "eval_type": "subjective",
        "keywords": ["morning", "help", "yes", "can", "assist", "sure", "welcome"],
    },
    {
        "id": "G03",
        "category": "greetings",
        "prompt": "Hi, GroundUpGPT.",
        "context": None,
        "expected": "friendly greeting / acknowledgement",
        "eval_type": "subjective",
        "keywords": ["hi", "hello", "groundupgpt", "welcome", "glad", "day"],
    },
    {
        "id": "G04",
        "category": "greetings",
        "prompt": "Farewell! Until next time.",
        "context": None,
        "expected": "farewell / parting acknowledgment",
        "eval_type": "subjective",
        "keywords": ["farewell", "bye", "goodbye", "next", "time", "take", "care"],
    },

    # 2. Basic Factual Questions (5)
    {
        "id": "F01",
        "category": "factual",
        "prompt": "What is the capital of Spain?",
        "context": None,
        "expected": "Madrid",
        "eval_type": "objective",
        "target_keywords": ["madrid"],
    },
    {
        "id": "F02",
        "category": "factual",
        "prompt": "What planet is closest to the Sun?",
        "context": None,
        "expected": "Mercury",
        "eval_type": "objective",
        "target_keywords": ["mercury"],
    },
    {
        "id": "F03",
        "category": "factual",
        "prompt": "What is the largest ocean on Earth?",
        "context": None,
        "expected": "Pacific Ocean",
        "eval_type": "objective",
        "target_keywords": ["pacific"],
    },
    {
        "id": "F04",
        "category": "factual",
        "prompt": "How many days are in a week?",
        "context": None,
        "expected": "7 / seven",
        "eval_type": "objective",
        "target_keywords": ["7", "seven"],
    },
    {
        "id": "F05",
        "category": "factual",
        "prompt": "What continent is Egypt located on?",
        "context": None,
        "expected": "Africa",
        "eval_type": "objective",
        "target_keywords": ["africa"],
    },

    # 3. Simple Classification (5)
    {
        "id": "C01",
        "category": "classification",
        "prompt": "Is a carrot a vegetable or a musical instrument?",
        "context": None,
        "expected": "vegetable",
        "eval_type": "objective",
        "target_keywords": ["vegetable"],
    },
    {
        "id": "C02",
        "category": "classification",
        "prompt": "Classify this statement as positive or negative: I love this sunny morning.",
        "context": None,
        "expected": "positive",
        "eval_type": "objective",
        "target_keywords": ["positive"],
    },
    {
        "id": "C03",
        "category": "classification",
        "prompt": "Which of these is an animal: rock, dog, cloud?",
        "context": None,
        "expected": "dog",
        "eval_type": "objective",
        "target_keywords": ["dog"],
    },
    {
        "id": "C04",
        "category": "classification",
        "prompt": "Is ice hot or cold?",
        "context": None,
        "expected": "cold",
        "eval_type": "objective",
        "target_keywords": ["cold"],
    },
    {
        "id": "C05",
        "category": "classification",
        "prompt": "Classify this vehicle as land, air, or sea: helicopter.",
        "context": None,
        "expected": "air",
        "eval_type": "objective",
        "target_keywords": ["air"],
    },

    # 4. Arithmetic (4)
    {
        "id": "A01",
        "category": "arithmetic",
        "prompt": "What is 2 + 2?",
        "context": None,
        "expected": "4 / four",
        "eval_type": "objective",
        "target_keywords": ["4", "four"],
    },
    {
        "id": "A02",
        "category": "arithmetic",
        "prompt": "What is 10 minus 3?",
        "context": None,
        "expected": "7 / seven",
        "eval_type": "objective",
        "target_keywords": ["7", "seven"],
    },
    {
        "id": "A03",
        "category": "arithmetic",
        "prompt": "What is 5 multiplied by 2?",
        "context": None,
        "expected": "10 / ten",
        "eval_type": "objective",
        "target_keywords": ["10", "ten"],
    },
    {
        "id": "A04",
        "category": "arithmetic",
        "prompt": "What is half of 20?",
        "context": None,
        "expected": "10 / ten",
        "eval_type": "objective",
        "target_keywords": ["10", "ten"],
    },

    # 5. One-Sentence Explanations (4)
    {
        "id": "E01",
        "category": "explanation",
        "prompt": "Explain what gravity is in one sentence.",
        "context": None,
        "expected": "force attracting physical bodies / mass",
        "eval_type": "subjective",
        "keywords": ["force", "attract", "mass", "earth", "pull", "object"],
    },
    {
        "id": "E02",
        "category": "explanation",
        "prompt": "What is water made of?",
        "context": None,
        "expected": "hydrogen and oxygen / H2O",
        "eval_type": "objective",
        "target_keywords": ["hydrogen", "oxygen", "h2o"],
    },
    {
        "id": "E03",
        "category": "explanation",
        "prompt": "What is the function of the human heart?",
        "context": None,
        "expected": "pumps blood through the body",
        "eval_type": "objective",
        "target_keywords": ["blood", "pump"],
    },
    {
        "id": "E04",
        "category": "explanation",
        "prompt": "Explain why the sky looks blue.",
        "context": None,
        "expected": "scattering of sunlight in the atmosphere",
        "eval_type": "subjective",
        "keywords": ["scatter", "light", "atmosphere", "blue", "sun"],
    },

    # 6. Context-Based Questions (5)
    {
        "id": "X01",
        "category": "context_qa",
        "prompt": "What time does the library close on weekdays?",
        "context": "The library opens at 9 AM and closes at 8 PM on weekdays.",
        "expected": "8 PM / eight",
        "eval_type": "objective",
        "target_keywords": ["8", "8 pm", "8pm", "eight"],
    },
    {
        "id": "X02",
        "category": "context_qa",
        "prompt": "How many cats does Alice have?",
        "context": "Alice has two cats named Luna and Oliver.",
        "expected": "2 / two",
        "eval_type": "objective",
        "target_keywords": ["two", "2"],
    },
    {
        "id": "X03",
        "category": "context_qa",
        "prompt": "Which university did Dr. Smith attend?",
        "context": "Dr. Smith graduated from Stanford University in 2015.",
        "expected": "Stanford",
        "eval_type": "objective",
        "target_keywords": ["stanford"],
    },
    {
        "id": "X04",
        "category": "context_qa",
        "prompt": "How many eggs are needed?",
        "context": "The recipe requires 3 eggs, 2 cups of flour, and 1 cup of sugar.",
        "expected": "3 / three",
        "eval_type": "objective",
        "target_keywords": ["3", "three"],
    },
    {
        "id": "X05",
        "category": "context_qa",
        "prompt": "Where was the package delivered?",
        "context": "The package was delivered to the front porch on Tuesday morning.",
        "expected": "front porch / porch",
        "eval_type": "objective",
        "target_keywords": ["porch", "front porch"],
    },

    # 7. Short Lists (4)
    {
        "id": "L01",
        "category": "lists",
        "prompt": "List three primary colors.",
        "context": None,
        "expected": "red, blue, yellow",
        "eval_type": "objective",
        "target_keywords": ["red", "blue", "yellow"],
    },
    {
        "id": "L02",
        "category": "lists",
        "prompt": "Name two seasons of the year.",
        "context": None,
        "expected": "spring, summer, autumn/fall, winter",
        "eval_type": "objective",
        "target_keywords": ["spring", "summer", "autumn", "fall", "winter"],
    },
    {
        "id": "L03",
        "category": "lists",
        "prompt": "List three common domestic pets.",
        "context": None,
        "expected": "dog, cat, fish, bird",
        "eval_type": "objective",
        "target_keywords": ["dog", "cat", "fish", "bird", "rabbit", "hamster"],
    },
    {
        "id": "L04",
        "category": "lists",
        "prompt": "Name two continents on Earth.",
        "context": None,
        "expected": "Asia, Africa, Europe, North America, South America, Antarctica, Australia",
        "eval_type": "objective",
        "target_keywords": ["asia", "africa", "europe", "america", "antarctica", "australia"],
    },
]


def compute_3gram_repetition_rate(text: str) -> float:
    """Compute ratio of duplicate 3-grams to total 3-grams in the text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return 0.0
    unique_count = len(set(trigrams))
    total_count = len(trigrams)
    return round(1.0 - (unique_count / total_count), 4)


def check_objective_correctness(response: str, target_keywords: List[str]) -> bool:
    """Check if generated response contains any expected target keywords for objective tasks."""
    resp_lower = response.lower()
    for kw in target_keywords:
        # Match as whole word or boundary
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, resp_lower):
            return True
    return False


def check_prompt_relevance(prompt: str, response: str, category_keywords: Optional[List[str]] = None) -> bool:
    """Check if response exhibits topic relevance to prompt content or domain keywords."""
    resp_words = set(re.findall(r"\b\w{3,}\b", response.lower()))
    prompt_words = set(re.findall(r"\b\w{3,}\b", prompt.lower())) - {"what", "which", "where", "how", "the", "and", "are", "for", "one"}
    
    # Direct word overlap with prompt
    if len(resp_words & prompt_words) > 0:
        return True
    
    # Or matches category/domain expected vocabulary
    if category_keywords:
        for kw in category_keywords:
            if kw.lower() in resp_words:
                return True
    return False


def check_invalid_delimiter_leak(text: str) -> bool:
    """Return True if any raw delimiter token (<|user|>, <|assistant|>, <|endoftext|>, or dangling <|) leaks."""
    delims = ["<|user|>", "<|assistant|>", "<|endoftext|>", "<|"]
    for d in delims:
        if d in text:
            return True
    return False


def run_benchmark_for_model(
    manager: ModelManager,
    is_sft: bool,
    config: GenerationConfig,
    config_label: str,
) -> Dict[str, Any]:
    """Execute all 31 prompts through the given model manager and compute metrics."""
    results: List[Dict[str, Any]] = []
    
    total_latency = 0.0
    total_tokens = 0
    objective_correct = 0
    objective_total = 0
    relevant_count = 0
    repetition_rates: List[float] = []
    eot_stopped_count = 0
    invalid_delim_count = 0
    empty_count = 0
    lengths: List[int] = []

    for item in BENCHMARK_PROMPTS:
        p_id = item["id"]
        category = item["category"]
        prompt_text = item["prompt"]
        context = item.get("context")
        eval_type = item["eval_type"]
        target_kws = item.get("target_keywords", [])
        expected_desc = item["expected"]

        t_start = time.perf_counter()
        if is_sft:
            res = generate_sft_response(
                generator=manager.generator,
                prompt=prompt_text,
                context=context,
                config=config,
            )
        else:
            raw_prompt = f"Context: {context}\n\nPrompt: {prompt_text}\nAnswer: " if context else f"{prompt_text} "
            res = manager.generator.generate(prompt=raw_prompt, config=config)
        elapsed = time.perf_counter() - t_start

        gen_text = res.generated_text.strip()
        gen_tokens = res.generated_tokens
        total_latency += elapsed
        total_tokens += gen_tokens
        lengths.append(gen_tokens)

        # Repetition
        rep_rate = compute_3gram_repetition_rate(gen_text)
        repetition_rates.append(rep_rate)

        # Delimiter leakage
        has_delim_leak = check_invalid_delimiter_leak(gen_text)
        if has_delim_leak:
            invalid_delim_count += 1

        # Empty response
        is_empty = len(gen_text) == 0
        if is_empty:
            empty_count += 1

        # End of turn detection (stopped before max_new_tokens)
        stopped_early = (gen_tokens < config.max_new_tokens)
        if stopped_early:
            eot_stopped_count += 1

        # Correctness & Relevance
        is_correct = False
        if eval_type == "objective":
            objective_total += 1
            is_correct = check_objective_correctness(gen_text, target_kws)
            if is_correct:
                objective_correct += 1

        is_relevant = check_prompt_relevance(prompt_text, gen_text, item.get("keywords") or target_kws)
        if is_relevant:
            relevant_count += 1

        results.append({
            "id": p_id,
            "category": category,
            "prompt": prompt_text,
            "context": context,
            "expected": expected_desc,
            "eval_type": eval_type,
            "generated_text": gen_text,
            "generated_tokens": gen_tokens,
            "elapsed_sec": round(elapsed, 4),
            "tokens_per_sec": round(gen_tokens / max(elapsed, 1e-9), 2),
            "is_correct": is_correct if eval_type == "objective" else None,
            "is_relevant": is_relevant,
            "repetition_rate": rep_rate,
            "stopped_early": stopped_early,
            "has_delimiter_leak": has_delim_leak,
        })

    avg_tokens_per_sec = total_tokens / max(total_latency, 1e-9)
    avg_length = sum(lengths) / max(len(lengths), 1)
    avg_repetition = sum(repetition_rates) / max(len(repetition_rates), 1)
    obj_accuracy = (objective_correct / max(objective_total, 1)) * 100.0
    relevance_rate = (relevant_count / len(BENCHMARK_PROMPTS)) * 100.0
    eot_rate = (eot_stopped_count / len(BENCHMARK_PROMPTS)) * 100.0
    delim_leak_rate = (invalid_delim_count / len(BENCHMARK_PROMPTS)) * 100.0

    summary = {
        "config_label": config_label,
        "is_sft": is_sft,
        "total_prompts": len(BENCHMARK_PROMPTS),
        "objective_prompts_count": objective_total,
        "objective_correct_count": objective_correct,
        "objective_accuracy_pct": round(obj_accuracy, 2),
        "relevance_rate_pct": round(relevance_rate, 2),
        "mean_3gram_repetition_rate": round(avg_repetition, 4),
        "eot_stopping_rate_pct": round(eot_rate, 2),
        "delimiter_leakage_rate_pct": round(delim_leak_rate, 2),
        "empty_responses_count": empty_count,
        "mean_generated_tokens": round(avg_length, 2),
        "total_wall_clock_sec": round(total_latency, 4),
        "tokens_per_sec": round(avg_tokens_per_sec, 2),
        "prompt_results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Conversational Benchmark & Inference Audit.")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "logs" / "conversational_benchmark_step29.json")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    logger.info(f"Running conversational quality benchmark on device: {device.upper()}")

    tok_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    base_ckpt = REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_best.pt"
    sft_ckpt = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"

    assert tok_path.is_file(), f"Missing {tok_path}"
    assert base_ckpt.is_file(), f"Missing {base_ckpt}"
    assert sft_ckpt.is_file(), f"Missing {sft_ckpt}"

    all_experiments: Dict[str, Any] = {}

    # Define Decoding Configurations to Compare
    configs_to_test = [
        ("Greedy (T=0.0)", GenerationConfig(max_new_tokens=40, temperature=0.0)),
        ("Default Sampling (T=0.7, p=0.9)", GenerationConfig(max_new_tokens=40, temperature=0.7, top_p=0.9, seed=42)),
        ("Calibrated SFT (T=0.7, p=0.9, rep_penalty=1.15)", GenerationConfig(max_new_tokens=40, temperature=0.7, top_p=0.9, seed=42, repetition_penalty=1.15)),
    ]

    # 1. Evaluate Pretrained Base Model (Step 3000)
    logger.info("=" * 70)
    logger.info("Evaluating Pretrained Base Model (Step 3000)...")
    logger.info("=" * 70)
    mgr_base = ModelManager()
    mgr_base.load_model(base_ckpt, tok_path, device_name=device, active_profile_key="5m_phase2_best")

    # Evaluate Base with Greedy and Default Sampling
    for label, cfg in [configs_to_test[0], configs_to_test[1]]:
        exp_key = f"base_step3000_{label}"
        logger.info(f"Running Base Model with {label}...")
        res = run_benchmark_for_model(mgr_base, is_sft=False, config=cfg, config_label=f"Base Pretrained - {label}")
        all_experiments[exp_key] = res
        logger.info(
            f"Base [{label}]: Obj Acc: {res['objective_accuracy_pct']}% | "
            f"Relevance: {res['relevance_rate_pct']}% | "
            f"Repetition: {res['mean_3gram_repetition_rate']:.4f} | "
            f"EOT Stop: {res['eot_stopping_rate_pct']}%"
        )

    mgr_base.unload_model()

    # 2. Evaluate SFT Best Model (Step 650)
    logger.info("=" * 70)
    logger.info("Evaluating Instruction-Tuned SFT Best Model (Step 650)...")
    logger.info("=" * 70)
    mgr_sft = ModelManager()
    mgr_sft.load_model(sft_ckpt, tok_path, device_name=device, active_profile_key="5m_sft_best")

    for label, cfg in configs_to_test:
        exp_key = f"sft_step650_{label}"
        logger.info(f"Running SFT Model with {label}...")
        res = run_benchmark_for_model(mgr_sft, is_sft=True, config=cfg, config_label=f"SFT Best - {label}")
        all_experiments[exp_key] = res
        logger.info(
            f"SFT [{label}]: Obj Acc: {res['objective_accuracy_pct']}% | "
            f"Relevance: {res['relevance_rate_pct']}% | "
            f"Repetition: {res['mean_3gram_repetition_rate']:.4f} | "
            f"EOT Stop: {res['eot_stopping_rate_pct']}%"
        )

    mgr_sft.unload_model()

    # Save to JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    benchmark_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "prompts_count": len(BENCHMARK_PROMPTS),
        "categories": ["greetings", "factual", "classification", "arithmetic", "explanation", "context_qa", "lists"],
        "experiments": all_experiments,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    logger.info(f"Saved complete conversational benchmark to {args.output_json}")

    # Print Summary Markdown Table
    print("\n" + "=" * 95)
    print("GROUNDUPGPT STEP 29: CONVERSATIONAL BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Experiment':<40} | {'Obj Acc':<8} | {'Relevance':<10} | {'Repetition':<10} | {'EOT Stop':<9} | {'Speed':<10}")
    print("-" * 95)
    for k, v in all_experiments.items():
        print(
            f"{v['config_label']:<40} | "
            f"{v['objective_accuracy_pct']:>6.1f}% | "
            f"{v['relevance_rate_pct']:>8.1f}% | "
            f"{v['mean_3gram_repetition_rate']:>10.4f} | "
            f"{v['eot_stopping_rate_pct']:>7.1f}% | "
            f"{v['tokens_per_sec']:>7.1f} t/s"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()

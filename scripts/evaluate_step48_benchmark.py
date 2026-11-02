"""Evaluation harness for GroundUpGPT Step 48 Frozen Conversational Benchmark.

Evaluates checkpoints on the 84-question frozen held-out benchmark across 12 categories:
1. Greetings / Acknowledgements
2. Short Direct Facts
3. Everyday Conversational Questions
4. Definitions
5. Basic Science
6. Simple Python Concepts
7. Elementary Classification
8. Single-Step Instructions
9. Small Lists
10. Simple Logical Relationships
11. Arithmetic (disjoint operands)
12. Concise Constraint Responses

Tests both Greedy (temp 0.0) and Calibrated (temp 0.8, top_p 0.9, rep_penalty 1.15) decoding.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, generate_sft_response
from src.model import GPTConfig, GroundUpGPT
from src.conversation_evaluation import (
    TaskType,
    _clean_text,
    _is_circular_tautology,
    _is_prompt_echo,
    evaluate_arithmetic,
    evaluate_binary_classification,
    evaluate_exact_factual,
    evaluate_n_item_list,
    evaluate_scientific_definition,
    evaluate_text_transformation,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "benchmark_conversational_step48.json"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def compute_3gram_repetition_rate(text: str) -> float:
    """Calculate the fraction of 3-grams that are duplicates."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique_trigrams = set(trigrams)
    duplicate_count = len(trigrams) - len(unique_trigrams)
    return duplicate_count / len(trigrams)


def grade_prompt(item: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    """Grade a benchmark item response using rigorous task-specific rubrics."""
    clean_r = _clean_text(raw_response)
    prompt = item["prompt"]
    task_type = item["task_type"]
    is_objective = item.get("is_objective", True)
    qid = item["id"]

    if not clean_r:
        return {
            "passed": False,
            "is_objective": is_objective,
            "requires_qualitative_review": not is_objective or task_type == "open_ended_conversational",
            "reason": "Empty or delimiter-only response",
            "task_type": task_type,
        }

    # Rejection of pure prompt echo
    if _is_prompt_echo(prompt, clean_r):
        return {
            "passed": False,
            "is_objective": is_objective,
            "requires_qualitative_review": not is_objective or task_type == "open_ended_conversational",
            "reason": f"Prompt echoed without answering: '{clean_r[:50]}'",
            "task_type": task_type,
        }

    # Open-ended conversational
    if not is_objective or task_type == "open_ended_conversational":
        norm_r = clean_r.lower()
        # Check if contains any acceptable equivalents or reasonable content
        # Check against common hallucination patterns like raw Python code blocks or gibberish
        if len(clean_r) < 3:
            return {
                "passed": False,
                "is_objective": False,
                "requires_qualitative_review": True,
                "reason": "Too short response (<3 chars)",
                "task_type": task_type,
            }
        
        # Check if response has some semantic alignment
        acc_equivs = item.get("acceptable_equivalents", [])
        has_relevant_concept = any(eq.lower() in norm_r for eq in acc_equivs)
        
        # Check for obvious incoherent jargon
        incoherent_indicators = ["def ", "class ", "import ", "http://", "```", "sold to today"]
        if any(ind in norm_r for ind in incoherent_indicators):
            return {
                "passed": False,
                "is_objective": False,
                "requires_qualitative_review": True,
                "reason": f"Incoherent or hallucinated jargon in reply: '{clean_r[:50]}'",
                "task_type": task_type,
            }

        return {
            "passed": has_relevant_concept,
            "is_objective": False,
            "requires_qualitative_review": True,
            "reason": "Relevant conversational response (heuristic keyword match)" if has_relevant_concept else "Plausible response but lacked key conversational concept",
            "task_type": task_type,
        }

    # Arithmetic
    if task_type == "numeric_arithmetic":
        res = evaluate_arithmetic(prompt, clean_r)
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    # Exact Factual
    if task_type == "exact_factual_qa":
        target = item["expected_answer"]
        aliases = item.get("acceptable_equivalents", [target])
        res = evaluate_exact_factual(prompt, clean_r, target_answer=target, aliases=aliases)
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    # Binary Classification
    if task_type == "binary_classification":
        # Determine correct vs incorrect from prompt and expected_answer
        correct = item["expected_answer"].lower()
        if " or " in prompt.lower():
            # e.g., "Is a sparrow a bird or a mammal?"
            opts = prompt.lower().split(" or ")
            # extract option words
            opt2 = re.findall(r"\b\w+\b", opts[1].replace("?", ""))[-1]
            opt1 = re.findall(r"\b\w+\b", opts[0])[-1]
            incorrect = opt2 if correct in opt1 else opt1
        elif "yes or no" in prompt.lower():
            incorrect = "no" if correct == "yes" else "yes"
        elif "true or false" in prompt.lower():
            incorrect = "false" if correct == "true" else "true"
        else:
            incorrect = "unknown"

        res = evaluate_binary_classification(prompt, clean_r, correct_category=correct, incorrect_category=incorrect)
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    # Scientific Definitions
    if task_type == "scientific_definitions":
        acc_equivs = item.get("acceptable_equivalents", [])
        concept_groups = [[eq] for eq in acc_equivs]
        defined_term = item.get("expected_answer", "").split()[0]
        res = evaluate_scientific_definition(
            prompt, clean_r, defined_term=defined_term, required_concept_groups=concept_groups, min_groups_required=1
        )
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    # Requested N-Item Lists
    if task_type == "requested_n_item_lists":
        valid_pool = set(item.get("acceptable_equivalents", []))
        # min count
        min_count = 2
        if "three" in prompt.lower():
            min_count = 3
        res = evaluate_n_item_list(prompt, clean_r, min_count=min_count, valid_pool=valid_pool)
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    # Deterministic Text Transformation
    if task_type == "deterministic_text_transformation":
        expected = item["expected_answer"]
        # Case sensitivity depends on prompt
        case_sensitive = "uppercase" in prompt.lower() or "capitalize" in prompt.lower()
        res = evaluate_text_transformation(prompt, clean_r, expected_substring=expected, case_sensitive=case_sensitive)
        return {"passed": res.passed, "is_objective": True, "requires_qualitative_review": False, "reason": res.reason, "task_type": task_type}

    return {"passed": False, "is_objective": is_objective, "requires_qualitative_review": not is_objective, "reason": f"Unhandled task type: {task_type}", "task_type": task_type}


def evaluate_checkpoint_on_benchmark(
    checkpoint_path: Path,
    device_str: str = "cuda",
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """Run full evaluation across Greedy and Calibrated decoding."""
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Loading checkpoint {checkpoint_path} on device {device}...")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    generator = TextGenerator(model, tokenizer, device=device)

    # Load frozen benchmark
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    prompts = benchmark_data["prompts"]

    decoding_modes = {
        "greedy": GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        ),
        "calibrated": GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            seed=42,
            stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        ),
    }

    eval_results = {}

    for mode_name, gen_cfg in decoding_modes.items():
        logger.info(f"Running evaluation mode: {mode_name} on {checkpoint_path.name}...")
        records = []
        total_time = 0.0
        total_tokens = 0
        obj_passed = 0
        obj_total = 0
        qual_passed = 0
        qual_total = 0
        repetition_rates = []
        clean_delimiter_stops = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start_mode_time = time.perf_counter()

        for item in prompts:
            prompt_text = item["prompt"]
            t0 = time.perf_counter()
            gen_res = generate_sft_response(generator, prompt=prompt_text, config=gen_cfg)
            elapsed = time.perf_counter() - t0

            raw_resp = gen_res.generated_text
            clean_resp = _clean_text(raw_resp)
            tokens_gen = gen_res.generated_tokens
            total_tokens += tokens_gen
            total_time += elapsed

            rep_rate = compute_3gram_repetition_rate(clean_resp)
            repetition_rates.append(rep_rate)

            # Check clean delimiter stop
            if gen_res.stop_reason in ("end_of_turn", "user_delimiter", "assistant_delimiter", "stop_sequence", "end_of_text"):
                clean_delimiter_stops += 1

            grade = grade_prompt(item, clean_resp)

            if grade["is_objective"]:
                obj_total += 1
                if grade["passed"]:
                    obj_passed += 1
            else:
                qual_total += 1
                if grade["passed"]:
                    qual_passed += 1

            records.append({
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt_text,
                "expected": item["expected_answer"],
                "task_type": item["task_type"],
                "is_objective": grade["is_objective"],
                "generated_text": clean_resp,
                "raw_text": raw_resp,
                "generated_tokens": tokens_gen,
                "elapsed_sec": round(elapsed, 4),
                "stop_reason": str(gen_res.stop_reason),
                "repetition_rate": round(rep_rate, 4),
                "passed": grade["passed"],
                "reason": grade["reason"],
            })

        mode_wall_time = time.perf_counter() - start_mode_time
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )

        avg_rep = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
        tok_per_sec = total_tokens / max(total_time, 1e-6)
        obj_accuracy = (obj_passed / obj_total) if obj_total > 0 else 0.0
        qual_accuracy = (qual_passed / qual_total) if qual_total > 0 else 0.0

        # Category breakdown
        cat_stats = {}
        for r in records:
            cat = r["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "passed": 0, "objective_total": 0, "objective_passed": 0}
            cat_stats[cat]["total"] += 1
            if r["passed"]:
                cat_stats[cat]["passed"] += 1
            if r["is_objective"]:
                cat_stats[cat]["objective_total"] += 1
                if r["passed"]:
                    cat_stats[cat]["objective_passed"] += 1

        eval_results[mode_name] = {
            "total_prompts": len(prompts),
            "primary_objective_accuracy": round(obj_accuracy, 4),
            "objective_total": obj_total,
            "objective_passed": obj_passed,
            "objective_accuracy": round(obj_accuracy, 4),
            "heuristic_phrase_match_indicator": round(qual_accuracy, 4),
            "heuristic_disclaimer": "Non-objective heuristic based on keyword presence; requires manual qualitative review. Not an indicator of verified semantic correctness.",
            "qualitative_total": qual_total,
            "qualitative_passed": qual_passed,
            "qualitative_accuracy": round(qual_accuracy, 4),
            "delimiter_stopping_rate": round(clean_delimiter_stops / len(prompts), 4),
            "avg_3gram_repetition_rate": round(avg_rep, 4),
            "total_tokens_generated": total_tokens,
            "avg_tokens_per_response": round(total_tokens / len(prompts), 2),
            "throughput_tokens_per_sec": round(tok_per_sec, 2),
            "wall_clock_time_sec": round(mode_wall_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "category_breakdown": cat_stats,
            "records": records,
        }

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "device": str(device),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(BENCHMARK_PATH.name),
        "benchmark_sha256": hashlib.sha256(open(BENCHMARK_PATH, "rb").read()).hexdigest(),
        "num_prompts": len(prompts),
        "results": eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Step 48 Frozen Benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=48, help="Max new tokens to generate")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = evaluate_checkpoint_on_benchmark(
        checkpoint_path=ckpt_path,
        device_str=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved evaluation results to {out_path}")
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {ckpt_path.name}")
    print("=" * 70)
    for mode in ["greedy", "calibrated"]:
        m = result["results"][mode]
        print(f"Mode: {mode.upper()}")
        print(f"  Primary Objective Accuracy:       {m['objective_passed']}/{m['objective_total']} ({m['objective_accuracy']*100:.1f}%) [STRICT OBJECTIVE]")
        print(f"  Heuristic Phrase Match Indicator: {m['qualitative_passed']}/{m['qualitative_total']} ({m['heuristic_phrase_match_indicator']*100:.1f}%) [Unverified Heuristic]")
        print(f"  Delimiter Stop Rate:              {m['delimiter_stopping_rate']*100:.1f}%")
        print(f"  3-Gram Repetition Rate:           {m['avg_3gram_repetition_rate']*100:.1f}%")
        print(f"  Avg Tokens / Response:            {m['avg_tokens_per_response']}")
        print(f"  Throughput:                       {m['throughput_tokens_per_sec']:.1f} tok/s")
        print(f"  Peak VRAM:                        {m['peak_vram_mb']:.1f} MB")
        print("-" * 70)


if __name__ == "__main__":
    main()

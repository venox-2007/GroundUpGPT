"""Comparative Language Generation Quality Benchmark for Step 35.

Evaluates Step 3968 (Phase 3 Best) vs Step 5949 (Phase 4 Best) on 16 diverse prompts
covering 6 distinct domains across both Greedy and Calibrated Sampling configurations.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT

PROMPTS = [
    # 1. Ordinary English prose
    {"id": "prose_1", "category": "Ordinary English Prose", "prompt": "The sun rose over the"},
    {"id": "prose_2", "category": "Ordinary English Prose", "prompt": "In the middle of the room,"},
    {"id": "prose_3", "category": "Ordinary English Prose", "prompt": "She walked down the street and"},
    # 2. Narrative continuation
    {"id": "narr_1", "category": "Narrative Continuation", "prompt": "Once upon a time in a small village,"},
    {"id": "narr_2", "category": "Narrative Continuation", "prompt": "He opened the mysterious door and saw"},
    {"id": "narr_3", "category": "Narrative Continuation", "prompt": "The dark storm clouds gathered as"},
    # 3. Science and geography
    {"id": "sci_1", "category": "Science & Geography", "prompt": "The capital of France is"},
    {"id": "sci_2", "category": "Science & Geography", "prompt": "The Earth revolves around the"},
    {"id": "sci_3", "category": "Science & Geography", "prompt": "The Atlantic Ocean lies between"},
    # 4. Short factual statements
    {"id": "fact_1", "category": "Short Factual Statements", "prompt": "Water is made of"},
    {"id": "fact_2", "category": "Short Factual Statements", "prompt": "A triangle has three"},
    {"id": "fact_3", "category": "Short Factual Statements", "prompt": "Oxygen is essential for"},
    # 5. Simple definitions
    {"id": "def_1", "category": "Simple Definitions", "prompt": "Biology is the study of"},
    {"id": "def_2", "category": "Simple Definitions", "prompt": "A library is a place where"},
    # 6. Basic question-like sentences
    {"id": "quest_1", "category": "Basic Questions", "prompt": "What is the reason why"},
    {"id": "quest_2", "category": "Basic Questions", "prompt": "How do birds fly through the"},
]

FACTUAL_TARGETS = {
    "The capital of France is": ["paris"],
    "The Earth revolves around the": ["sun"],
    "The Atlantic Ocean lies between": ["america", "europe", "africa"],
    "Water is made of": ["hydrogen", "oxygen", "h2o", "atoms", "molecules"],
    "A triangle has three": ["sides", "angles", "vertices", "corners", "points"],
    "Oxygen is essential for": ["life", "living", "breathing", "respiration", "animals", "humans"],
    "Biology is the study of": ["life", "living organisms", "living things", "cells", "plants", "animals"],
    "A library is a place where": ["books", "read", "information", "study"],
}


def contains_complete_sentence(text: str) -> bool:
    """Automated Heuristic: Checks if text contains at least one sentence terminator (. ? !)

    with non-empty text preceding and succeeding it (or ending at terminator).
    """
    clean = text.strip()
    match = re.search(r"[a-zA-Z0-9,\'\"\s]+[.!?](\s|$)", clean)
    return match is not None


def calculate_repetition_rate(text: str, n: int = 3) -> float:
    """Automated Heuristic: Proportion of repeated n-grams in the tokenized string."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)
    return round(1.0 - (len(unique_ngrams) / len(ngrams)), 3)


def detect_malformed_tokens(text: str) -> List[str]:
    """Automated Heuristic: Flags non-standard words (e.g. repeated consonant clusters, invalid symbols)."""
    words = re.findall(r"\b[A-Za-z]+\b", text)
    malformed = []
    for w in words:
        wl = w.lower()
        # Repeated character 4+ times (e.g. 'aaaa', 'zzzz')
        if re.search(r"(.)\1{3,}", wl):
            malformed.append(w)
        # Consonant clusters of 5+ consonants in English (excluding common Welsh/foreign words)
        elif re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", wl):
            malformed.append(w)
    return list(set(malformed))


def evaluate_model_on_prompts(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    prompts: List[Dict[str, str]],
    device: str,
) -> Dict[str, Any]:
    gen = TextGenerator(model, tokenizer=tokenizer, device=device)

    configs = {
        "greedy": GenerationConfig(max_new_tokens=45, temperature=0.0),
        "sampling": GenerationConfig(max_new_tokens=45, temperature=0.8, top_k=40, top_p=0.9, seed=1337),
    }

    eval_results: Dict[str, List[Dict[str, Any]]] = {"greedy": [], "sampling": []}

    for cfg_name, cfg in configs.items():
        for p_item in prompts:
            p_text = p_item["prompt"]
            out = gen.generate(p_text, config=cfg)
            full = out.full_text
            completion = out.generated_text

            has_complete = contains_complete_sentence(full)
            rep_rate = calculate_repetition_rate(full, n=3)
            malformed = detect_malformed_tokens(full)

            # Factual check if applicable
            factual_correct = None
            if p_text in FACTUAL_TARGETS:
                expected = FACTUAL_TARGETS[p_text]
                factual_correct = any(t in full.lower() for t in expected)

            eval_results[cfg_name].append({
                "id": p_item["id"],
                "category": p_item["category"],
                "prompt": p_text,
                "completion": completion,
                "full_text": full,
                "has_complete_sentence": has_complete,
                "repetition_rate_3gram": rep_rate,
                "malformed_words": malformed,
                "factual_correct": factual_correct,
            })

    return eval_results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = Path("checkpoints/bpe_vocab4096.json")
    tokenizer = BPETokenizer.load(tok_path)

    # 1. Load Step 3968 Model
    path_3968 = Path("checkpoints/27m_phase3/model_27m_best.pt")
    ckpt_3968 = torch.load(path_3968, map_location=device, weights_only=False)
    cfg_3968 = GPTConfig(**ckpt_3968["model_config"])
    model_3968 = GroundUpGPT(cfg_3968).to(device)
    model_3968.load_state_dict(ckpt_3968["model_state_dict"])
    model_3968.eval()

    # 2. Load Step 5949 Model
    path_5949 = Path("checkpoints/27m_phase4/model_27m_best.pt")
    ckpt_5949 = torch.load(path_5949, map_location=device, weights_only=False)
    cfg_5949 = GPTConfig(**ckpt_5949["model_config"])
    model_5949 = GroundUpGPT(cfg_5949).to(device)
    model_5949.load_state_dict(ckpt_5949["model_state_dict"])
    model_5949.eval()

    print("[*] Running comparative evaluation on 16 prompts...")
    results_3968 = evaluate_model_on_prompts(model_3968, tokenizer, PROMPTS, device=device)
    results_5949 = evaluate_model_on_prompts(model_5949, tokenizer, PROMPTS, device=device)

    # Calculate Summary Statistics
    summary: Dict[str, Any] = {"step_3968": {}, "step_5949": {}}

    for name, res in [("step_3968", results_3968), ("step_5949", results_5949)]:
        for cfg_name in ["greedy", "sampling"]:
            items = res[cfg_name]
            total = len(items)
            complete_count = sum(1 for it in items if it["has_complete_sentence"])
            avg_rep = sum(it["repetition_rate_3gram"] for it in items) / total
            malformed_count = sum(len(it["malformed_words"]) for it in items)
            factual_items = [it for it in items if it["factual_correct"] is not None]
            factual_correct_count = sum(1 for it in factual_items if it["factual_correct"] is True)

            summary[name][cfg_name] = {
                "complete_sentence_ratio": round(complete_count / total, 3),
                "avg_3gram_repetition_rate": round(avg_rep, 3),
                "total_malformed_words": malformed_count,
                "factual_correctness_ratio": round(factual_correct_count / len(factual_items), 3),
                "factual_correct_count": factual_correct_count,
                "factual_total": len(factual_items),
            }

    output = {
        "summary": summary,
        "step_3968_results": results_3968,
        "step_5949_results": results_5949,
    }

    out_file = Path("logs/generation_quality_step35.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[*] Assessment saved to: {out_file}")
    print("\n=== SUMMARY COMPARISON ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Audit the Dolly SFT dataset distribution and content for Step 39."""

import json
import re
from collections import Counter
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    train_cache_path = REPO_ROOT / "data" / "cache_sft_27m_v1" / "train_sft_tokens.pt"
    val_cache_path = REPO_ROOT / "data" / "cache_sft_27m_v1" / "val_sft_tokens.pt"
    raw_dolly_path = REPO_ROOT / "data" / "raw_sft" / "databricks-dolly-15k.jsonl"
    out_dir = REPO_ROOT / "logs" / "diagnostics_step39"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cache = torch.load(train_cache_path, weights_only=False)
    val_cache = torch.load(val_cache_path, weights_only=False)

    with open(raw_dolly_path, "r", encoding="utf-8") as f:
        dolly_lines = [json.loads(line) for line in f]

    results = {}

    for split_name, cache in [
        ("train", train_cache),
        ("val", val_cache),
        ("combined", train_cache + val_cache),
    ]:
        n = len(cache)
        prompt_lens = [ex["prompt_tokens_count"] + 1 for ex in cache]
        resp_lens = [ex["response_tokens_count"] for ex in cache]
        total_lens = [ex["total_tokens_count"] for ex in cache]

        under_20 = sum(1 for r in resp_lens if r < 20)
        under_40 = sum(1 for r in resp_lens if r < 40)
        under_80 = sum(1 for r in resp_lens if r < 80)

        single_sentence = 0
        greetings = 0
        thank_yous = 0
        arithmetic = 0
        what_is = 0
        list_prompts = 0
        definitions = 0
        classifications = 0
        open_explanations = 0
        long_context = 0

        categories = Counter(ex.get("category", "unknown") for ex in cache)

        sample_greetings = []
        sample_thank_yous = []
        sample_arithmetic = []

        for ex in cache:
            meta = ex.get("metadata", {})
            line_idx = meta.get("source_line", 1) - 1
            raw = dolly_lines[line_idx]
            inst = raw["instruction"].strip()
            inst_lower = inst.lower()
            resp = raw["response"].strip()
            ctx = raw.get("context", "").strip()

            sents = [s for s in re.split(r"[.!?]+", resp) if s.strip()]
            if len(sents) <= 1:
                single_sentence += 1

            if any(w in inst_lower for w in ["hello", "hi ", "hey ", "how are you", "good morning"]):
                greetings += 1
                if len(sample_greetings) < 5:
                    sample_greetings.append({"instruction": inst, "response": resp[:100]})

            if any(w in inst_lower for w in ["thank you", "thanks", "appreciate"]):
                thank_yous += 1
                if len(sample_thank_yous) < 5:
                    sample_thank_yous.append({"instruction": inst, "response": resp[:100]})

            # Arithmetic heuristic
            if any(re.search(pat, inst_lower) for pat in [r"\b\d+\s*[\+\-\*\/]\s*\d+\b", r"\badd\b", r"\bsubtract\b", r"\bmultiply\b", r"\bdivide\b", r"\bcalculate\b", r"\barithmetic\b", r"\bhow many\b"]):
                arithmetic += 1
                if len(sample_arithmetic) < 5:
                    sample_arithmetic.append({"instruction": inst, "response": resp[:100]})

            if inst_lower.startswith("what is ") or inst_lower.startswith("what are "):
                what_is += 1

            if "define " in inst_lower or inst_lower.startswith("define ") or "definition" in inst_lower:
                definitions += 1

            if "classify" in inst_lower or ex.get("category") == "classification":
                classifications += 1

            if "explain" in inst_lower or "why" in inst_lower:
                open_explanations += 1

            if len(ctx) > 300:
                long_context += 1

            if "list " in inst_lower or inst_lower.startswith("list"):
                list_prompts += 1

        split_stats = {
            "total_examples": n,
            "category_distribution": dict(categories),
            "category_percentages": {k: round(v / n * 100, 2) for k, v in categories.items()},
            "prompt_length": {
                "mean": round(sum(prompt_lens) / n, 2),
                "median": sorted(prompt_lens)[n // 2],
                "min": min(prompt_lens),
                "max": max(prompt_lens),
            },
            "response_length": {
                "mean": round(sum(resp_lens) / n, 2),
                "median": sorted(resp_lens)[n // 2],
                "min": min(resp_lens),
                "max": max(resp_lens),
                "under_20_tokens_count": under_20,
                "under_20_tokens_pct": round(under_20 / n * 100, 2),
                "under_40_tokens_count": under_40,
                "under_40_tokens_pct": round(under_40 / n * 100, 2),
                "under_80_tokens_count": under_80,
                "under_80_tokens_pct": round(under_80 / n * 100, 2),
                "single_sentence_count": single_sentence,
                "single_sentence_pct": round(single_sentence / n * 100, 2),
            },
            "task_subtypes": {
                "greetings_count": greetings,
                "greetings_pct": round(greetings / n * 100, 2),
                "thank_you_count": thank_yous,
                "thank_you_pct": round(thank_yous / n * 100, 2),
                "arithmetic_count": arithmetic,
                "arithmetic_pct": round(arithmetic / n * 100, 2),
                "what_is_count": what_is,
                "what_is_pct": round(what_is / n * 100, 2),
                "definitions_count": definitions,
                "definitions_pct": round(definitions / n * 100, 2),
                "classifications_count": classifications,
                "classifications_pct": round(classifications / n * 100, 2),
                "open_explanations_count": open_explanations,
                "open_explanations_pct": round(open_explanations / n * 100, 2),
                "long_context_count": long_context,
                "long_context_pct": round(long_context / n * 100, 2),
                "list_prompts_count": list_prompts,
                "list_prompts_pct": round(list_prompts / n * 100, 2),
            },
            "samples": {
                "sample_greetings": sample_greetings,
                "sample_thank_yous": sample_thank_yous,
                "sample_arithmetic": sample_arithmetic,
            },
        }
        results[split_name] = split_stats

    # Check for duplicate prompts
    all_prompts = [dolly_lines[ex["metadata"]["source_line"] - 1]["instruction"].strip().lower() for ex in train_cache]
    prompt_counts = Counter(all_prompts)
    duplicates = {p: c for p, c in prompt_counts.items() if c > 1}
    results["prompt_duplicates_in_train"] = {
        "distinct_prompts": len(prompt_counts),
        "total_prompts": len(all_prompts),
        "duplicate_prompts_count": len(duplicates),
        "duplicate_prompts_samples": dict(list(duplicates.items())[:10]),
    }

    out_file = out_dir / "sft_dataset_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved analysis to {out_file}")
    comb = results["combined"]
    print("\n=== DATASET ANALYSIS SUMMARY ===")
    print(f"Total Examples: {comb['total_examples']}")
    print(f"Categories: {comb['category_percentages']}")
    print(f"Prompt Length: mean={comb['prompt_length']['mean']}, median={comb['prompt_length']['median']}")
    print(f"Response Length: mean={comb['response_length']['mean']}, median={comb['response_length']['median']}")
    print(f"Responses < 20 tokens: {comb['response_length']['under_20_tokens_pct']}%")
    print(f"Responses < 40 tokens: {comb['response_length']['under_40_tokens_pct']}%")
    print(f"Responses < 80 tokens: {comb['response_length']['under_80_tokens_pct']}%")
    print(f"Single Sentence: {comb['response_length']['single_sentence_pct']}%")
    print(f"Greetings in prompt: {comb['task_subtypes']['greetings_count']} ({comb['task_subtypes']['greetings_pct']}%)")
    print(f"Thank you in prompt: {comb['task_subtypes']['thank_you_count']} ({comb['task_subtypes']['thank_you_pct']}%)")
    print(f"Arithmetic in prompt: {comb['task_subtypes']['arithmetic_count']} ({comb['task_subtypes']['arithmetic_pct']}%)")
    print(f"\"What is/are\": {comb['task_subtypes']['what_is_count']} ({comb['task_subtypes']['what_is_pct']}%)")
    print(f"Definitions: {comb['task_subtypes']['definitions_count']} ({comb['task_subtypes']['definitions_pct']}%)")
    print(f"List prompts: {comb['task_subtypes']['list_prompts_count']} ({comb['task_subtypes']['list_prompts_pct']}%)")
    print(f"Duplicate prompts in train: {results['prompt_duplicates_in_train']['duplicate_prompts_count']}")

if __name__ == "__main__":
    main()

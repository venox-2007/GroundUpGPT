"""Compile and analyze 3-way model comparison across Step 48 benchmark.

Compares:
1. Model A: Original 27M SFT Baseline (Step 600)
2. Model B: Step 40 Anchor SFT (Step 250)
3. Model C: Step 48 Neural v2 SFT (Step 350)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "logs" / "autonomous_steps_44_46"


def main():
    orig_path = LOG_DIR / "step48_baseline_orig_sft.json"
    anchor_path = LOG_DIR / "step48_baseline_anchor_sft.json"
    neural_v2_path = LOG_DIR / "step48_neural_v2_benchmark_eval.json"

    with open(orig_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    with open(anchor_path, "r", encoding="utf-8") as f:
        anchor_data = json.load(f)
    with open(neural_v2_path, "r", encoding="utf-8") as f:
        neural_data = json.load(f)

    orig_records = {r["id"]: r for r in orig_data["results"]["greedy"]["records"]}
    anchor_records = {r["id"]: r for r in anchor_data["results"]["greedy"]["records"]}
    neural_records = {r["id"]: r for r in neural_data["results"]["greedy"]["records"]}

    comparison_records = []

    for qid in sorted(orig_records.keys()):
        r_orig = orig_records[qid]
        r_anc = anchor_records[qid]
        r_neu = neural_records[qid]

        item = {
            "id": qid,
            "category": r_orig["category"],
            "prompt": r_orig["prompt"],
            "expected": r_orig["expected"],
            "task_type": r_orig["task_type"],
            "is_objective": r_orig["is_objective"],
            "orig_sft_generated": r_orig["generated_text"],
            "orig_sft_passed": r_orig["passed"],
            "orig_sft_stop": r_orig["stop_reason"],
            "anchor_sft_generated": r_anc["generated_text"],
            "anchor_sft_passed": r_anc["passed"],
            "anchor_sft_stop": r_anc["stop_reason"],
            "neural_v2_generated": r_neu["generated_text"],
            "neural_v2_passed": r_neu["passed"],
            "neural_v2_stop": r_neu["stop_reason"],
        }
        comparison_records.append(item)

    out_file = LOG_DIR / "step48_3way_model_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    print(f"Compiled {len(comparison_records)} comparative items to {out_file}\n")

    indices = [0, 1, 2, 3, 5, 7, 8, 9, 10, 14, 15, 21, 22, 28, 29, 35, 36, 42, 43, 49, 50, 56, 57, 63, 64, 70, 71, 77, 78, 83]
    for idx in indices:
        it = comparison_records[idx]
        print(f"=== [{it['id']}] {it['prompt']} ===")
        print("  Expected:   ", it["expected"])
        print("  Orig SFT:   ", repr(it["orig_sft_generated"]))
        print("  Anchor SFT: ", repr(it["anchor_sft_generated"]))
        print("  Neural v2:  ", repr(it["neural_v2_generated"]))
        print("  Passed: Orig=", it["orig_sft_passed"], "Anchor=", it["anchor_sft_passed"], "Neural_v2=", it["neural_v2_passed"])
        print()


if __name__ == "__main__":
    main()

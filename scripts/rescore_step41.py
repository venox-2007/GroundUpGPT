"""Re-score Step 40 benchmark generations using the strict, task-specific evaluation engine.

Computes exact objective accuracy, category breakdowns, length distributions,
natural stop rates, delimiter leakage, and repetition metrics across:
- Frozen 35 benchmark prompts
- 56 Held-out anchor transfer test prompts
- 20 Seen anchor training prompts
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conversation_evaluation import TaskType, grade_benchmark_item


def calculate_repetition_rate(text: str, n: int = 4) -> float:
    """Calculate n-gram repetition fraction in text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ngrams = set(ngrams)
    return round((1.0 - len(unique_ngrams) / len(ngrams)) * 100.0, 1)


def rescore_all_responses(
    input_comparisons_path: Path,
    output_metrics_path: Path,
    output_regraded_comparisons_path: Path,
) -> Dict[str, Any]:
    with open(input_comparisons_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    suites = ["frozen_35", "anchor_transfer_test", "seen_anchor_train"]
    models = ["baseline_step600", "target_anchors_best"]
    decodings = ["greedy", "calibrated"]

    regraded_comparisons: List[Dict[str, Any]] = []

    # Statistics tracking
    suite_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for s in suites:
        suite_stats[s] = {}
        for m in models:
            suite_stats[s][m] = {}
            for d in decodings:
                suite_stats[s][m][d] = {
                    "total_prompts": 0,
                    "objective_scored": 0,
                    "objective_correct": 0,
                    "objective_incorrect": 0,
                    "unscored_qualitative": 0,
                    "objective_accuracy_pct": 0.0,
                    "by_task_type": {},
                    "by_category": {},
                    "lengths": [],
                    "mean_response_tokens": 0.0,
                    "median_response_tokens": 0,
                    "natural_stop_count": 0,
                    "max_token_cutoff_count": 0,
                    "natural_stop_rate_pct": 0.0,
                    "repetition_rates": [],
                    "mean_repetition_pct": 0.0,
                    "delimiter_leakage_count": 0,
                }

    for it in items:
        iid = it["id"]
        suite = it["suite"]
        cat = it["category"]
        p = it["prompt"]
        exp = it["expected_guidance"]

        regraded_item = {
            "id": iid,
            "suite": suite,
            "category": cat,
            "prompt": p,
            "expected_guidance": exp,
            "models": {},
        }

        for m in models:
            regraded_item["models"][m] = {}
            for d in decodings:
                model_data = it["models"][m][d]
                resp_text = model_data["text"]
                r_len = model_data["token_length"]

                eval_res = grade_benchmark_item(
                    item_id=iid,
                    prompt=p,
                    response=resp_text,
                    category=cat,
                    suite=suite,
                    expected_guidance=exp,
                )

                rep_rate = calculate_repetition_rate(resp_text)
                has_delim = any(delim in resp_text for delim in ("<|endoftext|>", "<|user|>", "<|assistant|>"))
                stopped_naturally = r_len < 52  # 52 was max_new_tokens

                st = suite_stats[suite][m][d]
                st["total_prompts"] += 1
                st["lengths"].append(r_len)
                st["repetition_rates"].append(rep_rate)
                if has_delim:
                    st["delimiter_leakage_count"] += 1
                if stopped_naturally:
                    st["natural_stop_count"] += 1
                else:
                    st["max_token_cutoff_count"] += 1

                ttype_str = eval_res.task_type.value
                st["by_task_type"].setdefault(ttype_str, {"total": 0, "correct": 0})
                st["by_task_type"][ttype_str]["total"] += 1

                st["by_category"].setdefault(cat, {"total": 0, "correct": 0})
                st["by_category"][cat]["total"] += 1

                if eval_res.is_objective:
                    st["objective_scored"] += 1
                    if eval_res.passed:
                        st["objective_correct"] += 1
                        st["by_task_type"][ttype_str]["correct"] += 1
                        st["by_category"][cat]["correct"] += 1
                    else:
                        st["objective_incorrect"] += 1
                else:
                    st["unscored_qualitative"] += 1

                regraded_item["models"][m][d] = {
                    "text": resp_text,
                    "token_length": r_len,
                    "repetition_pct": rep_rate,
                    "stopped_naturally": stopped_naturally,
                    "delimiter_leakage": has_delim,
                    "grade": eval_res.to_dict(),
                }

        regraded_comparisons.append(regraded_item)

    # Compute aggregate percentages
    for s in suites:
        for m in models:
            for d in decodings:
                st = suite_stats[s][m][d]
                obj_tot = st["objective_scored"]
                obj_pass = st["objective_correct"]
                st["objective_accuracy_pct"] = round((obj_pass / max(1, obj_tot)) * 100.0, 1) if obj_tot else 0.0

                lens = st["lengths"]
                st["mean_response_tokens"] = round(sum(lens) / max(1, len(lens)), 1)
                st["median_response_tokens"] = sorted(lens)[len(lens) // 2] if lens else 0

                tot_p = st["total_prompts"]
                st["natural_stop_rate_pct"] = round((st["natural_stop_count"] / max(1, tot_p)) * 100.0, 1)

                reps = st["repetition_rates"]
                st["mean_repetition_pct"] = round(sum(reps) / max(1, len(reps)), 1)

                # Format task type and category percentages
                for k, v in st["by_task_type"].items():
                    v["accuracy_pct"] = round((v["correct"] / max(1, v["total"])) * 100.0, 1)
                for k, v in st["by_category"].items():
                    v["accuracy_pct"] = round((v["correct"] / max(1, v["total"])) * 100.0, 1)

                # Don't serialize the raw list of lengths/rates in summary
                del st["lengths"]
                del st["repetition_rates"]

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(suite_stats, f, indent=2)

    with open(output_regraded_comparisons_path, "w", encoding="utf-8") as f:
        json.dump(regraded_comparisons, f, indent=2)

    return suite_stats


def main() -> None:
    comp_path = REPO_ROOT / "logs" / "anchor_sft_step40" / "response_comparisons.json"
    out_metrics = REPO_ROOT / "logs" / "anchor_sft_step40" / "regraded_metrics_step41.json"
    out_comps = REPO_ROOT / "logs" / "anchor_sft_step40" / "regraded_comparisons_step41.json"

    stats = rescore_all_responses(comp_path, out_metrics, out_comps)
    print("Re-scoring complete. Summary:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

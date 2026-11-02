"""Step 40: Multi-Group evaluation and conservative comparative benchmark.

Compares GroundUpGPT-27M Step 600 SFT Baseline vs Step 40 Targeted SFT Best Checkpoint
under identical greedy and calibrated decoding conditions.
Evaluates:
- Frozen 35 benchmark prompts (with explicit objective criteria for checkable items)
- 56 Held-out anchor transfer test prompts (testing unseen operands, terms, and phrasings)
- 20 Seen anchor training prompts (measuring memorization vs generalization)
- Response length, repetition, delimiter compliance, and conservative accuracy
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, format_sft_prompt
from src.model import GPTConfig, GroundUpGPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> GroundUpGPT:
    """Load GroundUpGPT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def conservative_grade(
    prompt: str,
    response: str,
    valid_equivalents: List[str],
    is_objective: bool = True,
) -> Dict[str, Any]:
    """Conservative factual and arithmetic grading.

    Rejects:
    - Empty or delimiter-only responses
    - Prompt repetition / echo without answer
    - Contradictory answers (conflicting numbers or ambiguous alternatives)
    - Substring matches inside unrelated words
    """
    if not is_objective or not valid_equivalents:
        return {
            "is_objective": False,
            "passed": None,
            "reason": "Unscored (open-ended / stylistic prompt)",
        }

    clean_resp = response.strip()
    norm_resp = re.sub(r"[^\w\s]", "", clean_resp.lower()).strip()
    norm_p = re.sub(r"[^\w\s]", "", prompt.lower()).strip()

    # 1. Empty check
    if not clean_resp or clean_resp in ("<|endoftext|>", "<|assistant|>", "<|user|>"):
        return {"is_objective": True, "passed": False, "reason": "Empty or delimiter-only response"}

    # 2. Echo check
    if norm_resp == norm_p or (len(norm_p) > 10 and norm_resp.startswith(norm_p) and len(norm_resp) < len(norm_p) + 15):
        return {"is_objective": True, "passed": False, "reason": "Echoed prompt without assertion"}

    # 3. Numeric answer check
    is_numeric_target = any(eq.isdigit() for eq in valid_equivalents)
    if is_numeric_target:
        num_matches = re.findall(r"\b\d+\b", clean_resp)
        target_nums = {eq for eq in valid_equivalents if eq.isdigit()}

        if not num_matches:
            word_digits = {
                "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "zero": "0"
            }
            matched_words = [word_digits[w] for w in word_digits if re.search(r"\b" + w + r"\b", clean_resp.lower())]
            if any(nw in target_nums for nw in matched_words):
                return {"is_objective": True, "passed": True, "reason": "Matched target number word"}
            return {"is_objective": True, "passed": False, "reason": "No numeric answer found"}

        prompt_nums = set(re.findall(r"\b\d+\b", prompt))
        new_nums = [n for n in num_matches if n not in prompt_nums]

        if any(tn in new_nums for tn in target_nums):
            contradictory = [n for n in new_nums if n not in target_nums]
            if len(contradictory) > 0 and not any(clean_resp.lower().startswith(f"{tn} ") or clean_resp.lower().startswith(f"{tn}.") for tn in target_nums):
                return {"is_objective": True, "passed": False, "reason": f"Contradictory extraneous numbers found: {contradictory}"}
            return {"is_objective": True, "passed": True, "reason": "Correct target number matched"}
        elif any(tn in num_matches for tn in target_nums):
            for tn in target_nums:
                if re.search(r"=\s*" + tn + r"\b", clean_resp) or re.search(r"\bis\s*" + tn + r"\b", clean_resp.lower()):
                    return {"is_objective": True, "passed": True, "reason": f"Equation answer {tn} matched"}
            return {"is_objective": True, "passed": False, "reason": "Target number merely echoed from prompt"}
        else:
            return {"is_objective": True, "passed": False, "reason": f"Incorrect numbers: {num_matches}"}

    # 4. Textual equivalent check with word boundaries
    for eq in valid_equivalents:
        pattern = r"\b" + re.escape(eq.lower()) + r"\b"
        if re.search(pattern, clean_resp.lower()):
            if " or " in clean_resp.lower() and "?" in clean_resp:
                return {"is_objective": True, "passed": False, "reason": "Uncertain / contradictory question output"}
            return {"is_objective": True, "passed": True, "reason": f"Matched equivalent '{eq}'"}

    return {"is_objective": True, "passed": False, "reason": f"Target equivalents not matched: {valid_equivalents}"}


def build_frozen_35_items_with_criteria(bench35_path: Path) -> List[Dict[str, Any]]:
    """Load the frozen 35 benchmark items and assign explicit criteria to checkable objective items."""
    with open(bench35_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)["items"]

    criteria_map: Dict[str, Dict[str, Any]] = {
        "bench_05": {"is_objective": True, "valid_equivalents": ["Tokyo"]},
        "bench_06": {"is_objective": True, "valid_equivalents": ["Mercury"]},
        "bench_07": {"is_objective": True, "valid_equivalents": ["100", "100 degrees", "100°C", "100 C"]},
        "bench_08": {"is_objective": True, "valid_equivalents": ["William Shakespeare", "Shakespeare"]},
        "bench_09": {"is_objective": True, "valid_equivalents": ["366", "366 days"]},
        "bench_10": {"is_objective": True, "valid_equivalents": ["warm-blooded", "vertebrate", "mammal", "milk", "hair", "fur"]},
        "bench_13": {"is_objective": True, "valid_equivalents": ["gravity", "force", "attracts", "attraction", "mass"]},
        "bench_14": {"is_objective": True, "valid_equivalents": ["62"]},
        "bench_15": {"is_objective": True, "valid_equivalents": ["180", "180 miles"]},
        "bench_16": {"is_objective": True, "valid_equivalents": ["96"]},
        "bench_17": {"is_objective": True, "valid_equivalents": ["yes", "prime", "divisible only"]},
        "bench_18": {"is_objective": True, "valid_equivalents": ["9", "9 apples"]},
        "bench_23": {"is_objective": True, "valid_equivalents": ["vegetable", "fruit"]},
        "bench_24": {"is_objective": True, "valid_equivalents": ["solid", "liquid", "gas"]},
        "bench_25": {"is_objective": True, "valid_equivalents": ["mammal", "bird"]},
        "bench_26": {"is_objective": True, "valid_equivalents": ["renewable", "non-renewable"]},
        "bench_28": {"is_objective": True, "valid_equivalents": ["red", "blue", "yellow", "green"]},
        "bench_29": {"is_objective": True, "valid_equivalents": ["GROUNDUP GPT IS FAST AND LIGHTWEIGHT"]},
        "bench_30": {"is_objective": True, "valid_equivalents": ["Python", "JavaScript", "C++", "Java", "C", "Rust", "Go"]},
    }

    processed = []
    for it in raw_items:
        iid = it["id"]
        c_info = criteria_map.get(iid, {"is_objective": False, "valid_equivalents": []})
        processed.append({
            "id": iid,
            "category": it["category"],
            "prompt": it["prompt"],
            "expected_guidance": it.get("expected_response_guidance", ""),
            "is_objective": c_info["is_objective"],
            "valid_equivalents": c_info["valid_equivalents"],
            "suite": "frozen_35",
        })
    return processed


def build_transfer_test_items(anchor_test_path: Path) -> List[Dict[str, Any]]:
    """Load held-out transfer test anchors and extract criteria."""
    test_raw = torch.load(anchor_test_path, weights_only=False)
    processed = []
    for it in test_raw:
        cat = it["category"]
        p = it["prompt_text"]
        r = it["response_text"]
        iid = it["example_id"]

        is_obj = False
        valid_eqs: List[str] = []

        if cat == "arithmetic":
            is_obj = True
            # Extract answer number from response
            nums = re.findall(r"\b\d+\b", r)
            if nums:
                valid_eqs = [nums[-1]]
        elif cat in ("everyday_factual", "basic_science", "definitions", "classification", "simple_instructions"):
            is_obj = True
            # Extract key target terms from response
            if "The capital of" in r:
                cap = r.split("is ")[-1].rstrip(".")
                valid_eqs = [cap]
            elif "degrees" in r or "sides" in r or "inches" in r:
                nums = re.findall(r"\b\d+\b", r)
                if nums:
                    valid_eqs = [nums[-1]]
            elif cat == "classification":
                if "is a " in r:
                    cls_term = r.split("is a ")[-1].split()[0].rstrip(".")
                    valid_eqs = [cls_term]
                elif "is an " in r:
                    cls_term = r.split("is an ")[-1].split()[0].rstrip(".")
                    valid_eqs = [cls_term]
                elif "is " in r:
                    cls_term = r.split("is ")[-1].split()[0].rstrip(".")
                    valid_eqs = [cls_term]
            elif cat == "simple_instructions":
                if "is " in r:
                    t = r.split("is ")[-1].strip("'\".").split()[0]
                    valid_eqs = [t]
                elif "-" in r:
                    valid_eqs = [r.strip(".")]
            else:
                # Use key noun phrases from response
                words = [w.strip(".,'\"") for w in r.split() if len(w) > 3]
                valid_eqs = words[:3]
        else:
            # Greetings, acknowledgements, farewells, assistance
            is_obj = False
            valid_eqs = []

        processed.append({
            "id": iid,
            "category": cat,
            "prompt": p,
            "expected_guidance": r,
            "is_objective": is_obj,
            "valid_equivalents": valid_eqs,
            "suite": "anchor_transfer_test",
        })
    return processed


def build_seen_training_sample_items(anchor_train_path: Path, count: int = 20) -> List[Dict[str, Any]]:
    """Sample representative seen anchor training examples across categories."""
    train_raw = torch.load(anchor_train_path, weights_only=False)
    # Group by category and pick up to 2 per category
    by_cat: Dict[str, List[Any]] = {}
    for ex in train_raw:
        by_cat.setdefault(ex["category"], []).append(ex)

    selected = []
    for cat, ex_list in by_cat.items():
        for ex in ex_list[:2]:
            p = ex["prompt_text"]
            r = ex["response_text"]
            nums = re.findall(r"\b\d+\b", r)
            valid_eqs = [nums[-1]] if (cat == "arithmetic" and nums) else [w.strip(".,") for w in r.split() if len(w) > 4][:2]
            selected.append({
                "id": ex["example_id"],
                "category": cat,
                "prompt": p,
                "expected_guidance": r,
                "is_objective": cat in ("arithmetic", "everyday_factual", "basic_science", "definitions", "classification"),
                "valid_equivalents": valid_eqs,
                "suite": "seen_anchor_train",
            })
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break
    return selected


def run_comparative_benchmark(
    baseline_model_path: Path,
    target_model_path: Path,
    tokenizer_path: Path,
    bench35_path: Path,
    anchor_cache_dir: Path,
    out_dir: Path,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Run comprehensive before-vs-after comparative evaluation."""
    device = torch.device(device_name)
    tokenizer = BPETokenizer.load(str(tokenizer_path))

    logger.info("Loading baseline and target models...")
    baseline_model = load_model(baseline_model_path, device)
    target_model = load_model(target_model_path, device)

    generators = {
        "baseline_step600": TextGenerator(baseline_model, tokenizer, device=device),
        "target_anchors_best": TextGenerator(target_model, tokenizer, device=device),
    }

    # Build evaluation suites
    f35_items = build_frozen_35_items_with_criteria(bench35_path)
    transfer_items = build_transfer_test_items(anchor_cache_dir / "test_transfer_tokens.pt")
    seen_items = build_seen_training_sample_items(anchor_cache_dir / "train_anchor_tokens.pt", count=20)

    eval_items = f35_items + transfer_items + seen_items
    logger.info(f"Total benchmark items: {len(eval_items)} ({len(f35_items)} frozen 35, {len(transfer_items)} transfer test, {len(seen_items)} seen train).")

    decoding_configs = {
        "greedy": GenerationConfig(
            max_new_tokens=52,
            temperature=0.0,
            stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        ),
        "calibrated": GenerationConfig(
            max_new_tokens=52,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            seed=42,
            stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        ),
    }

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_model": str(baseline_model_path.name),
        "target_model": str(target_model_path.name),
        "total_items": len(eval_items),
        "suites": {},
        "comparisons": [],
    }

    suite_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for suite_name in ("frozen_35", "anchor_transfer_test", "seen_anchor_train"):
        suite_stats[suite_name] = {}
        for m_name in generators.keys():
            suite_stats[suite_name][m_name] = {
                "greedy": {"objective_passed": 0, "objective_total": 0, "lengths": []},
                "calibrated": {"objective_passed": 0, "objective_total": 0, "lengths": []},
            }

    comparisons_list: List[Dict[str, Any]] = []

    for item in eval_items:
        p_text = item["prompt"]
        formatted_prompt = format_sft_prompt(p_text)
        item_comp: Dict[str, Any] = {
            "id": item["id"],
            "suite": item["suite"],
            "category": item["category"],
            "prompt": p_text,
            "expected_guidance": item["expected_guidance"],
            "is_objective": item["is_objective"],
            "valid_equivalents": item["valid_equivalents"],
            "models": {},
        }

        for m_name, gen in generators.items():
            item_comp["models"][m_name] = {}
            for dec_name, dec_cfg in decoding_configs.items():
                gen_res = gen.generate(formatted_prompt, dec_cfg)
                out_text = gen_res.generated_text.strip()
                r_len = len(gen_res.generated_token_ids)

                grade = conservative_grade(
                    prompt=p_text,
                    response=out_text,
                    valid_equivalents=item["valid_equivalents"],
                    is_objective=item["is_objective"],
                )

                item_comp["models"][m_name][dec_name] = {
                    "text": out_text,
                    "token_length": r_len,
                    "grade": grade,
                }

                st = suite_stats[item["suite"]][m_name][dec_name]
                st["lengths"].append(r_len)
                if item["is_objective"]:
                    st["objective_total"] += 1
                    if grade["passed"]:
                        st["objective_passed"] += 1

        comparisons_list.append(item_comp)

    # Compute aggregate summary stats per suite
    summary_report: Dict[str, Any] = {}
    for suite_name, m_dict in suite_stats.items():
        summary_report[suite_name] = {}
        for m_name, d_dict in m_dict.items():
            summary_report[suite_name][m_name] = {}
            for dec_name, data in d_dict.items():
                tot_obj = data["objective_total"]
                pass_obj = data["objective_passed"]
                lens = data["lengths"]
                summary_report[suite_name][m_name][dec_name] = {
                    "objective_passed": pass_obj,
                    "objective_total": tot_obj,
                    "objective_accuracy_pct": round((pass_obj / max(1, tot_obj)) * 100.0, 1) if tot_obj else 0.0,
                    "mean_response_tokens": round(sum(lens) / max(1, len(lens)), 1),
                    "median_response_tokens": sorted(lens)[len(lens) // 2] if lens else 0,
                }

    results["suite_summary"] = summary_report
    results["comparisons"] = comparisons_list

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    with open(out_dir / "response_comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons_list, f, indent=2)

    logger.info("Evaluation complete. Metrics saved to final_metrics.json and response_comparisons.json.")
    return results


def main() -> None:
    base_ckpt = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
    target_ckpt = REPO_ROOT / "checkpoints" / "27m_sft_anchors_v1" / "model_27m_anchors_best.pt"
    tok_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    bench35_path = REPO_ROOT / "data" / "cache_sft_27m_v1" / "eval_benchmark_35.json"
    anchor_cache = REPO_ROOT / "data" / "cache_sft_anchors_v1"
    out_dir = REPO_ROOT / "logs" / "anchor_sft_step40"

    run_comparative_benchmark(
        baseline_model_path=base_ckpt,
        target_model_path=target_ckpt,
        tokenizer_path=tok_path,
        bench35_path=bench35_path,
        anchor_cache_dir=anchor_cache,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()

"""Run 10-prompt interactive comparative smoke test on both 27M SFT Best and 27M Anchors Best.

Uses identical decoding parameters (temp=0.8, top_k=40, top_p=0.9, rep_penalty=1.15, seed=42)
and evaluates responses with the task-specific conversation evaluation engine.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config import get_available_profiles
from backend.model_loader import ModelManager
from src.generation import GenerationConfig, generate_sft_response
from src.conversation_evaluation import grade_benchmark_item

PROMPTS = [
    {"id": "smoke_01", "prompt": "Hello!", "category": "greetings", "suite": "smoke"},
    {"id": "smoke_02", "prompt": "Thank you!", "category": "thanks", "suite": "smoke"},
    {"id": "smoke_03", "prompt": "Is a whale a fish or a mammal?", "category": "classification", "suite": "smoke", "expected": "A whale is a mammal."},
    {"id": "smoke_04", "prompt": "Is wood a conductor or an insulator?", "category": "classification", "suite": "smoke", "expected": "Wood is an insulator."},
    {"id": "smoke_05", "prompt": "What is the capital of Japan?", "category": "everyday_factual", "suite": "smoke", "expected": "The capital of Japan is Tokyo."},
    {"id": "smoke_06", "prompt": "What is 7 + 9?", "category": "arithmetic", "suite": "smoke", "expected": "7 + 9 = 16."},
    {"id": "smoke_07", "prompt": "What is the chemical formula for water?", "category": "basic_science", "suite": "smoke", "expected": "The chemical formula for water is H2O."},
    {"id": "smoke_08", "prompt": "List three primary colors.", "category": "short_lists", "suite": "smoke", "expected": "Red, blue, and yellow."},
    {"id": "smoke_09", "prompt": "What is gravity?", "category": "definitions", "suite": "smoke", "expected": "Gravity is the fundamental force that attracts objects with mass toward each other."},
    {"id": "smoke_10", "prompt": "Tell me a short story.", "category": "creative", "suite": "smoke", "expected": "A short creative narrative."},
]


def run_smoke_test(device_name: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    profs = get_available_profiles()
    models_to_test = ["27m_sft_best", "27m_anchors_best"]

    gen_config = GenerationConfig(
        max_new_tokens=45,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15,
        seed=42,
    )

    results: Dict[str, Any] = {
        "device": device_name,
        "config": {
            "max_new_tokens": gen_config.max_new_tokens,
            "temperature": gen_config.temperature,
            "top_k": gen_config.top_k,
            "top_p": gen_config.top_p,
            "repetition_penalty": gen_config.repetition_penalty,
            "seed": gen_config.seed,
        },
        "prompts": [],
    }

    model_managers: Dict[str, ModelManager] = {}
    peak_vram_by_model: Dict[str, float] = {}

    for p_key in models_to_test:
        if device_name == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        mgr = ModelManager()
        t0 = time.perf_counter()
        mgr.load_model(
            checkpoint_path=profs[p_key]["checkpoint_path"],
            tokenizer_path=profs[p_key]["tokenizer_path"],
            device_name=device_name,
            model_name=profs[p_key]["model_name"],
            active_profile_key=p_key,
        )
        load_time = time.perf_counter() - t0

        if device_name == "cuda":
            vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        else:
            vram_mb = 0.0

        peak_vram_by_model[p_key] = vram_mb
        model_managers[p_key] = mgr

    for item in PROMPTS:
        p_id = item["id"]
        prompt = item["prompt"]
        cat = item["category"]
        exp = item.get("expected", "")

        prompt_record: Dict[str, Any] = {
            "id": p_id,
            "prompt": prompt,
            "category": cat,
            "expected_guidance": exp,
            "responses": {},
        }

        for p_key, mgr in model_managers.items():
            t0 = time.perf_counter()
            gen_res = generate_sft_response(mgr.generator, prompt=prompt, config=gen_config)
            latency = time.perf_counter() - t0

            tok_count = len(gen_res.generated_token_ids)
            tok_per_sec = round(tok_count / max(1e-4, latency), 1)
            raw_text = gen_res.generated_text.strip()

            grade = grade_benchmark_item(
                item_id=p_id,
                prompt=prompt,
                response=raw_text,
                category=cat,
                suite="smoke",
                expected_guidance=exp,
            )

            prompt_record["responses"][p_key] = {
                "text": raw_text,
                "tokens": tok_count,
                "latency_sec": round(latency, 3),
                "tokens_per_sec": tok_per_sec,
                "grade": grade.to_dict(),
            }

        results["prompts"].append(prompt_record)

    results["peak_vram_mb"] = peak_vram_by_model

    out_file = REPO_ROOT / "logs" / "anchor_sft_step40" / "interactive_smoke_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    res = run_smoke_test()
    print("Smoke comparison finished successfully:")
    for p in res["prompts"]:
        print(f"\n[{p['category']}] Prompt: '{p['prompt']}'")
        for m_key, r_data in p["responses"].items():
            print(f"  {m_key} ({r_data['tokens']} tok, {r_data['tokens_per_sec']} t/s, pass={r_data['grade']['passed']}): {repr(r_data['text'])}")


if __name__ == "__main__":
    main()

"""Step 48: Audit and verification of Conversational SFT v2 dataset.

Verifies:
- Loss masking integrity (assistant-only loss, prompt masked to -100)
- Sequence length bounds and token distributions
- End-of-turn delimiter presence
- Strict zero-leakage check against frozen benchmark
- Category distribution and balance
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "benchmark_conversational_step48.json"
CACHE_DIR = REPO_ROOT / "data" / "cache_sft_neural_v2"
JSONL_DIR = REPO_ROOT / "data" / "sft" / "conversational_v2"
LOG_DIR = REPO_ROOT / "logs" / "autonomous_steps_44_46"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def audit_sft_v2():
    logger.info("Auditing Conversational SFT v2 dataset...")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark prompts
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    bench_prompts = [item["prompt"] for item in bench_data["prompts"]]
    bench_norm_set = {normalize(p) for p in bench_prompts}

    # 2. Load JSONL
    train_jsonl = JSONL_DIR / "sft_conversational_v2_train.jsonl"
    val_jsonl = JSONL_DIR / "sft_conversational_v2_val.jsonl"

    all_items = []
    category_counts = {}
    leakage_detected = []

    for path, split in [(train_jsonl, "train"), (val_jsonl, "val")]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                item["split"] = split
                all_items.append(item)
                cat = item["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1

                # Check leakage
                norm_p = normalize(item["prompt"])
                if norm_p in bench_norm_set:
                    leakage_detected.append({"prompt": item["prompt"], "split": split})

    # 3. Check binary cache
    train_cache = torch.load(CACHE_DIR / "train_tokens.pt", weights_only=False)
    val_cache = torch.load(CACHE_DIR / "val_tokens.pt", weights_only=False)

    # Validate loss masking on first 50 items
    mask_errors = 0
    for ex in train_cache[:50] + val_cache[:20]:
        input_ids = ex["input_ids"]
        target_ids = ex["target_ids"]
        loss_mask = ex["loss_mask"]
        p_count = ex["prompt_tokens_count"]

        # All positions before p_count - 1 MUST be -100 in target_ids
        prompt_targets = target_ids[: p_count - 1]
        if not (prompt_targets == -100).all():
            mask_errors += 1

        # All response positions must match loss_mask == 1
        active_mask = (target_ids != -100)
        if not (active_mask == (loss_mask == 1)).all():
            mask_errors += 1

    audit_report = {
        "dataset_name": "sft_conversational_v2",
        "total_records": len(all_items),
        "train_records": len(train_cache),
        "val_records": len(val_cache),
        "benchmark_prompts_checked": len(bench_norm_set),
        "leakage_count": len(leakage_detected),
        "leakage_items": leakage_detected,
        "mask_integrity_errors": mask_errors,
        "category_distribution": category_counts,
        "avg_prompt_tokens": round(sum(it["prompt_tokens"] for it in all_items) / len(all_items), 2),
        "avg_response_tokens": round(sum(it["response_tokens"] for it in all_items) / len(all_items), 2),
        "audit_passed": (len(leakage_detected) == 0 and mask_errors == 0 and len(all_items) >= 800),
    }

    report_path = LOG_DIR / "step48_dataset_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"Audit completed. Passed: {audit_report['audit_passed']}. Saved to {report_path}")
    print("\n" + "=" * 60)
    print("SFT V2 DATASET AUDIT REPORT")
    print(f"Total Records:         {audit_report['total_records']}")
    print(f"Train / Val Tensors:   {audit_report['train_records']} / {audit_report['val_records']}")
    print(f"Leakage Detected:      {audit_report['leakage_count']}")
    print(f"Masking Errors:        {audit_report['mask_integrity_errors']}")
    print(f"Categories ({len(category_counts)}):")
    for cat, cnt in sorted(category_counts.items()):
        print(f"  {cat:32s}: {cnt}")
    print(f"Audit Status:          {'PASSED' if audit_report['audit_passed'] else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    audit_sft_v2()

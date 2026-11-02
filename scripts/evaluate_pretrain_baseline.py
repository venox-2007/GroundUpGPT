"""Pretraining Baseline Evaluation for GroundUpGPT-27M (Step 49).

Evaluates the unadapted Step 5949 checkpoint (checkpoints/27m_phase4/model_27m_best.pt)
BEFORE any Step 49 training begins across all validation tracks:
1. Track A: Original Combined Pretraining Validation (data/cache_27m_v1/val_tokens.pt)
2. Track B: Original Gutenberg Literary Validation
3. Track C: Original Simple Wikipedia Validation
4. Track D: New Expansion Validation (data/cache_pretrain_v2/val_tokens.pt)

Outputs:
  logs/pretraining_step49/baseline_validation.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.amp import autocast

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_baseline")

INSPECTION_PROMPTS = [
    # General / narrative
    "The sun rose over the",
    "Once upon a time in a small village,",
    "He opened the mysterious door and saw",
    # Science & Nature
    "The Earth revolves around the",
    "Water is composed of",
    "Photosynthesis is the process by which",
    "Gravity is the force that",
    # Geography & Facts
    "The capital of France is",
    "The Atlantic Ocean lies between",
    "Mount Everest is the highest",
    # Definitions
    "Biology is the study of",
    "A computer algorithm is a",
    "An atom is the smallest unit of",
]


def evaluate_windows(
    model: GroundUpGPT,
    tokens: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float]:
    """Evaluate mean cross-entropy loss and perplexity on deterministic token windows."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack([tokens[j : j + block_size] for j in batch_idx]).to(device, dtype=torch.long)
            y = torch.stack([tokens[j + 1 : j + block_size + 1] for j in batch_idx]).to(device, dtype=torch.long)
            with autocast(device_type="cuda" if use_cuda else "cpu", enabled=use_cuda):
                _, loss = model(x, y)
            total_loss += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

    mean_loss = total_loss / max(1, total_samples)
    ppl = math.exp(min(20.0, mean_loss))
    return round(mean_loss, 4), round(ppl, 2)


def generate_continuations(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    prompts: List[str],
    device: torch.device,
) -> List[Dict[str, str]]:
    generator = TextGenerator(model, tokenizer=tokenizer, device=device)
    cfg = GenerationConfig(
        max_new_tokens=36,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.1,
        seed=42,
    )
    samples = []
    for p in prompts:
        out = generator.generate(p, cfg)
        text = out.generated_text if hasattr(out, "generated_text") else str(out)
        samples.append({"prompt": p, "completion": text})
    return samples


def run_baseline_evaluation(
    checkpoint_path: Path,
    tokenizer_path: Path,
    original_val_path: Path,
    new_val_path: Path,
    output_path: Path,
    block_size: int = 256,
    num_windows: int = 250,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for baseline evaluation...")

    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    source_step = ckpt.get("step", 5949)
    source_sha256 = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
    logger.info(f"Loaded base model: {checkpoint_path.name} (step={source_step}, sha256={source_sha256[:16]}...)")

    # 2. Load Tokenizer
    tokenizer = BPETokenizer().load(str(tokenizer_path))

    # 3. Load Validation Tensors
    orig_val_tokens = torch.load(original_val_path, weights_only=False)
    new_val_tokens = torch.load(new_val_path, weights_only=False)

    logger.info(f"Original val tokens: {len(orig_val_tokens):,}")
    logger.info(f"New val tokens:      {len(new_val_tokens):,}")

    # Track B (Gutenberg) vs Track C (Wiki) boundary in original val
    # From cache_metadata.json: Gutenberg val is first 1,118,258 tokens
    gutenberg_cutoff = 1118258
    orig_gutenberg_tokens = orig_val_tokens[:gutenberg_cutoff]
    orig_wiki_tokens = orig_val_tokens[gutenberg_cutoff:]

    # Deterministic evaluation window indices
    def get_eval_indices(t_len: int, n_win: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        max_idx = max(0, t_len - block_size - 1)
        k = min(n_win, max_idx)
        return torch.randint(0, max_idx, (k,), generator=g)

    idx_orig_all = get_eval_indices(len(orig_val_tokens), num_windows, seed=1001)
    idx_orig_gut = get_eval_indices(len(orig_gutenberg_tokens), num_windows, seed=1002)
    idx_orig_wiki = get_eval_indices(len(orig_wiki_tokens), num_windows, seed=1003)
    idx_new_val = get_eval_indices(len(new_val_tokens), num_windows, seed=1004)

    # 4. Evaluate Tracks
    logger.info("Evaluating Track A (Original Combined Val)...")
    loss_orig_all, ppl_orig_all = evaluate_windows(model, orig_val_tokens, idx_orig_all, block_size, device=device)

    logger.info("Evaluating Track B (Original Gutenberg Val)...")
    loss_orig_gut, ppl_orig_gut = evaluate_windows(model, orig_gutenberg_tokens, idx_orig_gut, block_size, device=device)

    logger.info("Evaluating Track C (Original Simple Wikipedia Val)...")
    loss_orig_wiki, ppl_orig_wiki = evaluate_windows(model, orig_wiki_tokens, idx_orig_wiki, block_size, device=device)

    logger.info("Evaluating Track D (New Expanded Simple Wikipedia Val)...")
    loss_new_val, ppl_new_val = evaluate_windows(model, new_val_tokens, idx_new_val, block_size, device=device)

    # 5. Qualitative Continuations
    logger.info("Generating qualitative continuations on base checkpoint...")
    continuations = generate_continuations(model, tokenizer, INSPECTION_PROMPTS, device=device)

    baseline_data = {
        "evaluation_name": "pretraining_v2_baseline_step5949",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": source_sha256,
        "checkpoint_step": source_step,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "block_size": block_size,
        "eval_windows_per_track": num_windows,
        "metrics": {
            "track_A_original_combined": {
                "description": "Original Combined Pretraining Validation (cache_27m_v1)",
                "loss": loss_orig_all,
                "perplexity": ppl_orig_all,
                "historical_step34_loss": 3.5768,
                "historical_step34_ppl": 35.76,
            },
            "track_B_original_gutenberg": {
                "description": "Original Gutenberg Literary Held-Out Books",
                "loss": loss_orig_gut,
                "perplexity": ppl_orig_gut,
                "historical_step34_loss": 4.4824,
            },
            "track_C_original_simplewiki": {
                "description": "Original Simple English Wikipedia Held-Out Articles",
                "loss": loss_orig_wiki,
                "perplexity": ppl_orig_wiki,
                "historical_step34_loss": 2.6171,
            },
            "track_D_new_expanded_val": {
                "description": "New Expanded Pretraining Validation (cache_pretrain_v2)",
                "loss": loss_new_val,
                "perplexity": ppl_new_val,
            },
            "composite_baseline_loss": round((loss_orig_all + loss_new_val) / 2.0, 4),
        },
        "inspection_continuations": continuations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)

    logger.info(f"Baseline evaluation saved to {output_path}")

    print("\n" + "=" * 70)
    print(" BASELINE PRETRAINING VALIDATION METRICS (STEP 5949)")
    print("=" * 70)
    print(f"Checkpoint:                {checkpoint_path.name} (Step {source_step})")
    print(f"Checkpoint SHA-256:        {source_sha256[:16]}...")
    print(f"Track A (Orig Combined):   Loss: {loss_orig_all:.4f} | PPL: {ppl_orig_all:.2f}")
    print(f"Track B (Orig Gutenberg):  Loss: {loss_orig_gut:.4f} | PPL: {ppl_orig_gut:.2f}")
    print(f"Track C (Orig SimpleWiki): Loss: {loss_orig_wiki:.4f} | PPL: {ppl_orig_wiki:.2f}")
    print(f"Track D (New Expansion):   Loss: {loss_new_val:.4f} | PPL: {ppl_new_val:.2f}")
    print(f"Composite Baseline Loss:   {baseline_data['metrics']['composite_baseline_loss']:.4f}")
    print("-" * 70)
    print("Sample Continuations:")
    for s in continuations[:4]:
        print(f"  [{s['prompt']}] -> {repr(s['completion'])}")
    print("=" * 70 + "\n")

    return baseline_data


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation for GroundUpGPT-27M (Step 49)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "27m_phase4" / "model_27m_best.pt",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json",
    )
    parser.add_argument(
        "--orig-val",
        type=Path,
        default=REPO_ROOT / "data" / "cache_27m_v1" / "val_tokens.pt",
    )
    parser.add_argument(
        "--new-val",
        type=Path,
        default=REPO_ROOT / "data" / "cache_pretrain_v2" / "val_tokens.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "logs" / "pretraining_step49" / "baseline_validation.json",
    )
    parser.add_argument("--num-windows", type=int, default=250)
    args = parser.parse_args()

    run_baseline_evaluation(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        original_val_path=args.orig_val,
        new_val_path=args.new_val,
        output_path=args.output,
        num_windows=args.num_windows,
    )


if __name__ == "__main__":
    main()

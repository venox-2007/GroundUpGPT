"""CLI script to fine-tune GroundUpGPT 5M model on Databricks Dolly 15k instructions.

Usage:
    python scripts/train_sft_5m.py [--max-steps 650] [--max-seconds 580] [--lr 1.5e-4]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, generate_sft_response
from src.model import GPTConfig, GroundUpGPT
from src.sft_trainer import SFTTrainer, SFTTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fixed unseen conversational evaluation prompts
BENCHMARK_PROMPTS = [
    {
        "category": "Greeting",
        "prompt": "Hello! How are you?",
        "context": "",
    },
    {
        "category": "Factual QA",
        "prompt": "What is the capital of France?",
        "context": "",
    },
    {
        "category": "Classification",
        "prompt": "Identify which of the following is a fruit: carrot, apple, potato",
        "context": "",
    },
    {
        "category": "Explanation",
        "prompt": "Why is the sky blue?",
        "context": "",
    },
    {
        "category": "List Generation",
        "prompt": "List three primary colors.",
        "context": "",
    },
    {
        "category": "Context QA",
        "prompt": "According to the text, when was the company founded?",
        "context": "Acme Corp was established in 1985 as an industrial equipment supplier.",
    },
]


def generate_benchmark_responses(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    device: str,
    max_new_tokens: int = 60,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Generate responses for fixed benchmark prompts using fixed seed and decoding settings."""
    model.eval()
    generator = TextGenerator(model, tokenizer, device=device)
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        seed=seed,
    )

    results = []
    for item in BENCHMARK_PROMPTS:
        cat = item["category"]
        prompt = item["prompt"]
        context = item["context"]
        gen_res = generate_sft_response(
            generator=generator,
            prompt=prompt,
            context=context if context else None,
            config=cfg,
        )
        results.append(
            {
                "category": cat,
                "prompt": prompt,
                "context": context,
                "response": gen_res.generated_text,
                "tokens_generated": gen_res.generated_tokens,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GroundUpGPT 5M Supervised Instruction Fine-Tuning"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/5m_phase2/model_5m_best.pt"),
        help="Path to source Phase 2 Step 3000 checkpoint",
    )
    parser.add_argument(
        "--sft-cache",
        type=Path,
        default=Path("data/cache_sft"),
        help="Path to pre-tokenized SFT dataset cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/5m_sft"),
        help="Path to store SFT checkpoints",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/bpe_vocab4096.json"),
        help="Path to BPE tokenizer",
    )
    parser.add_argument(
        "--lm-val-path",
        type=Path,
        default=Path("data/cache_expanded/val_tokens.pt"),
        help="Path to expanded validation tokens for LM benchmark",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=650,
        help="Maximum optimizer steps to train (target 500-800)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1.5e-5,
        help="Minimum learning rate at end of cosine decay",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=16,
        help="Micro-batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch: micro_batch * grad_accum)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=50,
        help="Steps between validations",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=580.0,
        help="Mandatory wall-clock limit in seconds (max 600s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Preflight Check
    if not args.base_checkpoint.is_file():
        logger.error(f"Base checkpoint missing: {args.base_checkpoint}")
        sys.exit(1)
    if not (args.sft_cache / "train_sft_tokens.pt").is_file():
        logger.error(f"SFT cache missing in {args.sft_cache}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(args.tokenizer)

    # 1. Establish Pre-SFT Baseline Responses
    logger.info("Generating pre-SFT baseline conversational responses...")
    base_ckpt = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    base_model = GroundUpGPT(GPTConfig(**base_ckpt["model_config"]))
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.to(device)

    pre_sft_responses = generate_benchmark_responses(
        base_model, tok, device=device
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Configure and Run SFT Training
    train_cfg = SFTTrainingConfig(
        base_checkpoint_path=args.base_checkpoint,
        sft_cache_dir=args.sft_cache,
        checkpoint_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        lm_val_path=args.lm_val_path,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        val_interval=args.val_interval,
        max_train_seconds=args.max_seconds,
        seed=args.seed,
        device=device,
    )

    trainer = SFTTrainer(train_cfg, tokenizer=tok)
    train_results = trainer.train()

    # 3. Generate Post-SFT Benchmark Responses from Best Checkpoint
    logger.info("Generating post-SFT responses from best checkpoint...")
    best_ckpt = torch.load(
        train_results["best_checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    sft_model = GroundUpGPT(GPTConfig(**best_ckpt["model_config"]))
    sft_model.load_state_dict(best_ckpt["model_state_dict"])
    sft_model.to(device)

    post_sft_responses = generate_benchmark_responses(
        sft_model, tok, device=device
    )

    # 4. Print Side-by-Side Comparison and Final Report
    print("\n" + "=" * 80)
    print("GROUNDUPGPT STEP 27 — SUPERVISED FINE-TUNING REPORT")
    print("=" * 80)
    print(f"Source Checkpoint:       {args.base_checkpoint} (Step {train_results['source_step']})")
    print(f"Tokenizer Fingerprint:   {tok.get_fingerprint()}")
    print(f"Device:                  {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"SFT Steps Completed:     {train_results['sft_steps']} / {args.max_steps}")
    print(f"Total GPU Wall Time:     {train_results['total_elapsed_seconds']:.2f} s ({train_results['total_elapsed_seconds']/60:.2f} mins)")
    print(f"Tokens Processed:        {train_results['tokens_processed']:,} active response tokens")
    print(f"Stop Reason:             {train_results['stop_reason']}")
    print("-" * 80)
    print("VALIDATION METRICS COMPARISON:")
    print(f"  Pre-SFT Baseline (Step 3000):")
    print(f"    - SFT Validation Loss:         {train_results['pre_sft_baseline']['sft_val_loss']:.4f}")
    print(f"    - SFT Validation Perplexity:   {train_results['pre_sft_baseline']['sft_val_ppl']:.2f}")
    print(f"    - Frozen LM Benchmark Loss:    {train_results['pre_sft_baseline']['lm_val_loss']:.4f}")
    print(f"    - Frozen LM Benchmark PPL:     {train_results['pre_sft_baseline']['lm_val_ppl']:.2f}")
    print(f"  Post-SFT Fine-Tuned (Best):")
    print(f"    - SFT Validation Loss:         {train_results['post_sft_metrics']['best_sft_val_loss']:.4f}")
    print(f"    - SFT Validation Perplexity:   {train_results['post_sft_metrics']['best_sft_val_ppl']:.2f}")
    print(f"    - Frozen LM Benchmark Loss:    {train_results['post_sft_metrics']['final_lm_val_loss']:.4f}")
    print(f"    - Frozen LM Benchmark PPL:     {train_results['post_sft_metrics']['final_lm_val_ppl']:.2f}")
    print("-" * 80)
    print("CHECKPOINT ARTIFACTS:")
    print(f"  Best Checkpoint:    {train_results['best_checkpoint_path']}")
    print(f"  Latest Checkpoint:  {train_results['latest_checkpoint_path']}")
    print("-" * 80)
    print("SIDE-BY-SIDE CONVERSATIONAL COMPARISON (BASE vs SFT):")

    for pre, post in zip(pre_sft_responses, post_sft_responses):
        cat = pre["category"]
        prompt = pre["prompt"]
        ctx = pre["context"]
        print(f"\n--- [{cat}] ---")
        print(f"User Prompt: {prompt}")
        if ctx:
            print(f"Context:     {ctx}")
        print(f"Base Output: {repr(pre['response'])}")
        print(f"SFT Output:  {repr(post['response'])}")

    print("=" * 80 + "\n")

    # Save summary report to JSON
    report_file = args.output_dir / "sft_training_summary.json"
    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_results": train_results,
        "pre_sft_responses": pre_sft_responses,
        "post_sft_responses": post_sft_responses,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Summary written to {report_file}")


if __name__ == "__main__":
    main()

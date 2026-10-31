"""Initial training experiment for the 5.05M parameter GroundUpGPT model.

Trains the full GroundUpGPT Transformer from scratch on the real Project Gutenberg corpus
using the 4,096-vocab production BPE tokenizer and GPU acceleration.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator
from src.model import GPTConfig, GroundUpGPT
from src.token_cache import load_token_cache
from src.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_5m_gpt")

TEST_PROMPTS = [
    "The meaning of life is",
    "Once upon a time in a distant land",
    "The scientific method requires",
]


def run_sampling(model: GroundUpGPT, tokenizer: BPETokenizer, tag: str) -> list[dict[str, str]]:
    """Generate sample completions using fixed prompts and settings for before/after comparison."""
    generator = TextGenerator(model=model, tokenizer=tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=40,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        seed=42,
    )
    samples = []
    print(f"\n--- Model Generation Samples ({tag}) ---")
    for prompt in TEST_PROMPTS:
        result = generator.generate(prompt=prompt, config=gen_config)
        clean_text = result.generated_text.replace("\n", "\\n")
        safe_repr = clean_text.encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"[{tag}] Prompt: '{prompt}'")
        print(f"[{tag}] Output: '{safe_repr}'\n")
        samples.append({
            "prompt": prompt,
            "generated_text": result.generated_text,
            "full_text": result.full_text,
            "tag": tag,
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 5.05M GroundUpGPT model from scratch.")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimizer steps to execute (default: 200).")
    parser.add_argument("--global-steps", type=int, default=1000, help="Planned total schedule steps (default: 1000).")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size (default: 16).")
    parser.add_argument("--block-size", type=int, default=256, help="Context length (default: 256).")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate (default: 6e-4).")
    parser.add_argument("--time-budget", type=float, default=1200.0, help="Max wall-clock budget in seconds (default: 1200).")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints" / "5m_initial", help="Checkpoint directory.")
    parser.add_argument("--logs-dir", type=Path, default=REPO_ROOT / "logs", help="Directory for metric summaries.")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / "checkpoints" / "bpe_vocab4096.json", help="Tokenizer JSON path.")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "cache", help="Token cache directory.")
    parser.add_argument("--resume-step", type=int, default=None, help="Optional step checkpoint to resume from.")

    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" GroundUpGPT — 5.05M Architecture Training Pipeline")
    print("=" * 70)

    # 1. Load Tokenizer & Token Cache
    if not args.tokenizer_path.is_file():
        logger.error(f"Tokenizer not found at {args.tokenizer_path}. Run Step 15 first.")
        sys.exit(1)
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    print(f" Loaded Tokenizer: {args.tokenizer_path.name} (Vocab size: {tokenizer.vocab_size})")

    pipeline = load_token_cache(args.cache_dir, tokenizer=tokenizer)
    print(f" Loaded Token Cache: {len(pipeline.train_tokens):,} train tokens, {len(pipeline.val_tokens):,} val tokens")

    # 2. Build 5.05M Model Architecture
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embed_dim=256,
        num_heads=4,
        num_layers=5,
        dropout=0.1,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(gpt_config)
    num_params = model.get_num_params(non_embedding=False)
    print(f" Model Parameter Count: {num_params:,} (Target: 5,051,904)")
    assert num_params == 5051904, f"Parameter count mismatch: {num_params} != 5051904"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Target Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 3. Before-Training Samples
    before_samples = run_sampling(model, tokenizer, "BEFORE")

    # 4. Configure Trainer
    train_config = TrainingConfig(
        max_steps=args.global_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        min_lr=args.lr * 0.1,
        warmup_steps=20,
        weight_decay=0.1,
        grad_clip=1.0,
        grad_accum_steps=args.accum_steps,
        eval_interval=25,
        eval_iters=10,
        checkpoint_interval=50,
        checkpoint_dir=str(args.checkpoint_dir),
        device=device,
        seed=42,
    )

    trainer = Trainer(model=model, pipeline=pipeline, config=train_config)

    # Optional resume
    if args.resume_step is not None:
        resume_ckpt = args.checkpoint_dir / f"step_{args.resume_step}.pt"
        print(f"[*] Resuming from checkpoint: {resume_ckpt}")
        trainer.load_checkpoint(resume_ckpt)

    # 5. Run Bounded Training Segment
    print(f"\n[*] Commencing training run (bounded to {args.steps} steps, budget {args.time_budget}s)...")
    results = trainer.train(
        stop_after_steps=args.steps,
        time_budget_seconds=args.time_budget,
    )

    # 6. After-Training Samples
    after_samples = run_sampling(model, tokenizer, "AFTER")

    # 7. Save Summary and History
    import dataclasses
    summary = {
        "model_parameters": num_params,
        "config": dataclasses.asdict(gpt_config),
        "training_config": dataclasses.asdict(train_config),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "completed_steps": results["total_steps"],
        "tokens_processed": results["tokens_processed"],
        "tokens_per_second": results["tokens_per_second"],
        "total_time_seconds": results["total_time_seconds"],
        "initial_train_loss": results["initial_train_loss"],
        "initial_val_loss": results["initial_val_loss"],
        "final_train_loss": results["final_train_loss"],
        "final_val_loss": results["final_val_loss"],
        "final_checkpoint": results["final_checkpoint"],
        "device": results["device"],
        "mixed_precision": results["mixed_precision"],
        "samples_before": before_samples,
        "samples_after": after_samples,
        "history": results["history"],
    }

    summary_file = args.logs_dir / "training_5m_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save CSV history
    csv_file = args.logs_dir / "training_5m_history.csv"
    if results["history"]:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results["history"][0].keys()))
            writer.writeheader()
            writer.writerows(results["history"])

    # Create convenient alias checkpoint
    alias_ckpt = args.checkpoint_dir / f"model_5m_step{results['total_steps']}.pt"
    if Path(results["final_checkpoint"]).exists():
        torch.save(torch.load(results["final_checkpoint"], weights_only=False), alias_ckpt)
        print(f"[*] Saved primary 5M model checkpoint to: {alias_ckpt}")

    print("\n" + "=" * 70)
    print(" Training Experiment Succeeded!")
    print("=" * 70)
    print(f" Completed Steps:      {results['total_steps']}")
    print(f" Processed Tokens:     {results['tokens_processed']:,}")
    print(f" Processing Speed:     {results['tokens_per_second']:,} tokens/sec")
    print(f" Elapsed Time:         {results['total_time_seconds']:.2f}s")
    print(f" Initial Loss (Train): {results['initial_train_loss']:.4f} | Val: {results['initial_val_loss']:.4f}")
    print(f" Final Loss (Train):   {results['final_train_loss']:.4f} | Val: {results['final_val_loss']:.4f}")
    print(f" Checkpoint Path:      {alias_ckpt}")
    print(f" Metrics Summary:      {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

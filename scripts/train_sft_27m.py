"""CLI script to fine-tune GroundUpGPT-27M on conversational instruction data (Step 37).

Executes bounded supervised fine-tuning (SFT) initialized from Step 5949 pretrained weights
(checkpoints/27m_phase4/model_27m_best.pt), monitoring held-out SFT validation loss and
evaluating pre/post conversational quality on a 35-prompt benchmark.

Usage:
    .venv/Scripts/python scripts/train_sft_27m.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    GenerationConfig,
    TextGenerator,
    format_sft_prompt,
    generate_sft_response,
)
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import SFTDataPipeline
from src.sft_trainer import (
    EXPECTED_TOKENIZER_FINGERPRINT,
    SFTTrainer,
    SFTTrainingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPECTED_BASE_SHA256 = (
    "3e2f62dd9d53bf8a2dffeb91d92f362d6873163c57429084b6a39bb4dcb0baa1"
)


def compute_sha256(file_path: Path) -> str:
    """Compute exact SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_isolated_smoke_test(device: str) -> None:
    """Run an isolated smoke test with a dummy miniature model to verify CUDA, AMP, and backward."""
    logger.info("[*] Running isolated preflight smoke test on disposable dummy model...")
    cfg = GPTConfig(
        vocab_size=4096,
        block_size=32,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
    )
    dummy_model = GroundUpGPT(cfg).to(device)
    dummy_model.train()
    optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device))

    x = torch.randint(0, 4096, (2, 16), device=device)
    y = torch.randint(0, 4096, (2, 16), device=device)
    y[:, :4] = -100

    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=("cuda" in device)):
        logits, _ = dummy_model(x)
        loss = F.cross_entropy(logits.view(-1, 4096), y.view(-1), ignore_index=-100)

    assert torch.isfinite(loss), "Smoke test loss is non-finite!"
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    del dummy_model, optimizer, scaler
    if "cuda" in device:
        torch.cuda.empty_cache()
    logger.info("[*] Preflight smoke test passed successfully. Environment verified.")


def generate_benchmark_suite(
    model: GroundUpGPT,
    tokenizer: BPETokenizer,
    benchmark_items: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int = 60,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate both greedy and stochastic sampling responses for all benchmark prompts."""
    model.eval()
    generator = TextGenerator(model, tokenizer, device=device)

    greedy_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )
    sampling_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=42,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )

    greedy_results = []
    sampling_results = []

    logger.info(f"Generating benchmark responses for {len(benchmark_items)} prompts...")
    t0 = time.time()
    for item in benchmark_items:
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        guidance = item.get("expected_response_guidance", "")

        # 1. Greedy argmax
        g_res = generate_sft_response(
            generator=generator,
            prompt=prompt,
            config=greedy_cfg,
        )
        greedy_results.append(
            {
                "id": p_id,
                "category": cat,
                "prompt": prompt,
                "response": g_res.generated_text,
                "tokens_generated": g_res.generated_tokens,
                "guidance": guidance,
            }
        )

        # 2. Stochastic sampling
        s_res = generate_sft_response(
            generator=generator,
            prompt=prompt,
            config=sampling_cfg,
        )
        sampling_results.append(
            {
                "id": p_id,
                "category": cat,
                "prompt": prompt,
                "response": s_res.generated_text,
                "tokens_generated": s_res.generated_tokens,
                "guidance": guidance,
            }
        )

    logger.info(f"Generated {len(benchmark_items)*2} responses in {time.time()-t0:.2f}s.")
    return {
        "greedy": greedy_results,
        "sampling": sampling_results,
    }


def grade_response(prompt: str, response: str, category: str) -> Dict[str, Any]:
    """Heuristic assessment of response quality, formatting, and repetition."""
    resp = response.strip()
    words = resp.split()

    # Repetition check (2-gram repetition)
    if len(words) >= 4:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        counts = Counter(bigrams)
        max_rep = max(counts.values()) if counts else 1
        rep_ratio = round(max_rep / len(bigrams), 2)
    else:
        rep_ratio = 0.0

    has_leaked_delimiters = any(d in resp for d in ["<|user|>", "<|assistant|>", "<|endoftext|>"])
    non_empty = len(resp) > 0

    return {
        "length_chars": len(resp),
        "word_count": len(words),
        "repetition_ratio": rep_ratio,
        "clean_stopping": not has_leaked_delimiters,
        "non_empty": non_empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GroundUpGPT-27M Supervised Instruction Fine-Tuning"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/27m_phase4/model_27m_best.pt"),
        help="Path to source Phase 4 Step 5949 checkpoint",
    )
    parser.add_argument(
        "--sft-cache",
        type=Path,
        default=Path("data/cache_sft_27m_v1"),
        help="Path to pre-tokenized SFT dataset cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/27m_sft_phase1"),
        help="Path to store SFT checkpoints",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/bpe_vocab4096.json"),
        help="Path to BPE tokenizer",
    )
    parser.add_argument(
        "--lm-track-a",
        type=Path,
        default=Path("data/cache_27m_v1/val_tokens.pt"),
        help="Path to pretraining validation tokens for Track A",
    )
    parser.add_argument(
        "--benchmark-prompts",
        type=Path,
        default=Path("data/cache_sft_27m_v1/eval_benchmark_35.json"),
        help="Path to 35-prompt conversational benchmark",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/training_27m_sft_phase1.json"),
        help="Path to output telemetry log",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum optimizer steps (target 600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=5e-6,
        help="Minimum learning rate at end of cosine decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=30,
        help="Linear warmup steps",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Micro-batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch: micro_batch * grad_accum = 16)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=100,
        help="Steps between validations",
    )
    parser.add_argument(
        "--eval-windows",
        type=int,
        default=48,
        help="Number of evaluation windows per LM track",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=580.0,
        help="Mandatory wall-clock limit in seconds (strict boundary <600s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # 1. Strict Preflight
    logger.info("=" * 70)
    logger.info("GroundUpGPT-27M Step 37 — SFT Experiment Preflight")
    logger.info("=" * 70)

    if not args.base_checkpoint.is_file():
        logger.error(f"Base checkpoint missing: {args.base_checkpoint}")
        sys.exit(1)
    base_sha = compute_sha256(args.base_checkpoint)
    logger.info(f"Source Checkpoint: {args.base_checkpoint}")
    logger.info(f"SHA-256:          {base_sha}")
    if base_sha != EXPECTED_BASE_SHA256:
        logger.error(f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_BASE_SHA256}, got {base_sha}")
        sys.exit(1)

    if not args.tokenizer.is_file():
        logger.error(f"Tokenizer missing: {args.tokenizer}")
        sys.exit(1)
    tok = BPETokenizer.load(args.tokenizer)
    tok_fp = tok.get_fingerprint()
    logger.info(f"Tokenizer Fingerprint: {tok_fp}")
    if tok_fp != EXPECTED_TOKENIZER_FINGERPRINT:
        logger.error(f"Tokenizer fingerprint mismatch! Expected {EXPECTED_TOKENIZER_FINGERPRINT}")
        sys.exit(1)

    train_tokens_path = args.sft_cache / "train_sft_tokens.pt"
    val_tokens_path = args.sft_cache / "val_sft_tokens.pt"
    if not train_tokens_path.is_file() or not val_tokens_path.is_file():
        logger.error(f"SFT cache files missing in {args.sft_cache}")
        sys.exit(1)

    if not args.benchmark_prompts.is_file():
        logger.error(f"Benchmark file missing: {args.benchmark_prompts}")
        sys.exit(1)

    with open(args.benchmark_prompts, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    benchmark_items = bench_data.get("items", [])
    logger.info(f"Loaded {len(benchmark_items)} benchmark prompts from {args.benchmark_prompts}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    logger.info(f"Compute Device: {device.upper()} ({device_name})")

    # 2. Isolated Preflight Smoke Test
    run_isolated_smoke_test(device)

    # 3. Pre-SFT Step 0 Baseline Evaluation
    logger.info("-" * 70)
    logger.info("Computing Step 0 Pre-SFT Untouched Baseline...")
    logger.info("-" * 70)

    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    m_cfg = GPTConfig(**base_ckpt["model_config"])
    base_model = GroundUpGPT(m_cfg)
    base_model.load_state_dict(base_ckpt["model_state_dict"], strict=True)
    base_model.to(device)

    # Generate baseline responses for all 35 prompts (both greedy and sampling)
    baseline_benchmark_results = generate_benchmark_suite(
        base_model, tok, benchmark_items, device=device
    )

    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 4. SFT Training
    logger.info("-" * 70)
    logger.info("Configuring SFT Trainer for 27M Parameters...")
    logger.info("-" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    train_cfg = SFTTrainingConfig(
        base_checkpoint_path=args.base_checkpoint,
        sft_cache_dir=args.sft_cache,
        checkpoint_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        lm_track_A_path=args.lm_track_a,
        best_checkpoint_filename="model_27m_sft_best.pt",
        latest_checkpoint_filename="model_27m_sft_latest.pt",
        expected_param_count=27_411_456,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        val_interval=args.val_interval,
        eval_windows=args.eval_windows,
        max_train_seconds=args.max_seconds,
        seed=args.seed,
        device=device,
    )

    trainer = SFTTrainer(train_cfg, tokenizer=tok)
    train_results = trainer.train()

    # 5. Post-SFT Evaluation on Best Checkpoint
    logger.info("-" * 70)
    logger.info(f"Loading Best SFT Checkpoint: {train_results['best_checkpoint_path']}")
    logger.info("-" * 70)

    best_ckpt = torch.load(train_results["best_checkpoint_path"], map_location="cpu", weights_only=False)
    best_model = GroundUpGPT(GPTConfig(**best_ckpt["model_config"]))
    best_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    best_model.to(device)

    post_benchmark_results = generate_benchmark_suite(
        best_model, tok, benchmark_items, device=device
    )

    del best_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # 6. Benchmark Comparisons & Grading
    logger.info("-" * 70)
    logger.info("Evaluating Conversational Quality Across 35 Prompts...")
    logger.info("-" * 70)

    comparisons: List[Dict[str, Any]] = []
    pre_greedy = baseline_benchmark_results["greedy"]
    post_greedy = post_benchmark_results["greedy"]
    pre_sampling = baseline_benchmark_results["sampling"]
    post_sampling = post_benchmark_results["sampling"]

    for idx, item in enumerate(benchmark_items):
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]
        guidance = item.get("expected_response_guidance", "")

        pg_resp = pre_greedy[idx]["response"]
        psg_resp = post_greedy[idx]["response"]
        psamp_resp = pre_sampling[idx]["response"]
        postsamp_resp = post_sampling[idx]["response"]

        grade_pre_g = grade_response(prompt, pg_resp, cat)
        grade_post_g = grade_response(prompt, psg_resp, cat)
        grade_pre_s = grade_response(prompt, psamp_resp, cat)
        grade_post_s = grade_response(prompt, postsamp_resp, cat)

        comparisons.append(
            {
                "id": p_id,
                "category": cat,
                "prompt": prompt,
                "guidance": guidance,
                "greedy": {
                    "pre_sft_response": pg_resp,
                    "post_sft_response": psg_resp,
                    "pre_grade": grade_pre_g,
                    "post_grade": grade_post_g,
                },
                "sampling": {
                    "pre_sft_response": psamp_resp,
                    "post_sft_response": postsamp_resp,
                    "pre_grade": grade_pre_s,
                    "post_grade": grade_post_s,
                },
            }
        )

    # 7. Summary Report & Telemetry Serialization
    telemetry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.base_checkpoint),
        "source_checkpoint_sha256": base_sha,
        "source_step": train_results["source_step"],
        "source_val_loss": train_results["source_val_loss"],
        "model_parameters": 27_411_456,
        "device": device_name,
        "training_results": train_results,
        "benchmark_comparisons": comparisons,
    }

    with open(args.log_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    logger.info(f"Telemetry logged atomically to {args.log_path}")

    # Print Full Terminal Summary
    print("\n" + "=" * 80)
    print("GROUNDUPGPT STEP 37 — 27M SUPERVISED FINE-TUNING REPORT")
    print("=" * 80)
    print(f"Source Pretrained Checkpoint:  {args.base_checkpoint} (Step {train_results['source_step']})")
    print(f"Source SHA-256:               {base_sha}")
    print(f"Tokenizer Fingerprint:        {tok_fp}")
    print(f"Model Parameters:             27,411,456 (8L/8H/512d/256ctx)")
    print(f"Compute Device:               {device_name}")
    print(f"Total SFT Steps:              {train_results['sft_steps']} / {args.max_steps}")
    print(f"Successful Optimizer Steps:   {train_results['successful_updates']}")
    print(f"Skipped Optimizer Steps:      {train_results['skipped_updates']}")
    print(f"Micro-batches Processed:      {train_results['micro_batches_processed']}")
    print(f"Sample Exposures:             {train_results['conversation_sample_exposures']}")
    print(f"Active Targets Processed:     {train_results['active_assistant_targets_processed']}")
    print(f"Total GPU Wall Time:          {train_results['total_elapsed_seconds']:.2f} s ({train_results['total_elapsed_seconds']/60:.2f} min)")
    print(f"Stop Reason:                  {train_results['stop_reason']}")
    print("-" * 80)
    print("HELD-OUT SFT VALIDATION METRICS:")
    print(f"  Pre-SFT Baseline (Step 0):   Loss: {train_results['pre_sft_baseline']['sft_val_loss']:.4f} | PPL: {train_results['pre_sft_baseline']['sft_val_ppl']:.2f}")
    print(f"  Post-SFT Best:               Loss: {train_results['post_sft_metrics']['best_sft_val_loss']:.4f} | PPL: {train_results['post_sft_metrics']['best_sft_val_ppl']:.2f}")
    print(f"  Post-SFT Final (Step {train_results['sft_steps']}):    Loss: {train_results['post_sft_metrics']['final_sft_val_loss']:.4f} | PPL: {train_results['post_sft_metrics']['final_sft_val_ppl']:.2f}")
    print("-" * 80)
    print("CATASTROPHIC FORGETTING AUDIT (PRETRAINING LM BENCHMARK TRACKS):")
    base_tracks = train_results["pre_sft_baseline"].get("lm_tracks", {})
    final_tracks = train_results["post_sft_metrics"].get("final_lm_tracks", {})
    print(f"  Track A (Combined Val):      {base_tracks.get('track_A_combined_loss', 0.0):.4f} (PPL {base_tracks.get('track_A_combined_ppl', 0.0):.2f}) -> {final_tracks.get('track_A_combined_loss', 0.0):.4f} (PPL {final_tracks.get('track_A_combined_ppl', 0.0):.2f})")
    print(f"  Track B (Orig Gutenberg):    {base_tracks.get('track_B_orig_gt_loss', 0.0):.4f} (PPL {base_tracks.get('track_B_orig_gt_ppl', 0.0):.2f}) -> {final_tracks.get('track_B_orig_gt_loss', 0.0):.4f} (PPL {final_tracks.get('track_B_orig_gt_ppl', 0.0):.2f})")
    print(f"  Track C (Wikipedia Val):     {base_tracks.get('track_C_wiki_loss', 0.0):.4f} (PPL {base_tracks.get('track_C_wiki_ppl', 0.0):.2f}) -> {final_tracks.get('track_C_wiki_loss', 0.0):.4f} (PPL {final_tracks.get('track_C_wiki_ppl', 0.0):.2f})")
    print("-" * 80)
    print("CHECKPOINTS PRESERVED:")
    print(f"  Best Checkpoint:             {train_results['best_checkpoint_path']}")
    print(f"  Latest Checkpoint:           {train_results['latest_checkpoint_path']}")
    print(f"  Telemetry Log:               {args.log_path}")
    print("-" * 80)
    print("UNEDITED CONVERSATIONAL COMPARISONS (SELECTION OF 15 BENCHMARK PROMPTS):")

    # Display at least 15 unedited comparisons
    selected_indices = [0, 1, 2, 4, 7, 9, 12, 14, 16, 18, 21, 24, 27, 30, 34]
    for sel_idx in selected_indices:
        c = comparisons[sel_idx]
        print(f"\n--- [{c['id'].upper()}] Category: {c['category']} ---")
        print(f"User Prompt: {c['prompt']}")
        print(f"Guidance:    {c['guidance']}")
        print(f"Pre-SFT Greedy:      {repr(c['greedy']['pre_sft_response'])}")
        print(f"Post-SFT Greedy:     {repr(c['greedy']['post_sft_response'])}")
        print(f"Pre-SFT Sampling:    {repr(c['sampling']['pre_sft_response'])}")
        print(f"Post-SFT Sampling:   {repr(c['sampling']['post_sft_response'])}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""Unit tests for Step 48: 27M Neural Conversational SFT v2.

Tests:
1. Frozen benchmark integrity and zero-leakage guarantee
2. Conversational SFT v2 dataset formatting and assistant-only loss masking
3. Model checkpoint loading and metadata integrity
4. Deterministic inference with clean delimiter termination
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, generate_sft_response
from src.model import GPTConfig, GroundUpGPT

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmarks" / "benchmark_conversational_step48.json"
CACHE_DIR = REPO_ROOT / "data" / "cache_sft_neural_v2"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "27m_sft_neural_v2" / "model_27m_neural_v2_best.pt"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def test_frozen_benchmark_integrity():
    """Verify that the frozen benchmark file exists, has 84 items, and matches expected categories."""
    assert BENCHMARK_PATH.exists(), f"Benchmark file missing: {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["num_items"] == 84
    assert len(data["prompts"]) == 84
    assert data["num_categories"] == 12

    # Check all 12 categories are present with exactly 7 items each
    cat_counts = {}
    for p in data["prompts"]:
        cat = p["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        assert "prompt" in p and len(p["prompt"]) > 3
        assert "expected_answer" in p
        assert "task_type" in p

    assert len(cat_counts) == 12
    for cat, count in cat_counts.items():
        assert count == 7, f"Category {cat} has {count} items, expected 7"


def test_sft_v2_dataset_zero_leakage():
    """Verify strict zero-leakage between SFT v2 training data and the frozen benchmark."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    bench_norm_set = {normalize(item["prompt"]) for item in bench_data["prompts"]}

    # Check JSONL files
    jsonl_dir = REPO_ROOT / "data" / "sft" / "conversational_v2"
    train_jsonl = jsonl_dir / "sft_conversational_v2_train.jsonl"
    val_jsonl = jsonl_dir / "sft_conversational_v2_val.jsonl"
    manifest_path = jsonl_dir / "dataset_manifest.json"

    assert train_jsonl.exists(), f"Train JSONL missing: {train_jsonl}"
    assert val_jsonl.exists(), f"Val JSONL missing: {val_jsonl}"
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["benchmark_collisions"] == 0

    total_records = 0
    for p in [train_jsonl, val_jsonl]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                norm_p = normalize(it["prompt"])
                assert norm_p not in bench_norm_set, f"Benchmark leak detected: {it['prompt']}"
                total_records += 1

    assert total_records >= 800

    # Also verify binary cache if present
    train_cache_file = CACHE_DIR / "train_tokens.pt"
    if train_cache_file.exists():
        train_cache = torch.load(train_cache_file, weights_only=False)
        assert len(train_cache) >= 700


def test_sft_v2_assistant_loss_masking():
    """Verify that prompt tokens are masked to -100 and assistant tokens have mask=1."""
    train_cache = torch.load(CACHE_DIR / "train_tokens.pt", weights_only=False)
    for ex in train_cache[:30]:
        target_ids = ex["target_ids"]
        loss_mask = ex["loss_mask"]
        p_count = ex["prompt_tokens_count"]

        # Prompt targets before response must be -100
        prompt_targets = target_ids[: p_count - 1]
        assert (prompt_targets == -100).all(), "Prompt targets must be -100"

        # Active targets must have loss_mask == 1
        active = (target_ids != -100)
        assert (active == (loss_mask == 1)).all(), "loss_mask must match active targets"


def test_model_27m_neural_v2_checkpoint_loading():
    """Verify that model_27m_neural_v2_best.pt exists and loads cleanly."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    assert "model_config" in ckpt
    assert "model_state_dict" in ckpt
    assert "step" in ckpt
    assert "global_step" in ckpt

    cfg = GPTConfig(**ckpt["model_config"])
    assert cfg.num_layers == 8
    assert cfg.num_heads == 8
    assert cfg.embed_dim == 512
    assert cfg.vocab_size == 4096

    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 27_411_456


def test_model_27m_neural_v2_inference():
    """Verify inference generation with stop sequence handling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    generator = TextGenerator(model, tokenizer, device=device)

    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        temperature=0.0,
        stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"],
    )

    result = generate_sft_response(generator, prompt="Thank you for your help!", config=gen_cfg)
    assert len(result.generated_text) > 0
    assert "<|user|>" not in result.generated_text
    assert "<|assistant|>" not in result.generated_text

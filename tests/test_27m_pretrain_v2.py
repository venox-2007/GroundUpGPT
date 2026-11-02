"""Unit tests for GroundUpGPT-27M Pretraining v2 (Step 49)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer
from scripts.audit_pretraining_corpus import run_corpus_audit
from scripts.evaluate_step48_benchmark import grade_prompt
from scripts.train_27m_pretrain_v2 import PretrainMixedSampler


def test_step48_evaluation_heuristic_disclaimers():
    """Verify open-ended conversational prompts are designated with requires_qualitative_review = True."""
    open_ended_item = {
        "id": "open_test_1",
        "category": "greetings_persona",
        "task_type": "open_ended_conversational",
        "prompt": "Hello! How are you today?",
        "expected_answer": "I am doing well, thank you!",
        "acceptable_equivalents": ["doing well", "hello", "hi"],
        "is_objective": False,
    }
    grade = grade_prompt(open_ended_item, "Hello there! I am doing well, thanks for asking.")
    assert grade["requires_qualitative_review"] is True
    assert grade["is_objective"] is False

    # Objective task should have requires_qualitative_review = False
    objective_item = {
        "id": "obj_test_1",
        "category": "short_direct_facts",
        "task_type": "exact_factual_qa",
        "prompt": "What is the capital of France?",
        "expected_answer": "Paris",
        "acceptable_equivalents": ["Paris"],
        "is_objective": True,
    }
    obj_grade = grade_prompt(objective_item, "The capital of France is Paris.")
    assert obj_grade["requires_qualitative_review"] is False
    assert obj_grade["is_objective"] is True
    assert obj_grade["passed"] is True


def test_sft_v2_concept_disjointness():
    """Verify that Conversational SFT v2 dataset has zero concept overlap between train and val."""
    train_jsonl = REPO_ROOT / "data" / "sft" / "conversational_v2" / "sft_conversational_v2_train.jsonl"
    val_jsonl = REPO_ROOT / "data" / "sft" / "conversational_v2" / "sft_conversational_v2_val.jsonl"
    assert train_jsonl.exists(), f"Missing {train_jsonl}"
    assert val_jsonl.exists(), f"Missing {val_jsonl}"

    train_concepts = set()
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            train_concepts.add(item["concept_id"])

    val_concepts = set()
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            val_concepts.add(item["concept_id"])

    assert len(train_concepts) > 0
    assert len(val_concepts) > 0
    overlap = train_concepts.intersection(val_concepts)
    assert len(overlap) == 0, f"Found concept overlap between train and val: {overlap}"


def test_corpus_audit_accounting():
    """Verify corpus audit correctly computes 5,949 pretraining token exposures and ~48% coverage."""
    audit_file = REPO_ROOT / "logs" / "pretraining_step49" / "corpus_audit.json"
    assert audit_file.exists(), f"Missing {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    exposures = audit["training_token_exposures"]
    assert exposures["pretraining_optimizer_steps"] == 5949
    assert exposures["tokens_per_step"] == 4096
    assert exposures["total_tokens_exposed"] == 24367104
    assert 48.0 <= exposures["single_epoch_coverage_pct"] <= 49.0
    assert exposures["corpus_unique_train_tokens"] == 50301446


def test_pretrain_mixed_sampler():
    """Verify that PretrainMixedSampler extracts valid tensor batches and tracks exposures."""
    # Synthetic tokens
    v1_fake = torch.arange(1000, dtype=torch.int32)
    v2_fake = torch.arange(1000, dtype=torch.int32)

    sampler = PretrainMixedSampler(
        v1_tokens=v1_fake,
        v2_tokens=v2_fake,
        mix_ratio_v2=0.5,
        block_size=64,
        seed=42,
    )

    batch_x, batch_y = sampler.sample_batch(batch_size=8)
    assert batch_x.shape == (8, 64)
    assert batch_y.shape == (8, 64)
    # Check shift invariant: batch_y is batch_x shifted by 1
    for i in range(8):
        diff = batch_y[i] - batch_x[i]
        assert (diff == 1).all()

    assert sampler.tokens_exposed_v1 + sampler.tokens_exposed_v2 == 8 * 64


def test_pretrain_v2_cache_integrity():
    """Verify pretrain_v2 cache files exist, have >20M tokens, and within valid vocab bounds."""
    cache_dir = REPO_ROOT / "data" / "cache_pretrain_v2"
    train_path = cache_dir / "train_tokens.pt"
    val_path = cache_dir / "val_tokens.pt"
    manifest_path = cache_dir / "source_manifest.json"

    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    train_tokens = torch.load(train_path, weights_only=False)
    val_tokens = torch.load(val_path, weights_only=False)

    assert train_tokens.ndim == 1
    assert val_tokens.ndim == 1
    assert train_tokens.dtype == torch.int32
    assert val_tokens.dtype == torch.int32

    total_tokens = len(train_tokens) + len(val_tokens)
    assert total_tokens >= 20_000_000, f"Expected at least 20M tokens, got {total_tokens:,}"

    assert (train_tokens >= 0).all() and (train_tokens < 4096).all()
    assert (val_tokens >= 0).all() and (val_tokens < 4096).all()


def test_continued_pretraining_checkpoint_provenance():
    """Verify that Step 49 continued pretraining checkpoint preserves full state and lineage."""
    ckpt_path = REPO_ROOT / "checkpoints" / "27m_pretrain_v2" / "model_27m_pretrain_v2_best.pt"
    assert ckpt_path.exists(), f"Missing {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 1500
    assert ckpt["source_step"] == 5949
    assert ckpt["global_step"] == 7449

    # State dicts
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scaler_state_dict" in ckpt
    assert "rng_state_torch" in ckpt

    # Lineage
    assert "parent_checkpoint_path" in ckpt
    assert "parent_checkpoint_sha256" in ckpt
    assert ckpt["parent_checkpoint_sha256"] == "3e2f62dd9d53bf8a2dffeb91d92f362d6873163c57429084b6a39bb4dcb0baa1"

    # Token exposure accounting
    assert ckpt["total_tokens_exposed"] == 12_288_000
    assert ckpt["tokens_exposed_v1"] > 0
    assert ckpt["tokens_exposed_v2"] > 0

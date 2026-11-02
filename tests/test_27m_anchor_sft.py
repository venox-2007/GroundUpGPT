"""Tests for Step 40: Targeted conversational fine-tuning with verified short-answer anchors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, format_sft_prompt
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    BlendedSFTDataPipeline,
    ConversationTurn,
    SFTDataPipeline,
    encode_conversation,
)
from scripts.evaluate_sft_27m_anchors import conservative_grade
from scripts.prepare_conversation_anchors import (
    EXPECTED_TOKENIZER_FINGERPRINT,
    build_raw_anchor_catalog,
    load_protected_prompts,
    normalize_prompt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok_path = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
    assert tok_path.exists(), "Tokenizer file missing"
    tok = BPETokenizer.load(str(tok_path))
    assert tok.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT
    return tok


def test_anchor_source_provenance_and_counts() -> None:
    catalog = build_raw_anchor_catalog()
    assert len(catalog) == 430, f"Expected 430 catalog items, got {len(catalog)}"

    valid_provenances = {"manual_curated", "deterministic_generated", "template_derived"}
    for item in catalog:
        assert item.provenance in valid_provenances, f"Invalid provenance {item.provenance} for {item.id}"
        assert item.factual_verification, f"Missing factual verification for {item.id}"
        assert item.split in ("train", "val", "test_transfer"), f"Invalid split {item.split} for {item.id}"


def test_arithmetic_ground_truth_calculation() -> None:
    catalog = build_raw_anchor_catalog()
    arith_items = [it for it in catalog if it.category == "arithmetic"]
    assert len(arith_items) == 70, f"Expected 70 arithmetic items, got {len(arith_items)}"

    for it in arith_items:
        assert it.provenance == "deterministic_generated"
        # Check equation format
        p = it.prompt
        r = it.response
        if "+" in p:
            nums = [int(n) for n in re_digits(p)]
            expected = sum(nums)
            assert str(expected) in r, f"Math mismatch in {p} -> {r}"
        elif "minus" in p or "-" in p:
            nums = [int(n) for n in re_digits(p)]
            expected = nums[0] - nums[1]
            assert str(expected) in r, f"Math mismatch in {p} -> {r}"
        elif "times" in p or "multiplied by" in p:
            nums = [int(n) for n in re_digits(p)]
            expected = nums[0] * nums[1]
            assert str(expected) in r, f"Math mismatch in {p} -> {r}"
        elif "divided by" in p:
            nums = [int(n) for n in re_digits(p)]
            expected = nums[0] // nums[1]
            assert str(expected) in r, f"Math mismatch in {p} -> {r}"


def re_digits(text: str) -> List[str]:
    import re
    return re.findall(r"\b\d+\b", text)


def test_deduplication_and_protection_isolation() -> None:
    protected, _ = load_protected_prompts()
    assert len(protected) > 500, "Protected prompts set too small"

    catalog = build_raw_anchor_catalog()
    seen_prompts = set()

    for item in catalog:
        norm_p = normalize_prompt(item.prompt)
        assert norm_p not in seen_prompts, f"Duplicate prompt in anchor catalog: {item.prompt}"
        seen_prompts.add(norm_p)

        if item.split == "train":
            assert norm_p not in protected, f"Protected evaluation prompt leaked into anchor training split: {item.prompt}"


def test_group_aware_splitting() -> None:
    catalog = build_raw_anchor_catalog()
    train_items = [it for it in catalog if it.split == "train"]
    val_items = [it for it in catalog if it.split == "val"]
    test_items = [it for it in catalog if it.split == "test_transfer"]

    assert len(train_items) == 321, f"Expected 321 train, got {len(train_items)}"
    assert len(val_items) == 53, f"Expected 53 val, got {len(val_items)}"
    assert len(test_items) == 56, f"Expected 56 test_transfer, got {len(test_items)}"

    # For arithmetic, ensure operand pairs in train vs val vs test are distinct
    arith_train_ops = {tuple(re_digits(it.prompt)) for it in train_items if it.category == "arithmetic"}
    arith_val_ops = {tuple(re_digits(it.prompt)) for it in val_items if it.category == "arithmetic"}
    arith_test_ops = {tuple(re_digits(it.prompt)) for it in test_items if it.category == "arithmetic"}

    assert not (arith_train_ops & arith_val_ops), "Arithmetic operand leakage between train and val"
    assert not (arith_train_ops & arith_test_ops), "Arithmetic operand leakage between train and test"
    assert not (arith_val_ops & arith_test_ops), "Arithmetic operand leakage between val and test"


def test_tokenizer_compliance_and_loss_masking(tokenizer: BPETokenizer) -> None:
    cache_path = REPO_ROOT / "data" / "cache_sft_anchors_v1" / "train_anchor_tokens.pt"
    assert cache_path.exists(), "Anchor training cache missing"
    cache = torch.load(cache_path, weights_only=False)

    for ex in cache[:30]:
        input_ids = ex["input_ids"]
        target_ids = ex["target_ids"]
        loss_mask = ex["loss_mask"]
        p_count = ex["prompt_tokens_count"]
        r_count = ex["response_tokens_count"]

        # Causal shift check
        assert input_ids.size(0) == target_ids.size(0)
        assert input_ids.size(0) == loss_mask.size(0)
        assert input_ids.size(0) == p_count + r_count - 1

        first_target_idx = p_count - 1
        # Prefix targets must be IGNORE_INDEX
        assert (target_ids[:first_target_idx] == IGNORE_INDEX).all()
        assert (loss_mask[:first_target_idx] == 0).all()

        # Response targets must be active
        assert (target_ids[first_target_idx:] != IGNORE_INDEX).all()
        assert (loss_mask[first_target_idx:] == 1).all()

        # Token values in range
        assert (input_ids >= 0).all() and (input_ids < tokenizer.vocab_size).all()


def test_blended_sft_pipeline_sampling_and_proportions(tokenizer: BPETokenizer) -> None:
    pipeline = BlendedSFTDataPipeline.load(
        dolly_cache_dir=REPO_ROOT / "data" / "cache_sft_27m_v1",
        anchor_cache_dir=REPO_ROOT / "data" / "cache_sft_anchors_v1",
        tokenizer=tokenizer,
        anchor_sample_prob=0.55,
        seed=1337,
    )

    # Sample 20 batches of 16
    for _ in range(20):
        x, y, mask = pipeline.get_batch(split="train", batch_size=16)
        assert x.shape == y.shape == mask.shape
        assert mask.sum() > 0

    stats = pipeline.get_exposure_summary()
    assert stats["total_samples_drawn"] == 320
    # Expected anchor sample fraction ~50-60%
    assert 0.45 <= stats["anchor_sample_fraction"] <= 0.65
    # Expected anchor target token fraction ~20-35%
    assert 0.18 <= stats["anchor_target_token_fraction"] <= 0.38
    assert len(stats["category_exposures"]) > 5


def test_conservative_grade_edge_cases() -> None:
    # 1. Correct arithmetic
    g1 = conservative_grade("What is 2 + 2?", "4.", ["4", "four"])
    assert g1["passed"] is True

    # 2. Equation format
    g2 = conservative_grade("What is 2 + 2?", "2 + 2 = 4.", ["4", "four"])
    assert g2["passed"] is True

    # 3. Contradictory extraneous numbers
    g3 = conservative_grade("What is 2 + 2?", "102 4 years old.", ["4", "four"])
    assert g3["passed"] is False

    # 4. Prompt echoing
    g4 = conservative_grade("What is 2 + 2?", "What is 2 + 2?", ["4", "four"])
    assert g4["passed"] is False

    # 5. Substring false positive
    g5 = conservative_grade("What is the capital of France?", "Comparison of capitals", ["Paris"])
    assert g5["passed"] is False

    # 6. Correct textual answer
    g6 = conservative_grade("What is the capital of France?", "The capital of France is Paris.", ["Paris"])
    assert g6["passed"] is True

    # 7. Unscored open-ended prompt
    g7 = conservative_grade("Tell me a story.", "Once upon a time...", [], is_objective=False)
    assert g7["is_objective"] is False
    assert g7["passed"] is None


def test_backward_compatibility_with_step600_baseline() -> None:
    base_path = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
    assert base_path.exists(), "Step 600 baseline missing"
    ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt.get("sft_step") == 600
    assert ckpt.get("model_config", {}).get("vocab_size") == 4096

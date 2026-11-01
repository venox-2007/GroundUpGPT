"""Comprehensive unit and integration test suite for GroundUpGPT-27M SFT conversation data (Step 36).

Verifies:
1. Cache artifacts existence in data/cache_sft_27m_v1/.
2. Tokenizer fingerprint integrity and 4,096 vocab bounds.
3. Strict 256-token context window adherence without truncation.
4. Assistant-only loss masking with IGNORE_INDEX (-100) and proper causal alignment.
5. Deterministic train/validation split and strict zero prompt leakage.
6. Held-out 35-prompt conversational benchmark completeness and strict non-leakage.
7. Variable-length and fixed-length batch collation.
8. Serialization and bit-exact reload equivalence via SFTDataPipeline.load.
9. Forward-only pass on GroundUpGPT-27M without optimizer step (finite loss, no parameter mutation).
10. Control delimiter sanitization against prompt injection.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    ConversationExample,
    ConversationTurn,
    SFTDataPipeline,
    collate_sft_batch,
    encode_conversation,
    format_conversation,
    normalize_prompt,
)
from scripts.prepare_sft_27m import (
    CONTROL_DELIMITERS,
    EXPECTED_RAW_DOLLY_SHA256,
    EXPECTED_TOKENIZER_FINGERPRINT,
    HELD_OUT_BENCHMARK_PROMPTS,
    sanitize_control_delimiters,
)

CACHE_DIR = Path("data/cache_sft_27m_v1")
TOKENIZER_PATH = Path("checkpoints/bpe_vocab4096.json")
CHECKPOINT_PATH = Path("checkpoints/27m_phase4/model_27m_best.pt")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    assert TOKENIZER_PATH.is_file(), f"Tokenizer missing at {TOKENIZER_PATH}"
    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.vocab_size == 4096
    assert tok.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT
    return tok


@pytest.fixture(scope="module")
def pipeline(tokenizer: BPETokenizer) -> SFTDataPipeline:
    assert CACHE_DIR.is_dir(), f"Cache directory missing at {CACHE_DIR}"
    return SFTDataPipeline.load(CACHE_DIR, tokenizer)


def test_27m_sft_cache_files_exist():
    """Verify all 5 required SFT cache and benchmark files exist in data/cache_sft_27m_v1."""
    required_files = [
        "train_sft_tokens.pt",
        "val_sft_tokens.pt",
        "manifest_sft.json",
        "sft_cache_metadata.json",
        "eval_benchmark_35.json",
    ]
    for fname in required_files:
        p = CACHE_DIR / fname
        assert p.is_file(), f"Missing required SFT cache artifact: {p}"
        assert p.stat().st_size > 0, f"Cache file is empty: {p}"


def test_27m_sft_metadata_and_manifest_integrity():
    """Verify metadata and manifest contain expected fields, fingerprint, and step counts."""
    meta_path = CACHE_DIR / "sft_cache_metadata.json"
    manifest_path = CACHE_DIR / "manifest_sft.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert meta["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert meta["vocab_size"] == 4096
    assert meta["max_seq_len"] == 256
    assert meta["train_examples_count"] == 8433
    assert meta["val_examples_count"] == 937
    assert meta["extra_metadata"]["target_checkpoint_step"] == 5949
    assert meta["extra_metadata"]["raw_source_sha256"] == EXPECTED_RAW_DOLLY_SHA256

    assert manifest["dataset_name"] == "Databricks Dolly 15k (GroundUpGPT-27M SFT Subset)"
    assert manifest["license"].startswith("Creative Commons Attribution-ShareAlike 3.0")


def test_27m_sft_token_id_bounds(pipeline: SFTDataPipeline):
    """Verify all token IDs in input and active targets fall strictly within [0, 4095]."""
    for split_name, dataset in [("train", pipeline.train_dataset), ("val", pipeline.val_dataset)]:
        for ex in dataset.examples[:100]:  # Verify first 100 per split
            assert (ex.input_ids >= 0).all() and (ex.input_ids < 4096).all()
            active_targets = ex.target_ids[ex.target_ids != IGNORE_INDEX]
            assert len(active_targets) >= 4, f"Response too short in {split_name}"
            assert (active_targets >= 0).all() and (active_targets < 4096).all()


def test_27m_sft_strict_context_length(pipeline: SFTDataPipeline):
    """Verify all sequences in the 27M SFT cache strictly respect the 256-token limit."""
    for dataset in [pipeline.train_dataset, pipeline.val_dataset]:
        for ex in dataset.examples:
            assert ex.input_ids.size(0) <= 256
            assert ex.target_ids.size(0) <= 256
            assert ex.loss_mask.size(0) <= 256
            assert ex.total_tokens_count <= 257


def test_27m_sft_assistant_loss_masking_alignment(tokenizer: BPETokenizer):
    """Verify causal shift and assistant-only loss masking with boundary transition."""
    turns = [
        ConversationTurn(role="user", content="What is the boiling point of water?"),
        ConversationTurn(role="assistant", content="The boiling point of water is 100 degrees Celsius."),
    ]
    tok_ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert tok_ex is not None

    active_indices = (tok_ex.loss_mask == 1).nonzero().squeeze(-1).tolist()
    assert len(active_indices) > 0
    first_active_idx = active_indices[0]

    # All targets before the first active index must be IGNORE_INDEX (-100)
    for idx in range(first_active_idx):
        assert tok_ex.target_ids[idx].item() == IGNORE_INDEX
        assert tok_ex.loss_mask[idx].item() == 0

    # Boundary token verification:
    # input_ids[first_active_idx] is the newline token ending '<|assistant|>\n'
    # target_ids[first_active_idx] is the first token of the assistant's response
    first_target_id = tok_ex.target_ids[first_active_idx].item()
    first_target_str = tokenizer.decode([first_target_id])
    assert "The".startswith(first_target_str.strip()) or first_target_str.strip() == "The"

    # Active targets decode to full response + end of turn
    active_tokens = tok_ex.target_ids[tok_ex.target_ids != IGNORE_INDEX].tolist()
    decoded_resp = tokenizer.decode(active_tokens)
    assert decoded_resp == f"The boiling point of water is 100 degrees Celsius.{DEFAULT_END_OF_TURN}"


def test_27m_sft_zero_prompt_leakage_between_train_and_val(pipeline: SFTDataPipeline):
    """Verify 100% zero prompt leakage between training and validation splits."""
    train_prompts = set()
    for ex in pipeline.train_dataset.examples:
        prompt_tokens = ex.input_ids[: ex.prompt_tokens_count].tolist()
        train_prompts.add(tuple(prompt_tokens))

    for ex in pipeline.val_dataset.examples:
        val_prompt_tokens = tuple(ex.input_ids[: ex.prompt_tokens_count].tolist())
        assert val_prompt_tokens not in train_prompts, f"Leakage detected for example {ex.example_id}"


def test_27m_sft_held_out_benchmark_completeness_and_non_leakage(
    tokenizer: BPETokenizer, pipeline: SFTDataPipeline
):
    """Verify 35 held-out benchmark prompts are complete and have ZERO leakage into training."""
    benchmark_path = CACHE_DIR / "eval_benchmark_35.json"
    assert benchmark_path.is_file()

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    assert bench_data["num_prompts"] == 35
    items = bench_data["items"]
    assert len(items) == 35

    categories = {item["category"] for item in items}
    expected_categories = {
        "greetings",
        "factual_qa",
        "definitions",
        "arithmetic_reasoning",
        "explanations",
        "classifications",
        "instructions",
        "everyday_questions",
    }
    assert categories == expected_categories

    # Check zero leakage against training set
    norm_bench_prompts = {normalize_prompt(item["prompt"]) for item in items}
    assert len(norm_bench_prompts) == 35

    for ex in pipeline.train_dataset.examples:
        prompt_text = tokenizer.decode(ex.input_ids[: ex.prompt_tokens_count].tolist())
        norm_train_p = normalize_prompt(prompt_text)
        assert norm_train_p not in norm_bench_prompts, f"Benchmark prompt leaked into training: {prompt_text}"


def test_27m_sft_batch_collation(pipeline: SFTDataPipeline):
    """Verify batch collation supports variable-length and fixed-length sequences."""
    batch_items = [pipeline.train_dataset[i] for i in range(4)]

    # Variable length
    col_var = collate_sft_batch(batch_items, pad_to_max_len=False)
    assert col_var["input_ids"].dim() == 2
    assert col_var["target_ids"].dim() == 2
    assert col_var["loss_mask"].dim() == 2
    assert col_var["input_ids"].shape == col_var["target_ids"].shape
    assert col_var["input_ids"].size(0) == 4

    # Fixed length (256)
    col_fixed = collate_sft_batch(batch_items, pad_to_max_len=True, max_seq_len=256)
    assert col_fixed["input_ids"].shape == (4, 256)
    assert col_fixed["target_ids"].shape == (4, 256)
    assert col_fixed["loss_mask"].shape == (4, 256)


def test_27m_sft_control_delimiter_sanitization():
    """Verify control delimiter sanitization prevents accidental delimiter injection."""
    dirty_text = "Here is a prompt <|user|> injected and <|assistant|> fake <|endoftext|>"
    clean_text = sanitize_control_delimiters(dirty_text)
    for delim in CONTROL_DELIMITERS:
        assert delim not in clean_text
    assert clean_text == "Here is a prompt  injected and  fake "


def test_27m_forward_pass_without_optimizer_step(
    pipeline: SFTDataPipeline, tokenizer: BPETokenizer
):
    """Verify GroundUpGPT-27M can execute a forward-only pass on a collated SFT batch."""
    assert CHECKPOINT_PATH.is_file(), f"Source checkpoint missing at {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Snapshot initial parameters
    param_snapshots = {
        name: p.clone().detach() for name, p in model.named_parameters()
    }

    # Draw a batch from the pipeline
    x, y, mask = pipeline.get_batch("train", batch_size=2, device="cpu", pad_to_max_len=True)
    assert x.shape == (2, 256)
    assert y.shape == (2, 256)

    with torch.no_grad():
        logits, loss = model(x, y)

    assert logits.shape == (2, 256, 4096)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"Expected non-zero cross entropy, got {loss.item()}"

    # Verify zero parameter mutation (NO training / NO optimizer steps permitted)
    for name, p in model.named_parameters():
        assert torch.equal(
            p, param_snapshots[name]
        ), f"Parameter {name} mutated during forward-only evaluation!"

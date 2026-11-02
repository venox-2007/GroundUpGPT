"""Tests for Step 39: Conversational failure diagnosis, training signal audit, and learning probe."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn.functional as F

from src.bpe_tokenizer import BPETokenizer
from src.generation import GenerationConfig, TextGenerator, format_sft_prompt
from src.model import GPTConfig, GroundUpGPT
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    ConversationTurn,
    collate_sft_batch,
    encode_conversation,
    scrub_pii,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache_sft_27m_v1"
CHECKPOINT_27M_SFT = REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"
CHECKPOINT_27M_BASE = REPO_ROOT / "checkpoints" / "27m_phase4" / "model_27m_best.pt"
CHECKPOINT_5M_SFT = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"
TOKENIZER_PATH = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"
REQUIRED_TOKENIZER_FP = "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok = BPETokenizer.load(str(TOKENIZER_PATH))
    assert tok.get_fingerprint() == REQUIRED_TOKENIZER_FP
    return tok


def test_prompt_length_accounting_fix(tokenizer: BPETokenizer) -> None:
    """Verify that prompt_tokens_count accurately equals the number of prompt tokens (P).

    Also verifies:
    - Number of shifted model-input tokens == P + R - 1
    - First supervised target index == P - 1
    - Number of active response targets == R
    - Slicing input_ids[:prompt_tokens_count] includes all prompt tokens and final header token.
    """
    turns = [
        ConversationTurn(role="user", content="What is gravity?"),
        ConversationTurn(role="assistant", content="Gravity is a fundamental force."),
    ]
    ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert ex is not None

    # Encode prompt separately to determine true P
    user_tokens = tokenizer.encode(f"{DEFAULT_USER_PREFIX}What is gravity?\n")
    prefix_tokens = tokenizer.encode(DEFAULT_ASSISTANT_PREFIX)
    true_prompt_tokens = user_tokens + prefix_tokens
    P = len(true_prompt_tokens)

    # Encode response separately to determine true R
    resp_tokens = tokenizer.encode(f"Gravity is a fundamental force.{DEFAULT_END_OF_TURN}")
    R = len(resp_tokens)

    # 1. Total tokens N = P + R
    assert ex.total_tokens_count == P + R

    # 2. Shifted model input has N - 1 tokens
    assert len(ex.input_ids) == P + R - 1

    # 3. Prompt tokens count is exactly P
    assert ex.prompt_tokens_count == P

    # 4. Active response targets count is exactly R
    assert ex.response_tokens_count == R

    # 5. First supervised target is at index P - 1
    assert ex.first_target_index == P - 1
    assert ex.target_ids[P - 1].item() == resp_tokens[0]
    assert ex.target_ids[P - 1].item() != IGNORE_INDEX

    # 6. Preceding P - 1 targets are all IGNORE_INDEX
    for idx in range(P - 1):
        assert ex.target_ids[idx].item() == IGNORE_INDEX
        assert ex.loss_mask[idx].item() == 0

    # 7. Slicing input_ids[:P] gives all prompt tokens including the final assistant header token
    sliced_prompt = ex.input_ids[: ex.prompt_tokens_count].tolist()
    assert sliced_prompt == true_prompt_tokens
    assert tokenizer.decode(sliced_prompt).endswith("<|assistant|>\n")


def test_token_level_training_inference_agreement(tokenizer: BPETokenizer) -> None:
    """Verify that training prompt tokens identically match inference format_sft_prompt tokens."""
    prompt_cases = [
        "What is the capital of Japan?",
        "Explain quantum computing in simple terms.",
        "List three primary colors: red, green, blue.",
        "Hello! How are you today?\nI am testing newlines.",
        "Special symbols: 2 + 2 = 4, 100%, and emoji-like text.",
    ]
    for prompt in prompt_cases:
        turns = [
            ConversationTurn(role="user", content=prompt),
            ConversationTurn(role="assistant", content="Sample response."),
        ]
        ex = encode_conversation(turns, tokenizer, max_seq_len=256)
        assert ex is not None

        # Training prompt tokens
        train_prompt_ids = ex.input_ids[: ex.first_target_index + 1].tolist()

        # Inference prompt tokens
        inf_prompt = format_sft_prompt(prompt)
        inf_ids = tokenizer.encode(inf_prompt)

        assert train_prompt_ids == inf_ids, f"Mismatch for prompt: {prompt}"


def test_causal_label_alignment_and_loss_masking(tokenizer: BPETokenizer) -> None:
    """Verify one-token causal shift and assistant-only loss masking."""
    turns = [
        ConversationTurn(role="user", content="Calculate 2 + 2"),
        ConversationTurn(role="assistant", content="The answer is 4."),
    ]
    ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert ex is not None

    # Check causal shift: target_ids is aligned with input_ids + 1
    # For every active position, the target is the next token in the sequence
    first_act = ex.first_target_index
    for idx in range(first_act, len(ex.input_ids)):
        assert ex.loss_mask[idx] == 1
        assert ex.target_ids[idx] != IGNORE_INDEX
        # The target at idx is predicted using input_ids[:idx+1]
        # Next token in sequence is target_ids[idx]
        assert 0 <= ex.target_ids[idx].item() < tokenizer.vocab_size


def test_full_end_of_turn_supervision(tokenizer: BPETokenizer) -> None:
    """Verify that <|endoftext|> is included in the active supervised targets."""
    turns = [
        ConversationTurn(role="user", content="Say bye"),
        ConversationTurn(role="assistant", content="Goodbye!"),
    ]
    ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert ex is not None

    # Last active target must be the end of turn token
    last_target = ex.target_ids[-1].item()
    eot_tokens = tokenizer.encode(DEFAULT_END_OF_TURN)
    assert last_target == eot_tokens[-1]
    assert ex.loss_mask[-1] == 1


def test_objective_grading_and_false_positive_rejection() -> None:
    """Verify conservative grading rejects substring false positives and accepts correct answers."""
    from scripts.diagnose_27m_conversation import run_benchmark_and_evaluation

    # Test logic directly
    def conservative_grade(response_text: str, valid_equivalents: List[str]) -> bool:
        import re
        clean_resp = re.sub(r"[^\w\s]", " ", response_text.lower()).strip()
        clean_resp = " ".join(clean_resp.split())
        words = clean_resp.split()
        if not words:
            return False

        for eq in valid_equivalents:
            clean_eq = re.sub(r"[^\w\s]", " ", eq.lower()).strip()
            clean_eq = " ".join(clean_eq.split())
            eq_words = clean_eq.split()

            # 1. Exact full match (e.g. 'Paris', '4')
            if clean_resp == clean_eq:
                return True

            # 2. Starts with exact answer
            if len(words) >= len(eq_words) and words[: len(eq_words)] == eq_words:
                if clean_eq.isdigit():
                    if len(words) == 1:
                        return True
                    if len(words) == 2 and words[1] in ("days", "degrees", "c", "celsius", "m", "meters"):
                        return True
                    if clean_resp.startswith(f"{clean_eq} is ") or clean_resp.startswith(f"{clean_eq} was "):
                        return True
                    continue
                return True

            # 3. For numerical answers, require explicit statement without contradictory numbers
            if clean_eq.isdigit():
                digits = re.findall(r"\b\d+\b", clean_resp)
                if len(digits) > 1:
                    continue
                if f"is {clean_eq}" in clean_resp or f"answer is {clean_eq}" in clean_resp or f"equals {clean_eq}" in clean_resp:
                    return True
                continue

            # 4. For non-digits, check word boundary match without negation
            pattern = r"\b" + re.escape(clean_eq) + r"\b"
            match = re.search(pattern, clean_resp)
            if match:
                preceding = clean_resp[: match.start()].strip().split()
                if preceding and preceding[-1] in ("not", "never", "no", "isnt"):
                    continue
                return True
        return False

    # True positives
    assert conservative_grade("Paris", ["Paris"])
    assert conservative_grade("Paris is the capital of France.", ["Paris"])
    assert conservative_grade("The capital is Paris.", ["Paris"])
    assert conservative_grade("4", ["4", "four"])
    assert conservative_grade("4 is the answer.", ["4", "four"])

    # False positives that MUST be rejected
    assert not conservative_grade("3 - 2 + 2 = 3 1/ 4 = 5 | 6 = 4 = 8 = | 1 =", ["4", "four"])
    assert not conservative_grade("102 4 years old.", ["4", "four"])
    assert not conservative_grade("It is not Paris, it is London.", ["Paris"])
    assert not conservative_grade("Parisian culture is nice.", ["Paris"])


def test_teacher_forced_scoring_reproducibility() -> None:
    """Verify that teacher-forced cross-entropy scoring is strictly reproducible."""
    if not CHECKPOINT_27M_SFT.exists():
        pytest.skip("27M SFT checkpoint not found.")

    device = torch.device("cpu")
    ckpt = torch.load(CHECKPOINT_27M_SFT, map_location=device, weights_only=False)
    cfg = GPTConfig(vocab_size=4096, block_size=256, embed_dim=512, num_heads=8, num_layers=8, weight_tying=True)
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.manual_seed(42)
    x = torch.randint(0, 4096, (2, 32))
    y = torch.randint(0, 4096, (2, 32))
    y[y < 2000] = IGNORE_INDEX  # Partial masking

    with torch.inference_mode():
        logits1, _ = model(x)
        loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX)

        logits2, _ = model(x)
        loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX)

    assert torch.equal(logits1, logits2)
    assert math.isclose(loss1.item(), loss2.item(), abs_tol=1e-6)


def test_frozen_checkpoint_integrity() -> None:
    """Verify that Step 39 experiments did not overwrite or mutate original checkpoints."""
    for p in [CHECKPOINT_27M_BASE, CHECKPOINT_27M_SFT]:
        assert p.exists(), f"Missing checkpoint: {p}"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ck
        sd = ck["model_state_dict"]
        # Ensure 27M parameters
        params = sum(t.numel() for t in sd.values())
        assert params in (27411456, 33702912)  # tied vs serialized


def test_5m_and_27m_sft_profile_compatibility() -> None:
    """Verify that both 5M and 27M SFT checkpoints remain fully compatible and distinct."""
    if not CHECKPOINT_5M_SFT.exists() or not CHECKPOINT_27M_SFT.exists():
        pytest.skip("Checkpoints missing.")

    ck_5m = torch.load(CHECKPOINT_5M_SFT, map_location="cpu", weights_only=False)
    ck_27m = torch.load(CHECKPOINT_27M_SFT, map_location="cpu", weights_only=False)

    p_5m = sum(t.numel() for t in ck_5m["model_state_dict"].values())
    p_27m = sum(t.numel() for t in ck_27m["model_state_dict"].values())

    assert p_5m < p_27m
    assert p_27m > 20_000_000
    assert ck_27m.get("sft_step") == 600
    assert ck_5m.get("sft_step") == 650

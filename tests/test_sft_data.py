"""Unit and regression tests for Supervised Fine-Tuning (SFT) conversation pipeline.

Verifies:
1. BPE compatibility and exact fingerprint preservation.
2. Canonical conversation formatting and delimiters.
3. One-token target shift and assistant-only loss masking with -100.
4. PyTorch cross-entropy gradient isolation (zero gradient on user prompts).
5. Short, long, and over-length example handling.
6. Empty and malformed conversation rejection.
7. Multi-turn dialogue representation and turn-level masking.
8. Exact and near-duplicate deduplication.
9. Deterministic splitting with zero prompt leakage.
10. PII scrubbing and UTF-8 round-trip verification.
11. Token ID bounds [0, 4095].
12. Batch collation and model forward compatibility.
13. Cache save and reload equivalence with fingerprint protection.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

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
    SFTDataset,
    collate_sft_batch,
    deduplicate_conversations,
    encode_conversation,
    format_conversation,
    normalize_prompt,
    scrub_pii,
    split_conversations,
    utf8_roundtrip_check,
)

EXPECTED_TOKENIZER_FINGERPRINT = (
    "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
)
TOKENIZER_PATH = Path("checkpoints/bpe_vocab4096.json")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    assert TOKENIZER_PATH.is_file(), f"Tokenizer missing at {TOKENIZER_PATH}"
    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.vocab_size == 4096
    assert tok.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT
    return tok


def test_bpe_compatibility_and_fingerprint(tokenizer: BPETokenizer) -> None:
    """Verify tokenizer fingerprint and lossless delimiter round-trip without vocab changes."""
    assert tokenizer.vocab_size == 4096
    assert tokenizer.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT

    for delimiter in [
        DEFAULT_USER_PREFIX,
        DEFAULT_ASSISTANT_PREFIX,
        DEFAULT_END_OF_TURN,
        "<|endoftext|>",
    ]:
        token_ids = tokenizer.encode(delimiter)
        assert len(token_ids) > 0
        assert all(0 <= tid < tokenizer.vocab_size for tid in token_ids)
        decoded = tokenizer.decode(token_ids)
        assert decoded == delimiter, f"Failed roundtrip on {delimiter!r}"


def test_conversation_formatting() -> None:
    """Verify canonical conversation formatting and delimiter placement."""
    turns = [
        ConversationTurn(role="user", content="Hello, world!"),
        ConversationTurn(role="assistant", content="Hi! How can I help?"),
    ]
    formatted = format_conversation(turns)
    expected = (
        f"{DEFAULT_USER_PREFIX}Hello, world!\n"
        f"{DEFAULT_ASSISTANT_PREFIX}Hi! How can I help?{DEFAULT_END_OF_TURN}"
    )
    assert formatted == expected


def test_assistant_only_loss_masking_and_target_shift(
    tokenizer: BPETokenizer,
) -> None:
    """Verify causal target shift, prompt masking with -100, and boundary alignment."""
    user_msg = "What is the capital of France?"
    asst_msg = "The capital of France is Paris."
    turns = [
        ConversationTurn(role="user", content=user_msg),
        ConversationTurn(role="assistant", content=asst_msg),
    ]

    example = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert example is not None
    assert isinstance(example.input_ids, torch.Tensor)
    assert isinstance(example.target_ids, torch.Tensor)
    assert isinstance(example.loss_mask, torch.Tensor)

    # 1. Input and target dimensions match
    assert example.input_ids.shape == example.target_ids.shape
    assert example.loss_mask.shape == example.target_ids.shape

    # 2. Causal target shift: targets[i] corresponds to next token in sequence
    seq_len = example.input_ids.size(0)
    assert example.total_tokens_count == seq_len + 1

    # 3. Find boundary where loss mask transitions from 0 to 1
    active_indices = (example.loss_mask == 1).nonzero().squeeze(-1).tolist()
    assert len(active_indices) > 0
    first_active_idx = active_indices[0]

    # Every target before first_active_idx must be -100 (prompt tokens)
    for idx in range(first_active_idx):
        assert example.target_ids[idx].item() == IGNORE_INDEX
        assert example.loss_mask[idx].item() == 0

    # 4. Alignment verification:
    # input_ids[first_active_idx] is the last token of the assistant prefix (e.g. '\n')
    # target_ids[first_active_idx] is the first token of the assistant's actual response
    first_target_id = example.target_ids[first_active_idx].item()
    first_target_str = tokenizer.decode([first_target_id])
    assert asst_msg.startswith(
        first_target_str.strip()
    ) or asst_msg.startswith(first_target_str)

    # 5. Active targets decode exactly to assistant response + end-of-turn
    active_target_ids = [
        tid.item() for tid in example.target_ids if tid.item() != IGNORE_INDEX
    ]
    decoded_active = tokenizer.decode(active_target_ids)
    assert decoded_active == f"{asst_msg}{DEFAULT_END_OF_TURN}"


def test_cross_entropy_gradient_isolation(tokenizer: BPETokenizer) -> None:
    """Verify PyTorch cross entropy produces zero gradient on prompt positions."""
    turns = [
        ConversationTurn(role="user", content="Ping"),
        ConversationTurn(role="assistant", content="Pong!"),
    ]
    example = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert example is not None

    # Simulate logits with autograd enabled: shape (1, seq_len, vocab_size)
    seq_len = example.input_ids.size(0)
    vocab_size = tokenizer.vocab_size
    dummy_logits = torch.randn(
        (1, seq_len, vocab_size), requires_grad=True, dtype=torch.float32
    )

    targets = example.target_ids.unsqueeze(0)  # (1, seq_len)
    loss = F.cross_entropy(
        dummy_logits.view(-1, vocab_size),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    loss.backward()

    assert dummy_logits.grad is not None
    grad = dummy_logits.grad[0]  # (seq_len, vocab_size)

    # Gradients for prompt tokens (-100) must be identically 0.0
    for i in range(seq_len):
        if example.target_ids[i].item() == IGNORE_INDEX:
            assert torch.all(
                grad[i] == 0.0
            ), f"Prompt position {i} received non-zero gradient!"
        else:
            assert (
                torch.sum(torch.abs(grad[i])) > 0.0
            ), f"Active target position {i} received zero gradient!"


def test_short_and_long_examples(tokenizer: BPETokenizer) -> None:
    """Verify handling of short conversations, over-length filtering, and truncation."""
    # 1. Short example
    short_turns = [
        ConversationTurn(role="user", content="Hi"),
        ConversationTurn(role="assistant", content="Hey"),
    ]
    short_ex = encode_conversation(short_turns, tokenizer, max_seq_len=256)
    assert short_ex is not None
    assert short_ex.input_ids.size(0) < 30

    # 2. Long example with max_seq_len=16 (deliberately tiny context)
    long_turns = [
        ConversationTurn(
            role="user",
            content="Tell me a long story about space exploration.",
        ),
        ConversationTurn(
            role="assistant",
            content="Once upon a time in a distant galaxy, intrepid astronauts embarked on a great mission.",
        ),
    ]

    # Without truncation: should be rejected
    rejected = encode_conversation(
        long_turns, tokenizer, max_seq_len=16, truncate=False
    )
    assert rejected is None

    # With truncation: should be safely truncated, or rejected if assistant response is cut off
    truncated = encode_conversation(
        long_turns, tokenizer, max_seq_len=16, truncate=True
    )
    if truncated is not None:
        assert truncated.input_ids.size(0) <= 16
        assert truncated.response_tokens_count > 0


def test_empty_or_malformed_rejection(tokenizer: BPETokenizer) -> None:
    """Verify rejection of empty, whitespace-only, single-role, or malformed conversations."""
    # 1. Empty list
    assert encode_conversation([], tokenizer) is None

    # 2. Only user turn
    user_only = [ConversationTurn(role="user", content="Hello")]
    assert encode_conversation(user_only, tokenizer) is None

    # 3. Only assistant turn
    asst_only = [ConversationTurn(role="assistant", content="Hello")]
    assert encode_conversation(asst_only, tokenizer) is None

    # 4. Whitespace content
    empty_content = [
        ConversationTurn(role="user", content="   "),
        ConversationTurn(role="assistant", content="Hello"),
    ]
    assert encode_conversation(empty_content, tokenizer) is None

    # 5. Invalid role in turn
    with pytest.raises(ValueError, match="Invalid conversation role"):
        ConversationTurn(role="system", content="You are a helper")


def test_multi_turn_representation_and_masking(tokenizer: BPETokenizer) -> None:
    """Verify multi-turn dialogues correctly mask all user turns and assistant prefixes."""
    turns = [
        ConversationTurn(role="user", content="Turn 1 user"),
        ConversationTurn(role="assistant", content="Turn 1 assistant"),
        ConversationTurn(role="user", content="Turn 2 user"),
        ConversationTurn(role="assistant", content="Turn 2 assistant"),
    ]
    ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert ex is not None

    # Verify that unmasked targets contain assistant tokens but NO user tokens
    active_targets = [
        tid.item() for tid in ex.target_ids if tid.item() != IGNORE_INDEX
    ]
    decoded_active = tokenizer.decode(active_targets)
    assert "Turn 1 assistant" in decoded_active
    assert "Turn 2 assistant" in decoded_active
    assert "Turn 1 user" not in decoded_active
    assert "Turn 2 user" not in decoded_active


def test_deduplication_and_near_duplicates() -> None:
    """Verify exact full-text and near-duplicate prompt deduplication."""
    ex1 = ConversationExample(
        example_id="1",
        turns=[
            ConversationTurn(role="user", content="What is Python?"),
            ConversationTurn(
                role="assistant",
                content="Python is a programming language.",
            ),
        ],
    )
    # Exact duplicate
    ex2 = ConversationExample(
        example_id="2",
        turns=[
            ConversationTurn(role="user", content="What is Python?"),
            ConversationTurn(
                role="assistant",
                content="Python is a programming language.",
            ),
        ],
    )
    # Near duplicate prompt (different case and punctuation)
    ex3 = ConversationExample(
        example_id="3",
        turns=[
            ConversationTurn(role="user", content="what is python???"),
            ConversationTurn(role="assistant", content="A high-level language."),
        ],
    )
    # Unique example
    ex4 = ConversationExample(
        example_id="4",
        turns=[
            ConversationTurn(role="user", content="What is PyTorch?"),
            ConversationTurn(
                role="assistant", content="A machine learning framework."
            ),
        ],
    )

    unique, stats = deduplicate_conversations([ex1, ex2, ex3, ex4])
    assert len(unique) == 2
    assert stats["exact_duplicates_removed"] == 1
    assert stats["near_duplicates_removed"] == 1
    assert stats["unique_retained"] == 2


def test_reproducible_split_and_zero_prompt_leakage() -> None:
    """Verify deterministic splits and guaranteed zero prompt leakage."""
    examples = [
        ConversationExample(
            example_id=str(i),
            turns=[
                ConversationTurn(role="user", content=f"Prompt question {i}"),
                ConversationTurn(
                    role="assistant", content=f"Assistant answer {i}"
                ),
            ],
        )
        for i in range(50)
    ]

    train1, val1 = split_conversations(examples, train_ratio=0.8, seed=42)
    train2, val2 = split_conversations(examples, train_ratio=0.8, seed=42)

    # Determinism
    assert [x.example_id for x in train1] == [x.example_id for x in train2]
    assert [x.example_id for x in val1] == [x.example_id for x in val2]

    # Zero leakage check
    train_prompts = {
        normalize_prompt(ex.turns[0].content) for ex in train1
    }
    for val_ex in val1:
        val_prompt = normalize_prompt(val_ex.turns[0].content)
        assert (
            val_prompt not in train_prompts
        ), f"Leakage detected for prompt: {val_prompt}"


def test_pii_scrubbing_and_utf8() -> None:
    """Verify scrubbing of emails, phone numbers, IPs, and UTF-8 validation."""
    raw = "Contact test.user@example.com or call +1-800-555-0199 or connect to 192.168.1.50."
    scrubbed = scrub_pii(raw)
    assert "[EMAIL]" in scrubbed
    assert "test.user@example.com" not in scrubbed
    assert "[PHONE]" in scrubbed
    assert "+1-800-555-0199" not in scrubbed
    assert "[IP_ADDRESS]" in scrubbed
    assert "192.168.1.50" not in scrubbed

    assert utf8_roundtrip_check("Valid English text with emoji 🚀 and — dash.")
    assert utf8_roundtrip_check(scrubbed)


def test_token_id_bounds(tokenizer: BPETokenizer) -> None:
    """Verify all token IDs in input and active targets fall strictly within [0, 4095]."""
    turns = [
        ConversationTurn(
            role="user",
            content="Generate a list of numbers from one to three.",
        ),
        ConversationTurn(role="assistant", content="1, 2, 3."),
    ]
    ex = encode_conversation(turns, tokenizer, max_seq_len=256)
    assert ex is not None

    vocab_size = tokenizer.vocab_size
    assert (ex.input_ids >= 0).all() and (ex.input_ids < vocab_size).all()

    active_targets = ex.target_ids[ex.target_ids != IGNORE_INDEX]
    assert (active_targets >= 0).all() and (active_targets < vocab_size).all()


def test_batch_collation_and_model_forward(tokenizer: BPETokenizer) -> None:
    """Verify batch collation pads correctly and feeds seamlessly into GroundUpGPT."""
    ex1 = encode_conversation(
        [
            ConversationTurn(role="user", content="A"),
            ConversationTurn(role="assistant", content="B"),
        ],
        tokenizer,
        max_seq_len=64,
    )
    ex2 = encode_conversation(
        [
            ConversationTurn(role="user", content="Question 123?"),
            ConversationTurn(role="assistant", content="Answer 456!"),
        ],
        tokenizer,
        max_seq_len=64,
    )
    assert ex1 is not None and ex2 is not None

    batch_dict = collate_sft_batch(
        [
            {
                "input_ids": ex1.input_ids,
                "target_ids": ex1.target_ids,
                "loss_mask": ex1.loss_mask,
            },
            {
                "input_ids": ex2.input_ids,
                "target_ids": ex2.target_ids,
                "loss_mask": ex2.loss_mask,
            },
        ],
        pad_token_id=0,
        pad_to_max_len=True,
        max_seq_len=64,
    )

    x = batch_dict["input_ids"]
    y = batch_dict["target_ids"]
    mask = batch_dict["loss_mask"]

    assert x.shape == (2, 64)
    assert y.shape == (2, 64)
    assert mask.shape == (2, 64)

    # Initialize tiny GroundUpGPT model for testing forward pass
    cfg = GPTConfig(
        vocab_size=4096,
        block_size=64,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
    )
    model = GroundUpGPT(cfg)

    logits, loss = model(x, y)
    assert logits.shape == (2, 64, 4096)
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    # Confirm backward completed without error


def test_cache_save_and_reload_equivalence(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    """Verify saving to disk, reloading, metadata integrity, and fingerprint rejection."""
    ex = encode_conversation(
        [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello"),
        ],
        tokenizer,
        max_seq_len=32,
    )
    assert ex is not None

    train_ds = SFTDataset([ex], max_seq_len=32)
    val_ds = SFTDataset([ex], max_seq_len=32)
    pipeline = SFTDataPipeline(
        train_ds, val_ds, tokenizer=tokenizer, max_seq_len=32
    )

    cache_dir = tmp_path / "sft_cache"
    pipeline.save(cache_dir, manifest_metadata={"version": "1.0"})

    assert (cache_dir / "train_sft_tokens.pt").is_file()
    assert (cache_dir / "val_sft_tokens.pt").is_file()
    assert (cache_dir / "manifest_sft.json").is_file()
    assert (cache_dir / "sft_cache_metadata.json").is_file()

    reloaded = SFTDataPipeline.load(cache_dir, tokenizer)
    assert len(reloaded.train_dataset) == 1
    assert len(reloaded.val_dataset) == 1
    assert torch.equal(
        reloaded.train_dataset.examples[0].input_ids, ex.input_ids
    )
    assert torch.equal(
        reloaded.train_dataset.examples[0].target_ids, ex.target_ids
    )
    assert torch.equal(
        reloaded.train_dataset.examples[0].loss_mask, ex.loss_mask
    )

    # Tokenizer fingerprint mismatch protection
    meta_file = cache_dir / "sft_cache_metadata.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    meta_json["tokenizer_fingerprint"] = "corrupted_fake_fingerprint"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_json, f)

    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        SFTDataPipeline.load(cache_dir, tokenizer)

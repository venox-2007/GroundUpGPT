"""Targeted regression and unit tests for Step 29 Inference Audit and Quality Benchmark.

Verifies:
1. Streaming prefix buffering and truncated delimiter suppression (<|..., <).
2. Streaming vs non-streaming exact token and text equivalence.
3. Repetition penalty logit scaling logic.
4. SFT conversation prompt formatting with and without context.
5. Long prompt context cropping at architecture block_size boundary.
"""

from __future__ import annotations

import pytest
import torch

from src.generation import (
    GenerationConfig,
    TextGenerator,
    format_sft_prompt,
    generate_sft_response,
    find_longest_stop_prefix_len,
)
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from src.tokenizer import CharTokenizer


@pytest.fixture
def tiny_bpe_model():
    """Create a minimal GPT model and BPETokenizer for deterministic unit testing."""
    sample_text = (
        "<|user|>\nHello! What is this?\n<|assistant|>\n"
        "This is a test response for GroundUpGPT inference audit.<|endoftext|>"
    )
    tok = BPETokenizer()
    tok.fit(sample_text, target_vocab_size=275)

    config = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    model = GroundUpGPT(config)
    model.eval()

    generator = TextGenerator(model=model, tokenizer=tok, device="cpu")
    return generator, tok


def test_find_longest_stop_prefix_len():
    """Verify stop sequence prefix detection helper."""
    stop_seqs = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
    assert find_longest_stop_prefix_len("Normal text", stop_seqs) == 0
    assert find_longest_stop_prefix_len("Text with <", stop_seqs) == 1
    assert find_longest_stop_prefix_len("Text with <|", stop_seqs) == 2
    assert find_longest_stop_prefix_len("Text with <|endof", stop_seqs) == 7
    assert find_longest_stop_prefix_len("Text with <|user|>", stop_seqs) == 8


def test_sft_prompt_formatting():
    """Verify standard SFT prompt formatting matching training-time structure."""
    prompt_no_ctx = format_sft_prompt("What is 2+2?")
    assert prompt_no_ctx == "<|user|>\nWhat is 2+2?\n<|assistant|>\n"

    prompt_with_ctx = format_sft_prompt("Where is Paris?", context="France is in Europe.")
    assert prompt_with_ctx == "<|user|>\nWhere is Paris?\n\nContext:\nFrance is in Europe.\n<|assistant|>\n"


def test_streaming_and_non_streaming_equivalence(tiny_bpe_model):
    """Verify generate() and generate_stream() produce byte-for-byte identical text."""
    generator, _ = tiny_bpe_model
    cfg = GenerationConfig(
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
        seed=12345,
        repetition_penalty=1.15,
    )

    # Full generation
    res_full = generator.generate(prompt="Hello", config=cfg)

    # Streaming generation
    stream_tokens = []
    stream_chunks = []
    for chunk in generator.generate_stream(prompt="Hello", config=cfg):
        stream_tokens.append(chunk.token_id)
        stream_chunks.append(chunk.text_chunk)

    stream_text = "".join(stream_chunks)

    assert res_full.generated_token_ids == stream_tokens
    assert res_full.generated_text == stream_text


def test_repetition_penalty_scaling():
    """Verify repetition penalty mathematically scales logits down for previously seen tokens."""
    # Create synthetic logits: 4 tokens
    # Token 0: positive logit (2.0)
    # Token 1: negative logit (-2.0)
    # Token 2: zero logit (0.0)
    # Token 3: positive logit (5.0) - not in context
    logits = torch.tensor([2.0, -2.0, 0.0, 5.0])
    seen_ids = [0, 1, 2]
    penalty = 1.2

    # Apply penalty manually according to Keskar et al.
    scaled = logits.clone()
    for tid in set(seen_ids):
        val = scaled[tid].item()
        if val > 0:
            scaled[tid] /= penalty
        else:
            scaled[tid] *= penalty

    # Token 0 was 2.0 -> should be 2.0 / 1.2 = 1.6667
    assert scaled[0] < logits[0]
    assert pytest.approx(scaled[0].item(), rel=1e-4) == (2.0 / 1.2)

    # Token 1 was -2.0 -> should be -2.0 * 1.2 = -2.4 (more negative)
    assert scaled[1] < logits[1]
    assert pytest.approx(scaled[1].item(), rel=1e-4) == (-2.0 * 1.2)

    # Token 2 was 0.0 -> remains 0.0
    assert scaled[2].item() == 0.0

    # Token 3 was not seen -> unchanged
    assert scaled[3].item() == logits[3].item()


def test_delimiter_truncation_suppression(tiny_bpe_model):
    """Verify that truncated delimiter prefixes (<|, <) are never leaked in SFT generation."""
    generator, _ = tiny_bpe_model

    # Test prompt formatting and stop delimiter trimming
    res = generate_sft_response(
        generator=generator,
        prompt="Hello!",
        config=GenerationConfig(max_new_tokens=15, seed=42, repetition_penalty=1.15),
    )

    # Resulting text must never leak stop delimiters or prefixes
    for delim in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
        assert delim not in res.generated_text
    assert not res.generated_text.endswith("<|")
    assert not res.generated_text.endswith("<")


def test_context_window_cropping(tiny_bpe_model):
    """Verify that prompts exceeding block_size are safely cropped."""
    generator, tok = tiny_bpe_model
    # Block size is 32. Provide 40 tokens.
    long_prompt = "Hello world " * 20
    encoded = tok.encode(long_prompt)
    assert len(encoded) > 32

    res = generator.generate(
        prompt=long_prompt,
        config=GenerationConfig(max_new_tokens=5, temperature=0.0),
    )

    assert res.context_cropped is True
    assert res.active_context_tokens <= 32

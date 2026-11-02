"""Tests for generation termination telemetry and stop_reason classification."""

import pytest
import torch

from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    GenerationConfig,
    GenerationResult,
    StreamChunk,
    TextGenerator,
    generate_sft_response,
)


@pytest.fixture
def dummy_gpt_and_tokenizer() -> tuple[GroundUpGPT, BPETokenizer]:
    """Provide a minimal model and fitted BPETokenizer including special tokens."""
    text = (
        "Hello world! This is a test corpus with <|user|> and <|assistant|> "
        "and <|endoftext|> and some numbers 12345."
    )
    tok = BPETokenizer().fit(text, target_vocab_size=320)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(cfg)
    return model, tok


def test_max_new_tokens_stop_reason(dummy_gpt_and_tokenizer: tuple[GroundUpGPT, BPETokenizer]) -> None:
    """Verify that reaching max_new_tokens without delimiter yields stop_reason='max_new_tokens'."""
    model, tok = dummy_gpt_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    cfg = GenerationConfig(max_new_tokens=5, stop_sequences=["<|endoftext|>"])
    res = gen.generate(prompt="Hello", config=cfg)

    assert isinstance(res, GenerationResult)
    assert res.generated_tokens == 5
    assert res.stop_reason == "max_new_tokens"


def test_stop_reason_end_of_turn(dummy_gpt_and_tokenizer: tuple[GroundUpGPT, BPETokenizer], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that encountering '<|endoftext|>' sets stop_reason='end_of_turn'."""
    model, tok = dummy_gpt_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    # Mock sample_next_token to generate tokens for 'abc', '<|endoftext|>', then 'extra'
    eot_ids = tok.encode("<|endoftext|>")
    seq_to_emit = tok.encode("Hi ") + eot_ids + tok.encode(" extra")
    call_idx = 0

    def mock_sample(*args, **kwargs) -> int:
        nonlocal call_idx
        token_id = seq_to_emit[call_idx % len(seq_to_emit)]
        call_idx += 1
        return token_id

    monkeypatch.setattr("src.generation.sample_next_token", mock_sample)

    cfg = GenerationConfig(max_new_tokens=20, stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"])
    res = gen.generate(prompt="Start: ", config=cfg)

    assert res.stop_reason == "end_of_turn"
    assert "<|endoftext|>" not in res.generated_text
    assert "extra" not in res.generated_text


def test_stop_reason_user_and_assistant_delimiters(dummy_gpt_and_tokenizer: tuple[GroundUpGPT, BPETokenizer], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that encountering '<|user|>' and '<|assistant|>' yields corresponding stop reasons."""
    model, tok = dummy_gpt_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    # 1. Test <|user|>
    user_ids = tok.encode("<|user|>")
    seq_user = tok.encode("Answer ") + user_ids
    call_idx = 0

    def mock_sample_user(*args, **kwargs) -> int:
        nonlocal call_idx
        tid = seq_user[call_idx % len(seq_user)]
        call_idx += 1
        return tid

    monkeypatch.setattr("src.generation.sample_next_token", mock_sample_user)
    cfg = GenerationConfig(max_new_tokens=20, stop_sequences=["<|endoftext|>", "<|user|>", "<|assistant|>"])
    res_user = gen.generate(prompt="Prompt", config=cfg)
    assert res_user.stop_reason == "user_delimiter"

    # 2. Test <|assistant|>
    asst_ids = tok.encode("<|assistant|>")
    seq_asst = tok.encode("Text ") + asst_ids
    call_idx = 0

    def mock_sample_asst(*args, **kwargs) -> int:
        nonlocal call_idx
        tid = seq_asst[call_idx % len(seq_asst)]
        call_idx += 1
        return tid

    monkeypatch.setattr("src.generation.sample_next_token", mock_sample_asst)
    res_asst = gen.generate(prompt="Prompt", config=cfg)
    assert res_asst.stop_reason == "assistant_delimiter"


def test_stop_reason_other(dummy_gpt_and_tokenizer: tuple[GroundUpGPT, BPETokenizer], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a custom stop sequence yields stop_reason='other'."""
    model, tok = dummy_gpt_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    stop_ids = tok.encode("STOP")
    seq = tok.encode("Hello ") + stop_ids
    call_idx = 0

    def mock_sample(*args, **kwargs) -> int:
        nonlocal call_idx
        tid = seq[call_idx % len(seq)]
        call_idx += 1
        return tid

    monkeypatch.setattr("src.generation.sample_next_token", mock_sample)
    cfg = GenerationConfig(max_new_tokens=20, stop_sequences=["STOP"])
    res = gen.generate(prompt="Prompt", config=cfg)
    assert res.stop_reason == "other"


def test_streaming_partial_delimiter_buffering(dummy_gpt_and_tokenizer: tuple[GroundUpGPT, BPETokenizer], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that partial delimiter is held in buffer during streaming and never leaked."""
    model, tok = dummy_gpt_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    # Tokens for '<|' then 'endoftext|>'
    part1_ids = tok.encode("<|")
    part2_ids = tok.encode("endoftext|>")
    seq = part1_ids + part2_ids
    call_idx = 0

    def mock_sample(*args, **kwargs) -> int:
        nonlocal call_idx
        tid = seq[call_idx % len(seq)]
        call_idx += 1
        return tid

    monkeypatch.setattr("src.generation.sample_next_token", mock_sample)
    cfg = GenerationConfig(max_new_tokens=len(seq), stop_sequences=["<|endoftext|>"])

    chunks: list[StreamChunk] = []
    for chunk in gen.generate_stream(prompt="Test: ", config=cfg):
        chunks.append(chunk)

    # Reconstructed text emitted to consumer must not contain partial or full delimiter
    emitted_text = "".join(c.text_chunk for c in chunks)
    assert "<|" not in emitted_text
    assert "<|endoftext|>" not in emitted_text
    assert chunks[-1].is_finished is True
    assert chunks[-1].stop_reason == "end_of_turn"

"""Unit and integration tests for autoregressive text generation in GroundUpGPT."""

from pathlib import Path
import pytest
import torch

from src.model import GPTConfig, GroundUpGPT
from src.tokenizer import CharTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.generation import (
    GenerationConfig,
    GenerationResult,
    StreamChunk,
    TextGenerator,
    sample_next_token,
    decode_token_sequence,
)


@pytest.fixture
def char_model_and_tokenizer() -> tuple[GroundUpGPT, CharTokenizer]:
    """Provide a small test model and fitted CharTokenizer."""
    tok = CharTokenizer().fit("abcdefghijklmnopqrstuvwxyz .,!?")
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(cfg)
    return model, tok


@pytest.fixture
def bpe_model_and_tokenizer() -> tuple[GroundUpGPT, BPETokenizer]:
    """Provide a small test model and fitted BPETokenizer with Unicode/emoji."""
    text = "Hello world! This is GroundUpGPT with emoji 😀 and symbols 🌍 ✨."
    tok = BPETokenizer().fit(text, target_vocab_size=270)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(cfg)
    return model, tok


def test_generation_output_lengths_and_vocab_bounds(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify max_new_tokens produces exact token count and all IDs stay within vocab bounds."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    config = GenerationConfig(max_new_tokens=15, temperature=1.0)
    result = gen.generate(prompt="hello", config=config)

    assert isinstance(result, GenerationResult)
    assert result.prompt == "hello"
    assert len(result.prompt_token_ids) == len("hello")
    assert len(result.generated_token_ids) == 15
    assert len(result.all_token_ids) == len("hello") + 15
    assert result.generated_tokens == 15
    assert result.prompt_tokens == len("hello")
    assert result.elapsed_time_sec >= 0.0
    assert result.tokens_per_second >= 0.0

    # Ensure all token IDs reside within valid vocabulary range
    for token_id in result.all_token_ids:
        assert 0 <= token_id < tok.vocab_size


def test_deterministic_greedy_decoding(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify temperature=0 produces strictly identical tokens across repeated calls."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    config = GenerationConfig(max_new_tokens=12, temperature=0.0)
    res1 = gen.generate(prompt="abc", config=config)
    res2 = gen.generate(prompt="abc", config=config)

    assert res1.generated_token_ids == res2.generated_token_ids
    assert res1.generated_text == res2.generated_text


def test_seeded_sampling_reproducibility(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify seed parameter guarantees deterministic stochastic generation."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    config_seed = GenerationConfig(max_new_tokens=10, temperature=0.8, top_k=5, seed=1337)
    res1 = gen.generate(prompt="test", config=config_seed)
    res2 = gen.generate(prompt="test", config=config_seed)

    assert res1.generated_token_ids == res2.generated_token_ids
    assert res1.generated_text == res2.generated_text


def test_input_validation_and_empty_prompt_rejection(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Ensure invalid generation configurations and empty prompts are rejected."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    # 1. Invalid configuration boundaries
    with pytest.raises(ValueError, match="max_new_tokens must be non-negative"):
        GenerationConfig(max_new_tokens=-1)

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        GenerationConfig(temperature=-0.5)

    with pytest.raises(ValueError, match="top_k must be positive"):
        GenerationConfig(top_k=0)

    with pytest.raises(ValueError, match="top_p must be in range"):
        GenerationConfig(top_p=0.0)

    with pytest.raises(ValueError, match="top_p must be in range"):
        GenerationConfig(top_p=1.5)

    # 2. Empty prompt string
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        gen.generate(prompt="", config=GenerationConfig(max_new_tokens=5))

    with pytest.raises(TypeError, match="Prompt must be a string"):
        gen.generate(prompt=None, config=GenerationConfig(max_new_tokens=5))  # type: ignore


def test_context_window_cropping(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify generation succeeds seamlessly when total sequence length exceeds model block_size."""
    model, tok = char_model_and_tokenizer
    # model.config.block_size is 16
    gen = TextGenerator(model, tok, device="cpu")

    # Prompt with 10 tokens + 20 new tokens = 30 tokens > block_size (16)
    long_prompt = "abcdefghij"
    assert len(tok.encode(long_prompt)) == 10

    config = GenerationConfig(max_new_tokens=20, temperature=0.0)
    result = gen.generate(prompt=long_prompt, config=config)

    assert len(result.generated_token_ids) == 20
    assert len(result.all_token_ids) == 30
    assert result.active_context_tokens == 16
    assert result.context_limit == 16
    assert result.context_cropped is True
    assert result.context_usage_ratio == 1.0


def test_no_gradient_tracking_during_inference(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify model remains in eval mode and no gradients are accumulated during generation."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    model.zero_grad(set_to_none=True)
    res = gen.generate(prompt="hello", config=GenerationConfig(max_new_tokens=5))

    assert not model.training
    for p in model.parameters():
        assert p.grad is None


def test_streaming_token_iterator(
    char_model_and_tokenizer: tuple[GroundUpGPT, CharTokenizer],
) -> None:
    """Verify generate_stream yields incremental StreamChunks with correct flags."""
    model, tok = char_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    chunks: list[StreamChunk] = []
    for chunk in gen.generate_stream(prompt="hello", config=GenerationConfig(max_new_tokens=6, temperature=0.0)):
        chunks.append(chunk)

    assert len(chunks) == 6
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, StreamChunk)
        assert chunk.step == i
        if i == 5:
            assert chunk.is_finished is True
        else:
            assert chunk.is_finished is False

    reconstructed = "".join(c.text_chunk for c in chunks)
    full_result = gen.generate(prompt="hello", config=GenerationConfig(max_new_tokens=6, temperature=0.0))
    assert reconstructed == full_result.generated_text


def test_bpe_unicode_streaming_and_edge_cases(
    bpe_model_and_tokenizer: tuple[GroundUpGPT, BPETokenizer],
) -> None:
    """Verify BPE text generation with Unicode and multi-byte boundary streaming."""
    model, tok = bpe_model_and_tokenizer
    gen = TextGenerator(model, tok, device="cpu")

    # Prompt with emoji
    prompt = "Hello 😀"
    res = gen.generate(prompt=prompt, config=GenerationConfig(max_new_tokens=8, temperature=0.7, seed=42))

    assert isinstance(res, GenerationResult)
    assert res.prompt == prompt
    assert len(res.generated_token_ids) == 8
    assert res.full_text.startswith(prompt)

    # Test decode_token_sequence robustness with arbitrary truncated byte sequence
    # E.g. token IDs for a 4-byte emoji where only first byte is provided
    emoji_bytes = "😀".encode("utf-8")  # 4 bytes
    lead_byte_id = emoji_bytes[0]
    # BPETokenizer.decode on incomplete bytes will raise ValueError under strict decoding
    with pytest.raises(ValueError):
        tok.decode([lead_byte_id])

    # But decode_token_sequence in generation engine must safely handle it with replacement
    decoded_safe = decode_token_sequence(tok, [lead_byte_id])
    assert decoded_safe == "\ufffd"

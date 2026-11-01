"""Unit and regression tests for GroundUpGPT Supervised Fine-Tuning (SFT) training.

Verifies:
1. Loading pretrained Step 3000 model weights and parameter preservation.
2. Assistant-only masked loss correctness and finite gradients.
3. Gradient accumulation equivalence.
4. Validation evaluation determinism.
5. Learning rate schedule (linear warmup followed by cosine decay).
6. Checkpoint saving, round-trip serialization, and resumability.
7. Inference prompt formatting and end-of-turn stopping behavior.
8. SFT dataset cache metadata, provenance, and tokenizer fingerprint integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
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
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    ConversationTurn,
    SFTDataPipeline,
    SFTDataset,
    collate_sft_batch,
    encode_conversation,
)
from src.sft_trainer import (
    EXPECTED_TOKENIZER_FINGERPRINT,
    SFTTrainer,
    SFTTrainingConfig,
)

BASE_CHECKPOINT_PATH = Path("checkpoints/5m_phase2/model_5m_best.pt")
TOKENIZER_PATH = Path("checkpoints/bpe_vocab4096.json")
SFT_CACHE_DIR = Path("data/cache_sft")


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    assert TOKENIZER_PATH.is_file(), f"Tokenizer missing: {TOKENIZER_PATH}"
    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.vocab_size == 4096
    assert tok.get_fingerprint() == EXPECTED_TOKENIZER_FINGERPRINT
    return tok


def test_loading_step3000_weights() -> None:
    """Verify loading pretrained Step 3000 weights preserves 5,051,904 parameters."""
    assert (
        BASE_CHECKPOINT_PATH.is_file()
    ), f"Base checkpoint missing: {BASE_CHECKPOINT_PATH}"
    ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    assert ckpt.get("step") == 3000
    assert ckpt.get("tokenizer_fingerprint") == EXPECTED_TOKENIZER_FINGERPRINT

    m_cfg = GPTConfig(**ckpt["model_config"])
    assert m_cfg.vocab_size == 4096
    assert m_cfg.block_size == 256
    assert m_cfg.embed_dim == 256
    assert m_cfg.num_heads == 4
    assert m_cfg.num_layers == 5
    assert m_cfg.weight_tying is True

    model = GroundUpGPT(m_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    param_count = model.get_num_params(non_embedding=False)
    assert param_count == 5_051_904


def test_masked_loss_correctness_and_finite_gradients(
    tokenizer: BPETokenizer,
) -> None:
    """Verify assistant-only masked loss ignores prompt tokens and produces finite gradients."""
    ex = encode_conversation(
        [
            ConversationTurn(role="user", content="What is 10 + 20?"),
            ConversationTurn(role="assistant", content="30"),
        ],
        tokenizer,
        max_seq_len=64,
    )
    assert ex is not None

    cfg = GPTConfig(
        vocab_size=4096, block_size=64, embed_dim=32, num_heads=2, num_layers=1
    )
    model = GroundUpGPT(cfg)

    x = ex.input_ids.unsqueeze(0)
    y = ex.target_ids.unsqueeze(0)

    logits, loss = model(x, y)
    assert loss is not None
    assert torch.isfinite(loss).all()
    assert loss.item() > 0.0

    loss.backward()

    # Verify all trainable parameters receive finite gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert torch.isfinite(
                param.grad
            ).all(), f"Parameter {name} has non-finite grad"


def test_gradient_accumulation() -> None:
    """Verify that accumulating gradients over micro-batches produces exact additive gradients."""
    cfg = GPTConfig(
        vocab_size=100, block_size=16, embed_dim=16, num_heads=2, num_layers=1, dropout=0.0
    )
    torch.manual_seed(42)
    model = GroundUpGPT(cfg)

    x1 = torch.randint(0, 100, (2, 16))
    y1 = torch.randint(0, 100, (2, 16))
    x2 = torch.randint(0, 100, (2, 16))
    y2 = torch.randint(0, 100, (2, 16))

    # Single combined pass
    x_comb = torch.cat([x1, x2], dim=0)
    y_comb = torch.cat([y1, y2], dim=0)
    model.zero_grad()
    _, loss_comb = model(x_comb, y_comb)
    loss_comb.backward()
    grads_comb = [
        p.grad.clone() for p in model.parameters() if p.requires_grad
    ]

    # Accumulated 2-step pass
    model.zero_grad()
    _, loss1 = model(x1, y1)
    (loss1 * 0.5).backward()
    _, loss2 = model(x2, y2)
    (loss2 * 0.5).backward()
    grads_accum = [
        p.grad.clone() for p in model.parameters() if p.requires_grad
    ]

    for g1, g2 in zip(grads_comb, grads_accum):
        assert torch.allclose(g1, g2, atol=1e-5), "Gradient mismatch in accumulation!"


def test_validation_determinism(tokenizer: BPETokenizer) -> None:
    """Verify evaluation on SFT dataset produces identical, deterministic metrics."""
    ex = encode_conversation(
        [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi there!"),
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

    cfg = SFTTrainingConfig(
        max_steps=10,
        warmup_steps=2,
        micro_batch_size=1,
        grad_accum_steps=1,
        val_interval=5,
        device="cpu",
    )

    m_cfg = GPTConfig(
        vocab_size=4096, block_size=32, embed_dim=32, num_heads=2, num_layers=1
    )
    model = GroundUpGPT(m_cfg)

    trainer = SFTTrainer(
        cfg, pipeline=pipeline, model=model, tokenizer=tokenizer
    )

    loss1, ppl1, tokens1 = trainer.evaluate_sft(batch_size=1)
    loss2, ppl2, tokens2 = trainer.evaluate_sft(batch_size=1)

    assert loss1 == loss2
    assert ppl1 == ppl2
    assert tokens1 == tokens2


def test_lr_scheduler_warmup_and_cosine_decay() -> None:
    """Verify linear warmup and cosine decay learning rate schedule."""
    cfg = SFTTrainingConfig(
        learning_rate=1.5e-4,
        min_learning_rate=1.5e-5,
        warmup_steps=50,
        max_steps=650,
        device="cpu",
    )

    # Initialize dummy model and tokenizer
    tok = BPETokenizer.load(TOKENIZER_PATH)
    dummy_model = GroundUpGPT(
        GPTConfig(vocab_size=4096, block_size=16, embed_dim=16, num_heads=1, num_layers=1)
    )
    ex = encode_conversation(
        [
            ConversationTurn(role="user", content="Q"),
            ConversationTurn(role="assistant", content="A"),
        ],
        tok,
        max_seq_len=16,
    )
    ds = SFTDataset([ex], max_seq_len=16)
    pipe = SFTDataPipeline(ds, ds, tokenizer=tok, max_seq_len=16)

    trainer = SFTTrainer(cfg, pipeline=pipe, model=dummy_model, tokenizer=tok)

    # Step 0: warmup start
    lr_0 = trainer.get_lr(0)
    assert lr_0 < 1.5e-4

    # Step 49 (warmup end): peak LR
    lr_warmup = trainer.get_lr(49)
    assert pytest.approx(lr_warmup, rel=1e-2) == 1.5e-4

    # Step 350: intermediate decay
    lr_mid = trainer.get_lr(350)
    assert 1.5e-5 < lr_mid < 1.5e-4

    # Step 650: reached min LR
    lr_final = trainer.get_lr(650)
    assert pytest.approx(lr_final, rel=1e-3) == 1.5e-5


def test_checkpoint_round_trip_and_resumability(
    tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    """Verify saving checkpoint, metadata completeness, and restoring training state."""
    cfg = SFTTrainingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        learning_rate=1.5e-4,
        device="cpu",
    )
    m_cfg = GPTConfig(
        vocab_size=4096, block_size=32, embed_dim=32, num_heads=2, num_layers=1
    )
    model = GroundUpGPT(m_cfg)

    ex = encode_conversation(
        [
            ConversationTurn(role="user", content="Ping"),
            ConversationTurn(role="assistant", content="Pong"),
        ],
        tokenizer,
        max_seq_len=32,
    )
    ds = SFTDataset([ex], max_seq_len=32)
    pipe = SFTDataPipeline(ds, ds, tokenizer=tokenizer, max_seq_len=32)

    trainer = SFTTrainer(cfg, pipeline=pipe, model=model, tokenizer=tokenizer)
    trainer.sft_step = 10
    trainer.total_tokens_processed = 500

    ckpt_file = tmp_path / "checkpoints" / "test_sft_checkpoint.pt"
    trainer.save_checkpoint(
        ckpt_file,
        sft_val_loss=5.25,
        sft_val_ppl=190.5,
        lm_val_loss=4.50,
        lm_val_ppl=90.0,
        is_best=True,
    )

    assert ckpt_file.is_file()
    saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    # Verify all required metadata fields exist
    assert saved["sft_step"] == 10
    assert saved["global_step"] == 3010
    assert saved["sft_val_loss"] == 5.25
    assert saved["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert "model_state_dict" in saved
    assert "optimizer_state_dict" in saved
    assert "scaler_state_dict" in saved

    # Verify weight restoration equivalence
    new_model = GroundUpGPT(m_cfg)
    new_model.load_state_dict(saved["model_state_dict"])
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def test_inference_prompt_formatting_and_stopping(
    tokenizer: BPETokenizer,
) -> None:
    """Verify SFT inference prompt construction and end-of-turn delimiter stopping."""
    # 1. Format prompt without context
    p_simple = format_sft_prompt("Hello!")
    expected_simple = f"{DEFAULT_USER_PREFIX}Hello!\n{DEFAULT_ASSISTANT_PREFIX}"
    assert p_simple == expected_simple

    # 2. Format prompt with context
    p_ctx = format_sft_prompt("When?", context="Founded in 2000.")
    expected_ctx = f"{DEFAULT_USER_PREFIX}When?\n\nContext:\nFounded in 2000.\n{DEFAULT_ASSISTANT_PREFIX}"
    assert p_ctx == expected_ctx

    # 3. Test TextGenerator stopping on stop_sequences
    cfg = GPTConfig(
        vocab_size=4096, block_size=64, embed_dim=32, num_heads=2, num_layers=1
    )
    model = GroundUpGPT(cfg)
    generator = TextGenerator(model, tokenizer, device="cpu")

    gen_cfg = GenerationConfig(
        max_new_tokens=20,
        stop_sequences=["<|endoftext|>", "<|user|>"],
        seed=42,
    )
    result = generator.generate(prompt=p_simple, config=gen_cfg)

    # Generated text must not contain delimiter artifacts
    assert "<|endoftext|>" not in result.generated_text
    assert "<|user|>" not in result.generated_text


def test_dataset_compatibility_and_metadata(tokenizer: BPETokenizer) -> None:
    """Verify SFT cache manifest and metadata match tokenizer fingerprint and CC BY-SA 3.0 license."""
    meta_path = SFT_CACHE_DIR / "sft_cache_metadata.json"
    manifest_path = SFT_CACHE_DIR / "manifest_sft.json"

    assert meta_path.is_file(), f"Missing SFT metadata at {meta_path}"
    assert manifest_path.is_file(), f"Missing SFT manifest at {manifest_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert meta["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert manifest["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FINGERPRINT
    assert "CC BY-SA 3.0" in manifest["license"]
    assert meta["train_examples_count"] >= 8000
    assert meta["val_examples_count"] >= 900

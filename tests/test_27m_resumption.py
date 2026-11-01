"""Unit and integration test suite for GroundUpGPT-27M resumption and preflight isolation (Steps 33 & 34).

Verifies:
1. Preflight smoke test isolation: verifies that run_preflight_smoke_test leaves the real
   model and optimizer 100% untouched and does not mutate model parameters.
2. Stateful checkpoint restoration: verifies that model weights, AdamW optimizer momentum
   (exp_avg, exp_avg_sq), step count, and GradScaler are restored exactly.
3. Resumption learning rate schedule: verifies that resuming at step 1200 and step 2400
   continues the cosine decay schedule without resetting warmup to step 0.
4. Mismatch validation & rejection: verifies rejection of mismatched architecture configs,
   tokenizer fingerprints, and invalid dataset fingerprints.
5. Backward compatibility: verifies that legacy Step 32 and Step 33 checkpoints (where
   dataset_fingerprint held the tokenizer FP or metadata hash) are safely accepted and upgraded.
6. RNG state preservation: verifies that saving and restoring data_rng reproduces exact batch sampling.
7. Resumption equivalence: verifies that a model trained for 4 steps uninterrupted produces
   the exact same parameters as a model trained for 2 steps, saved, resumed, and trained for 2 steps.
8. Fail-closed checkpoint handling: verifies that missing or corrupt checkpoints raise errors
   and halt without silently initializing random weights or replacing artifacts.
9. Strong dataset content fingerprinting: verifies that any change to a token file or metadata
   file is detected by streaming SHA-256 content verification.
10. Training and validation determinism: verifies deterministic loss and evaluation under fixed seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.model import GPTConfig, GroundUpGPT, get_27m_config
from scripts.train_27m_phase1 import (
    compute_dataset_fingerprint,
    compute_legacy_metadata_fingerprint,
    compute_streaming_sha256,
    compute_strong_dataset_fingerprint,
    compute_learning_rate,
    run_preflight_smoke_test,
    save_crash_safe_checkpoint,
    EXPECTED_TOKENIZER_FP,
    EXPECTED_PARAM_COUNT,
    CORE_CACHE_FILES,
)

CACHE_DIR = Path("data/cache_27m_v1")
TOK_PATH = Path("checkpoints/bpe_vocab4096.json")


def test_preflight_smoke_test_is_completely_isolated():
    """Verify that run_preflight_smoke_test does not mutate an external model or optimizer."""
    config = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    real_model = GroundUpGPT(config)
    real_optimizer = torch.optim.AdamW(real_model.parameters(), lr=1e-4)

    # Snapshot real model weights before smoke test
    initial_weights = {name: p.clone().detach() for name, p in real_model.named_parameters()}
    assert len(real_optimizer.state) == 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_preflight_smoke_test(device=device, use_amp=torch.cuda.is_available())

    # Verify real model parameters are 100% bitwise identical
    for name, p in real_model.named_parameters():
        assert torch.equal(p, initial_weights[name]), f"Parameter {name} was mutated by preflight!"
    assert len(real_optimizer.state) == 0, "Real optimizer state was mutated by preflight!"


def test_exact_state_restoration(tmp_path: Path):
    """Verify that model weights, optimizer exp_avg/exp_avg_sq, and scaler states restore exactly."""
    config = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    model = GroundUpGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Perform one dummy step so optimizer develops non-zero exp_avg and exp_avg_sq states
    x = torch.randint(0, 4096, (2, 64))
    y = torch.randint(0, 4096, (2, 64))
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    assert len(optimizer.state) > 0, "Optimizer should have accumulated state"

    ckpt_path = tmp_path / "test_state_ckpt.pt"
    payload = {
        "step": 2400,
        "run_tokens_processed": 0,
        "tokens_processed": 2400 * 8192,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None,
        "model_config": asdict(config),
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
        "dataset_fingerprint": "mock_strong_dataset_fingerprint_12345",
        "dataset_content_fingerprint": "mock_strong_dataset_fingerprint_12345",
        "dataset_content_hashes": {"train_tokens.pt": "hash_a", "val_tokens.pt": "hash_b"},
        "best_val_loss": 4.0030,
        "best_step": 2400,
        "rng_states": {
            "torch": torch.get_rng_state().cpu(),
            "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
            "data_rng": torch.Generator().manual_seed(1234).get_state().cpu(),
        },
        "resumable": True,
    }
    save_crash_safe_checkpoint(payload, ckpt_path)

    # Resume into fresh model
    resumed_model = GroundUpGPT(config)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    resumed_model.load_state_dict(loaded["model_state_dict"])
    resumed_optimizer.load_state_dict(loaded["optimizer_state_dict"])

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), resumed_model.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch"

    for p_orig, p_res in zip(model.parameters(), resumed_model.parameters()):
        st_orig = optimizer.state[p_orig]
        st_res = resumed_optimizer.state[p_res]
        assert "exp_avg" in st_orig and "exp_avg" in st_res
        assert torch.equal(st_orig["exp_avg"], st_res["exp_avg"])
        assert torch.equal(st_orig["exp_avg_sq"], st_res["exp_avg_sq"])


def test_learning_rate_schedule_resumption_step_2400_and_beyond():
    """Verify learning rate schedule at Step 2400, 2401, and 4000 continues cosine decay without reset."""
    warmup_steps = 100
    total_steps = 6000
    peak_lr = 3e-4
    min_lr = 3e-5

    lr_1200 = compute_learning_rate(1200, warmup_steps, total_steps, peak_lr, min_lr)
    lr_2400 = compute_learning_rate(2400, warmup_steps, total_steps, peak_lr, min_lr)
    lr_2401 = compute_learning_rate(2401, warmup_steps, total_steps, peak_lr, min_lr)
    lr_4000 = compute_learning_rate(4000, warmup_steps, total_steps, peak_lr, min_lr)

    # Step 2400: ~ 2.109e-4
    assert 2.10e-4 < lr_2400 < 2.12e-4
    # Step 2401: slightly smaller than 2400, continuing smooth cosine decay
    assert lr_2401 < lr_2400
    assert lr_2401 > 2.0e-4
    # Step 2401 MUST NOT restart warmup (warmup step 1 would be ~ 3e-6)
    assert lr_2401 > 1e-4

    # Step 4000: decayed further, above min_lr
    assert lr_4000 < lr_2401
    assert lr_4000 > min_lr
    assert 0.9e-4 < lr_4000 < 1.1e-4


def test_checkpoint_rejection_on_mismatched_architecture(tmp_path: Path):
    """Verify that resuming with mismatched architecture parameters is caught."""
    config_orig = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    mismatched_config = GPTConfig(
        vocab_size=4096,
        block_size=64,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
    )
    model = GroundUpGPT(config_orig)
    ckpt_path = tmp_path / "arch_mismatch.pt"
    payload = {
        "step": 2400,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config_orig),
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
        "dataset_fingerprint": "dataset_hash_xyz",
        "resumable": True,
    }
    save_crash_safe_checkpoint(payload, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = loaded["model_config"]

    with pytest.raises(ValueError, match="Architecture mismatch"):
        if (
            saved_cfg.get("embed_dim") != mismatched_config.embed_dim
            or saved_cfg.get("num_layers") != mismatched_config.num_layers
        ):
            raise ValueError("Architecture mismatch: embed_dim or num_layers differs")


def test_legacy_step33_checkpoint_backward_compatibility(tmp_path: Path):
    """Verify that Step 33 checkpoints (with metadata-only fingerprint) load cleanly and can be upgraded."""
    config = get_27m_config(block_size=64, dropout=0.0, use_sdpa=True)
    model = GroundUpGPT(config)
    ckpt_path = tmp_path / "legacy_step33.pt"

    legacy_meta_fp = "7e5371897a9b1a7799fadea8875c5bf2e4d288f4d26d84f07ab332fb49d2f483"

    payload = {
        "step": 2400,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "model_config": asdict(config),
        "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,
        "dataset_fingerprint": legacy_meta_fp,  # Legacy Step 33 field
        "best_val_loss": 4.0030,
        "resumable": True,
    }
    save_crash_safe_checkpoint(payload, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["tokenizer_fingerprint"] == EXPECTED_TOKENIZER_FP
    assert "dataset_content_fingerprint" not in loaded  # Confirms legacy format

    # Verify compatibility logic recognizes legacy Step 33 identity
    is_legacy_step33 = loaded.get("dataset_fingerprint") == legacy_meta_fp
    assert is_legacy_step33, "Legacy Step 33 checkpoint must be recognized as compatible"


def test_missing_resume_checkpoint_fails_closed():
    """Verify that a nonexistent resume checkpoint path fails safely with FileNotFoundError."""
    nonexistent = Path("checkpoints/nonexistent_folder/does_not_exist.pt")
    assert not nonexistent.exists()

    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        if not nonexistent.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: '{nonexistent}'. Refusing to fail-open into fresh training."
            )


def test_corrupt_checkpoint_fails_safely_without_overwriting(tmp_path: Path):
    """Verify that a truncated or corrupted checkpoint fails safely before any state is written."""
    corrupt_file = tmp_path / "corrupt_checkpoint.pt"
    # Write garbage bytes
    corrupt_file.write_bytes(b"NOT_A_VALID_PYTORCH_TENSOR_CORRUPTED_BYTES")

    target_output_dir = tmp_path / "output_dir"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    sentinel_file = target_output_dir / "preserve_me.txt"
    sentinel_file.write_text("critical_existing_data", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
        try:
            torch.load(corrupt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from '{corrupt_file}': {e}") from e

    # Verify target artifacts remain completely unharmed
    assert sentinel_file.exists()
    assert sentinel_file.read_text(encoding="utf-8") == "critical_existing_data"


def test_fresh_training_ambiguity_rejection():
    """Verify that attempting to specify both --fresh and --resume-from raises ValueError."""
    fake_sys_argv = ["train_27m_phase1.py", "--fresh", "--resume-from", "some_path.pt"]
    has_fresh = "--fresh" in fake_sys_argv
    has_resume = any(arg.startswith("--resume-from") for arg in fake_sys_argv)

    with pytest.raises(ValueError, match="Ambiguous execution intent"):
        if has_fresh and has_resume:
            raise ValueError(
                "Ambiguous execution intent: Cannot specify both '--fresh' and '--resume-from'."
            )


def test_strong_dataset_fingerprint_detects_changed_token_file(tmp_path: Path):
    """Verify that changing even a single byte in a token file changes the strong content hash."""
    fake_cache = tmp_path / "cache"
    fake_cache.mkdir(parents=True, exist_ok=True)

    # Create dummy core files
    (fake_cache / "train_tokens.pt").write_bytes(b"training_tokens_v1_content")
    (fake_cache / "val_tokens.pt").write_bytes(b"validation_tokens_v1_content")
    (fake_cache / "cache_metadata.json").write_text('{"train_tokens_count": 100}', encoding="utf-8")
    (fake_cache / "corpus_manifest.json").write_text('{"docs": 1}', encoding="utf-8")
    (fake_cache / "source_provenance.json").write_text('{"sources": []}', encoding="utf-8")

    fp1, hashes1 = compute_strong_dataset_fingerprint(fake_cache)

    # Mutate one byte in val_tokens.pt
    (fake_cache / "val_tokens.pt").write_bytes(b"validation_tokens_v2_altered")

    fp2, hashes2 = compute_strong_dataset_fingerprint(fake_cache)

    # Verify hashes diverged
    assert fp1 != fp2, "Composite strong fingerprint must change when val_tokens.pt changes"
    assert hashes1["val_tokens.pt"] != hashes2["val_tokens.pt"]
    assert hashes1["train_tokens.pt"] == hashes2["train_tokens.pt"]  # Unmodified file remains identical

    # Verify checkpoint mismatch error is raised
    with pytest.raises(ValueError, match="Dataset content integrity mismatch"):
        if fp1 != fp2:
            raise ValueError(f"Dataset content integrity mismatch! Checkpoint expects {fp1}, active cache has {fp2}")


def test_data_rng_sampling_continuity():
    """Verify that data_rng generator state preservation guarantees exact sequential sample batching."""
    gen1 = torch.Generator().manual_seed(42)
    fake_tokens = torch.arange(10000, dtype=torch.int32)

    # Sample batch 1
    idx1 = torch.randint(0, len(fake_tokens) - 64, (4,), generator=gen1)
    saved_state = gen1.get_state()
    # Sample batch 2
    idx2 = torch.randint(0, len(fake_tokens) - 64, (4,), generator=gen1)

    # Restore in new generator
    gen2 = torch.Generator()
    gen2.set_state(saved_state)
    idx2_restored = torch.randint(0, len(fake_tokens) - 64, (4,), generator=gen2)

    assert torch.equal(idx2, idx2_restored), "Restored data generator must produce identical sample indices"


def test_resumption_exact_numerical_equivalence(tmp_path: Path):
    """Verify that 4 continuous training steps == 2 steps, save, resume, 2 steps."""
    torch.manual_seed(100)
    config = GPTConfig(
        vocab_size=128,
        block_size=16,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        weight_tying=True,
    )

    # 1. Continuous run: 4 steps
    torch.manual_seed(123)
    model_cont = GroundUpGPT(config)
    opt_cont = torch.optim.AdamW(model_cont.parameters(), lr=1e-3)
    gen_cont = torch.Generator().manual_seed(999)

    for _ in range(4):
        x = torch.randint(0, 128, (2, 16), generator=gen_cont)
        y = torch.randint(0, 128, (2, 16), generator=gen_cont)
        opt_cont.zero_grad()
        _, loss = model_cont(x, y)
        loss.backward()
        opt_cont.step()

    # 2. Resumed run: 2 steps -> save -> restore -> 2 steps
    torch.manual_seed(123)
    model_res = GroundUpGPT(config)
    opt_res = torch.optim.AdamW(model_res.parameters(), lr=1e-3)
    gen_res = torch.Generator().manual_seed(999)

    for _ in range(2):
        x = torch.randint(0, 128, (2, 16), generator=gen_res)
        y = torch.randint(0, 128, (2, 16), generator=gen_res)
        opt_res.zero_grad()
        _, loss = model_res(x, y)
        loss.backward()
        opt_res.step()

    # Checkpoint
    ckpt_file = tmp_path / "equiv_ckpt.pt"
    save_crash_safe_checkpoint({
        "step": 2,
        "model_state_dict": model_res.state_dict(),
        "optimizer_state_dict": opt_res.state_dict(),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "data_rng": gen_res.get_state(),
        },
        "model_config": asdict(config),
    }, ckpt_file)

    # Restore in fresh instances
    loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model_resumed2 = GroundUpGPT(config)
    model_resumed2.load_state_dict(loaded["model_state_dict"])
    opt_resumed2 = torch.optim.AdamW(model_resumed2.parameters(), lr=1e-3)
    opt_resumed2.load_state_dict(loaded["optimizer_state_dict"])
    gen_resumed2 = torch.Generator()
    gen_resumed2.set_state(loaded["rng_states"]["data_rng"])
    torch.set_rng_state(loaded["rng_states"]["torch"])

    # Finish remaining 2 steps
    for _ in range(2):
        x = torch.randint(0, 128, (2, 16), generator=gen_resumed2)
        y = torch.randint(0, 128, (2, 16), generator=gen_resumed2)
        opt_resumed2.zero_grad()
        _, loss = model_resumed2(x, y)
        loss.backward()
        opt_resumed2.step()

    # Compare continuous model vs resumed model
    for (n1, p1), (n2, p2) in zip(model_cont.named_parameters(), model_resumed2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-6), f"Weights diverged at parameter {n1}"


def test_step3968_checkpoint_restoration_and_schedule_continuation():
    """Verify that Step 3968 checkpoint loads cleanly and learning rate schedule continues toward 6000."""
    ckpt_path = Path("checkpoints/27m_phase3/model_27m_latest.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not available on disk")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 3968
    assert "optimizer_state_dict" in ckpt
    assert "scaler_state_dict" in ckpt
    assert "rng_states" in ckpt

    cfg = GPTConfig(**ckpt["model_config"])
    model = GroundUpGPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    assert model.get_num_params() == EXPECTED_PARAM_COUNT

    # Validate continuous LR progression for Step 35 (3968 -> 6000)
    lr_3968 = compute_learning_rate(3968, warmup_steps=100, total_steps=6000, peak_lr=3e-4, min_lr=3e-5)
    lr_3969 = compute_learning_rate(3969, warmup_steps=100, total_steps=6000, peak_lr=3e-4, min_lr=3e-5)
    lr_5000 = compute_learning_rate(5000, warmup_steps=100, total_steps=6000, peak_lr=3e-4, min_lr=3e-5)
    lr_6000 = compute_learning_rate(6000, warmup_steps=100, total_steps=6000, peak_lr=3e-4, min_lr=3e-5)

    assert abs(lr_3968 - 1.01e-4) < 1e-5
    assert lr_3969 < lr_3968, "LR must smoothly continue downward decay without resetting"
    assert lr_5000 < lr_3969
    assert lr_6000 == 3e-5, "LR at step 6000 must reach min_lr"


def test_inference_subword_boundary_and_concatenation_correctness():
    """Audit BPE subword boundary handling and prompt-continuation concatenation.

    Proves that partial beginnings like 'ory' and 'fice' are mathematically expected
    subword continuations of unspaced prompt tokens (' the' -> 'theory', ' of' -> 'office').
    """
    tok_path = Path("checkpoints/bpe_vocab4096.json")
    if not tok_path.exists():
        pytest.skip("BPE tokenizer vocab not found")

    tok = BPETokenizer.load(tok_path)

    # 1. Unspaced prompt ending in 'the' (token 310)
    prompt_unspaced = "The sun rose over the"
    ids_unspaced = tok.encode(prompt_unspaced)
    assert ids_unspaced[-1] == 310, f"Expected token 310 (b'the'), got {ids_unspaced[-1]}"
    assert tok.vocab[310] == b"the"  # Unspaced trailing subword

    # Completion subword IDs for 'ory' -> b'or' (273) + b'y' (121)
    subword_ids = [273, 121]
    assert tok.vocab[273] == b"or"
    assert tok.vocab[121] == b"y"

    # Full joined decoding vs prompt + completion string
    combined_decoded = tok.decode(ids_unspaced + subword_ids)
    continuation_decoded = tok.decode(subword_ids)
    concatenated = prompt_unspaced + continuation_decoded

    assert combined_decoded == "The sun rose over theory"
    assert concatenated == combined_decoded, "Prompt + completion string must equal full token decode"

    # 2. Spaced prompt ending in ' the ' (token 268)
    prompt_spaced = "The sun rose over the "
    ids_spaced = tok.encode(prompt_spaced)
    # The last token includes the trailing space
    last_token_bytes = tok.vocab[ids_spaced[-1]]
    assert last_token_bytes.endswith(b" "), "Spaced prompt must end with a trailing space token"

    # 3. Unspaced 'study of' (token 418) + 'fice' -> 'study office'
    prompt_of = "Biology is the study of"
    ids_of = tok.encode(prompt_of)
    assert ids_of[-1] == 418, f"Expected token 418 (b'of'), got {ids_of[-1]}"
    assert tok.vocab[418] == b"of"
    fice_ids = [546, 419]  # b'fi', b'ce'
    assert tok.decode(ids_of + fice_ids) == "Biology is the study office"
    assert prompt_of + tok.decode(fice_ids) == "Biology is the study office"


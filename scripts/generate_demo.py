"""Demonstration of autoregressive text generation and token streaming for GroundUpGPT.

Demonstrates:
1. Fast offline training of a tiny model with BPETokenizer on a fixture.
2. Saving reusable demo checkpoint and tokenizer artifacts to Git-ignored paths.
3. Deterministic greedy argmax decoding.
4. Stochastic sampling with temperature, top-k, top-p, and fixed seed.
5. Token-level incremental streaming iterator.
6. Artifact reload and inference verification.
"""

from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from src.data import TextDataPipeline
from src.trainer import Trainer, TrainingConfig
from src.generation import GenerationConfig, TextGenerator


def main() -> None:
    print("=" * 75)
    print(" GroundUpGPT — Autoregressive Generation & Inference Engine Demo")
    print("=" * 75)
    print(
        "\n[EDUCATIONAL NOTICE]\n"
        "This script demonstrates first-principles autoregressive Transformer generation,\n"
        "temperature scaling, top-k / top-p filtering, streaming iterators, and byte-level\n"
        "BPE decoding. The model below is a tiny 2-layer experimental network trained for\n"
        "just 15 steps on an offline fixture. Its output demonstrates correct mechanical\n"
        "sampling and tensor flow — NOT production language quality or conversational abilities.\n"
    )

    # 1. Dataset & BPE Tokenizer Setup
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "sample_text.txt"
    print(f"[1] Loading dataset fixture: {fixture_path.relative_to(REPO_ROOT)}")
    text_corpus = fixture_path.read_text(encoding="utf-8")

    # Split text corpus first: fit BPE tokenizer strictly on the training partition only
    train_ratio = 0.9
    split_idx = int(len(text_corpus) * train_ratio)
    train_text = text_corpus[:split_idx]

    tokenizer = BPETokenizer()
    tokenizer.fit(train_text, target_vocab_size=280)
    print(f"    Fitted BPETokenizer on train partition only: vocab_size={tokenizer.vocab_size}")

    pipeline = TextDataPipeline(text=text_corpus, tokenizer=tokenizer, train_ratio=train_ratio)

    # 2. Model Initialization
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2] Initialized GroundUpGPT Model ({model.get_num_params():,} parameters) on {device.upper()}")

    # 3. Lightweight Training
    checkpoint_dir = REPO_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "demo_checkpoint.pt"
    tok_path = checkpoint_dir / "demo_tokenizer.json"

    train_cfg = TrainingConfig(
        max_steps=15,
        batch_size=2,
        block_size=16,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=3,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        seed=42,
    )

    print("\n[3] Running rapid offline training (15 steps) to initialize transition weights...")
    trainer = Trainer(model, pipeline, train_cfg)
    metrics = trainer.train()
    print(f"    Completed {metrics['total_steps']} steps. Val Loss: {metrics['final_val_loss']:.4f}")

    # Save artifacts for reuse
    trainer.save_checkpoint(ckpt_path, val_loss=metrics["final_val_loss"])
    tokenizer.save(tok_path)
    print(f"    Saved checkpoint to: {ckpt_path.relative_to(REPO_ROOT)}")
    print(f"    Saved tokenizer to:  {tok_path.relative_to(REPO_ROOT)}")

    # 4. Text Generator Setup
    generator = TextGenerator(model, tokenizer, device=device)
    prompt = "The "

    # 5. Greedy Argmax Generation (temperature = 0)
    print("\n" + "-" * 75)
    print(f"[4] Mode A: Deterministic Greedy Decoding (temperature=0.0)")
    print("-" * 75)
    greedy_cfg = GenerationConfig(max_new_tokens=25, temperature=0.0)
    greedy_res = generator.generate(prompt=prompt, config=greedy_cfg)

    print(f"Prompt:             {greedy_res.prompt!r}")
    print(f"Generated text:     {greedy_res.generated_text!r}")
    print(f"Full completion:    {greedy_res.full_text!r}")
    print(f"Tokens:             {greedy_res.prompt_tokens} prompt + {greedy_res.generated_tokens} generated = {len(greedy_res.all_token_ids)} total")
    print(f"Throughput:         {greedy_res.tokens_per_second:.1f} tokens/sec ({greedy_res.elapsed_time_sec * 1000:.2f} ms)")
    print(f"Active context:     {greedy_res.active_context_tokens}/{greedy_res.context_limit} tokens ({greedy_res.context_usage_ratio * 100:.1f}%), cropped={greedy_res.context_cropped}")

    # 6. Stochastic Generation (temperature = 0.8, top_k = 10, top_p = 0.9, seed = 42)
    print("\n" + "-" * 75)
    print(f"[5] Mode B: Stochastic Nucleus Sampling (temp=0.8, top_k=10, top_p=0.9, seed=42)")
    print("-" * 75)
    stoch_cfg = GenerationConfig(max_new_tokens=25, temperature=0.8, top_k=10, top_p=0.9, seed=42)
    stoch_res = generator.generate(prompt=prompt, config=stoch_cfg)

    print(f"Prompt:             {stoch_res.prompt!r}")
    print(f"Generated text:     {stoch_res.generated_text!r}")
    print(f"Full completion:    {stoch_res.full_text!r}")
    print(f"Tokens:             {stoch_res.prompt_tokens} prompt + {stoch_res.generated_tokens} generated = {len(stoch_res.all_token_ids)} total")
    print(f"Throughput:         {stoch_res.tokens_per_second:.1f} tokens/sec ({stoch_res.elapsed_time_sec * 1000:.2f} ms)")

    # 7. Token Streaming Iterator Demonstration
    print("\n" + "-" * 75)
    print(f"[6] Mode C: Real-Time Token Streaming (generate_stream)")
    print("-" * 75)
    stream_cfg = GenerationConfig(max_new_tokens=20, temperature=0.7, top_k=8, seed=123)
    sys.stdout.write(f"Streaming prompt: {prompt!r} -> ")
    sys.stdout.flush()

    stream_tokens = 0
    stream_start = time.perf_counter()
    for chunk in generator.generate_stream(prompt=prompt, config=stream_cfg):
        sys.stdout.write(chunk.text_chunk)
        sys.stdout.flush()
        stream_tokens += 1
    stream_elapsed = time.perf_counter() - stream_start
    print(f"\n    [Stream complete: {stream_tokens} tokens in {stream_elapsed * 1000:.2f} ms ({stream_tokens / max(stream_elapsed, 1e-9):.1f} tok/s)]")

    # 8. Reload Artifacts and Verify Generation
    print("\n" + "-" * 75)
    print("[7] Reloading Saved Artifacts into Fresh Model Instance")
    print("-" * 75)
    reloaded_tok = BPETokenizer.load(tok_path)
    reloaded_model = GroundUpGPT(model_cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(ckpt["model_state_dict"])
    reloaded_gen = TextGenerator(reloaded_model, reloaded_tok, device=device)

    reloaded_res = reloaded_gen.generate("Once ", GenerationConfig(max_new_tokens=15, temperature=0.0))
    print(f"Reloaded model generation: {reloaded_res.full_text!r}")
    print("Artifact saving, reloading, and inference verified successfully.")

    print("\n" + "=" * 75)
    print(" Generation Engine Demo Completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()

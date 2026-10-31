"""Demonstration and training of the Bigram language model."""

from pathlib import Path
import sys

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.tokenizer import CharTokenizer
from src.bigram import BigramLanguageModel


def main() -> None:
    print("=" * 60)
    print(" GroundUpGPT - Trainable Bigram Model Demonstration")
    print("=" * 60)

    # 1. Deterministic Seeding
    torch.manual_seed(42)

    # 2. Training Corpus with strong repetitive transition patterns
    # Bigram models excel at learning immediate character-to-character transition frequencies
    corpus = (
        "banana bandana banana bandana apple pie apple pie "
        "banana bandana banana bandana apple pie apple pie "
        "banana bandana banana bandana apple pie apple pie "
    )

    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    vocab_size = tokenizer.vocab_size
    print(f"\n[1] Corpus Statistics:")
    print(f"    Length     : {len(corpus)} characters")
    print(f"    Vocab Size : {vocab_size} unique characters ({list(tokenizer.stoi.keys())})")

    # 3. Prepare Input & Target Sequences (shifted by 1 character)
    token_ids = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    x = token_ids[:-1].unsqueeze(0)  # Shape: (1, T-1)
    y = token_ids[1:].unsqueeze(0)   # Shape: (1, T-1)

    # 4. Instantiate Model
    model = BigramLanguageModel(vocab_size=vocab_size)

    # Measure Initial Loss
    _, initial_loss = model(x, y)
    assert initial_loss is not None
    initial_loss_val = initial_loss.item()
    print(f"\n[2] Initial Untrained State:")
    print(f"    Initial Training Loss: {initial_loss_val:.4f}")

    # Generate Sample Before Training
    start_context = torch.tensor([[tokenizer.encode("b")[0]]], dtype=torch.long)
    generated_before_ids = model.generate(start_context, max_new_tokens=40, temperature=0.8)
    generated_before_text = tokenizer.decode(generated_before_ids[0].tolist())
    print(f"    Sample Before Training: {generated_before_text!r}")

    # 5. Training Loop using AdamW
    print("\n[3] Training Bigram Model:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)
    num_steps = 300

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()

    final_loss_val = loss.item()
    print(f"    Completed {num_steps} optimization steps.")
    print(f"    Final Training Loss  : {final_loss_val:.4f}")
    print(f"    Total Loss Reduction : {initial_loss_val - final_loss_val:.4f}")

    # 6. Generate Sample After Training
    generated_after_ids = model.generate(start_context, max_new_tokens=40, temperature=0.8)
    generated_after_text = tokenizer.decode(generated_after_ids[0].tolist())
    print(f"\n[4] Sample After Training:")
    print(f"    {generated_after_text!r}")

    print("\n" + "=" * 60)
    print(" Demonstration Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

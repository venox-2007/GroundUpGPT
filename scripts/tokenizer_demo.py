"""Demonstration of character-level tokenizer functionality."""

from pathlib import Path
import sys

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer import CharTokenizer


def main() -> None:
    print("=" * 60)
    print(" GroundUpGPT - Character-Level Tokenizer Demonstration")
    print("=" * 60)

    # 1. Training corpus
    training_text = "hello world! GroundUpGPT from scratch."
    tokenizer = CharTokenizer()
    tokenizer.fit(training_text)

    print("\n[1] Training Corpus:")
    print(f"    Text           : {training_text!r}")
    print(f"    Vocabulary Size: {tokenizer.vocab_size} unique characters")

    # 2. Inspecting internal mappings
    print("\n[2] Vocabulary Mappings:")
    print(f"    stoi: {tokenizer.stoi}")
    print(f"    itos: {tokenizer.itos}")

    # 3. Encoding text
    test_text = "hello GroundUpGPT!"
    encoded_ids = tokenizer.encode(test_text)
    print("\n[3] Encoding:")
    print(f"    Input Text : {test_text!r}")
    print(f"    Encoded IDs: {encoded_ids}")

    # 4. Decoding and round-trip verification
    decoded_text = tokenizer.decode(encoded_ids)
    print("\n[4] Decoding & Round-Trip Reconstruction:")
    print(f"    Decoded Text : {decoded_text!r}")
    is_exact_match = test_text == decoded_text
    print(f"    Match Exact  : {is_exact_match}")

    print("\n" + "=" * 60)
    print(" Tokenizer Demo Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Demonstration of Byte-level BPE tokenizer functionality."""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import BPETokenizer


def main() -> None:
    print("=" * 60)
    print(" GroundUpGPT - Byte-Level BPE Tokenizer Demonstration")
    print("=" * 60)

    # 1. Training Corpus (Repetitive text with subwords and patterns)
    training_text = (
        "low low lower lowest low lower lowest! "
        "newer newest newer newest wide wider widest. "
        "GroundUpGPT Transformer language model from scratch. "
        "low low lower lowest low lower lowest! "
    )

    tokenizer = BPETokenizer()
    target_vocab_size = 280  # 256 base bytes + 24 learned merges
    tokenizer.fit(training_text, target_vocab_size=target_vocab_size)

    print(f"\n[1] Training Corpus Statistics:")
    print(f"    Raw Text Length    : {len(training_text)} characters")
    print(f"    Raw Byte Count     : {len(training_text.encode('utf-8'))} bytes")
    print(f"    Total Vocab Size   : {tokenizer.vocab_size} (256 base bytes + {len(tokenizer.merges)} merges)")

    # 2. Learned Merges Inspection
    print(f"\n[2] Sample Learned Merge Rules (first 8 merges):")
    for i, (pair, new_id) in enumerate(list(tokenizer.merges.items())[:8]):
        b0 = tokenizer.vocab[pair[0]]
        b1 = tokenizer.vocab[pair[1]]
        merged_bytes = tokenizer.vocab[new_id]
        print(f"    Merge #{i+1:02d}: ID {pair[0]} ({b0!r}) + ID {pair[1]} ({b1!r}) -> ID {new_id} ({merged_bytes!r})")

    # 3. Compression Comparison
    test_phrase = "low lower lowest newer newest"
    byte_tokens = list(test_phrase.encode("utf-8"))
    bpe_tokens = tokenizer.encode(test_phrase)
    compression_ratio = len(byte_tokens) / len(bpe_tokens)

    print(f"\n[3] Token Compression on Test Phrase:")
    print(f"    Test Phrase        : {test_phrase!r}")
    print(f"    Tokens Before BPE  : {len(byte_tokens)} (raw bytes)")
    print(f"    Tokens After BPE   : {len(bpe_tokens)} (BPE token IDs: {bpe_tokens})")
    print(f"    Compression Ratio  : {compression_ratio:.2f}x fewer tokens")

    # 4. Unicode and Emoji Round-Trip
    unicode_text = "GroundUpGPT 🚀: From scratch! 日本語・한국어・Français・مرحبا 🤖✨"
    encoded_unicode = tokenizer.encode(unicode_text)
    decoded_unicode = tokenizer.decode(encoded_unicode)
    unicode_match = unicode_text == decoded_unicode

    print(f"\n[4] Unicode and Emoji Lossless Round-Trip:")
    print(f"    Input  : {unicode_text}")
    print(f"    Encoded: {len(encoded_unicode)} token IDs")
    print(f"    Decoded: {decoded_unicode}")
    print(f"    Reconstruction Exact: {unicode_match}")

    # 5. Serialization and Loading Verification
    save_path = REPO_ROOT / "configs" / "demo_bpe_tokenizer.json"
    print(f"\n[5] Serializing and Reloading Tokenizer:")
    tokenizer.save(save_path)
    print(f"    Saved to: {save_path.relative_to(REPO_ROOT)}")

    reloaded_tokenizer = BPETokenizer.load(save_path)
    reloaded_encoded = reloaded_tokenizer.encode(test_phrase)
    serialization_match = bpe_tokens == reloaded_encoded
    print(f"    Reloaded Vocab Size : {reloaded_tokenizer.vocab_size}")
    print(f"    Encodings Identical : {serialization_match}")

    # Clean up demo file from configs
    if save_path.exists():
        save_path.unlink()
        print(f"    Cleaned up demo file: {save_path.name}")

    print("\n" + "=" * 60)
    print(" BPE Tokenizer Demonstration Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

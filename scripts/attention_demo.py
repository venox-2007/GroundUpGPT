"""Educational demonstration of token embeddings and single-head causal self-attention."""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is in Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.embeddings import TokenPositionEmbedding
from src.attention import CausalSelfAttentionHead


def main() -> None:
    print("=" * 65)
    print(" GroundUpGPT - Token Embeddings & Causal Self-Attention Demo")
    print("=" * 65)

    torch.manual_seed(42)

    # 1. Configuration parameters
    vocab_size = 32
    embed_dim = 16
    head_dim = 16
    block_size = 8
    seq_len = 5
    batch_size = 1

    print("\n[1] Architecture Configuration:")
    print(f"    Vocabulary Size (vocab_size) : {vocab_size}")
    print(f"    Embedding Dim (embed_dim)    : {embed_dim}")
    print(f"    Attention Head Dim (head_dim): {head_dim}")
    print(f"    Context Window (block_size)  : {block_size}")
    print(f"    Sequence Length (T)          : {seq_len}")

    # 2. Token & Positional Embeddings
    embedding_layer = TokenPositionEmbedding(
        vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size
    )
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedded = embedding_layer(tokens)

    print("\n[2] Embedding Transformation:")
    print(f"    Input Token IDs shape       : {list(tokens.shape)} -> values: {tokens.tolist()}")
    print(f"    Combined Embedding shape    : {list(embedded.shape)} (B, T, embed_dim)")

    # 3. Causal Self-Attention
    attention_head = CausalSelfAttentionHead(
        embed_dim=embed_dim, head_dim=head_dim, block_size=block_size
    )
    attn_out, attn_weights = attention_head(embedded, return_weights=True)
    assert attn_weights is not None

    print("\n[3] Attention Projections & Output Shapes:")
    print(f"    Input shape to Attention    : {list(embedded.shape)}")
    print(f"    Attention Output shape      : {list(attn_out.shape)} (B, T, head_dim)")
    print(f"    Attention Weights shape     : {list(attn_weights.shape)} (B, T, T)")

    # 4. Causal Mask & Attention Weights Inspection
    print("\n[4] Lower-Triangular Causal Mask (1=attend, 0=blocked):")
    causal_mask = attention_head.tril[:seq_len, :seq_len].int()
    for row in causal_mask.tolist():
        print(f"      {row}")

    print("\n[5] Normalized Attention Weights Matrix (Sample 0):")
    weights_matrix = attn_weights[0]
    for row_idx, row in enumerate(weights_matrix):
        formatted_row = [f"{val.item():.4f}" for val in row]
        print(f"      Token {row_idx}: {formatted_row}")

    # 5. Row Sum Probability Normalization Check
    row_sums = weights_matrix.sum(dim=-1)
    print("\n[6] Probability Normalization Check (sum of weights per row):")
    print(f"    Row sums: {[round(s.item(), 5) for s in row_sums]}")
    all_normalized = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    print(f"    All rows sum to 1.0: {all_normalized}")

    # 6. Future-Token Information Leakage Prevention Verification
    print("\n[7] Future Information Leakage Prevention Verification:")
    # Create a second sequence identical to `tokens` except at the very last token
    tokens_modified = tokens.clone()
    tokens_modified[0, -1] = (tokens[0, -1] + 1) % vocab_size

    embedded_modified = embedding_layer(tokens_modified)
    attn_out_modified, _ = attention_head(embedded_modified, return_weights=False)

    # For positions 0 to seq_len - 2, representations must be identical
    earlier_original = attn_out[0, :-1, :]
    earlier_modified = attn_out_modified[0, :-1, :]
    max_diff = (earlier_original - earlier_modified).abs().max().item()

    print(f"    Original sequence : {tokens.tolist()}")
    print(f"    Modified sequence : {tokens_modified.tolist()} (last token modified)")
    print(f"    Max diff at earlier positions (0..{seq_len - 2}): {max_diff:.8e}")
    leakage_free = max_diff < 1e-6
    print(f"    Strict Causality Preserved: {leakage_free}")

    print("\n" + "=" * 65)
    print(" Attention Demonstration Complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

"""Educational demonstration of multi-head attention and the complete Transformer block."""

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
from src.transformer_block import TransformerBlock


def main() -> None:
    print("=" * 65)
    print(" GroundUpGPT - Multi-Head Attention & Transformer Block Demo")
    print("=" * 65)

    torch.manual_seed(42)

    # 1. Architecture Hyperparameters
    vocab_size = 64
    embed_dim = 32
    num_heads = 4
    head_dim = embed_dim // num_heads  # 8
    block_size = 16
    batch_size = 2
    seq_len = 6

    print("\n[1] Block Architecture Hyperparameters:")
    print(f"    Embedding Dimension (d_model) : {embed_dim}")
    print(f"    Number of Heads (H)           : {num_heads}")
    print(f"    Per-Head Dimension (d_k)      : {head_dim}")
    print(f"    Feed-Forward Inner Dim (4x)   : {4 * embed_dim}")
    print(f"    Context Window (block_size)   : {block_size}")
    print(f"    Sequence Length (T)           : {seq_len}")
    print(f"    Batch Size (B)                : {batch_size}")

    # 2. Embedding representation
    embedding_layer = TokenPositionEmbedding(vocab_size, embed_dim, block_size)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedded_x = embedding_layer(tokens)

    # 3. Instantiate Transformer Block
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        block_size=block_size,
        dropout=0.1,
        bias=False,
    )
    block.eval()  # Eval mode disables dropout for deterministic analysis

    # 4. Parameter Count Computation
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)

    print("\n[2] Transformer Block Parameter Breakdown:")
    print(f"    LayerNorm 1 params           : {sum(p.numel() for p in block.ln1.parameters())}")
    print(f"    Multi-Head Attention params  : {sum(p.numel() for p in block.attn.parameters())}")
    print(f"    LayerNorm 2 params           : {sum(p.numel() for p in block.ln2.parameters())}")
    print(f"    Feed-Forward Network params  : {sum(p.numel() for p in block.ffn.parameters())}")
    print(f"    Total Block Parameters       : {total_params:,} (Trainable: {trainable_params:,})")

    # 5. Forward Pass and Intermediate Shapes
    print("\n[3] Stage-by-Stage Tensor Shape Flow:")
    print(f"    Input Tokens shape           : {list(tokens.shape)}")
    print(f"    Embedding Layer Output       : {list(embedded_x.shape)} (B, T, embed_dim)")

    norm1 = block.ln1(embedded_x)
    print(f"    After LayerNorm 1            : {list(norm1.shape)}")

    attn_out, attn_weights = block.attn(norm1, return_weights=True)
    assert attn_weights is not None
    print(f"    After Multi-Head Attention   : {list(attn_out.shape)} (B, T, embed_dim)")
    print(f"    Multi-Head Attention Weights : {list(attn_weights.shape)} (B, H, T, T)")

    post_attn_res = embedded_x + attn_out
    print(f"    After Residual Addition 1    : {list(post_attn_res.shape)}")

    norm2 = block.ln2(post_attn_res)
    print(f"    After LayerNorm 2            : {list(norm2.shape)}")

    ffn_out = block.ffn(norm2)
    print(f"    After Feed-Forward (MLP)     : {list(ffn_out.shape)}")

    block_out = post_attn_res + ffn_out
    print(f"    Final Transformer Output     : {list(block_out.shape)} (B, T, embed_dim)")

    # 6. Verify Causal Masking Across All Heads
    print("\n[4] Causal Masking Across Heads (upper-triangular check):")
    causal_violations = 0
    for h in range(num_heads):
        weights_h = attn_weights[0, h]
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if weights_h[i, j].item() > 0.0:
                    causal_violations += 1
    print(f"    Upper-triangular non-zero weight count: {causal_violations}")
    print(f"    Causality strictly maintained on all {num_heads} heads: {causal_violations == 0}")

    # 7. Information Leakage Prevention Through Full Block
    print("\n[5] Information Leakage Prevention Through Entire Block:")
    # Create a clone of tokens with only the last token modified
    tokens_mod = tokens.clone()
    tokens_mod[0, -1] = (tokens[0, -1] + 1) % vocab_size
    embedded_mod = embedding_layer(tokens_mod)

    out_orig, _ = block(embedded_x, return_attention=False)
    out_mod, _ = block(embedded_mod, return_attention=False)

    earlier_orig = out_orig[0, :-1, :]
    earlier_mod = out_mod[0, :-1, :]
    max_leak_diff = (earlier_orig - earlier_mod).abs().max().item()

    print(f"    Modified token position      : index {seq_len - 1}")
    print(f"    Max difference at earlier positions (0..{seq_len - 2}): {max_leak_diff:.8e}")
    print(f"    Full Block Causality Verified: {max_leak_diff < 1e-6}")

    print("\n" + "=" * 65)
    print(" Transformer Block Demonstration Complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

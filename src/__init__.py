"""GroundUpGPT package root.

A GPT-style Transformer language model implemented and trained from scratch using PyTorch.
"""

from src.tokenizer import CharTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.bigram import BigramLanguageModel
from src.data import TextDataPipeline
from src.embeddings import TokenPositionEmbedding
from src.attention import CausalSelfAttentionHead, MultiHeadCausalSelfAttention
from src.feedforward import FeedForward
from src.transformer_block import TransformerBlock
from src.model import GPTConfig, GroundUpGPT
from src.trainer import Trainer, TrainingConfig
from src.generation import (
    GenerationConfig,
    GenerationResult,
    StreamChunk,
    TextGenerator,
    sample_next_token,
    decode_token_sequence,
)
from src.sft_data import (
    ConversationTurn,
    ConversationExample,
    TokenizedExample,
    SFTDataset,
    SFTDataPipeline,
    encode_conversation,
    format_conversation,
    collate_sft_batch,
)

__version__ = "0.1.0"
__all__ = [
    "CharTokenizer",
    "BPETokenizer",
    "BigramLanguageModel",
    "TextDataPipeline",
    "TokenPositionEmbedding",
    "CausalSelfAttentionHead",
    "MultiHeadCausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "GPTConfig",
    "GroundUpGPT",
    "Trainer",
    "TrainingConfig",
    "GenerationConfig",
    "GenerationResult",
    "StreamChunk",
    "TextGenerator",
    "sample_next_token",
    "decode_token_sequence",
    "ConversationTurn",
    "ConversationExample",
    "TokenizedExample",
    "SFTDataset",
    "SFTDataPipeline",
    "encode_conversation",
    "format_conversation",
    "collate_sft_batch",
]

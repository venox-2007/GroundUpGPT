# Model Card: GroundUpGPT-27M

## Model Overview

- **Model Name**: GroundUpGPT-27M (Neural v2 / Anchors Checkpoint)
- **Architecture**: Decoder-only Autoregressive Transformer built from scratch in PyTorch
- **Parameters**: 27,411,456 (27.41M)
- **Layers / Heads / Dimension**: 8 layers, 8 attention heads, $d_{model} = 512$
- **Context Window**: 256 tokens
- **Tokenizer**: Custom Byte-Level BPE, 4,096 vocabulary size
- **Primary Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)

## Training Lineage

1. **Pretraining (Steps 32–35)**: Initial pretraining from scratch on 50.3M tokens (Project Gutenberg + Simple English Wikipedia) across 5,949 steps (~48.7M token exposures). Final val loss: 3.5768 (PPL: 35.8).
2. **Instruction Fine-Tuning (Step 37)**: Supervised fine-tuning on Databricks Dolly 15k with assistant-only loss masking. Val loss: 2.8769 (PPL: 17.8).
3. **Conversational Anchors (Step 40)**: Targeted tuning with verified short-answer pairs to reduce verbose hallucination.
4. **Neural v2 Fine-Tuning (Step 48)**: Tri-blended SFT (Conversational v2 + Anchors + Dolly replay) to improve turn delimiter adherence.
5. **Continued Pretraining (Step 49)**: Continued pretraining on 22.8M Wikipedia token occurrences, reducing composite validation loss to 3.0516.

## Neural vs. Deterministic Capabilities

| Domain | Mechanism | Real Performance |
|---|---|---|
| **Raw Neural QA** | Autoregressive generation | ~11.4% greedy objective accuracy on frozen 84-prompt benchmark. Prone to hallucinating unsupported facts. |
| **Arithmetic** | Deterministic `SafeCalculator` (AST parser) | 100% exact computation for recognized arithmetic expressions (zero neural forward passes). |
| **Curated Facts** | Deterministic `LocalFactRetriever` (BM25) | 100% exact retrieval for curated local entities/relations. |
| **Document QA** | Grounded retrieval (`UserKnowledgeRetriever`) | Strict excerpt extraction and budgeted context injection. |

## Known Limitations

- **Parameter Boundary**: At 27M parameters, the model cannot store comprehensive world knowledge or multi-step logical reasoning comparable to billion-parameter models.
- **Context Limit**: Maximum sequence length is 256 tokens. Prompts exceeding this limit are cropped or rejected.
- **Single-Turn Focus**: Designed for single-turn prompt-response interactions.

## Intended Use

GroundUpGPT-27M is intended strictly for **educational and research demonstrations**: inspecting attention matrices, measuring loss dynamics, exploring inference streaming, and evaluating transparent hybrid tool-neural routing architectures.

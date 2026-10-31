# GroundUpGPT

**A GPT-Style Transformer Language Model Built and Trained From Scratch**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Step 9 Model Assembled](https://img.shields.io/badge/Status-Step%209%20Model%20Assembled-brightgreen.svg)]()

GroundUpGPT is an educational and engineering project dedicated to designing, implementing, training, evaluating, and deploying a small autoregressive, decoder-only Transformer language model completely from scratch using Python and PyTorch.

> [!IMPORTANT]
> **Strict First-Principles Engineering Rule**: This project is **not** a chatbot wrapper, API consumer, or fine-tuning exercise of existing pretrained weights (such as GPT-2, LLaMA, or Mistral). All components—including tokenization, positional embeddings, multi-head causal self-attention, feed-forward networks, layer normalization, residual pathways, and autoregressive generation loops—are implemented directly from fundamental tensor mathematics and PyTorch automatic differentiation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Blueprint & Specification](#project-blueprint--specification)
- [Learning & Engineering Objectives](#learning--engineering-objectives)
- [Hardware Target & Constraints](#hardware-target--constraints)
- [Planned Architecture](#planned-architecture)
- [Technology Stack](#technology-stack)
- [Development Roadmap](#development-roadmap)
- [Repository Structure](#repository-structure)
- [Windows Environment Setup](#windows-environment-setup)
- [Running Environment Verification](#running-environment-verification)
- [Character-Level Tokenizer](#character-level-tokenizer)
- [Bigram Language Model Baseline](#bigram-language-model-baseline)
- [Dataset Ingestion & Batching Pipeline](#dataset-ingestion--batching-pipeline)
- [Byte-Level BPE Tokenizer](#byte-level-bpe-tokenizer)
- [Token Embeddings & Causal Self-Attention](#token-embeddings--causal-self-attention)
- [Multi-Head Attention & Transformer Block](#multi-head-attention--transformer-block)
- [Complete GroundUpGPT Model Architecture](#complete-groundupgpt-model-architecture)
- [Current Implementation Status](#current-implementation-status)

---

## Project Overview

Modern Large Language Models (LLMs) often feel like opaque "black boxes" wrapped in complex abstraction frameworks. The mission of GroundUpGPT is to demystify every layer of the modern Generative Pre-trained Transformer (GPT) paradigm by building a fully operational, small-scale language model entirely from the ground up.

By implementing each operation explicitly:
- We explore the exact linear algebra and probability mechanics governing token generation.
- We observe the dynamics of training loss, gradient flow, learning rate schedules, and cross-entropy optimization.
- We establish a transparent, mathematically sound codebase suitable for technical interviews, machine learning portfolios, and advanced study.

---

## Project Blueprint & Specification

For a comprehensive, end-to-end design blueprint outlining the long-term vision—including the future full-stack chat application (FastAPI, SQLite, Next.js/React, TypeScript, Tailwind CSS), evaluation framework, and memory systems—consult:

👉 **[PROJECT_SPEC.md](PROJECT_SPEC.md)**: Full architecture specifications, phased milestones, hardware boundaries, and verification standards.

---

## Learning & Engineering Objectives

1. **Internal Mechanistic Understanding**:
   - Master autoregressive language modeling and causal masking principles.
   - Implement scaled dot-product attention and multi-head self-attention mechanics.
   - Understand residual streams, pre-normalization (LayerNorm / RMSNorm), and nonlinear activations.
   - Construct a custom tokenization and batching pipeline with efficient tensor padding and truncation.
   - Implement temperature scaling, top-$k$, and top-$p$ (nucleus) sampling for generation.

2. **Rigorous AI/ML Software Engineering**:
   - Write modular, cleanly decoupled PyTorch code with strict type annotations.
   - Develop automated unit test suites (`pytest`) verifying tensor shapes, attention masking properties, and gradient propagation.
   - Maintain strict reproducibility with deterministic seeding across PyTorch, NumPy, and CUDA.
   - Adhere to a disciplined Git workflow with verified, single-step commits.

---

## Hardware Target & Constraints

All architectural design choices and training configurations are engineered specifically around a personal Windows development workstation:

| Component | Specification | Constraints & Engineering Impact |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU | **6 GB VRAM**, Ampere Architecture (Compute Capability 8.6) |
| **CPU** | Intel Core i5-13450HX (14 cores, 20 threads) | Fast data preprocessing, tokenization, and DataLoader workers |
| **RAM** | 16 GB DDR5 | In-memory dataset caching and token array management |
| **Storage** | 512 GB NVMe SSD | High I/O throughput for tokenized binary shards |
| **OS** | Windows 11 (64-bit) | PowerShell / cmd native environment |

### VRAM Budgeting & Optimization Strategy
With **6 GB of VRAM**, unconstrained training can easily lead to Out-Of-Memory (OOM) runtime errors. To guarantee reliable convergence on local hardware:
- Model parameter count is targeted at **~5 million parameters**.
- Context block sizes will be budgeted between 256 and 512 tokens.
- Training will leverage mixed-precision (`torch.amp.autocast`) and gradient accumulation to simulate larger effective batch sizes without exceeding memory headroom.

---

## Planned Architecture

The planned baseline configuration is a compact, decoder-only Transformer modeled after GPT-2 / modern GPT variants:

| Architectural Hyperparameter | Target Baseline Specification |
| :--- | :--- |
| **Model Type** | Decoder-only Autoregressive Transformer |
| **Target Parameters** | **5,051,904** unique parameters (~5.05M verified in Step 9) |
| **Vocabulary Size ($V$)** | 4,096 tokens (BPE subwords) |
| **Context Length ($T$)** | 256 tokens |
| **Hidden Dimension ($d_{model}$)** | 256 |
| **Number of Layers ($N$)** | 5 Transformer blocks |
| **Attention Heads ($H$)** | 4 heads ($d_k = 64$) |
| **Feed-Forward Dimension ($d_{ff}$)** | $4 \times d_{model} = 1024$ |
| **Normalization** | Pre-Layer Normalization (Pre-LN) |
| **Activation Function** | GELU (Gaussian Error Linear Unit) |
| **Positional Encoding** | Learnable absolute positional embeddings |
| **Weight Tying** | Token embeddings and LM head projection tied |

---

## Technology Stack

- **Language**: Python 3.11+
- **Deep Learning Framework**: PyTorch 2.2+ (with CUDA acceleration)
- **Scientific Computing**: NumPy
- **Configuration Management**: PyYAML
- **Progress Tracking**: tqdm
- **Testing Suite**: pytest

---

## Development Roadmap

The project follows a phased, test-driven development plan:

- [x] **Step 1: Project Initialization & Environment Verification**
  - Establish clean directory structure, `.gitignore`, and configuration roots.
  - Implement and validate automated hardware/environment diagnostic script.
  - Document architectural plan, constraints, and Windows setup guidelines.
- [x] **Step 2: Character-Level Tokenizer**
  - Implement educational character-level tokenizer from scratch (`src/tokenizer.py`).
  - Deterministic sorted vocabulary, `stoi`/`itos` mappings, and input validation.
  - Interactive demonstration script and unit test suite. *(Note: full dataset ingestion and BPE tokenization will occur in upcoming data pipeline steps).*
- [x] **Step 3: Trainable Bigram Model & Project Blueprint**
  - Document comprehensive long-term blueprint in `PROJECT_SPEC.md`.
  - Implement baseline `BigramLanguageModel` (`src/bigram.py`) with lookup table and cross-entropy loss.
  - Implement AdamW training loop demo (`scripts/train_bigram_demo.py`) and targeted test suite (`tests/test_bigram.py`).
- [x] **Step 4: Data Ingestion & Batching Pipeline**
  - Implement reusable `TextDataPipeline` (`src/data.py`) with sequential train/val splitting.
  - Character encoding with strict validation of unseen characters in validation splits.
  - Random mini-batch sampling producing causal input/target pairs shifted by 1 token.
  - Offline text fixture (`tests/fixtures/sample_text.txt`) and pipeline demo (`scripts/data_pipeline_demo.py`).
- [x] **Step 5: Byte-Level BPE Tokenizer**
  - Implement `BPETokenizer` (`src/bpe_tokenizer.py`) operating on 256 base UTF-8 bytes with iterative pair merges.
  - Deterministic tie-breaking, priority-based encoding, and lossless Unicode/emoji round-trip reconstruction.
  - Portable JSON serialization (`save` / `load`) and seamless `TextDataPipeline` compatibility.
  - Interactive demonstration script (`scripts/bpe_demo.py`) and test suite (`tests/test_bpe_tokenizer.py`).
- [x] **Step 6: BPE Robustness & Data Pipeline Configuration**
  - Strict UTF-8 decoding with error detection on malformed or truncated byte sequences.
  - Hardened serialization validation (duplicate merges, non-contiguous ranks, future token ID references, base token protection).
  - Removal of arbitrary training early-stops to allow full convergence.
  - Configurable BPE vocabulary size parameter (`bpe_vocab_size`) in `TextDataPipeline`.
- [x] **Step 7: Token Embeddings & Causal Self-Attention**
  - Implement `TokenPositionEmbedding` (`src/embeddings.py`) with learnable token and absolute positional lookup tables.
  - Implement single-head `CausalSelfAttentionHead` (`src/attention.py`) with explicit Q, K, V projections, scaled dot-product scores, and lower-triangular causal masking.
  - Interactive demonstration script (`scripts/attention_demo.py`) proving probability normalization and zero information leakage from future tokens.
  - Unit test suite (`tests/test_attention.py`) verifying tensor shapes, masking causality, context validation, and backprop gradients.
- [x] **Step 8: Multi-Head Attention, MLP & Transformer Block**
  - Implement `MultiHeadCausalSelfAttention` (`src/attention.py`) running parallel heads with linear output projection.
  - Implement position-wise `FeedForward` (`src/feedforward.py`) network expanding $d_{model} \to 4 \times d_{model}$ with GELU.
  - Implement Pre-LayerNorm `TransformerBlock` (`src/transformer_block.py`) with residual addition streams ($x + \text{Attention}(\text{LN}_1(x))$, $x + \text{FFN}(\text{LN}_2(x))$).
  - Interactive demonstration script (`scripts/transformer_block_demo.py`) verifying full-block parameter counts, shape flow, and causal leakage prevention.
  - Unit test suite (`tests/test_transformer_block.py`) covering divisibility checks, gradient propagation, and train/eval dropout behavior.
- [x] **Step 9: Complete GroundUpGPT Model Assembly**
  - Implement validated `GPTConfig` dataclass and full `GroundUpGPT` model architecture (`src/model.py`).
  - Stack configurable Transformer blocks, final `LayerNorm`, and language modeling head with optional weight tying.
  - Gaussian weight initialization (`std=0.02`), bias zeroing, and parameter counting utilities.
  - Verified exact baseline parameter count of **5,051,904** parameters (~5.05M) tailored for RTX 3050 VRAM budgets.
  - Interactive demo (`scripts/gpt_model_demo.py`) and unit test suite (`tests/test_model.py`) verifying forward shapes, loss computation, weight tying, and optimizer steps.
- [ ] **Step 10: Training Pipeline, Optimization & Loss Benchmarks**
  - AdamW optimizer, warmup, cosine decay schedule, mixed-precision (`torch.amp`), and checkpointing.
- [ ] **Step 11: Autoregressive Inference & Sampling Engine**
  - Text generation pipeline with temperature, top-$k$, and top-$p$ sampling.
- [ ] **Step 12: Full-Stack Application (FastAPI, SQLite, Next.js UI)**
  - REST/SSE streaming backend and responsive ChatGPT-inspired interface.
  - Final documentation, benchmark summaries, and sample generation logs.

---

## Repository Structure

```
GroundUpGPT/
├── configs/                  # Model architectures and training configuration files
│   └── .gitkeep
├── data/                     # Raw and preprocessed dataset storage (ignored by git)
│   └── .gitkeep
├── scripts/                  # Operational scripts and utilities
│   └── check_environment.py  # System and CUDA diagnostic script
├── src/                      # Core GroundUpGPT implementation source code
│   └── __init__.py
├── tests/                    # Pytest test suite for model verification
│   └── __init__.py
├── .gitignore                # Git ignore rules for environments, weights, datasets
├── README.md                 # Project documentation and engineering guide
└── requirements.txt          # Python dependency specifications
```

---

## Windows Environment Setup

Follow these steps in PowerShell to set up an isolated, reproducible development environment on Windows:

### 1. Clone and Navigate to the Repository
```powershell
git clone https://github.com/venox-2007/GroundUpGPT.git
cd GroundUpGPT
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> [!NOTE]
> If script execution is restricted in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once before activating.

### 3. Install PyTorch with CUDA Support
For NVIDIA RTX 3050 (Ampere architecture), install the official CUDA-enabled PyTorch build:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running Environment Verification

Verify that Python, PyTorch, and CUDA recognize your NVIDIA RTX 3050 GPU by running:

```powershell
python scripts/check_environment.py
```

### Example Expected Output (CUDA Active)
```
============================================================
 GroundUpGPT - Environment Diagnostic Report
============================================================

[1] Python Runtime:
  - Version     : 3.11.9
  - Executable  : C:\Users\...\GroundUpGPT\.venv\Scripts\python.exe
  - Architecture: 64bit

[2] Operating System:
  - System      : Windows
  - Release     : 10/11
  - Platform    : Windows-10-...

[3] Deep Learning Framework (PyTorch):
  - Version     : 2.x.x+cu124
  - CUDA Support: Available

[4] GPU Hardware Diagnostics:
  - Device Count: 1
  - GPU [0] Name    : NVIDIA GeForce RTX 3050 Laptop GPU
  - GPU [0] VRAM    : 6.00 GB
  - Compute Capability: 8.6
  - Multi-Processors  : 16
  - PyTorch CUDA Version : 12.4
  - cuDNN Version        : 90100

============================================================
 Diagnostic Complete.
============================================================
```

---

## Character-Level Tokenizer

The project features a deterministic, educational `CharTokenizer` implemented purely in standard Python (`src/tokenizer.py`):
- **Deterministic Vocabulary**: Extracts unique characters from the training text and sorts them deterministically.
- **Bi-directional Mappings**: Maintains human-readable `stoi` (string-to-index) and `itos` (index-to-string) dictionaries.
- **Input Validation**: Raises informative `ValueError` exceptions for empty training corpora, unseen characters during encoding, and out-of-range IDs during decoding.

### Running the Tokenizer Demo
```powershell
python scripts/tokenizer_demo.py
```

---

## Bigram Language Model Baseline

The Bigram model (`src/bigram.py`) serves as the fundamental baseline for autoregressive sequence modeling:
- **Concept**: Next-token probabilities are conditioned exclusively on the current token ($P(w_t \mid w_{t-1})$) without multi-step history or attention.
- **Learnable Embedding Table**: Utilizes an `nn.Embedding(vocab_size, vocab_size)` lookup table where row $i$ represents unnormalized logits over the vocabulary for token $i$.
- **Loss & Backpropagation**: Computes PyTorch cross-entropy loss against target token sequences shifted by 1 position.
- **Autoregressive Generation**: Employs multinomial sampling (`torch.multinomial`) over temperature-scaled softmax distributions.

### Running the Bigram Training Demo
```powershell
python scripts/train_bigram_demo.py
```

---

## Dataset Ingestion & Batching Pipeline

The data pipeline (`src/data.py`) provides efficient, reliable sequence batching for training:
- **Sequential Splitting**: Splits text into non-overlapping training and validation partitions (default 90/10).
- **Strict Vocabulary Fitting**: Fits tokenizers exclusively on the training split, throwing an explicit error if the validation partition contains unseen characters.
- **Causal Next-Token Shifting**: Samples random starting indices within partition boundaries and constructs input (`x`) and target (`y`) sequences where `y[b, t] == x[b, t+1]`.
- **Extensible Architecture**: Decoupled from specific tokenizer implementations to support future subword tokenization (BPE) seamlessly.

### Running the Data Pipeline Demo
```powershell
python scripts/data_pipeline_demo.py
```

---

## Byte-Level BPE Tokenizer

The custom `BPETokenizer` (`src/bpe_tokenizer.py`) compresses text into subword units using byte-level Byte Pair Encoding:
- **Base Vocabulary (0–255)**: Grounded in all 256 individual UTF-8 byte values, preventing out-of-vocabulary failures for any character, symbol, or language.
- **Deterministic Merges**: Iteratively identifies the most frequent adjacent token pairs using strict frequency and lexicographical tie-breaking, merging them into new tokens (IDs starting at 256).
- **Priority-Ordered Encoding**: Encodes text by applying learned merges in their original training priority order.
- **Lossless Reconstruction**: Restores exact strings upon decoding, including multi-byte Unicode and complex emojis.
- **Portability**: Supports saving and loading learned rules via standard JSON metadata files (`save()` / `load()`).

### Running the BPE Demo
```powershell
python scripts/bpe_demo.py
```

---

## Token Embeddings & Causal Self-Attention

Step 7 implements the mathematical core of the Transformer's representation layer:
- **Token & Positional Embeddings (`src/embeddings.py`)**: Combines a discrete token lookup table (`nn.Embedding(vocab_size, embed_dim)`) with a learnable absolute position table (`nn.Embedding(block_size, embed_dim)`), enabling the model to retain positional ordering along the sequence dimension.
- **Single-Head Causal Self-Attention (`src/attention.py`)**:
  - Implements explicit Query ($Q$), Key ($K$), and Value ($V$) linear projections ($W_q, W_k, W_v$).
  - Computes raw attention affinities: $S = \frac{Q K^T}{\sqrt{d_k}}$.
  - Imposes strict autoregressive causality by masking the upper-triangular region ($j > i$) to $-\infty$ *before* softmax, ensuring token $i$ only attends to tokens $j \le i$.
  - Computes normalized probability weights: $A = \text{softmax}(S, \text{dim}=-1)$.
  - Aggregates values into contextualized representations: $O = A V$.

### Running the Attention Demo
```powershell
python scripts/attention_demo.py
```

---

## Multi-Head Attention & Transformer Block

Step 8 constructs the foundational repeating building block of the GPT architecture:
- **Multi-Head Causal Attention (`src/attention.py`)**: Runs multiple independent attention heads in parallel ($d_k = d_{model} / H$), allowing the model to jointly attend to information from different representation subspaces at different positions. Head outputs are concatenated and linearly projected back to $d_{model}$ through a learned residual projection.
- **Position-Wise Feed-Forward Network (`src/feedforward.py`)**: A 2-layer MLP expanding the representation space ($d_{model} \to 4 \times d_{model}$) with Gaussian Error Linear Unit (GELU) activation before projecting back to $d_{model}$.
- **Pre-LayerNorm Architecture (`src/transformer_block.py`)**: Places `nn.LayerNorm` prior to both the attention and feed-forward sublayers ($x + \text{Attention}(\text{LN}_1(x))$ and $x + \text{FFN}(\text{LN}_2(x))$), providing direct gradient highways through the residual streams to ensure stable convergence in deep models.

### Running the Transformer Block Demo
```powershell
python scripts/transformer_block_demo.py
```

---

## Complete GroundUpGPT Model Architecture

Step 9 integrates all preceding layers into the unified `GroundUpGPT` model (`src/model.py`) with configuration managed via `GPTConfig`:
- **Module Hierarchy**:
  1. `TokenPositionEmbedding`: Input token IDs $(B, T) \to$ Combined representations $(B, T, d_{model})$.
  2. `Dropout`: Applied to input representations ($p = 0.1$).
  3. `nn.ModuleList([TransformerBlock...])`: Stack of $N$ Pre-LN Transformer blocks with residual streams.
  4. `nn.LayerNorm(embed_dim)`: Final normalization layer stabilizing output representations.
  5. `nn.Linear(embed_dim, vocab_size, bias=False)`: Language modeling head projecting hidden states to next-token logits $(B, T, V)$.
- **Weight Tying**: Shares parameters between `embeddings.token_embedding.weight` and `lm_head.weight`, substantially reducing parameter footprints while improving language modeling representation alignment.
- **Initialization**: Gaussian initialization ($\mathcal{N}(0, 0.02^2)$) for linear and embedding layers; biases zeroed; LayerNorm weights initialized to 1 and biases to 0.
- **Hardware-Calibrated 5M Baseline Parameter Count**:
  - `vocab_size = 4096`, `block_size = 256`, `embed_dim = 256`, `num_heads = 4`, `num_layers = 5`, `weight_tying = True`.
  - Exactly **5,051,904 unique trainable parameters (~5.05M)** perfectly matched for 6GB VRAM on the NVIDIA RTX 3050 Laptop GPU.
- **Untrained Status**: *The model architecture is fully assembled and verified, but currently contains random initial weights.* Training pipeline optimization will occur in Step 10.

### Running the Model Demo
```powershell
python scripts/gpt_model_demo.py
```

---

## Current Implementation Status

- **Completed Steps**:
  - **Step 1 — Project Initialization**: Repository structure, Git ignore rules, CUDA hardware verification (`scripts/check_environment.py`).
  - **Step 2 — Character-Level Tokenizer**: Custom `CharTokenizer` in `src/tokenizer.py`, demo in `scripts/tokenizer_demo.py`, test coverage in `tests/test_tokenizer.py`.
  - **Step 3 — Bigram Model & Project Blueprint**: Project blueprint (`PROJECT_SPEC.md`), trainable `BigramLanguageModel` in `src/bigram.py`, training demo in `scripts/train_bigram_demo.py`, and unit test suite in `tests/test_bigram.py`.
  - **Step 4 — Dataset Ingestion & Batching**: Reusable `TextDataPipeline` in `src/data.py`, offline text fixture in `tests/fixtures/sample_text.txt`, batching demo in `scripts/data_pipeline_demo.py`, and unit test suite in `tests/test_data.py`.
  - **Step 5 — Byte-Level BPE Tokenizer**: Custom `BPETokenizer` in `src/bpe_tokenizer.py`, demo in `scripts/bpe_demo.py`, unit test suite in `tests/test_bpe_tokenizer.py`, and `TextDataPipeline` compatibility.
  - **Step 6 — BPE Robustness & Data Pipeline Configuration**: Strict UTF-8 decoding, hardened JSON schema validation, removal of early-stop limits, and explicit `bpe_vocab_size` configuration in `src/data.py`.
  - **Step 7 — Token Embeddings & Causal Self-Attention**: `TokenPositionEmbedding` in `src/embeddings.py`, single-head `CausalSelfAttentionHead` in `src/attention.py`, interactive demo in `scripts/attention_demo.py`, and unit test suite in `tests/test_attention.py`.
  - **Step 8 — Multi-Head Attention & Transformer Block**: `MultiHeadCausalSelfAttention` in `src/attention.py`, `FeedForward` in `src/feedforward.py`, Pre-LayerNorm `TransformerBlock` in `src/transformer_block.py`, demo in `scripts/transformer_block_demo.py`, and unit test suite in `tests/test_transformer_block.py`.
  - **Step 9 — Complete GroundUpGPT Model Assembly**: Validated `GPTConfig` dataclass and full `GroundUpGPT` architecture in `src/model.py`, weight tying, Gaussian initialization, verified 5.05M parameter baseline, demo in `scripts/gpt_model_demo.py`, and unit test suite in `tests/test_model.py`.
- **Pending Implementation**:
  - *Full GPU training pipeline (Step 10), evaluation benchmarks, generation sampling engine, FastAPI backend, SQLite persistence, and Next.js frontend are pending implementation.* Development will proceed to Step 10 upon instruction.

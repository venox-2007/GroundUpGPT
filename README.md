# GroundUpGPT

**A GPT-Style Transformer Language Model Built and Trained From Scratch**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Completed (Steps 1-17)](https://img.shields.io/badge/Status-Completed%20(Steps%201--17)-brightgreen.svg)]()

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
- [GPU Training Infrastructure & Optimizer](#gpu-training-infrastructure--optimizer)
- [Autoregressive Text Generation & Streaming Engine](#autoregressive-text-generation--streaming-engine)
- [Local FastAPI Inference Backend & SSE Streaming](#local-fastapi-inference-backend--sse-streaming)
- [Web Chat Interface (Next.js & Tailwind)](#web-chat-interface-nextjs--tailwind)
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
- [x] **Step 10: GPU Training Infrastructure**
  - Implement reusable `Trainer` and `TrainingConfig` (`src/trainer.py`) supporting PyTorch AMP fp16 on CUDA.
  - AdamW optimizer with decoupled weight decay (applied to 2D matrix weights, skipped for biases/LayerNorms).
  - Learning rate scheduler featuring linear warmup followed by cosine decay down to `min_lr`.
  - Gradient accumulation with scaled loss calculation and gradient norm clipping.
  - Resilient checkpoint save/resume with strict validation of architecture, tokenizer type, vocabulary size, and context length.
  - Interactive training demo script (`scripts/train_gpt.py`) and test suite (`tests/test_trainer.py`).
- [x] **Step 11: Autoregressive Text Generation & Inference Engine**
  - Implement reusable `TextGenerator` and `GenerationConfig` (`src/generation.py`) under `torch.inference_mode()`.
  - Deterministic greedy decoding (`temperature=0.0`), temperature scaling, top-$k$ filtering, and nucleus top-$p$ sampling.
  - Sliding context window cropping (`idx[:, -block_size:]`) preventing positional overflow.
  - Token-level streaming iterator (`generate_stream`) and comprehensive execution metrics (tokens/sec, context usage ratio).
  - Resilient byte-level BPE boundary decoding preventing crashes on incomplete multi-byte Unicode.
  - Checkpoint hardening: RNG state preservation, deterministic tokenizer fingerprinting, and complete architectural validation.
  - Interactive generation demo script (`scripts/generate_demo.py`) and test suite (`tests/test_generation.py`).
- [x] **Step 12: Local FastAPI Inference Backend & SSE Streaming**
  - Modular backend package (`backend/`) powered by FastAPI, Pydantic, and Uvicorn.
  - Endpoints: `GET /api/health`, `GET /api/model/info`, `POST /api/generate`, `POST /api/generate/stream`.
  - Inference-only `ModelManager` loading model once at startup, validating tokenizer fingerprint, and serializing execution via `asyncio.Lock`.
  - Real-time Server-Sent Events (SSE) with `token`, `done`, and `error` events and graceful client disconnect handling.
  - Secure local CORS configuration, configurable environment settings (`.env.example`), and targeted test suite (`tests/test_api.py`).
- [x] **Step 13: Web Chat Frontend (Next.js & Tailwind)**
  - Responsive ChatGPT-inspired web interface built with Next.js App Router, React, TypeScript, and Tailwind CSS (`frontend/`).
  - Real-time Server-Sent Events (SSE) streaming client using `ReadableStream` with robust chunk boundary parsing.
  - Interactive generation controls: real-time `Stop` button via `AbortController` (preserving partial outputs) and `Regenerate`.
  - Live execution telemetry: generated tokens, generation speed (`tok/s`), elapsed time, active context tokens, and context-cropped indicator.
  - In-sidebar sampling controls: temperature, nucleus top-$p$, top-$k$, max new tokens, and random seed.
  - Session chat management: clear conversation and keyboard shortcuts (Enter to send, Shift+Enter for newline).
- [ ] **Step 14: Persistent Storage & SQLite Conversation History**

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

## GPU Training Infrastructure & Optimizer

Step 10 implements a complete, hardware-calibrated training engine (`src/trainer.py`) engineered specifically for local execution on the NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) and CPU environments:
- **`TrainingConfig` Dataclass**: Strict configuration container encapsulating batch size, context length, gradient accumulation steps, optimizer settings (`learning_rate`, `min_learning_rate`, `weight_decay`, `betas`, `eps`), cosine decay scheduler (`warmup_steps`, `max_steps`), gradient clipping (`max_grad_norm`), checkpoint management, mixed-precision flags, and device selection.
- **AdamW Optimization with Weight Decay Decoupling**: Parameters are split into two groups: 2D weight matrices (linear projections, embeddings) receive decoupled weight decay, while 1D vectors (biases, LayerNorm scale and shifts) are explicitly exempted.
- **Warmup + Cosine Decay Learning Rate Scheduler**: Linearly scales learning rate from 0 up to `learning_rate` during `warmup_steps`, followed by cosine decay down to `min_learning_rate` at `max_steps`.
- **Automatic Mixed Precision (AMP)**: Leverages `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` for float16 mixed-precision acceleration, halving memory bandwidth demands and maximizing Tensor Core utilization on Ampere GPUs.
- **Gradient Accumulation & Norm Clipping**: Enables effective batch sizes larger than physical VRAM by scaling loss by `1 / grad_accum_steps`, with `clip_grad_norm_` applied before optimizer updates.
- **Safe & Validated Checkpointing**: Checkpoint files store full training state (`step`, `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `train_losses`, `val_losses`, `best_val_loss`, and embedded `GPTConfig` & tokenizer metadata). Strict compatibility validation prevents resuming training into a model with mismatched architecture, vocabulary size, or context window.
- **Training Status**: *The training infrastructure, optimizer, scheduler, and checkpoint save/resume cycles have been thoroughly verified with targeted unit tests and an offline smoke run. Full model pretraining on large-scale corpora is planned for subsequent steps.*

### Running the Training Demo
```powershell
python scripts/train_gpt.py
```

---

## Autoregressive Text Generation & Streaming Engine

Step 11 introduces a high-performance, reusable autoregressive inference engine (`src/generation.py`) for generating text token by token from conditioned prompts:
- **`TextGenerator` & `GenerationConfig`**: Encapsulates model inference under `torch.inference_mode()` with `model.eval()`. Accepts `max_new_tokens`, `temperature`, `top_k`, `top_p`, and `seed`.
- **Sampling Mechanics**:
  - **Greedy Argmax Decoding (`temperature = 0.0`)**: Selects the highest-probability token deterministically via `argmax(logits)`.
  - **Temperature Scaling (`temperature > 0.0`)**: Controls probability entropy by scaling logits by `1.0 / temperature`.
  - **Top-$k$ Truncation**: Restricts candidate tokens to the $k$ highest logit values, setting remaining positions to $-\infty$.
  - **Nucleus Top-$p$ Filtering**: Computes cumulative softmax probability mass and dynamically clips the candidate tail once cumulative density exceeds threshold $p$, ensuring at least the top-1 candidate is always preserved.
  - **Sliding Context Window Cropping**: Crops sequences to `idx[:, -block_size:]` prior to model forward passes, preventing out-of-bounds context window exceptions.
- **Incremental Streaming Iterator (`generate_stream`)**: Yields `StreamChunk` objects in real-time as individual tokens are predicted. Employs Python incremental UTF-8 decoders to smoothly buffer multi-byte byte fragments (such as emojis or non-ASCII characters) across token boundaries without crashing or weakening `BPETokenizer`'s strict decoding invariants.
- **Telemetry & Throughput Tracking**: Returns structured `GenerationResult` metadata with prompt token counts, generated token counts, total duration, tokens-per-second throughput, and context window utilization ratio.
- **Hardened Checkpoint Verification**: Checkpoints explicitly save and restore dedicated RNG generator states, require matching SHA-256 tokenizer identity fingerprints, and validate complete architectural consistency (`bias`, `weight_tying`, dimensions).
- **Generation Quality Disclaimer**:
  > [!NOTE]
  > The demonstration script trains a tiny 2-layer experimental model for 15 steps on an offline fixture to prove mechanical sampling and streaming validity. Generating coherent, grammatically fluid English requires substantial pretraining on large corpora, which is planned for upcoming dataset scaling phases.

### Running the Generation Demo
```powershell
python scripts/generate_demo.py
```

---

## Local FastAPI Inference Backend & SSE Streaming

Step 12 builds a robust, modular local serving architecture in `backend/` powered by FastAPI, Pydantic v2, and Uvicorn:
- **Backend Architecture**:
  - `backend/config.py`: Centralized configuration loading host, port, artifact paths, CORS origins, and safety limits from environment variables (`.env.example` provided).
  - `backend/schemas.py`: Strict Pydantic models for request validation (`GenerateRequest`) and structured responses (`GenerateResponse`, `ModelInfoResponse`, `HealthResponse`).
  - `backend/model_loader.py`: Inference-only `ModelManager` that loads the model checkpoint and tokenizer once at startup, strictly validates architecture and tokenizer SHA-256 fingerprints, and serializes inference execution via `asyncio.Lock` to avoid GPU memory collision.
  - `backend/app.py`: FastAPI server exposing health, metadata, generation, and streaming routes.
  - `backend/main.py`: Executable CLI launcher for starting the Uvicorn server locally.
- **REST & SSE Endpoints**:
  - `GET /api/health`: Reports server operational status, whether a model is loaded, the active device, and package version.
  - `GET /api/model/info`: Exposes full model hyperparameters, parameter counts, context limits, tokenizer type, and checkpoint provenance.
  - `POST /api/generate`: Executes synchronous autoregressive generation, returning full text alongside prompt/generated token counts, elapsed time, throughput, and context usage telemetry.
  - `POST /api/generate/stream`: Real-time Server-Sent Events (SSE) streaming yielding incremental `token` events as characters are predicted, followed by a final `done` event with complete telemetry, with graceful cancellation if the client disconnects.
- **Security & Concurrency Discipline**:
  - Binds strictly to `127.0.0.1` by default.
  - Restricts CORS to designated local origins (`http://localhost:3000`, `http://127.0.0.1:3000`).
  - Rejects arbitrary remote filesystem paths or checkpoint uploads via API.

### Running the Inference Backend Locally
```powershell
# 1. Ensure demo model artifacts are generated
python scripts/generate_demo.py

# 2. Start the FastAPI backend
python -m backend.main

# Or run via Uvicorn directly:
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

---

## Web Chat Interface (Next.js & Tailwind)

Step 13 delivers a clean, responsive web chat interface built with modern Next.js 14 (App Router), React 18, TypeScript, and Tailwind CSS (`frontend/`):
- **User Interface Features**:
  - **ChatGPT-Inspired Layout**: Dark-themed slate palette with teal accents, responsive collapsible sidebar, and clean typography.
  - **Sidebar Telemetry & Status**: Live connection indicator, parameter counts, context window limit, device (`CUDA`/`CPU`), and checkpoint metadata.
  - **Interactive Hyperparameter Tuning**: In-sidebar controls for `temperature` (0.00 to 2.00), nucleus `top_p` (0.10 to 1.00), `top_k`, `max_new_tokens` (5 to 128), and optional random `seed`.
  - **Real-Time Token Streaming**: Consumes `/api/generate/stream` SSE events with boundary-safe chunk reassembly, rendering live token deltas and animated cursor.
  - **Interactive Controls**: Dedicated `Stop` generation button (powered by `AbortController`, safely preserving received text) and `Regenerate response`.
  - **Telemetry Badges**: Displays genuine backend metrics on completions: generated tokens, generation throughput (`tok/s`), elapsed latency, active context window, and a warning badge if the prompt exceeded `block_size` and was cropped.
  - **Session Management**: Local React state message history with `Clear Conversation` capability.
  - **Honest Educational Notice**: Explicitly identifies the underlying model as a lightweight educational prototype predicting raw token completions, with no claims of multi-turn conversational reasoning.

### One-Click Launchers (Windows)

GroundUpGPT includes root batch launchers for instant local startup:

- **`Start GroundUpGPT.bat`**:
  - Double-click from Windows Explorer (handles paths with spaces and OneDrive directories).
  - Verifies Python virtual environment and Node.js availability.
  - Checks if services are already running to prevent duplicate processes.
  - Automatically installs missing frontend dependencies on first run.
  - Offers to generate lightweight demo artifacts if missing (never retrains on normal startup).
  - Starts FastAPI backend (`127.0.0.1:8000`) and Next.js frontend (`localhost:3000`) in separate console windows with live logs.
  - Waits for health checks to pass, then automatically opens `http://localhost:3000` in your default browser.
- **`Stop GroundUpGPT.bat`**:
  - Safely terminates only GroundUpGPT backend and frontend processes listening on ports 8000 and 3000 without affecting unrelated Python or Node.js applications.

### Manual Startup (PowerShell)

Alternatively, open two separate Windows PowerShell windows:

```powershell
# Window 1: Start Backend (Port 8000)
python -m backend.main

# Window 2: Start Frontend (Port 3000)
cd frontend
npm.cmd run dev
```

Visit `http://localhost:3000` in your browser.

---

## Model Evaluation & Profile Selection

Step 17 introduces a quantitative evaluation framework (`scripts/evaluate_model.py`) and dynamic multi-model profile serving across the FastAPI backend and Next.js Web UI.

### 1. Comparative Model Evaluation

To benchmark validation loss, perplexity ($\exp(\text{loss})$), parameter counts, generation throughput, and qualitative completions between the Step 13 demo model and the Step 16 5.05M model:

```powershell
# Run comparative evaluation on held-out validation data
python scripts/evaluate_model.py --profile compare

# Or evaluate a single profile specifically:
python scripts/evaluate_model.py --profile 5m
python scripts/evaluate_model.py --profile demo
```

The script produces a side-by-side terminal comparison and records machine-readable JSON metrics to `logs/evaluation_comparison.json`.

#### Measured Comparative Performance

| Metric | Demo Model (Step 13) | 5.05M Model (Step 16) | Notes |
| :--- | :--- | :--- | :--- |
| **Parameters** | 34,880 | **5,051,904** | 144x parameter scaleup |
| **Vocabulary Size** | 280 tokens | **4,096 tokens** | Full byte-level BPE vocabulary |
| **Context Window** | 32 tokens | **256 tokens** | 8x longer sequence context |
| **Architecture** | 2 Layers / 4 Heads | **5 Layers / 4 Heads** | Scaled Transformer stack |
| **Training Steps** | 15 steps | **200 steps** | 3,276,800 tokens processed |
| **Validation Loss** | 5.3150 | **6.1029** | Evaluated on held-out test splits |
| **Validation Perplexity** | 203.36 | **447.15** | Initial random perplexity was >4,200 |
| **Inference Throughput** | ~1,050 tok/s | **~228 tok/s** | Real-time generation on CUDA |

### 2. Model Selection in Backend and Web UI

GroundUpGPT supports dynamic profile selection across all trained checkpoints:

1. **Automatic Discovery & Default Profile**:
   When `checkpoints/5m_phase2/model_5m_best.pt` exists, the backend automatically defaults to serving the Phase 2 Best 5.05M model (Step 3000). If missing, it smoothly falls back through `5m_best` (Step 1000) $\to$ `5m_step200` $\to$ `demo`.
2. **Environment Variable Configuration**:
   ```powershell
   # Force Phase 2 Best profile
   $env:GROUNDUPGPT_MODEL_PROFILE = "5m_phase2_best"
   python -m backend.main

   # Force Step 1000 profile
   $env:GROUNDUPGPT_MODEL_PROFILE = "5m_best"
   python -m backend.main

   # Force Step 200 baseline
   $env:GROUNDUPGPT_MODEL_PROFILE = "5m_step200"
   python -m backend.main

   # Force demo profile
   $env:GROUNDUPGPT_MODEL_PROFILE = "demo"
   python -m backend.main
   ```
3. **Interactive Web UI Switcher**:
   The Web UI sidebar includes a **Model Profile** grid with buttons for each discovered model (`Phase 2 Best`, `Phase 2 (3K)`, `Step 1000`, `Step 200 (5M)`, `Demo (35K)`). Clicking switches the active model dynamically via `POST /api/model/load`, updating active parameters, step, context window, dual-track validation losses, and tokenizer in real time without restarting the server.

---

## Phase 2 Continued Pretraining & Quality Optimization (Steps 22–25)

In Autonomous Development Batch 3, GroundUpGPT underwent dataset expansion, rigorous quality auditing, and Phase 2 pretraining:

### 1. Expanded Real English Pretraining Corpus (Step 23)
- **Sources**: 62 Project Gutenberg classic works (57 training books + 5 held-out validation books), covering narrative fiction, philosophy, science, history, and essays.
- **Strict Data Isolation**: Zero data leakage from the original 3 held-out validation books (`pg35`, `pg36`, `pg5200`), which remain exclusively in `data/cache/val_tokens.pt` as an independent regression benchmark.
- **Corpus Statistics**:
  - Total Clean Training Bytes: **50.97 MB** (50,188,501 characters)
  - Total Training Tokens: **13,871,198 tokens** (within 10M–20M target)
  - Expanded Validation Tokens: **1,118,258 tokens** (5 held-out books)
  - Total Raw Download: **49.37 MB** (well below 100 MB budget)
- **Tokenizer Preservation**: 100% compatibility with existing 4,096-vocab byte-level BPE tokenizer (fingerprint `4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72`).

### 2. Phase 2 Continued Pretraining (Step 24)
- **Resumed From**: `checkpoints/5m_continued/model_5m_best.pt` (Step 1000).
- **Schedule**: Explicit Phase 2 cosine decay schedule with 50-step linear warmup to a conservative peak learning rate of $2.5 \times 10^{-4}$ decaying to $3.0 \times 10^{-5}$.
- **Hardware & Telemetry**: Trained on NVIDIA GeForce RTX 3050 6GB Laptop GPU.
  - Steps: Step 1000 $\to$ Step 3000 (+2,000 steps)
  - Additional Tokens Processed: **32,768,000 tokens**
  - Elapsed Time: **508.95 seconds** (~8.48 min, strictly respecting the 15-minute cap)
  - Peak VRAM: **744.7 MB**
  - Average Throughput: **64,383 tokens/second**
- **Dual-Track Evaluation**:
  - Track 1 (Original Frozen Val): Loss dropped from 5.3404 $\to$ **4.8185** (**-40.7% perplexity reduction** from 208.59 $\to$ 123.78).
  - Track 2 (Expanded Val): Loss dropped from 5.3043 $\to$ **4.5330** (**-53.8% perplexity reduction** from 201.19 $\to$ 93.04).

### 3. Quantitative Model Comparison (Step 25)

| Metric | Step 200 Baseline | Step 1000 Baseline | Phase 2 Best (Step 3000) | Overall Change |
| :--- | :--- | :--- | :--- | :--- |
| **Total Steps** | 200 | 1,000 | **3,000** | +2,800 steps |
| **Parameters** | 5,051,904 | 5,051,904 | **5,051,904** | Preserved |
| **Tokens Seen** | 3.28M | 15.56M | **48.33M** | +32.77M in Phase 2 |
| **Track 1 Val Loss (Frozen)** | 6.1520 | 5.3404 | **4.8185** | **-21.7% from Step 200** |
| **Track 1 Perplexity** | 469.68 | 208.59 | **123.78** | **-73.6% from Step 200** |
| **Track 2 Val Loss (Expanded)**| ~6.05 | 5.3043 | **4.5330** | **-14.5% from Step 1000** |
| **Track 2 Perplexity** | ~424 | 201.19 | **93.04** | **-53.8% from Step 1000** |
| **Generation Speed** | ~228 tok/s | ~106 tok/s | **~77 tok/s** | Real-time on RTX 3050 |

### 4. Qualitative Linguistic Assessment
- **Word Correctness**: Dramatically improved. Misspelled words and random token fragments from early checkpoints are virtually absent.
- **Grammar**: Sentence structures show proper subject-verb agreement, clause punctuation, dialogue quotation formatting, and pronoun references.
- **Local Coherence**: Contextually coherent across 10–25 tokens (e.g. narrative style consistent with 19th-century English literature).
- **Repetition**: Significantly reduced compared to Step 200, though greedy sampling ($T=0$) can still loop on certain repetitive phrases.
- **Limitations**: The model is an autoregressive token predictor trained from scratch on 48M tokens. It does not possess reasoning, factual recall, or instruction-following capabilities.

---

## Supervised Instruction Fine-Tuning (SFT) — Steps 26 & 27

### 1. Instruction Dataset & Conversation Formatting (Step 26)
- **Dataset**: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (`databricks-dolly-15k.jsonl`).
- **License & Attribution**: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0). Attribution: `Databricks Dolly 15k, © 2023 Databricks, Inc., licensed under CC BY-SA 3.0`.
- **Integrity Checksum**: SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` (12.48 MB raw).
- **Conversation Delimiters**:
  - User header: `<|user|>\n`
  - Assistant header: `<|assistant|>\n`
  - Turn termination: `<|endoftext|>\n`
  All delimiters encode into standard byte-level BPE tokens without modifying the 4,096 vocabulary or fingerprint (`4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72`).
- **Assistant-Only Masking**: Prompt and delimiter tokens are mapped to target `-100` (`IGNORE_INDEX`). Cross-entropy loss and backpropagated gradients are computed strictly on assistant response tokens.
- **Filtered & Cached Subset**: 9,373 valid examples $\le 256$ tokens (8,435 train, 938 validation, 0 prompt leakage) cached in `data/cache_sft/`.

### 2. First Supervised Fine-Tuning Run (Step 27)
- **Source Model**: Pretrained Step 3000 best checkpoint (`checkpoints/5m_phase2/model_5m_best.pt`).
- **Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- **Training Schedule**: 650 optimizer steps (effective batch size 32 via micro-batch 16 with 2 gradient accumulation steps). Peak LR: $1.5 \times 10^{-4}$ with 50-step linear warmup decaying to $1.5 \times 10^{-5}$ via cosine schedule.
- **Wall-Clock Time**: **97.13 seconds** (1.62 minutes, strictly respecting the 10-minute cap).
- **Active Tokens Trained**: **1,659,824 response tokens**.

#### Quantitative Validation Results

| Metric | Pre-SFT Baseline (Step 3000) | SFT Fine-Tuned (Step 650 Best) | Relative Change |
| :--- | :---: | :---: | :---: |
| **SFT Validation Loss (Held-Out)** | 6.5486 | **3.8380** | **-41.4% loss reduction** |
| **SFT Validation Perplexity** | 698.30 | **46.43** | **-93.4% ($15\times$ perplexity drop)** |
| **Frozen LM Benchmark Loss** | 4.5685 | 5.2730 | +15.4% (distribution shift) |
| **Frozen LM Benchmark Perplexity**| 96.40 | 194.99 | Catastrophic forgetting stabilized |

#### Qualitative Side-by-Side Assessment

| Task Category | User Prompt | Pretrained Base (Step 3000) | SFT Model (Step 650) |
| :--- | :--- | :--- | :--- |
| **Greeting** | *Hello! How are you?* | Uppercase Victorian OCR fragments: `'HORESTITIRIT SENG THE...'` | Modern conversational text: `'Tosta is a raby for an ability to an activitive sportal sport...'` |
| **Factual QA** | *What is the capital of France?* | Incoherent repetition: `'HORENT MORIT SENG THE STENCENG...'` | Stopped on `<|endoftext|>`: `'The AS States is one of the most regions of San Oceans...'` |
| **Classification** | *Identify which of the following is a fruit: carrot, apple, potato* | Continuous 19th-century prose: `'The whole rank is unexpected to serve...'` | Stopped on `<|endoftext|>`, formatted list: `'Tammos, Sunk, Suel Clas, San Banada'` |
| **List Generation**| *List three primary colors.* | Repetitive header text: `'HORENT MORIT SENG THE STENCENGS...'` | Stopped on `<|endoftext|>`, comma-separated: `'Trames, Had, Canada Clas, San Banada...'` |

### 3. Reproducible SFT Commands

```bash
# 1. Prepare and cache SFT dataset (Dolly 15k, filtered to 256 tokens)
python scripts/prepare_sft_dataset.py

# 2. Run bounded SFT fine-tuning (650 steps, max 10 minutes)
python scripts/train_sft_5m.py --max-steps 650 --max-seconds 580 --output-dir checkpoints/5m_sft
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
  - **Step 10 — GPU Training Infrastructure**: `Trainer` and `TrainingConfig` in `src/trainer.py`, AdamW optimizer with selective weight decay, cosine decay LR schedule with linear warmup, AMP float16 support, gradient accumulation, gradient norm clipping, architecture-checked checkpointing, smoke demo in `scripts/train_gpt.py`, and test suite in `tests/test_trainer.py`.
  - **Step 11 — Autoregressive Text Generation**: `TextGenerator` and `GenerationConfig` in `src/generation.py`, greedy decoding, temperature scaling, top-$k$, nucleus top-$p$, streaming iterator, stop sequence detection, delimiter residue stripping, checkpoint RNG/fingerprint hardening, demo in `scripts/generate_demo.py`, and test suite in `tests/test_generation.py`.
  - **Step 12 — Local FastAPI Inference Backend**: Modular `backend/` package, `GET /api/health`, `GET /api/model/info`, `POST /api/generate`, `POST /api/generate/stream`, Server-Sent Events, Pydantic v2 schemas, model startup loader, concurrency locking, and test suite in `tests/test_api.py`.
  - **Step 13 — Web Chat Frontend & Launchers**: Modern Next.js 14 App Router, React 18, TypeScript, and Tailwind CSS chat application in `frontend/`, real-time SSE streaming with `AbortController` cancellation, live token metrics, in-sidebar sampling controls, responsive interface, and hardened one-click Windows launchers (`Start GroundUpGPT.bat` and `Stop GroundUpGPT.bat`).
  - **Step 14 — Real Language Modeling Corpus Preparation**: Reusable dataset pipeline in `src/data_prep.py` and CLI in `scripts/prepare_dataset.py`, preparing 15 classic public-domain English literary works from Project Gutenberg (5.94 MB clean text, 5.84M characters, 12 train / 3 val documents, zero-leak document-level partitioning, and complete SHA-256 manifest in `data/processed/manifest.json`).
  - **Step 15 — Production BPE Tokenizer and Token Cache**: Fitted production 4,096-vocabulary byte-level BPE tokenizer on training documents (`checkpoints/bpe_vocab4096.json`, fingerprint `4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72`), optimized O(N) linked-list/heap merge algorithm, memoized piece encoding, and pre-tokenized binary token caches in `data/cache/` (1,361,831 train tokens, 168,741 val tokens, 3.89x compression ratio, and `PretokenizedDataPipeline` with causal one-token shifts).
  - **Step 16 — Initial Training of Full 5.05M GroundUpGPT Model**: Scaled training experiment using `scripts/train_5m_gpt.py`, processing 3,276,800 tokens over 200 optimization steps (effective batch size 64, context length 256) on an NVIDIA RTX 3050 6GB Laptop GPU. Cross-entropy loss successfully decreased from 8.3522 down to 5.8608 train / 6.1189 validation at 37,009.3 tokens/sec with peak VRAM footprint of only 744.7 MB. Saved checkpoint `checkpoints/5m_initial/model_5m_step200.pt`.
  - **Step 17 — Model Evaluation & Seamless Web UI Selection**: Quantitative evaluation suite in `scripts/evaluate_model.py`, computing validation cross-entropy loss and perplexity ($\exp(\text{loss})$), generation throughput benchmarks, and comparative reporting between demo and 5M models. Integrated dynamic profile selection across backend, Web UI, and Windows launcher.
  - **Step 18 — Inference Safety & Stability**: Hardened model switching against untrusted/arbitrary file path injection by enforcing trusted profiles (`extra = "forbid"`), prevented partial state corruption on failed loads, and engineered cooperative stream cancellation with bounded backpressure queues and strict lock preservation. Verified with 6/6 focused unit tests (`tests/test_inference_safety.py`).
  - **Step 19 — Reproducible Language Model Evaluation**: Built deterministic frozen validation benchmark on held-out Project Gutenberg corpus (64 fixed windows, 16,384 tokens, seed 1337) preventing evaluation/training contamination. Isolated evaluation sampling from training batch RNG, enforced strict tokenizer fingerprint verification, and established Step 200 baseline metrics (`logs/baseline_step200_evaluation.json`: Loss 6.1520, PPL 469.67). Verified with 4/4 unit tests (`tests/test_reproducible_evaluation.py`).
  - **Step 20 — Continued Pretraining with Monitored Checkpoints**: Resumed 5.05M GroundUpGPT pretraining from Step 200 through Step 1,000 on RTX 3050 GPU (processing an additional 12,288,000 tokens at 61,770.6 tok/s across 198.9s). Preserved 1,000-step cosine decay schedule, optimizer state, scaler, and isolated RNG generators. Validation loss dropped from 6.1520 to 5.4116, and perplexity dropped from 469.67 to 223.98 (-52.3% reduction). Saved protected checkpoints (`model_5m_best.pt`, `model_5m_latest.pt`, `step_300.pt`...`step_1000.pt`).
  - **Step 21 — Best Model Selection & Full UI Integration**: Enabled dynamic checkpoint discovery and runtime profile switching for `5m_best` (Step 1000, loss 5.4116), `5m_latest`, `5m_step200`, and `demo`. Updated Next.js sidebar with live model buttons, active step badges, real-time validation loss displays, and full streaming support across switches. Verified with 5/5 focused unit tests (`tests/test_model_selection.py`) and full repository test suite (118/118 passing).
  - **Step 22 — Model Quality Audit & Baseline Hardening**: Deduplicated model profile discovery ensuring distinct model artifacts are exposed only once in API and UI, bound active profile identity strictly to backend state, verified 100% zero-leak isolation between 12 training and 3 validation books (`manifest.json`), resolved metadata discrepancy in evaluation script, and audited deterministic baselines on GPU (Step 200 loss: 6.1520, PPL: 469.68; Step 1000 loss: 5.3404, PPL: 208.59). Saved audit report in `logs/baseline_audit_step22.json`. Verified with 20/20 focused tests.
  - **Step 23 — Expand Licensed English Pretraining Corpus**: Curated 50 additional public-domain works from Project Gutenberg (49.37 MB raw download, zero leakage of original validation books `pg35`, `pg36`, `pg5200`). Prepared 62 total documents (57 train, 5 expanded val) yielding **13,871,198 clean training tokens** and **1,118,258 expanded validation tokens**. Reused exact 4,096-vocab BPE tokenizer without retraining (fingerprint verified). Saved binary cache in `data/cache_expanded/`. Verified with 6/6 unit tests (`tests/test_expanded_data.py`).
  - **Step 24 — Quality-Focused Continued Pretraining (Phase 2)**: Resumed from Step 1000 best checkpoint and trained through Step 3000 (+2,000 steps, +32,768,000 tokens processed) on RTX 3050 Laptop GPU in 508.95s (8.48 min, within 15-min cap). Implemented explicit Phase 2 cosine decay schedule (warmup to 2.5e-4 decaying to 3.0e-5). Tracked dual validation benchmarks: Track 1 (Original Frozen Val) dropped from 5.3404 to **4.8185** (PPL: 123.78, **-40.7%** reduction); Track 2 (Expanded Val) dropped from 5.3043 to **4.5330** (PPL: 93.04, **-53.8%** reduction). Saved checkpoints (`checkpoints/5m_phase2/`) and telemetry (`logs/training_phase2_5m.json`). Verified with 4/4 unit tests (`tests/test_phase2_pretraining.py`).
  - **Step 25 — Model Comparison & Web UI Integration**: Executed comparative benchmarking (`scripts/compare_models_step25.py`) across Step 1000 Baseline, Phase 2 Best, and Phase 2 Latest. Extended backend profile discovery to expose all distinct checkpoints (`5m_phase2_best`, `5m_phase2_latest`, `5m_best`, `5m_latest`, `5m_step200`, `demo`) with automatic preference for Phase 2 Best. Updated Next.js UI sidebar to render active step, dual-track validation losses (Orig Val, Exp Val), and perplexity. Verified Windows launcher preflight (19/19 tests), Next.js production build (`npm run build`), and full repository test suite (131/131 passing).
  - **Step 26 — Supervised Conversation Dataset & Training Pipeline**: Sourced human-generated Databricks Dolly 15k dataset (CC BY-SA 3.0), implemented `<|user|>\n`, `<|assistant|>\n`, `<|endoftext|>\n` delimiters with 100% BPE lossless roundtrip, assistant-only loss masking with `-100`, regex PII scrubbing, exact and near deduplication, zero-leak train/val splitting, and cached 9,373 examples (8,435 train, 938 validation, 1,180,136 total tokens, 63.14% active response tokens) in `data/cache_sft/`. Verified with 13/13 focused unit tests (`tests/test_sft_data.py`).
  - **Step 27 — First Supervised Instruction Fine-Tuning Run**: Executed bounded SFT run on RTX 3050 Laptop GPU from Step 3000 base checkpoint through 650 optimizer steps (1.62 min wall time, well under 10-min budget). Token-weighted assistant-only SFT validation loss dropped from 6.5486 to **3.8380** (perplexity dropped from 698.30 to **46.43**). Monitored frozen LM benchmark for catastrophic forgetting (loss 4.5685 -> 5.2730 / PPL 96.40 -> 195.00). Verified stop-sequence termination at `<|endoftext|>`, evaluated 6 unseen conversational benchmark prompts, and saved protected checkpoints (`checkpoints/5m_sft/`). Verified with 8/8 unit tests (`tests/test_sft_training.py`).
- **Roadmap Complete**: Phase 7 (Steps 26 & 27) successfully completed! All 27 steps complete.

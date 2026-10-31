# GroundUpGPT

**A GPT-Style Transformer Language Model Built and Trained From Scratch**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Step 4 Ingested](https://img.shields.io/badge/Status-Step%204%20Ingested-brightgreen.svg)]()

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
| **Target Parameters** | ~5,000,000 (5M) *(Provisional target; will be strictly verified once architecture dimensions are selected)* |
| **Vocabulary Size ($V$)** | ~4,096 to 8,192 tokens (custom BPE or byte-level) |
| **Context Length ($T$)** | 256 – 512 tokens |
| **Hidden Dimension ($d_{model}$)** | 256 – 384 |
| **Number of Layers ($N$)** | 6 – 8 Transformer blocks |
| **Attention Heads ($H$)** | 4 – 8 (Head dimension $d_k = d_{model} / H$) |
| **Feed-Forward Dimension ($d_{ff}$)** | $4 \times d_{model}$ (1024 – 1536) |
| **Normalization** | Pre-Layer Normalization (Pre-LN) or RMSNorm |
| **Activation Function** | GELU (Gaussian Error Linear Unit) |
| **Positional Encoding** | Learned absolute positional embeddings or Rotary (RoPE) |

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
- [ ] **Step 5: Core Transformer Architectural Blocks**
  - Positional embeddings and token embeddings.
  - Causal multi-head self-attention module with triangular masking.
  - Position-wise feed-forward network (FFN) and residual normalization blocks.
- [ ] **Step 6: Full Model Assembly & Weight Initialization**
  - Complete `GroundUpGPT` model class (~5M target parameters).
- [ ] **Step 7: Training Pipeline & Optimization**
  - AdamW optimizer, warmup, cosine decay schedule, mixed-precision (`torch.amp`), and checkpointing.
- [ ] **Step 8: Evaluation & Autoregressive Inference Engine**
  - Text generation pipeline with temperature, top-$k$, and top-$p$ sampling.
- [ ] **Step 9: Full-Stack Application (FastAPI, SQLite, Next.js UI)**
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

## Current Implementation Status

- **Completed Steps**:
  - **Step 1 — Project Initialization**: Repository structure, Git ignore rules, CUDA hardware verification (`scripts/check_environment.py`).
  - **Step 2 — Character-Level Tokenizer**: Custom `CharTokenizer` in `src/tokenizer.py`, demo in `scripts/tokenizer_demo.py`, test coverage in `tests/test_tokenizer.py`.
  - **Step 3 — Bigram Model & Project Blueprint**: Project blueprint (`PROJECT_SPEC.md`), trainable `BigramLanguageModel` in `src/bigram.py`, training demo in `scripts/train_bigram_demo.py`, and unit test suite in `tests/test_bigram.py`.
  - **Step 4 — Dataset Ingestion & Batching**: Reusable `TextDataPipeline` in `src/data.py`, offline text fixture in `tests/fixtures/sample_text.txt`, batching demo in `scripts/data_pipeline_demo.py`, and unit test suite in `tests/test_data.py`.
- **Pending Implementation**:
  - *Transformer attention layers, multi-head mechanisms, full GPT model class, and web chat application layers have not yet been implemented.* Development will proceed to Step 5 upon instruction.

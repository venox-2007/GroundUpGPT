# GroundUpGPT — Complete Project Specification & Blueprint

## 1. Executive Summary & Vision

**GroundUpGPT** is an end-to-end, educational, and portfolio-grade AI engineering project dedicated to designing, implementing, training, evaluating, and serving a decoder-only Transformer language model from scratch. 

Built strictly from first principles using Python and PyTorch without pretrained weights or wrapper libraries (such as Hugging Face Transformers, LangChain, or Ollama), the project establishes mechanistic understanding of modern LLMs while delivering a full-stack, production-quality chat application.

---

## 2. Core Architecture & Technology Stack

The project spans two primary domains: the **Custom AI Engine** and the **Full-Stack Application Layer**.

### A. Custom AI Engine (Backend Core)
- **Deep Learning Framework**: PyTorch 2.x (pure tensor math, explicit autograd).
- **Core Modules**:
  - Custom Tokenization: Educational character-level tokenizer, followed by byte-pair encoding (BPE).
  - Neural Network: Embedding layers, learned / rotary position embeddings, causal multi-head self-attention, pre-layer normalization (LayerNorm / RMSNorm), MLP with GELU/SwiGLU activations, and residual streams.
  - Training Pipeline: AdamW optimizer, cosine decay learning rate schedule with linear warmup, mixed-precision (`torch.amp`), gradient clipping, and stateful checkpointing.
  - Inference: Autoregressive decoding loop with temperature scaling, top-$k$, and top-$p$ (nucleus) sampling.

### B. Application & Serving Layer
- **API Backend**: FastAPI (asynchronous endpoints, Server-Sent Events / WebSockets for token streaming).
- **Database & Persistence**: SQLite (via SQLAlchemy or SQLModel) for persistent conversation threads, message histories, and user settings.
- **Frontend User Interface**: Next.js / React with TypeScript and Tailwind CSS.
  - Clean ChatGPT-inspired interface.
  - Streaming markdown responses with syntax highlighting.
  - Context window visualizer and real-time generation speed (tokens/sec).
  - Stop/regenerate controls and interactive hyperparameter adjustment sliders.

---

## 3. Hardware Constraints & Realistic Model Quality

### A. Hardware Budget
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (**6 GB VRAM**, Ampere, Compute Capability 8.6).
- **CPU**: Intel Core i5-13450HX (14 cores, 20 threads).
- **RAM**: 16 GB DDR5.
- **Storage**: 512 GB SSD.

### B. Model Quality & Performance Boundaries
- **Parameter Target**: ~5 Million parameters ($5 \times 10^6$) *(Provisional target; subject to benchmarking and exact architectural parameter formulas)*.
- **Context Window**: 256 to 512 tokens.
- **Realistic Expectations**:
  - The model will effectively capture local grammatical rules, character/subword transitions, and domain-specific stylistic patterns (e.g., Shakespeare, TinyStories, or code snippets).
  - **No False Claims**: It will **not** achieve general reasoning, world knowledge, or multi-turn conversational alignment comparable to commercial multi-billion parameter models (e.g., GPT-4 or Claude). Honesty and engineering rigor take precedence over exaggerated claims.

---

## 4. Feature Status Matrix

| Feature | Category | Status | Notes |
| :--- | :--- | :--- | :--- |
| Project Scaffolding & Git Workflow | Core | **Implemented** | Initialized in Step 1 |
| CUDA Diagnostic Utility | Tooling | **Implemented** | Verified RTX 3050 6GB support |
| Character-Level Tokenizer | AI Engine | **Implemented** | Pure Python `CharTokenizer` (Step 2) |
| Bigram Language Model Baseline | AI Engine | **Implemented** | Learnable lookup table & CE loss (Step 3) |
| Dataset Ingestion & Batching Pipeline | AI Engine | **Implemented** | Sequential splitting & mini-batch sampler (Step 4) |
| Byte-Pair Encoding (BPE) Tokenizer | AI Engine | **Implemented** | Byte-level subword BPE & lossless UTF-8 (Step 5) |
| Token & Positional Embeddings | AI Engine | **Implemented** | Learnable token + absolute position lookup (Step 7) |
| Single-Head Causal Self-Attention | AI Engine | **Implemented** | Explicit Q, K, V with triangular masking (Step 7) |
| Multi-Head Self-Attention & MLP | AI Engine | **Implemented** | Parallel heads, linear projection, and GELU FFN (Step 8) |
| Full Transformer Blocks & Residuals | AI Engine | **Implemented** | Modular Pre-LN block with dual residual streams (Step 8) |
| Complete GroundUpGPT Model Class | AI Engine | **Implemented** | Stacked Pre-LN blocks, final LayerNorm, LM head, weight tying (Step 9) |
| Training Loop & LR Scheduler | AI Engine | **Implemented** | AdamW, warmup, cosine decay, AMP fp16, gradient accumulation, checkpointing (Step 10) |
| Autoregressive Generation Engine | AI Engine | **Implemented** | Greedy decoding, temperature, top-$k$, top-$p$, streaming iterator (Step 11) |
| FastAPI Serving & SSE Streaming | Backend | **Implemented** | Local FastAPI server with SSE token streaming & model loader (Step 12) |
| Evaluation & Perplexity Benchmarks | AI Engine | *Planned* | Validation loss tracking & perplexity curves |
| Next.js / Tailwind Chat Frontend | Frontend | **Implemented** | Interactive ChatGPT-style UI with real-time SSE streaming (Step 13) |
| Hardened Windows Launchers | Tooling | **Implemented** | Safe process identity tracking & port conflict protection (Step 13 Preflight) |
| Real Language Modeling Corpus | AI Engine | **Implemented** | 15-book Project Gutenberg collection with zero-leak doc split (Step 14) |
| Production BPE & Token Cache | AI Engine | **Implemented** | 4096-vocab BPE tokenizer & pre-tokenized cache with causal shifts (Step 15) |
| Full 5M Model Training | AI Engine | **Implemented** | Bounded pretraining experiment on RTX 3050 (Step 16) |
| Model Evaluation & UI Selection | Full-Stack | **Implemented** | Quantitative perplexity benchmarking & live Web UI profile switching (Step 17) |

---

## 5. Phased Implementation Roadmap

1. **Phase 1: Foundations & Tokenization** *(Steps 1–6)*
   - Step 1: Project initialization, environment setup, CUDA verification.
   - Step 2: Educational character-level tokenizer and unit tests.
   - Step 3: Trainable Bigram language model baseline and project blueprint.
   - Step 4: Text dataset ingestion, sequential splitting, and batching pipeline.
   - Step 5: Byte-level Byte Pair Encoding (BPE) tokenizer and serialization.
   - Step 6: BPE robustness (strict UTF-8, hardened load validation) & configurable data pipeline.
2. **Phase 2: Core Transformer Architecture** *(Completed)*
   - Step 7: Positional & token embeddings, single-head causal self-attention module. *(Implemented)*
   - Step 8: Multi-head attention, MLP layers, Pre-LayerNorm, and residual Transformer block. *(Implemented)*
   - Step 9: Full `GroundUpGPT` model class, exact 5,051,904 parameter calibration, weight initialization. *(Implemented)*
3. **Phase 3: Optimization, Training & Evaluation** *(Completed)*
   - Step 10: Training pipeline with AdamW, mixed precision (`amp`), gradient accumulation, and checkpoint management. *(Implemented)*
   - Step 11: Autoregressive text generation, greedy decoding, temperature sampling, top-$k$, top-$p$, streaming iterator, and hardened checkpoints. *(Implemented)*
   - Step 12: Local FastAPI inference backend with real-time SSE streaming, model loader, and CORS security. *(Implemented)*
4. **Phase 4: Full-Stack Application & Real Pretraining** *(Completed)*
   - Step 13: Next.js/React frontend with streaming UI, metrics dashboard, parameter controls, and hardened Windows launchers. *(Implemented)*
   - Step 14: Real public-domain English corpus preparation (15 Project Gutenberg documents, deduplication, document-level train/validation split). *(Implemented)*
   - Step 15: Production 4096-vocab BPE tokenizer, O(N) linked-list/heap merges, and persistent pre-tokenized cache. *(Implemented)*
   - Step 16: Initial training run of full 5.05M GroundUpGPT model on RTX 3050 GPU (200 steps, 3.28M tokens, loss dropped 8.35 -> 5.86). *(Implemented)*
   - Step 17: Model evaluation tool (`scripts/evaluate_model.py`), perplexity analysis, and dynamic UI model selection (`backend/app.py`, `frontend/app`). *(Implemented)*
5. **Phase 5: Reliability, Continued Pretraining & Checkpoint Lifecycle** *(In Progress)*
   - Step 18: Inference safety and stability (trusted profile switching, bounded backpressure queue, lock coordination on cancellation). *(Implemented)*
   - Step 19: Reproducible language model evaluation benchmark with deterministic sampling and metadata integrity. *(Implemented)*
   - Step 20: Continued pretraining from Step 200 to Step 1000 with monitored checkpoints, fixed benchmark evaluation, and lr-schedule resumption (Val Loss dropped from 6.1520 to 5.4116, Perplexity improved 469.67 -> 223.98). *(Implemented)*
   - Step 21: Best-checkpoint discovery, UI integration, and end-to-end full test suite verification. *(Planned)*

---

## 6. Engineering Quality & Verification Standards

- **Strict Type Annotations**: Every internal module must employ Python type hints.
- **Unit Testing First**: Every architectural module must have targeted tests in `tests/` verifying shapes, gradient flow, and edge cases.
- **Determinism**: Every training script must support fixed seeding (`torch.manual_seed`) for reproducible validation.
- **Clean Commits**: Atomic single-step commits following Conventional Commits format.
- **Resource Discipline**: Zero memory leaks, explicit CUDA cache clearing, and defensive batch sizing to prevent VRAM OOM.

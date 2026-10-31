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
| Causal Multi-Head Self-Attention | AI Engine | *Planned* | Triangular masked scaled dot-product |
| Full Transformer Blocks & Residuals | AI Engine | *Planned* | Modular Pre-LN architecture |
| Training Loop & LR Scheduler | AI Engine | *Planned* | AdamW, warmup, cosine decay |
| Evaluation & Perplexity Benchmarks | AI Engine | *Planned* | Validation loss tracking |
| Autoregressive Generation Engine | AI Engine | *Planned* | Temperature, top-$k$, top-$p$ |
| FastAPI Serving & SSE Streaming | Backend | *Planned* | Non-blocking token generator |
| SQLite Conversation Storage | Backend | *Planned* | Thread management & settings persistence |
| Next.js / Tailwind Chat Frontend | Frontend | *Planned* | Interactive ChatGPT-style UI |
| User-Controllable Memory System | Experimental | *Planned* | Key facts retrieval in prompt injection |

---

## 5. Phased Implementation Roadmap

1. **Phase 1: Foundations & Tokenization** *(Steps 1–5)*
   - Step 1: Project initialization, environment setup, CUDA verification.
   - Step 2: Educational character-level tokenizer and unit tests.
   - Step 3: Trainable Bigram language model baseline and project blueprint.
   - Step 4: Text dataset ingestion, sequential splitting, and batching pipeline.
   - Step 5: Byte-level Byte Pair Encoding (BPE) tokenizer and serialization.
2. **Phase 2: Core Transformer Architecture** *(Planned)*
   - Step 6: Positional & token embeddings, causal self-attention modules.
   - Step 7: MLP layers, LayerNorm/RMSNorm, and residual assembly into Transformer blocks.
   - Step 8: Full `GroundUpGPT` model class, parameter count verification (~5M), weight initialization.
3. **Phase 3: Optimization, Training & Evaluation** *(Planned)*
   - Step 9: Training pipeline with AdamW, mixed precision (`amp`), and checkpoint management.
   - Step 10: Validation benchmarking, cross-entropy loss tracking, and perplexity curves.
   - Step 11: Generation sampler with temperature, top-$k$, and top-$p$.
4. **Phase 4: Full-Stack Application & Deployment** *(Planned)*
   - Step 12: FastAPI backend with Server-Sent Events (SSE) streaming.
   - Step 13: SQLite schema and persistence for chats and runtime settings.
   - Step 14: Next.js/React frontend with streaming UI, metrics dashboard, and parameter controls.
   - Step 15: Final portfolio packaging, comprehensive benchmarks, and deployment guide.

---

## 6. Engineering Quality & Verification Standards

- **Strict Type Annotations**: Every internal module must employ Python type hints.
- **Unit Testing First**: Every architectural module must have targeted tests in `tests/` verifying shapes, gradient flow, and edge cases.
- **Determinism**: Every training script must support fixed seeding (`torch.manual_seed`) for reproducible validation.
- **Clean Commits**: Atomic single-step commits following Conventional Commits format.
- **Resource Discipline**: Zero memory leaks, explicit CUDA cache clearing, and defensive batch sizing to prevent VRAM OOM.

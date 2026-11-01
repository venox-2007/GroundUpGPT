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
| SFT Conversation Data Pipeline | AI Engine | **Implemented** | Databricks Dolly 15k, assistant-only masked loss, zero-leak splits (Step 26) |
| Instruction Fine-Tuning (SFT) | AI Engine | **Implemented** | Bounded SFT run, 650 steps, PPL 698 -> 46.4, LM benchmark tracking (Step 27) |
| Instruction-Tuned Web Chat | Full-Stack | **Implemented** | SFT profile discovery, prompt turns, prefix buffering, mode indicators (Step 28) |

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
5. **Phase 5: Reliability, Continued Pretraining & Checkpoint Lifecycle** *(Completed)*
   - Step 18: Inference safety and stability (trusted profile switching, bounded backpressure queue, lock coordination on cancellation). *(Implemented)*
   - Step 19: Reproducible language model evaluation benchmark with deterministic sampling and metadata integrity. *(Implemented)*
   - Step 20: Continued pretraining from Step 200 to Step 1000 with monitored checkpoints, fixed benchmark evaluation, and lr-schedule resumption (Val Loss dropped from 6.1520 to 5.4116, Perplexity improved 469.67 -> 223.98). *(Implemented)*
   - Step 21: Best-checkpoint discovery, UI integration (Best 5M, Latest 5M, Step 200 5M, Demo), Windows launcher compatibility, and end-to-end full test suite verification (118/118 passed). *(Implemented)*
6. **Phase 6: Quality Optimization & Expanded Pretraining** *(Completed)*
   - Step 22: Quality audit and baseline hardening (profile deduplication, validation split zero-leak verification, metadata alignment, and deterministic baseline evaluation on Step 200/1000). *(Implemented)*
   - Step 23: Expand real English pretraining corpus (50 new Gutenberg books, 49.37 MB download, 13.87M clean train tokens, 1.12M expanded val tokens, zero-leak verification against pg35/36/5200, tokenizer fingerprint match). *(Implemented)*
   - Step 24: Quality-focused Phase 2 pretraining on expanded corpus with dual-track validation (Step 1000 -> Step 3000, 32.77M tokens, Track 1 loss dropped 5.3404 -> 4.8185, Track 2 loss dropped 5.3043 -> 4.5330 / PPL 93.04). *(Implemented)*
   - Step 25: Model comparison across Step 1000 Baseline, Phase 2 Best, and Phase 2 Latest, trusted profile integration (`5m_phase2_best` default), UI sidebar dual validation display, launcher preflight verification, and full repository test suite pass (131/131 passed). *(Implemented)*
7. **Phase 7: Supervised Instruction Fine-Tuning (SFT) & Web Chat Integration** *(Steps 26–28)*
   - Step 26: Supervised conversation dataset and tokenization pipeline. Sourced Databricks Dolly 15k (CC BY-SA 3.0), implemented `<|user|>\n`, `<|assistant|>\n`, `<|endoftext|>\n` delimiters, PII scrubbing, deduplication, zero-leak splitting, assistant-only loss masking with `-100`, and cached 9,373 examples (8,435 train, 938 val, 1.18M tokens). *(Implemented)*
   - Step 27: First bounded supervised instruction fine-tuning run. Fine-tuned Step 3000 model weights on Dolly 15k for 650 steps (1.62 min wall time, RTX 3050 GPU). SFT validation loss dropped from 6.5486 to 3.8380 (PPL dropped 698.30 -> 46.43). Monitored frozen LM benchmark (loss 4.5685 -> 5.2730 / PPL 96.40 -> 195.00), tested end-of-turn delimiter stopping, and generated direct side-by-side conversational comparisons. *(Implemented)*
   - Step 28: Instruction-tuned web chat integration. Backward-compatible checkpoint loading (`bpe` and `BPETokenizer`), canonical profiles (`5m_sft_best`, `5m_sft_latest`), conversational prompt wrapping, streaming delimiter prefix buffering (`find_longest_stop_prefix_len`), exact token-weighted gradient accumulation, and responsive Next.js chat interface with mode badges, telemetry, and dynamic input placeholder. *(Implemented)*
8. **Phase 8: Conversational Quality Benchmark & Inference Audit** *(Step 29)*
   - Step 29: Real conversational quality benchmark and inference audit. Executed 31-prompt benchmark across 7 domains comparing Step 3000 Base vs Step 650 SFT across Greedy, Default Sampling, and Calibrated SFT. Verified token alignment (100% match across 500 cached Dolly samples), eliminated delimiter leakage on truncated generation, added repetition penalty (1.15) to break degenerate greedy cycles, and established honest capability baseline (turn-taking and syntax learned, world knowledge and multi-step reasoning severely limited by 5.05M capacity). *(Implemented)*
9. **Phase 9: Scaled Architecture & GPU Feasibility** *(Step 30)*
   - Step 30: Scaled GPT architecture and GPU feasibility benchmark. Implemented GroundUpGPT-27M (27,411,456 parameters, 8 layers, 8 heads, $d_{model}=512$, context 256). Profiled micro-batch memory and attention mechanisms on RTX 3050 Laptop GPU (6GB VRAM). PyTorch SDPA delivered a 1.23x throughput speedup and saved 451 MB VRAM over manual attention at batch 16. Completed 20-step smoke test in 9.53s (initial loss 8.4100 -> 7.3484, 17,199 tok/s, peak VRAM 1,266 MB allocated / 1,378 MB reserved, 4.76 GB headroom). Evaluated training data sufficiency (0.51 tokens/parameter on 13.87M corpus, 39.5x deficit against Chinchilla optimal) and formulated Step 31 data diversification recommendation. *(Implemented)*
10. **Phase 10: Scaled Pretraining Data & 27M Pretraining** *(Steps 31–32)*
   - Step 31: Expanded, diverse pretraining corpus for GroundUpGPT-27M. Built versioned multi-source pretraining cache (`data/cache_27m_v1/`) combining Project Gutenberg classic literature and philosophy (13.87M train tokens, 1.12M val tokens) with Simple English Wikipedia modern factual expository articles (CC BY-SA 4.0 / GFDL, 52,346 train docs, 36.43M train tokens) yielding **50,301,446 unique training tokens** and **2,682,495 held-out validation tokens** (52,983,941 total unique tokens across 54,584 clean documents). Reused exact 4,096-token BPE tokenizer (fingerprint: `4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72`). Enforced strict zero-leakage guarantee with 10-word shingle index across all 8 forbidden validation books, excluding 19 overlapping articles. Fully compatible with `PretokenizedDataPipeline` for Step 32 pretraining. *(Implemented)*

---

## 6. Engineering Quality & Verification Standards

- **Strict Type Annotations**: Every internal module must employ Python type hints.
- **Unit Testing First**: Every architectural module must have targeted tests in `tests/` verifying shapes, gradient flow, and edge cases.
- **Determinism**: Every training script must support fixed seeding (`torch.manual_seed`) for reproducible validation.
- **Clean Commits**: Atomic single-step commits following Conventional Commits format.
- **Resource Discipline**: Zero memory leaks, explicit CUDA cache clearing, and defensive batch sizing to prevent VRAM OOM.

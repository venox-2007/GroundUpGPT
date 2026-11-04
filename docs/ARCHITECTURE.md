# GroundUpGPT System Architecture

GroundUpGPT integrates a full-stack local web application with an autoregressive Transformer and deterministic execution tools.

## Architecture Diagram

```
+--------------------------------------------------------------+
|                Next.js 14 Frontend (:3000)                   |
|  - ChatGPT-style Minimal UI (Composer, Sidebar, Settings)     |
|  - Document Extraction (+ Button: TXT, MD, PDF, DOCX)        |
|  - SSE Event Consumer (Streaming token deltas)               |
+------------------------------+-------------------------------+
                               | HTTP / SSE Requests
                               v
+--------------------------------------------------------------+
|                 FastAPI Backend (:8000)                      |
|  - Endpoints: /api/assistant/respond, /stream, /history      |
|  - Model Profile Management & Concurrency Locking            |
+------------------------------+-------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|            Assistant Router (Intent Classification)          |
|                                                              |
|  +--------------------+---------------------+--------------+ |
|  | Arithmetic Query   | Curated Fact Query  | Document QA  | |
|  |        v           |         v           |      v       | |
|  |  SafeCalculator    |  LocalFactRetriever | UserKnowledge| |
|  |    (AST Eval)      |    (Local BM25)     | (BM25 Chunks)| |
|  +--------------------+---------------------+--------------+ |
|                               | (Fallback / General Chat)    |
|                               v                              |
|             GroundUpGPT-27M Autoregressive Model             |
|         (Custom PyTorch Transformer + BPE Tokenizer)         |
+------------------------------+-------------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   [SSE Stream Generator]             [SQLite Chat DB]
   Real-time token delivery           Opt-in persistence
   (tok/s, latency metrics)           (chat_history.db)
```

## Core Components

1. **Next.js Web Client (`frontend/`)**: Renders a clean interface, handles local document parsing, manages streaming via `fetch` + `ReadableStream`, and maintains local active message state.
2. **FastAPI Application (`backend/`)**: Exposes REST and SSE endpoints. Manages single-flight model evaluation locks to prevent GPU VRAM collision.
3. **Assistant Router (`backend/assistant_router.py`)**: Classifies user prompts to route queries safely:
   - Arithmetic queries bypass neural models and run through `SafeCalculator`.
   - Curated factual questions are answered via `LocalFactRetriever`.
   - Uploaded files are chunked into token-budgeted segments and indexed with BM25.
   - Conversational prompts stream tokens autoregressively from GroundUpGPT-27M.
4. **Local SQLite Persistence (`src/chat_history.py`)**: Opt-in conversation store using WAL mode for zero background disk writes unless enabled by user in Settings.

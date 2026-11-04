# Contributing to GroundUpGPT

Thank you for your interest in contributing to GroundUpGPT! GroundUpGPT is an educational and engineering project building a decoder-only Transformer from scratch using Python and PyTorch.

## Development Environment Setup

1. **Prerequisites**:
   - Python 3.11+
   - Node.js 18+ (for frontend)
   - CUDA-compatible GPU optional (RTX 3050 6GB or similar recommended for training; CPU supported for inference)

2. **Python Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Frontend Dependencies**:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

## Contribution Workflow

1. **Branching**: Create a focused topic branch from `main`:
   ```powershell
   git checkout -b feature/your-topic-name
   ```
2. **First-Principles Standard**: All core modeling components (attention, tokenization, embeddings, layers) must be written from fundamental mathematics without external pretrained weights (no HuggingFace pretrained checkpoints).
3. **Atomic Commits**: Keep changes minimal, verifiable, and clearly documented with descriptive commit messages.

## Targeted Testing

Before submitting changes, run targeted test suites relevant to your modifications:

- **API & Backend**:
  ```powershell
  pytest tests/test_api.py tests/test_assistant_api.py -v
  ```
- **Tokenizer & Models**:
  ```powershell
  pytest tests/test_bpe_tokenizer.py tests/test_scaled_architecture.py -v
  ```
- **Document & Tools**:
  ```powershell
  pytest tests/test_step50_chatgpt_features.py tests/test_assistant_router.py -v
  ```
- **Frontend Validation**:
  ```powershell
  cd frontend; npm run build; cd ..
  ```

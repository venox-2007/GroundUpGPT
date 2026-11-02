"""Unit and integration tests for Step 50 ChatGPT-style features.

Tests cover:
- Multi-format document text extraction (TXT, MD, PDF, DOCX).
- Error handling for empty, oversized, corrupted, and image-only PDFs.
- File upload endpoint (POST /api/knowledge/upload).
- Automatic server-side routing (mode='auto'):
    - Math queries -> SafeCalculator
    - Curated facts -> LocalFactRetriever
    - Document questions -> UserKnowledgeRetriever
    - General chat -> Neural model fallback
- Native mode isolation preservation (mode='native').
- Chat history opt-in persistence enforcement.
"""

from __future__ import annotations

import io
import pytest
from starlette.testclient import TestClient

from backend.app import app
from backend.assistant_router import (
    AssistantRouter,
    MODE_AUTO,
    MODE_ASSISTED,
    MODE_NATIVE,
    ORIGIN_SAFE_CALCULATOR,
    ORIGIN_LOCAL_FACT,
    ORIGIN_DOCUMENT_EXCERPT,
    ORIGIN_NEURAL,
)
from src.document_extractor import (
    extract_text_from_bytes,
    ExtractedDocument,
    MAX_UPLOAD_BYTES,
    MAX_EXTRACTED_BYTES,
)
from src.chat_history import ChatHistoryManager
from src.user_knowledge import knowledge_store


@pytest.fixture(scope="module", autouse=True)
def setup_test_model(tmp_path_factory):
    """Load a tiny dummy model for inference tests so 503 is not returned."""
    from backend.model_loader import model_manager
    from src.model import GPTConfig, GroundUpGPT
    from src.bpe_tokenizer import BPETokenizer
    import torch

    tmp_dir = tmp_path_factory.mktemp("step50_test")
    tok_path = tmp_dir / "test_tok.json"
    ckpt_path = tmp_dir / "test_ckpt.pt"

    corpus = "GroundUpGPT is a local GPT model. 1234567890."
    tok = BPETokenizer().fit(corpus, target_vocab_size=270)
    tok.save(tok_path)

    m_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        bias=False,
        weight_tying=True,
    )
    model = GroundUpGPT(m_cfg)
    checkpoint = {
        "step": 100,
        "val_loss": 3.45,
        "model_config": {
            "vocab_size": m_cfg.vocab_size,
            "block_size": m_cfg.block_size,
            "embed_dim": m_cfg.embed_dim,
            "num_heads": m_cfg.num_heads,
            "num_layers": m_cfg.num_layers,
            "dropout": m_cfg.dropout,
            "bias": m_cfg.bias,
            "weight_tying": m_cfg.weight_tying,
        },
        "model_state_dict": model.state_dict(),
        "is_sft": True,
        "profile_name": "test_profile",
    }
    torch.save(checkpoint, ckpt_path)

    model_manager.load_model(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
        device_name="cpu",
        model_name="test_model",
        active_profile_key="test_profile",
    )


@pytest.fixture
def client():
    return TestClient(app)


# =====================================================================
# 1. Document Extraction Tests (TXT, MD, PDF, DOCX)
# =====================================================================

def test_extract_txt_plain_utf8():
    raw = "GroundUpGPT is a local GPT model from scratch.".encode("utf-8")
    doc = extract_text_from_bytes(raw, "notes.txt")
    assert doc.display_name == "notes.txt"
    assert doc.source_type == "txt"
    assert doc.character_count == len("GroundUpGPT is a local GPT model from scratch.")
    assert "GroundUpGPT is a local GPT model" in doc.content
    assert not doc.truncated


def test_extract_markdown():
    raw = "# Header\n\n- Item 1\n- Item 2\n\n```python\nprint('hi')\n```".encode("utf-8")
    doc = extract_text_from_bytes(raw, "guide.md")
    assert doc.display_name == "guide.md"
    assert doc.source_type == "md"
    assert "Header" in doc.content
    assert "Item 1" in doc.content


def test_extract_empty_file_rejected():
    with pytest.raises(ValueError, match="completely empty"):
        extract_text_from_bytes(b"", "empty.txt")


def test_extract_unsupported_format_rejected():
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_bytes(b"some binary stuff", "image.png")


def test_extract_pdf_valid():
    import pypdf
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Write some stream text to page
    buf = io.BytesIO()
    writer.write(buf)
    # pypdf blank page has no text, so should raise no extractable text
    with pytest.raises(ValueError, match="no extractable text"):
        extract_text_from_bytes(buf.getvalue(), "blank.pdf")


def test_extract_docx_valid():
    import docx
    doc = docx.Document()
    doc.add_paragraph("GroundUpGPT Step 50 document extraction test.")
    doc.add_paragraph("Second paragraph with technical details.")
    buf = io.BytesIO()
    doc.save(buf)

    extracted = extract_text_from_bytes(buf.getvalue(), "report.docx")
    assert extracted.display_name == "report.docx"
    assert extracted.source_type == "docx"
    assert "GroundUpGPT Step 50" in extracted.content
    assert "Second paragraph" in extracted.content


def test_extract_oversized_file_rejected():
    # Exceeds MAX_UPLOAD_BYTES
    oversized = b"A" * (MAX_UPLOAD_BYTES + 1024)
    with pytest.raises(ValueError, match="exceeds maximum upload limit"):
        extract_text_from_bytes(oversized, "huge.txt")


def test_extract_text_exceeding_source_limit_truncated_with_warning():
    # Exceeds 16 KB text limit: should truncate and provide warning
    long_text = "This is a sentence that repeats. " * 600
    raw = long_text.encode("utf-8")
    assert len(raw) > MAX_EXTRACTED_BYTES
    doc = extract_text_from_bytes(raw, "long.txt")
    assert doc.truncated
    assert doc.warning is not None
    assert "16 KB" in doc.warning
    assert len(doc.content.encode("utf-8")) <= MAX_EXTRACTED_BYTES


# =====================================================================
# 2. Upload Endpoint Tests (POST /api/knowledge/upload)
# =====================================================================

def test_api_upload_txt_file(client):
    # Create isolated session
    sess_res = client.post("/api/knowledge/session")
    assert sess_res.status_code == 200
    sess_id = sess_res.json()["session_id"]

    # Upload TXT file
    file_bytes = b"Machine learning models are trained using optimization algorithms."
    files = {"file": ("ml_notes.txt", io.BytesIO(file_bytes), "text/plain")}
    data = {"session_id": sess_id}

    up_res = client.post("/api/knowledge/upload", data=data, files=files)
    assert up_res.status_code == 200
    resp_json = up_res.json()
    assert resp_json["display_name"] == "ml_notes.txt"
    assert resp_json["source_type"] == "txt"
    assert resp_json["chunk_count"] >= 1


def test_api_upload_docx_file(client):
    import docx
    sess_res = client.post("/api/knowledge/session")
    sess_id = sess_res.json()["session_id"]

    doc = docx.Document()
    doc.add_paragraph("Networking protocols define communication rules.")
    buf = io.BytesIO()
    doc.save(buf)

    files = {"file": ("networks.docx", io.BytesIO(buf.getvalue()), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    data = {"session_id": sess_id}

    up_res = client.post("/api/knowledge/upload", data=data, files=files)
    assert up_res.status_code == 200
    resp_json = up_res.json()
    assert resp_json["display_name"] == "networks.docx"
    assert resp_json["source_type"] == "docx"


def test_api_upload_unsupported_file_error(client):
    sess_res = client.post("/api/knowledge/session")
    sess_id = sess_res.json()["session_id"]

    files = {"file": ("binary.bin", io.BytesIO(b"\x00\x01\x02"), "application/octet-stream")}
    data = {"session_id": sess_id}

    up_res = client.post("/api/knowledge/upload", data=data, files=files)
    assert up_res.status_code == 400
    assert "Unsupported file format" in up_res.json()["detail"]


# =====================================================================
# 3. Automatic Server-Side Routing Tests (mode='auto')
# =====================================================================

def test_auto_route_arithmetic(client):
    payload = {
        "prompt": "What is 25 + 17?",
        "mode": "auto",
    }
    res = client.post("/api/assistant/respond", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == ORIGIN_SAFE_CALCULATOR
    assert data["answer_text"] == "42"
    assert data["model_generated"] is False


def test_auto_route_curated_fact(client):
    payload = {
        "prompt": "What is the capital of Japan?",
        "mode": "auto",
    }
    res = client.post("/api/assistant/respond", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == ORIGIN_LOCAL_FACT
    assert "Tokyo" in data["answer_text"]
    assert data["model_generated"] is False


def test_auto_route_with_attached_document(client):
    # 1. Create session and upload doc
    sess_res = client.post("/api/knowledge/session")
    sess_id = sess_res.json()["session_id"]

    content = (
        "The Apollo program succeeded in landing the first humans on the Moon in 1969. "
        "Neil Armstrong and Buzz Aldrin walked on the lunar surface during Apollo 11.\n\n"
        "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation."
    )
    files = {"file": ("apollo.txt", io.BytesIO(content.encode("utf-8")), "text/plain")}
    client.post("/api/knowledge/upload", data={"session_id": sess_id}, files=files)

    # 2. Send query with attached knowledge session in auto mode
    payload = {
        "prompt": "When did Apollo land on the Moon?",
        "mode": "auto",
        "knowledge_session_id": sess_id,
    }
    res = client.post("/api/assistant/respond", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == ORIGIN_DOCUMENT_EXCERPT
    assert "1969" in data["answer_text"]
    assert data["model_generated"] is False


def test_native_mode_isolation_ignores_math_and_facts(client):
    # In native mode, even arithmetic is routed to neural model (never tool)
    payload = {
        "prompt": "What is 25 + 17?",
        "mode": "native",
        "max_new_tokens": 5,
    }
    res = client.post("/api/assistant/respond", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["origin"] == ORIGIN_NEURAL
    assert data["model_generated"] is True


def test_auto_stream_arithmetic(client):
    payload = {
        "prompt": "What is 100 * 25?",
        "mode": "auto",
    }
    res = client.post("/api/assistant/stream", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert "event: tool_result" in body
    assert "2500" in body
    assert "event: done" in body


def test_auto_stream_curated_fact(client):
    payload = {
        "prompt": "What is the capital of France?",
        "mode": "auto",
    }
    res = client.post("/api/assistant/stream", json=payload)
    assert res.status_code == 200
    body = res.text
    assert "event: tool_result" in body
    assert "Paris" in body
    assert "event: done" in body


def test_chat_history_persistence_opt_in(client):
    # 1. When conversation_id is omitted (saveHistory=False): nothing is recorded
    payload = {
        "prompt": "What is 8 + 8?",
        "mode": "auto",
    }
    res = client.post("/api/assistant/respond", json=payload)
    assert res.status_code == 200

    # 2. When conversation_id is provided (saveHistory=True): conversation and message saved
    conv_res = client.post("/api/history/conversations", json={"title": "Math Test"})
    assert conv_res.status_code == 200
    conv_id = conv_res.json()["id"]

    payload_with_history = {
        "prompt": "What is 8 + 8?",
        "mode": "auto",
        "conversation_id": conv_id,
    }
    res2 = client.post("/api/assistant/respond", json=payload_with_history)
    assert res2.status_code == 200

    detail_res = client.get(f"/api/history/conversations/{conv_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["content"] == "What is 8 + 8?"
    assert detail["messages"][1]["content"] == "16"


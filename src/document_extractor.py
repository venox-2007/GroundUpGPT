"""Local document text extraction for GroundUpGPT.

Supports safe extraction from:
- Plain Text (.txt)
- Markdown (.md, .markdown)
- PDF (.pdf) with extractable text layers (pypdf)
- Word Documents (.docx) (python-docx)

Operates purely locally without cloud document APIs or heavy OCR runtimes.
Detects image-only/scanned documents, password encryption, and corrupted archives.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# System limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB raw file upload limit
MAX_EXTRACTED_BYTES = 16 * 1024       # 16 KB text limit per ingested source (matches user_knowledge)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


@dataclass
class ExtractedDocument:
    """Structured result of local document text extraction."""

    display_name: str
    source_type: str  # "txt", "md", "pdf", "docx"
    content: str
    raw_bytes: int
    character_count: int
    truncated: bool = False
    page_count: Optional[int] = None
    warning: Optional[str] = None


def extract_text_from_bytes(
    raw_data: bytes,
    filename: str,
) -> ExtractedDocument:
    """Extract clean text from raw file bytes safely and deterministically.

    Args:
        raw_data: Raw byte content of uploaded file.
        filename: Original file name used to determine extension and display name.

    Returns:
        ExtractedDocument containing extracted text and metadata.

    Raises:
        ValueError: If file is oversized, unsupported, password-protected, or has no extractable text.
    """
    raw_bytes_len = len(raw_data)
    if raw_bytes_len == 0:
        raise ValueError("Attached file is completely empty (0 bytes).")

    if raw_bytes_len > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(
            f"File size ({raw_bytes_len / (1024 * 1024):.2f} MB) exceeds maximum upload limit of {max_mb} MB."
        )

    clean_filename = Path(filename).name.strip()
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats are: TXT, Markdown, PDF, and DOCX."
        )

    # 1. Plain Text or Markdown
    if suffix in (".txt", ".md", ".markdown"):
        source_type = "md" if suffix in (".md", ".markdown") else "txt"
        try:
            text = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_data.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file with UTF-8 or Latin-1: {e}")

        if "\x00" in text:
            raise ValueError("File appears to contain binary data or null bytes and cannot be parsed as text.")

        extracted_text = text.strip()
        page_count = None

    # 2. PDF Document
    elif suffix == ".pdf":
        source_type = "pdf"
        try:
            import pypdf
        except ImportError:
            raise ValueError("PDF parsing library (pypdf) is not installed in the local environment.")

        try:
            pdf_stream = io.BytesIO(raw_data)
            reader = pypdf.PdfReader(pdf_stream)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PDF file: {e}")

        if reader.is_encrypted:
            raise ValueError(
                "PDF is password-protected or encrypted. Please provide an unencrypted document."
            )

        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError("PDF document contains 0 pages.")

        pages_text = []
        for idx, page in enumerate(reader.pages):
            try:
                pt = page.extract_text() or ""
                if pt.strip():
                    pages_text.append(pt.strip())
            except Exception as e:
                logger.warning(f"Failed to extract text from PDF page {idx + 1}: {e}")

        extracted_text = "\n\n".join(pages_text).strip()
        if not extracted_text:
            raise ValueError(
                "PDF contains no extractable text. Scanned or image-only documents without a text layer are unsupported."
            )

    # 3. Word Document (.docx)
    elif suffix == ".docx":
        source_type = "docx"
        try:
            import docx
        except ImportError:
            raise ValueError("DOCX parsing library (python-docx) is not installed in the local environment.")

        try:
            docx_stream = io.BytesIO(raw_data)
            doc = docx.Document(docx_stream)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid DOCX document: {e}")

        parts = []
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    parts.append(" | ".join(row_cells))

        extracted_text = "\n\n".join(parts).strip()
        page_count = None
        if not extracted_text:
            raise ValueError("DOCX document contains no extractable text.")

    else:
        raise ValueError(f"Unsupported format: {suffix}")

    # Enforce safe character / byte bounds with explicit notification
    content_bytes = extracted_text.encode("utf-8")
    truncated = False
    warning = None

    if len(content_bytes) > MAX_EXTRACTED_BYTES:
        truncated = True
        # Cleanly truncate to within MAX_EXTRACTED_BYTES boundary
        truncated_bytes = content_bytes[:MAX_EXTRACTED_BYTES]
        extracted_text = truncated_bytes.decode("utf-8", errors="ignore")
        warning = (
            f"Document text exceeded {MAX_EXTRACTED_BYTES // 1024} KB limit. "
            f"First {MAX_EXTRACTED_BYTES // 1024} KB loaded for context retrieval."
        )

    return ExtractedDocument(
        display_name=clean_filename,
        source_type=source_type,
        content=extracted_text,
        raw_bytes=raw_bytes_len,
        character_count=len(extracted_text),
        truncated=truncated,
        page_count=page_count,
        warning=warning,
    )

"""Dataset preparation pipeline for GroundUpGPT.

Handles automated download, boilerplate cleaning, normalization, document-level
deduplication, and reproducible train/validation splitting for a genuine
public-domain English literary corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import ssl
from typing import Dict, List, Optional, Tuple
import urllib.request

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()

logger = logging.getLogger(__name__)

# Curated catalog of classic public-domain English literary works from Project Gutenberg.
# All works are in the public domain under US copyright law.
GUTENBERG_CATALOG: List[Dict[str, str]] = [
    {
        "id": "pg1342",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    },
    {
        "id": "pg11",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    },
    {
        "id": "pg1661",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    },
    {
        "id": "pg84",
        "title": "Frankenstein; or, The Modern Prometheus",
        "author": "Mary Wollstonecraft Shelley",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    },
    {
        "id": "pg345",
        "title": "Dracula",
        "author": "Bram Stoker",
        "url": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    },
    {
        "id": "pg98",
        "title": "A Tale of Two Cities",
        "author": "Charles Dickens",
        "url": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    },
    {
        "id": "pg174",
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "url": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    },
    {
        "id": "pg35",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "url": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
    },
    {
        "id": "pg36",
        "title": "The War of the Worlds",
        "author": "H. G. Wells",
        "url": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
    },
    {
        "id": "pg74",
        "title": "The Adventures of Tom Sawyer",
        "author": "Mark Twain",
        "url": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    },
    {
        "id": "pg219",
        "title": "Heart of Darkness",
        "author": "Joseph Conrad",
        "url": "https://www.gutenberg.org/cache/epub/219/pg219.txt",
    },
    {
        "id": "pg43",
        "title": "The Strange Case of Dr. Jekyll and Mr. Hyde",
        "author": "Robert Louis Stevenson",
        "url": "https://www.gutenberg.org/cache/epub/43/pg43.txt",
    },
    {
        "id": "pg120",
        "title": "Treasure Island",
        "author": "Robert Louis Stevenson",
        "url": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
    },
    {
        "id": "pg5200",
        "title": "Metamorphosis",
        "author": "Franz Kafka (trans. David Wyllie)",
        "url": "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",
    },
    {
        "id": "pg2500",
        "title": "Siddhartha",
        "author": "Hermann Hesse (trans. Gunther Olesch et al.)",
        "url": "https://www.gutenberg.org/cache/epub/2500/pg2500.txt",
    },
]

USER_AGENT = "GroundUpGPT-DataPrep/1.0 (Educational Transformer Project; +https://github.com/venox-2007/GroundUpGPT)"


@dataclass
class CorpusDocument:
    """Represents a single document in the training/validation corpus."""

    doc_id: str
    title: str
    author: str
    raw_sha256: str
    raw_bytes: int
    clean_sha256: str
    clean_chars: int
    clean_bytes: int
    split: str = "train"
    clean_text: Optional[str] = None


def compute_sha256(content: bytes | str) -> str:
    """Compute hex SHA-256 digest of bytes or a UTF-8 string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def clean_gutenberg_text(raw_text: str) -> str:
    """Clean Project Gutenberg raw text by removing headers/footers and normalizing whitespace.

    Processing decisions:
    1. Strip standard Project Gutenberg start and end boilerplate notices.
    2. Normalize line breaks (\\r\\n and \\r -> \\n).
    3. Trim trailing whitespace from each line.
    4. Compress consecutive blank lines (3+ newlines -> 2 newlines) while preserving paragraph breaks.
    5. Preserve all natural punctuation, capitalization, dialogue dashes, and quotes.

    Args:
        raw_text: Raw downloaded Project Gutenberg text.

    Returns:
        Cleaned, normalized English text.
    """
    # 1. Normalize line endings to Unix \n
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Strip Project Gutenberg header boilerplate
    # Patterns like: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        text,
        re.IGNORECASE,
    )
    if start_match:
        text = text[start_match.end() :]

    # 3. Strip Project Gutenberg footer boilerplate
    # Patterns like: *** END OF THE PROJECT GUTENBERG EBOOK ... ***
    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        text,
        re.IGNORECASE,
    )
    if end_match:
        text = text[: end_match.start()]

    # 4. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 5. Compress 3+ consecutive newlines to 2 newlines (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


import time

def download_book(
    book_meta: Dict[str, str], raw_dir: Path, timeout: float = 30.0, max_retries: int = 3
) -> Path:
    """Download a single Project Gutenberg book into raw_dir with retries.

    Args:
        book_meta: Dictionary containing 'id', 'url', 'title', 'author'.
        raw_dir: Directory where raw text files are saved.
        timeout: Socket timeout in seconds.
        max_retries: Maximum download retry attempts.

    Returns:
        Path to the saved raw file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{book_meta['id']}.txt"

    if dest.exists() and dest.stat().st_size > 1000:
        logger.info(f"Using cached raw book: {book_meta['id']} ({dest.name})")
        return dest

    logger.info(f"Downloading {book_meta['title']} from {book_meta['url']}...")
    req = urllib.request.Request(book_meta["url"], headers={"User-Agent": USER_AGENT})

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as response:
                content = response.read()

            # Verify UTF-8 decode
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                content = content.decode("utf-8-sig", errors="replace").encode("utf-8")

            with open(dest, "wb") as f:
                f.write(content)

            logger.info(f"Saved {dest.name} ({len(content)} bytes)")
            time.sleep(1.0)  # Gentle delay between requests to respect rate limits
            return dest
        except Exception as err:
            last_err = err
            logger.warning(
                f"Download attempt {attempt}/{max_retries} failed for {book_meta['id']}: {err}. Retrying in {attempt * 2}s..."
            )
            time.sleep(attempt * 2)

    raise RuntimeError(f"Failed to download {book_meta['id']} after {max_retries} attempts: {last_err}")


def deduplicate_documents(
    docs: List[CorpusDocument],
) -> Tuple[List[CorpusDocument], List[CorpusDocument]]:
    """Perform document-level deduplication based on clean text SHA-256 hash.

    Args:
        docs: List of CorpusDocument instances.

    Returns:
        Tuple of (unique_docs, duplicate_docs).
    """
    seen_hashes: set[str] = set()
    unique_docs: List[CorpusDocument] = []
    duplicate_docs: List[CorpusDocument] = []

    for doc in docs:
        if doc.clean_sha256 in seen_hashes:
            logger.warning(
                f"Duplicate document detected and skipped: {doc.doc_id} ('{doc.title}')"
            )
            duplicate_docs.append(doc)
        else:
            seen_hashes.add(doc.clean_sha256)
            unique_docs.append(doc)

    return unique_docs, duplicate_docs


def split_documents(
    docs: List[CorpusDocument], val_ratio: float = 0.20, seed: int = 42
) -> Tuple[List[CorpusDocument], List[CorpusDocument]]:
    """Reproducibly split documents into training and validation partitions.

    Ensures document-level isolation: no document or author passage in validation
    overlaps with the training partition.

    Args:
        docs: List of CorpusDocument instances.
        val_ratio: Fraction of documents allocated to validation (e.g. 0.20).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_docs, val_docs).
    """
    if not docs:
        raise ValueError("Cannot split an empty document list.")

    rng = random.Random(seed)
    shuffled = list(docs)
    rng.shuffle(shuffled)

    num_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_docs = shuffled[:num_val]
    train_docs = shuffled[num_val:]

    # Mark split attribute
    for doc in train_docs:
        doc.split = "train"
    for doc in val_docs:
        doc.split = "val"

    # Enforce strictly disjoint document IDs
    train_ids = {d.doc_id for d in train_docs}
    val_ids = {d.doc_id for d in val_docs}
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(f"Data leak detected! Overlapping documents: {overlap}")

    return train_docs, val_docs


def prepare_corpus(
    catalog: List[Dict[str, str]] = GUTENBERG_CATALOG,
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
    val_ratio: float = 0.20,
    seed: int = 42,
    download: bool = True,
) -> Dict:
    """Download, clean, deduplicate, split, and save the genuine English corpus.

    Args:
        catalog: List of book metadata dictionaries.
        raw_dir: Directory for cached raw texts.
        output_dir: Directory for processed train.txt, val.txt, and manifest.json.
        val_ratio: Validation fraction (by document count).
        seed: Random seed for reproducible document partitioning.
        download: Whether to download missing books from source URLs.

    Returns:
        Corpus manifest dictionary.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_docs: List[CorpusDocument] = []

    for meta in catalog:
        raw_file = raw_dir / f"{meta['id']}.txt"
        if not raw_file.exists():
            if download:
                download_book(meta, raw_dir)
            else:
                logger.warning(
                    f"Raw file {raw_file} missing and download=False. Skipping."
                )
                continue

        raw_bytes = raw_file.read_bytes()
        raw_sha = compute_sha256(raw_bytes)

        # Decode strictly, fall back to utf-8-sig
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("utf-8-sig", errors="replace")

        # Skip empty or trivially small corrupt files
        if len(raw_text.strip()) < 500:
            logger.warning(
                f"Skipping corrupt or trivial file {raw_file} ({len(raw_text)} chars)"
            )
            continue

        clean_text = clean_gutenberg_text(raw_text)
        clean_sha = compute_sha256(clean_text)

        doc = CorpusDocument(
            doc_id=meta["id"],
            title=meta.get("title", meta["id"]),
            author=meta.get("author", "Unknown"),
            raw_sha256=raw_sha,
            raw_bytes=len(raw_bytes),
            clean_sha256=clean_sha,
            clean_chars=len(clean_text),
            clean_bytes=len(clean_text.encode("utf-8")),
            clean_text=clean_text,
        )
        loaded_docs.append(doc)

    if not loaded_docs:
        raise RuntimeError("No valid documents found or downloaded.")

    # Deduplicate documents
    unique_docs, duplicate_docs = deduplicate_documents(loaded_docs)

    # Split documents into train and val
    train_docs, val_docs = split_documents(unique_docs, val_ratio=val_ratio, seed=seed)

    # Write train.txt
    train_path = output_dir / "train.txt"
    train_content = "\n\n<|endoftext|>\n\n".join(d.clean_text for d in train_docs) + "\n"
    train_path.write_text(train_content, encoding="utf-8")

    # Write val.txt
    val_path = output_dir / "val.txt"
    val_content = "\n\n<|endoftext|>\n\n".join(d.clean_text for d in val_docs) + "\n"
    val_path.write_text(val_content, encoding="utf-8")

    # Calculate statistics
    train_chars = sum(d.clean_chars for d in train_docs)
    train_bytes = sum(d.clean_bytes for d in train_docs)
    val_chars = sum(d.clean_chars for d in val_docs)
    val_bytes = sum(d.clean_bytes for d in val_docs)
    total_chars = train_chars + val_chars
    total_bytes = train_bytes + val_bytes

    # Document details without memory-heavy clean_text
    def serialize_doc(d: CorpusDocument) -> Dict:
        data = asdict(d)
        data.pop("clean_text", None)
        return data

    manifest = {
        "dataset_name": "GroundUpGPT Classic English Literature Corpus",
        "description": "Public domain English literary texts from Project Gutenberg for foundational GPT pretraining.",
        "license": "Public Domain (works published prior to 1928; free redistribution under Project Gutenberg License)",
        "source": "https://www.gutenberg.org",
        "date_prepared": "2026-10-31",
        "val_ratio": val_ratio,
        "seed": seed,
        "statistics": {
            "total_documents": len(unique_docs),
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "duplicates_removed": len(duplicate_docs),
            "total_characters": total_chars,
            "total_bytes": total_bytes,
            "train_characters": train_chars,
            "train_bytes": train_bytes,
            "val_characters": val_chars,
            "val_bytes": val_bytes,
            "train_path": str(train_path.relative_to(output_dir.parent.parent)),
            "val_path": str(val_path.relative_to(output_dir.parent.parent)),
            "train_sha256": compute_sha256(train_content),
            "val_sha256": compute_sha256(val_content),
        },
        "documents": [serialize_doc(d) for d in unique_docs],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        f"Corpus preparation complete: {len(unique_docs)} documents, {total_chars:,} chars, {total_bytes:,} bytes."
    )
    logger.info(f"Manifest written to {manifest_path}")

    return manifest

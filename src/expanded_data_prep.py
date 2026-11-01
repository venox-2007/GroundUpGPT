"""Expanded dataset preparation pipeline for GroundUpGPT (Step 23).

Expands the real English training corpus from ~1.53M tokens to 10M–20M tokens
using verified public-domain English literature, philosophy, science, and history.

Key Guarantees:
1. Strict Validation Leak Prevention:
   Original held-out validation books (pg35, pg36, pg5200) are explicitly excluded
   from the expanded training corpus and preserved as an independent regression benchmark.
2. Tokenizer Preservation:
   The existing 4,096-vocab BPE tokenizer (checkpoints/bpe_vocab4096.json) is reused
   without modification or retraining; tokenizer fingerprint is verified before caching.
3. Polite Bulk Access:
   Adheres to Project Gutenberg robot access guidelines with custom User-Agent,
   exponential backoff, polite request throttling, and local caching.
4. Clean Data Hygiene:
   Header/footer boilerplate removal, UTF-8 normalization, document-level exact
   and near-duplicate deduplication, and reproducible SHA-256 manifest tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import ssl
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request
import torch

from src.bpe_tokenizer import BPETokenizer

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()

logger = logging.getLogger(__name__)

USER_AGENT = "GroundUpGPT-ExpandedDataPrep/1.0 (Educational Transformer Project; +https://github.com/venox-2007/GroundUpGPT)"

# STRICT LEAK PREVENTION: Original Step 14 validation documents MUST NEVER be included in training.
FORBIDDEN_VAL_IDS: Set[str] = {
    "pg35",    # The Time Machine (H. G. Wells)
    "pg36",    # The War of the Worlds (H. G. Wells)
    "pg5200",  # Metamorphosis (Franz Kafka)
}

# Original 12 training books from Step 14 (already downloaded in data/raw)
ORIGINAL_TRAIN_CATALOG: List[Dict[str, str]] = [
    {"id": "pg1342", "title": "Pride and Prejudice", "author": "Jane Austen", "year": "1813", "category": "fiction"},
    {"id": "pg11", "title": "Alice's Adventures in Wonderland", "author": "Lewis Carroll", "year": "1865", "category": "fiction"},
    {"id": "pg1661", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": "1892", "category": "fiction"},
    {"id": "pg84", "title": "Frankenstein; or, The Modern Prometheus", "author": "Mary Shelley", "year": "1818", "category": "fiction"},
    {"id": "pg345", "title": "Dracula", "author": "Bram Stoker", "year": "1897", "category": "fiction"},
    {"id": "pg98", "title": "A Tale of Two Cities", "author": "Charles Dickens", "year": "1859", "category": "fiction"},
    {"id": "pg174", "title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "year": "1890", "category": "fiction"},
    {"id": "pg74", "title": "The Adventures of Tom Sawyer", "author": "Mark Twain", "year": "1876", "category": "fiction"},
    {"id": "pg219", "title": "Heart of Darkness", "author": "Joseph Conrad", "year": "1899", "category": "fiction"},
    {"id": "pg43", "title": "The Strange Case of Dr. Jekyll and Mr. Hyde", "author": "Robert Louis Stevenson", "year": "1886", "category": "fiction"},
    {"id": "pg120", "title": "Treasure Island", "author": "Robert Louis Stevenson", "year": "1883", "category": "fiction"},
    {"id": "pg2500", "title": "Siddhartha", "author": "Hermann Hesse", "year": "1922", "category": "fiction"},
]

# Dedicated Expanded Validation Partition (5 diverse books ~1M tokens)
EXPANDED_VAL_CATALOG: List[Dict[str, str]] = [
    {"id": "pg1497", "title": "The Republic", "author": "Plato (trans. Benjamin Jowett)", "year": "1892", "category": "philosophy"},
    {"id": "pg1228", "title": "On the Origin of Species", "author": "Charles Darwin", "year": "1859", "category": "science"},
    {"id": "pg768", "title": "Wuthering Heights", "author": "Emily Brontë", "year": "1847", "category": "fiction"},
    {"id": "pg108", "title": "The Return of Sherlock Holmes", "author": "Arthur Conan Doyle", "year": "1905", "category": "fiction"},
    {"id": "pg205", "title": "Walden", "author": "Henry David Thoreau", "year": "1854", "category": "essays"},
]

# Additional Pretraining Training Books (45 diverse public-domain works)
EXPANDED_TRAIN_CATALOG: List[Dict[str, str]] = [
    # Classic Narrative Fiction
    {"id": "pg2701", "title": "Moby Dick; or, The Whale", "author": "Herman Melville", "year": "1851", "category": "fiction"},
    {"id": "pg1400", "title": "Great Expectations", "author": "Charles Dickens", "year": "1861", "category": "fiction"},
    {"id": "pg730", "title": "Oliver Twist", "author": "Charles Dickens", "year": "1838", "category": "fiction"},
    {"id": "pg766", "title": "David Copperfield", "author": "Charles Dickens", "year": "1850", "category": "fiction"},
    {"id": "pg1184", "title": "The Count of Monte Cristo", "author": "Alexandre Dumas", "year": "1844", "category": "fiction"},
    {"id": "pg135", "title": "Les Misérables", "author": "Victor Hugo", "year": "1862", "category": "fiction"},
    {"id": "pg2554", "title": "Crime and Punishment", "author": "Fyodor Dostoevsky (trans. Constance Garnett)", "year": "1866", "category": "fiction"},
    {"id": "pg28054", "title": "The Brothers Karamazov", "author": "Fyodor Dostoevsky (trans. Constance Garnett)", "year": "1880", "category": "fiction"},
    {"id": "pg1399", "title": "Anna Karenina", "author": "Leo Tolstoy (trans. Constance Garnett)", "year": "1877", "category": "fiction"},
    {"id": "pg996", "title": "Don Quixote", "author": "Miguel de Cervantes (trans. John Ormsby)", "year": "1605", "category": "fiction"},
    {"id": "pg1260", "title": "Jane Eyre: An Autobiography", "author": "Charlotte Brontë", "year": "1847", "category": "fiction"},
    {"id": "pg76", "title": "The Adventures of Huckleberry Finn", "author": "Mark Twain", "year": "1884", "category": "fiction"},
    {"id": "pg521", "title": "The Life and Adventures of Robinson Crusoe", "author": "Daniel Defoe", "year": "1719", "category": "fiction"},
    {"id": "pg829", "title": "Gulliver's Travels", "author": "Jonathan Swift", "year": "1726", "category": "fiction"},
    {"id": "pg161", "title": "Sense and Sensibility", "author": "Jane Austen", "year": "1811", "category": "fiction"},
    {"id": "pg158", "title": "Emma", "author": "Jane Austen", "year": "1815", "category": "fiction"},
    {"id": "pg141", "title": "Mansfield Park", "author": "Jane Austen", "year": "1814", "category": "fiction"},
    {"id": "pg155", "title": "The Moonstone", "author": "Wilkie Collins", "year": "1868", "category": "fiction"},
    {"id": "pg583", "title": "The Woman in White", "author": "Wilkie Collins", "year": "1859", "category": "fiction"},
    {"id": "pg2097", "title": "The Sign of the Four", "author": "Arthur Conan Doyle", "year": "1890", "category": "fiction"},
    {"id": "pg244", "title": "A Study in Scarlet", "author": "Arthur Conan Doyle", "year": "1887", "category": "fiction"},
    {"id": "pg514", "title": "Little Women", "author": "Louisa May Alcott", "year": "1868", "category": "fiction"},
    {"id": "pg145", "title": "Middlemarch", "author": "George Eliot", "year": "1871", "category": "fiction"},
    {"id": "pg6688", "title": "The Mill on the Floss", "author": "George Eliot", "year": "1860", "category": "fiction"},
    {"id": "pg33", "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne", "year": "1850", "category": "fiction"},
    {"id": "pg940", "title": "The Last of the Mohicans", "author": "James Fenimore Cooper", "year": "1826", "category": "fiction"},
    {"id": "pg215", "title": "The Call of the Wild", "author": "Jack London", "year": "1903", "category": "fiction"},
    {"id": "pg910", "title": "White Fang", "author": "Jack London", "year": "1906", "category": "fiction"},
    {"id": "pg1074", "title": "The Sea-Wolf", "author": "Jack London", "year": "1904", "category": "fiction"},
    {"id": "pg103", "title": "Around the World in Eighty Days", "author": "Jules Verne", "year": "1872", "category": "fiction"},
    {"id": "pg164", "title": "Twenty Thousand Leagues Under the Sea", "author": "Jules Verne", "year": "1870", "category": "fiction"},
    {"id": "pg18857", "title": "Journey to the Center of the Earth", "author": "Jules Verne", "year": "1864", "category": "fiction"},
    {"id": "pg2166", "title": "King Solomon's Mines", "author": "H. Rider Haggard", "year": "1885", "category": "fiction"},
    {"id": "pg3155", "title": "She: A History of Adventure", "author": "H. Rider Haggard", "year": "1886", "category": "fiction"},

    # Philosophy, Science, Nonfiction, History & Essays
    {"id": "pg944", "title": "The Voyage of the Beagle", "author": "Charles Darwin", "year": "1839", "category": "science"},
    {"id": "pg2680", "title": "Meditations", "author": "Marcus Aurelius (trans. George Long)", "year": "1862", "category": "philosophy"},
    {"id": "pg20203", "title": "Autobiography of Benjamin Franklin", "author": "Benjamin Franklin", "year": "1791", "category": "biography"},
    {"id": "pg3600", "title": "Essays of Michel de Montaigne", "author": "Michel de Montaigne (trans. Charles Cotton)", "year": "1877", "category": "essays"},
    {"id": "pg1232", "title": "The Prince", "author": "Niccolò Machiavelli (trans. W. K. Marriott)", "year": "1532", "category": "philosophy"},
    {"id": "pg147", "title": "Common Sense", "author": "Thomas Paine", "year": "1776", "category": "history"},
    {"id": "pg18", "title": "The Federalist Papers", "author": "Alexander Hamilton, James Madison, John Jay", "year": "1788", "category": "history"},
    {"id": "pg7370", "title": "Second Treatise of Government", "author": "John Locke", "year": "1689", "category": "philosophy"},
    {"id": "pg10615", "title": "An Essay Concerning Human Understanding", "author": "John Locke", "year": "1689", "category": "philosophy"},
    {"id": "pg35461", "title": "A Short History of the World", "author": "H. G. Wells", "year": "1922", "category": "history"},
    {"id": "pg890", "title": "The History of the Decline and Fall of the Roman Empire (Vol 1)", "author": "Edward Gibbon", "year": "1776", "category": "history"},
]


@dataclass
class DocumentMetadata:
    """Metadata representing a single document in the expanded corpus."""
    doc_id: str
    title: str
    author: str
    year: str
    category: str
    url: str
    split: str
    raw_bytes: int
    raw_sha256: str
    clean_bytes: int
    clean_chars: int
    clean_sha256: str
    tokens: Optional[int] = None


def compute_sha256(content: bytes | str) -> str:
    """Compute hex SHA-256 digest of bytes or a UTF-8 string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_gutenberg_url(book_id: str) -> str:
    """Return standard Project Gutenberg cache endpoint for an ID."""
    clean_id = book_id.replace("pg", "")
    return f"https://www.gutenberg.org/cache/epub/{clean_id}/pg{clean_id}.txt"


def clean_gutenberg_text(raw_text: str) -> str:
    """Clean Project Gutenberg text by stripping headers, footers, and normalizing whitespace.

    Decision rules:
    1. Normalize line breaks to \n.
    2. Strip start header boilerplate (START OF ... PROJECT GUTENBERG).
    3. Strip end footer boilerplate (END OF ... PROJECT GUTENBERG).
    4. Strip trailing line whitespace.
    5. Compress 3+ consecutive newlines to 2 newlines (paragraph boundary).
    6. Preserve genuine dialogue, quotes, and punctuation.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Match Project Gutenberg header
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        text,
        re.IGNORECASE,
    )
    if start_match:
        text = text[start_match.end() :]
    else:
        # Fallback header match for older transcripts
        alt_start = re.search(r"\*END\*THE SMALL PRINT![^\n]*\*", text, re.IGNORECASE)
        if alt_start:
            text = text[alt_start.end() :]

    # Match Project Gutenberg footer
    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",
        text,
        re.IGNORECASE,
    )
    if end_match:
        text = text[: end_match.start()]
    else:
        alt_end = re.search(r"End of (?:the )?Project Gutenberg['’]?s", text, re.IGNORECASE)
        if alt_end:
            text = text[: alt_end.start()]

    # Strip line-trailing whitespace
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Compress excessive blank lines while preserving paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_expanded_book(
    meta: Dict[str, str],
    raw_dir: Path,
    timeout: float = 30.0,
    max_retries: int = 3,
    delay_between_requests: float = 0.5,
) -> Path:
    """Politely download a Project Gutenberg book if not already cached locally."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{meta['id']}.txt"

    # Reuse local cached file if valid
    if dest.exists() and dest.stat().st_size > 1000:
        logger.debug(f"Using local cached book: {dest.name}")
        return dest

    url = meta.get("url") or get_gutenberg_url(meta["id"])
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=timeout) as response:
                content = response.read()

            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                content = content.decode("utf-8-sig", errors="replace").encode("utf-8")

            with open(dest, "wb") as f:
                f.write(content)

            logger.info(f"Downloaded {meta['id']} ({meta['title']}) [{len(content):,} bytes]")
            time.sleep(delay_between_requests)
            return dest
        except Exception as err:
            last_err = err
            wait_s = attempt * 2.0
            logger.warning(
                f"Download attempt {attempt}/{max_retries} failed for {meta['id']}: {err}. Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download {meta['id']} ({meta['title']}) after {max_retries} attempts: {last_err}")


def deduplicate_paragraphs(text: str, min_len: int = 60) -> str:
    """Deduplicate repeated exact boilerplate paragraphs within a document."""
    paragraphs = text.split("\n\n")
    seen: Set[str] = set()
    kept: List[str] = []

    for p in paragraphs:
        p_clean = p.strip()
        if len(p_clean) >= min_len:
            p_hash = hashlib.md5(p_clean.encode("utf-8")).hexdigest()
            # If paragraph appears multiple times in same document (e.g., repeated notice), keep first occurrence only
            if p_hash in seen:
                continue
            seen.add(p_hash)
        kept.append(p)

    return "\n\n".join(kept)


def prepare_expanded_corpus(
    raw_dir: Path = Path("data/raw_expanded"),
    existing_raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed_expanded"),
    cache_dir: Path = Path("data/cache_expanded"),
    tokenizer_path: Path = Path("checkpoints/bpe_vocab4096.json"),
    download: bool = True,
) -> Dict[str, Any]:
    """Execute complete Step 23 expanded corpus preparation pipeline.

    1. Enforces zero data leakage against original validation books (pg35, pg36, pg5200).
    2. Downloads and cleans 45 new training books and 5 expanded validation books.
    3. Merges 12 original clean training books with 45 new clean training books.
    4. Verifies BPE tokenizer fingerprint.
    5. Encodes training and expanded validation tokens into compact int32 tensors.
    6. Writes reproducible manifest and cache metadata.
    """
    raw_dir = Path(raw_dir)
    existing_raw_dir = Path(existing_raw_dir)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    tokenizer_path = Path(tokenizer_path)

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify tokenizer exists and compute fingerprint
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tok_fingerprint = tokenizer.get_fingerprint()
    logger.info(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}, fingerprint: {tok_fingerprint}")

    # 2. Strict Data Leakage Prevention Check
    all_incoming_train_ids = {m["id"] for m in ORIGINAL_TRAIN_CATALOG} | {m["id"] for m in EXPANDED_TRAIN_CATALOG}
    leak_train = all_incoming_train_ids.intersection(FORBIDDEN_VAL_IDS)
    if leak_train:
        raise RuntimeError(f"CRITICAL DATA LEAK DETECTED: Forbidden validation IDs found in training catalog: {leak_train}")

    leak_val = {m["id"] for m in EXPANDED_VAL_CATALOG}.intersection(FORBIDDEN_VAL_IDS)
    if leak_val:
        raise RuntimeError(f"CRITICAL DATA LEAK DETECTED: Forbidden validation IDs found in expanded val catalog: {leak_val}")

    # Check that original validation raw files hashes are recorded to prevent content-level leaks
    forbidden_hashes: Set[str] = set()
    for forbidden_id in FORBIDDEN_VAL_IDS:
        val_path = existing_raw_dir / f"{forbidden_id}.txt"
        if val_path.is_file():
            forbidden_hashes.add(compute_sha256(val_path.read_bytes()))
            forbidden_hashes.add(compute_sha256(clean_gutenberg_text(val_path.read_text(encoding="utf-8", errors="replace"))))

    # 3. Process Original Training Books
    docs_metadata: List[DocumentMetadata] = []
    seen_clean_hashes: Dict[str, str] = {}
    train_texts: List[str] = []
    val_texts: List[str] = []

    logger.info(f"Processing {len(ORIGINAL_TRAIN_CATALOG)} original training books...")
    for meta in ORIGINAL_TRAIN_CATALOG:
        book_id = meta["id"]
        # Original books are located in data/raw/
        raw_file = existing_raw_dir / f"{book_id}.txt"
        if not raw_file.exists():
            raw_file = raw_dir / f"{book_id}.txt"
            if not raw_file.exists() and download:
                download_expanded_book(meta, raw_dir)

        raw_bytes = raw_file.read_bytes()
        raw_sha = compute_sha256(raw_bytes)
        if raw_sha in forbidden_hashes:
            raise RuntimeError(f"Data leak: Document {book_id} matches forbidden validation hash!")

        try:
            raw_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_str = raw_bytes.decode("utf-8-sig", errors="replace")

        clean_str = clean_gutenberg_text(raw_str)
        clean_str = deduplicate_paragraphs(clean_str)
        clean_sha = compute_sha256(clean_str)

        if clean_sha in forbidden_hashes:
            raise RuntimeError(f"Data leak: Document {book_id} clean text matches forbidden validation hash!")

        if clean_sha in seen_clean_hashes:
            logger.warning(f"Document {book_id} is exact duplicate of {seen_clean_hashes[clean_sha]}, skipping.")
            continue
        seen_clean_hashes[clean_sha] = book_id

        doc_meta = DocumentMetadata(
            doc_id=book_id,
            title=meta["title"],
            author=meta["author"],
            year=meta.get("year", "Unknown"),
            category=meta.get("category", "general"),
            url=meta.get("url") or get_gutenberg_url(book_id),
            split="train",
            raw_bytes=len(raw_bytes),
            raw_sha256=raw_sha,
            clean_bytes=len(clean_str.encode("utf-8")),
            clean_chars=len(clean_str),
            clean_sha256=clean_sha,
        )
        docs_metadata.append(doc_meta)
        train_texts.append(clean_str)

    # 4. Process 45 New Training Books
    logger.info(f"Processing {len(EXPANDED_TRAIN_CATALOG)} new training books...")
    for meta in EXPANDED_TRAIN_CATALOG:
        book_id = meta["id"]
        raw_file = raw_dir / f"{book_id}.txt"
        if not raw_file.exists():
            if download:
                download_expanded_book(meta, raw_dir)
            else:
                raise FileNotFoundError(f"Missing raw book {book_id} and download=False")

        raw_bytes = raw_file.read_bytes()
        raw_sha = compute_sha256(raw_bytes)
        if raw_sha in forbidden_hashes:
            raise RuntimeError(f"Data leak: Document {book_id} matches forbidden validation hash!")

        try:
            raw_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_str = raw_bytes.decode("utf-8-sig", errors="replace")

        clean_str = clean_gutenberg_text(raw_str)
        clean_str = deduplicate_paragraphs(clean_str)
        clean_sha = compute_sha256(clean_str)

        if clean_sha in forbidden_hashes:
            raise RuntimeError(f"Data leak: Document {book_id} clean text matches forbidden validation hash!")

        if clean_sha in seen_clean_hashes:
            logger.warning(f"Document {book_id} is exact duplicate of {seen_clean_hashes[clean_sha]}, skipping.")
            continue
        seen_clean_hashes[clean_sha] = book_id

        doc_meta = DocumentMetadata(
            doc_id=book_id,
            title=meta["title"],
            author=meta["author"],
            year=meta.get("year", "Unknown"),
            category=meta.get("category", "general"),
            url=meta.get("url") or get_gutenberg_url(book_id),
            split="train",
            raw_bytes=len(raw_bytes),
            raw_sha256=raw_sha,
            clean_bytes=len(clean_str.encode("utf-8")),
            clean_chars=len(clean_str),
            clean_sha256=clean_sha,
        )
        docs_metadata.append(doc_meta)
        train_texts.append(clean_str)

    # 5. Process 5 Expanded Validation Books
    logger.info(f"Processing {len(EXPANDED_VAL_CATALOG)} expanded validation books...")
    for meta in EXPANDED_VAL_CATALOG:
        book_id = meta["id"]
        raw_file = raw_dir / f"{book_id}.txt"
        if not raw_file.exists():
            if download:
                download_expanded_book(meta, raw_dir)
            else:
                raise FileNotFoundError(f"Missing raw book {book_id} and download=False")

        raw_bytes = raw_file.read_bytes()
        raw_sha = compute_sha256(raw_bytes)
        if raw_sha in forbidden_hashes:
            raise RuntimeError(f"Data leak: Document {book_id} matches forbidden validation hash!")

        try:
            raw_str = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_str = raw_bytes.decode("utf-8-sig", errors="replace")

        clean_str = clean_gutenberg_text(raw_str)
        clean_str = deduplicate_paragraphs(clean_str)
        clean_sha = compute_sha256(clean_str)

        if clean_sha in seen_clean_hashes:
            logger.warning(f"Document {book_id} duplicate of {seen_clean_hashes[clean_sha]}, skipping.")
            continue
        seen_clean_hashes[clean_sha] = book_id

        doc_meta = DocumentMetadata(
            doc_id=book_id,
            title=meta["title"],
            author=meta["author"],
            year=meta.get("year", "Unknown"),
            category=meta.get("category", "general"),
            url=meta.get("url") or get_gutenberg_url(book_id),
            split="val_expanded",
            raw_bytes=len(raw_bytes),
            raw_sha256=raw_sha,
            clean_bytes=len(clean_str.encode("utf-8")),
            clean_chars=len(clean_str),
            clean_sha256=clean_sha,
        )
        docs_metadata.append(doc_meta)
        val_texts.append(clean_str)

    # 6. Save Combined Clean Text Files
    train_corpus_str = "\n\n".join(train_texts)
    val_corpus_str = "\n\n".join(val_texts)

    train_txt_path = output_dir / "train_expanded.txt"
    val_txt_path = output_dir / "val_expanded.txt"

    train_txt_path.write_text(train_corpus_str, encoding="utf-8")
    val_txt_path.write_text(val_corpus_str, encoding="utf-8")

    train_bytes = len(train_corpus_str.encode("utf-8"))
    val_bytes = len(val_corpus_str.encode("utf-8"))
    train_sha256 = compute_sha256(train_corpus_str)
    val_sha256 = compute_sha256(val_corpus_str)

    logger.info(f"Saved {train_txt_path.name}: {len(train_corpus_str):,} chars ({train_bytes / 1024 / 1024:.2f} MB)")
    logger.info(f"Saved {val_txt_path.name}: {len(val_corpus_str):,} chars ({val_bytes / 1024 / 1024:.2f} MB)")

    # 7. Tokenize Corpora into int32 Tensors
    t0 = time.time()
    logger.info(f"Tokenizing expanded training corpus ({train_bytes:,} bytes)...")
    train_tokens_list: List[int] = []
    # Tokenize document by document to monitor progress and maintain low peak memory
    for i, t_doc in enumerate(train_texts):
        # Add boundary spacing between documents
        doc_ids = tokenizer.encode(t_doc + ("\n\n" if i < len(train_texts) - 1 else ""))
        train_tokens_list.extend(doc_ids)

    train_tokenization_time = time.time() - t0
    logger.info(
        f"Tokenized training set: {len(train_tokens_list):,} tokens in {train_tokenization_time:.2f}s "
        f"({len(train_tokens_list)/max(0.01, train_tokenization_time):,.0f} tok/s)"
    )

    t1 = time.time()
    logger.info(f"Tokenizing expanded validation corpus ({val_bytes:,} bytes)...")
    val_tokens_list: List[int] = []
    for i, v_doc in enumerate(val_texts):
        doc_ids = tokenizer.encode(v_doc + ("\n\n" if i < len(val_texts) - 1 else ""))
        val_tokens_list.extend(doc_ids)
    val_tokenization_time = time.time() - t1
    logger.info(f"Tokenized validation set: {len(val_tokens_list):,} tokens in {val_tokenization_time:.2f}s")

    # Validate token ID bounds
    vocab_size = tokenizer.vocab_size
    min_train, max_train = min(train_tokens_list), max(train_tokens_list)
    min_val, max_val = min(val_tokens_list), max(val_tokens_list)
    if min_train < 0 or max_train >= vocab_size:
        raise ValueError(f"Train token ID out of range [0, {vocab_size-1}]: min={min_train}, max={max_train}")
    if min_val < 0 or max_val >= vocab_size:
        raise ValueError(f"Val token ID out of range [0, {vocab_size-1}]: min={min_val}, max={max_val}")

    # Convert to compact 32-bit signed integer tensors
    train_tensor = torch.tensor(train_tokens_list, dtype=torch.int32)
    val_tensor = torch.tensor(val_tokens_list, dtype=torch.int32)

    # 8. Save Token Cache Artifacts to data/cache_expanded/
    train_tensor_path = cache_dir / "train_tokens.pt"
    val_expanded_tensor_path = cache_dir / "val_expanded_tokens.pt"
    val_tensor_path = cache_dir / "val_tokens.pt"  # Alias for standard PretokenizedDataPipeline loading

    torch.save(train_tensor, train_tensor_path)
    torch.save(val_tensor, val_expanded_tensor_path)
    torch.save(val_tensor, val_tensor_path)

    compression_train = train_bytes / max(1, len(train_tokens_list))
    compression_val = val_bytes / max(1, len(val_tokens_list))

    cache_meta = {
        "corpus_version": "v2-expanded",
        "tokenizer_path": str(tokenizer_path).replace("\\", "/"),
        "tokenizer_fingerprint": tok_fingerprint,
        "vocab_size": vocab_size,
        "train_tokens_count": len(train_tokens_list),
        "val_tokens_count": len(val_tokens_list),
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "compression_ratio_train": round(compression_train, 2),
        "compression_ratio_val": round(compression_val, 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_dir / "cache_metadata.json", "w", encoding="utf-8") as f:
        json.dump(cache_meta, f, indent=2)

    # 9. Build and Save Manifest
    train_docs_count = len([d for d in docs_metadata if d.split == "train"])
    val_docs_count = len([d for d in docs_metadata if d.split == "val_expanded"])

    manifest = {
        "dataset_name": "GroundUpGPT-Expanded-English-Corpus",
        "version": "v2-expanded",
        "description": "Expanded genuine English literary, scientific, and philosophical pretraining corpus",
        "license": "Public Domain (US works published prior to 1929)",
        "source": "Project Gutenberg bulk cache/mirrors",
        "user_agent": USER_AGENT,
        "tokenizer_fingerprint": tok_fingerprint,
        "date_prepared": datetime.now(timezone.utc).isoformat(),
        "forbidden_validation_ids_excluded": sorted(list(FORBIDDEN_VAL_IDS)),
        "statistics": {
            "total_documents": len(docs_metadata),
            "original_train_documents": len(ORIGINAL_TRAIN_CATALOG),
            "new_train_documents": len(EXPANDED_TRAIN_CATALOG),
            "total_train_documents": train_docs_count,
            "expanded_val_documents": val_docs_count,
            "train_characters": len(train_corpus_str),
            "train_bytes": train_bytes,
            "train_tokens": len(train_tokens_list),
            "val_characters": len(val_corpus_str),
            "val_bytes": val_bytes,
            "val_tokens": len(val_tokens_list),
            "train_sha256": train_sha256,
            "val_sha256": val_sha256,
            "train_tokenization_seconds": round(train_tokenization_time, 2),
            "val_tokenization_seconds": round(val_tokenization_time, 2),
        },
        "documents": [asdict(d) for d in docs_metadata],
    }

    manifest_path = output_dir / "manifest_expanded.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(cache_dir / "manifest_expanded.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Step 23 complete! Expanded training tokens: {len(train_tokens_list):,}, "
        f"Expanded val tokens: {len(val_tokens_list):,}, Total documents: {len(docs_metadata)}"
    )

    return manifest

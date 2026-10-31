"""Unit tests for GroundUpGPT dataset preparation pipeline."""

import json
from pathlib import Path

import pytest

from src.data_prep import (
    CorpusDocument,
    clean_gutenberg_text,
    compute_sha256,
    deduplicate_documents,
    prepare_corpus,
    split_documents,
)


def test_clean_gutenberg_text():
    raw_sample = (
        "The Project Gutenberg eBook of Sample Book\r\n"
        "Some Legal Preamble text here.\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***\r\n"
        "\r\n"
        "CHAPTER I.\r\n"
        "\r\n\r\n\r\n"
        "It is a truth universally acknowledged, that a single man in possession "
        "of a good fortune, must be in want of a wife.   \r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***\r\n"
        "Project Gutenberg license footer and legal text.\r\n"
    )

    clean = clean_gutenberg_text(raw_sample)

    assert "Legal Preamble" not in clean
    assert "license footer" not in clean
    assert "*** START" not in clean
    assert "*** END" not in clean
    assert "CHAPTER I." in clean
    assert (
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
        in clean
    )
    # Line endings normalized
    assert "\r" not in clean
    # Multiple blank lines compressed
    assert "\n\n\n" not in clean


def test_deduplicate_documents():
    doc1 = CorpusDocument(
        doc_id="doc1",
        title="Title 1",
        author="Author 1",
        raw_sha256="abc",
        raw_bytes=100,
        clean_sha256="hash_identical",
        clean_chars=50,
        clean_bytes=50,
        clean_text="Identical text body.",
    )
    doc2 = CorpusDocument(
        doc_id="doc2_duplicate",
        title="Title 2",
        author="Author 2",
        raw_sha256="def",
        raw_bytes=100,
        clean_sha256="hash_identical",
        clean_chars=50,
        clean_bytes=50,
        clean_text="Identical text body.",
    )
    doc3 = CorpusDocument(
        doc_id="doc3_unique",
        title="Title 3",
        author="Author 3",
        raw_sha256="ghi",
        raw_bytes=120,
        clean_sha256="hash_unique",
        clean_chars=60,
        clean_bytes=60,
        clean_text="Different unique text body.",
    )

    unique, duplicates = deduplicate_documents([doc1, doc2, doc3])
    assert len(unique) == 2
    assert len(duplicates) == 1
    assert duplicates[0].doc_id == "doc2_duplicate"
    assert {d.doc_id for d in unique} == {"doc1", "doc3_unique"}


def test_split_documents_isolation_and_reproducibility():
    docs = [
        CorpusDocument(
            doc_id=f"doc_{i}",
            title=f"Book {i}",
            author="Author",
            raw_sha256=f"raw_{i}",
            raw_bytes=1000,
            clean_sha256=f"clean_{i}",
            clean_chars=800,
            clean_bytes=800,
        )
        for i in range(10)
    ]

    train1, val1 = split_documents(docs, val_ratio=0.20, seed=42)
    train2, val2 = split_documents(docs, val_ratio=0.20, seed=42)
    train_diff, val_diff = split_documents(docs, val_ratio=0.20, seed=999)

    # Disjoint document sets (zero leak)
    train_ids = {d.doc_id for d in train1}
    val_ids = {d.doc_id for d in val1}
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == len(docs)
    assert len(val_ids) == 2  # 20% of 10

    # Deterministic reproducibility
    assert [d.doc_id for d in train1] == [d.doc_id for d in train2]
    assert [d.doc_id for d in val1] == [d.doc_id for d in val2]

    # Different seed yields different partition order
    assert [d.doc_id for d in train1] != [d.doc_id for d in train_diff]


def test_prepare_corpus_end_to_end_local(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()

    # Create 4 synthetic mock books
    catalog = []
    for i in range(4):
        book_id = f"mock_{i}"
        raw_file = raw_dir / f"{book_id}.txt"
        text = (
            f"Header {i}\n"
            f"*** START OF THE PROJECT GUTENBERG EBOOK MOCK {i} ***\n"
            f"This is the genuine content of book {i}. " * 30 + "\n"
            f"*** END OF THE PROJECT GUTENBERG EBOOK MOCK {i} ***\n"
            f"Footer {i}\n"
        )
        raw_file.write_text(text, encoding="utf-8")
        catalog.append(
            {
                "id": book_id,
                "title": f"Mock Book {i}",
                "author": f"Author {i}",
                "url": f"https://example.com/{book_id}.txt",
            }
        )

    manifest = prepare_corpus(
        catalog=catalog,
        raw_dir=raw_dir,
        output_dir=out_dir,
        val_ratio=0.25,
        seed=123,
        download=False,
    )

    assert (out_dir / "train.txt").exists()
    assert (out_dir / "val.txt").exists()
    assert (out_dir / "manifest.json").exists()

    stats = manifest["statistics"]
    assert stats["total_documents"] == 4
    assert stats["val_documents"] == 1
    assert stats["train_documents"] == 3
    assert stats["duplicates_removed"] == 0

    # Check SHA-256 consistency
    train_content = (out_dir / "train.txt").read_text(encoding="utf-8")
    assert compute_sha256(train_content) == stats["train_sha256"]

    # Check document isolation
    train_docs = {
        d["doc_id"] for d in manifest["documents"] if d["split"] == "train"
    }
    val_docs = {d["doc_id"] for d in manifest["documents"] if d["split"] == "val"}
    assert len(train_docs.intersection(val_docs)) == 0

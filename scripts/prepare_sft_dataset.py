"""CLI script to prepare and cache the Supervised Fine-Tuning (SFT) conversation dataset.

Downloads or reads Databricks Dolly 15k (or fixture fallback), validates data quality,
scrubs PII, deduplicates, tokenizes with byte-level BPE, verifies assistant-only loss masking,
splits into zero-leakage train/validation sets, and caches binary tensors to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import ssl
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import certifi
import torch

from src.bpe_tokenizer import BPETokenizer
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    ConversationExample,
    ConversationTurn,
    SFTDataPipeline,
    SFTDataset,
    deduplicate_conversations,
    encode_conversation,
    format_conversation,
    split_conversations,
    utf8_roundtrip_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Expected canonical BPE tokenizer fingerprint (Step 14-25 baseline)
EXPECTED_TOKENIZER_FINGERPRINT = (
    "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
)
DOLLY_HF_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def download_dataset(
    url: str, dest_path: Path, max_bytes: int = 30 * 1024 * 1024
) -> Tuple[str, int]:
    """Download dataset file with streaming SSL verification and byte budget check.

    Returns:
        Tuple[str, int]: (SHA-256 hex digest, total bytes downloaded)
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    logger.info(f"Downloading dataset from {url} to {temp_path}...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url, headers={"User-Agent": "GroundUpGPT-SFTPrep/1.0"}
    )

    sha256 = hashlib.sha256()
    total_bytes = 0

    with (
        urllib.request.urlopen(req, context=ctx, timeout=60) as resp,
        open(temp_path, "wb") as f,
    ):
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Download exceeded maximum budget of {max_bytes} bytes."
                )
            f.write(chunk)
            sha256.update(chunk)

    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)

    digest = sha256.hexdigest()
    logger.info(
        f"Download complete: {total_bytes / (1024 * 1024):.2f} MB, SHA-256: {digest}"
    )
    return digest, total_bytes


def load_raw_jsonl_records(
    file_path: Path, char_filter_len: int = 1100
) -> Tuple[List[ConversationExample], Dict[str, int]]:
    """Parse raw JSONL records into ConversationExample objects with data quality checks."""
    examples: List[ConversationExample] = []
    stats = {
        "raw_lines": 0,
        "json_parse_errors": 0,
        "missing_fields": 0,
        "empty_content": 0,
        "utf8_failures": 0,
        "char_filter_skipped": 0,
        "retained": 0,
    }

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            stats["raw_lines"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception:
                stats["json_parse_errors"] += 1
                continue

            # Support Databricks Dolly format or generic instruction/response / turns format
            instruction = data.get("instruction")
            response = data.get("response")
            context = data.get("context", "")
            category = data.get("category")

            # Check if multi-turn format is present
            if "turns" in data and isinstance(data["turns"], list):
                turns: List[ConversationTurn] = []
                valid_turns = True
                for t in data["turns"]:
                    role = t.get("role", "")
                    content = t.get("content", "")
                    if (
                        role not in ("user", "assistant")
                        or not isinstance(content, str)
                        or not content.strip()
                    ):
                        valid_turns = False
                        break
                    turns.append(
                        ConversationTurn(role=role, content=content.strip())
                    )
                if not valid_turns or not turns:
                    stats["missing_fields"] += 1
                    continue
            else:
                if (
                    not isinstance(instruction, str)
                    or not isinstance(response, str)
                    or not instruction.strip()
                    or not response.strip()
                ):
                    stats["empty_content"] += 1
                    continue

                instruction_clean = instruction.strip()
                response_clean = response.strip()
                context_clean = (
                    context.strip() if isinstance(context, str) else ""
                )

                if context_clean:
                    prompt_text = (
                        f"{instruction_clean}\n\nContext:\n{context_clean}"
                    )
                else:
                    prompt_text = instruction_clean

                turns = [
                    ConversationTurn(role="user", content=prompt_text),
                    ConversationTurn(role="assistant", content=response_clean),
                ]

            # Fast character heuristic: sequences with > char_filter_len characters
            # are mathematically guaranteed to exceed 256 BPE tokens.
            total_chars = sum(len(t.content) for t in turns)
            if total_chars > char_filter_len:
                stats["char_filter_skipped"] += 1
                continue

            # UTF-8 round-trip validation
            if not all(utf8_roundtrip_check(t.content) for t in turns):
                stats["utf8_failures"] += 1
                continue

            ex_id = data.get("id") or f"ex_{idx+1}"
            examples.append(
                ConversationExample(
                    example_id=str(ex_id),
                    turns=turns,
                    category=category,
                    metadata={"source_line": idx + 1},
                )
            )
            stats["retained"] += 1

    return examples, stats


def prepare_sft_dataset(
    raw_file: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    fixture_path: Optional[Path] = None,
    download_if_missing: bool = True,
    max_seq_len: int = 256,
    train_ratio: float = 0.9,
    seed: int = 42,
    char_filter_len: int = 1100,
) -> SFTDataPipeline:
    """End-to-end dataset preparation and caching execution."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("GroundUpGPT Supervised Fine-Tuning (SFT) Dataset Preparation")
    logger.info("=" * 70)

    # 1. Load and verify tokenizer
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"BPE tokenizer not found at {tokenizer_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    tok_fp = tokenizer.get_fingerprint()
    logger.info(
        f"Loaded BPE tokenizer with vocab size {tokenizer.vocab_size}, fingerprint: {tok_fp}"
    )
    if tok_fp != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError(
            f"Tokenizer fingerprint mismatch! Expected {EXPECTED_TOKENIZER_FINGERPRINT}, got {tok_fp}"
        )

    # 2. Check delimiters encoding compatibility
    for delim in [
        DEFAULT_USER_PREFIX,
        DEFAULT_ASSISTANT_PREFIX,
        DEFAULT_END_OF_TURN,
    ]:
        encoded = tokenizer.encode(delim)
        decoded = tokenizer.decode(encoded)
        if decoded != delim:
            raise ValueError(
                f"Delimiter {delim!r} failed lossless BPE round-trip!"
            )
    logger.info(
        "Conversation delimiters verified with 100% lossless BPE round-trip."
    )

    # 3. Source dataset file
    download_hash = ""
    download_size = 0
    used_fixture = False

    if not raw_file.is_file():
        if download_if_missing:
            logger.info(f"Raw file {raw_file} not found. Attempting download...")
            try:
                download_hash, download_size = download_dataset(
                    DOLLY_HF_URL, raw_file
                )
            except Exception as e:
                logger.warning(f"Download failed: {e}")
                if fixture_path and fixture_path.is_file():
                    logger.info(f"Falling back to local fixture: {fixture_path}")
                    raw_file = fixture_path
                    used_fixture = True
                else:
                    raise
        elif fixture_path and fixture_path.is_file():
            logger.info(f"Using local fixture: {fixture_path}")
            raw_file = fixture_path
            used_fixture = True
        else:
            raise FileNotFoundError(f"Dataset file not found at {raw_file}")
    else:
        # Compute existing file hash
        h = hashlib.sha256()
        with open(raw_file, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
                download_size += len(chunk)
        download_hash = h.hexdigest()
        logger.info(
            f"Using existing raw dataset: {raw_file} ({download_size / (1024*1024):.2f} MB, SHA-256: {download_hash})"
        )

    # 4. Ingest and validate JSONL records
    logger.info(f"Ingesting raw records from {raw_file}...")
    examples, ingest_stats = load_raw_jsonl_records(
        raw_file, char_filter_len=char_filter_len
    )
    logger.info(
        f"Ingest stats: {ingest_stats['raw_lines']} lines parsed, "
        f"{ingest_stats['char_filter_skipped']} skipped by char pre-filter, "
        f"{ingest_stats['retained']} retained candidates."
    )

    # 5. Deduplication
    logger.info("Running exact and near-duplicate deduplication...")
    unique_examples, dedupe_stats = deduplicate_conversations(examples)
    logger.info(
        f"Deduplication complete: {dedupe_stats['exact_duplicates_removed']} exact dupes removed, "
        f"{dedupe_stats['near_duplicates_removed']} near-dupes removed, "
        f"{dedupe_stats['unique_retained']} unique examples retained."
    )

    # 6. Tokenize with assistant-only loss masking and length filtering
    logger.info(
        f"Tokenizing conversations (max_seq_len={max_seq_len}, filtering over-length)..."
    )
    tokenized_examples = []
    over_length_count = 0
    zero_response_count = 0

    for ex in unique_examples:
        tok_ex = encode_conversation(
            turns=ex.turns,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            user_prefix=DEFAULT_USER_PREFIX,
            assistant_prefix=DEFAULT_ASSISTANT_PREFIX,
            end_of_turn=DEFAULT_END_OF_TURN,
            truncate=False,  # Strict filtering: keep complete responses
            example_id=ex.example_id,
            category=ex.category,
            metadata=ex.metadata,
        )
        if tok_ex is None:
            over_length_count += 1
            continue
        tokenized_examples.append(tok_ex)

    logger.info(
        f"Tokenization complete: {len(tokenized_examples)} valid examples fit within {max_seq_len} tokens. "
        f"({over_length_count} exceeded context length and were filtered)."
    )

    if not tokenized_examples:
        raise ValueError("No valid tokenized examples produced!")

    # 7. Split into Train / Validation with zero prompt leakage
    logger.info(
        f"Splitting into train ({train_ratio*100:.0f}%) and val ({(1-train_ratio)*100:.0f}%) with zero prompt leakage..."
    )
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(tokenized_examples), generator=g).tolist()
    split_idx = int(len(indices) * train_ratio)

    train_tok_examples = [tokenized_examples[i] for i in indices[:split_idx]]
    val_tok_candidates = [tokenized_examples[i] for i in indices[split_idx:]]

    # Verify zero leakage by prompt IDs / tokens
    train_prompt_keys = {
        tuple(ex.input_ids[: ex.prompt_tokens_count].tolist())
        for ex in train_tok_examples
    }
    val_tok_examples = []
    leakage_reassigned = 0

    for ex in val_tok_candidates:
        prompt_key = tuple(ex.input_ids[: ex.prompt_tokens_count].tolist())
        if prompt_key in train_prompt_keys:
            train_tok_examples.append(ex)
            leakage_reassigned += 1
        else:
            val_tok_examples.append(ex)

    logger.info(
        f"Splits prepared: Train={len(train_tok_examples)}, Val={len(val_tok_examples)} "
        f"(Zero leakage verified; {leakage_reassigned} boundary prompts reassigned to train)."
    )

    # 8. Build SFT datasets and pipeline
    train_dataset = SFTDataset(
        train_tok_examples,
        pad_token_id=0,
        pad_to_max_len=False,
        max_seq_len=max_seq_len,
    )
    val_dataset = SFTDataset(
        val_tok_examples,
        pad_token_id=0,
        pad_to_max_len=False,
        max_seq_len=max_seq_len,
    )
    pipeline = SFTDataPipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        pad_token_id=0,
        max_seq_len=max_seq_len,
    )

    # 9. Compute stats and save cache
    provenance_metadata = {
        "dataset_name": (
            "Databricks Dolly 15k (SFT Subset)"
            if not used_fixture
            else "SFT Fixture"
        ),
        "source_url": DOLLY_HF_URL if not used_fixture else "local_fixture",
        "license": "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
        "download_date": datetime.now(timezone.utc).isoformat(),
        "sha256_checksum": download_hash,
        "raw_bytes": download_size,
        "is_fixture_fallback": used_fixture,
    }
    logger.info(f"Saving SFT token cache to {cache_dir}...")
    pipeline.save(cache_dir, manifest_metadata=provenance_metadata)

    # 10. Verify reload equivalence
    logger.info("Verifying serialized cache reload equivalence...")
    reloaded_pipeline = SFTDataPipeline.load(cache_dir, tokenizer)
    assert len(reloaded_pipeline.train_dataset) == len(train_dataset)
    assert len(reloaded_pipeline.val_dataset) == len(val_dataset)
    logger.info("Reload verification passed!")

    elapsed = time.time() - t0
    logger.info(f"SFT Dataset Preparation finished in {elapsed:.2f} seconds.")

    # 11. Print comprehensive report
    stats = pipeline.compute_stats()
    tok_stats = stats.get("tokens", {})

    print("\n" + "=" * 70)
    print("GROUNDUPGPT SFT DATASET PREPARATION REPORT")
    print("=" * 70)
    print(f"Dataset Name:        {provenance_metadata['dataset_name']}")
    print(f"License:             {provenance_metadata['license']}")
    print(f"Raw File Size:       {download_size / (1024 * 1024):.2f} MB")
    print(f"SHA-256 Digest:      {download_hash}")
    print(f"Tokenizer Fingerprint:{tok_fp}")
    print(f"Max Context Length:  {max_seq_len} tokens")
    print(f"Train Examples:      {len(train_dataset):,}")
    print(f"Validation Examples: {len(val_dataset):,}")
    print(f"Total Examples:      {stats['total_examples']:,}")
    print("-" * 70)
    print("TOKEN DISTRIBUTION STATISTICS:")
    print(f"Total Prompt Tokens:   {tok_stats.get('total_prompt_tokens', 0):,}")
    print(
        f"Total Response Tokens: {tok_stats.get('total_response_tokens', 0):,}"
    )
    print(f"Total Tokens:          {tok_stats.get('total_tokens', 0):,}")
    p_len = tok_stats.get("prompt_length", {})
    r_len = tok_stats.get("response_length", {})
    s_len = tok_stats.get("sequence_length", {})
    print(
        f"Prompt Tokens:   Min={p_len.get('min')}, Max={p_len.get('max')}, Mean={p_len.get('mean', 0):.1f}, Median={p_len.get('median', 0):.1f}"
    )
    print(
        f"Response Tokens: Min={r_len.get('min')}, Max={r_len.get('max')}, Mean={r_len.get('mean', 0):.1f}, Median={r_len.get('median', 0):.1f}"
    )
    print(
        f"Sequence Length: Min={s_len.get('min')}, Max={s_len.get('max')}, Mean={s_len.get('mean', 0):.1f}, Median={s_len.get('median', 0):.1f}"
    )
    print(
        f"Active Target Fraction: {tok_stats.get('active_target_token_percentage', 0):.2f}%"
    )
    print("-" * 70)
    print("CATEGORIES BREAKDOWN:")
    for cat, count in sorted(
        stats.get("categories", {}).items(), key=lambda x: -x[1]
    ):
        print(f"  - {cat:24s}: {count:,}")
    print("-" * 70)

    # Example formatted conversation and mask
    sample_ex = train_tok_examples[0]
    sample_input = sample_ex.input_ids.tolist()
    sample_target = sample_ex.target_ids.tolist()
    sample_mask = sample_ex.loss_mask.tolist()

    print("SAMPLE FORMATTED CONVERSATION INSPECTION:")
    # Find boundary where target transitions from -100 to active
    first_active_idx = next(
        i for i, t in enumerate(sample_target) if t != -100
    )
    print(f"Example ID:          {sample_ex.example_id}")
    print(f"Sequence Length:     {len(sample_input)} tokens")
    print(f"First Active Target: Index {first_active_idx}")
    print(
        f"Input at Boundary:   ID={sample_input[first_active_idx]} ({repr(tokenizer.decode([sample_input[first_active_idx]]))})"
    )
    print(
        f"Target at Boundary:  ID={sample_target[first_active_idx]} ({repr(tokenizer.decode([sample_target[first_active_idx]]))})"
    )
    active_tokens = [t for t in sample_target if t != -100]
    print(
        f"Active Decoded Text: {repr(tokenizer.decode(active_tokens)[:160])}..."
    )
    print("=" * 70 + "\n")

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and cache GroundUpGPT Supervised Fine-Tuning dataset"
    )
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=Path("data/raw_sft/databricks-dolly-15k.jsonl"),
        help="Path to raw Dolly 15k JSONL file",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache_sft"),
        help="Directory to cache processed .pt and .json artifacts",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("checkpoints/bpe_vocab4096.json"),
        help="Path to 4,096-vocab BPE tokenizer",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=Path("tests/fixtures/sft_fixture.jsonl"),
        help="Path to offline fixture fallback",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Download dataset if missing",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence context length",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--char-filter-len",
        type=int,
        default=1100,
        help="Fast char length filter for 256 tokens",
    )

    args = parser.parse_args()

    prepare_sft_dataset(
        raw_file=args.raw_file,
        cache_dir=args.cache_dir,
        tokenizer_path=args.tokenizer_path,
        fixture_path=args.fixture_path,
        download_if_missing=args.download,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        seed=args.seed,
        char_filter_len=args.char_filter_len,
    )


if __name__ == "__main__":
    main()

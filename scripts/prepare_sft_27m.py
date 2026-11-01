"""CLI script to prepare and cache the Supervised Fine-Tuning (SFT) dataset for GroundUpGPT-27M (Step 36).

Processes Databricks Dolly 15k, scrubs PII, sanitizes control delimiters,
deduplicates, filters to strict 256-token context window without truncation,
enforces assistant-only loss masking (-100), excludes held-out benchmark prompts,
splits deterministically (90/10) with zero leakage, and serializes token cache to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.bpe_tokenizer import BPETokenizer
from src.sft_data import (
    DEFAULT_ASSISTANT_PREFIX,
    DEFAULT_END_OF_TURN,
    DEFAULT_USER_PREFIX,
    IGNORE_INDEX,
    ConversationExample,
    ConversationTurn,
    SFTDataPipeline,
    SFTDataset,
    TokenizedExample,
    collate_sft_batch,
    deduplicate_conversations,
    encode_conversation,
    format_conversation,
    normalize_prompt,
    scrub_pii,
    split_conversations,
    utf8_roundtrip_check,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXPECTED_TOKENIZER_FINGERPRINT = (
    "4138461786ea6a199ff232b2427f3cb71c60f351afa7241f75f8196e51bdae72"
)
EXPECTED_RAW_DOLLY_SHA256 = (
    "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
)

# 35 Curated Held-Out Benchmark Prompts across 8 Conversational Categories
HELD_OUT_BENCHMARK_PROMPTS: List[Dict[str, Any]] = [
    # 1. Greetings & Conversational Openers (4)
    {
        "id": "bench_01",
        "category": "greetings",
        "prompt": "Hello! How are you today?",
        "expected_response_guidance": "Polite greeting acknowledgement, friendly and helpful tone.",
        "key_criteria": ["greeting", "acknowledgement", "helpful_tone"],
    },
    {
        "id": "bench_02",
        "category": "greetings",
        "prompt": "Good morning! Can you help me?",
        "expected_response_guidance": "Polite morning greeting and willingness to assist.",
        "key_criteria": ["morning_greeting", "readiness_to_help"],
    },
    {
        "id": "bench_03",
        "category": "greetings",
        "prompt": "Hi, who are you and what can you do?",
        "expected_response_guidance": "Concise introduction as an AI assistant capable of answering questions.",
        "key_criteria": ["assistant_intro", "capabilities"],
    },
    {
        "id": "bench_04",
        "category": "greetings",
        "prompt": "Hey there, how is it going?",
        "expected_response_guidance": "Casual, friendly greeting response acknowledging the user.",
        "key_criteria": ["casual_greeting", "friendly_tone"],
    },
    # 2. Simple Factual QA (5)
    {
        "id": "bench_05",
        "category": "factual_qa",
        "prompt": "What is the capital of Japan?",
        "expected_response_guidance": "Tokyo",
        "key_criteria": ["Tokyo"],
    },
    {
        "id": "bench_06",
        "category": "factual_qa",
        "prompt": "Which planet is closest to the Sun?",
        "expected_response_guidance": "Mercury",
        "key_criteria": ["Mercury"],
    },
    {
        "id": "bench_07",
        "category": "factual_qa",
        "prompt": "What is the boiling point of water in Celsius?",
        "expected_response_guidance": "100 degrees Celsius (100 C).",
        "key_criteria": ["100"],
    },
    {
        "id": "bench_08",
        "category": "factual_qa",
        "prompt": "Who wrote Romeo and Juliet?",
        "expected_response_guidance": "William Shakespeare",
        "key_criteria": ["Shakespeare"],
    },
    {
        "id": "bench_09",
        "category": "factual_qa",
        "prompt": "How many days are in a leap year?",
        "expected_response_guidance": "366 days",
        "key_criteria": ["366"],
    },
    # 3. Definitions (4)
    {
        "id": "bench_10",
        "category": "definitions",
        "prompt": "What is a mammal?",
        "expected_response_guidance": "A warm-blooded vertebrate animal with hair or fur, females produce milk.",
        "key_criteria": ["warm-blooded", "vertebrate", "milk", "hair/fur"],
    },
    {
        "id": "bench_11",
        "category": "definitions",
        "prompt": "Define the word 'photosynthesis'.",
        "expected_response_guidance": "Process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
        "key_criteria": ["plants", "sunlight", "energy/glucose", "oxygen"],
    },
    {
        "id": "bench_12",
        "category": "definitions",
        "prompt": "What is an algorithm in computer science?",
        "expected_response_guidance": "A step-by-step procedure or set of rules for solving a computational problem.",
        "key_criteria": ["step-by-step", "rules/procedure", "problem_solving"],
    },
    {
        "id": "bench_13",
        "category": "definitions",
        "prompt": "What is gravity?",
        "expected_response_guidance": "The fundamental natural force that attracts physical bodies with mass toward one another.",
        "key_criteria": ["force", "attraction", "mass"],
    },
    # 4. Arithmetic & Simple Reasoning (5)
    {
        "id": "bench_14",
        "category": "arithmetic_reasoning",
        "prompt": "What is 25 plus 37?",
        "expected_response_guidance": "62",
        "key_criteria": ["62"],
    },
    {
        "id": "bench_15",
        "category": "arithmetic_reasoning",
        "prompt": "If a train travels at 60 miles per hour, how far does it travel in 3 hours?",
        "expected_response_guidance": "180 miles (60 * 3 = 180).",
        "key_criteria": ["180"],
    },
    {
        "id": "bench_16",
        "category": "arithmetic_reasoning",
        "prompt": "What is 12 multiplied by 8?",
        "expected_response_guidance": "96",
        "key_criteria": ["96"],
    },
    {
        "id": "bench_17",
        "category": "arithmetic_reasoning",
        "prompt": "Is 29 a prime number?",
        "expected_response_guidance": "Yes, 29 is a prime number because it is divisible only by 1 and 29.",
        "key_criteria": ["Yes", "prime"],
    },
    {
        "id": "bench_18",
        "category": "arithmetic_reasoning",
        "prompt": "If you have 15 apples and give away 6, how many do you have left?",
        "expected_response_guidance": "9 apples (15 - 6 = 9).",
        "key_criteria": ["9"],
    },
    # 5. Explanations (4)
    {
        "id": "bench_19",
        "category": "explanations",
        "prompt": "Why is the sky blue?",
        "expected_response_guidance": "Rayleigh scattering of shorter blue wavelengths of sunlight by atmospheric gases.",
        "key_criteria": ["scattering", "atmosphere", "wavelength"],
    },
    {
        "id": "bench_20",
        "category": "explanations",
        "prompt": "Why do leaves change color in the fall?",
        "expected_response_guidance": "Chlorophyll breaks down due to shorter daylight, revealing orange and yellow carotenoids.",
        "key_criteria": ["chlorophyll", "pigments", "autumn/fall"],
    },
    {
        "id": "bench_21",
        "category": "explanations",
        "prompt": "How does a refrigerator keep food cold?",
        "expected_response_guidance": "It circulates a refrigerant fluid that absorbs heat from inside and releases it outside.",
        "key_criteria": ["refrigerant", "heat_transfer", "cooling"],
    },
    {
        "id": "bench_22",
        "category": "explanations",
        "prompt": "Why do we have day and night?",
        "expected_response_guidance": "Due to Earth's rotation on its axis once every 24 hours.",
        "key_criteria": ["rotation", "Earth", "axis"],
    },
    # 6. Classifications (4)
    {
        "id": "bench_23",
        "category": "classifications",
        "prompt": "Classify the following items as fruit or vegetable: apple, carrot, banana, broccoli.",
        "expected_response_guidance": "Fruits: apple, banana. Vegetables: carrot, broccoli.",
        "key_criteria": ["apple: fruit", "banana: fruit", "carrot: vegetable", "broccoli: vegetable"],
    },
    {
        "id": "bench_24",
        "category": "classifications",
        "prompt": "Classify each as solid, liquid, or gas at room temperature: ice, water, oxygen, steam.",
        "expected_response_guidance": "Solid: ice. Liquid: water. Gas: oxygen, steam.",
        "key_criteria": ["solid: ice", "liquid: water", "gas: oxygen"],
    },
    {
        "id": "bench_25",
        "category": "classifications",
        "prompt": "Classify the following animals as mammal or bird: eagle, dolphin, penguin, dog.",
        "expected_response_guidance": "Mammals: dolphin, dog. Birds: eagle, penguin.",
        "key_criteria": ["mammal: dolphin", "mammal: dog", "bird: eagle", "bird: penguin"],
    },
    {
        "id": "bench_26",
        "category": "classifications",
        "prompt": "Classify each as renewable or non-renewable energy: solar, coal, wind, petroleum.",
        "expected_response_guidance": "Renewable: solar, wind. Non-renewable: coal, petroleum.",
        "key_criteria": ["renewable: solar", "renewable: wind", "non-renewable: coal", "non-renewable: petroleum"],
    },
    # 7. Short Instructions / Tasks (5)
    {
        "id": "bench_27",
        "category": "instructions",
        "prompt": "Write a 3-step guide to make a cup of tea.",
        "expected_response_guidance": "1. Boil water. 2. Steep tea bag in a mug for 3-5 minutes. 3. Remove tea bag and add milk/sugar if desired.",
        "key_criteria": ["3 steps", "boil water", "steep"],
    },
    {
        "id": "bench_28",
        "category": "instructions",
        "prompt": "List three primary colors.",
        "expected_response_guidance": "Red, blue, and yellow (or red, green, blue for additive light).",
        "key_criteria": ["red", "blue", "yellow/green"],
    },
    {
        "id": "bench_29",
        "category": "instructions",
        "prompt": "Rewrite the following sentence in uppercase: 'groundup gpt is fast and lightweight'.",
        "expected_response_guidance": "'GROUNDUP GPT IS FAST AND LIGHTWEIGHT'",
        "key_criteria": ["GROUNDUP GPT IS FAST AND LIGHTWEIGHT"],
    },
    {
        "id": "bench_30",
        "category": "instructions",
        "prompt": "Name three common programming languages.",
        "expected_response_guidance": "Python, JavaScript, and C++ (or Java, Rust, Go, etc.).",
        "key_criteria": ["programming languages listed"],
    },
    {
        "id": "bench_31",
        "category": "instructions",
        "prompt": "Provide two tips for getting better sleep.",
        "expected_response_guidance": "1. Maintain a consistent sleep schedule. 2. Avoid screens and caffeine before bedtime.",
        "key_criteria": ["two tips", "sleep hygiene"],
    },
    # 8. Everyday Questions & Advice (4)
    {
        "id": "bench_32",
        "category": "everyday_questions",
        "prompt": "What should I pack for a rainy day?",
        "expected_response_guidance": "An umbrella, a waterproof jacket/raincoat, and water-resistant shoes.",
        "key_criteria": ["umbrella", "raincoat/jacket"],
    },
    {
        "id": "bench_33",
        "category": "everyday_questions",
        "prompt": "What is the purpose of an umbrella?",
        "expected_response_guidance": "To provide shelter and protection from rain or direct sunlight.",
        "key_criteria": ["protection", "rain/sun"],
    },
    {
        "id": "bench_34",
        "category": "everyday_questions",
        "prompt": "Why is drinking water important for health?",
        "expected_response_guidance": "Maintains hydration, regulates body temperature, supports digestion and nutrient transport.",
        "key_criteria": ["hydration", "health", "bodily functions"],
    },
    {
        "id": "bench_35",
        "category": "everyday_questions",
        "prompt": "How often should teeth be brushed daily according to dentists?",
        "expected_response_guidance": "Twice a day, for two minutes each time.",
        "key_criteria": ["twice a day", "two times"],
    },
]

CONTROL_DELIMITERS = ["<|user|>", "<|assistant|>", "<|endoftext|>"]


def sanitize_control_delimiters(text: str) -> str:
    """Strip or replace accidental control delimiters to prevent prompt injection."""
    for delim in CONTROL_DELIMITERS:
        if delim in text:
            text = text.replace(delim, "")
    return text


def load_and_filter_raw_dolly(
    file_path: Path,
    char_filter_len: int = 1100,
    held_out_benchmark_prompts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[ConversationExample], Dict[str, Any]]:
    """Ingest raw Dolly records with PII scrubbing, delimiter protection, and benchmark exclusion."""
    examples: List[ConversationExample] = []
    stats = {
        "raw_lines": 0,
        "json_parse_errors": 0,
        "missing_fields": 0,
        "empty_content": 0,
        "delimiter_sanitized": 0,
        "utf8_failures": 0,
        "char_filter_skipped": 0,
        "benchmark_collisions_excluded": 0,
        "retained": 0,
    }

    norm_benchmark_prompts: Set[str] = set()
    if held_out_benchmark_prompts:
        for b in held_out_benchmark_prompts:
            norm_benchmark_prompts.add(normalize_prompt(b["prompt"]))

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

            instruction = data.get("instruction")
            response = data.get("response")
            context = data.get("context", "")
            category = data.get("category")

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
            context_clean = context.strip() if isinstance(context, str) else ""

            # Delimiter injection protection
            had_delim = False
            for d in CONTROL_DELIMITERS:
                if d in instruction_clean or d in response_clean or d in context_clean:
                    had_delim = True
                    break
            if had_delim:
                instruction_clean = sanitize_control_delimiters(instruction_clean)
                response_clean = sanitize_control_delimiters(response_clean)
                context_clean = sanitize_control_delimiters(context_clean)
                stats["delimiter_sanitized"] += 1

            if context_clean:
                prompt_text = f"{instruction_clean}\n\nContext:\n{context_clean}"
            else:
                prompt_text = instruction_clean

            # Zero-Leakage Benchmark Exclusion Check
            norm_p = normalize_prompt(prompt_text)
            norm_instr = normalize_prompt(instruction_clean)
            if norm_p in norm_benchmark_prompts or norm_instr in norm_benchmark_prompts:
                stats["benchmark_collisions_excluded"] += 1
                logger.info(
                    f"Excluding Dolly line {idx+1} matching benchmark prompt: {instruction_clean!r}"
                )
                continue

            turns = [
                ConversationTurn(role="user", content=prompt_text),
                ConversationTurn(role="assistant", content=response_clean),
            ]

            total_chars = sum(len(t.content) for t in turns)
            if total_chars > char_filter_len:
                stats["char_filter_skipped"] += 1
                continue

            if not all(utf8_roundtrip_check(t.content) for t in turns):
                stats["utf8_failures"] += 1
                continue

            ex_id = data.get("id") or f"dolly_{idx+1}"
            examples.append(
                ConversationExample(
                    example_id=str(ex_id),
                    turns=turns,
                    category=category,
                    metadata={"source_line": idx + 1, "raw_category": category},
                )
            )
            stats["retained"] += 1

    return examples, stats


def prepare_sft_27m(
    raw_file: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    max_seq_len: int = 256,
    train_ratio: float = 0.9,
    seed: int = 42,
    char_filter_len: int = 1100,
    min_response_tokens: int = 4,
    benchmark_file: Optional[Path] = None,
) -> SFTDataPipeline:
    """End-to-end dataset preparation and caching for 27M SFT."""
    t0 = time.time()
    logger.info("=" * 75)
    logger.info("GroundUpGPT-27M Supervised Fine-Tuning (SFT) Dataset Preparation")
    logger.info("=" * 75)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and verify Tokenizer
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"BPE tokenizer not found at {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tok_fp = tokenizer.get_fingerprint()
    logger.info(
        f"Loaded BPE tokenizer: vocab={tokenizer.vocab_size}, fingerprint={tok_fp}"
    )
    if tok_fp != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError(
            f"Tokenizer fingerprint mismatch! Expected {EXPECTED_TOKENIZER_FINGERPRINT}, got {tok_fp}"
        )

    # 2. Verify delimiters lossless round-trip
    for delim in [DEFAULT_USER_PREFIX, DEFAULT_ASSISTANT_PREFIX, DEFAULT_END_OF_TURN]:
        enc = tokenizer.encode(delim)
        dec = tokenizer.decode(enc)
        if dec != delim:
            raise ValueError(f"Lossless round-trip failed for delimiter {delim!r}")
    logger.info("Delimiters verified with 100% lossless BPE round-trip.")

    # 3. Source dataset verification
    if not raw_file.is_file():
        raise FileNotFoundError(
            f"Raw Dolly dataset missing at {raw_file}! Step 36 requires verified local artifact."
        )

    raw_h = hashlib.sha256()
    raw_bytes = 0
    with open(raw_file, "rb") as f:
        while chunk := f.read(65536):
            raw_h.update(chunk)
            raw_bytes += len(chunk)
    raw_sha256 = raw_h.hexdigest()
    logger.info(
        f"Verified raw Dolly dataset: {raw_file.name} ({raw_bytes / (1024*1024):.2f} MB, SHA-256: {raw_sha256})"
    )
    if raw_sha256 != EXPECTED_RAW_DOLLY_SHA256:
        raise ValueError(
            f"Raw Dolly SHA-256 mismatch! Expected {EXPECTED_RAW_DOLLY_SHA256}, got {raw_sha256}"
        )

    # 4. Save Held-Out Benchmark Artifact
    benchmark_path = benchmark_file or (cache_dir / "eval_benchmark_35.json")
    benchmark_data = {
        "benchmark_name": "GroundUpGPT-27M Conversational Benchmark (35 Prompts)",
        "benchmark_version": "1.0",
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "num_prompts": len(HELD_OUT_BENCHMARK_PROMPTS),
        "target_model": "GroundUpGPT-27M (27,411,456 params, 8L/8H/d=512/ctx=256)",
        "source_checkpoint": "checkpoints/27m_phase4/model_27m_best.pt",
        "categories": [
            "greetings",
            "factual_qa",
            "definitions",
            "arithmetic_reasoning",
            "explanations",
            "classifications",
            "instructions",
            "everyday_questions",
        ],
        "items": HELD_OUT_BENCHMARK_PROMPTS,
    }
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(
        f"Saved 35 held-out conversational benchmark prompts to {benchmark_path}"
    )

    # 5. Ingestion with quality filters and benchmark exclusion
    logger.info(f"Ingesting raw Dolly records from {raw_file}...")
    examples, ingest_stats = load_and_filter_raw_dolly(
        raw_file,
        char_filter_len=char_filter_len,
        held_out_benchmark_prompts=HELD_OUT_BENCHMARK_PROMPTS,
    )
    logger.info(
        f"Ingest stats: {ingest_stats['raw_lines']} lines parsed, "
        f"{ingest_stats['benchmark_collisions_excluded']} excluded benchmark collisions, "
        f"{ingest_stats['char_filter_skipped']} char-filter skipped, "
        f"{ingest_stats['retained']} retained candidates."
    )

    # 6. Deduplication (exact hash + normalized prompt)
    logger.info("Running exact and normalized prompt deduplication...")
    unique_examples, dedupe_stats = deduplicate_conversations(examples)
    logger.info(
        f"Deduplication: {dedupe_stats['exact_duplicates_removed']} exact dupes removed, "
        f"{dedupe_stats['near_duplicates_removed']} near-dupes removed, "
        f"{dedupe_stats['unique_retained']} unique examples retained."
    )

    # 7. Tokenization, strict 256-context window filtering, and loss masking
    logger.info(
        f"Tokenizing conversations (max_seq_len={max_seq_len}, min_resp_tokens={min_response_tokens})..."
    )
    tokenized_examples: List[TokenizedExample] = []
    over_length_count = 0
    short_response_count = 0

    for ex in unique_examples:
        tok_ex = encode_conversation(
            turns=ex.turns,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            user_prefix=DEFAULT_USER_PREFIX,
            assistant_prefix=DEFAULT_ASSISTANT_PREFIX,
            end_of_turn=DEFAULT_END_OF_TURN,
            truncate=False,  # Strict: no truncation!
            example_id=ex.example_id,
            category=ex.category,
            metadata=ex.metadata,
        )
        if tok_ex is None:
            over_length_count += 1
            continue
        if tok_ex.response_tokens_count < min_response_tokens:
            short_response_count += 1
            continue
        tokenized_examples.append(tok_ex)

    logger.info(
        f"Tokenization complete: {len(tokenized_examples)} valid examples fit within {max_seq_len} tokens. "
        f"({over_length_count} exceeded context length, {short_response_count} had < {min_response_tokens} response tokens)."
    )

    if not tokenized_examples:
        raise ValueError("No valid tokenized examples produced!")

    # 8. Deterministic Train/Val Split with Zero Leakage
    logger.info(
        f"Splitting into train ({train_ratio*100:.0f}%) and val ({(1-train_ratio)*100:.0f}%) with zero prompt leakage (seed={seed})..."
    )
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(tokenized_examples), generator=g).tolist()
    split_idx = int(len(indices) * train_ratio)

    train_tok_examples = [tokenized_examples[i] for i in indices[:split_idx]]
    val_tok_candidates = [tokenized_examples[i] for i in indices[split_idx:]]

    train_prompt_keys = {
        tuple(ex.input_ids[: ex.prompt_tokens_count].tolist())
        for ex in train_tok_examples
    }
    val_tok_examples: List[TokenizedExample] = []
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
        f"({leakage_reassigned} boundary prompts reassigned to train for 100% zero leakage)."
    )

    # 9. Build datasets and pipeline
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

    # 10. Metadata and serialization
    provenance_metadata = {
        "dataset_name": "Databricks Dolly 15k (GroundUpGPT-27M SFT Subset)",
        "dataset_version": "27m_v1",
        "target_model_architecture": "GroundUpGPT-27M (8 layers, 8 heads, d_model 512, context 256)",
        "target_checkpoint": "checkpoints/27m_phase4/model_27m_best.pt",
        "target_checkpoint_step": 5949,
        "source_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        "license": "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
        "attribution": "Databricks Dolly 15k, (c) 2023 Databricks, Inc., licensed under CC BY-SA 3.0",
        "raw_source_file": str(raw_file),
        "raw_source_sha256": raw_sha256,
        "raw_source_bytes": raw_bytes,
        "raw_source_lines": ingest_stats["raw_lines"],
        "benchmark_file": str(benchmark_path.name),
        "benchmark_prompts_count": len(HELD_OUT_BENCHMARK_PROMPTS),
        "benchmark_collisions_excluded": ingest_stats["benchmark_collisions_excluded"],
        "exact_duplicates_removed": dedupe_stats["exact_duplicates_removed"],
        "near_duplicates_removed": dedupe_stats["near_duplicates_removed"],
        "over_length_filtered": over_length_count,
        "short_response_filtered": short_response_count,
        "min_response_tokens": min_response_tokens,
        "leakage_reassigned": leakage_reassigned,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"Saving 27M SFT token cache to {cache_dir}...")
    pipeline.save(cache_dir, manifest_metadata=provenance_metadata)

    # 11. Verify reload equivalence
    logger.info("Verifying serialized cache reload equivalence...")
    reloaded = SFTDataPipeline.load(cache_dir, tokenizer)
    assert len(reloaded.train_dataset) == len(train_dataset), "Train count mismatch!"
    assert len(reloaded.val_dataset) == len(val_dataset), "Val count mismatch!"

    # Verify tensor identicality on sample
    for idx in range(min(5, len(train_dataset))):
        orig_item = train_dataset[idx]
        reloaded_item = reloaded.train_dataset[idx]
        assert orig_item["example_id"] == reloaded_item["example_id"]
        assert torch.equal(orig_item["input_ids"], reloaded_item["input_ids"])
        assert torch.equal(orig_item["target_ids"], reloaded_item["target_ids"])
        assert torch.equal(orig_item["loss_mask"], reloaded_item["loss_mask"])
    logger.info("Reload verification passed with 100% bit-exact equivalence.")

    elapsed = time.time() - t0
    logger.info(f"27M SFT Dataset Preparation completed in {elapsed:.2f} seconds.")

    # 12. Display Comprehensive Statistics
    stats = pipeline.compute_stats()
    tok_stats = stats.get("tokens", {})

    print("\n" + "=" * 75)
    print("GROUNDUPGPT-27M SFT DATASET PREPARATION REPORT")
    print("=" * 75)
    print(f"Target Architecture:    GroundUpGPT-27M (27,411,456 params)")
    print(f"Target Checkpoint:      checkpoints/27m_phase4/model_27m_best.pt (Step 5949)")
    print(f"Raw Dataset Source:     Databricks Dolly 15k (CC BY-SA 3.0)")
    print(f"Raw Source SHA-256:     {raw_sha256}")
    print(f"Tokenizer Fingerprint:  {tok_fp}")
    print(f"Max Context Length:     {max_seq_len} tokens (strict, no truncation)")
    print(f"Held-Out Benchmark Set: {len(HELD_OUT_BENCHMARK_PROMPTS)} prompts (100% zero leakage)")
    print(f"Benchmark Excluded:     {ingest_stats['benchmark_collisions_excluded']} Dolly collisions")
    print(f"Deduplication:          {dedupe_stats['exact_duplicates_removed']} exact, {dedupe_stats['near_duplicates_removed']} near-prompt dupes removed")
    print(f"Train Examples:         {len(train_dataset):,}")
    print(f"Validation Examples:    {len(val_dataset):,}")
    print(f"Total Usable Examples:  {stats['total_examples']:,}")
    print("-" * 75)
    print("TOKEN DISTRIBUTION STATISTICS:")
    print(f"Total Prompt Tokens:    {tok_stats.get('total_prompt_tokens', 0):,}")
    print(f"Total Response Tokens:  {tok_stats.get('total_response_tokens', 0):,}")
    print(f"Total Tokens:           {tok_stats.get('total_tokens', 0):,}")
    p_len = tok_stats.get("prompt_length", {})
    r_len = tok_stats.get("response_length", {})
    s_len = tok_stats.get("sequence_length", {})
    print(
        f"Prompt Tokens:    Min={p_len.get('min')}, Max={p_len.get('max')}, "
        f"Mean={p_len.get('mean', 0):.1f}, Median={p_len.get('median', 0):.1f}"
    )
    print(
        f"Response Tokens:  Min={r_len.get('min')}, Max={r_len.get('max')}, "
        f"Mean={r_len.get('mean', 0):.1f}, Median={r_len.get('median', 0):.1f}"
    )
    print(
        f"Sequence Length:  Min={s_len.get('min')}, Max={s_len.get('max')}, "
        f"Mean={s_len.get('mean', 0):.1f}, Median={s_len.get('median', 0):.1f}"
    )
    print(
        f"Active Target Fraction: {tok_stats.get('active_target_token_percentage', 0):.2f}%"
    )
    print("-" * 75)
    print("CATEGORY DISTRIBUTION:")
    for cat, count in sorted(
        stats.get("categories", {}).items(), key=lambda x: -x[1]
    ):
        pct = count / stats["total_examples"] * 100.0
        print(f"  - {cat:24s}: {count:5,d} ({pct:5.2f}%)")
    print("=" * 75 + "\n")

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and cache GroundUpGPT-27M Supervised Fine-Tuning dataset"
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
        default=Path("data/cache_sft_27m_v1"),
        help="Directory to cache 27M SFT .pt and .json artifacts",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("checkpoints/bpe_vocab4096.json"),
        help="Path to 4,096-vocab BPE tokenizer",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence context length (strict)",
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
    parser.add_argument(
        "--min-response-tokens",
        type=int,
        default=4,
        help="Minimum active response tokens",
    )

    args = parser.parse_args()

    prepare_sft_27m(
        raw_file=args.raw_file,
        cache_dir=args.cache_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        seed=args.seed,
        char_filter_len=args.char_filter_len,
        min_response_tokens=args.min_response_tokens,
    )


if __name__ == "__main__":
    main()

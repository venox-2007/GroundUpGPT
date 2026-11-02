"""Supervised Fine-Tuning (SFT) conversation dataset and tokenization pipeline.

This module provides data structures, formatting, PII cleaning, tokenization,
assistant-only loss masking, quality filtering, deduplication, and PyTorch dataset
utilities for conversational instruction-following in GroundUpGPT.

In Step 27, this module's SFTDataPipeline and SFTDataset will be consumed
directly by the training loop to fine-tune the 5.05M-parameter GPT model
without updating user prompt tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data import Dataset

from src.bpe_tokenizer import BPETokenizer

logger = logging.getLogger(__name__)

# Standard conversation delimiters for GroundUpGPT
DEFAULT_USER_PREFIX: str = "<|user|>\n"
DEFAULT_ASSISTANT_PREFIX: str = "<|assistant|>\n"
DEFAULT_END_OF_TURN: str = "<|endoftext|>\n"

# PyTorch CrossEntropyLoss ignore index
IGNORE_INDEX: int = -100

# Regex patterns for basic PII scrubbing
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"
)
IP_REGEX = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
)


def scrub_pii(text: str) -> str:
    """Scrub accidental private identifiers (emails, phone numbers, IP addresses).

    Args:
        text: Input string.

    Returns:
        Sanitized string with identifiable patterns replaced by placeholder tokens.
    """
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    text = PHONE_REGEX.sub("[PHONE]", text)
    text = IP_REGEX.sub("[IP_ADDRESS]", text)
    return text


def utf8_roundtrip_check(text: str) -> bool:
    """Verify that a text string survives a lossless UTF-8 round-trip encode/decode."""
    try:
        encoded = text.encode("utf-8", errors="strict")
        decoded = encoded.decode("utf-8", errors="strict")
        return decoded == text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def normalize_prompt(text: str) -> str:
    """Normalize a prompt string for collision, near-duplicate, and leakage checks.

    Lowercases, strips punctuation, and collapses consecutive whitespace.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


@dataclass
class ConversationTurn:
    """Single turn in a dialogue.

    Attributes:
        role: Role of the speaker ('user' or 'assistant').
        content: Text content of the turn.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        role_lower = self.role.strip().lower()
        if role_lower not in ("user", "assistant"):
            raise ValueError(
                f"Invalid conversation role: {self.role!r}. Must be 'user' or 'assistant'."
            )
        self.role = role_lower
        if not isinstance(self.content, str):
            raise TypeError(f"content must be a string, got {type(self.content)}")


@dataclass
class ConversationExample:
    """Full conversational example consisting of one or more turns.

    Attributes:
        example_id: Unique string identifier.
        turns: Ordered sequence of conversation turns.
        category: Optional task category (e.g., 'open_qa', 'classification').
        metadata: Optional dictionary of provenance metadata.
    """

    example_id: str
    turns: List[ConversationTurn]
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("ConversationExample must have at least one turn.")


@dataclass
class TokenizedExample:
    """Tokenized SFT example ready for causal LM with assistant-only loss masking.

    Attributes:
        example_id: Unique example identifier.
        input_ids: 1D Tensor of input token IDs (shape T).
        target_ids: 1D Tensor of causal target token IDs (shape T), with prompt
                    and padding tokens replaced by IGNORE_INDEX (-100).
        loss_mask: 1D Tensor (shape T), with 1 for active assistant targets and 0 for ignored tokens.
        prompt_tokens_count: Number of tokens belonging to prompt and prefix delimiters.
        response_tokens_count: Number of active assistant response tokens.
        total_tokens_count: Total tokens in the sequence before causal shift (input_ids + 1).
        category: Task category.
        metadata: Optional metadata.
    """

    example_id: str
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor
    prompt_tokens_count: int
    response_tokens_count: int
    total_tokens_count: int
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.input_ids.shape != self.target_ids.shape:
            raise ValueError(
                f"input_ids shape {list(self.input_ids.shape)} != target_ids shape {list(self.target_ids.shape)}"
            )
        if self.loss_mask.shape != self.target_ids.shape:
            raise ValueError(
                f"loss_mask shape {list(self.loss_mask.shape)} != target_ids shape {list(self.target_ids.shape)}"
            )

    @property
    def first_target_index(self) -> int:
        """0-based index of the first active assistant target token in target_ids (P - 1)."""
        return self.prompt_tokens_count - 1

    @property
    def prompt_targets_count(self) -> int:
        """Number of masked prompt targets (P - 1) before the first active response target."""
        return self.prompt_tokens_count - 1

    @property
    def prompt_input_ids(self) -> torch.Tensor:
        """The full prompt portion of input_ids (P tokens, including the final assistant header token)."""
        return self.input_ids[: self.prompt_tokens_count]


def format_conversation(
    turns: Sequence[ConversationTurn],
    user_prefix: str = DEFAULT_USER_PREFIX,
    assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    end_of_turn: str = DEFAULT_END_OF_TURN,
) -> str:
    """Format an ordered sequence of conversation turns into canonical string representation.

    Args:
        turns: Ordered turns.
        user_prefix: Delimiter prepended to user turns.
        assistant_prefix: Delimiter prepended to assistant turns.
        end_of_turn: Delimiter appended after assistant turns.

    Returns:
        Formatted string representation.
    """
    formatted_parts: List[str] = []
    for turn in turns:
        cleaned_content = scrub_pii(turn.content.strip())
        if turn.role == "user":
            formatted_parts.append(f"{user_prefix}{cleaned_content}\n")
        elif turn.role == "assistant":
            formatted_parts.append(f"{assistant_prefix}{cleaned_content}{end_of_turn}")
    return "".join(formatted_parts)


def encode_conversation(
    turns: Sequence[ConversationTurn],
    tokenizer: BPETokenizer,
    max_seq_len: int = 256,
    user_prefix: str = DEFAULT_USER_PREFIX,
    assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    end_of_turn: str = DEFAULT_END_OF_TURN,
    truncate: bool = False,
    example_id: str = "",
    category: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[TokenizedExample]:
    """Encode a conversation sequence into input and target tensors with assistant-only loss masking.

    Math & Alignment:
        Given token sequence tokens = [t_0, t_1, ..., t_{N-1}]:
        - Model input: x = tokens[:-1] (length N-1).
        - Causal LM target: targets = tokens[1:] (length N-1).
        - Each target[i] represents predicting token tokens[i+1] from context tokens[:i+1].
        - We assign target[i] = tokens[i+1] IF AND ONLY IF tokens[i+1] is an assistant response token.
        - Otherwise, target[i] = IGNORE_INDEX (-100).
        - The boundary token at the end of the assistant prefix (e.g. newline in '<|assistant|>\\n')
          is the last prompt token in x. The target at that position is the first response token,
          which correctly trains the model to begin generating the assistant response!
        - If after masking or truncation no active assistant targets remain, the example is rejected.

    Args:
        turns: Ordered conversation turns.
        tokenizer: Fitted BPETokenizer instance.
        max_seq_len: Maximum sequence length (block_size, e.g. 256).
        user_prefix: Delimiter prepended to user turns.
        assistant_prefix: Delimiter prepended to assistant turns.
        end_of_turn: Delimiter appended after assistant turns.
        truncate: Whether to truncate over-length examples instead of rejecting them.
        example_id: Identifier for the example.
        category: Category label.
        metadata: Provenance metadata.

    Returns:
        TokenizedExample if valid and meets all criteria, or None if rejected.
    """
    if not turns:
        return None

    # Verify at least one user turn and at least one assistant turn
    has_user = any(t.role == "user" and t.content.strip() for t in turns)
    has_assistant = any(t.role == "assistant" and t.content.strip() for t in turns)
    if not (has_user and has_assistant):
        return None

    tokens: List[int] = []
    # Boolean mask aligned 1:1 with tokens: True if token belongs to assistant response
    is_response_token: List[bool] = []

    for turn in turns:
        content = scrub_pii(turn.content.strip())
        if not content:
            return None

        if turn.role == "user":
            user_text = f"{user_prefix}{content}\n"
            turn_tokens = tokenizer.encode(user_text)
            tokens.extend(turn_tokens)
            is_response_token.extend([False] * len(turn_tokens))
        elif turn.role == "assistant":
            # Assistant header delimiter (not part of response to learn)
            prefix_tokens = tokenizer.encode(assistant_prefix)
            tokens.extend(prefix_tokens)
            is_response_token.extend([False] * len(prefix_tokens))

            # Assistant response content + end-of-turn delimiter
            response_text = f"{content}{end_of_turn}"
            resp_tokens = tokenizer.encode(response_text)
            tokens.extend(resp_tokens)
            is_response_token.extend([True] * len(resp_tokens))

    total_tokens_count = len(tokens)
    if total_tokens_count < 2:
        return None

    # Validate token ID bounds
    vocab_size = tokenizer.vocab_size
    for tok_id in tokens:
        if not (0 <= tok_id < vocab_size):
            raise ValueError(
                f"Token ID {tok_id} out of bounds for vocab size {vocab_size}."
            )

    # Maximum token sequence length for causal shift to produce input length <= max_seq_len
    # tokens sequence of length N produces input_ids of length N-1.
    # Therefore, max tokens length is max_seq_len + 1.
    max_tokens_len = max_seq_len + 1
    if len(tokens) > max_tokens_len:
        if not truncate:
            return None
        tokens = tokens[:max_tokens_len]
        is_response_token = is_response_token[:max_tokens_len]

    # Causal LM shift
    input_ids_list = tokens[:-1]
    candidate_targets = tokens[1:]
    target_mask_flags = is_response_token[1:]

    # Assistant-only loss masking: set non-response positions to IGNORE_INDEX (-100)
    target_ids_list = [
        tok_id if flag else IGNORE_INDEX
        for tok_id, flag in zip(candidate_targets, target_mask_flags)
    ]
    loss_mask_list = [1 if flag else 0 for flag in target_mask_flags]

    active_response_count = sum(loss_mask_list)
    # Reject if no active assistant targets remain (e.g. response was fully truncated)
    if active_response_count == 0:
        return None

    # Prompt tokens count (P) spans all tokens prior to the assistant response in tokens (len(tokens) - R).
    # Note: len(input_ids_list) is P + R - 1, so len(input_ids_list) - active_response_count equals P - 1
    # (the index of the first active target, first_target_index), NOT the prompt token count.
    prompt_tokens_count = len(tokens) - active_response_count

    return TokenizedExample(
        example_id=example_id,
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        target_ids=torch.tensor(target_ids_list, dtype=torch.long),
        loss_mask=torch.tensor(loss_mask_list, dtype=torch.long),
        prompt_tokens_count=prompt_tokens_count,
        response_tokens_count=active_response_count,
        total_tokens_count=len(tokens),
        category=category,
        metadata=metadata,
    )


def deduplicate_conversations(
    examples: Sequence[ConversationExample],
) -> Tuple[List[ConversationExample], Dict[str, int]]:
    """Deduplicate conversation examples by full-text SHA-256 and normalized prompt.

    Args:
        examples: List of conversation examples.

    Returns:
        Tuple[List[ConversationExample], Dict[str, int]]:
            - Filtered list of unique conversation examples.
            - Metrics dictionary of duplicate removal counts.
    """
    seen_hashes: Set[str] = set()
    seen_prompts: Set[str] = set()
    unique_examples: List[ConversationExample] = []

    exact_duplicates = 0
    near_duplicates = 0

    for ex in examples:
        # Full text hash for exact duplicate detection
        formatted = format_conversation(ex.turns)
        full_hash = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        if full_hash in seen_hashes:
            exact_duplicates += 1
            continue
        seen_hashes.add(full_hash)

        # Extract all user turn text to detect duplicate prompts
        user_text = " ".join(t.content for t in ex.turns if t.role == "user")
        norm_p = normalize_prompt(user_text)
        if norm_p and norm_p in seen_prompts:
            near_duplicates += 1
            continue
        if norm_p:
            seen_prompts.add(norm_p)

        unique_examples.append(ex)

    metrics = {
        "initial_count": len(examples),
        "exact_duplicates_removed": exact_duplicates,
        "near_duplicates_removed": near_duplicates,
        "unique_retained": len(unique_examples),
    }
    return unique_examples, metrics


def split_conversations(
    examples: Sequence[ConversationExample],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[ConversationExample], List[ConversationExample]]:
    """Deterministically split conversation examples into train and validation sets with zero prompt leakage.

    Args:
        examples: Input examples.
        train_ratio: Fraction for training (e.g. 0.9).
        seed: RNG seed for reproducible shuffling.

    Returns:
        Tuple of (train_examples, val_examples).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            f"train_ratio must be between 0.0 and 1.0, got {train_ratio}"
        )

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(examples), generator=g).tolist()

    shuffled = [examples[i] for i in indices]
    split_idx = int(len(shuffled) * train_ratio)

    train_candidates = shuffled[:split_idx]
    val_candidates = shuffled[split_idx:]

    # Zero Leakage Guarantee: check that no prompt in val_candidates exists in train
    train_prompts = {
        normalize_prompt(" ".join(t.content for t in ex.turns if t.role == "user"))
        for ex in train_candidates
    }

    train_final = list(train_candidates)
    val_final: List[ConversationExample] = []

    for ex in val_candidates:
        norm_p = normalize_prompt(
            " ".join(t.content for t in ex.turns if t.role == "user")
        )
        if norm_p in train_prompts:
            # Reassign colliding prompt to train to prevent leakage
            train_final.append(ex)
        else:
            val_final.append(ex)

    return train_final, val_final


class SFTDataset(Dataset):
    """PyTorch Dataset holding tokenized supervised fine-tuning examples.

    Args:
        examples: List of TokenizedExample instances.
        pad_token_id: Integer token ID used for input padding (default: 0).
        pad_to_max_len: Whether __getitem__ pads to fixed max_seq_len.
        max_seq_len: Target sequence length when padding (default: 256).
    """

    def __init__(
        self,
        examples: Sequence[TokenizedExample],
        pad_token_id: int = 0,
        pad_to_max_len: bool = False,
        max_seq_len: int = 256,
    ) -> None:
        self.examples = list(examples)
        self.pad_token_id = pad_token_id
        self.pad_to_max_len = pad_to_max_len
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        input_ids = ex.input_ids
        target_ids = ex.target_ids
        loss_mask = ex.loss_mask

        if self.pad_to_max_len:
            seq_len = input_ids.size(0)
            if seq_len < self.max_seq_len:
                pad_len = self.max_seq_len - seq_len
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (pad_len,), self.pad_token_id, dtype=torch.long
                        ),
                    ]
                )
                target_ids = torch.cat(
                    [
                        target_ids,
                        torch.full(
                            (pad_len,), IGNORE_INDEX, dtype=torch.long
                        ),
                    ]
                )
                loss_mask = torch.cat(
                    [loss_mask, torch.zeros((pad_len,), dtype=torch.long)]
                )

        return {
            "example_id": ex.example_id,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "prompt_tokens_count": ex.prompt_tokens_count,
            "response_tokens_count": ex.response_tokens_count,
            "total_tokens_count": ex.total_tokens_count,
            "category": ex.category or "unknown",
        }


def collate_sft_batch(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int = 0,
    pad_to_max_len: bool = False,
    max_seq_len: int = 256,
) -> Dict[str, torch.Tensor]:
    """Collate a batch of tokenized SFT examples into padded 2D batch tensors.

    Args:
        batch: List of item dictionaries from SFTDataset.
        pad_token_id: Token ID for padding inputs.
        pad_to_max_len: If True, pad to max_seq_len; otherwise to max length in batch.
        max_seq_len: Maximum block size (default: 256).

    Returns:
        Dictionary containing:
            - 'input_ids': Tensor (batch_size, seq_len) of input IDs.
            - 'target_ids': Tensor (batch_size, seq_len) of target IDs (-100 on prompt & pad).
            - 'loss_mask': Tensor (batch_size, seq_len) of 1/0 active target flags.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    if pad_to_max_len:
        target_len = max_seq_len
    else:
        target_len = max(item["input_ids"].size(0) for item in batch)
        if target_len > max_seq_len:
            target_len = max_seq_len

    batch_size = len(batch)
    padded_inputs = torch.full(
        (batch_size, target_len), pad_token_id, dtype=torch.long
    )
    padded_targets = torch.full(
        (batch_size, target_len), IGNORE_INDEX, dtype=torch.long
    )
    padded_mask = torch.zeros((batch_size, target_len), dtype=torch.long)

    for i, item in enumerate(batch):
        inp = item["input_ids"][:target_len]
        tgt = item["target_ids"][:target_len]
        msk = item["loss_mask"][:target_len]
        length = inp.size(0)

        padded_inputs[i, :length] = inp
        padded_targets[i, :length] = tgt
        padded_mask[i, :length] = msk

    return {
        "input_ids": padded_inputs,
        "target_ids": padded_targets,
        "loss_mask": padded_mask,
    }


class SFTDataPipeline:
    """End-to-end dataset pipeline managing train and validation splits for SFT.

    Step 27 Consumption Contract:
        During Step 27 fine-tuning, the trainer can directly invoke:
            x, targets, loss_mask = pipeline.get_batch(split="train", batch_size=32, device="cuda")
            logits, loss = model(x, targets)
        Because prompt and padding tokens in `targets` have value -100, PyTorch's
        F.cross_entropy automatically masks them out, training the model strictly
        on assistant response tokens.

    Attributes:
        train_dataset: SFTDataset for training.
        val_dataset: SFTDataset for validation.
        tokenizer: BPETokenizer instance.
        pad_token_id: Integer token ID for padding.
        max_seq_len: Maximum block size (default: 256).
    """

    def __init__(
        self,
        train_dataset: SFTDataset,
        val_dataset: SFTDataset,
        tokenizer: BPETokenizer,
        pad_token_id: int = 0,
        max_seq_len: int = 256,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 32,
        device: Union[str, torch.device] = "cpu",
        pad_to_max_len: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch with assistant-only loss masking.

        Args:
            split: 'train' or 'val'.
            batch_size: Number of examples in the batch.
            device: Target torch device.
            pad_to_max_len: Whether to pad to fixed max_seq_len (recommended for GPU).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - input_ids: (batch_size, seq_len)
                - target_ids: (batch_size, seq_len) where prompt/pad tokens are -100
                - loss_mask: (batch_size, seq_len) where 1 indicates response token
        """
        dataset = self.train_dataset if split == "train" else self.val_dataset
        if len(dataset) == 0:
            raise ValueError(f"Split {split!r} contains 0 examples.")

        # Random sampling with replacement if batch_size > len(dataset)
        indices = torch.randint(
            len(dataset), (batch_size,), generator=generator
        ).tolist()
        batch_items = [dataset[i] for i in indices]

        collated = collate_sft_batch(
            batch_items,
            pad_token_id=self.pad_token_id,
            pad_to_max_len=pad_to_max_len,
            max_seq_len=self.max_seq_len,
        )

        x = collated["input_ids"].to(device)
        y = collated["target_ids"].to(device)
        mask = collated["loss_mask"].to(device)

        return x, y, mask

    def compute_stats(self) -> Dict[str, Any]:
        """Compute comprehensive token length and category statistics across splits."""
        stats: Dict[str, Any] = {
            "train_examples": len(self.train_dataset),
            "val_examples": len(self.val_dataset),
            "total_examples": len(self.train_dataset) + len(self.val_dataset),
        }

        all_examples = self.train_dataset.examples + self.val_dataset.examples
        if not all_examples:
            return stats

        prompt_lens = [ex.prompt_tokens_count for ex in all_examples]
        resp_lens = [ex.response_tokens_count for ex in all_examples]
        total_lens = [ex.total_tokens_count for ex in all_examples]

        import numpy as np

        stats["tokens"] = {
            "total_prompt_tokens": int(sum(prompt_lens)),
            "total_response_tokens": int(sum(resp_lens)),
            "total_tokens": int(sum(total_lens)),
            "prompt_length": {
                "min": int(min(prompt_lens)),
                "max": int(max(prompt_lens)),
                "mean": float(np.mean(prompt_lens)),
                "median": float(np.median(prompt_lens)),
            },
            "response_length": {
                "min": int(min(resp_lens)),
                "max": int(max(resp_lens)),
                "mean": float(np.mean(resp_lens)),
                "median": float(np.median(resp_lens)),
            },
            "sequence_length": {
                "min": int(min(total_lens)),
                "max": int(max(total_lens)),
                "mean": float(np.mean(total_lens)),
                "median": float(np.median(total_lens)),
            },
            "active_target_token_percentage": float(
                sum(resp_lens) / max(1, sum(total_lens)) * 100.0
            ),
        }

        categories: Dict[str, int] = {}
        for ex in all_examples:
            cat = ex.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        stats["categories"] = categories

        return stats

    def save(
        self,
        save_dir: Union[str, Path],
        manifest_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize tokenized train and validation datasets to disk.

        Saves:
            - train_sft_tokens.pt
            - val_sft_tokens.pt
            - manifest_sft.json
            - sft_cache_metadata.json

        Args:
            save_dir: Directory path for cache.
            manifest_metadata: Optional provenance dictionary.
        """
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save train examples
        train_data = [
            {
                "example_id": ex.example_id,
                "input_ids": ex.input_ids,
                "target_ids": ex.target_ids,
                "loss_mask": ex.loss_mask,
                "prompt_tokens_count": ex.prompt_tokens_count,
                "response_tokens_count": ex.response_tokens_count,
                "total_tokens_count": ex.total_tokens_count,
                "category": ex.category,
                "metadata": ex.metadata,
            }
            for ex in self.train_dataset.examples
        ]
        torch.save(train_data, out_dir / "train_sft_tokens.pt")

        # 2. Save val examples
        val_data = [
            {
                "example_id": ex.example_id,
                "input_ids": ex.input_ids,
                "target_ids": ex.target_ids,
                "loss_mask": ex.loss_mask,
                "prompt_tokens_count": ex.prompt_tokens_count,
                "response_tokens_count": ex.response_tokens_count,
                "total_tokens_count": ex.total_tokens_count,
                "category": ex.category,
                "metadata": ex.metadata,
            }
            for ex in self.val_dataset.examples
        ]
        torch.save(val_data, out_dir / "val_sft_tokens.pt")

        # 3. Compute stats
        stats = self.compute_stats()

        # 4. Manifest and Metadata
        tok_fp = self.tokenizer.get_fingerprint()
        meta = {
            "tokenizer_fingerprint": tok_fp,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_len": self.max_seq_len,
            "pad_token_id": self.pad_token_id,
            "train_examples_count": len(self.train_dataset),
            "val_examples_count": len(self.val_dataset),
            "statistics": stats,
            "extra_metadata": manifest_metadata or {},
        }

        with open(
            out_dir / "sft_cache_metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(meta, f, indent=2)

        manifest = {
            "dataset_name": (manifest_metadata or {}).get(
                "dataset_name", "Databricks Dolly 15k (SFT Subset)"
            ),
            "license": "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
            "attribution": "Databricks Dolly 15k, (c) 2023 Databricks, Inc., licensed under CC BY-SA 3.0",
            "format": "user_prefix='<|user|>\\n', assistant_prefix='<|assistant|>\\n', end_of_turn='<|endoftext|>\\n'",
            "tokenizer_fingerprint": tok_fp,
            "max_seq_len": self.max_seq_len,
            "train_examples": len(self.train_dataset),
            "val_examples": len(self.val_dataset),
            "statistics": stats,
            "consumption_notes": (
                "In Step 27, load via SFTDataPipeline.load(cache_dir, tokenizer). "
                "Use pipeline.get_batch(split='train') to obtain (input_ids, target_ids, loss_mask). "
                "Pass directly to GroundUpGPT(input_ids, target_ids) with default ignore_index=-100."
            ),
        }
        with open(out_dir / "manifest_sft.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"SFT dataset cache successfully written to {out_dir}")

    @classmethod
    def load(
        cls,
        save_dir: Union[str, Path],
        tokenizer: BPETokenizer,
        pad_token_id: int = 0,
        pad_to_max_len: bool = False,
    ) -> SFTDataPipeline:
        """Load serialized SFT tokenized dataset cache and verify tokenizer compatibility.

        Args:
            save_dir: Directory containing cached .pt and .json artifacts.
            tokenizer: BPETokenizer instance to verify against fingerprint.
            pad_token_id: Integer token ID for padding.
            pad_to_max_len: Whether datasets should pad to fixed max length.

        Returns:
            Initialized SFTDataPipeline.

        Raises:
            FileNotFoundError: If cache files are missing.
            ValueError: If tokenizer fingerprint mismatches the cache.
        """
        cache_path = Path(save_dir)
        train_file = cache_path / "train_sft_tokens.pt"
        val_file = cache_path / "val_sft_tokens.pt"
        meta_file = cache_path / "sft_cache_metadata.json"

        if (
            not train_file.is_file()
            or not val_file.is_file()
            or not meta_file.is_file()
        ):
            raise FileNotFoundError(
                f"SFT cache artifacts missing in {cache_path}. "
                f"Expected train_sft_tokens.pt, val_sft_tokens.pt, and sft_cache_metadata.json."
            )

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        expected_fp = tokenizer.get_fingerprint()
        cached_fp = meta.get("tokenizer_fingerprint")
        if cached_fp != expected_fp:
            raise ValueError(
                f"Tokenizer fingerprint mismatch! Cache was prepared with {cached_fp}, "
                f"but loaded tokenizer fingerprint is {expected_fp}."
            )

        max_seq_len = meta.get("max_seq_len", 256)

        def _normalize_prompt_count(d: Dict[str, Any]) -> int:
            p_count = d.get("prompt_tokens_count", 0)
            r_count = d.get("response_tokens_count", 0)
            inp_len = d["input_ids"].size(0)
            tot_len = d.get("total_tokens_count", inp_len + 1)
            # Legacy cache stored p_count = inp_len - r_count = P - 1
            if p_count + r_count == inp_len and p_count + r_count < tot_len:
                return p_count + 1
            return p_count

        train_raw = torch.load(train_file, weights_only=True)
        train_examples = [
            TokenizedExample(
                example_id=d["example_id"],
                input_ids=d["input_ids"],
                target_ids=d["target_ids"],
                loss_mask=d["loss_mask"],
                prompt_tokens_count=_normalize_prompt_count(d),
                response_tokens_count=d["response_tokens_count"],
                total_tokens_count=d["total_tokens_count"],
                category=d.get("category"),
                metadata=d.get("metadata"),
            )
            for d in train_raw
        ]

        val_raw = torch.load(val_file, weights_only=True)
        val_examples = [
            TokenizedExample(
                example_id=d["example_id"],
                input_ids=d["input_ids"],
                target_ids=d["target_ids"],
                loss_mask=d["loss_mask"],
                prompt_tokens_count=_normalize_prompt_count(d),
                response_tokens_count=d["response_tokens_count"],
                total_tokens_count=d["total_tokens_count"],
                category=d.get("category"),
                metadata=d.get("metadata"),
            )
            for d in val_raw
        ]

        train_ds = SFTDataset(
            train_examples,
            pad_token_id=pad_token_id,
            pad_to_max_len=pad_to_max_len,
            max_seq_len=max_seq_len,
        )
        val_ds = SFTDataset(
            val_examples,
            pad_token_id=pad_token_id,
            pad_to_max_len=pad_to_max_len,
            max_seq_len=max_seq_len,
        )

        return cls(
            train_dataset=train_ds,
            val_dataset=val_ds,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            max_seq_len=max_seq_len,
        )


class BlendedSFTDataPipeline:
    """Blended SFT data pipeline combining a large general instruction dataset (Dolly 15k)
    with a compact verified conversational anchor dataset.

    Samples mini-batches dynamically from both source pools with configurable sampling weights,
    guaranteeing substantial exposure to both sources while tracking sample counts, token exposures,
    and category-level distributions.
    """

    def __init__(
        self,
        dolly_pipeline: SFTDataPipeline,
        anchor_train_examples: List[Dict[str, Any]],
        anchor_val_examples: List[Dict[str, Any]],
        anchor_test_examples: Optional[List[Dict[str, Any]]] = None,
        anchor_sample_prob: float = 0.55,
        pad_token_id: int = 0,
        pad_to_max_len: bool = False,
        seed: int = 42,
    ) -> None:
        self.dolly_pipeline = dolly_pipeline
        self.anchor_train_examples = anchor_train_examples
        self.anchor_val_examples = anchor_val_examples
        self.anchor_test_examples = anchor_test_examples or []
        self.anchor_sample_prob = float(anchor_sample_prob)
        self.pad_token_id = pad_token_id
        self.pad_to_max_len = pad_to_max_len
        self.rng = random.Random(seed)

        # Exposure tracking
        self.dolly_samples_drawn: int = 0
        self.anchor_samples_drawn: int = 0
        self.dolly_tokens_drawn: int = 0
        self.anchor_tokens_drawn: int = 0
        self.category_exposure_counts: Dict[str, int] = {}

    @property
    def val_dataset(self) -> List[Any]:
        """Provides access to the primary held-out validation split (Dolly val)."""
        return self.dolly_pipeline.val_dataset

    @property
    def val_anchor_dataset(self) -> List[Any]:
        """Access to the held-out conversational anchor validation split."""
        return self.anchor_val_examples

    @property
    def test_transfer_dataset(self) -> List[Any]:
        """Access to the held-out conversational anchor transfer test split."""
        return self.anchor_test_examples

    @property
    def max_seq_len(self) -> int:
        return self.dolly_pipeline.max_seq_len

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 16,
        device: Union[str, torch.device] = "cpu",
        pad_to_max_len: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the blended pipeline."""
        pad_flag = self.pad_to_max_len if pad_to_max_len is None else pad_to_max_len

        if split == "train":
            items: List[Any] = []
            for _ in range(batch_size):
                if self.anchor_train_examples and self.rng.random() < self.anchor_sample_prob:
                    idx = self.rng.randrange(len(self.anchor_train_examples))
                    ex = self.anchor_train_examples[idx]
                    self.anchor_samples_drawn += 1
                    r_toks = ex.get("response_tokens_count", 0)
                    self.anchor_tokens_drawn += int(r_toks)
                    cat = ex.get("category", "anchor")
                    self.category_exposure_counts[cat] = self.category_exposure_counts.get(cat, 0) + 1
                    items.append(ex)
                else:
                    idx = self.rng.randrange(len(self.dolly_pipeline.train_dataset))
                    ex = self.dolly_pipeline.train_dataset[idx]
                    self.dolly_samples_drawn += 1
                    r_toks = getattr(ex, "response_tokens_count", None)
                    if r_toks is None and isinstance(ex, dict):
                        r_toks = ex.get("response_tokens_count", 0)
                    self.dolly_tokens_drawn += int(r_toks or 0)
                    cat = getattr(ex, "category", None) or (ex.get("category") if isinstance(ex, dict) else "dolly") or "dolly"
                    self.category_exposure_counts[cat] = self.category_exposure_counts.get(cat, 0) + 1
                    items.append(ex)

            collated = collate_sft_batch(
                items,
                pad_token_id=self.pad_token_id,
                pad_to_max_len=pad_flag,
                max_seq_len=self.dolly_pipeline.max_seq_len,
            )
            return (
                collated["input_ids"].to(device),
                collated["target_ids"].to(device),
                collated["loss_mask"].to(device),
            )

        elif split in ("val", "val_dolly"):
            return self.dolly_pipeline.get_batch(
                split="val",
                batch_size=batch_size,
                device=device,
                pad_to_max_len=pad_flag,
                generator=generator,
            )

        elif split == "val_anchor":
            if not self.anchor_val_examples:
                raise ValueError("No anchor validation examples available.")
            indices = [self.rng.randrange(len(self.anchor_val_examples)) for _ in range(batch_size)]
            items = [self.anchor_val_examples[i] for i in indices]
            collated = collate_sft_batch(
                items,
                pad_token_id=self.pad_token_id,
                pad_to_max_len=pad_flag,
                max_seq_len=self.dolly_pipeline.max_seq_len,
            )
            return (
                collated["input_ids"].to(device),
                collated["target_ids"].to(device),
                collated["loss_mask"].to(device),
            )

        elif split == "test_transfer":
            if not self.anchor_test_examples:
                raise ValueError("No anchor test transfer examples available.")
            indices = [self.rng.randrange(len(self.anchor_test_examples)) for _ in range(batch_size)]
            items = [self.anchor_test_examples[i] for i in indices]
            collated = collate_sft_batch(
                items,
                pad_token_id=self.pad_token_id,
                pad_to_max_len=pad_flag,
                max_seq_len=self.dolly_pipeline.max_seq_len,
            )
            return (
                collated["input_ids"].to(device),
                collated["target_ids"].to(device),
                collated["loss_mask"].to(device),
            )
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_exposure_summary(self) -> Dict[str, Any]:
        """Return quantitative accounting of sample exposures and target token exposures."""
        tot_samples = self.dolly_samples_drawn + self.anchor_samples_drawn
        tot_tokens = self.dolly_tokens_drawn + self.anchor_tokens_drawn
        return {
            "dolly_samples_drawn": self.dolly_samples_drawn,
            "anchor_samples_drawn": self.anchor_samples_drawn,
            "total_samples_drawn": tot_samples,
            "anchor_sample_fraction": self.anchor_samples_drawn / max(1, tot_samples),
            "dolly_target_tokens_drawn": self.dolly_tokens_drawn,
            "anchor_target_tokens_drawn": self.anchor_tokens_drawn,
            "total_target_tokens_drawn": tot_tokens,
            "anchor_target_token_fraction": self.anchor_tokens_drawn / max(1, tot_tokens),
            "category_exposures": dict(self.category_exposure_counts),
        }

    @classmethod
    def load(
        cls,
        dolly_cache_dir: Union[str, Path],
        anchor_cache_dir: Union[str, Path],
        tokenizer: BPETokenizer,
        anchor_sample_prob: float = 0.55,
        pad_token_id: int = 0,
        pad_to_max_len: bool = False,
        seed: int = 42,
    ) -> BlendedSFTDataPipeline:
        """Load Dolly SFT cache and anchor SFT cache into a unified blended pipeline."""
        dolly_pipe = SFTDataPipeline.load(
            save_dir=dolly_cache_dir,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            pad_to_max_len=pad_to_max_len,
        )

        anchor_dir = Path(anchor_cache_dir)
        train_file = anchor_dir / "train_anchor_tokens.pt"
        val_file = anchor_dir / "val_anchor_tokens.pt"
        test_file = anchor_dir / "test_transfer_tokens.pt"

        anchor_train = torch.load(train_file, weights_only=False) if train_file.exists() else []
        anchor_val = torch.load(val_file, weights_only=False) if val_file.exists() else []
        anchor_test = torch.load(test_file, weights_only=False) if test_file.exists() else []

        return cls(
            dolly_pipeline=dolly_pipe,
            anchor_train_examples=anchor_train,
            anchor_val_examples=anchor_val,
            anchor_test_examples=anchor_test,
            anchor_sample_prob=anchor_sample_prob,
            pad_token_id=pad_token_id,
            pad_to_max_len=pad_to_max_len,
            seed=seed,
        )

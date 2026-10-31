"""Byte-level Byte Pair Encoding (BPE) tokenizer implemented from scratch.

Operates directly on UTF-8 bytes to guarantee lossless round-trip reconstruction
for any arbitrary text, including Unicode and emoji.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union


class BPETokenizer:
    """Byte-level Byte Pair Encoding (BPE) Tokenizer.

    Attributes:
        merges (dict[tuple[int, int], int]): Mapping from token ID pair to merged ID.
        ranks (dict[tuple[int, int], int]): Priority rank for each learned merge.
        vocab (dict[int, bytes]): Mapping from token ID to its corresponding bytes.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.ranks: dict[tuple[int, int], int] = {}
        # Base vocabulary consists of all 256 raw UTF-8 byte values (IDs 0..255)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._is_fitted: bool = False

    @property
    def vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)

    @staticmethod
    def _get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
        """Compute frequency counts for all adjacent token pairs."""
        counts: dict[tuple[int, int], int] = {}
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """Replace all consecutive occurrences of pair with new token idx in ids."""
        new_ids: list[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def fit(self, text: str, target_vocab_size: int = 300) -> BPETokenizer:
        """Train BPE merge rules on the input text corpus.

        Args:
            text: Input training text.
            target_vocab_size: Desired total vocabulary size (must be >= 256).

        Returns:
            BPETokenizer: Fitted tokenizer instance.

        Raises:
            ValueError: If training text is empty or target_vocab_size < 256.
        """
        if not text:
            raise ValueError("Cannot fit BPETokenizer on empty text.")
        if target_vocab_size < 256:
            raise ValueError(f"target_vocab_size must be at least 256, got {target_vocab_size}")

        num_merges = target_vocab_size - 256
        raw_bytes = text.encode("utf-8")
        ids = list(raw_bytes)

        # Reset vocabulary and merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.ranks = {}

        for rank in range(num_merges):
            if len(ids) <= 4:
                break

            stats = self._get_stats(ids)
            if not stats:
                break

            # Deterministic tie-breaking:
            # Sort by frequency descending (-freq), then by lexicographical order of token ID pair (p[0], p[1])
            best_pair = min(stats.keys(), key=lambda p: (-stats[p], p[0], p[1]))

            # Stop if the most frequent pair occurs less than 2 times
            if stats[best_pair] < 2:
                break

            new_id = 256 + rank
            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.ranks[best_pair] = rank
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        self._is_fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        """Encode input text into a sequence of BPE token IDs.

        Applies learned merge rules in their learned priority order.

        Args:
            text: Input text string.

        Returns:
            list[int]: Sequence of integer token IDs.
        """
        if not text:
            return []

        # Convert text into initial byte IDs [0..255]
        ids = list(text.encode("utf-8"))
        if len(ids) < 2 or not self.merges:
            return ids

        # Iteratively merge the highest priority pair until no mergeable pairs exist
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find candidate pair with the lowest rank (earliest learned merge)
            eligible_pairs = [p for p in stats if p in self.ranks]
            if not eligible_pairs:
                break

            best_pair = min(eligible_pairs, key=lambda p: self.ranks[p])
            idx = self.merges[best_pair]
            ids = self._merge(ids, best_pair, idx)

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of token IDs back into a UTF-8 text string.

        Args:
            ids: Sequence of integer token IDs.

        Returns:
            str: Losslessly reconstructed text string.

        Raises:
            ValueError: If any token ID is invalid (not present in vocabulary).
        """
        if not ids:
            return ""

        byte_parts: list[bytes] = []
        for idx_pos, token_id in enumerate(ids):
            if token_id not in self.vocab:
                raise ValueError(
                    f"Invalid token ID {token_id} at index {idx_pos}. "
                    f"Token ID must be in range [0, {self.vocab_size - 1}]."
                )
            byte_parts.append(self.vocab[token_id])

        raw_bytes = b"".join(byte_parts)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, file_path: Union[str, Path]) -> None:
        """Save learned BPE merge rules and metadata to a portable JSON file.

        Args:
            file_path: Destination JSON file path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "merges": [
                [pair[0], pair[1], new_id, self.ranks[pair]]
                for pair, new_id in self.merges.items()
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> BPETokenizer:
        """Load a saved BPETokenizer from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            BPETokenizer: Restored tokenizer instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is malformed or corrupted.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            raise ValueError(f"Failed to parse tokenizer JSON file: {err}") from err

        if not isinstance(data, dict) or data.get("type") != "byte_bpe" or "merges" not in data:
            raise ValueError("Malformed tokenizer JSON: missing required 'byte_bpe' structure.")

        tokenizer = cls()
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.merges = {}
        tokenizer.ranks = {}

        # Sort merges by rank to reconstruct vocabulary deterministically
        try:
            sorted_merges = sorted(data["merges"], key=lambda m: m[3])
            for p0, p1, new_id, rank in sorted_merges:
                pair = (int(p0), int(p1))
                new_id = int(new_id)
                rank = int(rank)
                tokenizer.merges[pair] = new_id
                tokenizer.ranks[pair] = rank
                tokenizer.vocab[new_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        except Exception as err:
            raise ValueError(f"Corrupted merge entries in tokenizer file: {err}") from err

        tokenizer._is_fitted = True
        return tokenizer

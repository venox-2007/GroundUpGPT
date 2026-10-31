"""Byte-level Byte Pair Encoding (BPE) tokenizer implemented from scratch.

Operates directly on UTF-8 bytes to guarantee lossless round-trip reconstruction
for any arbitrary text, including Unicode and emoji.
"""

from __future__ import annotations

import heapq
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

    def get_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint for base vocabulary and learned merges."""
        import hashlib

        items = sorted(
            [(p[0], p[1], idx) for p, idx in self.merges.items()],
            key=lambda x: x[2],
        )
        payload = f"BPETokenizer:{self.vocab_size}:{json.dumps(items)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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

    def _fit_fast(self, raw_bytes: list[int], num_merges: int) -> None:
        """Optimized first-principles BPE merge learner using linked lists and lazy priority queue."""
        from collections import defaultdict

        n = len(raw_bytes)
        if n < 2 or num_merges <= 0:
            return

        # Doubly-linked list node pointers
        prev_i = list(range(-1, n - 1))
        next_i = list(range(1, n + 1))
        next_i[-1] = -1
        val = list(raw_bytes)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_locs: dict[tuple[int, int], set[int]] = defaultdict(set)

        for i in range(n - 1):
            p = (val[i], val[i + 1])
            pair_counts[p] += 1
            pair_locs[p].add(i)

        heap: list[tuple[int, int, int, tuple[int, int]]] = []
        for p, c in pair_counts.items():
            if c >= 2:
                heap.append((-c, p[0], p[1], p))
        heapq.heapify(heap)

        for rank in range(num_merges):
            best_pair = None
            while heap:
                neg_c, p0, p1, p = heapq.heappop(heap)
                curr_c = pair_counts.get(p, 0)
                if curr_c < 2:
                    continue
                if -neg_c == curr_c:
                    best_pair = p
                    break
                else:
                    heapq.heappush(heap, (-curr_c, p[0], p[1], p))

            if best_pair is None or pair_counts.get(best_pair, 0) < 2:
                # Stop when no pairs occur at least twice
                break

            new_id = 256 + rank
            self.merges[best_pair] = new_id
            self.ranks[best_pair] = rank
            self.vocab[new_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

            # Process all occurrences of best_pair in left-to-right order
            locs = sorted(pair_locs[best_pair])
            del pair_counts[best_pair]
            del pair_locs[best_pair]

            for pos in locs:
                j = next_i[pos]
                if (
                    j == -1
                    or val[pos] != best_pair[0]
                    or val[j] != best_pair[1]
                ):
                    continue

                p_idx = prev_i[pos]
                k = next_i[j]

                # Remove old left pair
                if p_idx != -1:
                    old_left = (val[p_idx], val[pos])
                    pair_counts[old_left] -= 1
                    pair_locs[old_left].discard(p_idx)

                # Remove old right pair
                if k != -1:
                    old_right = (val[j], val[k])
                    pair_counts[old_right] -= 1
                    pair_locs[old_right].discard(j)

                # Merge nodes
                val[pos] = new_id
                next_i[pos] = k
                if k != -1:
                    prev_i[k] = pos

                # Add new left pair
                if p_idx != -1:
                    new_left = (val[p_idx], new_id)
                    pair_counts[new_left] += 1
                    pair_locs[new_left].add(p_idx)
                    if pair_counts[new_left] >= 2:
                        heapq.heappush(
                            heap,
                            (
                                -pair_counts[new_left],
                                new_left[0],
                                new_left[1],
                                new_left,
                            ),
                        )

                # Add new right pair
                if k != -1:
                    new_right = (new_id, val[k])
                    pair_counts[new_right] += 1
                    pair_locs[new_right].add(pos)
                    if pair_counts[new_right] >= 2:
                        heapq.heappush(
                            heap,
                            (
                                -pair_counts[new_right],
                                new_right[0],
                                new_right[1],
                                new_right,
                            ),
                        )

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
            raise ValueError(
                f"target_vocab_size must be at least 256, got {target_vocab_size}"
            )

        num_merges = target_vocab_size - 256
        raw_bytes = list(text.encode("utf-8"))

        # Reset vocabulary and merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.ranks = {}

        if num_merges > 0:
            self._fit_fast(raw_bytes, num_merges)

        self._is_fitted = True
        return self

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """Encode a single slice/line of bytes using priority-ranked merges."""
        ids = list(chunk_bytes)
        if len(ids) < 2 or not self.merges:
            return ids

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            eligible_pairs = [p for p in stats if p in self.ranks]
            if not eligible_pairs:
                break
            best_pair = min(eligible_pairs, key=lambda p: self.ranks[p])
            idx = self.merges[best_pair]
            ids = self._merge(ids, best_pair, idx)

        return ids

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
        if not self.merges:
            return list(text.encode("utf-8"))

        # For multi-line text, process line-by-line with memoization cache for high throughput
        if "\n" in text:
            lines = text.splitlines(keepends=True)
            cache: dict[bytes, list[int]] = {}
            all_ids: list[int] = []
            for line in lines:
                raw_b = line.encode("utf-8")
                if raw_b not in cache:
                    cache[raw_b] = self._encode_chunk(raw_b)
                all_ids.extend(cache[raw_b])
            return all_ids

        return self._encode_chunk(text.encode("utf-8"))

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
        try:
            return raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as err:
            raise ValueError(
                f"Failed to decode byte sequence into valid UTF-8 string: {err}"
            ) from err

    def save(self, file_path: Union[str, Path]) -> None:
        """Save learned BPE merge rules and metadata to a portable JSON file.

        Args:
            file_path: Destination path for the saved JSON tokenizer file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialized format: list of [tokenA, tokenB, merged_id, rank]
        merges_serialized = [
            [pair[0], pair[1], token_id, self.ranks[pair]]
            for pair, token_id in sorted(self.merges.items(), key=lambda x: self.ranks[x[0]])
        ]

        data = {
            "version": "1.0",
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "merges": merges_serialized,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> BPETokenizer:
        """Load and strictly validate a saved BPE tokenizer from JSON file.

        Performs comprehensive integrity validation:
        1. Confirms required top-level schema fields.
        2. Rejects duplicate merge ranks, duplicate pairs, or duplicate token IDs.
        3. Validates contiguous merge ranks starting from 0.
        4. Validates contiguous token IDs starting from 256.
        5. Validates dependency causality: merge pairs must reference existing IDs.
        6. Confirms final reconstructed vocabulary size matches declared vocab_size.

        Args:
            file_path: Path to the JSON tokenizer file.

        Returns:
            BPETokenizer: Reconstructed and validated tokenizer instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If metadata, ranks, IDs, merges, or vocab sizes are corrupted.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            raise ValueError(f"Failed to parse tokenizer JSON file: {err}") from err

        # 1. Structural and metadata validation
        if not isinstance(data, dict):
            raise ValueError("Malformed tokenizer JSON: root must be an object.")
        if data.get("type") != "byte_bpe":
            raise ValueError(f"Unsupported tokenizer type: {data.get('type')!r}, expected 'byte_bpe'.")
        if not isinstance(data.get("version"), str):
            raise ValueError("Malformed tokenizer JSON: missing or invalid 'version' field.")

        declared_vocab_size = data.get("vocab_size")
        if not isinstance(declared_vocab_size, int) or declared_vocab_size < 256:
            raise ValueError(
                f"Malformed tokenizer JSON: invalid vocab_size {declared_vocab_size!r}."
            )

        merges_data = data.get("merges")
        if not isinstance(merges_data, list):
            raise ValueError("Malformed tokenizer JSON: 'merges' must be a list.")

        expected_vocab_size = 256 + len(merges_data)
        if declared_vocab_size != expected_vocab_size:
            raise ValueError(
                f"Inconsistent declared vocab_size {declared_vocab_size} does not match "
                f"base 256 bytes + {len(merges_data)} merges ({expected_vocab_size})."
            )

        tokenizer = cls()
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.merges = {}
        tokenizer.ranks = {}

        seen_pairs: set[tuple[int, int]] = set()
        seen_new_ids: set[int] = set()
        seen_ranks: set[int] = set()

        # 2. Sort by rank to reconstruct deterministically
        try:
            sorted_merges = sorted(merges_data, key=lambda m: m[3])
        except Exception as err:
            raise ValueError(f"Malformed merge entries in tokenizer file: {err}") from err

        # 3. Step-by-step merge validation and vocabulary reconstruction
        for expected_rank, entry in enumerate(sorted_merges):
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise ValueError(
                    f"Invalid merge entry format at rank {expected_rank}: {entry!r}. "
                    "Expected [p0, p1, new_id, rank]."
                )

            p0, p1, new_id, rank = entry
            if not all(isinstance(v, int) for v in (p0, p1, new_id, rank)):
                raise ValueError(f"All merge elements must be integers, got {entry!r}.")

            # Rank contiguity and duplicate check
            if rank in seen_ranks:
                raise ValueError(f"Duplicate merge rank detected: {rank}.")
            if rank != expected_rank:
                raise ValueError(
                    f"Non-contiguous merge rank: expected {expected_rank}, got {rank}."
                )
            seen_ranks.add(rank)

            # Generated token ID validation
            expected_new_id = 256 + rank
            if new_id < 256:
                raise ValueError(
                    f"Cannot overwrite base byte token ID {new_id}. "
                    "Learned token IDs must be >= 256."
                )
            if new_id in seen_new_ids:
                raise ValueError(f"Duplicate merged token ID detected: {new_id}.")
            if new_id != expected_new_id:
                raise ValueError(
                    f"Non-contiguous token ID: expected {expected_new_id}, got {new_id}."
                )
            seen_new_ids.add(new_id)

            # Merge pair uniqueness and dependency validation
            pair = (p0, p1)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate merge pair detected: {pair}.")
            if p0 not in tokenizer.vocab or p1 not in tokenizer.vocab:
                raise ValueError(
                    f"Merge references unavailable or future token IDs: pair={pair}, "
                    f"available vocabulary size={len(tokenizer.vocab)}."
                )

            seen_pairs.add(pair)
            tokenizer.merges[pair] = new_id
            tokenizer.ranks[pair] = rank
            tokenizer.vocab[new_id] = tokenizer.vocab[p0] + tokenizer.vocab[p1]

        if len(tokenizer.vocab) != declared_vocab_size:
            raise ValueError(
                f"Reconstructed vocabulary size {len(tokenizer.vocab)} does not match "
                f"declared vocab_size {declared_vocab_size}."
            )

        tokenizer._is_fitted = True
        return tokenizer

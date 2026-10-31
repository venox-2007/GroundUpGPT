"""Character-level tokenizer implemented from scratch.

A deterministic, educational tokenizer that maps unique characters to integer
token IDs and vice versa.
"""

from __future__ import annotations

from typing import Sequence


class CharTokenizer:
    """Deterministic character-level tokenizer.

    Attributes:
        stoi (dict[str, int]): Mapping from string character to token ID.
        itos (dict[int, str]): Mapping from token ID to string character.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self._is_fitted: bool = False

    @property
    def vocab_size(self) -> int:
        """Return the total number of unique characters in the vocabulary."""
        if not self._is_fitted:
            raise ValueError("Tokenizer has not been fitted yet. Call fit() first.")
        return len(self.stoi)

    def fit(self, text: str) -> CharTokenizer:
        """Build deterministic vocabulary from unique characters in sorted order.

        Args:
            text: Input training text corpus.

        Returns:
            CharTokenizer: The fitted tokenizer instance.

        Raises:
            ValueError: If the training text is empty.
        """
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text.")

        unique_chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}
        self._is_fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        """Convert a text string into a list of integer token IDs.

        Args:
            text: Input string to tokenize.

        Returns:
            list[int]: Sequence of integer token IDs.

        Raises:
            ValueError: If tokenizer is unfitted or text contains unknown characters.
        """
        if not self._is_fitted:
            raise ValueError("Tokenizer has not been fitted yet. Call fit() first.")

        token_ids: list[int] = []
        for idx, ch in enumerate(text):
            if ch not in self.stoi:
                raise ValueError(
                    f"Unknown character '{ch}' at position {idx}. "
                    f"Character not present in tokenizer vocabulary."
                )
            token_ids.append(self.stoi[ch])
        return token_ids

    def decode(self, ids: Sequence[int]) -> str:
        """Convert a sequence of integer token IDs back into a text string.

        Args:
            ids: Sequence of token IDs to decode.

        Returns:
            str: Reconstructed text string.

        Raises:
            ValueError: If tokenizer is unfitted or an ID is invalid.
        """
        if not self._is_fitted:
            raise ValueError("Tokenizer has not been fitted yet. Call fit() first.")

        chars: list[str] = []
        for idx, token_id in enumerate(ids):
            if token_id not in self.itos:
                raise ValueError(
                    f"Invalid token ID {token_id} at index {idx}. "
                    f"Valid token IDs are in range [0, {self.vocab_size - 1}]."
                )
            chars.append(self.itos[token_id])
        return "".join(chars)

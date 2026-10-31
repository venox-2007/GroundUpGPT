"""Configuration settings for the GroundUpGPT local inference server."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Server and inference configuration settings.

    Attributes:
        host: Host IP address to bind server (default: 127.0.0.1).
        port: Port number to bind server (default: 8000).
        checkpoint_path: Path to the model checkpoint file.
        tokenizer_path: Path to the tokenizer JSON file.
        cors_origins: List of permitted origins for cross-origin requests.
        max_prompt_chars: Maximum allowed prompt length in characters.
        max_new_tokens_limit: Maximum allowed max_new_tokens per request.
        device: Target execution device ('auto', 'cuda', or 'cpu').
    """

    host: str = field(
        default_factory=lambda: os.getenv("GROUNDUPGPT_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("GROUNDUPGPT_PORT", "8000"))
    )
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_CHECKPOINT_PATH",
            str(REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"),
        )
    )
    tokenizer_path: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_TOKENIZER_PATH",
            str(REPO_ROOT / "checkpoints" / "demo_tokenizer.json"),
        )
    )
    cors_origins: List[str] = field(
        default_factory=lambda: json.loads(
            os.getenv(
                "GROUNDUPGPT_CORS_ORIGINS",
                '["http://localhost:3000", "http://127.0.0.1:3000"]',
            )
        )
    )
    max_prompt_chars: int = field(
        default_factory=lambda: int(os.getenv("GROUNDUPGPT_MAX_PROMPT_CHARS", "2048"))
    )
    max_new_tokens_limit: int = field(
        default_factory=lambda: int(os.getenv("GROUNDUPGPT_MAX_NEW_TOKENS", "512"))
    )
    device: str = field(
        default_factory=lambda: os.getenv("GROUNDUPGPT_DEVICE", "auto")
    )
    auto_load_on_startup: bool = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_AUTO_LOAD", "true"
        ).lower() in ("1", "true", "yes")
    )


def get_settings() -> Settings:
    """Retrieve active server settings instance."""
    return Settings()

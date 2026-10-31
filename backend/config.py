"""Configuration settings for the GroundUpGPT local inference server."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "5m_initial": {
        "model_name": "GroundUpGPT-5M (5.05M)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "5m": {
        "model_name": "GroundUpGPT-5M (5.05M)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "demo": {
        "model_name": "GroundUpGPT-Demo (35K)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "demo_checkpoint.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "demo_tokenizer.json"),
    },
}


def resolve_default_profile() -> str:
    """Determine the default model profile based on environment and available checkpoints."""
    env_profile = os.getenv("GROUNDUPGPT_MODEL_PROFILE")
    if env_profile:
        return env_profile.lower().strip()

    # If 5M trained checkpoint exists, prefer it
    p5m = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"
    if p5m.is_file():
        return "5m_initial"
    return "demo"


def get_trusted_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve trusted profile dictionary by key, or None if untrusted/unknown."""
    return MODEL_PROFILES.get(profile_name.lower().strip())


def register_trusted_profile(key: str, profile_dict: Dict[str, Any]) -> None:
    """Register a trusted profile on the server side (for testing or server configuration)."""
    MODEL_PROFILES[key.lower().strip()] = profile_dict


def resolve_profile_paths(profile_name: str) -> Dict[str, str]:
    """Retrieve checkpoint, tokenizer, and display name for a given profile."""
    prof = MODEL_PROFILES.get(profile_name, MODEL_PROFILES["demo"])
    return {
        "model_name": prof["model_name"],
        "checkpoint_path": prof["checkpoint_path"],
        "tokenizer_path": prof["tokenizer_path"],
    }


@dataclass
class Settings:
    """Server and inference configuration settings.

    Attributes:
        host: Host IP address to bind server (default: 127.0.0.1).
        port: Port number to bind server (default: 8000).
        model_profile: Active model profile ('demo' or '5m_initial').
        model_name: Display name for the loaded model.
        checkpoint_path: Path to the model checkpoint file.
        tokenizer_path: Path to the tokenizer JSON file.
        cors_origins: List of permitted origins for cross-origin requests.
        max_prompt_chars: Maximum allowed prompt length in characters.
        max_new_tokens_limit: Maximum allowed max_new_tokens per request.
        device: Target execution device ('auto', 'cuda', or 'cpu').
        auto_load_on_startup: Whether to load model on server start.
    """

    host: str = field(
        default_factory=lambda: os.getenv("GROUNDUPGPT_HOST", "127.0.0.1")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("GROUNDUPGPT_PORT", "8000"))
    )
    model_profile: str = field(default_factory=resolve_default_profile)
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_MODEL_NAME",
            resolve_profile_paths(resolve_default_profile())["model_name"],
        )
    )
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_CHECKPOINT_PATH",
            resolve_profile_paths(resolve_default_profile())["checkpoint_path"],
        )
    )
    tokenizer_path: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDUPGPT_TOKENIZER_PATH",
            resolve_profile_paths(resolve_default_profile())["tokenizer_path"],
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

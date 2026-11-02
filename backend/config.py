"""Configuration settings for the GroundUpGPT local inference server."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "27m_anchors_best": {
        "model_name": "GroundUpGPT-27M Anchors Best (Experimental Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "27m_sft_anchors_v1" / "model_27m_anchors_best.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "27m_anchors_latest": {
        "model_name": "GroundUpGPT-27M Anchors Latest (Experimental Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "27m_sft_anchors_v1" / "model_27m_anchors_latest.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "27m_sft_best": {
        "model_name": "GroundUpGPT-27M SFT Best (Experimental Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_best.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "27m_sft_latest": {
        "model_name": "GroundUpGPT-27M SFT Latest (Experimental Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "27m_sft_phase1" / "model_27m_sft_latest.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "5m_sft_best": {
        "model_name": "GroundUpGPT-5M SFT Best (Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "5m_sft_latest": {
        "model_name": "GroundUpGPT-5M SFT Latest (Chat)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_latest.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
        "is_sft": True,
    },
    "5m_phase2_best": {
        "model_name": "GroundUpGPT-5M Phase 2 Best (Step 3000)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_best.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "5m_phase2_latest": {
        "model_name": "GroundUpGPT-5M Phase 2 Latest (Step 3000)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_latest.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "5m_best": {
        "model_name": "GroundUpGPT-5M Continued Best (Step 1000)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_best.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "5m_latest": {
        "model_name": "GroundUpGPT-5M Continued Latest (Step 1000)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_latest.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
    "5m_step200": {
        "model_name": "GroundUpGPT-5M Baseline (Step 200)",
        "checkpoint_path": str(REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"),
        "tokenizer_path": str(REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"),
    },
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


CANONICAL_PROFILE_ORDER: List[str] = [
    "27m_sft_best",
    "27m_sft_latest",
    "27m_anchors_best",
    "27m_anchors_latest",
    "5m_sft_best",
    "5m_sft_latest",
    "5m_phase2_best",
    "5m_phase2_latest",
    "5m_best",
    "5m_latest",
    "5m_step200",
    "demo",
]

PROFILE_ALIASES: Dict[str, str] = {
    "5m_initial": "5m_step200",
    "5m": "5m_step200",
}


def resolve_default_profile() -> str:
    """Determine the default model profile based on environment and available checkpoints."""
    env_profile = os.getenv("GROUNDUPGPT_MODEL_PROFILE")
    if env_profile:
        return env_profile.lower().strip()

    p_bpe = REPO_ROOT / "checkpoints" / "bpe_vocab4096.json"

    # 1. Prefer SFT Best 5M checkpoint if available
    p5m_sft_best = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_best.pt"
    if p5m_sft_best.is_file() and p_bpe.is_file():
        return "5m_sft_best"

    # 2. Prefer SFT Latest 5M checkpoint if available
    p5m_sft_latest = REPO_ROOT / "checkpoints" / "5m_sft" / "model_5m_sft_latest.pt"
    if p5m_sft_latest.is_file() and p_bpe.is_file():
        return "5m_sft_latest"

    # 3. Prefer Phase 2 Best 5M checkpoint if available
    p5m_p2_best = REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_best.pt"
    if p5m_p2_best.is_file() and p_bpe.is_file():
        return "5m_phase2_best"

    # 4. Prefer Phase 2 Latest 5M checkpoint if available
    p5m_p2_latest = REPO_ROOT / "checkpoints" / "5m_phase2" / "model_5m_latest.pt"
    if p5m_p2_latest.is_file() and p_bpe.is_file():
        return "5m_phase2_latest"

    # 5. Prefer Best continued 5M checkpoint (Step 1000) if available
    p5m_best = REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_best.pt"
    if p5m_best.is_file() and p_bpe.is_file():
        return "5m_best"

    # 6. Prefer Latest continued 5M checkpoint if available
    p5m_latest = REPO_ROOT / "checkpoints" / "5m_continued" / "model_5m_latest.pt"
    if p5m_latest.is_file() and p_bpe.is_file():
        return "5m_latest"

    # 7. Prefer Initial Step 200 checkpoint if available
    p5m_step200 = REPO_ROOT / "checkpoints" / "5m_initial" / "model_5m_step200.pt"
    if p5m_step200.is_file() and p_bpe.is_file():
        return "5m_step200"

    # 8. Fall back to demo checkpoint
    return "demo"


def get_available_profiles() -> Dict[str, Dict[str, Any]]:
    """Return only distinct canonical profiles whose checkpoint and tokenizer files exist on disk.

    Guarantees no duplicate profiles pointing to the same underlying model checkpoint artifact.
    """
    available = {}
    seen_checkpoints = set()

    for key in CANONICAL_PROFILE_ORDER:
        if key not in MODEL_PROFILES:
            continue
        prof = MODEL_PROFILES[key]
        ckpt_p = Path(prof["checkpoint_path"]).resolve()
        tok_p = Path(prof["tokenizer_path"]).resolve()
        if ckpt_p.is_file() and tok_p.is_file():
            if str(ckpt_p) not in seen_checkpoints:
                available[key] = prof
                seen_checkpoints.add(str(ckpt_p))

    # Check for custom registered profiles (excluding known aliases and duplicate artifacts)
    for k, v in MODEL_PROFILES.items():
        if k in available or k in PROFILE_ALIASES:
            continue
        ckpt_p = Path(v["checkpoint_path"]).resolve()
        tok_p = Path(v["tokenizer_path"]).resolve()
        if ckpt_p.is_file() and tok_p.is_file():
            if str(ckpt_p) not in seen_checkpoints:
                available[k] = v
                seen_checkpoints.add(str(ckpt_p))

    return available


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

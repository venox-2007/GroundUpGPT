"""Inference-only model loader and concurrency manager for GroundUpGPT."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging
import torch

from src.model import GPTConfig, GroundUpGPT
from src.bpe_tokenizer import BPETokenizer
from src.tokenizer import CharTokenizer
from src.trainer import get_tokenizer_fingerprint
from src.generation import TextGenerator
from backend.schemas import ModelInfoResponse

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle, checkpoint restoration, and inference serialization.

    Attributes:
        model: Loaded GroundUpGPT instance.
        tokenizer: Loaded BPETokenizer or CharTokenizer.
        generator: TextGenerator inference engine.
        model_info: Structured ModelInfoResponse dataclass.
        lock: Asyncio concurrency lock serializing inference calls.
    """

    def __init__(self) -> None:
        self.model: Optional[GroundUpGPT] = None
        self.tokenizer: Optional[Union[BPETokenizer, CharTokenizer]] = None
        self.generator: Optional[TextGenerator] = None
        self.model_info: Optional[ModelInfoResponse] = None
        self.device_name: str = "cpu"
        self.lock: asyncio.Lock = asyncio.Lock()
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are successfully initialized and ready."""
        return self._is_loaded

    def load_model(
        self,
        checkpoint_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        device_name: str = "auto",
        model_name: Optional[str] = None,
    ) -> None:
        """Load trained model and matching tokenizer strictly from local trusted artifact paths.

        Note: PyTorch checkpoints contain pickled structures; only load checkpoints created
        by trusted local GroundUpGPT runs.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            tokenizer_path: Path to tokenizer JSON file.
            device_name: 'auto', 'cuda', or 'cpu'.

        Raises:
            FileNotFoundError: If checkpoint or tokenizer file does not exist.
            ValueError: If checkpoint metadata fails architectural or tokenizer validation.
        """
        ckpt_path = Path(checkpoint_path).resolve()
        tok_path = Path(tokenizer_path).resolve()

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{ckpt_path}'. "
                f"Please generate the demo artifacts first by running: python scripts/generate_demo.py"
            )

        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file not found at '{tok_path}'. "
                f"Please generate the demo artifacts first by running: python scripts/generate_demo.py"
            )

        # 1. Device Resolution
        if device_name == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device_name

        logger.info(f"Loading checkpoint from '{ckpt_path}' onto {target_device}...")

        # 2. Checkpoint Loading (trusted local artifact)
        checkpoint: Dict[str, Any] = torch.load(
            ckpt_path, map_location=target_device, weights_only=False
        )

        # 3. Model Architecture Reconstruction
        saved_model_cfg = checkpoint.get("model_config")
        if not saved_model_cfg:
            raise ValueError(f"Checkpoint at '{ckpt_path}' lacks required 'model_config' metadata.")

        gpt_config = GPTConfig(**saved_model_cfg)
        model = GroundUpGPT(gpt_config)

        # 4. Strict State Dict Restoration
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(target_device)
        model.eval()

        # 5. Tokenizer Restoration & Validation
        logger.info(f"Loading tokenizer from '{tok_path}'...")
        saved_tok_type = checkpoint.get("tokenizer_type", "BPETokenizer")
        if saved_tok_type == "BPETokenizer":
            tokenizer = BPETokenizer.load(tok_path)
        elif saved_tok_type == "CharTokenizer":
            raise NotImplementedError("CharTokenizer file loading not supported via JSON path.")
        else:
            raise ValueError(f"Unknown tokenizer type '{saved_tok_type}' in checkpoint.")

        # Validate tokenizer against checkpoint metadata
        saved_vocab_size = checkpoint.get("tokenizer_vocab_size")
        if saved_vocab_size is not None and tokenizer.vocab_size != saved_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size mismatch: checkpoint trained with {saved_vocab_size}, "
                f"but loaded tokenizer has {tokenizer.vocab_size}."
            )

        saved_fp = checkpoint.get("tokenizer_fingerprint")
        curr_fp = get_tokenizer_fingerprint(tokenizer)
        if saved_fp is not None and saved_fp != curr_fp:
            raise ValueError(
                f"Tokenizer fingerprint mismatch: checkpoint requires {saved_fp[:16]}..., "
                f"loaded tokenizer produced {curr_fp[:16]}..."
            )

        # Ensure model vocab_size exactly matches loaded tokenizer's vocabulary size
        if gpt_config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"Model architecture vocab_size ({gpt_config.vocab_size}) does not match "
                f"loaded tokenizer vocabulary size ({tokenizer.vocab_size})."
            )

        # 6. Initialize TextGenerator Engine
        generator = TextGenerator(model, tokenizer, device=target_device)

        # 7. Construct Metadata Summary
        display_profile = model_name
        if display_profile is None:
            if gpt_config.vocab_size == 4096 and gpt_config.num_layers == 5:
                display_profile = "GroundUpGPT-5M (5.05M)"
            elif gpt_config.vocab_size <= 512 and gpt_config.num_layers <= 3:
                display_profile = "GroundUpGPT-Demo (35K)"
            else:
                display_profile = f"GroundUpGPT ({model.get_num_params():,} params)"

        model_info = ModelInfoResponse(
            model_name="GroundUpGPT",
            profile_name=display_profile,
            vocab_size=gpt_config.vocab_size,
            block_size=gpt_config.block_size,
            embed_dim=gpt_config.embed_dim,
            num_heads=gpt_config.num_heads,
            num_layers=gpt_config.num_layers,
            weight_tying=gpt_config.weight_tying,
            total_parameters=model.get_num_params(),
            tokenizer_type=saved_tok_type,
            tokenizer_vocab_size=tokenizer.vocab_size,
            tokenizer_fingerprint=curr_fp,
            device=target_device,
            checkpoint_step=checkpoint.get("step"),
            checkpoint_val_loss=checkpoint.get("val_loss"),
        )

        # 8. Atomically assign loaded components only after all validations pass
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.model_info = model_info
        self.device_name = target_device
        self._is_loaded = True
        logger.info(
            f"Successfully initialized GroundUpGPT ({model.get_num_params():,} parameters) "
            f"on {target_device}."
        )

    def unload_model(self) -> None:
        """Unload current model and reset manager state."""
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_info = None
        self._is_loaded = False
        self.device_name = "cpu"


# Global singleton instance managing model memory
model_manager = ModelManager()

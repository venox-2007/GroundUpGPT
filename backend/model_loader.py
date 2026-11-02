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
        self.active_profile_key: Optional[str] = None
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
        active_profile_key: Optional[str] = None,
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
        if saved_tok_type.lower() in ("bpe", "bpetokenizer"):
            tokenizer = BPETokenizer.load(tok_path)
        elif saved_tok_type.lower() in ("char", "chartokenizer"):
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
        is_anchor_sft = checkpoint.get("anchor_sft_step") is not None
        legacy_sft_step = checkpoint.get("sft_step")
        is_sft = is_anchor_sft or (legacy_sft_step is not None)
        model_mode = "chat" if is_sft else "completion"

        anchor_sft_steps = checkpoint.get("anchor_sft_step")
        if is_anchor_sft:
            pretrain_steps = checkpoint.get("source_pretrain_step")
            original_sft_steps = checkpoint.get("prev_sft_step")
            source_step = pretrain_steps
            sft_step = (original_sft_steps or 0) + (anchor_sft_steps or 0)
            global_step = checkpoint.get("global_step")
            checkpoint_step = global_step
            source_val_loss = checkpoint.get("source_val_loss")

            anchor_val_loss = checkpoint.get("anchor_val_loss")
            anchor_val_perplexity = checkpoint.get("anchor_val_ppl")
            dolly_val_loss = checkpoint.get("dolly_val_loss")
            dolly_val_perplexity = checkpoint.get("dolly_val_ppl")
            transfer_test_loss = checkpoint.get("transfer_test_loss")
            transfer_test_perplexity = checkpoint.get("transfer_test_ppl")
            lm_val_loss = checkpoint.get("pretrain_lm_loss")
            lm_val_perplexity = checkpoint.get("pretrain_lm_ppl")
            sft_val_loss = anchor_val_loss
            sft_val_perplexity = anchor_val_perplexity
        else:
            pretrain_steps = checkpoint.get("source_step")
            original_sft_steps = legacy_sft_step
            sft_step = legacy_sft_step
            source_step = checkpoint.get("source_step")
            global_step = checkpoint.get("global_step")
            if global_step is None and source_step is not None and sft_step is not None:
                global_step = source_step + sft_step
            elif global_step is None:
                global_step = checkpoint.get("step")

            checkpoint_step = checkpoint.get("step")
            if checkpoint_step is None and is_sft:
                checkpoint_step = global_step

            source_val_loss = checkpoint.get("source_val_loss")
            anchor_val_loss = None
            anchor_val_perplexity = None
            dolly_val_loss = checkpoint.get("sft_val_loss") or checkpoint.get("best_sft_val_loss")
            dolly_val_perplexity = checkpoint.get("sft_val_perplexity") or checkpoint.get("best_sft_val_perplexity")
            transfer_test_loss = None
            transfer_test_perplexity = None
            lm_val_loss = checkpoint.get("lm_benchmark_val_loss")
            lm_val_perplexity = checkpoint.get("lm_benchmark_perplexity")
            sft_val_loss = dolly_val_loss
            sft_val_perplexity = dolly_val_perplexity

        display_profile = model_name
        num_params = model.get_num_params()
        if display_profile is None:
            if is_anchor_sft:
                display_profile = f"GroundUpGPT-27M Anchors (Step +{anchor_sft_steps})"
            elif is_sft:
                if num_params > 20_000_000:
                    display_profile = f"GroundUpGPT-27M SFT (Step {sft_step})"
                else:
                    display_profile = f"GroundUpGPT-5M SFT (Step {sft_step})"
            elif num_params > 20_000_000:
                display_profile = "GroundUpGPT-27M"
            elif gpt_config.vocab_size == 4096 and gpt_config.num_layers == 5:
                display_profile = "GroundUpGPT-5M (5.05M)"
            elif gpt_config.vocab_size <= 512 and gpt_config.num_layers <= 3:
                display_profile = "GroundUpGPT-Demo (35K)"
            else:
                display_profile = f"GroundUpGPT ({num_params:,} params)"

        model_info = ModelInfoResponse(
            model_name="GroundUpGPT",
            profile_name=display_profile,
            active_profile=active_profile_key,
            vocab_size=gpt_config.vocab_size,
            block_size=gpt_config.block_size,
            embed_dim=gpt_config.embed_dim,
            num_heads=gpt_config.num_heads,
            num_layers=gpt_config.num_layers,
            weight_tying=gpt_config.weight_tying,
            total_parameters=num_params,
            tokenizer_type=saved_tok_type,
            tokenizer_vocab_size=tokenizer.vocab_size,
            tokenizer_fingerprint=curr_fp,
            device=target_device,
            checkpoint_step=checkpoint_step,
            checkpoint_val_loss=checkpoint.get("val_loss") or sft_val_loss,
            original_val_loss=checkpoint.get("original_val_loss"),
            expanded_val_loss=checkpoint.get("expanded_val_loss"),
            checkpoint_perplexity=checkpoint.get("perplexity") or checkpoint.get("expanded_perplexity") or sft_val_perplexity,
            global_step=global_step,
            sft_step=sft_step,
            source_step=source_step,
            source_val_loss=source_val_loss,
            is_sft=is_sft,
            model_mode=model_mode,
            sft_val_loss=sft_val_loss,
            sft_val_perplexity=sft_val_perplexity,
            lm_val_loss=lm_val_loss,
            lm_val_perplexity=lm_val_perplexity,
            pretrain_steps=pretrain_steps,
            original_sft_steps=original_sft_steps,
            anchor_sft_steps=anchor_sft_steps,
            anchor_val_loss=anchor_val_loss,
            anchor_val_perplexity=anchor_val_perplexity,
            dolly_val_loss=dolly_val_loss,
            dolly_val_perplexity=dolly_val_perplexity,
            transfer_test_loss=transfer_test_loss,
            transfer_test_perplexity=transfer_test_perplexity,
            is_anchor_sft=is_anchor_sft,
        )

        # 8. Atomically assign loaded components only after all validations pass
        old_model = self.model
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.model_info = model_info
        self.active_profile_key = active_profile_key
        self.device_name = target_device
        self._is_loaded = True

        # Release previous model memory
        del old_model
        if target_device == "cuda" or (torch.cuda.is_available() and self.device_name == "cuda"):
            torch.cuda.empty_cache()

        logger.info(
            f"Successfully initialized GroundUpGPT ({num_params:,} parameters) "
            f"on {target_device}."
        )

    def unload_model(self) -> None:
        """Unload current model and reset manager state."""
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_info = None
        self.active_profile_key = None
        self._is_loaded = False
        self.device_name = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global singleton instance managing model memory
model_manager = ModelManager()

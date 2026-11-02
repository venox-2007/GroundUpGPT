"""Pydantic data models for API request and response serialization."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

from backend.config import get_settings

_settings = get_settings()


class GenerateRequest(BaseModel):
    """Parameters for autoregressive text generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=_settings.max_prompt_chars,
        description="Input text prompt to condition autoregressive generation.",
    )
    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=_settings.max_new_tokens_limit,
        description="Maximum number of new tokens to generate past prompt.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Softmax temperature (>0.0 for sampling, 0.0 for greedy argmax).",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k filtering threshold (keeps top k probability tokens).",
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus top-p filtering threshold (cumulative probability mass).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic pseudo-random sampling.",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=5.0,
        description="Optional repetition penalty factor (>1.0 penalizes repeated tokens; None disables).",
    )


class GenerateResponse(BaseModel):
    """Complete generation output and execution telemetry."""

    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    active_context_tokens: int
    context_limit: int
    context_cropped: bool
    elapsed_time_sec: float
    tokens_per_second: float
    stop_reason: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Architectural metadata and checkpoint information for loaded model."""

    model_name: str
    profile_name: Optional[str] = None
    active_profile: Optional[str] = None
    vocab_size: int
    block_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    weight_tying: bool
    total_parameters: int
    tokenizer_type: str
    tokenizer_vocab_size: int
    tokenizer_fingerprint: str
    device: str
    checkpoint_step: Optional[int] = None
    checkpoint_val_loss: Optional[float] = None
    original_val_loss: Optional[float] = None
    expanded_val_loss: Optional[float] = None
    checkpoint_perplexity: Optional[float] = None
    global_step: Optional[int] = None
    sft_step: Optional[int] = None
    source_step: Optional[int] = None
    source_val_loss: Optional[float] = None
    is_sft: bool = False
    model_mode: str = "completion"
    sft_val_loss: Optional[float] = None
    sft_val_perplexity: Optional[float] = None
    lm_val_loss: Optional[float] = None
    lm_val_perplexity: Optional[float] = None
    pretrain_steps: Optional[int] = None
    original_sft_steps: Optional[int] = None
    anchor_sft_steps: Optional[int] = None
    anchor_val_loss: Optional[float] = None
    anchor_val_perplexity: Optional[float] = None
    dolly_val_loss: Optional[float] = None
    dolly_val_perplexity: Optional[float] = None
    transfer_test_loss: Optional[float] = None
    transfer_test_perplexity: Optional[float] = None
    is_anchor_sft: bool = False


class HealthResponse(BaseModel):
    """API health and server operational status."""

    status: str
    model_loaded: bool
    device: str
    version: str


class LoadModelRequest(BaseModel):
    """Parameters for dynamically switching model profiles strictly among trusted predefined profiles."""

    profile: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Target trusted model profile key ('5m_best', '5m_latest', '5m_step200', 'demo', etc.).",
    )

    model_config = {
        "extra": "forbid"
    }


class AssistantRequest(BaseModel):
    """Parameters for user requests submitted to GroundUpGPT assistant router."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=_settings.max_prompt_chars,
        description="User question or input prompt to process.",
    )
    mode: str = Field(
        default="native",
        description="Active chat mode: 'native' (default neural GPT) or 'assisted' (opt-in tool-assisted).",
    )
    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=_settings.max_new_tokens_limit,
        description="Maximum new tokens to generate if neural path is selected.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Softmax temperature for neural sampling.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k filtering threshold for neural sampling.",
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus top-p filtering threshold for neural sampling.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic pseudo-random sampling.",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=5.0,
        description="Optional repetition penalty factor for neural sampling.",
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("native", "assisted"):
            raise ValueError(f"Invalid mode '{v}'. Must be either 'native' or 'assisted'.")
        return v


class AssistantResponse(BaseModel):
    """Structured response from GroundUpGPT assistant router."""

    request_id: str = Field(..., description="Unique request tracking identifier.")
    mode: str = Field(..., description="Active execution mode: 'native' or 'assisted'.")
    origin: str = Field(
        ...,
        description="Execution origin: 'neural', 'safe_calculator', 'local_fact', or 'tool_error'.",
    )
    answer_text: str = Field(..., description="Returned assistant answer or generated text.")
    model_generated: bool = Field(
        ..., description="True if response was produced by neural model, False if tool-computed."
    )
    model_profile: Optional[str] = Field(
        default=None, description="Profile name of the active loaded neural model."
    )
    tool_name: Optional[str] = Field(
        default=None, description="Name of the deterministic tool if tool-computed."
    )
    fact_id: Optional[str] = Field(
        default=None, description="Fact ID if answered from curated local facts."
    )
    provenance: Optional[str] = Field(
        default=None, description="Provenance source of the factual record."
    )
    evaluated_expression: Optional[str] = Field(
        default=None, description="Parsed mathematical expression if evaluated by calculator."
    )
    stop_reason: Optional[str] = Field(
        default=None, description="Generation stop reason if neural model was used."
    )
    prompt_tokens: Optional[int] = Field(
        default=None, description="Prompt token count (0 or null for tools)."
    )
    generated_tokens: Optional[int] = Field(
        default=None, description="Tokens generated by neural model (0 or null for tools)."
    )
    total_tokens: Optional[int] = Field(
        default=None, description="Total tokens processed in neural context."
    )
    elapsed_time_sec: float = Field(..., description="Wall-clock execution time in seconds.")
    tokens_per_second: Optional[float] = Field(
        default=None, description="Generation throughput for neural execution."
    )
    error_code: Optional[str] = Field(
        default=None, description="Error code if request resulted in a tool error or limitation."
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional structured provenance or tool evaluation details."
    )



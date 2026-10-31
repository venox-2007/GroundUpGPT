"""FastAPI application providing local inference and streaming APIs for GroundUpGPT."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import __version__
from backend.config import get_settings
from backend.model_loader import model_manager
from backend.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.generation import GenerationConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown procedures."""
    settings = get_settings()
    logger.info("Initializing GroundUpGPT inference server...")

    if settings.auto_load_on_startup and not model_manager.is_loaded:
        try:
            model_manager.load_model(
                checkpoint_path=settings.checkpoint_path,
                tokenizer_path=settings.tokenizer_path,
                device_name=settings.device,
            )
        except FileNotFoundError as err:
            logger.warning(
                f"Artifact missing: {err}\n"
                "Server running in uninitialized mode. Generate demo artifacts via:\n"
                "  python scripts/generate_demo.py\n"
            )
        except Exception as exc:
            logger.error(f"Failed to load model during startup: {exc}", exc_info=True)

    yield

    logger.info("GroundUpGPT inference server shutting down.")


settings = get_settings()

app = FastAPI(
    title="GroundUpGPT Inference API",
    description="Local inference and Server-Sent Events streaming engine for GroundUpGPT.",
    version=__version__,
    lifespan=lifespan,
)

# Secure CORS configuration restricted to specified local development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def get_health() -> HealthResponse:
    """Return operational server status and model loading state."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        device=model_manager.device_name,
        version=__version__,
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["System"])
async def get_model_info() -> ModelInfoResponse:
    """Return model architecture, parameters, context window, and tokenizer metadata."""
    if not model_manager.is_loaded or model_manager.model_info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please generate demo artifacts first.",
        )
    return model_manager.model_info


@app.post("/api/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Execute complete autoregressive text generation and return all metrics."""
    if not model_manager.is_loaded or model_manager.generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' to generate artifacts.",
        )

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        seed=request.seed,
    )

    try:
        # Concurrency safety: serialize GPU access for local single-user inference
        async with model_manager.lock:
            result = model_manager.generator.generate(prompt=request.prompt, config=config)

        return GenerateResponse(
            prompt=result.prompt,
            generated_text=result.generated_text,
            full_text=result.full_text,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            total_tokens=len(result.all_token_ids),
            active_context_tokens=result.active_context_tokens,
            context_limit=result.context_limit,
            context_cropped=result.context_cropped,
            elapsed_time_sec=round(result.elapsed_time_sec, 4),
            tokens_per_second=round(result.tokens_per_second, 2),
        )
    except Exception as exc:
        logger.error(f"Generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )


@app.post("/api/generate/stream", tags=["Inference"])
async def generate_text_stream(
    request: GenerateRequest,
    raw_request: Request,
) -> StreamingResponse:
    """Stream generated tokens in real-time using standards-compliant Server-Sent Events (SSE)."""
    if not model_manager.is_loaded or model_manager.generator is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' to generate artifacts.",
        )

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        seed=request.seed,
    )

    async def sse_event_stream() -> AsyncIterator[str]:
        prompt_tokens_count = 0
        try:
            prompt_ids = model_manager.generator.tokenizer.encode(request.prompt)
            prompt_tokens_count = len(prompt_ids)
        except Exception as encode_err:
            error_payload = json.dumps({"error": f"Failed to encode prompt: {encode_err}"})
            yield f"event: error\ndata: {error_payload}\n\n"
            return

        generated_tokens_count = 0
        start_time = time.perf_counter()

        try:
            # Concurrency safety: acquire lock for duration of streaming job
            async with model_manager.lock:
                for chunk in model_manager.generator.generate_stream(
                    prompt=request.prompt, config=config
                ):
                    # Check for early client disconnection
                    if await raw_request.is_disconnected():
                        logger.info("Client disconnected during stream. Halting generation.")
                        break

                    generated_tokens_count += 1
                    token_payload = json.dumps(
                        {
                            "token_id": chunk.token_id,
                            "text": chunk.text_chunk,
                            "step": chunk.step,
                            "generated_tokens": generated_tokens_count,
                        }
                    )
                    yield f"event: token\ndata: {token_payload}\n\n"

            # Compute accurate final token metrics
            elapsed_time = time.perf_counter() - start_time
            total_tokens = prompt_tokens_count + generated_tokens_count
            block_size = model_manager.model.config.block_size
            active_context = min(total_tokens, block_size)
            context_cropped = total_tokens > block_size
            tokens_per_sec = (
                generated_tokens_count / max(elapsed_time, 1e-9)
                if generated_tokens_count > 0
                else 0.0
            )

            done_payload = json.dumps(
                {
                    "prompt_tokens": prompt_tokens_count,
                    "generated_tokens": generated_tokens_count,
                    "total_tokens": total_tokens,
                    "active_context_tokens": active_context,
                    "context_limit": block_size,
                    "context_cropped": context_cropped,
                    "elapsed_time_sec": round(elapsed_time, 4),
                    "tokens_per_second": round(tokens_per_sec, 2),
                }
            )
            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as exc:
            logger.error(f"Streaming inference error: {exc}", exc_info=True)
            error_payload = json.dumps({"error": f"Streaming inference error: {str(exc)}"})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

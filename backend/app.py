"""FastAPI application providing local inference and streaming APIs for GroundUpGPT."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import threading
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import __version__
from backend.config import get_settings, MODEL_PROFILES
from backend.model_loader import model_manager
from backend.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LoadModelRequest,
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
                model_name=settings.model_name,
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


@app.get("/api/model/profiles", tags=["System"])
async def get_model_profiles() -> dict:
    """Return available model profiles and which one is active."""
    from pathlib import Path
    available = {}
    for k, v in MODEL_PROFILES.items():
        if k == "5m":
            continue
        available[k] = {
            "name": v["model_name"],
            "checkpoint_path": v["checkpoint_path"],
            "tokenizer_path": v["tokenizer_path"],
            "checkpoint_exists": Path(v["checkpoint_path"]).is_file(),
            "tokenizer_exists": Path(v["tokenizer_path"]).is_file(),
        }
    return {
        "active_model": model_manager.model_info.model_name if model_manager.model_info else None,
        "profiles": available,
    }


@app.post("/api/model/load", response_model=ModelInfoResponse, tags=["System"])
async def load_model_endpoint(req: LoadModelRequest) -> ModelInfoResponse:
    """Dynamically load or switch active model profile at runtime."""
    target_ckpt = req.checkpoint_path
    target_tok = req.tokenizer_path
    target_name = req.model_name

    if req.profile:
        prof_key = req.profile.lower().strip()
        if prof_key not in MODEL_PROFILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown profile '{req.profile}'. Available profiles: {list(MODEL_PROFILES.keys())}",
            )
        prof = MODEL_PROFILES[prof_key]
        target_ckpt = target_ckpt or prof["checkpoint_path"]
        target_tok = target_tok or prof["tokenizer_path"]
        target_name = target_name or prof["model_name"]

    target_ckpt = target_ckpt or settings.checkpoint_path
    target_tok = target_tok or settings.tokenizer_path
    target_device = req.device or settings.device

    async with model_manager.lock:
        try:
            model_manager.load_model(
                checkpoint_path=target_ckpt,
                tokenizer_path=target_tok,
                device_name=target_device,
                model_name=target_name,
            )
        except FileNotFoundError as err:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {exc}",
            )

    if model_manager.model_info is None:
        raise HTTPException(status_code=500, detail="Model info failed to initialize.")
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
        # Offload synchronous PyTorch inference to worker thread while holding model lock
        async with model_manager.lock:
            result = await asyncio.to_thread(
                model_manager.generator.generate,
                prompt=request.prompt,
                config=config,
            )

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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()

        def inference_worker() -> None:
            """Run synchronous token generation in a dedicated thread to unblock event loop."""
            try:
                assert model_manager.generator is not None
                for chunk in model_manager.generator.generate_stream(
                    prompt=request.prompt, config=config
                ):
                    if cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", chunk))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        try:
            # Concurrency safety: acquire lock for duration of streaming job
            async with model_manager.lock:
                worker_thread = threading.Thread(target=inference_worker, daemon=True)
                worker_thread.start()

                try:
                    while True:
                        if await raw_request.is_disconnected():
                            logger.info("Client disconnected during stream. Halting generation.")
                            cancel_event.set()
                            break

                        try:
                            msg_type, payload = await asyncio.wait_for(queue.get(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue

                        if msg_type == "token":
                            chunk = payload
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
                        elif msg_type == "done":
                            break
                        elif msg_type == "error":
                            raise payload  # Handled by outer try-except
                finally:
                    cancel_event.set()
                    worker_thread.join(timeout=1.0)

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

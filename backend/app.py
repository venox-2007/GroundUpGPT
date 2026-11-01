"""FastAPI application providing local inference and streaming APIs for GroundUpGPT."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import queue
import threading
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import __version__
from backend.config import (
    get_settings,
    MODEL_PROFILES,
    PROFILE_ALIASES,
    get_trusted_profile,
)
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
            canonical_startup = PROFILE_ALIASES.get(
                settings.model_profile.lower().strip(),
                settings.model_profile.lower().strip(),
            )
            model_manager.load_model(
                checkpoint_path=settings.checkpoint_path,
                tokenizer_path=settings.tokenizer_path,
                device_name=settings.device,
                model_name=settings.model_name,
                active_profile_key=canonical_startup,
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
    """Return available model profiles and which one is active. Only returns profiles whose files exist."""
    from backend.config import get_available_profiles
    available_map = get_available_profiles()
    profiles_dict = {}
    for k, v in available_map.items():
        profiles_dict[k] = {
            "name": v["model_name"],
            "checkpoint_path": v["checkpoint_path"],
            "tokenizer_path": v["tokenizer_path"],
            "checkpoint_exists": True,
            "tokenizer_exists": True,
            "is_sft": bool(v.get("is_sft", False)),
        }
    return {
        "active_model": model_manager.model_info.profile_name if (model_manager.model_info and model_manager.model_info.profile_name) else (model_manager.model_info.model_name if model_manager.model_info else None),
        "active_profile": model_manager.active_profile_key,
        "profiles": profiles_dict,
    }


@app.post("/api/model/load", response_model=ModelInfoResponse, tags=["System"])
async def load_model_endpoint(req: LoadModelRequest) -> ModelInfoResponse:
    """Dynamically load or switch active model profile at runtime.

    Strictly restricted to predefined trusted profiles in MODEL_PROFILES.
    Arbitrary client-supplied file paths and device overrides are strictly rejected.
    """
    prof = get_trusted_profile(req.profile)
    if not prof:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown profile '{req.profile}'. Untrusted profiles are rejected. Available profiles: {list(MODEL_PROFILES.keys())}",
        )

    async with model_manager.lock:
        try:
            canonical_profile_key = PROFILE_ALIASES.get(
                req.profile.lower().strip(),
                req.profile.lower().strip(),
            )
            model_manager.load_model(
                checkpoint_path=prof["checkpoint_path"],
                tokenizer_path=prof["tokenizer_path"],
                device_name=settings.device,
                model_name=prof["model_name"],
                active_profile_key=canonical_profile_key,
            )
        except FileNotFoundError as err:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(err))
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {exc}",
            )

    if model_manager.model_info is None:
        raise HTTPException(status_code=500, detail="Model info failed to initialize.")
    return model_manager.model_info


@app.post("/api/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(request: GenerateRequest, raw_request: Request) -> GenerateResponse:
    """Execute complete autoregressive text generation and return all metrics.

    Guarantees that inference lock is held until inference is completely finished,
    even if the client request is cancelled or disconnected.
    """
    if not model_manager.is_loaded or model_manager.generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' to generate artifacts.",
        )

    is_sft = bool(model_manager.model_info and getattr(model_manager.model_info, "is_sft", False))

    if is_sft:
        from src.generation import format_sft_prompt
        prompt_to_run = format_sft_prompt(request.prompt)
    else:
        prompt_to_run = request.prompt

    prompt_ids = model_manager.generator.tokenizer.encode(prompt_to_run)
    prompt_tokens_count = len(prompt_ids)
    block_size = model_manager.model.config.block_size if model_manager.model else 256
    if is_sft and prompt_tokens_count >= block_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt length ({prompt_tokens_count} tokens including conversation delimiters) exceeds or equals the model's maximum context window of {block_size} tokens. Please shorten your message.",
        )

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        seed=request.seed,
        repetition_penalty=request.repetition_penalty,
    )

    worker_result = {}
    worker_error = []

    def run_generate() -> None:
        try:
            assert model_manager.generator is not None
            if is_sft:
                from src.generation import generate_sft_response
                res = generate_sft_response(model_manager.generator, prompt=request.prompt, config=config)
            else:
                res = model_manager.generator.generate(prompt=request.prompt, config=config)
            worker_result["result"] = res
        except Exception as exc:
            worker_error.append(exc)

    # Concurrency and cancellation safety: lock must NOT be released while worker thread is active
    async with model_manager.lock:
        thread = threading.Thread(
            target=run_generate,
            name=f"generate-worker-{time.time_ns()}",
            daemon=True,
        )
        thread.start()
        try:
            while thread.is_alive():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Non-streaming generation cancelled; waiting for worker to finish safely.")
            while thread.is_alive():
                await asyncio.sleep(0.01)
            raise

    if worker_error:
        exc = worker_error[0]
        logger.error(f"Generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    result = worker_result["result"]
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


@app.post("/api/generate/stream", tags=["Inference"])
async def generate_text_stream(
    request: GenerateRequest,
    raw_request: Request,
) -> StreamingResponse:
    """Stream generated tokens in real-time using standards-compliant Server-Sent Events (SSE).

    Implements backpressure with a bounded queue and ensures that streaming cancellation
    cooperatively halts generation and never releases the inference lock prematurely.
    """
    if not model_manager.is_loaded or model_manager.generator is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' to generate artifacts.",
        )

    is_sft = bool(model_manager.model_info and getattr(model_manager.model_info, "is_sft", False))

    if is_sft:
        from src.generation import format_sft_prompt
        prompt_to_run = format_sft_prompt(request.prompt)
        stop_seqs = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
    else:
        prompt_to_run = request.prompt
        stop_seqs = None

    try:
        prompt_ids = model_manager.generator.tokenizer.encode(prompt_to_run)
        prompt_tokens_count = len(prompt_ids)
    except Exception as encode_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to encode prompt: {encode_err}",
        )

    block_size = model_manager.model.config.block_size if model_manager.model else 256
    if is_sft and prompt_tokens_count >= block_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt length ({prompt_tokens_count} tokens including conversation delimiters) exceeds or equals the model's maximum context window of {block_size} tokens. Please shorten your message.",
        )

    config = GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        seed=request.seed,
        stop_sequences=stop_seqs,
        repetition_penalty=request.repetition_penalty,
    )

    async def sse_event_stream() -> AsyncIterator[str]:
        generated_tokens_count = 0
        start_time = time.perf_counter()
        sync_queue: queue.Queue = queue.Queue(maxsize=16)
        cancel_event = threading.Event()
        worker_done = threading.Event()

        def inference_worker() -> None:
            """Run synchronous token generation in a dedicated thread with backpressure and cancellation."""
            try:
                assert model_manager.generator is not None
                for chunk in model_manager.generator.generate_stream(
                    prompt=prompt_to_run, config=config
                ):
                    if cancel_event.is_set():
                        break
                    # Bounded queue with backpressure: retry put while monitoring cancel_event
                    while not cancel_event.is_set():
                        try:
                            sync_queue.put(("token", chunk), timeout=0.05)
                            break
                        except queue.Full:
                            continue

                if not cancel_event.is_set():
                    while not cancel_event.is_set():
                        try:
                            sync_queue.put(("done", None), timeout=0.05)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                try:
                    sync_queue.put(("error", exc), timeout=0.1)
                except queue.Full:
                    pass
            finally:
                worker_done.set()

        try:
            # Concurrency safety: acquire lock for duration of streaming job
            async with model_manager.lock:
                worker_thread = threading.Thread(
                    target=inference_worker,
                    name=f"stream-worker-{time.time_ns()}",
                    daemon=True,
                )
                worker_thread.start()

                try:
                    while True:
                        if await raw_request.is_disconnected():
                            logger.info("Client disconnected during stream. Halting generation.")
                            cancel_event.set()
                            break

                        try:
                            msg_type, payload = await asyncio.to_thread(
                                sync_queue.get, True, 0.05
                            )
                        except queue.Empty:
                            if worker_done.is_set() and sync_queue.empty():
                                break
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
                            raise payload
                finally:
                    # 1. Signal cancellation to worker thread
                    cancel_event.set()

                    # 2. Drain any residual items to unblock worker if waiting on put
                    while not sync_queue.empty():
                        try:
                            sync_queue.get_nowait()
                        except queue.Empty:
                            break

                    # 3. Asynchronously wait for worker thread to exit completely without blocking event loop
                    while worker_thread.is_alive():
                        try:
                            await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            while worker_thread.is_alive():
                                await asyncio.sleep(0.01)
                            raise

            # 4. Only once worker thread is completely terminated and lock released:
            # Compute accurate final token metrics
            elapsed_time = time.perf_counter() - start_time
            total_tokens = prompt_tokens_count + generated_tokens_count
            block_size = model_manager.model.config.block_size if model_manager.model else 256
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

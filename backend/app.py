"""FastAPI application providing local inference and streaming APIs for GroundUpGPT."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import queue
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import __version__
from backend.assistant_router import (
    assistant_router,
    MODE_NATIVE,
    MODE_ASSISTED,
    ORIGIN_NEURAL,
    ORIGIN_SAFE_CALCULATOR,
    ORIGIN_LOCAL_FACT,
    ORIGIN_TOOL_ERROR,
    ORIGIN_DOCUMENT_EXCERPT,
    ORIGIN_GROUNDED_NEURAL,
)
from backend.config import (
    get_settings,
    MODEL_PROFILES,
    PROFILE_ALIASES,
    get_trusted_profile,
)
from backend.model_loader import model_manager
from backend.schemas import (
    AssistantRequest,
    AssistantResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LoadModelRequest,
    ModelInfoResponse,
    KnowledgeSourceCreateRequest,
    KnowledgeSourceResponse,
    KnowledgeRetrieveRequest,
    KnowledgeRetrieveResponse,
)
from src.generation import GenerationConfig
from src.user_knowledge import knowledge_store
from src.user_knowledge_retrieval import UserKnowledgeRetriever

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
        stop_reason=getattr(result, "stop_reason", None),
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
        stream_stop_reason = "max_new_tokens"
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
                            if chunk.stop_reason:
                                stream_stop_reason = chunk.stop_reason
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
                    "stop_reason": stream_stop_reason,
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


# ---------------------------------------------------------------------------
# Dedicated Assistant Routing Endpoints (Native vs Tool-Assisted)
# ---------------------------------------------------------------------------

@app.get("/api/assistant/telemetry", tags=["Assistant"])
async def get_assistant_telemetry() -> dict:
    """Return execution telemetry counters for validating zero model calls."""
    return model_manager.get_telemetry_counters()


@app.post("/api/assistant/respond", response_model=AssistantResponse, tags=["Assistant"])
async def assistant_respond(request: AssistantRequest) -> AssistantResponse:
    """Execute conservative deterministic routing to answer via tools or fallback to neural GPT."""
    req_id = f"ast-{uuid.uuid4().hex[:10]}"
    start_time = time.perf_counter()

    # 1. Validate mode strictly
    if request.mode not in (MODE_NATIVE, MODE_ASSISTED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Must be strictly 'native' or 'assisted'.",
        )

    # 2. Knowledge Evidence Mode (0 model calls)
    if request.knowledge_mode == "evidence":
        session = knowledge_store.get_or_create_session(request.knowledge_session_id or "default")
        retriever = UserKnowledgeRetriever.from_session(session)
        k_res = retriever.retrieve(request.prompt)
        elapsed = time.perf_counter() - start_time
        if k_res.status == "relevant_passage" and k_res.chunk:
            ans_text = k_res.chunk.text
            err = None
        elif k_res.status == "conflicting_evidence":
            msg = k_res.conflict_details.get("message") if k_res.conflict_details else "Contradictory records detected."
            ans_text = f"⚠️ Conflicting Evidence Detected across active documents:\n{msg}\n\nPlease check your source documents to resolve the contradiction."
            err = "conflicting_evidence"
        else:
            ans_text = f"No unambiguous evidence passage found in active knowledge documents (status: {k_res.status})."
            err = k_res.error_code or k_res.status

        return AssistantResponse(
            request_id=req_id,
            mode=request.mode,
            origin=ORIGIN_DOCUMENT_EXCERPT,
            answer_text=ans_text,
            model_generated=False,
            model_profile=None,
            tool_name="UserKnowledgeRetriever",
            fact_id=None,
            provenance=k_res.source_display_name,
            evaluated_expression=None,
            stop_reason="tool_complete",
            prompt_tokens=0,
            generated_tokens=0,
            total_tokens=0,
            elapsed_time_sec=round(elapsed, 4),
            tokens_per_second=0.0,
            error_code=err,
            details={
                "source_id": k_res.source_id,
                "source_display_name": k_res.source_display_name,
                "chunk_id": k_res.chunk.chunk_id if k_res.chunk else None,
                "start_char": k_res.chunk.start_char if k_res.chunk else None,
                "end_char": k_res.chunk.end_char if k_res.chunk else None,
                "score": k_res.score,
                "status": k_res.status,
                "conflict_detected": k_res.conflict_detected,
                "conflicting_sources": k_res.conflicting_sources,
                "conflict_details": k_res.conflict_details,
                "candidates": k_res.candidates,
            },
        )

    # 3. Run deterministic router classification (unless grounded knowledge is active and terminal router is not arithmetic)
    decision = assistant_router.classify_and_route(request.prompt, mode=request.mode)

    # If decision is terminal (safe_calculator, local_fact, or tool_error):
    # ZERO model calls! No model lock acquired!
    if decision.is_terminal and request.knowledge_mode != "grounded":
        elapsed = time.perf_counter() - start_time
        return AssistantResponse(
            request_id=req_id,
            mode=request.mode,
            origin=decision.origin,
            answer_text=decision.answer_text or "",
            model_generated=False,
            model_profile=None,
            tool_name=decision.tool_name,
            fact_id=decision.fact_id,
            provenance=decision.provenance,
            evaluated_expression=decision.evaluated_expression,
            stop_reason="tool_complete",
            prompt_tokens=0,
            generated_tokens=0,
            total_tokens=0,
            elapsed_time_sec=round(elapsed, 4),
            tokens_per_second=0.0,
            error_code=decision.error_code,
            details=decision.details,
        )

    # If arithmetic was identified, calculator always takes terminal precedence even in grounded mode
    if decision.is_terminal and decision.origin == ORIGIN_SAFE_CALCULATOR:
        elapsed = time.perf_counter() - start_time
        return AssistantResponse(
            request_id=req_id,
            mode=request.mode,
            origin=decision.origin,
            answer_text=decision.answer_text or "",
            model_generated=False,
            model_profile=None,
            tool_name=decision.tool_name,
            fact_id=decision.fact_id,
            provenance=decision.provenance,
            evaluated_expression=decision.evaluated_expression,
            stop_reason="tool_complete",
            prompt_tokens=0,
            generated_tokens=0,
            total_tokens=0,
            elapsed_time_sec=round(elapsed, 4),
            tokens_per_second=0.0,
            error_code=decision.error_code,
            details=decision.details,
        )

    grounded_chunk = None
    retrieval_details = None
    if request.knowledge_mode == "grounded":
        session = knowledge_store.get_or_create_session(request.knowledge_session_id or "default")
        retriever = UserKnowledgeRetriever.from_session(session)
        k_res = retriever.retrieve(request.prompt)
        if k_res.status == "conflicting_evidence":
            # Hallucination Guardrail: do not feed conflicting facts to neural model
            elapsed = time.perf_counter() - start_time
            msg = k_res.conflict_details.get("message") if k_res.conflict_details else "Contradictory records detected."
            return AssistantResponse(
                request_id=req_id,
                mode=request.mode,
                origin=ORIGIN_DOCUMENT_EXCERPT,
                answer_text=f"⚠️ Grounding Guardrail: Conflicting Evidence Detected:\n{msg}\n\nNeural generation halted to prevent hallucinating between contradictory source documents.",
                model_generated=False,
                model_profile=None,
                tool_name="UserKnowledgeRetriever",
                fact_id=None,
                provenance=k_res.source_display_name,
                evaluated_expression=None,
                stop_reason="tool_complete",
                prompt_tokens=0,
                generated_tokens=0,
                total_tokens=0,
                elapsed_time_sec=round(elapsed, 4),
                tokens_per_second=0.0,
                error_code="conflicting_evidence",
                details={
                    "conflict_detected": True,
                    "conflicting_sources": k_res.conflicting_sources,
                    "conflict_details": k_res.conflict_details,
                    "status": k_res.status,
                    "candidates": k_res.candidates,
                },
            )
        elif k_res.status == "relevant_passage" and k_res.chunk:
            grounded_chunk = k_res.chunk
            retrieval_details = {
                "source_id": k_res.source_id,
                "source_display_name": k_res.source_display_name,
                "chunk_id": k_res.chunk.chunk_id,
                "start_char": k_res.chunk.start_char,
                "end_char": k_res.chunk.end_char,
                "score": k_res.score,
                "status": k_res.status,
                "conflict_detected": False,
                "candidates": k_res.candidates,
            }

    # 4. Neural generation path (Native, Fallback, or Grounded)
    if not model_manager.is_loaded or model_manager.generator is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' or load a checkpoint profile.",
        )

    is_sft = bool(model_manager.model_info and getattr(model_manager.model_info, "is_sft", False))

    context_str = grounded_chunk.text if grounded_chunk else None

    if is_sft:
        from src.generation import format_sft_prompt
        from src.user_knowledge import budget_context_bpe
        if context_str:
            budgeted_context = budget_context_bpe(
                prompt=request.prompt,
                context=context_str,
                tokenizer=model_manager.generator.tokenizer,
                max_total_prompt_tokens=190,
            )
            context_str = budgeted_context
            prompt_to_run = format_sft_prompt(request.prompt, context=budgeted_context)
        else:
            prompt_to_run = format_sft_prompt(request.prompt)
    else:
        prompt_to_run = request.prompt

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
            detail=f"Prompt length ({prompt_tokens_count} tokens) exceeds or equals maximum context of {block_size} tokens.",
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
            model_manager.inference_call_count += 1
            if is_sft:
                from src.generation import generate_sft_response
                res = generate_sft_response(
                    model_manager.generator,
                    prompt=request.prompt,
                    context=context_str,
                    config=config,
                )
            else:
                res = model_manager.generator.generate(prompt=request.prompt, config=config)
            model_manager.tokens_generated_count += res.generated_tokens
            worker_result["result"] = res
        except Exception as exc:
            worker_error.append(exc)

    async with model_manager.lock:
        thread = threading.Thread(
            target=run_generate,
            name=f"ast-generate-{time.time_ns()}",
            daemon=True,
        )
        thread.start()
        try:
            while thread.is_alive():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            while thread.is_alive():
                await asyncio.sleep(0.01)
            raise

    if worker_error:
        exc = worker_error[0]
        logger.error(f"Assistant generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    res = worker_result["result"]
    elapsed = time.perf_counter() - start_time
    total_tokens = res.prompt_tokens + res.generated_tokens
    tok_per_sec = res.generated_tokens / max(res.elapsed_time_sec, 1e-9) if res.generated_tokens > 0 else 0.0

    active_profile = model_manager.active_profile_key or (
        model_manager.model_info.profile_name if model_manager.model_info else "GroundUpGPT"
    )

    final_origin = ORIGIN_GROUNDED_NEURAL if grounded_chunk else ORIGIN_NEURAL
    final_tool_name = "UserKnowledgeRetriever" if grounded_chunk else None
    final_provenance = retrieval_details.get("source_display_name") if retrieval_details else None

    return AssistantResponse(
        request_id=req_id,
        mode=request.mode,
        origin=final_origin,
        answer_text=res.generated_text,
        model_generated=True,
        model_profile=active_profile,
        tool_name=final_tool_name,
        fact_id=None,
        provenance=final_provenance,
        evaluated_expression=None,
        stop_reason=getattr(res, "stop_reason", None) or "max_new_tokens",
        prompt_tokens=res.prompt_tokens,
        generated_tokens=res.generated_tokens,
        total_tokens=total_tokens,
        elapsed_time_sec=round(res.elapsed_time_sec, 4),
        tokens_per_second=round(tok_per_sec, 2),
        error_code=None,
        details=retrieval_details,
    )


@app.post("/api/assistant/stream", tags=["Assistant"])
async def assistant_stream(
    request: AssistantRequest,
    raw_request: Request,
) -> StreamingResponse:
    """Stream assistant response using SSE, delivering tool results directly or streaming neural tokens."""
    req_id = f"ast-{uuid.uuid4().hex[:10]}"
    start_time = time.perf_counter()

    # 1. Validate mode strictly
    if request.mode not in (MODE_NATIVE, MODE_ASSISTED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode '{request.mode}'. Must be strictly 'native' or 'assisted'.",
        )

    # 2. Knowledge Evidence Mode (0 model calls)
    if request.knowledge_mode == "evidence":
        session = knowledge_store.get_or_create_session(request.knowledge_session_id or "default")
        retriever = UserKnowledgeRetriever.from_session(session)
        k_res = retriever.retrieve(request.prompt)
        elapsed = time.perf_counter() - start_time
        if k_res.status == "relevant_passage" and k_res.chunk:
            ans_text = k_res.chunk.text
            err = None
        elif k_res.status == "conflicting_evidence":
            msg = k_res.conflict_details.get("message") if k_res.conflict_details else "Contradictory records detected."
            ans_text = f"⚠️ Conflicting Evidence Detected across active documents:\n{msg}\n\nPlease check your source documents to resolve the contradiction."
            err = "conflicting_evidence"
        else:
            ans_text = f"No unambiguous evidence passage found in active knowledge documents (status: {k_res.status})."
            err = k_res.error_code or k_res.status

        async def evidence_stream() -> AsyncIterator[str]:
            payload = json.dumps({
                "request_id": req_id,
                "mode": request.mode,
                "origin": ORIGIN_DOCUMENT_EXCERPT,
                "answer_text": ans_text,
                "model_generated": False,
                "model_profile": None,
                "tool_name": "UserKnowledgeRetriever",
                "fact_id": None,
                "provenance": k_res.source_display_name,
                "evaluated_expression": None,
                "stop_reason": "tool_complete",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "elapsed_time_sec": round(elapsed, 4),
                "tokens_per_second": 0.0,
                "error_code": err,
                "details": {
                    "source_id": k_res.source_id,
                    "source_display_name": k_res.source_display_name,
                    "chunk_id": k_res.chunk.chunk_id if k_res.chunk else None,
                    "start_char": k_res.chunk.start_char if k_res.chunk else None,
                    "end_char": k_res.chunk.end_char if k_res.chunk else None,
                    "score": k_res.score,
                    "status": k_res.status,
                    "conflict_detected": k_res.conflict_detected,
                    "conflicting_sources": k_res.conflicting_sources,
                    "conflict_details": k_res.conflict_details,
                    "candidates": k_res.candidates,
                },
            })
            yield f"event: tool_result\ndata: {payload}\n\n"
            done_payload = json.dumps({
                "request_id": req_id,
                "mode": request.mode,
                "origin": ORIGIN_DOCUMENT_EXCERPT,
                "model_generated": False,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "active_context_tokens": 0,
                "context_limit": 256,
                "context_cropped": False,
                "elapsed_time_sec": round(elapsed, 4),
                "tokens_per_second": 0.0,
                "stop_reason": "tool_complete",
            })
            yield f"event: done\ndata: {done_payload}\n\n"

        return StreamingResponse(
            evidence_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 3. Run deterministic router classification
    decision = assistant_router.classify_and_route(request.prompt, mode=request.mode)

    # If decision is terminal:
    # ZERO model calls! No model lock acquired!
    if decision.is_terminal and (request.knowledge_mode != "grounded" or decision.origin == ORIGIN_SAFE_CALCULATOR):
        async def tool_event_stream() -> AsyncIterator[str]:
            elapsed = time.perf_counter() - start_time
            tool_payload = json.dumps({
                "request_id": req_id,
                "mode": request.mode,
                "origin": decision.origin,
                "answer_text": decision.answer_text or "",
                "model_generated": False,
                "model_profile": None,
                "tool_name": decision.tool_name,
                "fact_id": decision.fact_id,
                "provenance": decision.provenance,
                "evaluated_expression": decision.evaluated_expression,
                "stop_reason": "tool_complete",
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "elapsed_time_sec": round(elapsed, 4),
                "tokens_per_second": 0.0,
                "error_code": decision.error_code,
                "details": decision.details,
            })
            yield f"event: tool_result\ndata: {tool_payload}\n\n"

            done_payload = json.dumps({
                "request_id": req_id,
                "mode": request.mode,
                "origin": decision.origin,
                "model_generated": False,
                "prompt_tokens": 0,
                "generated_tokens": 0,
                "total_tokens": 0,
                "active_context_tokens": 0,
                "context_limit": 256,
                "context_cropped": False,
                "elapsed_time_sec": round(elapsed, 4),
                "tokens_per_second": 0.0,
                "stop_reason": "tool_complete",
            })
            yield f"event: done\ndata: {done_payload}\n\n"

        return StreamingResponse(
            tool_event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Check grounded guardrail before starting worker thread
    grounded_chunk = None
    retrieval_details = None
    if request.knowledge_mode == "grounded":
        session = knowledge_store.get_or_create_session(request.knowledge_session_id or "default")
        retriever = UserKnowledgeRetriever.from_session(session)
        k_res = retriever.retrieve(request.prompt)
        if k_res.status == "conflicting_evidence":
            # Hallucination Guardrail in streaming mode
            elapsed = time.perf_counter() - start_time
            msg = k_res.conflict_details.get("message") if k_res.conflict_details else "Contradictory records detected."

            async def guardrail_stream() -> AsyncIterator[str]:
                payload = json.dumps({
                    "request_id": req_id,
                    "mode": request.mode,
                    "origin": ORIGIN_DOCUMENT_EXCERPT,
                    "answer_text": f"⚠️ Grounding Guardrail: Conflicting Evidence Detected:\n{msg}\n\nNeural generation halted to prevent hallucinating between contradictory source documents.",
                    "model_generated": False,
                    "model_profile": None,
                    "tool_name": "UserKnowledgeRetriever",
                    "fact_id": None,
                    "provenance": k_res.source_display_name,
                    "evaluated_expression": None,
                    "stop_reason": "tool_complete",
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_time_sec": round(elapsed, 4),
                    "tokens_per_second": 0.0,
                    "error_code": "conflicting_evidence",
                    "details": {
                        "conflict_detected": True,
                        "conflicting_sources": k_res.conflicting_sources,
                        "conflict_details": k_res.conflict_details,
                        "status": k_res.status,
                        "candidates": k_res.candidates,
                    },
                })
                yield f"event: tool_result\ndata: {payload}\n\n"
                done_payload = json.dumps({
                    "request_id": req_id,
                    "mode": request.mode,
                    "origin": ORIGIN_DOCUMENT_EXCERPT,
                    "model_generated": False,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "total_tokens": 0,
                    "active_context_tokens": 0,
                    "context_limit": 256,
                    "context_cropped": False,
                    "elapsed_time_sec": round(elapsed, 4),
                    "tokens_per_second": 0.0,
                    "stop_reason": "tool_complete",
                })
                yield f"event: done\ndata: {done_payload}\n\n"

            return StreamingResponse(
                guardrail_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        elif k_res.status == "relevant_passage" and k_res.chunk:
            grounded_chunk = k_res.chunk
            retrieval_details = {
                "source_id": k_res.source_id,
                "source_display_name": k_res.source_display_name,
                "chunk_id": k_res.chunk.chunk_id,
                "start_char": k_res.chunk.start_char,
                "end_char": k_res.chunk.end_char,
                "score": k_res.score,
                "status": k_res.status,
                "conflict_detected": False,
                "candidates": k_res.candidates,
            }

    # 4. Neural generation streaming path (Native or Grounded)
    if not model_manager.is_loaded or model_manager.generator is None or model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run 'python scripts/generate_demo.py' or load a checkpoint profile.",
        )

    is_sft = bool(model_manager.model_info and getattr(model_manager.model_info, "is_sft", False))

    context_str = grounded_chunk.text if grounded_chunk else None

    if is_sft:
        from src.generation import format_sft_prompt
        from src.user_knowledge import budget_context_bpe
        if context_str:
            budgeted_context = budget_context_bpe(
                prompt=request.prompt,
                context=context_str,
                tokenizer=model_manager.generator.tokenizer,
                max_total_prompt_tokens=190,
            )
            prompt_to_run = format_sft_prompt(request.prompt, context=budgeted_context)
        else:
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
            detail=f"Prompt length ({prompt_tokens_count} tokens) exceeds or equals maximum context of {block_size} tokens.",
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

    active_profile = model_manager.active_profile_key or (
        model_manager.model_info.profile_name if model_manager.model_info else "GroundUpGPT"
    )

    async def sse_event_stream() -> AsyncIterator[str]:
        generated_tokens_count = 0
        stream_stop_reason = "max_new_tokens"
        stream_start_time = time.perf_counter()
        sync_queue: queue.Queue = queue.Queue(maxsize=16)
        cancel_event = threading.Event()
        worker_done = threading.Event()

        def inference_worker() -> None:
            try:
                assert model_manager.generator is not None
                model_manager.inference_call_count += 1
                for chunk in model_manager.generator.generate_stream(
                    prompt=prompt_to_run, config=config
                ):
                    if cancel_event.is_set():
                        break
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
            async with model_manager.lock:
                worker_thread = threading.Thread(
                    target=inference_worker,
                    name=f"ast-stream-{time.time_ns()}",
                    daemon=True,
                )
                worker_thread.start()

                try:
                    while True:
                        if await raw_request.is_disconnected():
                            logger.info("Client disconnected during assistant stream. Halting generation.")
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
                            if chunk.stop_reason:
                                stream_stop_reason = chunk.stop_reason
                            token_payload = json.dumps({
                                "token_id": chunk.token_id,
                                "text": chunk.text_chunk,
                                "step": chunk.step,
                                "generated_tokens": generated_tokens_count,
                            })
                            yield f"event: token\ndata: {token_payload}\n\n"
                        elif msg_type == "done":
                            break
                        elif msg_type == "error":
                            raise payload
                finally:
                    cancel_event.set()
                    while not sync_queue.empty():
                        try:
                            sync_queue.get_nowait()
                        except queue.Empty:
                            break

                    while worker_thread.is_alive():
                        try:
                            await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            while worker_thread.is_alive():
                                await asyncio.sleep(0.01)
                            raise

            model_manager.tokens_generated_count += generated_tokens_count

            elapsed_time = time.perf_counter() - stream_start_time
            total_tokens = prompt_tokens_count + generated_tokens_count
            block_size_val = model_manager.model.config.block_size if model_manager.model else 256
            active_context = min(total_tokens, block_size_val)
            context_cropped = total_tokens > block_size_val
            tokens_per_sec = (
                generated_tokens_count / max(elapsed_time, 1e-9)
                if generated_tokens_count > 0
                else 0.0
            )

            final_origin = ORIGIN_GROUNDED_NEURAL if grounded_chunk else ORIGIN_NEURAL
            final_tool_name = "UserKnowledgeRetriever" if grounded_chunk else None
            final_provenance = retrieval_details.get("source_display_name") if retrieval_details else None

            done_payload = json.dumps({
                "request_id": req_id,
                "mode": request.mode,
                "origin": final_origin,
                "model_generated": True,
                "model_profile": active_profile,
                "tool_name": final_tool_name,
                "provenance": final_provenance,
                "prompt_tokens": prompt_tokens_count,
                "generated_tokens": generated_tokens_count,
                "total_tokens": total_tokens,
                "active_context_tokens": active_context,
                "context_limit": block_size_val,
                "context_cropped": context_cropped,
                "elapsed_time_sec": round(elapsed_time, 4),
                "tokens_per_second": round(tokens_per_sec, 2),
                "stop_reason": stream_stop_reason,
                "details": retrieval_details,
            })
            yield f"event: done\ndata: {done_payload}\n\n"

        except Exception as exc:
            logger.error(f"Assistant streaming error: {exc}", exc_info=True)
            error_payload = json.dumps({"error": f"Assistant streaming error: {str(exc)}"})
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


# ---------------------------------------------------------------------------
# User Knowledge Endpoints (Ingestion, Retrieval, Management)
# ---------------------------------------------------------------------------

@app.post("/api/knowledge/source", response_model=KnowledgeSourceResponse, tags=["Knowledge"])
async def create_knowledge_source(request: KnowledgeSourceCreateRequest) -> KnowledgeSourceResponse:
    """Ingest a user-provided text document or pasted knowledge excerpt."""
    try:
        source = knowledge_store.add_source(
            session_id=request.session_id,
            display_name=request.display_name,
            content=request.content,
            source_type=request.source_type,
        )
        return KnowledgeSourceResponse(
            source_id=source.source_id,
            session_id=source.session_id,
            display_name=source.display_name,
            source_type=source.source_type,
            content_hash=source.content_hash,
            character_count=source.character_count,
            token_count=source.token_count,
            chunk_count=len(source.chunks),
            created_at=source.created_at,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )


@app.get("/api/knowledge/sources", response_model=List[KnowledgeSourceResponse], tags=["Knowledge"])
async def list_knowledge_sources(session_id: str = "default") -> List[KnowledgeSourceResponse]:
    """List all ingested documents in the specified knowledge session."""
    sources = knowledge_store.list_sources(session_id)
    return [
        KnowledgeSourceResponse(
            source_id=s.source_id,
            session_id=s.session_id,
            display_name=s.display_name,
            source_type=s.source_type,
            content_hash=s.content_hash,
            character_count=s.character_count,
            token_count=s.token_count,
            chunk_count=len(s.chunks),
            created_at=s.created_at,
        )
        for s in sources
    ]


@app.delete("/api/knowledge/source/{source_id}", tags=["Knowledge"])
async def delete_knowledge_source(source_id: str, session_id: str = "default") -> dict:
    """Delete a single document from the active session by ID."""
    deleted = knowledge_store.delete_source(session_id, source_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_id}' not found in session '{session_id}'.",
        )
    return {"success": True, "deleted_source_id": source_id}


@app.delete("/api/knowledge/sources", tags=["Knowledge"])
async def clear_knowledge_sources(session_id: str = "default") -> dict:
    """Clear all documents from the active session."""
    knowledge_store.clear_session(session_id)
    return {"success": True, "session_id": session_id}


@app.post("/api/knowledge/retrieve", response_model=KnowledgeRetrieveResponse, tags=["Knowledge"])
async def retrieve_knowledge(request: KnowledgeRetrieveRequest) -> KnowledgeRetrieveResponse:
    """Retrieve relevant chunks from session documents using BM25."""
    session = knowledge_store.get_or_create_session(request.session_id)
    retriever = UserKnowledgeRetriever.from_session(session)
    res = retriever.retrieve(request.query, top_k=request.top_k)
    return KnowledgeRetrieveResponse(
        query=res.query,
        status=res.status,
        chunk_id=res.chunk.chunk_id if res.chunk else None,
        source_id=res.chunk.source_id if res.chunk else None,
        source_display_name=res.source_display_name,
        text=res.chunk.text if res.chunk else None,
        start_char=res.chunk.start_char if res.chunk else None,
        end_char=res.chunk.end_char if res.chunk else None,
        score=res.score,
        is_ambiguous=res.is_ambiguous,
        candidates=res.candidates,
        error_code=res.error_code,
    )


# GroundUpGPT Local API Examples

The FastAPI backend runs locally at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## 1. Health & Server Status Check

Check backend operational status, active inference device (`cuda` or `cpu`), and whether weights are loaded.

```bash
curl -X GET http://127.0.0.1:8000/api/health
```

**Example Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "0.1.0"
}
```

## 2. Assistant Response (JSON)

Submit a prompt to the assistant router. Set `mode` to `"auto"` for intelligent tool routing, `"assisted"` for deterministic tools with neural fallback, or `"native"` for pure neural generation.

```bash
curl -X POST http://127.0.0.1:8000/api/assistant/respond \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 15 * 8?",
    "mode": "auto",
    "temperature": 0.8,
    "max_new_tokens": 64
  }'
```

**Example Response**:
```json
{
  "reply": "120",
  "origin": "safe_calculator",
  "origin_label": "SafeCalculator — Computed",
  "provenance": "AST mathematical expression evaluation",
  "execution_time_ms": 0.42
}
```

## 3. Streaming Assistant Response (Server-Sent Events)

Receive real-time token deltas over SSE. Yields individual token chunks and a terminating metadata payload.

```bash
curl -N -X POST http://127.0.0.1:8000/api/assistant/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello! What can you do?",
    "mode": "native",
    "temperature": 0.8,
    "max_new_tokens": 40
  }'
```

**Stream Output**:
```
data: {"token": "I"}

data: {"token": " am"}

data: {"token": " GroundUpGPT"}

data: {"done": true, "origin": "neural", "generated_tokens": 3, "throughput_tok_per_sec": 128.5}
```

## 4. Pure Autoregressive Completion

Direct completion endpoint bypassing conversational delimiter wrappers:

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The theory of relativity states that",
    "max_new_tokens": 32,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
  }'
```

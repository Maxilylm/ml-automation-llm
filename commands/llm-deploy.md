# /llm-deploy

Deploy LLM applications as API endpoints, Streamlit chat UIs, or Docker containers.

## Usage

```
/llm-deploy [--target api|streamlit|docker] [--model <model>] [--port 8000]
```

- `--target`: deployment target (default: api)
- `--model`: model to serve (auto-detected from project if not specified)
- `--port`: port for API/Streamlit (default: 8000 for API, 8501 for Streamlit)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. Detect project LLM setup:
   - Check for fine-tuned model in `models/`
   - Check for RAG pipeline in `src/rag_pipeline.py`
   - Check for prompt templates in `prompts/`
4. **Credential check — HARD FAIL if missing:**
   ```python
   import os
   needed = []
   if model.startswith("claude"):
       needed = [("ANTHROPIC_API_KEY", "anthropic")]
   elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
       needed = [("OPENAI_API_KEY", "openai")]
   for env_var, pkg in needed:
       if not os.environ.get(env_var):
           raise SystemExit(
               f"ERROR: {env_var} is not set.\n"
               f"Set it with: export {env_var}=<your-key>\n"
               "Deployment aborted — the service would fail at runtime without credentials."
           )
   # Verify the key actually works with a minimal API call before proceeding
   if "ANTHROPIC_API_KEY" in [v for v, _ in needed]:
       import anthropic
       try:
           anthropic.Anthropic().messages.create(
               model=model, max_tokens=5,
               messages=[{"role": "user", "content": "ping"}]
           )
           print(f"✓ ANTHROPIC_API_KEY verified — credential works.")
       except anthropic.AuthenticationError:
           raise SystemExit("ERROR: ANTHROPIC_API_KEY is set but invalid. Check the key value.")
   elif "OPENAI_API_KEY" in [v for v, _ in needed]:
       import openai
       try:
           openai.OpenAI().chat.completions.create(
               model=model, max_tokens=5,
               messages=[{"role": "user", "content": "ping"}]
           )
           print(f"✓ OPENAI_API_KEY verified — credential works.")
       except openai.AuthenticationError:
           raise SystemExit("ERROR: OPENAI_API_KEY is set but invalid. Check the key value.")
   ```
5. Report: detected setup, recommended deployment target

### Stage 1: Application Scaffolding

Based on `--target`:

**API (FastAPI):**
1. Generate `src/app.py`:
   - FastAPI application with CORS middleware
   - `/health` endpoint (GET) — service status
   - `/v1/chat/completions` endpoint (POST) — OpenAI-compatible chat API
   - `/v1/completions` endpoint (POST) — completion API
   - Request/response models with Pydantic
   - Token counting and usage tracking
   - **slowapi rate limiting** (v1.1.0) — per-route limits with `429 Too Many Requests` + `Retry-After` header:
     ```python
     from slowapi import Limiter, _rate_limit_exceeded_handler
     from slowapi.util import get_remote_address
     from slowapi.errors import RateLimitExceeded

     limiter = Limiter(key_func=get_remote_address)
     app = FastAPI()
     app.state.limiter = limiter
     app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

     @app.post("/v1/chat/completions")
     @limiter.limit(f"{config['rate_limit_rpm']}/minute")
     async def chat_completions(request: Request, body: ChatRequest):
         ...
     ```
   - Request logging to `logs/requests.jsonl`
2. Generate `src/llm_service.py`:
   - Model loading (API client or local model)
   - Prompt template application
   - RAG integration (if pipeline exists)
   - Response streaming support
3. Generate `requirements.txt` with deployment dependencies

**Streamlit Chat UI:**
1. Generate `src/streamlit_app.py`:
   - Chat interface with `st.chat_message` / `st.chat_input`
   - System prompt configuration sidebar
   - Model parameter controls (temperature, max_tokens, top_p)
   - Token usage display
   - RAG source display (if pipeline exists)
   - Export conversation button

   **Chat state management:**
   Use the pending_question pattern to prevent duplicate messages from multiple input sources (buttons + chat_input):
   ```python
   if "pending_question" not in st.session_state:
       st.session_state.pending_question = None
   # Buttons and chat_input both write to pending_question
   # Single processing point reads and clears it
   ```
   Never append directly to chat history from multiple input sources. Funnel everything through a single pending state variable.

   **Caching rules:**
   - Cache with `@st.cache_resource`: embedding models, numpy arrays, tokenizers, config dicts
   - NEVER cache: database connections (ChromaDB, SQLite), HTTP client sessions, file handles, anything wrapping an OS file descriptor
   - Reason: `@st.cache_resource` preserves the Python object but not the OS resources it holds. Cached connections produce "Broken pipe" errors across Streamlit reruns.

   **Vector store for Streamlit:**
   For RAG-enabled Streamlit apps with < 10K chunks, use in-memory vector search instead of ChromaDB. Call `search_in_memory()` from `llm_utils`. This eliminates SQLite connection state issues entirely.

   **Multi-pass architecture:**
   For data analytics chatbots, generate a two-pass pipeline:
   - Pass 1: LLM decides if computation is needed based on the question
   - Pass 2: If yes, generates code → sandboxed execution → LLM formats result

   This bridges the gap between static RAG and ad-hoc analytical questions. Choose an inference provider with low latency — two sequential calls must stay under the user's patience threshold (~5 seconds).

2. Generate `.streamlit/config.toml` with theme settings

**Docker:**
1. Generate `Dockerfile`:
   - Multi-stage build (builder + runtime)
   - Python dependencies installation
   - Application copy
   - Health check endpoint
   - Non-root user
2. Generate `docker-compose.yml`:
   - Application service
   - Volume mounts for models and data
   - Environment variable configuration
   - Port mapping
3. Generate `.dockerignore`

### Stage 2: Configuration

1. Generate `config/deploy_config.json`:
   ```json
   {
     "target": "<target>",
     "model": "<model>",
     "port": 8000,
     "max_concurrent_requests": 10,
     "request_timeout_seconds": 60,
     "max_tokens_per_request": 4096,
     "rate_limit_rpm": 60,
     "log_requests": true,
     "cors_origins": ["*"]
   }
   ```
2. If target is `streamlit`, add `streamlit_config` block to `deploy_config.json`:
   ```json
   "streamlit_config": {
     "state_management": "pending_question",
     "cache_resource": ["embedding_models", "numpy_arrays", "tokenizers", "config_dicts"],
     "never_cache": ["db_connections", "http_sessions", "file_handles"],
     "vector_store": "in_memory",
     "vector_store_threshold_chunks": 10000,
     "multi_pass_enabled": false,
     "multi_pass_latency_budget_seconds": 5
   }
   ```
3. Generate `config/model_config.json` with model-specific settings

### Stage 3: Monitoring Setup

1. Generate `src/monitoring.py`:
   - Request count, latency histogram, error rate
   - Token usage tracking (input/output per request)
   - Cost estimation per request
   - Prometheus metrics endpoint (`/metrics`) if API target
2. Generate `src/logging_config.py`:
   - Structured JSON logging
   - Request/response logging (with PII masking)
   - Error logging with stack traces

### Stage 4: Testing

1. Generate `tests/test_api.py` (for API target):
   - Health check test
   - Chat completion test
   - Rate limiting test
   - Error handling test
2. Generate `tests/test_smoke.py`:
   - End-to-end smoke test (send request, verify response)
   - Latency check (response under threshold)

### Stage 4b: Live Inference Verification

**RUN the smoke test** — do not just generate it. After the application code is written:

**For API target:**
```python
import subprocess, time, requests, sys

proc = subprocess.Popen(
    ["uvicorn", "src.app:app", "--host", "127.0.0.1", f"--port", str(port)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(3)  # wait for startup

errors = []
try:
    # 1. Health check
    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
    if r.status_code != 200:
        errors.append(f"Health check failed: HTTP {r.status_code}")
    else:
        print("✓ /health — OK")

    # 2. Inference call
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }, timeout=30)
    if r.status_code != 200:
        errors.append(f"Inference call failed: HTTP {r.status_code} — {r.text[:200]}")
    else:
        print(f"✓ /v1/chat/completions — OK (response: {r.json()})")
except Exception as e:
    errors.append(f"Connection error: {e}")
finally:
    proc.terminate()
    proc.wait()

if errors:
    print("INFERENCE VERIFICATION FAILED:")
    for err in errors:
        print(f"  - {err}")
    print("Fix the errors above before considering this deployment complete.")
else:
    print("✓ Live inference verified — service starts and responds correctly.")
```

**For Streamlit target:**
```python
import subprocess, time, sys

proc = subprocess.Popen(
    ["streamlit", "run", "src/streamlit_app.py",
     "--server.headless", "true", "--server.port", str(port)],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(5)
returncode = proc.poll()
if returncode is not None:
    stderr = proc.stderr.read().decode()
    print(f"STREAMLIT STARTUP FAILED (exit {returncode}):\n{stderr}")
    print("Fix the errors above before considering this deployment complete.")
else:
    proc.terminate()
    proc.wait()
    print("✓ Streamlit app starts without errors.")
```

**For Docker target:** Run `docker compose up --build -d`, wait 10s, call health endpoint, then `docker compose down`.

If verification fails, re-spawn developer agent with the error output and ask it to fix `src/app.py` (or `src/streamlit_app.py`). Max 2 fix iterations before reporting as failed.

### Stage 5: Report

```python
from ml_utils import save_agent_report
save_agent_report("rag-builder", {
    "status": "completed",
    "target": target,
    "model": model,
    "generated_files": generated_files,
    "port": port,
    "startup_command": startup_cmd,
    "test_command": test_cmd,
    "recommendations": recommendations
})
```

Print: deployment target, generated files list, startup command, test command.
Print startup instructions:
- API: `uvicorn src.app:app --host 0.0.0.0 --port 8000`
- Streamlit: `streamlit run src/streamlit_app.py`
- Docker: `docker compose up --build`

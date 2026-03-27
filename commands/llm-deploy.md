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
   - Check for API keys in environment
4. Report: detected setup, recommended deployment target

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
   - Rate limiting middleware
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
   - Conversation history management
   - System prompt configuration sidebar
   - Model parameter controls (temperature, max_tokens, top_p)
   - Token usage display
   - RAG source display (if pipeline exists)
   - Export conversation button
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
2. Generate `config/model_config.json` with model-specific settings

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

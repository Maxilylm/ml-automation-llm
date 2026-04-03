---
name: llm-deploy
description: "Deploy LLM applications as API endpoints, Streamlit chat UIs, or Docker containers. Includes health checks and monitoring setup. Use: /llm-deploy [--target api|streamlit|docker]"
aliases: [deploy llm, llm api, llm serve, genai deploy]
extends: spark
user_invocable: true
---

# LLM Deploy

Deploy LLM-powered applications to production. Supports FastAPI/Flask REST endpoints, Streamlit chat UI, and Docker containerization. Includes health check endpoints, request/response logging, token usage tracking, rate limiting configuration, and basic monitoring setup.

## When to Use

- You have a working LLM application and need to expose it as a REST API or chat interface.
- You want a containerized deployment with health checks and monitoring out of the box.
- You need to scaffold a Streamlit chat app with proper state management and caching.
- You are preparing a production deployment with logging, rate limiting, and token tracking.

## Streamlit Deployment Patterns

When deploying as a Streamlit chat UI, these patterns prevent common issues:
- **Chat state**: use the `pending_question` pattern — funnel all input sources (buttons, chat_input) through a single state variable to prevent duplicate messages.
- **Caching**: cache embedding models with `@st.cache_resource`, but NEVER cache database connections (ChromaDB, SQLite) — they break across Streamlit reruns.
- **Vector store**: for RAG apps with <10K chunks, use in-memory search (`search_in_memory()`) to avoid SQLite broken pipe errors entirely.
- **Multi-pass**: for analytics chatbots, use a two-pass architecture (retrieve + optional sandboxed code execution) to handle ad-hoc questions that pre-built chunks can't answer.

## Workflow

1. **Env Check** -- Detect project structure, verify dependencies, confirm the LLM integration works locally.
2. **Application Scaffolding** -- Generate deployment target code. For Streamlit: chat UI with pending_question state management, proper caching, and optional code sandbox. For API: FastAPI with `/predict` and `/health`. For Docker: multi-stage build.
3. **Configuration** -- Set up environment variables, rate limiting, CORS, logging, caching rules, and Streamlit config.
4. **Deploy** -- Start locally and smoke test. Verify Streamlit chat handles multiple input sources without duplicates.
5. **Report** -- Deployment summary with endpoint URLs, configuration notes, production checklist.

## Report Bus Integration

The deployment workflow publishes deployment metadata for operational tracking:

```python
from ml_utils import save_agent_report

save_agent_report("llm_deployer", {
    "stage": "llm-deploy",
    "target": "api",
    "framework": "fastapi",
    "endpoints": ["/predict", "/health"],
    "docker_image": "llm-app:latest",
    "port": 8000,
    "status": "deployed"
})
```

## Full Specification

See `commands/llm-deploy.md` for the complete workflow.

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
- You need to scaffold a Streamlit demo for stakeholder review or internal testing.
- You are preparing a production deployment with logging, rate limiting, and token tracking.

## Workflow

1. **Env Check** -- Detect project structure, verify dependencies (fastapi, streamlit, docker), confirm the LLM integration works locally.
2. **Application Scaffolding** -- Generate the deployment target code: FastAPI app with `/predict` and `/health` endpoints, Streamlit chat UI, or Dockerfile with multi-stage build. Wire in the existing LLM pipeline.
3. **Configuration** -- Set up environment variables, rate limiting, CORS, logging format, and token usage tracking. Generate a `.env.example` with required keys.
4. **Deploy** -- For API: start the server locally and run a smoke test. For Docker: build the image and verify the container starts. For Streamlit: launch the app and confirm the chat interface renders.
5. **Report** -- Produce a deployment summary with endpoint URLs, container image details, configuration notes, and production checklist. Save to report bus.

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

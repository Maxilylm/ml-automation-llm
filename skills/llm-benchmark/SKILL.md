---
name: llm-benchmark
description: "Benchmark LLM performance on standard tasks (QA, summarization, classification). Compare models on quality, latency, and cost. Use: /llm-benchmark <model> [--tasks qa,summarization,classification]"
aliases: [benchmark llm, compare llm, llm comparison, model benchmark]
extends: spark
user_invocable: true
---

# LLM Benchmark

Benchmark one or more LLM models on standard NLP tasks including question answering, summarization, classification, and code generation. Measures quality metrics, latency, token usage, and cost per task. Produces comparison tables and recommendations for model selection.

## When to Use

- You need to choose between LLM providers or model sizes for a specific use case.
- You want latency and cost data alongside quality metrics before committing to a model.
- You are evaluating a new model release against your current production model.
- You need a structured benchmark report for stakeholder review.

## Real API Call Support (v1.1.0)

Benchmarks now make real API calls to LLM providers. Configure API keys in the environment or project config:

```json
// config/llm_benchmark_config.json
{
  "providers": {
    "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "base_url": null},
    "openai": {"api_key_env": "OPENAI_API_KEY", "base_url": null},
    "local": {"api_key_env": null, "base_url": "http://localhost:8000/v1"}
  },
  "defaults": {
    "max_samples": 50,
    "timeout_seconds": 60,
    "max_retries": 3,
    "retry_delay_seconds": 5
  }
}
```

Model identifiers are resolved to providers automatically:
- `claude-*` → Anthropic SDK (`anthropic.Anthropic`)
- `gpt-*` → OpenAI SDK (`openai.OpenAI`)
- `http://...` or `localhost` → OpenAI-compatible local endpoint

If an API key is missing for a provider, skip that model with a warning instead of failing the entire benchmark.

## Workflow

1. **Env Check** -- Verify API keys for each model under test (check env vars per provider config), confirm task datasets are available, install metric dependencies (`anthropic`, `openai` SDKs as needed).
2. **Task Setup** -- Load or generate evaluation datasets for each requested task (QA pairs, summarization sources, classification examples). Configure sampling and batch sizes.
3. **Run Benchmarks** -- For each model-task combination: send **real API requests** to the provider, measure wall-clock latency, record token counts from the API response, compute quality metrics (accuracy, BLEU, ROUGE, F1). Handle rate limits with exponential backoff and retries.
4. **Comparison** -- Build cross-model comparison tables ranked by quality, latency, and cost-per-1K-tokens. Highlight Pareto-optimal models.
5. **Report** -- Generate a benchmark report with tables, charts, and a recommendation summary. Save to report bus.

## Report Bus Integration

The llm-evaluator agent publishes benchmark results for model selection decisions:

```python
from ml_utils import save_agent_report

save_agent_report("llm_evaluator", {
    "stage": "llm-benchmark",
    "models_tested": ["claude-sonnet", "gpt-4o-mini"],
    "tasks": ["qa", "summarization"],
    "results": {
        "claude-sonnet": {"qa_accuracy": 0.88, "latency_p50_ms": 320, "cost_per_1k": 0.015},
        "gpt-4o-mini": {"qa_accuracy": 0.84, "latency_p50_ms": 280, "cost_per_1k": 0.010}
    },
    "recommended": "claude-sonnet",
    "status": "complete"
})
```

## Full Specification

See `commands/llm-benchmark.md` for the complete workflow.

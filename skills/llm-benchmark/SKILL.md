---
name: llm-benchmark
description: "Benchmark LLM performance on standard tasks (QA, summarization, classification). Compare models on quality, latency, and cost. Use: /llm-benchmark <model> [--tasks qa,summarization,classification]"
aliases: [benchmark llm, compare llm, llm comparison, model benchmark]
extends: ml-automation
user_invocable: true
---

# LLM Benchmark

Benchmark one or more LLM models on standard NLP tasks including question answering, summarization, classification, and code generation. Measures quality metrics, latency, token usage, and cost per task. Produces comparison tables and recommendations for model selection.

## When to Use

- You need to choose between LLM providers or model sizes for a specific use case.
- You want latency and cost data alongside quality metrics before committing to a model.
- You are evaluating a new model release against your current production model.
- You need a structured benchmark report for stakeholder review.

## Workflow

1. **Env Check** -- Verify API keys for each model under test, confirm task datasets are available, install metric dependencies.
2. **Task Setup** -- Load or generate evaluation datasets for each requested task (QA pairs, summarization sources, classification examples). Configure sampling and batch sizes.
3. **Run Benchmarks** -- For each model-task combination: send requests, measure wall-clock latency, record token counts, compute quality metrics (accuracy, BLEU, ROUGE, F1). Handle rate limits and retries.
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

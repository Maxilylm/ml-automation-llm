# /llm-benchmark

Benchmark LLM performance on standard tasks. Compare models on quality, latency, and cost.

## Usage

```
/llm-benchmark <model> [--tasks qa,summarization,classification] [--dataset <path>] [--compare <model2>]
```

- `model`: model identifier (e.g., `claude-sonnet-4-20250514`, `gpt-4o`, `llama-3-8b`)
- `--tasks`: comma-separated task list (default: all)
- `--dataset`: custom evaluation dataset (JSONL)
- `--compare`: second model for head-to-head comparison

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. **Load provider config** (v1.1.0) — read `config/llm_benchmark_config.json` if present, else use defaults
4. **Verify API keys** for each model under test:
   - `claude-*` models → check `ANTHROPIC_API_KEY` env var, initialize `anthropic.Anthropic()`
   - `gpt-*` models → check `OPENAI_API_KEY` env var, initialize `openai.OpenAI()`
   - `http://` / `localhost` models → verify endpoint responds to `GET /health`
   - If a key is missing, warn and skip that model (don't fail the entire benchmark)
5. Install SDKs if missing: `pip install anthropic openai` as needed
6. If custom `--dataset` provided, validate format

### Stage 1: Task Setup

For each task in `--tasks`:

1. **Question Answering (qa)**
   - Dataset: SQuAD-style (context, question, answer)
   - Metrics: exact match, F1, latency
   - Prompt: context + question, expect extracted answer

2. **Summarization**
   - Dataset: documents with reference summaries
   - Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, latency
   - Prompt: document, expect concise summary

3. **Classification**
   - Dataset: text with labels
   - Metrics: accuracy, macro-F1, per-class precision/recall, latency
   - Prompt: text, expect label from predefined set

4. **Code Generation**
   - Dataset: problem descriptions with test cases
   - Metrics: pass@1, pass@5, latency
   - Prompt: problem description, expect working code

5. **Reasoning**
   - Dataset: multi-step logic problems
   - Metrics: accuracy, chain-of-thought quality, latency
   - Prompt: problem, expect step-by-step solution

### Stage 2: Benchmark Execution

For each task and model:

1. **Send real API requests** (v1.1.0) to each provider:
   - Anthropic: `client.messages.create(model=model, messages=[...], max_tokens=...)`
   - OpenAI: `client.chat.completions.create(model=model, messages=[...])`
   - Local: POST to `{base_url}/v1/chat/completions`
2. Record per-sample: prediction, latency_ms, input_tokens, output_tokens (from API response usage fields)
3. Compute task-specific metrics
4. Calculate cost: `(input_tokens * input_price + output_tokens * output_price) / 1000`
5. Handle errors with exponential backoff: rate limits (wait + retry, max 3 retries), timeouts (skip sample), API errors (log and continue)

### Stage 3: Comparison (if --compare provided)

1. Run same benchmark on second model
2. Compute head-to-head:
   - Win/loss/tie per sample
   - Statistical significance (paired bootstrap test)
   - Quality delta, latency ratio, cost ratio
3. Generate comparison chart (text-based table)

### Stage 4: Report

```python
from ml_utils import save_agent_report
save_agent_report("llm-evaluator", {
    "status": "completed",
    "models_tested": models,
    "tasks": task_results,
    "summary": {
        "best_quality": best_quality_model,
        "best_latency": best_latency_model,
        "best_cost": best_cost_model
    },
    "comparison": comparison_results,
    "recommendations": recommendations
})
```

Write detailed results to `reports/llm_benchmark_results.json`.
Print summary table: model x task matrix with metrics.
Print recommendation: best model per task and overall.

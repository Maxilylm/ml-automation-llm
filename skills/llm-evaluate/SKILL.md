---
name: llm-evaluate
description: "Evaluate LLM outputs with standard metrics (BLEU, ROUGE, BERTScore, faithfulness, hallucination detection). Supports reference-based and reference-free evaluation. Use: /llm-evaluate <predictions_file> [--reference <ref>] [--metrics bleu,rouge,bertscore]"
aliases: [eval llm, llm metrics, llm quality, genai eval]
extends: ml-automation
user_invocable: true
---

# LLM Evaluate

Run comprehensive evaluation on LLM-generated outputs. Computes text generation metrics (BLEU, ROUGE, BERTScore), detects hallucinations against source documents, and scores faithfulness. Supports both reference-based evaluation (comparing against gold answers) and reference-free evaluation (self-consistency, coherence).

## When to Use

- You have LLM-generated predictions and want to measure quality against reference answers.
- You need to detect hallucinations or score faithfulness in generated text.
- You want to compare output quality across prompt variants or model versions.
- You need a reproducible evaluation report before deploying a generative pipeline.

## Workflow

1. **Env Check** -- Verify Python environment, install missing metric libraries (nltk, rouge-score, bert-score), confirm llm_utils is available.
2. **Load Data** -- Read the predictions file and optional reference file. Validate row counts match, detect format (JSONL, CSV, plain text).
3. **Compute Metrics** -- Calculate requested metrics (BLEU, ROUGE-1/2/L, BERTScore). For reference-free mode, run self-consistency and coherence checks. Aggregate corpus-level and per-sample scores.
4. **Report** -- Generate a structured evaluation report with metric tables, worst-sample highlights, and distribution summaries. Save to the report bus.

## Report Bus Integration

The llm-evaluator agent writes its results to the shared report bus so downstream agents (e.g., prompt-engineer) can consume scores:

```python
from ml_utils import save_agent_report

save_agent_report("llm_evaluator", {
    "stage": "llm-evaluate",
    "metrics": {"corpus_bleu": 0.42, "rouge1_f1": 0.61, "bertscore_f1": 0.87},
    "sample_count": 500,
    "worst_samples": [{"idx": 23, "bleu": 0.02}],
    "status": "complete"
})
```

## Full Specification

See `commands/llm-evaluate.md` for the complete workflow.

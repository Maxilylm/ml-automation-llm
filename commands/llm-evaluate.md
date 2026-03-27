# /llm-evaluate

Evaluate LLM outputs with standard metrics. Supports reference-based and reference-free evaluation.

## Usage

```
/llm-evaluate <predictions_file> [--reference <ref_file>] [--metrics bleu,rouge,bertscore] [--source <source_file>]
```

- `predictions_file`: JSONL file with `prediction` field (one per line)
- `--reference`: JSONL file with `reference` field for reference-based metrics
- `--metrics`: comma-separated list (default: all available)
- `--source`: source documents for faithfulness/hallucination evaluation

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. Verify predictions file exists and is valid JSONL
4. If `--reference` provided, verify reference file exists and row counts match

### Stage 1: Load and Validate Data

1. Parse predictions JSONL — extract `prediction` field from each line
2. If `--reference` provided: parse references JSONL — extract `reference` field
3. If `--source` provided: parse source documents for faithfulness evaluation
4. Report: sample count, avg prediction length, preview of first 3 samples

### Stage 2: Reference-Based Metrics (if --reference provided)

1. **BLEU Score**
   - Compute corpus-level BLEU (1-gram through 4-gram)
   - Compute per-sample BLEU scores
   - Report: corpus BLEU, mean per-sample BLEU, min/max

2. **ROUGE Score**
   - Compute ROUGE-1, ROUGE-2, ROUGE-L (precision, recall, F1)
   - Per-sample scores
   - Report: corpus-level F1 for each ROUGE variant

3. **BERTScore**
   - Compute precision, recall, F1 using contextual embeddings
   - Model: `microsoft/deberta-xlarge-mnli` (default) or user-specified
   - Report: mean P/R/F1, distribution stats

4. **Exact Match / F1** (for QA tasks)
   - Token-level F1 and exact match percentage
   - Normalized comparison (lowercase, strip punctuation)

### Stage 3: Reference-Free Metrics

1. **Coherence** — assess logical flow and consistency within each prediction
2. **Fluency** — grammar and naturalness scoring
3. **Self-Consistency** — generate multiple responses, measure agreement

### Stage 4: Faithfulness and Hallucination (if --source provided)

1. Extract claims from each prediction (sentence-level decomposition)
2. For each claim, check entailment against source documents
3. Compute faithfulness ratio: `supported_claims / total_claims`
4. Flag hallucinated claims with source document context
5. Report: faithfulness score, hallucination count, flagged samples

### Stage 5: Toxicity Screening

1. Check each prediction for harmful content categories:
   - Hate speech, harassment, self-harm, sexual content, violence
2. Compute per-category scores and aggregate toxicity score
3. Flag samples exceeding threshold (default: 0.5)

### Stage 6: Report

```python
from ml_utils import save_agent_report
save_agent_report("llm-evaluator", {
    "status": "completed",
    "sample_count": len(predictions),
    "metrics": {
        "bleu": {"corpus": corpus_bleu, "mean": mean_bleu},
        "rouge": {"rouge1_f1": r1, "rouge2_f1": r2, "rougeL_f1": rl},
        "bertscore": {"precision": p, "recall": r, "f1": f1},
        "faithfulness": faithfulness_score,
        "hallucination_count": hallucination_count,
        "toxicity": {"aggregate": agg_score, "flagged_count": flagged}
    },
    "flagged_samples": flagged_samples[:20],
    "recommendations": recommendations
})
```

Write evaluation report to `reports/llm_evaluation_report.json`.

Print summary table with all computed metrics.

# /prompt-engineer

Systematic prompt optimization workflow. Design, iterate, and test prompts.

## Usage

```
/prompt-engineer <task_description> [--model claude|gpt] [--iterations 5] [--eval-dataset <path>]
```

- `task_description`: natural language description of the LLM task
- `--model`: target model family (default: claude)
- `--iterations`: max optimization iterations (default: 5)
- `--eval-dataset`: JSONL with `input` and `expected_output` fields for testing

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. If `--eval-dataset` provided, verify file exists and is valid JSONL

### Stage 1: Task Analysis

1. Parse the task description to identify:
   - Task type (classification, extraction, generation, summarization, QA, code)
   - Input format and expected output format
   - Key constraints and requirements
2. Check for existing prompts in the project (`prompts/`, `*.prompt`, template strings)
3. Report: identified task type, input/output schema, existing prompt inventory

### Stage 2: Baseline Prompt Design

1. Generate an initial prompt template with:
   - **System prompt**: role definition, context, constraints, output format
   - **User prompt template**: input placeholders, instruction framing
   - **Output schema**: expected structure (JSON, markdown, plain text)
2. Apply task-specific best practices:
   - Classification: include label definitions, examples per class
   - Extraction: specify entity types, output JSON schema
   - Summarization: length constraints, focus areas
   - QA: context handling, "I don't know" behavior
   - Code: language, style guide, error handling expectations
3. **Prompt Quality Audit:**
   - Run `audit_system_prompt()` from llm_utils on the generated system prompt
   - Check for common anti-patterns that degrade output quality:
     - "If the context doesn't contain the answer, say so honestly" → causes excessive hedging. LLM says "context doesn't contain X" even when it does.
     - Generic persona ("You are a helpful assistant") → adds no value. Define specific role and domain expertise.
     - Negative-only instructions ("don't hallucinate") without positive alternatives → model doesn't know what to do instead.
     - Vague length instructions ("be concise") → specify exact format: "2-3 sentences" or "max 5 bullet points."
   - Apply fixes for any detected anti-patterns before proceeding to evaluation
   - Key principle: LLMs take instructions literally. Be specific about what NOT to do, but always pair with what TO do instead.
4. Save baseline as `prompts/v1_baseline.prompt`

### Stage 3: Few-Shot Example Selection (if eval-dataset provided)

1. Analyze eval dataset for diversity (input length, topic, difficulty)
2. Select few-shot examples that maximize coverage:
   - K-means clustering on input embeddings
   - Pick one example per cluster
   - Respect token budget (leave room for actual input)
3. Format examples in the prompt template
4. Save as `prompts/v2_fewshot.prompt`

### Stage 4: Iterative Optimization

For each iteration (up to `--iterations`):

1. **Evaluate current prompt** on eval dataset (if provided):
   - Run predictions on all eval samples
   - Compute task-appropriate metrics (accuracy, ROUGE, exact match)
   - Identify failure patterns (common error types)

2. **Generate variant** based on failure analysis:
   - Clarify ambiguous instructions
   - Add constraints to reduce hallucination
   - Adjust few-shot examples for failed categories
   - Try chain-of-thought for reasoning failures
   - Refine output format instructions
   - Run `audit_system_prompt()` on each variant before testing — catch anti-patterns early
   - If the LLM is over-hedging (saying "I don't have information" when it does), check for overly cautious safety instructions and soften them with positive framing

3. **A/B test** variant vs. current best:
   - Run both prompts on eval dataset
   - Compare metrics with bootstrap confidence intervals
   - If variant wins significantly: promote to current best
   - If tie or loss: keep current, try different optimization direction

4. Log iteration: prompt version, metrics, changes made, outcome

### Stage 5: Final Report

```python
from ml_utils import save_agent_report
save_agent_report("prompt-engineer", {
    "status": "completed",
    "task_type": task_type,
    "iterations_run": iterations_run,
    "versions": [
        {"version": "v1", "metrics": {...}, "changes": "baseline"},
        {"version": "v2", "metrics": {...}, "changes": "added few-shot"},
        ...
    ],
    "best_version": best_version,
    "best_metrics": best_metrics,
    "prompt_file": best_prompt_path,
    "recommendations": recommendations
})
```

Write all prompt versions to `prompts/` directory.
Print comparison table of all versions with metrics.
Print the winning prompt in full.

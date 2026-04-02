---
name: prompt-engineer
description: "Systematic prompt optimization workflow. Design, iterate, and A/B test prompts with tracked metrics per version. Use: /prompt-engineer <task_description> [--model claude|gpt] [--iterations 5]"
aliases: [optimize prompt, prompt design, prompt iteration, prompt test]
extends: spark
user_invocable: true
---

# Prompt Engineer

Systematic prompt engineering workflow that designs, optimizes, and tests prompts for LLM applications. Generates prompt variants, runs A/B tests against evaluation datasets, tracks performance metrics per version, and recommends the best-performing prompt with statistical confidence.

## When to Use

- You need to write or improve a prompt for a specific LLM task (classification, extraction, summarization, etc.).
- You want to compare multiple prompt strategies with tracked metrics across iterations.
- You have an evaluation dataset and want data-driven prompt selection instead of guesswork.
- You are switching models (e.g., GPT to Claude) and need to re-optimize prompts for the new backend.

## Workflow

1. **Env Check** -- Verify API keys, confirm target model is accessible, validate evaluation dataset format.
2. **Task Analysis** -- Parse the task description, identify input/output schema, select initial prompt strategies (zero-shot, few-shot, chain-of-thought).
3. **Iteration Loop** -- For each iteration: generate prompt variant, run against evaluation dataset, compute quality metrics (accuracy, BLEU, faithfulness), log results. Applies techniques like self-refinement, example curation, and instruction tuning.
4. **A/B Comparison** -- Rank prompt variants by metric, compute confidence intervals, select the winner.
5. **Report** -- Produce a versioned prompt history with per-iteration metrics, the recommended prompt, and deployment instructions. Save to report bus.

## Report Bus Integration

The prompt-engineer agent publishes iteration results so other agents can audit prompt evolution:

```python
from ml_utils import save_agent_report

save_agent_report("prompt_engineer", {
    "stage": "prompt-engineer",
    "iterations_run": 5,
    "best_prompt_version": 3,
    "best_score": {"accuracy": 0.91, "bleu": 0.48},
    "model": "claude-sonnet",
    "status": "complete"
})
```

## Full Specification

See `commands/prompt-engineer.md` for the complete workflow.

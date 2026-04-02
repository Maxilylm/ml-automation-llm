---
name: llm-finetune
description: "Prepare data and configure fine-tuning jobs for LLMs. Dataset formatting, validation, training configuration, and job monitoring. Use: /llm-finetune <dataset> --base-model <model> [--method lora|full]"
aliases: [finetune llm, fine-tune, lora, llm training]
extends: ml-automation
user_invocable: true
---

# LLM Fine-Tune

Prepare datasets and configure fine-tuning jobs for large language models. Handles data formatting (JSONL with prompt/completion pairs), dataset validation and deduplication, training configuration (LoRA vs full fine-tuning), hyperparameter selection, job submission, and training run monitoring.

## When to Use

- You have a domain-specific dataset and want to adapt a base LLM for better task performance.
- You need to validate and clean a fine-tuning dataset before submitting a training job.
- You want to compare LoRA (parameter-efficient) vs full fine-tuning for your use case.
- You are setting up a reproducible training configuration with tracked hyperparameters.

## Workflow

1. **Env Check** -- Verify Python environment, confirm base model access (API key or local weights), install training dependencies (peft, transformers, trl).
2. **Data Validation** -- Load the dataset, run `validate_finetune_dataset()` from llm_utils. Check format (chat messages or prompt/completion), flag duplicates, empty fields, and token length outliers. Produce a data quality summary.
3. **Training Config** -- Select fine-tuning method (LoRA or full). Generate training configuration with learning rate, batch size, epochs, warmup steps. For LoRA: set rank, alpha, target modules.
4. **Job Submission** -- Submit the training job to the configured backend (OpenAI API, Hugging Face, or local). Record job ID and estimated completion time.
5. **Report** -- Produce a fine-tuning summary with dataset stats, training config, job status, and next-step instructions. Save to report bus.

## Report Bus Integration

The fine-tuning workflow publishes job metadata for tracking and downstream evaluation:

```python
from ml_utils import save_agent_report

save_agent_report("llm_finetuner", {
    "stage": "llm-finetune",
    "base_model": "meta-llama/Llama-3-8B",
    "method": "lora",
    "dataset_samples": 2500,
    "dataset_valid": True,
    "config": {"lr": 2e-4, "epochs": 3, "lora_rank": 16},
    "job_id": "ft-abc123",
    "status": "submitted"
})
```

## Full Specification

See `commands/llm-finetune.md` for the complete workflow.

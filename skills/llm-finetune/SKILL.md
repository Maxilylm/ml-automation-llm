---
name: llm-finetune
description: "Prepare data and configure fine-tuning jobs for LLMs. Dataset formatting, validation, training configuration, and job monitoring."
aliases: [finetune llm, fine-tune, lora, llm training]
extends: ml-automation
user_invocable: true
---

# LLM Fine-Tune

Prepare datasets and configure fine-tuning jobs for large language models. Handles data formatting (JSONL with prompt/completion pairs), dataset validation and deduplication, training configuration (LoRA vs full fine-tuning), hyperparameter selection, job submission, and training run monitoring.

## Full Specification

See `commands/llm-finetune.md` for the complete workflow.

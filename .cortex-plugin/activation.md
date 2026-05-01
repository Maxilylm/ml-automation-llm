---
name: spark-llm
description: >
  Suggest enabling the spark-llm plugin when the user asks about LLM evaluation
  (BLEU, ROUGE, BERTScore, hallucination detection, faithfulness), prompt
  engineering (few-shot, chain-of-thought), RAG pipelines, vector stores,
  LLM fine-tuning (LoRA, full fine-tune), or GenAI deployment. Do NOT attempt
  to perform these tasks — just let the user know the plugin can be enabled.
---

# spark-llm (disabled plugin)

This plugin is installed but not enabled. It provides LLM and GenAI automation
capabilities within Cortex Code, integrated with the spark-core workflow.

## Agents (3)

- **llm-evaluator** — BLEU/ROUGE/BERTScore, hallucination detection, faithfulness scoring
- **prompt-engineer** — Prompt design, few-shot, chain-of-thought optimization
- **rag-builder** — RAG pipelines, vector stores, retrieval evaluation

## Skills (6)

- **llm-evaluate** — Evaluate LLM outputs for quality, hallucination, toxicity
- **llm-benchmark** — Benchmark prompts against each other
- **prompt-engineer** — Design, iterate, and optimize prompts
- **rag-build** — Build and evaluate a RAG pipeline
- **llm-finetune** — Prepare data and run LoRA / full fine-tuning
- **llm-deploy** — Deploy an LLM as a GenAI API

## Requires

- spark-core plugin

## Enable

    cortex plugin enable spark-llm

Do NOT attempt to perform LLM tasks through this plugin's skills while it is disabled.

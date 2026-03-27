---
name: llm-evaluate
description: "Evaluate LLM outputs with standard metrics (BLEU, ROUGE, BERTScore, faithfulness, hallucination detection). Supports reference-based and reference-free evaluation."
aliases: [eval llm, llm metrics, llm quality, genai eval]
extends: ml-automation
user_invocable: true
---

# LLM Evaluate

Run comprehensive evaluation on LLM-generated outputs. Computes text generation metrics (BLEU, ROUGE, BERTScore), detects hallucinations against source documents, and scores faithfulness. Supports both reference-based evaluation (comparing against gold answers) and reference-free evaluation (self-consistency, coherence).

## Full Specification

See `commands/llm-evaluate.md` for the complete workflow.

# spark-llm — Cortex Code Extension

LLM and GenAI automation. Prompt engineering, RAG pipelines, LLM evaluation, fine-tuning workflows, and GenAI deployment. Requires spark-core installed.

## Available Agents

| Agent | When to use |
|---|---|
| `llm-evaluator` | User wants to evaluate LLM outputs, detect hallucinations, compute BLEU/ROUGE/BERTScore, or measure faithfulness |
| `prompt-engineer` | User wants to design prompts, apply few-shot techniques, chain-of-thought, or optimize prompt performance |
| `rag-builder` | User wants to build a RAG pipeline, set up a vector store, or evaluate retrieval quality |

## Available Skills

| Skill | Trigger |
|---|---|
| `/llm-evaluate` | "evaluate LLM outputs", "check hallucinations", "score these completions", "measure faithfulness" |
| `/llm-benchmark` | "benchmark this prompt", "compare prompts", "run LLM benchmark" |
| `/prompt-engineer` | "improve this prompt", "design a prompt", "apply few-shot", "chain-of-thought" |
| `/rag-build` | "build a RAG pipeline", "set up vector search", "create a retrieval system" |
| `/llm-finetune` | "fine-tune this model", "prepare fine-tuning data", "run LoRA fine-tuning" |
| `/llm-deploy` | "deploy this LLM", "serve a language model", "create a GenAI API" |

## Routing

- LLM output quality, hallucinations, metrics → `llm-evaluator`
- Prompt design and optimization → `prompt-engineer`
- Retrieval-augmented generation → `rag-builder`
- Fallback → spark-core orchestrator

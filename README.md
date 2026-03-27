# ml-automation-llm

LLM and GenAI automation extension for [ml-automation](https://github.com/Maxilylm/ml-automation-core).

## Prerequisites

- [ml-automation](https://github.com/Maxilylm/ml-automation-core) core plugin (>= v1.8.0)
- Claude Code CLI
- LLM API access (OpenAI, Anthropic, or local models) for evaluation and deployment commands

## Installation

```bash
claude plugin add /path/to/ml-automation-llm
```

## What's Included

### Agents

| Agent | Purpose | Hooks Into |
|---|---|---|
| `llm-evaluator` | LLM output quality, hallucination detection, BLEU/ROUGE/BERTScore | `after-evaluation` |
| `prompt-engineer` | Prompt design, optimization, A/B testing | `after-eda` |
| `rag-builder` | RAG pipeline construction, vector store setup, retrieval evaluation | *(direct invocation)* |

### Commands

| Command | Purpose |
|---|---|
| `/llm-evaluate` | Evaluate LLM outputs with standard metrics |
| `/prompt-engineer` | Systematic prompt optimization workflow |
| `/rag-build` | Build a RAG pipeline from documents |
| `/llm-benchmark` | Benchmark LLM performance on standard tasks |
| `/llm-finetune` | Prepare data and configure fine-tuning jobs |
| `/llm-deploy` | Deploy LLM applications (API, Streamlit, Docker) |

## Getting Started

```bash
# Evaluate LLM outputs
/llm-evaluate predictions.jsonl --reference gold.jsonl --metrics bleu,rouge,bertscore

# Optimize prompts
/prompt-engineer "classify customer support tickets" --iterations 5 --eval-dataset test.jsonl

# Build a RAG pipeline
/rag-build ./documents/ --vector-store chroma --chunk-size 512

# Benchmark models
/llm-benchmark claude-sonnet-4-20250514 --tasks qa,summarization --compare gpt-4o

# Prepare fine-tuning
/llm-finetune training_data.jsonl --base-model gpt-4o-mini --method lora

# Deploy an LLM app
/llm-deploy --target api --port 8000
```

## How It Integrates

When installed alongside the core plugin:

1. **Automatic routing** -- Tasks mentioning LLM evaluation, prompt engineering, RAG, or GenAI are routed to LLM agents
2. **Core workflow hooks** -- When running `/team-coldstart`:
   - `prompt-engineer` fires at `after-eda` to detect and optimize existing prompts
   - `llm-evaluator` fires at `after-evaluation` to add LLM-specific metrics
3. **Core agent reuse** -- Commands use eda-analyst, developer, ml-theory-advisor from core

## License

MIT

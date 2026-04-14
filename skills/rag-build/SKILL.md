---
name: rag-build
description: "Build a RAG pipeline from documents with smart backend selection (in-memory for <10K chunks, vector DB for larger), domain-aware chunking, optional code sandbox for ad-hoc analytics, and retrieval evaluation. Use this whenever building Q&A systems, knowledge bases, document search, or data chatbots."
aliases: [rag pipeline, rag setup, vector store, knowledge base build, chatbot, qa system]
extends: spark
user_invocable: true
---

# RAG Build

Build a complete Retrieval-Augmented Generation pipeline. Supports three chunking strategies (recursive, sentence, domain-aware), four vector backends (in-memory, FAISS, ChromaDB, Pinecone), optional sandboxed code execution for ad-hoc analytics, and IR metric evaluation.

## When to Use

- You have documents and want a grounded question-answering system.
- You're building a data/analytics chatbot that needs retrieval, computation, AND tool use.
- You need to choose the right vector backend for your scale (in-memory for small KBs, DB for large).
- You want domain-structured chunks that each answer one category of question.

## Embedding Model Selection (v1.1.0)

Use **sentence-transformers** as the default embedding model instead of TF-IDF. Sentence-transformers produce dense semantic vectors that capture meaning ("car" ≈ "automobile"), whereas TF-IDF only matches exact tokens.

| Model | Dimensions | Speed | Quality | When to Use |
|-------|-----------|-------|---------|-------------|
| `all-MiniLM-L6-v2` (default) | 384 | Fast | Good | General-purpose, <10K chunks |
| `all-mpnet-base-v2` | 768 | Medium | Better | When quality matters more than speed |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Alternative to MiniLM |
| TF-IDF fallback | sparse | Instant | Basic | Only if `sentence-transformers` cannot be installed |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
```

Install: `pip install sentence-transformers`. If installation fails (e.g., no torch), fall back to TF-IDF with a warning.

## Key Design Decisions

| Chunk count | Recommended backend | Why |
|-------------|-------------------|-----|
| < 100 | `memory` (default) | Numpy dot product is instant, zero dependencies, no broken pipes in Streamlit |
| 100 - 10K | `faiss` | Fast approximate search, single file, no server |
| > 10K | `chroma` or `pinecone` | Persistent storage, metadata filtering |

For **data/analytics** projects: use `--chunking domain` to build topic-structured chunks instead of splitting by character count. One chunk = one topic = one type of question answered well.

## Workflow

1. **Env Check** -- Verify environment, auto-select backend based on chunk count estimate.
2. **Document Loading** -- Read documents (PDF, MD, TXT, HTML, CSV, JSONL). Parse and normalize.
3. **Chunking** -- Three strategies:
   - `recursive` (default): paragraph > sentence > word splitting
   - `sentence`: sentence-boundary chunking with overlap
   - `domain`: structured chunks via `build_structured_chunks()` — each chunk maps to one question type. Include both percentages AND raw counts in every statistical chunk.
4. **Embedding & Indexing** -- Generate embeddings, build index. In-memory backend stores L2-normalized embeddings and uses `search_in_memory()` for cosine similarity.
5. **Retrieval Evaluation** -- Recall@k, MRR, NDCG@10 via `evaluate_retrieval()`.
6. **Tool-Augmented RAG** -- Register domain-specific tools (query_data, compute_stats, search_docs) via `ToolRegistry`. LLM selects tools, framework executes safely, results fed back. Code sandbox as fallback for ad-hoc queries.
7. **Pipeline Assembly** -- Generate `rag_pipeline.py` with `search()`, `setup_tools()`, `generate()`, `rag()`.
8. **Report** -- Save pipeline metadata to report bus.

## Report Bus Integration

```python
from ml_utils import save_agent_report
save_agent_report("rag-builder", {
    "status": "completed",
    "chunks": {"count": 45, "strategy": "domain"},
    "backend": "memory",
    "has_code_sandbox": True,
    "retrieval_metrics": {"recall_at_5": 0.91, "mrr": 0.85},
})
```

## Full Specification

Usage: `/rag-build <documents_path> [--vector-store memory|chroma|faiss|pinecone] [--chunking recursive|sentence|domain] [--chunk-size 512]`

Agent: **rag-builder**

See `commands/rag-build.md` for the complete workflow.

---
name: rag-build
description: "Build a RAG pipeline from documents. Chunking, embedding, vector store setup, retrieval evaluation, and end-to-end testing. Use: /rag-build <documents_path> [--vector-store chroma|faiss|pinecone] [--chunk-size 512]"
aliases: [rag pipeline, rag setup, vector store, knowledge base build]
extends: ml-automation
user_invocable: true
---

# RAG Build

Build a complete Retrieval-Augmented Generation pipeline from a document collection. Handles document loading, chunking strategy selection, embedding generation, vector store indexing, retrieval evaluation (recall@k, MRR), and end-to-end RAG quality testing with faithfulness scoring.

## When to Use

- You have a corpus of documents and want to build a question-answering system grounded in that corpus.
- You need to choose between vector store backends (Chroma, FAISS, Pinecone) and chunking strategies.
- You want to evaluate retrieval quality with IR metrics before wiring up the generation layer.
- You are adding a knowledge base to an existing LLM application.

## Workflow

1. **Env Check** -- Verify Python environment, install embedding and vector store dependencies, confirm document path exists.
2. **Document Loading** -- Read documents from the specified path (PDF, Markdown, plain text, HTML). Parse and normalize content, extract metadata.
3. **Chunking** -- Split documents using the selected strategy (fixed-size, recursive, sentence-based). Configure chunk size and overlap. Uses `chunk_documents()` from llm_utils.
4. **Embedding** -- Generate vector embeddings for all chunks using the configured model (sentence-transformers or OpenAI). Build the vector store index via `create_embedding_index()`.
5. **Retrieval Evaluation** -- If test queries are available, run recall@k, MRR, and NDCG@10 evaluation using `evaluate_retrieval()`. Flag low-recall queries for chunk tuning.
6. **Report** -- Produce a RAG pipeline summary with chunk stats, index metadata, retrieval scores, and recommended next steps. Save to report bus.

## Report Bus Integration

The rag-builder agent publishes pipeline metadata for downstream consumers:

```python
from ml_utils import save_agent_report

save_agent_report("rag_builder", {
    "stage": "rag-build",
    "document_count": 120,
    "chunk_count": 1843,
    "vector_store": "chroma",
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieval_metrics": {"recall_at_5": 0.82, "mrr": 0.74, "ndcg_at_10": 0.69},
    "status": "complete"
})
```

## Full Specification

See `commands/rag-build.md` for the complete workflow.

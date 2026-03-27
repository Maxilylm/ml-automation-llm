---
name: rag-build
description: "Build a RAG pipeline from documents. Chunking, embedding, vector store setup, retrieval evaluation, and end-to-end testing."
aliases: [rag pipeline, rag setup, vector store, knowledge base build]
extends: ml-automation
user_invocable: true
---

# RAG Build

Build a complete Retrieval-Augmented Generation pipeline from a document collection. Handles document loading, chunking strategy selection, embedding generation, vector store indexing, retrieval evaluation (recall@k, MRR), and end-to-end RAG quality testing with faithfulness scoring.

## Full Specification

See `commands/rag-build.md` for the complete workflow.

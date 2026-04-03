# /rag-build

Build a RAG pipeline from a document collection.

## Usage

```
/rag-build <documents_path> [--vector-store memory|chroma|faiss|pinecone] [--chunk-size 512] [--overlap 50] [--chunking recursive|sentence|domain] [--embedding-model <model>]
```

- `documents_path`: directory containing documents (PDF, TXT, MD, HTML, DOCX)
- `--vector-store`: vector store backend (default: memory)
- `--chunk-size`: target chunk size in tokens (default: 512)
- `--overlap`: chunk overlap in tokens (default: 50)
- `--chunking`: chunking strategy (default: recursive)
- `--embedding-model`: embedding model identifier — provider-agnostic, resolved at runtime

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. Verify documents directory exists and contains readable files
4. Check required packages based on `--vector-store`:
   - memory: `numpy` (no external DB needed)
   - chroma: `chromadb`
   - faiss: `faiss-cpu` or `faiss-gpu`
   - pinecone: `pinecone-client`
5. **Backend recommendation** — after document loading (Stage 1) produces a chunk count estimate, recommend a backend if the user did not explicitly set `--vector-store`:
   - Chunk count < 100 → recommend `memory` (no database needed, zero dependencies)
   - Chunk count 100–10K → recommend `faiss` (fast local ANN search)
   - Chunk count > 10K → recommend `chroma` or `pinecone` (persistent storage with metadata filtering)

### Stage 1: Document Loading

1. Scan `documents_path` for supported file types:
   - `.txt`, `.md` — read directly
   - `.pdf` — extract text (PyPDF2 or pdfplumber)
   - `.html` — parse with BeautifulSoup, extract text
   - `.docx` — extract with python-docx
   - `.csv`, `.jsonl` — load structured data as text records
2. Report: file count by type, total character count, avg document length
3. Flag empty or unreadable files

### Stage 2: Chunking

1. Select chunking strategy based on `--chunking` (or auto-detect from document types):
   - **Recursive** (default): recursive character splitting (paragraph > sentence > word)
   - **Sentence**: sentence-boundary splitting — better for Q&A where answers live in single sentences
   - **Markdown/HTML** (auto): header-aware splitting (preserve section structure)
   - **Structured data** (auto): one record per chunk
   - **Domain**: hand-crafted topic-based chunks for data/ML/analytics projects (see below)
2. Apply `--chunk-size` and `--overlap` parameters
3. Add metadata to each chunk: source file, chunk index, section header (if applicable)
4. Report: total chunks, avg chunk size, chunk size distribution

#### Domain Chunking Strategy

For data, ML, and analytics projects, hand-craft chunks around the questions stakeholders will ask, not around document structure. One chunk = one topic = one type of question answered well.

A single chunk covering many topics dilutes its embedding — it is "about everything" so it matches nothing strongly. Domain chunking solves this by organizing knowledge into tightly scoped units.

When using domain chunking, call `build_structured_chunks()` from `llm_utils` to generate topic-aligned chunks from structured data or analysis outputs.

**Redundant representations are required.** Every statistical chunk must include both percentages AND raw counts. For example: "74.2% survival rate (233 of 314 female passengers)". LLMs are more confident when context provides multiple representations of the same fact — a percentage alone forces the model to guess the denominator, reducing answer quality.

### Stage 3: Embedding Generation

1. Load embedding model (`--embedding-model`)
2. Generate embeddings for all chunks (batch processing)
3. Normalize embeddings (L2 norm)
4. Report: embedding dimension, generation time, tokens processed

### Stage 4: Vector Store Indexing

1. Initialize vector store:
   - **In-Memory** (default): Store L2-normalized embeddings as a numpy array. Query with `search_in_memory()` — cosine similarity via dot product. Zero connection state, zero dependencies, zero broken pipes in Streamlit. Best for knowledge bases under ~10K chunks.
   - **ChromaDB**: create local persistent collection in `vector_stores/chroma/`
   - **FAISS**: build HNSW index, save to `vector_stores/index.faiss`
   - **Pinecone**: create/upsert to cloud index (requires API key)
2. Insert all chunk embeddings with metadata
3. Report: index size, insertion time, vector count

> **ChromaDB + Streamlit caution:** ChromaDB's `PersistentClient` holds an internal SQLite connection. When cached via `@st.cache_resource`, the connection becomes stale across Streamlit reruns. Use the in-memory backend for Streamlit apps, or create a fresh ChromaDB client per request.

### Stage 5: Retrieval Evaluation

1. Generate synthetic test queries from document content (sample 10-20 chunks, create questions)
2. For each query:
   - Embed query
   - Retrieve top-k (k=5, 10, 20)
   - Check if source chunk is in retrieved set
3. Compute metrics:
   - **Recall@5, Recall@10, Recall@20**
   - **MRR** (Mean Reciprocal Rank)
   - **NDCG@10** (Normalized Discounted Cumulative Gain)
4. Report: retrieval metrics table, failure cases

### Stage 5.5: Tool-Augmented RAG Setup

Pure retrieval is insufficient for real-world applications. Users ask questions that need computation, database queries, or external lookups — not just document search. Use `ToolRegistry` from `llm_utils` to give the LLM discrete, safe capabilities.

#### Tool Registry

1. Create a `ToolRegistry` and register domain-specific tools:
   ```python
   from llm_utils import ToolRegistry

   registry = ToolRegistry()

   @registry.tool("query_data", "Run a query on the dataset and return results")
   def query_data(query: str, limit: int = 10) -> dict:
       result = df.query(query).head(limit)
       return {"rows": len(result), "data": result.to_dict("records")}

   @registry.tool("compute_stats", "Compute statistics for a column")
   def compute_stats(column: str) -> dict:
       return {"mean": df[column].mean(), "median": df[column].median(), ...}

   @registry.tool("search_docs", "Search the knowledge base")
   def search_docs(question: str, k: int = 5) -> list:
       # uses search_in_memory() under the hood
       return retrieve(question, k)
   ```

2. Export schemas for the LLM:
   - For models with native function-calling: `registry.get_schemas(format="openai")`
   - For models without: `registry.get_tool_descriptions()` → inject into system prompt

3. Parse tool calls from LLM output with `parse_tool_calls()` — handles JSON blocks, XML tags, and function-call syntax.

#### Agentic Tool Loop

For complex questions, use `run_tool_loop()` — the LLM decides which tool to call, the framework executes it, and feeds the result back for the next turn. Repeats until the LLM produces a final answer.

```python
from llm_utils import run_tool_loop

result = run_tool_loop(
    registry=registry,
    llm_call=my_llm_function,  # any provider — just needs (messages, tools=None) -> str
    messages=[{"role": "user", "content": "What's the average age of 1st class survivors?"}],
    max_iterations=5,
)
print(result["final_response"])  # stakeholder-friendly answer
print(result["tool_calls"])      # audit trail of what tools were used
```

#### Code Sandbox (Fallback)

For ad-hoc computation that doesn't fit a named tool, include a `run_code` tool backed by `create_sandbox()` + `execute_sandboxed()`:
- Restricts builtins to safe operations — no file I/O, no network, no imports beyond allowed list
- DataFrame is deep-copied before execution
- Captured output + errors returned to LLM for formatting

> **Design principle:** Prefer named tools over raw code execution. Named tools have clear schemas, are easier to audit, and give the LLM a better signal of what's available. Use the code sandbox as a fallback for truly novel queries.

### Stage 6: Pipeline Assembly

1. Generate `src/rag_pipeline.py` with:
   - `ingest(documents_path)` — load, chunk, embed, index
   - `search(question, k=5)` — uses in-memory search by default, falls back to configured vector DB
   - `setup_tools(df=None)` — create ToolRegistry with domain-specific tools (query_data, compute_stats, search_docs, and optionally run_code)
   - `generate(question, context)` — call LLM with retrieved context (provider-agnostic)
   - `rag(question)` — end-to-end: search → tool loop (if tools registered) → generate final answer
2. Generate `src/rag_config.json` with all configuration parameters
3. Generate basic test script `tests/test_rag.py`

> **Streamlit integration:** If deploying as a Streamlit app, cache the embedding model with `@st.cache_resource` but never cache database connections. Database clients hold connection state that goes stale across Streamlit reruns.

### Stage 7: Report

```python
from ml_utils import save_agent_report
save_agent_report("rag-builder", {
    "status": "completed",
    "documents": {"count": doc_count, "types": type_counts},
    "chunks": {"count": chunk_count, "avg_size": avg_size},
    "embedding_model": embedding_model,
    "vector_store": vector_store_type,
    "retrieval_metrics": {
        "recall_at_5": r5, "recall_at_10": r10,
        "mrr": mrr, "ndcg_at_10": ndcg
    },
    "pipeline_files": ["src/rag_pipeline.py", "src/rag_config.json"],
    "recommendations": recommendations
})
```

Print summary: document stats, chunk stats, retrieval metrics, generated files.

# /rag-build

Build a RAG pipeline from a document collection.

## Usage

```
/rag-build <documents_path> [--vector-store chroma|faiss|pinecone] [--chunk-size 512] [--overlap 50] [--embedding-model <model>]
```

- `documents_path`: directory containing documents (PDF, TXT, MD, HTML, DOCX)
- `--vector-store`: vector store backend (default: chroma)
- `--chunk-size`: target chunk size in tokens (default: 512)
- `--overlap`: chunk overlap in tokens (default: 50)
- `--embedding-model`: embedding model (default: `all-MiniLM-L6-v2`)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. Verify documents directory exists and contains readable files
4. Check required packages based on `--vector-store`:
   - chroma: `chromadb`
   - faiss: `faiss-cpu` or `faiss-gpu`
   - pinecone: `pinecone-client`

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

1. Select chunking strategy based on document types:
   - **Markdown/HTML**: header-aware splitting (preserve section structure)
   - **Plain text**: recursive character splitting (paragraph > sentence > word)
   - **Structured data**: one record per chunk
2. Apply `--chunk-size` and `--overlap` parameters
3. Add metadata to each chunk: source file, chunk index, section header (if applicable)
4. Report: total chunks, avg chunk size, chunk size distribution

### Stage 3: Embedding Generation

1. Load embedding model (`--embedding-model`)
2. Generate embeddings for all chunks (batch processing)
3. Normalize embeddings (L2 norm)
4. Report: embedding dimension, generation time, tokens processed

### Stage 4: Vector Store Indexing

1. Initialize vector store:
   - **ChromaDB**: create local persistent collection in `vector_stores/chroma/`
   - **FAISS**: build HNSW index, save to `vector_stores/index.faiss`
   - **Pinecone**: create/upsert to cloud index (requires API key)
2. Insert all chunk embeddings with metadata
3. Report: index size, insertion time, vector count

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

### Stage 6: Pipeline Assembly

1. Generate `src/rag_pipeline.py` with:
   - `ingest(documents_path)` — load, chunk, embed, index
   - `query(question, k=5)` — retrieve and format context
   - `generate(question, context)` — call LLM with retrieved context
   - `rag(question)` — end-to-end retrieve + generate
2. Generate `src/rag_config.json` with all configuration parameters
3. Generate basic test script `tests/test_rag.py`

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

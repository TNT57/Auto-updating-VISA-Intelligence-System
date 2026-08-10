# 🏗️ Architecture Documentation

## System Overview

The Auto-Updating 485 Visa Intelligence System is a multi-phase RAG application that combines document ingestion, semantic search, LLM-powered generation, and automated web monitoring.

---

## Phase 1: Core RAG Pipeline ✅

### Data Flow

```
PDFs → Extract Text → Chunk → Embed → Store in ChromaDB
                                            ↓
User Query → Embed → Similarity Search → Top-K Chunks
                                            ↓
                    Chunks + Query → LLM Prompt → Answer + Citations
```

### Components

#### 1. PDF Loader (`src/ingestion/pdf_loader.py`)
- **Library**: pdfplumber
- **Purpose**: Extract text from PDF files with page-level metadata
- **Output**: List of `{text, page_number, source_file}` dictionaries
- **Error handling**: Graceful handling of corrupted/encrypted PDFs

#### 2. Text Chunker (`src/ingestion/text_chunker.py`)
- **Library**: `langchain-text-splitters` RecursiveCharacterTextSplitter
- **Strategy**: Recursive splitting with 1000-char chunks, 200-char overlap
- **Metadata preservation**: Each chunk carries source file + page number
- **Why recursive**: Handles natural document boundaries (paragraphs, sentences)

#### 3. Vector Store Manager (`src/ingestion/vectorstore_manager.py`)
- **Library**: ChromaDB + Sentence-Transformers
- **Model**: `all-mpnet-base-v2` (420MB, 768 dimensions)
- **Distance metric**: cosine (`hnsw:space`), so `1 - distance` is a usable
  0-1 relevance score. Chroma's default squared-L2 ranges 0-4 and would make
  most scores read as 0%.
- **Storage**: Persistent local directory (`database/vectorstore/`)
- **Collection**: Named `visa_485_documents`
- **Operations**: Add chunks, query similarity, get stats, reset, delete by source
- **Changing the model or metric** requires a rebuild:
  `python scripts/initial_setup.py --rebuild`

#### 4. Retriever (`src/retrieval/retriever.py`)
- **Purpose**: Bridge between user queries and vector store
- **Process**: Embed query → ChromaDB similarity search → Format results
- **Returns**: `QueryResult` with relevance scores and source metadata
- **Configurable**: Number of results (default: 5)

#### 5. LLM Client (`src/generation/llm_client.py`)
- **Primary**: Groq API (Llama 3.3 70B) — free, ~500 tokens/sec
- **Fallback**: OpenAI GPT-4o-mini — cheap, high quality
- **Streaming**: Token-by-token response for real-time UX
- **Temperature**: 0.3 for factual accuracy

#### 6. Prompt Templates (`src/generation/prompt_templates.py`)
- **System prompt**: Legal document assistant persona
- **Citation enforcement**: Must cite source document + page number
- **Disclaimer**: Built-in "not legal advice" framing
- **Context window**: Optimized for ~5 chunks of 1000 chars each

### Configuration (`src/utils/config.py`)
- **Pydantic Settings**: Type-safe configuration with validation
- **Environment variables**: Loaded from `.env` file
- **Defaults**: Sensible defaults for all parameters
- **Key settings**: API keys, model names, chunk sizes, file paths

---

## Phase 2: Auto-Update System ✅

### Scraping Architecture

```
GitHub Actions (cron: 2 AM daily)
         │
         ▼
┌─────────────────┐
│  Scraper        │──▶ Fetch monitored URLs
│  (httpx+BS4)    │──▶ Download new/modified PDFs
└────────┬────────┘──▶ Save HTML snapshots
         │
         ▼
┌─────────────────┐
│  Change Detector │──▶ Compare hashes (Level 1)
│  (difflib)       │──▶ Compare text chunks (Level 2)
└────────┬────────┘──▶ Extract & compare fields (Level 3)
         │
         ▼
┌─────────────────┐
│  Classifier     │──▶ 🔴 Critical (processing times, eligibility)
│                 │──▶ 🟡 Important (fees, documents)
└────────┬────────┘──▶ 🟢 Minor (formatting, contacts)
         │
         ▼
   Store in SQLite + Update ChromaDB
         │
         ▼
┌─────────────────┐
│  Alert Manager  │──▶ Discord webhook (severity ≥ ALERT_MIN_SEVERITY)
│  (httpx)        │──▶ Log delivery, mark change notified=2
└─────────────────┘
```

### State persistence

Change detection needs the previous scrape to still exist on the next run.
GitHub Actions runners are ephemeral and `database/` is gitignored, so the
workflow restores `database/changes.db` from `actions/cache` before each run
and saves it after. That one file holds both the page snapshots
(`page_snapshots`) and the PDF ingestion hashes (`documents`), so it is the
only state the pipeline depends on. HTML files under
`data/raw/html_snapshots/` are a local debugging archive only.

### Monitored URLs
- Main 485 visa page
- Processing times page
- Document checklists
- Relevant forms (e.g., Form 1574)

### Change Detection Levels
1. **Level 1 — File hash**: Fast binary comparison (has anything changed?)
2. **Level 2 — Text diff**: Detailed content comparison (what changed?)
3. **Level 3 — Semantic**: Structured field extraction (what does it mean?)

---

## Alert System ✅

- **Discord webhooks**: one message per run, with an embed per change
  (`src/alerts/alert_manager.py`)
- **Severity filter**: `ALERT_MIN_SEVERITY` (default `IMPORTANT`)
- **Deduplication**: `changes.notified` is set to 2 on success, so a re-run
  never alerts on the same change twice. A failed send leaves the change
  pending for the next run.
- **Audit trail**: every attempt is written to `alert_logs` and surfaced on
  the Alerts page

### Not yet implemented
- **Email digests**: SMTP settings exist in `config.py` but no channel uses them
- **Temporal RAG**: time-aware queries ("what were the requirements in
  January?"). `src/temporal/` is an empty placeholder.

---

## Database Schema

### SQLite — Change Tracking

```sql
CREATE TABLE changes (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    severity TEXT,         -- 'CRITICAL', 'IMPORTANT', 'MINOR'
    change_type TEXT,      -- 'content_update', 'content_added', ...
    old_value TEXT,
    new_value TEXT,
    summary TEXT,
    detected_at DATETIME,
    source_url TEXT,
    notified INTEGER       -- 0=no, 1=pending, 2=sent
);

CREATE TABLE page_snapshots (
    id INTEGER PRIMARY KEY,
    url TEXT,
    title TEXT,
    content TEXT,          -- extracted page text, diffed on the next run
    content_hash TEXT,
    scraped_at DATETIME
);

CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    source_url TEXT,
    file_path TEXT,
    file_hash TEXT,        -- skips re-ingesting unchanged PDFs
    version INTEGER,
    ingested_at DATETIME,
    last_modified DATETIME
);

CREATE TABLE alert_logs (
    id INTEGER PRIMARY KEY,
    change_id INTEGER,
    channel TEXT,          -- 'discord'
    status TEXT,           -- 'sent', 'failed'
    sent_at DATETIME,
    error_message TEXT
);
```

### ChromaDB — Vector Store

```
Collection: visa_485_documents (cosine distance)
├── documents: [chunk_text, ...]
├── embeddings: [768-dim vectors, ...]
├── metadatas: [{source, page_number, chunk_index, doc_type}, ...]
└── ids: [uuid, ...]
```

---

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Components receive dependencies, not create them
3. **Graceful Degradation**: System works even if some components fail
4. **Cost Optimization**: Free-tier everything for portfolio project
5. **Local-First**: Embeddings run locally, only LLM needs API
6. **Extensibility**: Easy to add new data sources, LLMs, or alert channels
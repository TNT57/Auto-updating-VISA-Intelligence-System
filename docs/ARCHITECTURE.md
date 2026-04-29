# 🏗️ Architecture Documentation

## System Overview

The Auto-Updating 485 Visa Intelligence System is a multi-phase RAG application that combines document ingestion, semantic search, LLM-powered generation, and automated web monitoring.

---

## Phase 1: Core RAG Pipeline (Current)

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
- **Library**: LangChain RecursiveCharacterTextSplitter
- **Strategy**: Recursive splitting with 1000-char chunks, 200-char overlap
- **Metadata preservation**: Each chunk carries source file + page number
- **Why recursive**: Handles natural document boundaries (paragraphs, sentences)

#### 3. Vector Store Manager (`src/ingestion/vectorstore_manager.py`)
- **Library**: ChromaDB + Sentence-Transformers
- **Model**: `all-MiniLM-L6-v2` (80MB, 384 dimensions)
- **Storage**: Persistent local directory (`database/vectorstore/`)
- **Collection**: Named `visa_485_documents`
- **Operations**: Add chunks, query similarity, get stats, reset

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

## Phase 2: Auto-Update System (Planned)

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
```

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

## Phase 3: Intelligence Layer (Planned)

### Temporal RAG
- Version-controlled document snapshots
- Time-aware queries: "What were the requirements in January?"
- Change timeline with Plotly visualizations

### Alert System
- **Discord webhooks**: Rich embeds with change details
- **Email**: SMTP-based daily/weekly digests
- **Severity filters**: Only get alerts you care about

---

## Database Schema

### SQLite — Change Tracking (Phase 2)

```sql
CREATE TABLE changes (
    id INTEGER PRIMARY KEY,
    detected_at DATETIME,
    source_url TEXT,
    change_type TEXT,      -- 'content', 'pdf', 'field'
    severity TEXT,         -- 'critical', 'important', 'minor'
    old_value TEXT,
    new_value TEXT,
    summary TEXT,
    ai_analysis TEXT
);

CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY,
    captured_at DATETIME,
    source_url TEXT,
    content_hash TEXT,
    file_path TEXT
);
```

### ChromaDB — Vector Store

```
Collection: visa_485_documents
├── documents: [chunk_text, ...]
├── embeddings: [384-dim vectors, ...]
├── metadatas: [{source, page, chunk_index}, ...]
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
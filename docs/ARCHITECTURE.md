# 🏗️ Architecture Documentation

## System Overview

The Auto-Updating 485 Visa Intelligence System is a multi-phase RAG application that combines document ingestion, semantic search, LLM-powered generation, and automated web monitoring.

The deployed Streamlit app serves the chat from a pre-built, committed index;
it never scrapes at runtime. A scheduled GitHub Actions job keeps that index
current.

---

## Phase 1: Core RAG Pipeline ✅

### Data Flow

```
Web pages (browser-rendered) + linked PDFs → Extract Text → Chunk → Embed → Store in ChromaDB
                                                                                  ↓
User Query → Expand into phrasings → Search each → Fuse rankings (RRF) → Top-K Chunks
                                                                                  ↓
          Deduped, stream-labelled chunks + Query → LLM Prompt → Answer → Grounding check
                                                                                  ↓
                                                                       Answer + source links
```

### Components

#### 1. PDF Loader (`src/ingestion/pdf_loader.py`)
- **Library**: pdfplumber
- **Purpose**: Extract text from PDF files with page-level metadata
- **Tables**: Extracted with pdfplumber and appended to the page text as
  markdown rows, so fee and score tables keep their structure
- **Output**: One `DocumentChunk` per page (`content`, `source`, `page_number`,
  `doc_type="pdf"`, plus `total_pages`, `file_hash`, `has_tables` metadata).
  `DocumentChunk` lives in `src/ingestion/models.py` so the retrieval path can
  import it without pdfplumber.
- **Error handling**: A PDF that fails to open is logged and skipped

#### 2. Text Chunker (`src/ingestion/text_chunker.py`)
- **Library**: `langchain-text-splitters` RecursiveCharacterTextSplitter
- **Strategy**: Recursive splitting with 1000-char chunks, 200-char overlap
  (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
- **Metadata preservation**: Each chunk carries source file + page number
- **Why recursive**: Handles natural document boundaries (paragraphs, sentences)

#### 3. Vector Store Manager (`src/ingestion/vectorstore_manager.py`)
- **Library**: ChromaDB + Sentence-Transformers
- **Model**: `all-mpnet-base-v2` (420MB, 768 dimensions), set by `EMBEDDING_MODEL`
- **Distance metric**: cosine (`hnsw:space`), so `1 - distance` is a usable
  0-1 relevance score. Chroma's default squared-L2 ranges 0-4 and would make
  most scores read as 0%.
- **Storage**: Persistent local directory (`database/vectorstore/`), committed
  to the repository so the deployed app ships with its index. A relative
  `CHROMA_PERSIST_DIR` is resolved against the project root.
- **Collection**: Named `visa_485_documents`
- **Operations**: Add chunks (upsert), query similarity, get stats, reset, delete by source
- **Changing the model or metric** requires a rebuild:
  `python scripts/initial_setup.py --rebuild`

#### 4. Ingestion Pipeline (`src/ingestion/pipeline.py`)
- **Purpose**: Shared ingestion steps used by both `scripts/fetch_and_index.py`
  and `scripts/daily_update.py`, so the two cannot drift apart
- **Pages**: `ingest_pages()` chunks each scraped page (`doc_type="webpage"`,
  source = URL) and replaces that URL's previous chunks
- **PDFs**: `ingest_pdfs()` skips a PDF whose hash matches the `documents`
  table, and skips PDFs that do not mention `PDF_RELEVANCE_KEYWORDS`
  (default `485`, `temporary graduate`). That filter keeps generic
  application forms out of the index.

#### 5. Retriever (`src/retrieval/retriever.py`)
- **Purpose**: Bridge between user queries and vector store
- **Process**: Expand query → ChromaDB similarity search per phrasing → fuse
  rankings → format results
- **Query expansion** (`src/retrieval/query_expansion.py`, set by
  `QUERY_EXPANSION`, default `llm`):
  - `llm` — the LLM rewrites the question into 3 phrasings in Home Affairs'
    own wording. Falls back to `synonyms` if the call fails or returns nothing.
  - `synonyms` — a built-in map of migration jargon to official wording
    (`DOMAIN_SYNONYMS`, e.g. "onshore" → "in Australia when you apply").
    No network, no cost.
  - `none` — the single original query.
  - The original question is always kept, plus at most 4 rewrites.
- **Overfetch**: Each phrasing asks Chroma for `n_results × OVERFETCH` (4)
  chunks, so fusion has deeper lists to compare
- **Reciprocal rank fusion**: Each chunk scores `Σ 1/(60 + rank)` across the
  phrasings. A chunk that ranks well for several phrasings beats one that ranks
  first for a single lucky wording. With only one phrasing, Chroma's order is
  used as is.
- **Relevance score**: `1 - distance`, clamped to 0-1, using the best distance
  the chunk got across phrasings
- **Stream labelling**: `RetrievalResult.stream` maps the source URL slug to the
  stream name (e.g. `post-higher-education-work` → "Post-Higher Education Work
  stream"), longest slug first. The fee differs by stream, so naming the stream
  lets the model give the breakdown instead of picking one number.
- **Exact-match dedupe**: `QueryResults.format_for_llm()` merges passages that
  are identical after whitespace normalisation into one, labelled with every
  stream it came from and marked as one shared rule. The match is exact, never
  fuzzy: two fee paragraphs that differ only in the amount must stay separate.
  This changes only what the LLM sees; `get_sources()` still cites every page.
- **Returns**: `QueryResults` with relevance scores and source metadata
- **Configurable**: Number of results (default: 5; a 1-10 slider on the Chat
  page when `PUBLIC_MODE=false`)
- **Measured with**: `scripts/eval_retrieval.py` (hit@1, hit@3, MRR)

#### 6. LLM Client (`src/generation/llm_client.py`)
- **Primary**: Groq API, model set by `GROQ_MODEL` (default
  `openai/gpt-oss-120b`). Configurable because Groq retired the previously
  hardcoded `llama-3.3-70b-versatile` on 2026-08-16, which broke generation.
- **Alternative**: OpenAI, model set by `OPENAI_MODEL` (default `gpt-4o-mini`).
  The provider is picked once at start-up: Groq if `GROQ_API_KEY` is set,
  otherwise OpenAI if `OPENAI_API_KEY` is set. It is not a per-request failover.
- **Streaming**: Token-by-token response for real-time UX
- **Retries**: `generate()` retries up to 3 times with exponential backoff
- **Temperature**: 0.3 for answers (0.0 for grounding, query expansion and
  severity classification)
- **Grounding check**: `verify_grounding()` makes a second call that re-reads
  the answer against the same context and labels it GROUNDED,
  PARTIALLY_GROUNDED or UNGROUNDED (UNKNOWN if the reply cannot be parsed)

#### 7. Prompt Templates (`src/generation/prompt_templates.py`)
- **System prompt**: 485 visa assistant persona; answer only from the provided
  context, and say so when the context does not contain the answer
- **No inline citations**: The model is told not to write citations; the UI
  shows sources from `QueryResults.get_sources()`
- **Answer shape**: Direct answer first, then detail. A value shared by several
  streams is stated once; values that differ are given per stream. Scored by
  `scripts/eval_answers.py` (`src/generation/answer_metrics.py`).
- **Follow-ups**: `FOLLOW_UP_TEMPLATE` adds the chat history
- **Disclaimer**: Not part of the prompt. A permanent "not legal advice" banner
  on the Chat page.
- **Context window**: Sized for ~5 chunks of 1000 chars each
- **Other templates**: `GROUNDING_PROMPT`, `SEVERITY_CLASSIFICATION_PROMPT`.
  `CHANGE_EXPLANATION_TEMPLATE` and `WEEKLY_SUMMARY_TEMPLATE` are defined but
  not used by the pipeline.

### Configuration (`src/utils/config.py`)
- **Pydantic Settings**: Type-safe configuration with validation
- **Environment variables**: Loaded from `.env` file (on Streamlit Cloud,
  `app/bootstrap.py` copies `st.secrets` into the environment first)
- **Defaults**: Sensible defaults for all parameters
- **Key settings**: API keys, model names (`GROQ_MODEL`, `EMBEDDING_MODEL`),
  `QUERY_EXPANSION`, `SCRAPER_BACKEND`, chunk sizes, file paths,
  `PUBLIC_MODE`, `ALERTS_ENABLED`

---

## Phase 2: Auto-Update System ✅

### Scraping Architecture

```
GitHub Actions (cron: 2 AM AEST daily)
         │
         ▼   scripts/daily_update.py --detect-only
┌─────────────────┐
│  Scraper        │──▶ Fetch monitored URLs (pluggable fetch backend)
│  (fetchers.py   │──▶ Extract main content (BeautifulSoup)
│   + BS4)        │──▶ Download new/modified PDFs (httpx, ETag)
└────────┬────────┘──▶ Save HTML snapshots (local debug archive)
         │
         ▼
┌─────────────────┐
│ Change Detector │──▶ Compare content hashes (Level 1)
│ (difflib)       │──▶ Diff lines into change blocks (Level 2)
└────────┬────────┘──▶ Classify severity (Level 3)
         │
         ▼
┌─────────────────┐
│  Classifier     │──▶ 🔴 Critical (processing times, eligibility)
│                 │──▶ 🟡 Important (fees, documents)
└────────┬────────┘──▶ 🟢 Minor (formatting, contacts)
         │
         ▼
   Record changes + save page snapshot in SQLite (changes.db)
         │
         ▼
   changed=true? ──no──▶ stop (no ML dependencies installed, nothing committed)
         │ yes
         ▼
   scripts/fetch_and_index.py → re-embed into ChromaDB
         │
         ▼
   Commit database/vectorstore + database/changes.db → push → Streamlit redeploys

┌─────────────────┐
│  Alert Manager  │──▶ Discord webhook (severity ≥ ALERT_MIN_SEVERITY)
│  (httpx)        │──▶ Log delivery, mark change notified=2
└─────────────────┘    Only when ALERTS_ENABLED=true (off by default and in CI)
```

### Daily flow (`.github/workflows/daily_scrape.yml`)

Cost is the design constraint: the pages change rarely, so the run is split
into a cheap pass that always runs and an expensive pass that runs only when
something changed.

1. **Cheap path (every run)**: install only the scraping dependencies
   (Scrapling, Playwright, BeautifulSoup, httpx, SQLAlchemy…) and Chromium,
   then run `python scripts/daily_update.py --detect-only`. This scrapes,
   detects and records changes, saves the new snapshots, and skips
   re-indexing. It writes `changed=true|false` to `$GITHUB_OUTPUT`, and exits
   non-zero if no page was fetched.
2. **Expensive path (only when `changed == 'true'`)**: install
   `requirements.txt` (ChromaDB, sentence-transformers, PyTorch), restore the
   embedding-model cache, run `scripts/fetch_and_index.py` to re-fetch and
   re-embed the monitored pages and any new or changed PDFs, then commit
   `database/vectorstore` and `database/changes.db` and push. The push
   triggers a Streamlit Cloud redeploy.

A no-change day stops after step 1: no ML install, no commit. The workflow sets
`SCRAPER_BACKEND=browser` and `MAX_LLM_CLASSIFICATIONS=0`, so CI makes no Groq
calls. A `concurrency` group stops two runs racing on the snapshots and the
commit.

Run locally without `--detect-only`, `daily_update.py` re-ingests by itself, but
still only when it detected a change.

### Fetch backends (`src/scraping/fetchers.py`)

The Home Affairs pages build their content in JavaScript: a plain HTTP fetch
returns a page that looks successful but holds almost no text. `SCRAPER_BACKEND`
selects the backend:

| Value | Behaviour |
|---|---|
| `auto` (default) | `browser` if Chromium is present, else `scrapling`, else `httpx` |
| `browser` | Playwright Chromium via Scrapling's `DynamicFetcher`. Renders JavaScript |
| `scrapling` | Browser TLS impersonation via Scrapling's `Fetcher` (curl_cffi). No JS rendering |
| `httpx` | Plain HTTP |

A missing or misspelled backend falls back with a loud warning instead of
failing the run. PDF downloads always use httpx, because they need raw bytes
and ETag conditional GET.

### State persistence

Change detection needs the previous scrape to still exist on the next run.
GitHub Actions runners are ephemeral, so `database/changes.db` is **committed
to the repository**, and the checkout restores the last committed snapshot as
the baseline. A cache was rejected: it works until it silently expires, and
then every run re-baselines and reports no changes forever — the original bug.

That one file holds both the page snapshots (`page_snapshots`) and the PDF
ingestion hashes (`documents`), so it is the only change-detection state the
pipeline depends on. Snapshots are pruned to `SNAPSHOT_HISTORY` (default 10)
per URL, so the file stays small. The vector store (`database/vectorstore/`)
is committed too, because the deployed app reads the committed index. Both are
committed by the workflow's "Commit refreshed index" step, which runs only when
`changed=true`. HTML files under `data/raw/html_snapshots/` are gitignored and
are a local debugging archive only.

### Monitored URLs
Set by `MONITORED_URLS`. The defaults are the main 485 page and its three stream
sub-pages:
- Main 485 visa page (`/temporary-graduate-485`)
- Post-Higher Education Work stream
- Post-Vocational Education Work stream
- Second Post-Higher Education Work stream

PDFs linked from these pages are downloaded too. `scripts/fetch_and_index.py
--crawl` also follows links under the `/temporary-graduate-485` prefix (capped
by `--max-pages`, default 25).

### Change Detection Levels
Implemented in `src/monitoring/change_detector.py`:

1. **Level 1 — Content hash**: SHA-256 of the whitespace-normalised page text,
   compared with the last snapshot (has anything changed?). A URL with no
   previous snapshot is stored as the baseline and reports no changes.
2. **Level 2 — Text diff**: `difflib.SequenceMatcher` over lines, giving
   `content_update`, `content_added` and `content_removed` blocks (what
   changed?). If the hash changed but no lines differ, one MINOR `formatting`
   change is recorded.
3. **Level 3 — Severity classification**: each block is labelled CRITICAL,
   IMPORTANT or MINOR (how important is it?). The LLM classifies up to
   `MAX_LLM_CLASSIFICATIONS` blocks (default 10) per page compared; the rest,
   and any block where the LLM is unavailable or fails, use the keyword lists
   (`CRITICAL_KEYWORDS`, `IMPORTANT_KEYWORDS`). The daily workflow sets the cap
   to 0, so CI uses keywords only.

---

## Alert System ⏸️ (built, parked)

- **Off by default**: `ALERTS_ENABLED=false`. `daily_update.py` only calls the
  alert manager when it is `true` and `DISCORD_WEBHOOK_URL` is set; the daily
  workflow leaves it off. Detected changes are still recorded and shown on the
  Changes page.
- **Discord webhooks**: one message per run, with an embed per change (up to
  10, then an "…and N more" line) (`src/alerts/alert_manager.py`)
- **Severity filter**: `ALERT_MIN_SEVERITY` (default `IMPORTANT`)
- **Deduplication**: `changes.notified` is set to 2 on success, so a re-run
  never alerts on the same change twice. A failed send leaves the change
  pending for the next run.
- **Audit trail**: every attempt is written to `alert_logs` and surfaced on
  the Alerts page (operator mode, `PUBLIC_MODE=false`)

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
├── metadatas: [{source, page_number, chunk_index, doc_type, ...}, ...]
└── ids: ["{source}_p{page}_c{chunk_index}_{i}", ...]
```

---

## Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Components accept their dependencies
   (`Retriever(vectorstore=...)`, `HomeAffairsScraper(fetcher=...)`,
   `ChangeDetector(db=...)`) and only build a default when none is passed
3. **Graceful Degradation**: System works even if some components fail — LLM
   query expansion falls back to synonyms, a missing fetch backend falls back
   with a warning, a failed grounding check does not block the answer. The
   exception is deliberate: a scrape that fetches no page fails the run.
4. **Cost Optimization**: Free-tier everything for portfolio project; the daily
   job installs ML dependencies and re-indexes only when the site changed
5. **Local-First**: Embeddings run locally, only LLM needs API
6. **Extensibility**: Easy to add new data sources, fetch backends
   (`BaseFetcher`), LLMs, or alert channels

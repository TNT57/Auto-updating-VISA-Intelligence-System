# 🛂 Auto-Updating VISA Intelligence System

> A side project exploring **RAG (Retrieval-Augmented Generation)** to build an intelligent, self-updating knowledge base for Australian immigration policy — because government websites change constantly, and keeping track manually is painful.

---

## What It Does

This system monitors the Australian Department of Home Affairs website for changes to visa policies (currently targeting the **485 Temporary Graduate visa**), automatically ingests updated documents into a vector database, and lets you chat with the latest policy information through an AI-powered interface.

| Feature | How It Works |
|---|---|
| **PDF Ingestion & Chunking** | Extracts text and tables from government PDFs using `pdfplumber`, splits into semantic chunks with `langchain-text-splitters` |
| **Vector Search (RAG)** | Embeds chunks with `SentenceTransformers` (`all-mpnet-base-v2`), stores in `ChromaDB`, retrieves relevant context for queries |
| **AI-Powered Chat** | Uses `Groq` (Llama 3.3 70B) to generate grounded answers, then re-checks each answer against its sources |
| **Web Scraping & Monitoring** | Scrapes the Home Affairs website on a schedule via `BeautifulSoup`, detects content changes |
| **Change Detection** | Stores page snapshots in `SQLite` and diffs them to flag policy updates, with LLM severity classification |
| **Alerts** | Discord webhook notifications for changes at or above a configurable severity. *Currently parked — see Project Status* |
| **Dashboard** | `Streamlit` multi-page app with chat, change log, and alert status |
| **Automated Pipeline** | GitHub Actions workflow runs daily scraping, change detection, re-ingestion, and alerting |

## Why This Project

I built this to solve a real problem I faced — Australian visa rules update frequently and the changes are buried in long PDF documents. Instead of manually checking and re-reading, I wanted a system that:

1. **Notifies me** when something changes
2. **Lets me ask questions** about the current rules in plain English
3. **Shows me exactly what changed** between versions

It also served as a hands-on way to learn and apply RAG patterns, vector databases, and LLM integration end-to-end.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                    │
│         ┌──────────┬──────────┬───────────┐             │
│         │   Chat   │ Changes  │  Alerts   │             │
│         └────┬─────┴────┬─────┴─────┬─────┘             │
└──────────────┼──────────┼───────────┼───────────────────┘
               │          │           │
       ┌───────▼──┐  ┌────▼────┐  ┌──▼──────────┐
       │   RAG    │  │ Change  │  │   Alert     │
       │ Pipeline │  │ Detector│  │  Manager    │
       └────┬─────┘  └────┬────┘  └─────────────┘
            │             │
    ┌───────▼─────┐  ┌────▼────────┐
    │  ChromaDB   │  │   SQLite    │
    │  (Vectors)  │  │ (Snapshots) │
    └──────▲──────┘  └────▲────────┘
           │              │
    ┌──────▼──────────────▼──────┐
    │     Ingestion Pipeline     │
    │  PDF Load → Chunk → Embed  │
    └──────────────▲─────────────┘
                   │
         ┌─────────▼──────────┐
         │   Web Scraper      │──── GitHub Actions (daily)
         │   (Home Affairs)   │
         └────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical breakdown.

## Tech Stack

| Category | Tools |
|---|---|
| **LLM & RAG** | langchain-text-splitters, Groq (Llama 3.3), SentenceTransformers |
| **Vector DB** | ChromaDB |
| **Data Processing** | pdfplumber, pypdf, BeautifulSoup4, pandas |
| **Backend** | Python, SQLAlchemy, httpx |
| **Frontend** | Streamlit (multi-page), Plotly |
| **Database** | SQLite |
| **CI/CD** | GitHub Actions, pytest, ruff |
| **Alerts** | Discord Webhooks |

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (for LLM-powered chat)
- Australian visa PDF documents (see step 4)

### Setup

> **⚠️ Always use the virtual environment!** This keeps all dependencies and caches
> inside the project folder on your current drive. Never run bare `pip install` — it
> pollutes your system Python (typically on C: drive) with gigabytes of packages.

```bash
# 1. Clone the repo
git clone https://github.com/TNT57/Auto-updating-VISA-Intelligence-System.git
cd Auto-updating-VISA-Intelligence-System

# 2. Create a virtual environment (keeps packages isolated in this project)
python -m venv .venv

# 3. Activate the virtual environment
#    Windows CMD:
.venv\Scripts\activate
#    Windows PowerShell:
.venv\Scripts\Activate.ps1
#    macOS / Linux:
#    source .venv/bin/activate

# 4. Install dependencies (INTO the venv, not global)
pip install -r requirements.txt

# 5. Configure your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Fetch the real content from the Home Affairs site and index it
python scripts/fetch_and_index.py --crawl

# 7. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Deploying to Streamlit Cloud

The app serves questions from a pre-built index; it does not scrape at
runtime. That shapes the deployment:

1. **The vector store is committed** (`database/vectorstore/`, ~7.5MB).
   Streamlit Cloud deploys from a git clone onto an ephemeral filesystem, so
   an ignored index would mean a deployed app with an empty knowledge base.
   Committing it also makes deploys reproducible — the app ships the exact
   corpus it was tested against. Rebuild and re-commit with
   `python scripts/fetch_and_index.py --crawl`.

2. **`requirements.txt` holds runtime dependencies only.** Scraping, PDF
   parsing, Playwright and dev tooling live in `requirements-dev.txt`: they
   build the index, they are not needed to query one, and they would waste a
   constrained container.

3. **Set secrets in the Streamlit Cloud UI**, not a `.env`. Under
   *Settings → Secrets*:

   ```toml
   GROQ_API_KEY = "gsk_..."
   QUERY_EXPANSION = "llm"
   ```

   `app/bootstrap.py` copies these into the environment before settings are
   read, so the same config code works locally and deployed. Real environment
   variables take precedence, so a local `.env` is unaffected.

4. **Main file path:** `app/streamlit_app.py`.

⚠️ **Memory.** `sentence-transformers` pulls in PyTorch, and the embedding
model (`all-mpnet-base-v2`, 768-dim) is ~420MB. On Streamlit Cloud's free
tier that is close to the limit and may fail to boot. If it does, switch to
ChromaDB's bundled ONNX model (`all-MiniLM-L6-v2`, 384-dim, no PyTorch): set
`EMBEDDING_MODEL=all-MiniLM-L6-v2`, rebuild with
`python scripts/initial_setup.py --rebuild` then
`python scripts/fetch_and_index.py --crawl`, and re-commit. Retrieval quality
drops; measure it with `scripts/eval_retrieval.py` before and after rather
than guessing.

### Building the knowledge base

`scripts/fetch_and_index.py` is what puts real content behind the chat. It
fetches the monitored pages, optionally follows links within the 485 section,
downloads any linked PDFs, and indexes all of it into ChromaDB.

```bash
python scripts/fetch_and_index.py                 # configured URLs + their PDFs
python scripts/fetch_and_index.py --crawl         # also follow in-section links
python scripts/fetch_and_index.py --dry-run       # show what it would fetch
python scripts/fetch_and_index.py --ask "What is the English requirement?"
```

Run it from a machine whose IP the site does not block — a home connection.
GitHub Actions runners are refused (see Troubleshooting). The crawl is bounded:
same host only, restricted to the 485 URL prefix, capped by `--max-pages`
(default 25), and it honours `robots.txt` and `SCRAPING_DELAY`.

Without this step the vector store only contains whatever PDFs you placed in
`data/raw/pdfs/` by hand, so the chat cannot answer from the live site.
`scripts/initial_setup.py` still exists for the PDFs-only path.

### Where Files Live (Cache & Storage)

This project is designed to keep **everything** inside its own directory — no stray
files on your C: drive.

| What | Location | Notes |
|---|---|---|
| Python packages | `.venv/` | Virtual environment (gitignored) |
| HuggingFace models | `.cache/huggingface/` | Embedding models cached here (gitignored) |
| PyTorch data | `.cache/torch/` | Torch hub cache (gitignored) |
| ChromaDB vectors | `database/vectorstore/` | Persistent vector DB (gitignored) |
| Raw PDFs / HTML | `data/raw/` | Your documents (gitignored) |

Cache directories are configured in two places so they always point inside the project:
1. **`.env`** — `HF_HOME` and `TORCH_HOME` environment variables
2. **`src/utils/config.py`** — programmatic fallback via `os.environ.setdefault()`

### Project Status

| Component | Status | Notes |
|---|---|---|
| PDF ingestion & chunking | ✅ Working | Includes table → markdown extraction |
| Vector store (ChromaDB) | ✅ Working | Cosine distance, `all-mpnet-base-v2` |
| RAG retrieval pipeline | ✅ Working | |
| LLM chat (Groq) | ✅ Working | Streaming, conversation memory, grounding check |
| Streamlit dashboard | ✅ Working | Chat and Changes pages |
| Web scraper | ✅ Working | 4 monitored URLs, conditional GET for PDFs |
| Change detection | ✅ Working | Snapshots persist in SQLite |
| Alert system (Discord) | ⏸️ Built, parked | Working and tested, but off by default — set `ALERTS_ENABLED=true` |
| Email alerts | ❌ Not implemented | SMTP settings exist in config but are unused |
| Alerts config page | ⚠️ Read-only | Shows status; configure via `.env`, not the UI |
| GitHub Actions daily scrape | ✅ Configured | State cached between runs |
| CI (tests + lint) | ✅ Configured | `.github/workflows/tests.yml` |

To run it yourself:
1. Add a `GROQ_API_KEY` to your `.env` file (free at [console.groq.com](https://console.groq.com/keys))
2. Place at least one PDF document in `data/raw/pdfs/`
3. Run `python scripts/initial_setup.py` to build the vector database
4. Launch with `streamlit run app/streamlit_app.py`

> **Upgrading an existing install?** The vector collection now uses cosine
> distance instead of Chroma's default squared-L2. Rebuild it once with
> `python scripts/initial_setup.py --rebuild`, otherwise relevance scores will
> keep reading 0%.

### Troubleshooting the daily scrape

**"Daily Scrape" emails you a failure every day.** Two separate causes:

1. **`git push` denied to `github-actions[bot]` (exit 128).** This is what
   actually failed the job. The workflow tried to commit results back to the
   repo without `permissions: contents: write` — and everything it tried to
   commit was gitignored anyway. **Fixed:** the push step is gone; state is
   cached instead.
2. **All four URLs returned `403 Forbidden`** on the runner. **Not fixed, and
   not fixable from GitHub Actions** — see below.

The run also used to count blocked pages as "scraped", so it reported
`Pages scraped: 4` on a run that fetched nothing. It now counts only real
successes and **exits non-zero if no page was fetched** — a monitoring job
that silently goes green is worse than one that fails.

### The 403 is an IP block, not a header problem

A bare `curl` with no special headers succeeds from a residential connection
but the same request 403s from a GitHub Actions runner. That rules out user
agent and header fingerprinting: the site is refusing the runner's **network**,
whether by datacenter IP range or geography.

Consequences:

- No fetch backend fixes this on GitHub Actions. `scrapling` changes the TLS
  fingerprint, not the source IP.
- **Run the pipeline where the IP is not blocked.** On Windows, Task Scheduler
  running `python scripts/daily_update.py` daily; on macOS/Linux, cron. The
  scheduled workflow is best treated as disabled until then.
- If CI-based scraping is genuinely needed later, the fix is a fetch that
  *originates elsewhere* — a hosted scraping API or an egress proxy. The
  `BaseFetcher` interface in `src/scraping/fetchers.py` exists so that can be
  added without touching the scraper or the pipeline.

### Fetch backends

`SCRAPER_BACKEND` selects how pages are fetched:

| Value | Behaviour |
|---|---|
| `auto` (default) | Use `scrapling` if installed, else `httpx` |
| `httpx` | Plain HTTP. Sufficient from an unblocked IP |
| `scrapling` | Browser TLS impersonation via curl_cffi, for stricter fingerprint checks |

An unavailable or misspelled backend falls back to `httpx` with a warning
rather than failing the run. Scrapling is optional — if it isn't installed,
everything still works.

### How the daily update stays stateful

Change detection only works if the previous scrape survives to the next run.
Page snapshots and PDF ingestion hashes both live in `database/changes.db`,
which the GitHub Actions workflow restores from cache at the start of every
run. That single file is the only state the pipeline needs; the HTML files
under `data/raw/html_snapshots/` are a local archive, not the source of truth.

## Project Structure

```
├── app/                        # Streamlit dashboard
│   ├── streamlit_app.py        # Main entry point & sidebar
│   └── pages/
│       ├── 1_Chat.py           # RAG-powered Q&A
│       ├── 2_Changes.py        # Change detection timeline
│       └── 3_Alerts.py         # Alert configuration
├── src/
│   ├── ingestion/              # PDF loading, chunking, vector store
│   ├── retrieval/              # Semantic search & retrieval
│   ├── generation/             # LLM client & prompt templates
│   ├── scraping/               # Home Affairs web scraper
│   ├── monitoring/             # Change detection engine
│   ├── alerts/                 # Notification manager
│   └── utils/                  # Config, logging, database
├── scripts/
│   ├── fetch_and_index.py      # Fetch the live site + PDFs → index for RAG
│   ├── initial_setup.py        # First-time setup (--rebuild to reset the vector DB)
│   └── daily_update.py         # Scrape → detect changes → re-ingest → alert
├── .github/workflows/
│   ├── daily_scrape.yml        # Automated daily scraping
│   └── tests.yml               # pytest + ruff on every push
├── docs/
│   └── ARCHITECTURE.md         # Detailed architecture docs
└── tests/                      # Unit + integration tests
```

## Development

```bash
pytest                      # unit tests (fast, fully mocked)
pytest -m integration       # real embeddings + real ChromaDB (slow)
ruff check .                # lint
```

## What I Learned

- Building an end-to-end RAG pipeline — from raw PDFs to grounded AI answers
- Trade-offs in chunking strategies and embedding models for document retrieval
- Web scraping with change detection and snapshot diffing
- Using free-tier LLM APIs (Groq) for production-quality inference
- Scheduling automated data pipelines with GitHub Actions

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as a personal side project to explore RAG systems and automated intelligence gathering.*
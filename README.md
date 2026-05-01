# 🛂 Auto-Updating VISA Intelligence System

> A side project exploring **RAG (Retrieval-Augmented Generation)** to build an intelligent, self-updating knowledge base for Australian immigration policy — because government websites change constantly, and keeping track manually is painful.

---

## What It Does

This system monitors the Australian Department of Home Affairs website for changes to visa policies (currently targeting the **485 Temporary Graduate visa**), automatically ingests updated documents into a vector database, and lets you chat with the latest policy information through an AI-powered interface.

| Feature | How It Works |
|---|---|
| **PDF Ingestion & Chunking** | Extracts text from government PDFs using `pdfplumber`, splits into semantic chunks with `LangChain` |
| **Vector Search (RAG)** | Embeds chunks with `SentenceTransformers`, stores in `ChromaDB`, retrieves relevant context for queries |
| **AI-Powered Chat** | Uses `Groq` (Llama 3) to generate grounded answers based on retrieved document context |
| **Web Scraping & Monitoring** | Scrapes the Home Affairs website on a schedule via `BeautifulSoup`, detects content changes |
| **Change Detection** | Stores historical snapshots in `SQLite` and diffs them to flag policy updates |
| **Alerts** | Sends notifications for detected changes via Discord webhooks |
| **Dashboard** | `Streamlit` multi-page app with chat, change log, and alert configuration |
| **Automated Pipeline** | GitHub Actions workflow runs daily scraping and re-ingestion |

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
| **LLM & RAG** | LangChain, Groq (Llama 3), SentenceTransformers |
| **Vector DB** | ChromaDB |
| **Data Processing** | pdfplumber, pypdf, BeautifulSoup4, pandas |
| **Backend** | Python, SQLAlchemy, APScheduler |
| **Frontend** | Streamlit (multi-page), Plotly |
| **Database** | SQLite |
| **CI/CD** | GitHub Actions |
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

# 6. Add visa PDF documents
# Download from: https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485
# Place them in: data/raw/pdfs/

# 7. Run the setup script (creates directories, builds vector DB, runs test query)
python scripts/initial_setup.py

# 8. Launch the dashboard
streamlit run app/streamlit_app.py
```

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

| Component | Status |
|---|---|
| PDF ingestion & chunking | ✅ Working |
| Vector store (ChromaDB) | ✅ Working |
| RAG retrieval pipeline | ✅ Working |
| LLM chat (Groq) | ✅ Working |
| Streamlit dashboard | ✅ Working |
| Web scraper | ✅ Working |
| Change detection | ✅ Working |
| Alert system (Discord) | ✅ Working |
| GitHub Actions daily scrape | ✅ Configured |

The system is **fully functional**. To test it yourself, you'll need to:
1. Add a `GROQ_API_KEY` to your `.env` file (free at [console.groq.com](https://console.groq.com/keys))
2. Place at least one PDF document in `data/raw/pdfs/`
3. Run `python scripts/initial_setup.py` to build the vector database
4. Launch with `streamlit run app/streamlit_app.py`

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
│   └── initial_setup.py        # First-time project setup
├── .github/workflows/
│   └── daily_scrape.yml        # Automated daily scraping
├── docs/
│   └── ARCHITECTURE.md         # Detailed architecture docs
└── tests/                      # Unit tests
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
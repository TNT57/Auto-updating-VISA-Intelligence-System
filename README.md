# 🛂 Auto-Updating 485 Visa Intelligence System

> **Never miss a 485 visa policy change again.**

An intelligent monitoring system that scrapes Australian immigration websites daily, detects policy changes, and provides an AI-powered Q&A chatbot with source citations using Retrieval-Augmented Generation (RAG).

---

## 🎯 What It Does

| Feature | Status | Description |
|---|---|---|
| 💬 **RAG Chatbot** | ✅ Phase 1 | Ask questions about the 485 visa with cited sources |
| 📊 **Change Detection** | 🔜 Phase 2 | Automatically detect policy changes from government websites |
| 🔔 **Smart Alerts** | 🔜 Phase 3 | Discord/email notifications for critical changes |
| 📈 **Dashboard** | 🔜 Phase 4 | System health monitoring and analytics |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                 STREAMLIT UI                         │
│  💬 Chat  │  📊 Changes  │  🔔 Alerts  │  📈 Health │
└──────────────────────┬──────────────────────────────┘
                       ↕
┌──────────────────────────────────────────────────────┐
│              RAG QUERY ENGINE                        │
│  Retriever (ChromaDB) → LLM (Groq) → Citations     │
└──────────────────────┬──────────────────────────────┘
                       ↕
┌──────────────────────────────────────────────────────┐
│           VECTOR DATABASE (ChromaDB)                 │
│  Embeddings (all-MiniLM-L6-v2) + Metadata           │
└──────────────────────┬──────────────────────────────┘
                       ↕
┌──────────────────────────────────────────────────────┐
│          DATA PIPELINE (Phase 2)                     │
│  Scraper → Diff Detector → Ingestion → Alerts       │
└──────────────────────┬──────────────────────────────┘
                       ↕
┌──────────────────────────────────────────────────────┐
│          DATA SOURCES                                │
│  homeaffairs.gov.au PDFs + Web Pages                │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core RAG Pipeline
| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (Llama 3.3 70B) | Answer generation (free, instant) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Local vector embeddings |
| **Vector DB** | ChromaDB | Persistent vector storage |
| **PDF Parsing** | pdfplumber | Text extraction with tables |
| **Framework** | LangChain | Text chunking & orchestration |

### Application
| Component | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Multi-page web application |
| **Database** | SQLite + SQLAlchemy | Change tracking |
| **Logging** | Loguru | Structured logging |
| **Config** | Pydantic + python-dotenv | Settings management |

### Automation (Phase 2+)
| Component | Technology | Purpose |
|---|---|---|
| **Scraping** | httpx + BeautifulSoup | Government website monitoring |
| **Scheduling** | GitHub Actions (cron) | Daily automated updates |
| **Alerts** | Discord Webhooks | Instant notifications |
| **Retries** | Tenacity | Robust error handling |

---

## 📁 Project Structure

```
visa-485-intelligence/
├── app/                          # Streamlit web application
│   ├── streamlit_app.py          # Main entry point
│   └── pages/                    # Multi-page app
│       ├── 1_💬_Chat.py          # RAG Q&A interface
│       ├── 2_📊_Changes.py       # Change timeline (Phase 2)
│       └── 3_🔔_Alerts.py        # Alert config (Phase 3)
│
├── src/                          # Core source code
│   ├── ingestion/                # PDF loading, chunking, vectorstore
│   │   ├── pdf_loader.py         # PDF text extraction with metadata
│   │   ├── text_chunker.py       # Recursive text splitting
│   │   └── vectorstore_manager.py # ChromaDB management
│   ├── retrieval/                # Semantic search
│   │   └── retriever.py          # Query engine with relevance scoring
│   ├── generation/               # LLM integration
│   │   ├── llm_client.py         # Groq/OpenAI client with streaming
│   │   └── prompt_templates.py   # Engineered prompts with citations
│   ├── scraping/                 # Web scraping (Phase 2)
│   ├── monitoring/               # Change detection (Phase 2)
│   ├── alerts/                   # Notifications (Phase 3)
│   ├── temporal/                 # Version control (Phase 3)
│   └── utils/                    # Shared utilities
│       ├── config.py             # Pydantic settings
│       ├── logger.py             # Loguru configuration
│       └── db_manager.py         # SQLAlchemy database
│
├── scripts/                      # Operational scripts
│   └── initial_setup.py          # First-time setup wizard
│
├── data/                         # Data storage (gitignored)
│   └── raw/pdfs/                 # Place 485 visa PDFs here
│
├── database/                     # Database storage (gitignored)
│   └── changes.db                # SQLite change tracking
│
├── .github/workflows/            # GitHub Actions
│   └── daily_scrape.yml          # Daily automation (Phase 2)
│
├── .env.example                  # Environment template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com/keys) (free)

### 1. Clone & Install

```bash
git clone https://github.com/TNT57/Auto-updating-VISA-Intelligence-System.git
cd Auto-updating-VISA-Intelligence-System

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Add Documents

Download 485 visa PDFs from [immi.homeaffairs.gov.au](https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485) and place them in `data/raw/pdfs/`.

### 4. Initialize

```bash
python scripts/initial_setup.py
```

### 5. Run

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` and start asking questions! 🎉

---

## 💡 Key Technical Decisions

### Why Groq (not OpenAI)?
- **Free tier** with generous limits
- **Instant inference** — Llama 3.3 70B runs at ~500 tokens/sec
- Easy swap to OpenAI via abstraction layer (change one env variable)

### Why ChromaDB (not Pinecone/Weaviate)?
- **Fully local** — no API costs or network latency
- **Persistent** — data survives restarts
- **Simple** — perfect for single-user portfolio project

### Why all-MiniLM-L6-v2 (not OpenAI embeddings)?
- **Runs locally** — no API costs
- **80MB model** — fast to download and load
- **Good quality** — fine for domain-specific RAG

### Why pdfplumber (not PyPDF2)?
- PyPDF2 is **deprecated**
- pdfplumber handles **tables** beautifully (common in visa docs)
- Better text extraction quality overall

---

## 🧠 How the RAG Pipeline Works

```
User Question
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│   Embed      │────▶│  ChromaDB        │
│   Question   │     │  Similarity      │
└─────────────┘     │  Search           │
                    └────────┬─────────┘
                             │ Top 5 chunks
                             ▼
                    ┌──────────────────┐
                    │  Context +       │
                    │  Question → LLM  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Answer with     │
                    │  Source Citations│
                    └──────────────────┘
```

1. **User asks** a question in natural language
2. **Question is embedded** using the same model as documents
3. **ChromaDB finds** the 5 most similar document chunks
4. **Chunks + question** are sent to the LLM with a carefully engineered prompt
5. **LLM generates** an answer with source citations (document name + page number)
6. **Answer displayed** in Streamlit with expandable source references

---

## 🔮 Roadmap

### Phase 2 — Auto-Update System (Weeks 3-4)
- [ ] Web scraper for `immi.homeaffairs.gov.au`
- [ ] Daily automated scraping via GitHub Actions
- [ ] Change detection with diff algorithm
- [ ] Severity classification (Critical/Important/Minor)

### Phase 3 — Intelligence Layer (Weeks 5-6)
- [ ] Temporal RAG — query across time ("What changed since January?")
- [ ] Discord webhook alerts for critical changes
- [ ] Email digest (daily/weekly)
- [ ] "What's new this week?" AI summaries

### Phase 4 — Production Polish (Weeks 7-8)
- [ ] Comprehensive error handling & retry logic
- [ ] Health monitoring dashboard
- [ ] Rate limiting (respect government servers)
- [ ] Deployment to Streamlit Community Cloud
- [ ] Demo video and blog post

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific module
pytest tests/test_ingestion.py -v
```

---

## 📊 Deployment Strategy ($0/month)

| Component | Platform | Cost |
|---|---|---|
| **Streamlit UI** | Streamlit Community Cloud | Free |
| **Daily Scraping** | GitHub Actions (cron) | Free |
| **Vector DB** | Persistent local storage | Free |
| **LLM** | Groq free tier | Free |
| **Alerts** | Discord webhooks | Free |

---

## ⚠️ Disclaimer

This system is for **informational purposes only**. It is NOT legal advice. Always consult a **registered migration agent** (MARA) for your specific visa situation. The AI may occasionally produce inaccurate information — always verify against official sources.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Data source**: [Australian Department of Home Affairs](https://immi.homeaffairs.gov.au)
- **LLM**: [Groq](https://groq.com) — free, fast inference
- **Embeddings**: [Sentence-Transformers](https://www.sbert.net)
- **Vector DB**: [ChromaDB](https://www.trychroma.com)
- **UI**: [Streamlit](https://streamlit.io)

---

Built with ❤️ for international students in Australia.
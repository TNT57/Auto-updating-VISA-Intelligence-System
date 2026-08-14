# 🛂 Auto-Updating VISA Intelligence System

> Ask questions about the Australian **Subclass 485 Temporary Graduate visa**
> and get answers drawn from the Department of Home Affairs' own pages, with a
> link to the source of every claim.

<!-- DEPLOY-LINK -->
### ▶️ [Try it live](https://auto-updating-visa-intelligence-system.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://auto-updating-visa-intelligence-system.streamlit.app/)

The deployed app serves the chat and nothing else — see
[Deploying to Streamlit Cloud](#deploying-to-streamlit-cloud). It may take a
few seconds to wake if it has been idle.

---

## What it does

Australian visa rules change often, and the answers are spread across pages
that render entirely in JavaScript. This project scrapes those pages, indexes
them, and answers questions from them.

Ask *"how much does it cost?"* and it replies:

```
- Post-Higher Education Work stream:        From AUD5,750.00
- Post-Vocational Education Work stream:    From AUD5,750.00
- Second Post-Higher Education Work stream: From AUD2,265.00
```

…because the fee genuinely differs by stream, and answering with one number
would be wrong. Every answer is re-checked against its sources before you see
it, and cites the pages it came from.

## How it works

Three stages. There is a full walkthrough with real output in
[`docs/PIPELINE_WALKTHROUGH.txt`](docs/PIPELINE_WALKTHROUGH.txt).

**1. Ingestion** — `scripts/fetch_and_index.py`

The Home Affairs pages build their content client-side: a plain HTTP fetch
returns 1.2MB of HTML containing about 1,400 characters of text, all
navigation, with the body reading "Loading". So pages are rendered in a real
browser, the content extracted, split into ~1,000-character overlapping
**chunks**, and each chunk converted into a 768-number **embedding** that
captures its meaning. Those live in a local ChromaDB vector store.

Generic application forms (Form 80, 1221, 956…) are linked from every visa
page and were 246 of 478 chunks while mentioning the visa zero times. They are
now filtered out on content, not filename.

**2. Retrieval** — `src/retrieval/`

Your question is embedded the same way, and the closest chunks come back.
The interesting part is **query expansion** — see below.

**3. Generation** — `src/generation/`

The top passages go to Llama 3.3 70B (via Groq's free tier) with instructions
to use only what it was given. A second call then re-reads the answer against
those same passages and labels it GROUNDED, PARTIALLY_GROUNDED or UNGROUNDED.
When the answer isn't in the corpus, it says so rather than inventing one.

### Query expansion: the fix that mattered most

Applicants and governments use different words for the same thing. Asked two
ways, the *same* passage scored very differently:

| Question | Rank | Relevance |
|---|---|---|
| "Do I need to be in Australia when I apply?" | 1st | 62% |
| "Does it matter, onshore or offshore?" | 3rd | 25% |

Same chunk, same model, same index. "Onshore" is migration-agent shorthand;
Home Affairs writes "be in Australia when you apply". Dense retrieval matches
on meaning, but it cannot bridge vocabulary it has never seen paired.

So before searching, the question is rewritten into several phrasings — by the
LLM, or by a built-in map of migration jargon — each one searched separately,
and the rankings combined with **reciprocal rank fusion**. A chunk that ranks
respectably for several phrasings beats one that ranks first for a single
lucky wording, so *agreement across vocabularies* becomes the signal.

Measured over 226 chunks of live content, 10 questions
(`python scripts/eval_retrieval.py`):

| Strategy | hit@1 | hit@3 | MRR |
|---|---|---|---|
| `none` — single query | 50% | 70% | 0.608 |
| `synonyms` — jargon map, free and instant | 80% | 90% | 0.833 |
| **`llm`** — LLM paraphrase (default) | **80%** | **100%** | **0.867** |

*hit@1* = the right passage ranked first. *hit@3* = it was in the top three.
*MRR* = mean of 1/rank, so first place scores 1.0, third scores 0.33 — one
number for "how near the top, usually". The onshore/offshore question moved
from 25% to 66% relevance.

Set with `QUERY_EXPANSION=llm|synonyms|none`.

## Why this project

Australian visa rules update frequently and the changes are buried across long
pages and PDFs. Instead of checking manually, I wanted something that:

1. **Answers questions** about the current rules in plain English
2. **Cites its sources** so any answer can be verified
3. **Notices when something changes** (built; parked — see Project Status)

It was also a way to learn RAG, vector databases and LLM integration
end-to-end.

## Architecture

```
  ASKING                                    KEEPING IT CURRENT
  ──────                                    ──────────────────
  Your question                             GitHub Actions, 2am daily
        │                                            │
        ▼                                            ▼
  ┌───────────────────┐                   ┌────────────────────┐
  │ Query expansion   │  rewrite into     │ Browser scraper    │
  │ + rank fusion     │  several          │ (Playwright —      │
  └─────────┬─────────┘  phrasings        │  the site is JS)   │
            │                             └─────────┬──────────┘
            ▼                                       ▼
  ┌───────────────────┐                   ┌────────────────────┐
  │ ChromaDB          │◀──── re-embed ────│ Changed?  ──no──▶ stop
  │ 226 chunks        │      only if      │ (diff vs snapshot) │  (~80s,
  │ (committed)       │      changed      └─────────┬──────────┘   no ML)
  └─────────┬─────────┘                             │ yes
            │ top passages                          ▼
            ▼                                  commit + push
  ┌───────────────────┐                             │
  │ Llama 3.3 70B     │                             ▼
  │ answer + cite     │                    Streamlit redeploys
  └─────────┬─────────┘
            ▼
  ┌───────────────────┐
  │ Grounding check   │  re-reads the answer against
  │ GROUNDED / not    │  the same passages
  └─────────┬─────────┘
            ▼
     Answer + source links
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical breakdown.

## Tech Stack

| Category | Tools |
|---|---|
| **LLM & RAG** | Groq (Llama 3.3 70B), SentenceTransformers (`all-mpnet-base-v2`), langchain-text-splitters |
| **Vector DB** | ChromaDB (cosine) |
| **Scraping** | Scrapling + Playwright Chromium (the site is JS-rendered), BeautifulSoup4 |
| **Data Processing** | pdfplumber, pypdf, pandas |
| **Backend** | Python 3.10+, SQLAlchemy, httpx, Pydantic Settings |
| **Frontend** | Streamlit, Plotly |
| **Database** | SQLite |
| **CI/CD** | GitHub Actions, pytest, ruff |
| **Alerts** | Discord webhooks *(built, parked)* |

## Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) — for the chat only;
  ingestion and retrieval run entirely locally
- Chromium for scraping: `python -m playwright install chromium`. Not needed
  to query the prebuilt index that ships with the repo

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

5. **Visitors see the chat and nothing else.** `PUBLIC_MODE` defaults to
   `true`, so the Changes and Alerts pages, the retrieval-depth slider and the
   chunk-count readouts — all operator tooling — are not merely hidden but
   unreachable, since `st.navigation` bypasses Streamlit's automatic page
   discovery. Set `PUBLIC_MODE=false` locally to get the operator view.

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
| Streamlit dashboard | ✅ Working | Chat only in public mode; Home/Changes/Alerts with `PUBLIC_MODE=false` |
| Web scraper | ✅ Working | Browser-rendered; 4 seed URLs, 7 pages with `--crawl` |
| Change detection | ✅ Working | Snapshots committed in `database/changes.db` |
| Alert system (Discord) | ⏸️ Built, parked | Working and tested, but off by default — set `ALERTS_ENABLED=true` |
| Email alerts | ❌ Not implemented | SMTP settings exist in config but are unused |
| Alerts config page | ⚠️ Read-only | Shows status; configure via `.env`, not the UI |
| GitHub Actions daily scrape | ✅ Working | Cheap detect pass daily; re-indexes and commits only on change |
| CI (tests + lint) | ✅ Working | `.github/workflows/tests.yml` |

### Known limitations / future work

- **HTML tables lose their structure.** The scraper flattens a `<table>` to one
  cell per line, so the English score grids arrive as a header list followed by
  a run of loose numbers and the LLM has to re-pair them by position. Answers
  spot-checked against the live page have been correct, helped by the chunk
  overlap keeping headers beside their values, but a merged or empty cell would
  break the alignment silently. `pdf_loader.py` already renders PDF tables as
  markdown; porting that to the HTML path would fix it properly.
- **Coverage is 7 pages** under the `/temporary-graduate-485` prefix. Processing
  times and the full fee schedule live elsewhere, so those questions get an
  honest "not in the provided documents" rather than an answer.

To run it yourself, follow [Setup](#setup) above — in short: create the venv,
add a `GROQ_API_KEY` to `.env`, run `python scripts/fetch_and_index.py --crawl`,
then `streamlit run app/streamlit_app.py`. The repository ships a prebuilt
index, so the fetch step is only needed to refresh it.

> **Upgrading an install from before August 2026?** The vector collection now
> uses cosine distance instead of Chroma's default squared-L2, and the
> embedding model changed. Rebuild once with
> `python scripts/initial_setup.py --rebuild` followed by
> `python scripts/fetch_and_index.py --crawl`, otherwise relevance scores read
> 0% or the collection rejects queries on a dimension mismatch.

### Why the scrape failed for four months

Every scheduled run from April to June failed. Three independent causes, and
none was what it first appeared to be — worth recording because two plausible
diagnoses were wrong.

1. **A single word in the User-Agent.** `USER_AGENT` was
   `Visa485IntelligenceBot/1.0`, and the site's WAF returns 403 to any agent
   string containing "Bot" or "Crawler". Verified by repeated trial:
   `Visa485IntelligenceBot/1.0` and `MyCrawler/1.0` 403 every time, while
   `VisaWatch/1.0`, `curl` and plain `python-httpx` all return 200. The default
   is now an honest identifier with a contact URL and no blocked keyword.

   *A bare `curl` succeeding from a home connection while the runner 403'd
   looked like an IP block. It was not — `curl` simply does not call itself a
   bot. GitHub Actions was never the problem, and the workflow now runs green.*

2. **Three of the four monitored URLs had 404'd.** Home Affairs restructured
   the 485 content into stream sub-pages; `documents-you-need`, `visa-fees` and
   `global-processing-times` no longer exist and do not redirect.

3. **`git push` denied to `github-actions[bot]`** (exit 128) — the workflow
   pushed without `permissions: contents: write`. That is what generated the
   daily failure email.

The run also counted blocked pages as "scraped", reporting `Pages scraped: 4`
on a run that fetched nothing. It now counts only real successes and **exits
non-zero if no page was fetched** — a monitoring job that silently goes green
is worse than one that fails.

### Fetch backends

The Home Affairs pages build their content client-side: a plain HTTP fetch
returns 1.2MB of HTML holding about 1,400 characters of text, all navigation,
with the body reading "Loading". Rendered in a browser the same page yields
about 8,000 characters. **A browser is not optional for this site** — the HTTP
backends return pages that look successful and are empty.

`SCRAPER_BACKEND` selects how pages are fetched:

| Value | Behaviour |
|---|---|
| `auto` (default) | `browser` if Chromium is present, else `scrapling`, else `httpx` |
| `browser` | Playwright Chromium via Scrapling's `DynamicFetcher`. Renders JavaScript |
| `scrapling` | Browser TLS impersonation via curl_cffi. No JS rendering |
| `httpx` | Plain HTTP |

An unavailable or misspelled backend degrades with a loud warning rather than
failing the run — but on this site a degraded fetch means empty pages, not
merely slower ones. Install the browser with
`python -m playwright install chromium`.

### How the daily update stays stateful

Change detection only works if the previous scrape survives to the next run.
Page snapshots and PDF ingestion hashes both live in `database/changes.db`,
which is **committed to the repository**. On an ephemeral runner an ignored
copy means every run re-baselines and reports no changes forever, which was
the original bug; a cache would work until it silently expired and
reintroduced it. The checkout already restores the last committed state, which
is the correct baseline. Snapshots are pruned to `SNAPSHOT_HISTORY` per URL, so
the file stays around 20KB.

## Project Structure

```
├── app/                        # Streamlit UI
│   ├── streamlit_app.py        # Router — decides which pages exist
│   ├── bootstrap.py            # Bridges st.secrets into the environment
│   └── pages/                  # Home, Chat, Changes, Alerts
├── src/
│   ├── ingestion/              # models, pdf_loader, text_chunker,
│   │                           #   vectorstore_manager, pipeline
│   ├── retrieval/              # retriever, query_expansion
│   ├── generation/             # llm_client, prompt_templates
│   ├── scraping/               # homeaffairs_scraper, fetchers (pluggable)
│   ├── monitoring/             # change_detector
│   ├── alerts/                 # alert_manager (parked)
│   └── utils/                  # config, logging, db_manager
├── scripts/
│   ├── fetch_and_index.py      # Fetch the live site → index for RAG
│   ├── inspect_pipeline.py     # Print each stage's real output
│   ├── eval_retrieval.py       # Score retrieval (hit@k, MRR)
│   ├── initial_setup.py        # Local PDFs only (--rebuild resets the store)
│   ├── daily_update.py         # Scrape → detect → re-index (--detect-only)
│   └── create_sample_pdf.py    # Synthetic test PDF — content is invented
├── database/
│   ├── vectorstore/            # ChromaDB — committed, ships with the app
│   └── changes.db              # Page snapshots — committed, the diff baseline
├── .github/workflows/
│   ├── daily_scrape.yml        # Detect daily; re-index only on change
│   └── tests.yml               # pytest + ruff on every push
├── docs/
│   ├── ARCHITECTURE.md         # Technical breakdown
│   └── PIPELINE_WALKTHROUGH.txt # Plain-language tour with real output
└── tests/                      # 216 unit + 30 integration
```

## Development

```bash
pytest                      # unit tests (fast, fully mocked)
pytest -m integration       # real embeddings + real ChromaDB (slow)
ruff check .                # lint
```

## What I Learned

- **Retrieval quality is usually not the embedding model.** The same passage
  ranked 1st at 62% and 3rd at 25% depending only on how the question was
  worded. Rewriting the query and fusing the rankings fixed what a bigger model
  would not have.
- **Measure before optimising.** `eval_retrieval.py` decided every retrieval
  change; one "obvious" improvement made the unexpanded baseline worse.
- **A monitoring job that silently goes green is worse than one that fails.**
  The scrape counted blocked pages as successes and reported "4 pages scraped"
  on runs that fetched nothing.
- **Verify against the real thing.** Mocked tests passed for months while the
  pipeline had never once fetched a page — the User-Agent contained "Bot" and
  the site's WAF refused it.
- **Corpus quality beats retrieval tuning.** Generic application forms were 246
  of 478 chunks; removing them moved hit@1 from 60% to 80%.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as a personal side project to explore RAG systems and automated intelligence gathering.*
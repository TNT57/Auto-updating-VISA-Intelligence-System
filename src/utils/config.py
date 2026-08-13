"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
Provides validated, type-safe settings for the entire application.
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Redirect ML framework caches into the project directory (not C: drive).
# These are set *before* any ML library is imported so they take effect.
# Values from .env take precedence; defaults land inside .cache/ at root.
# ---------------------------------------------------------------------------
_CACHE_DIR = BASE_DIR / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_CACHE_DIR / "torch"))


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ---- LLM Configuration ----
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ---- Embedding Model ----
    embedding_model: str = Field(
        default="all-mpnet-base-v2", alias="EMBEDDING_MODEL"
    )

    # ---- Discord Alerts ----
    # Off by default: the project's focus is the RAG side, and change
    # notification is parked. The alerting code is intact and tested — flip
    # this to true (and set DISCORD_WEBHOOK_URL) to switch it back on.
    alerts_enabled: bool = Field(default=False, alias="ALERTS_ENABLED")
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")
    # Only changes at or above this severity trigger an alert.
    alert_min_severity: str = Field(default="IMPORTANT", alias="ALERT_MIN_SEVERITY")

    # ---- Email Alerts (Phase 3) ----
    smtp_server: str = Field(default="smtp.gmail.com", alias="SMTP_SERVER")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    email_address: str = Field(default="", alias="EMAIL_ADDRESS")
    email_password: str = Field(default="", alias="EMAIL_PASSWORD")
    alert_recipients: str = Field(default="", alias="ALERT_RECIPIENTS")

    # ---- Scraping Configuration ----
    scraping_delay: int = Field(default=5, alias="SCRAPING_DELAY")
    # Fetch backend: auto | httpx | scrapling. "auto" prefers scrapling when
    # installed. See src/scraping/fetchers.py.
    scraper_backend: str = Field(default="auto", alias="SCRAPER_BACKEND")
    # The WAF 403s any User-Agent containing "Bot" or "Crawler" — verified by
    # trial on 2026-08-13. The original "Visa485IntelligenceBot/1.0" was the
    # sole cause of four months of failed scheduled runs. This identifies the
    # client honestly, with a contact URL, without tripping that filter.
    user_agent: str = Field(
        default=(
            "Visa485Intelligence/1.0 "
            "(+https://github.com/TNT57/Auto-updating-VISA-Intelligence-System)"
        ),
        alias="USER_AGENT",
    )

    # ---- Monitored URLs ----
    # Verified live 2026-08-13. The previous list pointed at
    # documents-you-need, visa-fees and global-processing-times, all of which
    # now 404 — Home Affairs restructured the 485 content into three stream
    # sub-pages. Re-check these if fetches start coming back 404.
    monitored_urls: list[str] = Field(
        default=[
            "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485",
            "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-higher-education-work",
            "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-vocational-education-work",
            "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/second-post-higher-education-work",
        ],
        alias="MONITORED_URLS",
    )

    # ---- Application Settings ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "database" / "vectorstore"),
        alias="CHROMA_PERSIST_DIR",
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # ---- Retrieval ----
    # Query expansion strategy: llm | synonyms | none. Measured on the 10
    # questions in scripts/eval_retrieval.py against a 478-chunk index:
    #   none      hit@1 60%  hit@3 70%  MRR 0.693
    #   synonyms  hit@1 60%  hit@3 80%  MRR 0.723
    #   llm       hit@1 70%  hit@3 90%  MRR 0.808
    # "llm" costs one extra call per question and falls back to "synonyms"
    # if the LLM is unavailable.
    query_expansion: str = Field(default="llm", alias="QUERY_EXPANSION")

    # ---- Change detection ----
    # How many page snapshots to retain per URL before pruning. Full page text
    # adds up across daily runs and changes.db is cached between CI runs.
    snapshot_history: int = Field(default=10, alias="SNAPSHOT_HISTORY")
    # Cap on LLM severity classifications per run. A page-wide rewrite can
    # produce dozens of diff blocks, which would otherwise burn Groq's rate
    # limit; blocks beyond this fall back to keyword classification.
    max_llm_classifications: int = Field(
        default=10, alias="MAX_LLM_CLASSIFICATIONS"
    )

    # ---- Derived Paths ----
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / "raw" / "pdfs"

    @property
    def raw_html_dir(self) -> Path:
        return self.data_dir / "raw" / "html_snapshots"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def archives_dir(self) -> Path:
        return self.data_dir / "archives"

    @property
    def database_dir(self) -> Path:
        return BASE_DIR / "database"

    @property
    def changes_db_path(self) -> Path:
        return self.database_dir / "changes.db"

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


# Singleton instance — import this wherever needed
settings = Settings()


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        settings.raw_pdf_dir,
        settings.raw_html_dir,
        settings.processed_dir,
        settings.archives_dir,
        Path(settings.chroma_persist_dir),
        settings.database_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
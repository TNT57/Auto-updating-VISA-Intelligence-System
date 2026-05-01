"""
Configuration management using Pydantic Settings.

Loads configuration from environment variables and .env file.
Provides validated, type-safe settings for the entire application.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


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
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # ---- Discord Alerts (Phase 3) ----
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")

    # ---- Email Alerts (Phase 3) ----
    smtp_server: str = Field(default="smtp.gmail.com", alias="SMTP_SERVER")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    email_address: str = Field(default="", alias="EMAIL_ADDRESS")
    email_password: str = Field(default="", alias="EMAIL_PASSWORD")
    alert_recipients: str = Field(default="", alias="ALERT_RECIPIENTS")

    # ---- Scraping Configuration ----
    scraping_delay: int = Field(default=5, alias="SCRAPING_DELAY")
    user_agent: str = Field(
        default="Visa485IntelligenceBot/1.0 (Educational Project)",
        alias="USER_AGENT",
    )

    # ---- Application Settings ----
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    chroma_persist_dir: str = Field(
        default=str(BASE_DIR / "database" / "vectorstore"),
        alias="CHROMA_PERSIST_DIR",
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

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
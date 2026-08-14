"""
Daily Update Script — Automated scraping, change detection, and re-ingestion.

This script is called by GitHub Actions (daily_scrape.yml) and can also be
run manually:
    python scripts/daily_update.py

Pipeline:
    1. Scrape all monitored URLs (download new PDFs, save HTML snapshots)
    2. Load previous snapshots and compare for changes
    3. Record detected changes in SQLite
    4. Re-ingest any updated/new PDFs into ChromaDB
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.ingestion.pipeline import ingest_pages, ingest_pdfs
from src.monitoring.change_detector import ChangeDetector
from src.scraping.homeaffairs_scraper import HomeAffairsScraper
from src.utils.config import ensure_directories, settings
from src.utils.db_manager import DatabaseManager


def load_previous_content(url: str, db: DatabaseManager) -> str | None:
    """
    Load the most recent stored snapshot content for a URL.

    Snapshots live in SQLite rather than on the filesystem. That matters for
    the scheduled GitHub Actions run: the runner is ephemeral, and the HTML
    snapshot directory is gitignored, so a file-based lookup would find
    nothing and every run would re-baseline instead of detecting changes.
    Keeping snapshots in changes.db means one cached file carries all the
    state the pipeline needs.
    """
    snapshot = db.get_latest_page_snapshot(url)
    if snapshot is None:
        return None
    return snapshot["content"]


# Ingestion lives in src/ingestion/pipeline.py so this script and
# fetch_and_index.py cannot drift apart. Aliased to the historical names.
reingest_updated_pdfs = ingest_pdfs
ingest_scraped_pages = ingest_pages


def run_daily_update() -> dict:
    """
    Execute the full daily update pipeline.

    Returns a summary dict with stats.
    """
    summary = {
        "pages_scraped": 0,
        "pages_failed": 0,
        "changes_detected": 0,
        "pdfs_downloaded": 0,
        "chunks_ingested": 0,
        "alerts_sent": 0,
        "errors": [],
    }

    logger.info("=" * 50)
    logger.info("Starting daily update pipeline")
    logger.info("=" * 50)

    # Ensure directories exist
    ensure_directories()

    # Initialize components
    db = DatabaseManager()
    db.create_tables()
    scraper = HomeAffairsScraper()
    detector = ChangeDetector(db)

    # Step 1: Scrape all monitored URLs
    logger.info("--- Step 1: Scraping monitored URLs ---")
    try:
        results = scraper.run_daily_scrape()
        # Count only pages that actually came back. `results` includes failed
        # fetches, so using len() here reported "4 pages scraped" on a run
        # where all four URLs returned 403 — and made the exit check below
        # think the run had succeeded.
        summary["pages_scraped"] = sum(
            1 for r in results if not r.get("error") and r.get("content")
        )
        summary["pages_failed"] = len(results) - summary["pages_scraped"]
    except Exception as exc:
        logger.error("Scraping failed: {}", exc)
        summary["errors"].append(f"Scraping: {exc}")
        results = []

    # Step 2: Detect changes
    logger.info("--- Step 2: Detecting changes ---")
    for result in results:
        if result.get("error"):
            summary["errors"].append(f"Scrape error ({result['url']}): {result['error']}")
            continue

        url = result["url"]
        new_content = result.get("content", "")

        if not new_content:
            logger.warning("No content extracted from {}", url)
            continue

        old_content = load_previous_content(url, db)
        changes = detector.compare_and_record(
            old_content=old_content,
            new_content=new_content,
            source_url=url,
        )
        summary["changes_detected"] += len(changes)

        for change in changes:
            emoji = {"CRITICAL": "🔴", "IMPORTANT": "🟡", "MINOR": "🟢"}.get(
                change["severity"], "⚪"
            )
            logger.info(
                "{} [{}] {} — {}",
                emoji,
                change["severity"],
                url,
                change.get("summary", "No summary"),
            )

        # Store this scrape as the baseline for the next run. Written after
        # the comparison so the diff above is always old-vs-new.
        db.save_page_snapshot(
            url=url,
            content=new_content,
            content_hash=result.get("content_hash") or "",
            title=result.get("title"),
        )
        db.prune_page_snapshots(url, keep=settings.snapshot_history)

    # Step 3: Re-ingest updated PDFs and scraped pages
    logger.info("--- Step 3: Re-ingesting documents ---")
    try:
        summary["chunks_ingested"] += reingest_updated_pdfs(db)
    except Exception as exc:
        logger.error("PDF re-ingestion failed: {}", exc)
        summary["errors"].append(f"Re-ingestion: {exc}")

    try:
        summary["chunks_ingested"] += ingest_scraped_pages(results)
    except Exception as exc:
        logger.error("Page ingestion failed: {}", exc)
        summary["errors"].append(f"Page ingestion: {exc}")

    # Step 4: Send alerts for significant changes.
    # Disabled by default (ALERTS_ENABLED). Change notification is parked while
    # the project focuses on the RAG side; detected changes are still recorded
    # in SQLite and visible on the Changes page, they just aren't pushed out.
    if not settings.alerts_enabled:
        logger.info("--- Step 4: Alerts disabled (ALERTS_ENABLED=false) ---")
    else:
        logger.info("--- Step 4: Sending alerts ---")
        try:
            from src.alerts.alert_manager import AlertManager

            summary["alerts_sent"] = AlertManager(db).send_pending_alerts()
        except Exception as exc:
            logger.error("Alerting failed: {}", exc)
            summary["errors"].append(f"Alerting: {exc}")

    # Summary
    logger.info("=" * 50)
    logger.info("Daily update complete")
    logger.info("  Pages scraped:    {}", summary["pages_scraped"])
    logger.info("  Changes detected: {}", summary["changes_detected"])
    logger.info("  Chunks ingested:  {}", summary["chunks_ingested"])
    logger.info("  Alerts sent:      {}", summary["alerts_sent"])
    if summary["errors"]:
        logger.warning("  Errors: {}", len(summary["errors"]))
        for err in summary["errors"]:
            logger.warning("    - {}", err)
    logger.info("=" * 50)

    return summary


def main():
    """Entry point for the daily update script."""
    summary = run_daily_update()

    # Print human-readable summary for GitHub Actions
    print("\n" + "=" * 60)
    print("📋 Daily Update Summary")
    print("=" * 60)
    print(f"  Pages scraped:    {summary['pages_scraped']}")
    print(f"  Pages failed:     {summary['pages_failed']}")
    print(f"  Changes detected: {summary['changes_detected']}")
    print(f"  Chunks ingested:  {summary['chunks_ingested']}")
    print(f"  Alerts sent:      {summary['alerts_sent']}")
    if summary["errors"]:
        print(f"  Errors:           {len(summary['errors'])}")
        for err in summary["errors"]:
            print(f"    - {err}")
    print()

    # Fail loudly when nothing was fetched. A run where every URL is blocked
    # must not report success — a silently-green daily job is worse than a
    # failing one, because nobody notices the monitoring has stopped.
    if summary["pages_scraped"] == 0:
        print("❌ No pages were scraped successfully — failing the run.")
        sys.exit(1)

    if summary["pages_failed"]:
        print(
            f"⚠️  {summary['pages_failed']} of "
            f"{summary['pages_failed'] + summary['pages_scraped']} URLs failed."
        )


if __name__ == "__main__":
    main()
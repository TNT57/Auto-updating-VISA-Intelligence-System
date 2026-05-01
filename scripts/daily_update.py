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

from src.scraping.homeaffairs_scraper import HomeAffairsScraper
from src.monitoring.change_detector import ChangeDetector
from src.utils.config import settings, ensure_directories
from src.utils.db_manager import DatabaseManager


def load_previous_content(url: str, db: DatabaseManager) -> str | None:
    """
    Load the most recent snapshot content for a URL from the HTML snapshot files.

    Falls back to checking the database for stored hashes.
    """
    html_dir = settings.raw_html_dir
    if not html_dir.exists():
        return None

    # Build the slug pattern for this URL
    import re
    path_part = url.split("immi.homeaffairs.gov.au", 1)[-1].strip("/")
    slug = re.sub(r"[^\w]+", "_", path_part).strip("_")[:80]

    # Find all snapshots for this URL, sorted by date (newest first)
    snapshots = sorted(
        html_dir.glob("*_{}.html".format(slug)),
        reverse=True,
    )

    if not snapshots:
        return None

    # Read the most recent snapshot (index 0 is today's, so index 1 is previous)
    previous = snapshots[0] if len(snapshots) == 1 else snapshots[1]
    try:
        from bs4 import BeautifulSoup
        html = previous.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        # Use the same extraction logic as the scraper
        return HomeAffairsScraper._extract_content(soup)
    except Exception as exc:
        logger.warning("Could not read previous snapshot {}: {}", previous.name, exc)
        return None


def reingest_updated_pdfs(db: DatabaseManager) -> int:
    """
    Check for new/updated PDFs and re-ingest them into ChromaDB.

    Returns the number of chunks added.
    """
    from src.ingestion.pdf_loader import PDFLoader, DocumentChunk
    from src.ingestion.text_chunker import TextChunker
    from src.ingestion.vectorstore_manager import VectorStoreManager

    pdf_dir = settings.raw_pdf_dir
    if not pdf_dir.exists():
        return 0

    loader = PDFLoader(pdf_dir)
    pdfs = loader.list_pdfs()
    if not pdfs:
        return 0

    total_chunks = 0
    chunker = TextChunker()
    vs = VectorStoreManager()

    for pdf_path in pdfs:
        file_hash = loader.compute_file_hash(pdf_path)

        # Check if this PDF has already been ingested with the same hash
        with db.get_session() as session:
            from src.utils.db_manager import Document
            existing = (
                session.query(Document)
                .filter(Document.file_path == str(pdf_path))
                .first()
            )
            if existing and existing.file_hash == file_hash:
                logger.debug("PDF unchanged, skipping: {}", pdf_path.name)
                continue

        # New or updated PDF — ingest it
        logger.info("Ingesting PDF: {}", pdf_path.name)
        pages = loader.load_pdf(pdf_path)
        if not pages:
            continue

        chunks = chunker.chunk_document(pages)
        vs.add_chunks(chunks)

        # Record in database
        db.add_document(
            file_path=str(pdf_path),
            file_hash=file_hash,
            source_url=None,
        )
        total_chunks += len(chunks)
        logger.info("Ingested {} chunks from {}", len(chunks), pdf_path.name)

    return total_chunks


def run_daily_update() -> dict:
    """
    Execute the full daily update pipeline.

    Returns a summary dict with stats.
    """
    summary = {
        "pages_scraped": 0,
        "changes_detected": 0,
        "pdfs_downloaded": 0,
        "chunks_ingested": 0,
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
        summary["pages_scraped"] = len(results)
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

    # Step 3: Re-ingest updated PDFs
    logger.info("--- Step 3: Re-ingesting updated PDFs ---")
    try:
        chunks = reingest_updated_pdfs(db)
        summary["chunks_ingested"] = chunks
    except Exception as exc:
        logger.error("PDF re-ingestion failed: {}", exc)
        summary["errors"].append(f"Re-ingestion: {exc}")

    # Summary
    logger.info("=" * 50)
    logger.info("Daily update complete")
    logger.info("  Pages scraped:    {}", summary["pages_scraped"])
    logger.info("  Changes detected: {}", summary["changes_detected"])
    logger.info("  Chunks ingested:  {}", summary["chunks_ingested"])
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
    print(f"  Changes detected: {summary['changes_detected']}")
    print(f"  Chunks ingested:  {summary['chunks_ingested']}")
    if summary["errors"]:
        print(f"  Errors:           {len(summary['errors'])}")
        for err in summary["errors"]:
            print(f"    - {err}")
    print()

    # Exit with error code if there were critical failures
    if summary["errors"] and summary["pages_scraped"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
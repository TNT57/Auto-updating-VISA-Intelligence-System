"""
Build the knowledge base from the live Home Affairs site.

Fetches the monitored pages, optionally follows links within the same section,
downloads any linked PDFs, and indexes everything into ChromaDB. This is the
step that puts *real* content behind the chat — without it the vector store
only holds whatever PDFs you placed in data/raw/pdfs/ by hand.

Run it from a machine whose IP the site does not block (a home connection —
GitHub Actions runners are refused; see README).

    python scripts/fetch_and_index.py                    # configured URLs + their PDFs
    python scripts/fetch_and_index.py --crawl            # also follow in-section links
    python scripts/fetch_and_index.py --dry-run          # show what it would fetch
    python scripts/fetch_and_index.py --ask "What is the English requirement?"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup  # noqa: E402
from loguru import logger  # noqa: E402

from src.ingestion.pipeline import ingest_pages, ingest_pdfs  # noqa: E402
from src.scraping.homeaffairs_scraper import HomeAffairsScraper  # noqa: E402
from src.utils.config import ensure_directories, settings  # noqa: E402
from src.utils.db_manager import DatabaseManager  # noqa: E402

# Only follow links under this prefix. Keeps a crawl inside the visa content
# and away from the rest of a very large government site.
CRAWL_PREFIX = "/visas/getting-a-visa/visa-listing/temporary-graduate-485"


def _robots_allows(url: str, user_agent: str) -> bool:
    """Check robots.txt. On any failure, assume allowed but say so."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception as exc:
        logger.warning("Could not read {}: {} — proceeding", robots_url, exc)
        return True
    return parser.can_fetch(user_agent, url)


def discover_links(html: str, base_url: str, seen: set[str]) -> list[str]:
    """Find in-section page links (not PDFs) that haven't been visited."""
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].split("#")[0].strip()
        if not href or href.lower().endswith(".pdf"):
            continue

        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        if parsed.netloc != urlparse(base_url).netloc:
            continue
        if not parsed.path.startswith(CRAWL_PREFIX):
            continue

        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean not in seen and clean not in found:
            found.append(clean)

    return found


def fetch_all(scraper: HomeAffairsScraper, crawl: bool, max_pages: int,
              download_pdfs: bool) -> tuple[list[dict], list[str]]:
    """
    Fetch the configured URLs, optionally following in-section links.

    Returns (page results, downloaded PDF paths).
    """
    queue = list(settings.monitored_urls)
    seen: set[str] = set()
    results: list[dict] = []
    pdf_paths: list[str] = []

    while queue and len(results) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        result = scraper.scrape_page(url)
        results.append(result)

        if result.get("error"):
            logger.warning("  failed: {} — {}", url, result["error"])
            continue

        chars = len(result.get("content") or "")
        logger.info("  ok: {} ({:,} chars)", url, chars)

        html = result.get("html") or ""

        if download_pdfs:
            for pdf_url in scraper._discover_pdf_links(html, url):
                path = scraper.download_pdf(pdf_url)
                if path:
                    pdf_paths.append(path)
                    logger.info("  pdf: {}", Path(path).name)

        if crawl:
            for link in discover_links(html, url, seen):
                if link not in queue and len(seen) + len(queue) < max_pages:
                    queue.append(link)

        if queue:
            time.sleep(scraper.delay)

    return results, pdf_paths


def smoke_test(question: str, n_results: int = 3) -> None:
    """Retrieve against the freshly built index so you can see it working."""
    from src.retrieval.retriever import Retriever

    print(f"\nQ: {question}")
    print("-" * 70)
    results = Retriever().retrieve(question, n_results=n_results)

    if not results.results:
        print("  No results — is the index empty?")
        return

    for i, r in enumerate(results.results, 1):
        label = r.metadata.get("title") or r.source
        print(f"  {i}. [{r.relevance_score:6.1%}] {label[:60]}")
    print("\n  Top passage:")
    for line in results.results[0].content.strip().splitlines()[:6]:
        print(f"    | {line[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live Home Affairs content and index it for RAG."
    )
    parser.add_argument("--crawl", action="store_true",
                        help="Also follow links within the 485 section.")
    parser.add_argument("--max-pages", type=int, default=25,
                        help="Cap on pages fetched (default: 25).")
    parser.add_argument("--no-pdfs", action="store_true",
                        help="Skip discovering and downloading linked PDFs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched, then stop.")
    parser.add_argument("--ask", metavar="QUESTION", action="append",
                        help="Run a retrieval smoke test afterwards. Repeatable.")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Skip the robots.txt check.")
    args = parser.parse_args()

    ensure_directories()

    print("=" * 70)
    print("Fetch & Index — building the knowledge base from the live site")
    print("=" * 70)

    if args.dry_run:
        print(f"\nWould fetch {len(settings.monitored_urls)} configured URL(s):")
        for url in settings.monitored_urls:
            print(f"  - {url}")
        print(f"\nCrawl in-section links: {args.crawl}")
        print(f"Download linked PDFs:   {not args.no_pdfs}")
        print(f"Page cap:               {args.max_pages}")
        return

    first_url = settings.monitored_urls[0]
    if not args.ignore_robots and not _robots_allows(first_url, settings.user_agent):
        print("\nrobots.txt disallows fetching these pages for this user agent.")
        print("Review it before continuing, or pass --ignore-robots.")
        sys.exit(2)

    db = DatabaseManager()
    db.create_tables()
    scraper = HomeAffairsScraper()

    print(f"\nBackend: {scraper.fetcher.name}   "
          f"delay: {scraper.delay}s   crawl: {args.crawl}\n")

    results, pdf_paths = fetch_all(
        scraper, crawl=args.crawl, max_pages=args.max_pages,
        download_pdfs=not args.no_pdfs,
    )

    ok = [r for r in results if not r.get("error") and r.get("content")]
    failed = [r for r in results if r.get("error")]

    print(f"\nIndexing {len(ok)} page(s) and {len(pdf_paths)} PDF(s)...")
    page_chunks = ingest_pages(results)
    pdf_chunks = ingest_pdfs(db)

    from src.ingestion.vectorstore_manager import VectorStoreManager

    total = VectorStoreManager().get_collection_stats()["total_chunks"]

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Pages fetched:    {len(ok)}")
    print(f"  Pages failed:     {len(failed)}")
    print(f"  PDFs downloaded:  {len(pdf_paths)}")
    print(f"  Chunks from pages:{page_chunks:>4}")
    print(f"  Chunks from PDFs: {pdf_chunks:>4}")
    print(f"  Total in index:   {total}")

    for r in failed:
        print(f"    ! {r['url']} — {r['error']}")

    if not ok:
        print("\nNothing was fetched. If these are 403s, the site is refusing")
        print("this machine's IP — see the Troubleshooting section of README.md.")
        sys.exit(1)

    for question in args.ask or []:
        smoke_test(question)

    if not args.ask:
        print("\nTry it:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

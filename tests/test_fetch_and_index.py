"""
Tests for the fetch-and-index entry point.

The crawl is the risky part — a link discoverer that escapes its section would
walk a very large government site — so the containment rules are covered
explicitly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485"


class TestDiscoverLinks:
    """Crawl containment: same host, inside the section, no PDFs, no repeats."""

    def _links(self, html, seen=None):
        from scripts.fetch_and_index import discover_links

        return discover_links(html, BASE, seen or set())

    def test_finds_in_section_links(self):
        html = f'<a href="{BASE}/eligibility">Eligibility</a>'
        assert self._links(html) == [f"{BASE}/eligibility"]

    def test_resolves_relative_links(self):
        html = '<a href="/visas/getting-a-visa/visa-listing/temporary-graduate-485/fees">Fees</a>'
        assert self._links(html) == [f"{BASE}/fees"]

    def test_rejects_other_hosts(self):
        html = '<a href="https://example.com/visas/getting-a-visa/visa-listing/temporary-graduate-485/x">x</a>'
        assert self._links(html) == []

    def test_rejects_links_outside_the_section(self):
        html = '<a href="https://immi.homeaffairs.gov.au/citizenship/apply">Citizenship</a>'
        assert self._links(html) == []

    def test_rejects_pdfs(self):
        """PDFs go through download_pdf, not the page crawler."""
        html = f'<a href="{BASE}/form-1234.pdf">Form</a>'
        assert self._links(html) == []

    def test_strips_fragments_and_dedupes(self):
        html = (
            f'<a href="{BASE}/fees#top">a</a>'
            f'<a href="{BASE}/fees#bottom">b</a>'
        )
        assert self._links(html) == [f"{BASE}/fees"]

    def test_skips_already_seen(self):
        html = f'<a href="{BASE}/fees">Fees</a>'
        assert self._links(html, seen={f"{BASE}/fees"}) == []


class TestFetchAll:
    def _scraper(self, pages: dict):
        """A scraper stub whose scrape_page serves from a url->html dict."""
        scraper = MagicMock()
        scraper.delay = 0

        def scrape_page(url):
            if url not in pages:
                return {"url": url, "error": "404", "content": None, "html": None}
            html = pages[url]
            return {
                "url": url, "html": html, "content": f"text of {url}",
                "content_hash": "h", "title": "T",
            }

        scraper.scrape_page.side_effect = scrape_page
        scraper._discover_pdf_links.return_value = []
        return scraper

    def test_fetches_only_configured_urls_without_crawl(self):
        from scripts.fetch_and_index import fetch_all

        pages = {BASE: f'<a href="{BASE}/fees">Fees</a>'}
        scraper = self._scraper(pages)

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [BASE]
            results, _ = fetch_all(scraper, crawl=False, max_pages=25,
                                   download_pdfs=False)

        assert [r["url"] for r in results] == [BASE]

    def test_crawl_follows_in_section_links(self):
        from scripts.fetch_and_index import fetch_all

        pages = {
            BASE: f'<a href="{BASE}/fees">Fees</a>',
            f"{BASE}/fees": "<p>no links</p>",
        }
        scraper = self._scraper(pages)

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [BASE]
            results, _ = fetch_all(scraper, crawl=True, max_pages=25,
                                   download_pdfs=False)

        assert {r["url"] for r in results} == {BASE, f"{BASE}/fees"}

    def test_max_pages_caps_the_crawl(self):
        from scripts.fetch_and_index import fetch_all

        pages = {BASE: "".join(
            f'<a href="{BASE}/p{i}">p{i}</a>' for i in range(50)
        )}
        for i in range(50):
            pages[f"{BASE}/p{i}"] = "<p/>"
        scraper = self._scraper(pages)

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [BASE]
            results, _ = fetch_all(scraper, crawl=True, max_pages=5,
                                   download_pdfs=False)

        assert len(results) <= 5

    def test_a_failed_page_does_not_stop_the_run(self):
        from scripts.fetch_and_index import fetch_all

        scraper = self._scraper({f"{BASE}/ok": "<p/>"})

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [f"{BASE}/dead", f"{BASE}/ok"]
            results, _ = fetch_all(scraper, crawl=False, max_pages=25,
                                   download_pdfs=False)

        assert len(results) == 2
        assert results[0]["error"] == "404"
        assert results[1].get("error") is None

    def test_pdfs_are_downloaded_when_enabled(self):
        from scripts.fetch_and_index import fetch_all

        scraper = self._scraper({BASE: "<p/>"})
        scraper._discover_pdf_links.return_value = [f"{BASE}/form.pdf"]
        scraper.download_pdf.return_value = "/tmp/form.pdf"

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [BASE]
            _, pdfs = fetch_all(scraper, crawl=False, max_pages=25,
                                download_pdfs=True)

        assert pdfs == ["/tmp/form.pdf"]

    def test_pdfs_skipped_when_disabled(self):
        from scripts.fetch_and_index import fetch_all

        scraper = self._scraper({BASE: "<p/>"})
        scraper._discover_pdf_links.return_value = [f"{BASE}/form.pdf"]

        with patch("scripts.fetch_and_index.settings") as s:
            s.monitored_urls = [BASE]
            _, pdfs = fetch_all(scraper, crawl=False, max_pages=25,
                                download_pdfs=False)

        assert pdfs == []
        scraper.download_pdf.assert_not_called()


class TestRobots:
    def test_unreadable_robots_does_not_block_the_run(self):
        from scripts.fetch_and_index import _robots_allows

        with patch("scripts.fetch_and_index.RobotFileParser") as mock_parser:
            mock_parser.return_value.read.side_effect = OSError("unreachable")
            assert _robots_allows("https://immi.test/page", "bot") is True

    def test_disallow_is_respected(self):
        from scripts.fetch_and_index import _robots_allows

        with patch("scripts.fetch_and_index.RobotFileParser") as mock_parser:
            mock_parser.return_value.can_fetch.return_value = False
            assert _robots_allows("https://immi.test/page", "bot") is False


class TestSharedIngestion:
    """daily_update must keep using the same ingestion code after the move."""

    def test_daily_update_aliases_point_at_the_shared_pipeline(self):
        import scripts.daily_update as du
        from src.ingestion.pipeline import ingest_pages, ingest_pdfs

        assert du.reingest_updated_pdfs is ingest_pdfs
        assert du.ingest_scraped_pages is ingest_pages

    def test_pdf_ingest_skips_unchanged_files(self, tmp_path):
        from src.ingestion.pipeline import ingest_pdfs
        from src.utils.db_manager import DatabaseManager

        db = DatabaseManager(db_path=str(tmp_path / "t.db"))
        db.create_tables()

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        fpdf = pytest.importorskip("fpdf")
        doc = fpdf.FPDF()
        doc.add_page()
        doc.set_font("Helvetica", size=12)
        doc.multi_cell(0, 8, "The visa fee is AUD 2,235.",
                       new_x="LMARGIN", new_y="NEXT")
        doc.output(str(pdf_dir / "fees.pdf"))

        with patch("src.ingestion.vectorstore_manager.VectorStoreManager") as vs:
            vs.return_value.add_chunks.return_value = 1
            first = ingest_pdfs(db, pdf_dir=pdf_dir)
            second = ingest_pdfs(db, pdf_dir=pdf_dir)

        assert first > 0
        assert second == 0, "unchanged PDF should not be re-ingested"

"""
Tests for the pluggable fetch backends.

Covers backend selection, the failure contract (fetchers return errors rather
than raising), and that the scraper reports a failed fetch instead of
pretending it succeeded.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════
# Backend selection
# ══════════════════════════════════════════════════════════════════════

class TestBackendSelection:
    """SCRAPER_BACKEND resolution, including graceful degradation."""

    def test_auto_prefers_scrapling_when_available(self):
        from src.scraping import fetchers

        with patch.object(fetchers, "available_backends",
                          return_value=["httpx", "scrapling"]):
            fetcher = fetchers.get_fetcher("auto")

        assert fetcher.name == "scrapling"

    def test_auto_falls_back_to_httpx_when_scrapling_missing(self):
        from src.scraping import fetchers

        with patch.object(fetchers, "available_backends", return_value=["httpx"]):
            fetcher = fetchers.get_fetcher("auto")

        assert fetcher.name == "httpx"
        fetcher.close()

    def test_explicit_httpx_is_respected(self):
        from src.scraping.fetchers import get_fetcher

        fetcher = get_fetcher("httpx")
        assert fetcher.name == "httpx"
        fetcher.close()

    def test_missing_scrapling_degrades_instead_of_crashing(self):
        """A degraded fetch beats no fetch — never fail the run over a backend."""
        from src.scraping import fetchers

        with patch.object(fetchers, "ScraplingFetcher", side_effect=ImportError):
            fetcher = fetchers.get_fetcher("scrapling")

        assert fetcher.name == "httpx"
        fetcher.close()

    def test_unknown_backend_falls_back_to_httpx(self):
        from src.scraping.fetchers import get_fetcher

        fetcher = get_fetcher("firecrawl-typo")
        assert fetcher.name == "httpx"
        fetcher.close()

    def test_httpx_is_always_available(self):
        from src.scraping.fetchers import available_backends

        assert "httpx" in available_backends()


# ══════════════════════════════════════════════════════════════════════
# Failure contract
# ══════════════════════════════════════════════════════════════════════

class TestFetchResult:
    """`ok` must be false for anything that isn't usable HTML."""

    def test_ok_requires_html_and_no_error(self):
        from src.scraping.fetchers import FetchResult

        assert FetchResult(url="u", html="<html/>", status_code=200).ok
        assert not FetchResult(url="u", error="boom").ok
        assert not FetchResult(url="u", html="", status_code=200).ok
        assert not FetchResult(url="u", html=None, status_code=204).ok


class TestHttpxFetcher:
    def test_http_error_becomes_a_result_not_an_exception(self):
        import httpx

        from src.scraping.fetchers import HttpxFetcher

        fetcher = HttpxFetcher()
        with patch.object(fetcher.client, "get",
                          side_effect=httpx.ConnectError("refused")):
            result = fetcher.fetch("https://immi.test/page")

        assert not result.ok
        assert "refused" in result.error
        fetcher.close()

    def test_successful_fetch_returns_html(self):
        from src.scraping.fetchers import HttpxFetcher

        resp = MagicMock(text="<html>hi</html>", status_code=200)
        resp.raise_for_status = MagicMock()

        fetcher = HttpxFetcher()
        with patch.object(fetcher.client, "get", return_value=resp):
            result = fetcher.fetch("https://immi.test/page")

        assert result.ok
        assert result.html == "<html>hi</html>"
        assert result.status_code == 200
        fetcher.close()


class TestScraplingFetcher:
    """Scrapling is optional, so these skip cleanly when it isn't installed."""

    def _fetcher(self):
        pytest.importorskip("scrapling.fetchers", reason="scrapling not installed")
        from src.scraping.fetchers import ScraplingFetcher

        return ScraplingFetcher()

    def test_successful_fetch_returns_html(self):
        fetcher = self._fetcher()
        resp = MagicMock(status=200, html_content="<html>ok</html>")

        with patch.object(fetcher, "_fetcher") as mock:
            mock.get.return_value = resp
            result = fetcher.fetch("https://immi.test/page")

        assert result.ok
        assert result.html == "<html>ok</html>"

    def test_403_is_reported_as_an_error(self):
        """The failure that has been breaking the scheduled run."""
        fetcher = self._fetcher()
        resp = MagicMock(status=403, body="")

        with patch.object(fetcher, "_fetcher") as mock:
            mock.get.return_value = resp
            result = fetcher.fetch("https://immi.test/page")

        assert not result.ok
        assert result.status_code == 403
        assert "403" in result.error

    def test_bytes_body_is_decoded_to_str(self):
        """
        Regression: Scrapling's `.body` is bytes, not str. Passing it straight
        through crashed the snapshot writer with
        "TypeError: data must be str, not bytes".
        """
        fetcher = self._fetcher()
        resp = MagicMock(status=200, body="<html>café</html>".encode(),
                         encoding="utf-8")
        del resp.html_content  # force the .body decode path

        with patch.object(fetcher, "_fetcher") as mock:
            mock.get.return_value = resp
            result = fetcher.fetch("https://immi.test/page")

        assert isinstance(result.html, str)
        assert "café" in result.html

    def test_html_content_subclass_is_normalised_to_str(self):
        """`.html_content` is a str subclass; downstream code wants plain str."""
        fetcher = self._fetcher()

        class TextHandler(str):
            pass

        resp = MagicMock(status=200, html_content=TextHandler("<html>ok</html>"))

        with patch.object(fetcher, "_fetcher") as mock:
            mock.get.return_value = resp
            result = fetcher.fetch("https://immi.test/page")

        assert type(result.html) is str
        assert result.html == "<html>ok</html>"

    def test_transport_exception_becomes_a_result(self):
        fetcher = self._fetcher()

        with patch.object(fetcher, "_fetcher") as mock:
            mock.get.side_effect = RuntimeError("tls handshake failed")
            result = fetcher.fetch("https://immi.test/page")

        assert not result.ok
        assert "tls handshake failed" in result.error


# ══════════════════════════════════════════════════════════════════════
# Scraper integration
# ══════════════════════════════════════════════════════════════════════

class TestScraperUsesFetcher:
    def _scraper(self, tmp_path, fetcher):
        from src.scraping.homeaffairs_scraper import HomeAffairsScraper

        return HomeAffairsScraper(
            output_dir=tmp_path / "html",
            pdf_dir=tmp_path / "pdfs",
            delay=0,
            fetcher=fetcher,
        )

    def test_failed_fetch_surfaces_as_an_error_result(self, tmp_path):
        from src.scraping.fetchers import FetchResult

        fetcher = MagicMock(name="fetcher")
        fetcher.name = "stub"
        fetcher.fetch.return_value = FetchResult(
            url="https://immi.test/page", status_code=403, error="HTTP 403"
        )

        result = self._scraper(tmp_path, fetcher).scrape_page("https://immi.test/page")

        assert result["error"] == "HTTP 403"
        assert result["status_code"] == 403
        assert result["content"] is None

    def test_successful_fetch_extracts_and_snapshots(self, tmp_path):
        from src.scraping.fetchers import FetchResult

        html = (
            "<html><head><title>Visa fees</title></head>"
            "<body><main><p>The charge is AUD 2,235.</p></main></body></html>"
        )
        fetcher = MagicMock(name="fetcher")
        fetcher.name = "stub"
        fetcher.fetch.return_value = FetchResult(
            url="https://immi.homeaffairs.gov.au/fees", html=html, status_code=200
        )

        result = self._scraper(tmp_path, fetcher).scrape_page(
            "https://immi.homeaffairs.gov.au/fees"
        )

        assert result.get("error") is None
        assert "2,235" in result["content"]
        assert result["title"] == "Visa fees"
        assert result["content_hash"]
        assert Path(result["snapshot_path"]).exists()

    def test_scraper_defaults_to_configured_backend(self, tmp_path):
        """No injected fetcher means the factory decides."""
        from src.scraping.homeaffairs_scraper import HomeAffairsScraper

        scraper = HomeAffairsScraper(
            output_dir=tmp_path / "html", pdf_dir=tmp_path / "pdfs", delay=0
        )

        assert scraper.fetcher.name in ("httpx", "scrapling")

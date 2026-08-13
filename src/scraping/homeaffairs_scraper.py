"""
Home Affairs web scraper — Phase 2.

Scrapes immi.homeaffairs.gov.au 485 visa pages, downloads PDFs with
conditional GET support, and saves HTML snapshots for change detection.
"""

import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from src.scraping.fetchers import DEFAULT_HEADERS, BaseFetcher, get_fetcher
from src.utils.config import settings


class HomeAffairsScraper:
    """Scraper for Australian immigration website (485 visa pages)."""

    def __init__(
        self,
        output_dir: Path | None = None,
        pdf_dir: Path | None = None,
        delay: int | None = None,
        fetcher: BaseFetcher | None = None,
    ):
        self.output_dir = output_dir or settings.raw_html_dir
        self.pdf_dir = pdf_dir or settings.raw_pdf_dir
        self.delay = delay if delay is not None else settings.scraping_delay

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Page fetches go through a pluggable backend (see fetchers.py) so the
        # strategy can change without touching the scraper.
        self.fetcher = fetcher or get_fetcher(timeout=30.0)

        # PDF downloads stay on httpx: they need raw bytes and conditional-GET
        # via ETag, which the HTML-oriented fetcher interface does not cover.
        self.client = httpx.Client(
            headers={"User-Agent": settings.user_agent, **DEFAULT_HEADERS},
            timeout=30.0,
            follow_redirects=True,
        )
        logger.info(
            "Scraper initialized — backend: {}, delay: {}s, output: {}",
            self.fetcher.name,
            self.delay,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_page(self, url: str) -> dict:
        """
        Scrape a single page and return structured content.

        Returns dict with keys:
            url, title, content (normalised text), html (raw),
            content_hash, snapshot_path, status_code, timestamp
        """
        logger.info("Scraping: {}", url)
        result = self.fetcher.fetch(url)

        if not result.ok:
            logger.error("Failed to fetch {}: {}", url, result.error)
            return {
                "url": url,
                "title": None,
                "content": None,
                "html": None,
                "content_hash": None,
                "snapshot_path": None,
                "status_code": result.status_code,
                "timestamp": datetime.utcnow().isoformat(),
                "error": result.error,
            }

        html = result.html
        soup = BeautifulSoup(html, "lxml")

        # Extract and normalise main content
        title = self._extract_title(soup)
        content = self._extract_content(soup)

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        snapshot_path = self._save_snapshot(url, html)

        result = {
            "url": url,
            "title": title,
            "content": content,
            "html": html,
            "content_hash": content_hash,
            "snapshot_path": str(snapshot_path),
            "status_code": result.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(
            "Scraped {} — {} chars, hash {}…",
            url,
            len(content),
            content_hash[:12],
        )
        return result

    def download_pdf(self, url: str) -> str | None:
        """
        Download a PDF file.  Returns the local file path, or None if
        the file was unchanged (304) or the download failed.
        """
        filename = url.rsplit("/", 1)[-1]
        # Sanitise filename
        filename = re.sub(r"[^\w.\-]", "_", filename)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"
        dest = self.pdf_dir / filename

        # Conditional GET using ETag / Last-Modified
        headers: dict = {}
        etag_path = dest.with_suffix(".etag")
        if dest.exists() and etag_path.exists():
            headers["If-None-Match"] = etag_path.read_text().strip()

        try:
            resp = self.client.get(url, headers=headers)
            if resp.status_code == 304:
                logger.info("PDF unchanged (304): {}", filename)
                return None
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to download PDF {}: {}", url, exc)
            return None

        dest.write_bytes(resp.content)

        # Persist ETag for future conditional requests
        etag = resp.headers.get("ETag")
        if etag:
            etag_path.write_text(etag)

        logger.info("Downloaded PDF: {} ({} KB)", filename, len(resp.content) // 1024)
        return str(dest)

    def run_daily_scrape(self, urls: list[str] | None = None) -> list[dict]:
        """
        Execute the daily scraping routine for all monitored URLs.

        Returns a list of scrape results (one per URL).
        Also discovers and downloads any linked PDFs.
        """
        urls = urls or settings.monitored_urls
        results: list[dict] = []

        logger.info("Starting daily scrape — {} URLs", len(urls))

        for i, url in enumerate(urls):
            result = self.scrape_page(url)
            results.append(result)

            # Discover PDF links on the page and download new ones
            if result.get("html"):
                pdf_links = self._discover_pdf_links(result["html"], url)
                for pdf_url in pdf_links:
                    self.download_pdf(pdf_url)

            # Rate limiting between requests (skip after last URL)
            if i < len(urls) - 1:
                logger.debug("Rate-limit sleep: {}s", self.delay)
                time.sleep(self.delay)

        logger.info("Daily scrape complete — {} pages processed", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """Extract page title."""
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    # A container has to hold at least this much text to be believed. Without
    # a floor, an empty-but-present <main> wins and extraction silently returns
    # "" — which is exactly what an unrendered JS page looks like.
    MIN_CONTENT_CHARS = 200

    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> str:
        """
        Extract and normalise the main content area.

        Tries known Home Affairs content containers in order, skipping any that
        match but are empty, and falls back to <body> stripped of chrome.
        """
        best = ""

        for selector in (
            "div#contentBox",       # current Home Affairs main content wrapper
            "div.region-content",
            "div#content",
            "div.main-content",
            "main",
            "article",
        ):
            container = soup.select_one(selector)
            if not container:
                continue

            # Remove nav, footer, sidebar noise
            for tag in container.select("nav, footer, .sidebar, .breadcrumb, .menu"):
                tag.decompose()

            text = container.get_text(separator="\n", strip=True)
            if len(text) >= HomeAffairsScraper.MIN_CONTENT_CHARS:
                return text
            # Matched but thin — keep the best candidate and keep looking.
            if len(text) > len(best):
                best = text

        if best:
            return best

        # Fallback: entire body, stripped of chaff
        body = soup.find("body")
        if body:
            for tag in body.select("nav, footer, header, .sidebar, .menu, script, style"):
                tag.decompose()
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    def _save_snapshot(self, url: str, html: str) -> Path:
        """Save raw HTML snapshot to disk with date-stamped filename."""
        # Build a short slug from the URL path
        path_part = url.split("immi.homeaffairs.gov.au", 1)[-1].strip("/")
        slug = re.sub(r"[^\w]+", "_", path_part).strip("_")[:80]
        # Full timestamp, not just the date — two runs on the same day would
        # otherwise overwrite each other's snapshot.
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{slug}.html"
        dest = self.output_dir / filename
        dest.write_text(html, encoding="utf-8")
        logger.debug("Snapshot saved: {}", dest.name)
        return dest

    def _discover_pdf_links(self, html: str, base_url: str) -> list[str]:
        """Find all PDF links on a page."""
        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if ".pdf" in href.lower():
                # Resolve relative URLs
                if href.startswith("/"):
                    href = "https://immi.homeaffairs.gov.au" + href
                elif not href.startswith("http"):
                    # Relative to page
                    href = base_url.rsplit("/", 1)[0] + "/" + href
                if href.startswith("http") and href not in links:
                    links.append(href)

        logger.debug("Discovered {} PDF links on {}", len(links), base_url)
        return links
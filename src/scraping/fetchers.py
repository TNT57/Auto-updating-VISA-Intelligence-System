"""
Pluggable HTTP fetch backends for the scraper.

The scraper needs different fetch strategies depending on where it runs:

  * Locally (residential IP), plain httpx is sufficient — immi.homeaffairs.gov.au
    serves ordinary HTTP clients fine.
  * Where the site's WAF is stricter about client fingerprints, Scrapling's
    `Fetcher` impersonates a real browser at the TLS level, which plain httpx
    cannot do.

Neither helps when the block is on the *source IP* — a datacenter range such as
GitHub Actions runners. That needs a fetch originating elsewhere, which is what
the `BaseFetcher` interface leaves room for: a remote-fetch backend can be added
without touching the scraper or the pipeline.

Select a backend with SCRAPER_BACKEND=auto|httpx|scrapling (default: auto,
which prefers Scrapling when it is installed and falls back to httpx).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from src.utils.config import settings

# Ordinary browser headers. These are not what unblocks the site — a bare
# request works from an unblocked IP — but they keep the scraper looking like
# a normal client rather than an unidentified script.
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}


@dataclass
class FetchResult:
    """Outcome of a single fetch. `error` is set iff the fetch failed."""
    url: str
    html: str | None = None
    status_code: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.html)


class BaseFetcher(ABC):
    """Fetches a URL and returns its HTML."""

    name: str = "base"

    @abstractmethod
    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL. Must not raise — failures come back as FetchResult.error."""

    def close(self) -> None:  # noqa: B027 — optional hook, not every backend
        """Release any held resources. No-op unless the backend holds any."""
        return None


class HttpxFetcher(BaseFetcher):
    """Plain HTTP via httpx. Sufficient from an unblocked IP."""

    name = "httpx"

    def __init__(self, timeout: float = 30.0):
        import httpx

        self._httpx = httpx
        self.client = httpx.Client(
            headers={"User-Agent": settings.user_agent, **DEFAULT_HEADERS},
            timeout=timeout,
            follow_redirects=True,
        )

    def fetch(self, url: str) -> FetchResult:
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
        except self._httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return FetchResult(url=url, status_code=status, error=str(exc))
        return FetchResult(url=url, html=resp.text, status_code=resp.status_code)

    def close(self) -> None:
        self.client.close()


class ScraplingFetcher(BaseFetcher):
    """
    Fetch via Scrapling's `Fetcher`, which impersonates a real browser's TLS
    fingerprint and header ordering through curl_cffi.

    Deliberately *not* StealthyFetcher: that launches a hardened Firefox to
    defeat interactive anti-bot challenges, which is disproportionate for four
    requests a day to a public information page and would make CI slow and
    brittle. `Fetcher` is pure HTTP and adds no browser dependency.
    """

    name = "scrapling"

    def __init__(self, timeout: float = 30.0):
        from scrapling.fetchers import Fetcher

        self._fetcher = Fetcher
        self.timeout = timeout

    def fetch(self, url: str) -> FetchResult:
        try:
            resp = self._fetcher.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": settings.user_agent, **DEFAULT_HEADERS},
                stealthy_headers=True,
            )
        except Exception as exc:  # scrapling raises transport-specific errors
            return FetchResult(url=url, error=str(exc))

        status = getattr(resp, "status", None)
        if status is None or status >= 400:
            return FetchResult(
                url=url,
                status_code=status,
                error=f"HTTP {status} for {url}",
            )

        # `.body` is raw bytes and `.html_content` is a TextHandler (a str
        # subclass), not a plain str. Everything downstream — BeautifulSoup,
        # the snapshot writer, hashing — expects str, so normalise here.
        html = getattr(resp, "html_content", None)
        if html is None:
            body = resp.body
            html = (
                body.decode(getattr(resp, "encoding", None) or "utf-8", "replace")
                if isinstance(body, bytes)
                else body
            )

        return FetchResult(url=url, html=str(html), status_code=status)


class BrowserFetcher(BaseFetcher):
    """
    Render the page in a real browser via Scrapling's DynamicFetcher
    (Playwright Chromium).

    immi.homeaffairs.gov.au builds its visa content client-side: a plain HTTP
    fetch of the 485 page returns 1.2MB of HTML containing ~1,400 characters of
    text, all navigation and footer, with the body reading "Loading". Rendering
    it yields ~8,000 characters of real content and 4x the links. So for this
    site a browser is not optional — the HTTP backends return a page that looks
    successful and is empty.

    Costs roughly ten seconds and a Chromium process per page, which is why it
    is not used for anything that a plain fetch can handle.
    """

    name = "browser"

    def __init__(self, timeout: float = 90.0):
        from scrapling.fetchers import DynamicFetcher

        self._fetcher = DynamicFetcher
        self.timeout = timeout

    def fetch(self, url: str) -> FetchResult:
        try:
            page = self._fetcher.fetch(
                url,
                headless=True,
                # Wait for in-flight XHR to settle, otherwise the content
                # containers are still empty when the HTML is captured.
                network_idle=True,
                timeout=int(self.timeout * 1000),  # Playwright wants ms
            )
        except Exception as exc:
            return FetchResult(url=url, error=f"{type(exc).__name__}: {exc}")

        status = getattr(page, "status", None)
        if status is not None and status >= 400:
            return FetchResult(
                url=url, status_code=status, error=f"HTTP {status} for {url}"
            )

        html = getattr(page, "html_content", None)
        if html is None:
            return FetchResult(url=url, status_code=status,
                               error="Browser returned no HTML")

        return FetchResult(url=url, html=str(html), status_code=status or 200)


@lru_cache(maxsize=1)
def _browser_available() -> bool:
    """
    True when both DynamicFetcher and a Chromium binary are present.

    Cached: the probe starts a Playwright driver process, so calling it per
    fetcher construction is both slow and noisy at interpreter shutdown.
    """
    try:
        from scrapling.fetchers import DynamicFetcher  # noqa: F401
    except ImportError:
        return False

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


def available_backends() -> list[str]:
    """Backends that can actually be constructed in this environment."""
    backends = ["httpx"]
    try:
        import scrapling.fetchers  # noqa: F401

        backends.append("scrapling")
    except ImportError:
        pass
    if _browser_available():
        backends.append("browser")
    return backends


def get_fetcher(backend: str | None = None, timeout: float = 30.0) -> BaseFetcher:
    """
    Build a fetcher for the requested backend.

    "auto" prefers the browser, because the target site renders its content
    with JavaScript and the HTTP backends return a page that looks successful
    but is empty. It falls back to scrapling, then httpx, so a missing Chromium
    degrades the run rather than failing it — but the caller is warned loudly,
    since a degraded fetch here means empty pages, not merely slower ones.
    """
    backend = (backend or settings.scraper_backend or "auto").lower()
    available = available_backends()

    if backend == "auto":
        for candidate in ("browser", "scrapling", "httpx"):
            if candidate in available:
                backend = candidate
                break

    if backend == "browser":
        if "browser" in available:
            logger.info("Fetch backend: browser (Playwright — renders JS)")
            return BrowserFetcher(timeout=max(timeout, 90.0))
        logger.warning(
            "SCRAPER_BACKEND=browser but no Chromium binary was found. "
            "Run: python -m playwright install chromium\n"
            "Falling back — note this site is JS-rendered, so the fallback "
            "will fetch pages successfully but extract almost no text."
        )
        backend = "scrapling" if "scrapling" in available else "httpx"

    if backend == "scrapling":
        if "scrapling" in available:
            logger.info("Fetch backend: scrapling (browser impersonation)")
            return ScraplingFetcher(timeout=timeout)
        logger.warning(
            "SCRAPER_BACKEND=scrapling but scrapling is not installed "
            "(pip install 'scrapling[fetchers]') — falling back to httpx"
        )
        backend = "httpx"

    if backend != "httpx":
        logger.warning("Unknown SCRAPER_BACKEND '{}' — using httpx", backend)

    logger.info("Fetch backend: httpx")
    return HttpxFetcher(timeout=timeout)
